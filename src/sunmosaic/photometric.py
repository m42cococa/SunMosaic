"""Equalise brightness between tiles before they are blended.

Two effects are corrected, in the order they are measured:

* a multiplicative gain per tile, from the median ratio inside each overlap (the etalon
  transmission and the exposure differ slightly between frames), and
* a residual additive sky offset per tile, measured on the gain-corrected frames.

A constant gain cannot describe the etalon's "sweet spot", which varies across each frame.
What it leaves behind is low-frequency, and that is exactly what the multi-band blend hides.
"""

from __future__ import annotations

import cv2
import numpy as np

from .types import PairMatch, Tile

SAMPLE_STEP = 16
DISK_ERODE_PX = 15
SKY_DILATE_PX = 60
SATURATION_ADU = 64000.0
MIN_SKY_SAMPLES = 2000
GAIN_LIMITS = (0.7, 1.4)


def _solve_relative(
    n: int, observations: list[tuple[int, int, float, float]],
) -> np.ndarray:
    """Least squares for values whose pairwise differences are observed.

    ``observations`` are ``(i, j, value_j_minus_i, weight)``.  The gauge is ``sum == 0``,
    which preserves the overall brightness of the mosaic.
    """
    if not observations:
        return np.zeros(n)
    rows, rhs = [], []
    for i, j, value, weight in observations:
        row = np.zeros(n)
        row[j] = 1.0
        row[i] = -1.0
        rows.append(row * weight)
        rhs.append(value * weight)
    rows.append(np.ones(n))  # gauge: the values sum to zero
    rhs.append(0.0)
    solution, *_ = np.linalg.lstsq(np.vstack(rows), np.array(rhs), rcond=None)
    return solution


def solve_gains_offsets(
    tiles: list[Tile], images: list[np.ndarray], masks: list[np.ndarray],
    disk_masks: list[np.ndarray], pairs: list[PairMatch], pedestal: float,
    equalize: str = "gain+offset",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return ``(gains, offsets, warnings)`` for the warped tiles."""
    n = len(tiles)
    warnings: list[str] = []
    gains = np.ones(n)
    offsets = np.zeros(n)
    if equalize == "off":
        return gains, offsets, warnings

    grays = [im if im.ndim == 2 else im.mean(axis=2, dtype=np.float32) for im in images]
    sky_mad = float(np.median([t.sky_mad for t in tiles]))
    floor = pedestal + 10.0 * sky_mad

    log_obs: list[tuple[int, int, float, float]] = []
    for pair in pairs:
        if not pair.accepted:
            continue
        i, j = pair.i, pair.j
        both = (disk_masks[i] > 0) & (disk_masks[j] > 0) & (masks[i] > 0) & (masks[j] > 0)
        sample = np.zeros_like(both)
        sample[::SAMPLE_STEP, ::SAMPLE_STEP] = True
        sel = both & sample
        vi, vj = grays[i][sel], grays[j][sel]
        good = (vi > floor) & (vj > floor) & (vi < SATURATION_ADU) & (vj < SATURATION_ADU)
        if good.sum() < 200:
            warnings.append(
                f"tiles {i} and {j} share too little usable disk to compare their brightness"
            )
            continue
        ratio = np.log((vj[good] - pedestal).clip(1e-3)) - np.log((vi[good] - pedestal).clip(1e-3))
        log_obs.append((i, j, float(np.median(ratio)), float(np.clip(pair.response, 0.3, 1.0))))

    if log_obs:
        log_gains = _solve_relative(n, log_obs)
        gains = np.exp(log_gains)
        clipped = np.clip(gains, *GAIN_LIMITS)
        if not np.allclose(clipped, gains):
            warnings.append(
                "brightness differences between frames are larger than expected; "
                "the correction was capped"
            )
        gains = clipped
    elif len(pairs) > 0:
        warnings.append("brightness could not be equalised; frames are used as they are")

    if equalize == "gain":
        return gains, offsets, warnings

    corrected = [(grays[k] - pedestal) / gains[k] + pedestal for k in range(n)]
    offsets, offset_warnings = solve_sky_offsets(corrected, masks, disk_masks, pairs)
    warnings.extend(offset_warnings)
    return gains, offsets, warnings


def solve_sky_offsets(
    corrected: list[np.ndarray], masks: list[np.ndarray], disk_masks: list[np.ndarray],
    pairs: list[PairMatch],
) -> tuple[np.ndarray, list[str]]:
    """Match the background level between frames, measured on the shared sky."""
    n = len(corrected)
    warnings: list[str] = []
    observations: list[tuple[int, int, float, float]] = []
    for pair in pairs:
        if not pair.accepted:
            continue
        i, j = pair.i, pair.j
        sky = (disk_masks[i] == 0) & (disk_masks[j] == 0) & (masks[i] > 0) & (masks[j] > 0)
        if sky.sum() < MIN_SKY_SAMPLES:
            continue
        diff = corrected[j][sky] - corrected[i][sky]
        observations.append(
            (i, j, float(np.median(diff)), float(np.clip(pair.response, 0.3, 1.0)))
        )
    if observations:
        return _solve_relative(n, observations), warnings
    if pairs:
        warnings.append("not enough sky is shared between frames to match their backgrounds")
    return np.zeros(n), warnings


def apply_correction(
    images: list[np.ndarray], gains: np.ndarray, offsets: np.ndarray, pedestal: float,
) -> list[np.ndarray]:
    """Apply the solved gains and offsets to the warped tiles."""
    out = []
    for k, image in enumerate(images):
        corrected = (image - pedestal) / float(gains[k]) + pedestal - float(offsets[k])
        out.append(corrected.astype(np.float32))
    return out


# --- Low-order illumination field -------------------------------------------------------
#
# A single number per frame cannot describe the etalon's "sweet spot", which varies smoothly
# across each frame.  Fitting a gentle polynomial in frame coordinates removes most of what
# is left.  Two things make this fit delicate, and both are handled below:
#
# * a smooth field shared by every frame is invisible to pairwise ratios, so the problem has
#   a null space; a ridge prior pulls the answer to the smallest field that explains the
#   overlaps rather than an arbitrary member of that family, and
# * features evolve between exposures, so the fit is done with Huber weights and falls back
#   to a lower degree when the coefficients grow implausible.

FIELD_TERMS = {1: 3, 2: 6}
RIDGE_SIGMA = {"linear": 0.15, "quadratic": 0.10}
HUBER_DELTA = 0.08
IRLS_PASSES = 3
MAX_FIELD_COEFF = 0.35
MAX_FIELD_SAMPLES = 4000
UV_CLIP = 1.6


def _basis(u: np.ndarray, v: np.ndarray, degree: int) -> np.ndarray:
    """Polynomial basis in normalised frame coordinates, constant term first."""
    terms = [np.ones_like(u), u, v]
    if degree >= 2:
        terms += [u * u, v * v, u * v]
    return np.stack(terms, axis=-1)


def _frame_coords(
    xs: np.ndarray, ys: np.ndarray, placement, tile, clip: float | None = UV_CLIP,
) -> tuple[np.ndarray, np.ndarray]:
    """Canvas pixels expressed in the frame's own coordinates, spanning -1 to 1."""
    u = 2.0 * (xs - placement.x) / max(tile.width - 1, 1) - 1.0
    v = 2.0 * (ys - placement.y) / max(tile.height - 1, 1) - 1.0
    if clip is not None:
        u = np.clip(u, -clip, clip)
        v = np.clip(v, -clip, clip)
    return u.astype(np.float64), v.astype(np.float64)


def _field_observations(
    tiles, images, masks, disk_masks, pairs, placements, pedestal: float, degree: int,
    sky_mad: float, rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Sampled log-ratios across every overlap, with their design rows."""
    n = len(tiles)
    width = FIELD_TERMS[degree]
    grays = [im if im.ndim == 2 else im.mean(axis=2, dtype=np.float32) for im in images]
    smooth = [cv2.GaussianBlur(g, (0, 0), 8.0) for g in grays]
    floor = pedestal + 10.0 * sky_mad

    rows: list[np.ndarray] = []
    values: list[np.ndarray] = []
    for pair in pairs:
        if not pair.accepted:
            continue
        i, j = pair.i, pair.j
        usable = (
            (disk_masks[i] > 0) & (disk_masks[j] > 0) & (masks[i] > 0) & (masks[j] > 0)
        )
        grid = np.zeros_like(usable)
        grid[::SAMPLE_STEP, ::SAMPLE_STEP] = True
        ys, xs = np.nonzero(usable & grid)
        if ys.size < 200:
            continue
        vi = smooth[i][ys, xs]
        vj = smooth[j][ys, xs]
        good = (vi > floor) & (vj > floor) & (vi < SATURATION_ADU) & (vj < SATURATION_ADU)
        ys, xs, vi, vj = ys[good], xs[good], vi[good], vj[good]
        if ys.size < 200:
            continue
        if ys.size > MAX_FIELD_SAMPLES:
            pick = rng.choice(ys.size, MAX_FIELD_SAMPLES, replace=False)
            ys, xs, vi, vj = ys[pick], xs[pick], vi[pick], vj[pick]

        ui, vvi = _frame_coords(xs, ys, placements[i], tiles[i])
        uj, vvj = _frame_coords(xs, ys, placements[j], tiles[j])
        block = np.zeros((ys.size, n * width))
        block[:, i * width : (i + 1) * width] = -_basis(ui, vvi, degree)
        block[:, j * width : (j + 1) * width] = _basis(uj, vvj, degree)
        rows.append(block)
        values.append(
            np.log(np.clip(vj - pedestal, 1e-3, None))
            - np.log(np.clip(vi - pedestal, 1e-3, None))
        )
    if not rows:
        return None
    design = np.vstack(rows)
    observed = np.concatenate(values)
    return design, observed, np.ones(observed.size)


def _solve_field(
    design: np.ndarray, observed: np.ndarray, n: int, degree: int,
) -> tuple[np.ndarray, float, float]:
    """Ridge-regularised, Huber-weighted least squares.  Returns coefficients and RMS."""
    width = FIELD_TERMS[degree]
    sigma = np.empty(width)
    sigma[0] = np.inf  # the constant term is free; only the shape is regularised
    sigma[1:3] = RIDGE_SIGMA["linear"]
    if degree >= 2:
        sigma[3:] = RIDGE_SIGMA["quadratic"]

    prior_rows = []
    for tile_index in range(n):
        for term in range(1, width):
            row = np.zeros(n * width)
            row[tile_index * width + term] = 1.0 / sigma[term]
            prior_rows.append(row)
    gauge = np.zeros(n * width)
    gauge[::width] = 1.0  # the constant terms sum to zero, preserving overall brightness
    prior = np.vstack(prior_rows + [gauge * 10.0])
    prior_rhs = np.zeros(prior.shape[0])

    weights = np.ones(observed.size)
    coefficients = np.zeros(n * width)
    for _ in range(IRLS_PASSES):
        scaled = design * weights[:, None]
        stacked = np.vstack([scaled, prior])
        rhs = np.concatenate([observed * weights, prior_rhs])
        coefficients, *_ = np.linalg.lstsq(stacked, rhs, rcond=None)
        residual = observed - design @ coefficients
        weights = np.minimum(1.0, HUBER_DELTA / np.maximum(np.abs(residual), 1e-9))

    before = float(np.sqrt(np.mean(observed**2)))
    after = float(np.sqrt(np.mean((observed - design @ coefficients) ** 2)))
    return coefficients.reshape(n, width), before, after


def solve_illumination_field(
    tiles, images, masks, disk_masks, pairs, placements, pedestal: float, degree: int,
) -> tuple[np.ndarray | None, int, list[str]]:
    """Fit a per-frame illumination field, dropping to a lower degree if it misbehaves.

    Returns ``(coefficients, degree_used, warnings)``; coefficients are ``(n, terms)`` in
    log space with the constant term first, or ``None`` when no degree was usable.
    """
    n = len(tiles)
    warnings: list[str] = []
    sky_mad = float(np.median([t.sky_mad for t in tiles]))
    rng = np.random.default_rng(0)

    for attempt in range(degree, 0, -1):
        sampled = _field_observations(
            tiles, images, masks, disk_masks, pairs, placements, pedestal, attempt,
            sky_mad, rng,
        )
        if sampled is None:
            warnings.append("not enough shared disk to measure the illumination across frames")
            return None, 0, warnings
        design, observed, _ = sampled
        coefficients, before, after = _solve_field(design, observed, n, attempt)

        worst = float(np.abs(coefficients[:, 1:]).max()) if coefficients.shape[1] > 1 else 0.0
        if worst > MAX_FIELD_COEFF:
            warnings.append(
                f"the illumination varies more across the frames than expected; "
                f"dropped from a {'quadratic' if attempt == 2 else 'linear'} correction"
            )
            continue
        if after >= before:
            warnings.append("the illumination correction did not improve the match; skipped")
            continue
        return coefficients, attempt, warnings

    warnings.append("no illumination correction could be fitted; frame gains are used alone")
    return None, 0, warnings


def apply_illumination_field(
    images: list[np.ndarray], coefficients: np.ndarray, placements, tiles,
    pedestal: float, degree: int,
) -> list[np.ndarray]:
    """Divide each warped frame by its fitted illumination field."""
    out = []
    for index, image in enumerate(images):
        height, width = image.shape[:2]
        ys, xs = np.mgrid[0:height, 0:width]
        u, v = _frame_coords(xs, ys, placements[index], tiles[index])
        log_gain = _basis(u, v, degree) @ coefficients[index]
        field = np.exp(log_gain).astype(np.float32)
        if image.ndim == 3:
            field = field[..., None]
        out.append(((image - pedestal) / field + pedestal).astype(np.float32))
    return out


def equalise(
    tiles, images: list[np.ndarray], masks: list[np.ndarray], disk_masks: list[np.ndarray],
    pairs: list[PairMatch], placements, pedestal: float, equalize: str,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray | None, int, list[str]]:
    """Apply the requested amount of brightness matching.

    Returns the corrected frames, the per-frame gain and background offset (for reporting),
    the illumination-field coefficients when one was fitted, the degree used, and warnings.
    """
    if equalize in ("linear", "quadratic"):
        degree = 1 if equalize == "linear" else 2
        coefficients, used, warnings = solve_illumination_field(
            tiles, images, masks, disk_masks, pairs, placements, pedestal, degree
        )
        if coefficients is not None:
            corrected = apply_illumination_field(
                images, coefficients, placements, tiles, pedestal, used
            )
            grays = [
                im if im.ndim == 2 else im.mean(axis=2, dtype=np.float32) for im in corrected
            ]
            offsets, offset_warnings = solve_sky_offsets(grays, masks, disk_masks, pairs)
            warnings.extend(offset_warnings)
            corrected = [
                (frame - float(offsets[k])).astype(np.float32)
                for k, frame in enumerate(corrected)
            ]
            return corrected, np.exp(coefficients[:, 0]), offsets, coefficients, used, warnings
        warnings.append("fell back to a single brightness number per frame")
        equalize = "gain+offset"

    gains, offsets, warnings = solve_gains_offsets(
        tiles, images, masks, disk_masks, pairs, pedestal, equalize
    )
    corrected = apply_correction(images, gains, offsets, pedestal)
    return corrected, gains, offsets, None, 0, warnings
