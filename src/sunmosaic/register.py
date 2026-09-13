"""Geometric registration: where does each tile sit on the canvas.

Strategy, in order:

1. Every tile that shows enough limb is placed by making the fitted solar centres coincide.
2. Tiles without a usable limb are placed by full-frame phase correlation against the
   tiles already placed (a central tile overlaps its neighbours enormously, which is the
   regime where full-frame correlation does work).
3. Every overlapping pair is refined by phase-correlating crops of the *predicted* overlap.
   This is what makes the thin 25 % overlaps work, where full-frame correlation fails.
4. All pairwise measurements are combined in one weighted least-squares solve, which also
   yields a loop-closure residual per pair.

Sign convention (pinned by tests/test_register.py):
``cv2.phaseCorrelate(A, B)`` returns ``(dx, dy)`` with ``t_A - t_B == (dx, dy)``.
"""

from __future__ import annotations

import itertools

import cv2
import numpy as np

from .errors import RegistrationError
from .preprocess import hanning
from .types import LimbFit, PairMatch, Params, Placement, Tile

FULL_PC_MIN_RESPONSE = 0.20
FULL_PC_MIN_OVERLAP_FRAC = 0.30
PRIOR_WEIGHT_TRUSTED = 0.02
PRIOR_WEIGHT_WEAK = 0.005


def coarse_from_limb(limbs: list[LimbFit | None]) -> list[tuple[float, float] | None]:
    """Origins that make the fitted solar centres coincide at (0, 0)."""
    return [None if lf is None else (-lf.cx, -lf.cy) for lf in limbs]


def _overlap_rect(
    origin_i: tuple[float, float], shape_i: tuple[int, int],
    origin_j: tuple[float, float], shape_j: tuple[int, int],
) -> tuple[float, float, float, float]:
    """Predicted overlap rectangle in canvas coordinates: (x0, y0, x1, y1)."""
    hi, wi = shape_i
    hj, wj = shape_j
    x0 = max(origin_i[0], origin_j[0])
    y0 = max(origin_i[1], origin_j[1])
    x1 = min(origin_i[0] + wi, origin_j[0] + wj)
    y1 = min(origin_i[1] + hi, origin_j[1] + hj)
    return x0, y0, x1, y1


def phase_correlate_full(hp_a: np.ndarray, hp_b: np.ndarray) -> tuple[float, float, float]:
    """Full-frame phase correlation of two high-passed tiles of possibly different size."""
    h = max(hp_a.shape[0], hp_b.shape[0])
    w = max(hp_a.shape[1], hp_b.shape[1])
    pad_a = np.zeros((h, w), np.float32)
    pad_b = np.zeros((h, w), np.float32)
    pad_a[: hp_a.shape[0], : hp_a.shape[1]] = hp_a
    pad_b[: hp_b.shape[0], : hp_b.shape[1]] = hp_b
    (dx, dy), response = cv2.phaseCorrelate(pad_a, pad_b, hanning(w, h))
    return float(dx), float(dy), float(response)


def refine_pair(
    tiles: list[Tile], origins: list[tuple[float, float]],
    i: int, j: int, params: Params,
) -> PairMatch | None:
    """Phase-correlate the predicted overlap of tiles i and j.

    Returns ``None`` when the tiles are not predicted to overlap enough to try.
    """
    shape_i = (tiles[i].height, tiles[i].width)
    shape_j = (tiles[j].height, tiles[j].width)
    x0, y0, x1, y1 = _overlap_rect(origins[i], shape_i, origins[j], shape_j)
    ow, oh = x1 - x0, y1 - y0
    if ow < params.min_overlap_px or oh < params.min_overlap_px:
        return None

    margin = int(min(params.crop_margin, max(16, 0.15 * min(ow, oh))))
    # Integer crop origins in each tile's own pixel coordinates.  Their difference carries
    # the sub-pixel rounding exactly, so nothing is lost by rounding here.
    ci_x = int(round(x0 + margin - origins[i][0]))
    ci_y = int(round(y0 + margin - origins[i][1]))
    cj_x = int(round(x0 + margin - origins[j][0]))
    cj_y = int(round(y0 + margin - origins[j][1]))
    ci_x, ci_y = max(ci_x, 0), max(ci_y, 0)
    cj_x, cj_y = max(cj_x, 0), max(cj_y, 0)

    cw = int(min(ow - 2 * margin, tiles[i].width - ci_x, tiles[j].width - cj_x))
    ch = int(min(oh - 2 * margin, tiles[i].height - ci_y, tiles[j].height - cj_y))
    if cw < 32 or ch < 32:
        return None

    crop_i = np.ascontiguousarray(tiles[i].highpass[ci_y : ci_y + ch, ci_x : ci_x + cw])
    crop_j = np.ascontiguousarray(tiles[j].highpass[cj_y : cj_y + ch, cj_x : cj_x + cw])
    (dx, dy), response = cv2.phaseCorrelate(crop_i, crop_j, hanning(cw, ch))

    # t_j - t_i = -(dx, dy) + c_i - c_j
    meas_x = -dx + (ci_x - cj_x)
    meas_y = -dy + (ci_y - cj_y)
    pred_x = origins[j][0] - origins[i][0]
    pred_y = origins[j][1] - origins[i][1]
    correction = float(np.hypot(meas_x - pred_x, meas_y - pred_y))

    accepted = True
    reason: str | None = None
    if response < params.min_response:
        accepted, reason = False, f"correlation response {response:.3f} below {params.min_response:.2f}"
    elif correction > params.crop_margin:
        accepted, reason = False, f"refinement of {correction:.1f} px exceeds the {params.crop_margin} px search margin"

    return PairMatch(
        i=i, j=j, overlap_w=int(ow), overlap_h=int(oh), method="crop-pc",
        dx=float(meas_x), dy=float(meas_y), response=float(response),
        accepted=accepted, reject_reason=reason, correction=correction,
    )


TEMPLATE_SCALE = 0.25
TEMPLATE_MIN_SCORE = 0.30


def _partial_mosaic(
    tiles: list[Tile], origins: list[tuple[float, float] | None], scale: float,
) -> tuple[np.ndarray, tuple[float, float]] | None:
    """A coarse, high-passed picture of everything placed so far."""
    placed = [k for k, o in enumerate(origins) if o is not None]
    if not placed:
        return None
    min_x = min(origins[k][0] for k in placed)
    min_y = min(origins[k][1] for k in placed)
    max_x = max(origins[k][0] + tiles[k].width for k in placed)
    max_y = max(origins[k][1] + tiles[k].height for k in placed)
    width = int(np.ceil((max_x - min_x) * scale))
    height = int(np.ceil((max_y - min_y) * scale))
    if width < 8 or height < 8:
        return None

    canvas = np.zeros((height, width), np.float32)
    counts = np.zeros((height, width), np.float32)
    for k in placed:
        small = cv2.resize(tiles[k].highpass, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_AREA)
        x0 = int(round((origins[k][0] - min_x) * scale))
        y0 = int(round((origins[k][1] - min_y) * scale))
        x1, y1 = min(x0 + small.shape[1], width), min(y0 + small.shape[0], height)
        if x1 <= x0 or y1 <= y0:
            continue
        canvas[y0:y1, x0:x1] += small[: y1 - y0, : x1 - x0]
        counts[y0:y1, x0:x1] += 1.0
    canvas /= np.maximum(counts, 1.0)
    return canvas, (min_x, min_y)


def place_by_template(
    tiles: list[Tile], origins: list[tuple[float, float] | None], index: int,
    scale: float = TEMPLATE_SCALE,
) -> tuple[tuple[float, float], float] | None:
    """Find a frame inside the mosaic built so far, by matching it as a template.

    This is the right tool for a frame that lies wholly within the others, such as a fifth
    view of the disk centre: it is contained rather than overlapping, so there is a genuine
    position to search for rather than a shift to measure.
    """
    built = _partial_mosaic(tiles, origins, scale)
    if built is None:
        return None
    mosaic, (min_x, min_y) = built
    template = cv2.resize(tiles[index].highpass, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_AREA)
    if template.shape[0] > mosaic.shape[0] or template.shape[1] > mosaic.shape[1]:
        return None  # not contained: this is a job for phase correlation

    scores = cv2.matchTemplate(mosaic, template, cv2.TM_CCOEFF_NORMED)
    _, best, _, location = cv2.minMaxLoc(scores)
    if best < TEMPLATE_MIN_SCORE:
        return None
    return (min_x + location[0] / scale, min_y + location[1] / scale), float(best)


def place_limbless(
    tiles: list[Tile], origins: list[tuple[float, float] | None],
    sources: list[str], warnings: list[str],
) -> None:
    """Place tiles that have no usable limb, by full-frame phase correlation.

    Modifies ``origins`` and ``sources`` in place.  Runs repeatedly so that a tile placed in
    one pass can anchor another in the next.
    """
    for _ in range(len(tiles)):
        pending = [k for k, o in enumerate(origins) if o is None]
        if not pending:
            return
        placed = [k for k, o in enumerate(origins) if o is not None]
        if not placed:
            # Nothing has a limb at all: seed the chain with tile 0.
            origins[0] = (0.0, 0.0)
            sources[0] = "phasecorr"
            warnings.append(
                "no tile shows enough limb; placement relies on image texture alone"
            )
            continue

        best = None
        for k in pending:
            matched = place_by_template(tiles, origins, k)
            if matched is not None:
                origin, score = matched
                if best is None or score > best[0]:
                    best = (score, k, origin, "template")
        if best is not None:
            _, k, origin, source = best
            origins[k] = origin
            sources[k] = source
            continue

        for k in pending:
            for ref in placed:
                dx, dy, response = phase_correlate_full(tiles[ref].highpass, tiles[k].highpass)
                cand_origin = (origins[ref][0] - dx, origins[ref][1] - dy)
                x0, y0, x1, y1 = _overlap_rect(
                    origins[ref], (tiles[ref].height, tiles[ref].width),
                    cand_origin, (tiles[k].height, tiles[k].width),
                )
                area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
                frac = area / float(tiles[k].height * tiles[k].width)
                if response < FULL_PC_MIN_RESPONSE or frac < FULL_PC_MIN_OVERLAP_FRAC:
                    continue
                if best is None or response > best[0]:
                    best = (response, k, cand_origin, "phasecorr")
        if best is None:
            return
        _, k, origin, source = best
        origins[k] = origin
        sources[k] = source


def solve_global(
    tiles: list[Tile], coarse: list[tuple[float, float]], limbs: list[LimbFit | None],
    pairs: list[PairMatch], params: Params,
) -> tuple[list[tuple[float, float]], list[str]]:
    """Weighted least-squares placement from all accepted pairs plus weak coarse priors.

    The priors keep the graph connected and fix the gauge; their weight is far below any
    real measurement, so they never override a matched pair.
    """
    n = len(tiles)
    accepted = [p for p in pairs if p.accepted]
    warnings: list[str] = []

    rows: list[np.ndarray] = []
    rhs_x: list[float] = []
    rhs_y: list[float] = []
    for pair in accepted:
        row = np.zeros(n)
        row[pair.j] = 1.0
        row[pair.i] = -1.0
        weight = float(np.clip(pair.response, 0.3, 1.0))
        rows.append(row * weight)
        rhs_x.append(pair.dx * weight)
        rhs_y.append(pair.dy * weight)

    for k in range(n):
        lf = limbs[k]
        weight = PRIOR_WEIGHT_TRUSTED if (lf is not None and lf.trusted) else PRIOR_WEIGHT_WEAK
        row = np.zeros(n)
        row[k] = 1.0
        rows.append(row * weight)
        rhs_x.append(coarse[k][0] * weight)
        rhs_y.append(coarse[k][1] * weight)

    a = np.vstack(rows)
    sol_x, *_ = np.linalg.lstsq(a, np.array(rhs_x), rcond=None)
    sol_y, *_ = np.linalg.lstsq(a, np.array(rhs_y), rcond=None)
    solved = [(float(sol_x[k]), float(sol_y[k])) for k in range(n)]

    for pair in pairs:
        if not pair.accepted:
            continue
        res_x = (solved[pair.j][0] - solved[pair.i][0]) - pair.dx
        res_y = (solved[pair.j][1] - solved[pair.i][1]) - pair.dy
        pair.residual_px = float(np.hypot(res_x, res_y))

    connected = {k for p in accepted for k in (p.i, p.j)}
    for k in range(n):
        if k in connected:
            continue
        lf = limbs[k]
        if lf is not None and lf.trusted:
            warnings.append(
                f"tile {k} ({tiles[k].name}) matched no other tile; it was placed from its "
                "limb alone, so check the seams around it"
            )
        else:
            raise RegistrationError(
                f"Tile '{tiles[k].name}' could not be placed: it shows no usable limb and "
                "does not correlate with any other tile. Check that the selected files are "
                "overlapping views of the same Sun."
            )

    worst = max((p.residual_px or 0.0) for p in accepted) if accepted else 0.0
    if worst > params.max_loop_resid:
        warnings.append(
            f"tile positions are not fully consistent with a pure shift (worst mismatch "
            f"{worst:.1f} px); the frames may be slightly rotated relative to each other"
        )
    return solved, warnings


def build_placements(
    tiles: list[Tile], limbs: list[LimbFit | None], params: Params,
) -> tuple[list[Placement], list[PairMatch], tuple[int, int], list[str]]:
    """Full registration: coarse placement, pairwise refinement, global solve, canvas box."""
    n = len(tiles)
    warnings: list[str] = []
    origins: list[tuple[float, float] | None] = coarse_from_limb(limbs)
    sources = ["limb" if o is not None else "unplaced" for o in origins]

    if any(o is None for o in origins):
        place_limbless(tiles, origins, sources, warnings)
    unplaced = [k for k, o in enumerate(origins) if o is None]
    if unplaced:
        names = ", ".join(tiles[k].name for k in unplaced)
        raise RegistrationError(
            f"Could not place these files: {names}. They show no usable limb and do not "
            "correlate with the other frames. Check that all files are overlapping views "
            "of the same Sun."
        )
    coarse: list[tuple[float, float]] = [o for o in origins if o is not None]

    pairs: list[PairMatch] = []
    for i, j in itertools.combinations(range(n), 2):
        match = refine_pair(tiles, coarse, i, j, params)
        if match is not None:
            pairs.append(match)

    if not any(p.accepted for p in pairs) and n > 1:
        best = max((p.response for p in pairs), default=0.0)
        if all(limbs[k] is None or not limbs[k].trusted for k in range(n)):
            raise RegistrationError(
                f"Could not register the files: no pair of frames matches (best correlation "
                f"{best:.2f}). Check that they are overlapping views of the same Sun."
            )
        warnings.append(
            f"no pair of frames could be matched on texture (best correlation {best:.2f}); "
            "placement relies on the limb fits alone"
        )

    solved, solve_warnings = solve_global(tiles, coarse, limbs, pairs, params)
    warnings.extend(solve_warnings)

    min_x = min(o[0] for o in solved)
    min_y = min(o[1] for o in solved)
    solved = [(o[0] - min_x, o[1] - min_y) for o in solved]
    canvas_w = int(np.ceil(max(o[0] + tiles[k].width for k, o in enumerate(solved))))
    canvas_h = int(np.ceil(max(o[1] + tiles[k].height for k, o in enumerate(solved))))

    placements = [Placement(x=o[0], y=o[1], source=sources[k]) for k, o in enumerate(solved)]
    return placements, pairs, (canvas_w, canvas_h), warnings


# --- Rotation refinement ----------------------------------------------------------------
#
# Everything above assumes the frames differ by a pure shift.  On an alt-azimuth mount the
# field rotates while the sequence is captured, and after a couple of minutes that can reach
# several pixels at the limb.  It shows up as pair measurements that cannot all be satisfied
# at once, which is exactly what the loop-closure residual measures.  When that residual is
# large, each pair is re-measured with ECC to recover the relative angle, the angles are
# solved globally, and the frames are rotated into a common orientation.  Everything
# downstream then stays translation-only.

ECC_MAX_SIZE = 512
ECC_ITERATIONS = 60
ECC_EPS = 1e-5
MIN_USEFUL_ROTATION_DEG = 0.01
MAX_PLAUSIBLE_ROTATION_DEG = 5.0


def _overlap_crops(
    tiles: list[Tile], origins: list[tuple[float, float]], i: int, j: int, params: Params,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Matching crops of the predicted overlap, downsampled for iterative fitting."""
    x0, y0, x1, y1 = _overlap_rect(
        origins[i], (tiles[i].height, tiles[i].width),
        origins[j], (tiles[j].height, tiles[j].width),
    )
    ow, oh = x1 - x0, y1 - y0
    if ow < params.min_overlap_px or oh < params.min_overlap_px:
        return None
    margin = int(min(params.crop_margin, max(16, 0.15 * min(ow, oh))))
    ci_x = max(int(round(x0 + margin - origins[i][0])), 0)
    ci_y = max(int(round(y0 + margin - origins[i][1])), 0)
    cj_x = max(int(round(x0 + margin - origins[j][0])), 0)
    cj_y = max(int(round(y0 + margin - origins[j][1])), 0)
    cw = int(min(ow - 2 * margin, tiles[i].width - ci_x, tiles[j].width - cj_x))
    ch = int(min(oh - 2 * margin, tiles[i].height - ci_y, tiles[j].height - cj_y))
    if cw < 64 or ch < 64:
        return None
    crop_i = tiles[i].highpass[ci_y : ci_y + ch, ci_x : ci_x + cw]
    crop_j = tiles[j].highpass[cj_y : cj_y + ch, cj_x : cj_x + cw]
    scale = min(1.0, ECC_MAX_SIZE / max(cw, ch))
    if scale < 1.0:
        crop_i = cv2.resize(crop_i, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        crop_j = cv2.resize(crop_j, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(crop_i), np.ascontiguousarray(crop_j)


def estimate_pair_rotation(
    tiles: list[Tile], origins: list[tuple[float, float]], i: int, j: int, params: Params,
) -> tuple[float, float] | None:
    """Angle of frame j relative to frame i, with its correlation score.

    The sign is pinned by ``tests/test_register.py``: the value returned is what has to be
    applied to bring the frame back, which is the negative of the angle it carries.
    """
    crops = _overlap_crops(tiles, origins, i, j, params)
    if crops is None:
        return None
    crop_i, crop_j = crops
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, ECC_ITERATIONS, ECC_EPS)
    try:
        score, warp = cv2.findTransformECC(
            crop_i, crop_j, warp, cv2.MOTION_EUCLIDEAN, criteria, None, 5
        )
    except cv2.error:
        return None
    angle = float(np.arctan2(warp[1, 0], warp[0, 0]))
    if abs(np.degrees(angle)) > MAX_PLAUSIBLE_ROTATION_DEG or not np.isfinite(score):
        return None
    return angle, float(score)


def solve_rotations(
    tiles: list[Tile], origins: list[tuple[float, float]], pairs: list[PairMatch],
    params: Params,
) -> tuple[np.ndarray, int]:
    """Per-frame angle from every overlapping pair.  Gauge: the angles sum to zero.

    Every pair with usable overlap is measured, not only the matched ones.  ECC converges
    on rotations large enough to defeat phase correlation, so this is what rescues a set
    whose frames stopped matching precisely because they had rotated.
    """
    n = len(tiles)
    rows, rhs = [], []
    measured = 0
    for pair in pairs:
        estimate = estimate_pair_rotation(tiles, origins, pair.i, pair.j, params)
        if estimate is None:
            continue
        angle, score = estimate
        row = np.zeros(n)
        row[pair.j] = 1.0
        row[pair.i] = -1.0
        weight = float(np.clip(score, 0.05, 1.0))
        rows.append(row * weight)
        rhs.append(angle * weight)
        measured += 1
    if measured < n - 1:
        return np.zeros(n), measured
    rows.append(np.ones(n))
    rhs.append(0.0)
    solution, *_ = np.linalg.lstsq(np.vstack(rows), np.array(rhs), rcond=None)
    return solution, measured


def derotate(tiles: list[Tile], angles: np.ndarray) -> list[Tile]:
    """Rotate each frame about its own centre into the common orientation."""
    rotated = []
    for tile, angle in zip(tiles, angles):
        if abs(np.degrees(angle)) < MIN_USEFUL_ROTATION_DEG:
            rotated.append(tile)
            continue
        centre = ((tile.width - 1) / 2.0, (tile.height - 1) / 2.0)
        matrix = cv2.getRotationMatrix2D(centre, float(np.degrees(angle)), 1.0)
        data = cv2.warpAffine(
            tile.data, matrix, (tile.width, tile.height), flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        np.clip(data, float(tile.data.min()), float(tile.data.max()), out=data)
        rotated.append(Tile(
            name=tile.name, data=np.ascontiguousarray(data.astype(np.float32)),
            native_dtype=tile.native_dtype, native_max=tile.native_max,
            warnings=list(tile.warnings),
        ))
    return rotated


def needs_rotation(
    tiles: list[Tile], pairs: list[PairMatch], params: Params,
) -> bool:
    """Whether the frames look inconsistent with a pure shift.

    Two symptoms: pair measurements that cannot all be satisfied at once, or a frame that
    stopped matching its neighbours altogether.  Both are what field rotation looks like.
    """
    accepted = [p for p in pairs if p.accepted]
    if not accepted:
        return len(pairs) > 0
    worst = max(p.residual_px or 0.0 for p in accepted)
    if worst > params.max_loop_resid:
        return True
    connected = {k for p in accepted for k in (p.i, p.j)}
    return any(k not in connected for k in range(len(tiles)))
