"""Optional finishing steps applied to a built mosaic: orientation and prominences.

Nothing here touches the mosaic pipeline.  A :class:`FinishParams` with default values
returns the built mosaic unchanged, so leaving the controls alone costs nothing.

Orientation: a mirror flip, a rotation about the fitted solar centre, and padding to a
square with the Sun in the middle.  Exposed areas are filled so the sky stays continuous.

Prominences: a copy of the image is brightened and shown only outside the disk, the disk
itself being taken from the untouched original inside a protective circle.  The limb glow
is removed first, because on real frames it saturates into a bright ring otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .blend import fill_uncovered
from .types import MosaicResult

FILL_SMOOTH_SIGMA = 25.0
FILL_NOISE_SCALE = 0.6
COVERAGE_ERODE_PX = 2
GLOW_RING_PX = 1.0
LIMB_SECTOR_DEG = 2.0
LIMB_SEARCH_IN_PX = 6.0  # beyond the largest offset, so an inward dent is still found
LIMB_SEARCH_OUT_PX = 14.0
LIMB_MAX_OFFSET_PX = 4.0
LIMB_MIN_ARC_PX = 10.0  # each sector spans at least this much limb, so it holds enough pixels
GLOW_SECTOR_DEG = 15.0  # glow is measured per sector: several prominence widths, yet follows scattered light
GLOW_FIELD_RING_PX = 2.0
GLOW_SAMPLE_LIMIT = 1_500_000  # pixels used to measure the glow; more adds time, not accuracy
FAR_SKY_OFFSET_PX = 200.0


@dataclass
class FinishParams:
    """What to do to the finished mosaic.  Defaults leave it exactly as built."""

    flip: bool = False  # mirror left-to-right, applied before the rotation
    rotation_deg: float = 0.0  # counter-clockwise, about the solar centre
    square: bool = False  # pad to a square with the Sun in the middle
    prominence_boost: float = 1.0  # 1 = off
    prominence_border_px: float = 1.0
    prominence_feather_px: float = 4.0
    remove_glow: bool = True

    @property
    def changes_geometry(self) -> bool:
        return self.flip or abs(self.rotation_deg) > 1e-9 or self.square

    @property
    def boosts(self) -> bool:
        return self.prominence_boost > 1.0 + 1e-9

    def key(self) -> tuple:
        return (self.flip, round(self.rotation_deg, 4), self.square,
                round(self.prominence_boost, 3), self.prominence_border_px,
                self.prominence_feather_px, self.remove_glow)


@dataclass
class FinishedImage:
    image: np.ndarray  # uint16, linear
    coverage: np.ndarray  # bool, True where real data
    sun_center: tuple[float, float]
    sun_radius: float
    prominences: np.ndarray | None  # uint16 composite, only when boosted
    filled_fraction: float
    params: FinishParams
    warnings: list[str] = field(default_factory=list)
    # The two layers the composite is made of, kept so the file can be edited as layers:
    # the boosted copy underneath, and a mask (65535 shows the untouched image) for the top.
    prominence_layer: np.ndarray | None = None  # uint16 boosted copy, only when boosted
    disk_mask: np.ndarray | None = None  # uint16, 65535 on the disk, 0 far outside it

    @property
    def side(self) -> tuple[int, int]:
        return int(self.image.shape[1]), int(self.image.shape[0])


# --- geometry ----------------------------------------------------------------------------


def _radius_map(shape: tuple[int, ...], center: tuple[float, float]) -> np.ndarray:
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]].astype(np.float32)
    return np.hypot(xx - center[0], yy - center[1])


def flip_horizontal(
    image: np.ndarray, coverage: np.ndarray, center: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    """Mirror left-to-right.  Lossless."""
    width = image.shape[1]
    return (
        np.ascontiguousarray(np.fliplr(image)),
        np.ascontiguousarray(np.fliplr(coverage)),
        (width - 1 - center[0], center[1]),
    )


def _is_right_angle(angle_deg: float) -> bool:
    return abs(angle_deg / 90.0 - round(angle_deg / 90.0)) < 1e-9


def _rotate_lossless(
    image: np.ndarray, coverage: np.ndarray, center: tuple[float, float], quarter_turns: int,
    out_shape: tuple[int, int], target: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    """Multiples of 90 degrees: rotate with ``np.rot90`` and paste, no resampling at all.

    The pivot is the Sun centre rounded to a whole pixel, so the disk may sit up to half a
    pixel off the square's centre.  That is the price of a turn that touches no value.
    """
    height, width = image.shape[:2]
    ys, xs = np.mgrid[0:height, 0:width]
    turned = np.rot90(image, quarter_turns)
    turned_cov = np.rot90(coverage, quarter_turns)
    turned_xs = np.rot90(xs, quarter_turns)
    turned_ys = np.rot90(ys, quarter_turns)
    cx, cy = int(round(center[0])), int(round(center[1]))
    hit = np.argwhere((turned_xs == cx) & (turned_ys == cy))
    pivot_y, pivot_x = (int(hit[0][0]), int(hit[0][1])) if len(hit) else (0, 0)

    out_h, out_w = out_shape
    tx, ty = int(round(target[0])), int(round(target[1]))
    off_x, off_y = tx - pivot_x, ty - pivot_y
    out = np.zeros(out_shape + image.shape[2:], np.float32)
    out_cov = np.zeros(out_shape, bool)
    src_x0, src_y0 = max(0, -off_x), max(0, -off_y)
    dst_x0, dst_y0 = max(0, off_x), max(0, off_y)
    w = min(turned.shape[1] - src_x0, out_w - dst_x0)
    h = min(turned.shape[0] - src_y0, out_h - dst_y0)
    if w > 0 and h > 0:
        out[dst_y0 : dst_y0 + h, dst_x0 : dst_x0 + w] = turned[src_y0 : src_y0 + h, src_x0 : src_x0 + w]
        out_cov[dst_y0 : dst_y0 + h, dst_x0 : dst_x0 + w] = turned_cov[src_y0 : src_y0 + h, src_x0 : src_x0 + w]
    return out, out_cov, (float(tx), float(ty))


def rotate_and_square(
    image: np.ndarray, coverage: np.ndarray, center: tuple[float, float],
    angle_deg: float, square: bool = True,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    """Turn the image counter-clockwise about the Sun centre; optionally pad to a square.

    With ``square`` the Sun lands in the middle of a square whose side is the larger of the
    two dimensions.  Without it the canvas keeps its size and the Sun stays where it was.
    Returns float32 data, a bool coverage, and the new Sun centre.
    """
    image = image.astype(np.float32, copy=False)
    height, width = image.shape[:2]
    if square:
        side = max(height, width)
        out_shape = (side, side)
        target = ((side - 1) / 2.0, (side - 1) / 2.0)
    else:
        out_shape = (height, width)
        target = center

    if _is_right_angle(angle_deg):
        quarter_turns = int(round(angle_deg / 90.0)) % 4
        return _rotate_lossless(image, coverage, center, quarter_turns, out_shape, target)

    matrix = cv2.getRotationMatrix2D((float(center[0]), float(center[1])), float(angle_deg), 1.0)
    matrix[0, 2] += target[0] - center[0]
    matrix[1, 2] += target[1] - center[1]
    size = (out_shape[1], out_shape[0])
    turned = cv2.warpAffine(
        image, matrix, size, flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
    )
    np.clip(turned, float(image.min()), float(image.max()), out=turned)
    cov = cv2.warpAffine(
        coverage.astype(np.uint8) * 255, matrix, size, flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    k = 2 * COVERAGE_ERODE_PX + 1
    cov = cv2.erode(cov, np.ones((k, k), np.uint8)) > 127
    return np.ascontiguousarray(turned), cov, (float(target[0]), float(target[1]))


# --- background ----------------------------------------------------------------------------


def sky_statistics(
    image: np.ndarray, coverage: np.ndarray, center: tuple[float, float], radius: float,
) -> tuple[float, float]:
    """Median and MAD of the real sky well away from the disk.

    Measured beyond ``FAR_SKY_OFFSET_PX`` from the limb when the image reaches that far.  A
    tightly cropped image does not, and then the outermost fifth of the sky it does show is
    used, which is the part least lit by the limb glow.  The disk is never part of the sample:
    including it once returned a "sky" four times too bright.
    """
    gray = image if image.ndim == 2 else image.mean(axis=2)
    rr = _radius_map(gray.shape, center)
    far = coverage & (rr > radius + FAR_SKY_OFFSET_PX)
    if far.sum() > 1000:
        sample = gray[far]
    else:
        outside = coverage & (rr > radius)
        if outside.sum() >= 100:
            sample = gray[outside & (rr >= np.percentile(rr[outside], 80))]
        else:
            sample = gray[coverage]
    if sample.size == 0:
        return 0.0, 1.0
    median = float(np.median(sample))
    mad = float(np.median(np.abs(sample - median)))
    return median, max(mad, 1e-3)


def fill_background(
    image: np.ndarray, coverage: np.ndarray, sky_mad: float, seed: int = 0,
    sigma: float = FILL_SMOOTH_SIGMA, noise_scale: float = FILL_NOISE_SCALE,
) -> np.ndarray:
    """Fill everything outside the real coverage so the sky stays continuous.

    The nearest real pixels are extended outward, smoothed so they do not streak, and given
    noise matched to the measured sky so the fill is not conspicuously flat.  Deterministic.
    """
    if bool(np.all(coverage)):
        return image.astype(np.float32, copy=False)
    extended = fill_uncovered(image.astype(np.float32, copy=False), coverage)
    smooth = cv2.GaussianBlur(extended, (0, 0), sigma)
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(smooth.shape).astype(np.float32) * float(sky_mad) * noise_scale
    sel = coverage[..., None] if image.ndim == 3 else coverage
    return np.where(sel, image, smooth + noise).astype(np.float32)


# --- prominences ---------------------------------------------------------------------------


@dataclass
class LimbModel:
    """Where the real limb is, and how bright the glow is beyond it."""

    offsets: np.ndarray  # px the real edge sits outside (+) or inside (-) the fitted circle, per sector
    profile: np.ndarray  # glow per angular sector and ring beyond the real edge, (sectors, rings)
    sector_deg: float = LIMB_SECTOR_DEG
    ring_px: float = GLOW_FIELD_RING_PX
    glow_sector_deg: float = GLOW_SECTOR_DEG


def sector_width(radius: float) -> float:
    """Sector size in degrees: 2, or wider on a small disk so every sector holds enough pixels."""
    if radius <= 0:
        return LIMB_SECTOR_DEG
    wanted = max(LIMB_SECTOR_DEG, float(np.degrees(LIMB_MIN_ARC_PX / radius)))
    return 360.0 / max(12, int(np.floor(360.0 / wanted)))


def limb_offsets(
    gray: np.ndarray, coverage: np.ndarray, center: tuple[float, float], radius: float,
    sector_deg: float | None = None,
) -> np.ndarray:
    """How far the real limb sits outside (+) or inside (-) the fitted circle, per sector.

    A mosaic's limb is not a perfect circle at the pixel level: seams, seeing and placement
    leave it wandering by a couple of pixels.  Where the falloff drops 2000 ADU per pixel,
    that is enough to turn a boosted copy into a jagged bright ring.  In each sector the edge
    is taken where brightness falls through the midpoint between the disk just inside the
    fitted limb and the sky, then smoothed around the circumference so a single prominence
    cannot drag it outward.  The number of sectors is ``len(result)``; each spans
    ``360 / len(result)`` degrees.
    """
    width = sector_width(radius) if sector_deg is None else float(sector_deg)
    n_sectors = int(round(360.0 / width))
    zeros = np.zeros(n_sectors, np.float32)
    rr = _radius_map(gray.shape, center)
    inside = coverage & (rr > radius - 2.0) & (rr <= radius)
    band = coverage & (rr > radius - LIMB_SEARCH_IN_PX) & (rr < radius + LIMB_SEARCH_OUT_PX)
    if inside.sum() < 50 or band.sum() < n_sectors * 20:
        return zeros
    sky, _ = sky_statistics(gray, coverage, center, radius)
    level = sky + 0.5 * (float(np.median(gray[inside])) - sky)

    ys, xs = np.nonzero(band)
    theta = np.degrees(np.arctan2(ys - center[1], xs - center[0])) % 360.0
    sector = np.minimum((theta / width).astype(np.int64), n_sectors - 1)
    step = 0.5
    n_rad = int(np.ceil((LIMB_SEARCH_IN_PX + LIMB_SEARCH_OUT_PX) / step))
    rbin = np.clip(((rr[ys, xs] - (radius - LIMB_SEARCH_IN_PX)) / step).astype(np.int64), 0, n_rad - 1)
    values = gray[ys, xs].astype(np.float32)
    key = sector * n_rad + rbin
    order = np.lexsort((values, key))
    key, values = key[order], values[order]
    cells = np.arange(n_sectors * n_rad)
    starts = np.searchsorted(key, cells)
    ends = np.searchsorted(key, cells, side="right")
    filled = ends > starts
    medians = np.full(cells.size, np.nan, np.float32)
    medians[filled] = values[(starts[filled] + ends[filled] - 1) // 2]
    medians = medians.reshape(n_sectors, n_rad)

    # A thin sector on a small disk, or a gap in coverage, leaves some radial cells empty.
    # Fill them along the radius, so a missing cell next to the limb cannot hide it.
    index = np.arange(n_rad)
    for row in range(n_sectors):
        good = np.isfinite(medians[row])
        if 2 <= good.sum() < n_rad:
            medians[row] = np.interp(index, index[good], medians[row, good])

    radii = radius - LIMB_SEARCH_IN_PX + index * step + step / 2.0
    crossing = (medians[:, :-1] >= level) & (medians[:, 1:] < level)
    # A sector can cross the level more than once: a dark filament just inside the limb, or a
    # bright prominence base just outside it.  The limb is by far the steepest of these, so
    # the crossing with the largest drop across it is the one taken.
    before = np.concatenate([medians[:, :1], medians[:, :-2]], axis=1)
    after = np.concatenate([medians[:, 2:], medians[:, -1:]], axis=1)
    drop = np.where(crossing & np.isfinite(before) & np.isfinite(after), before - after, -np.inf)
    # Only sectors with a usable crossing count.  argmax over a row with none would quietly
    # return the innermost cell and report a false edge 6 px inside the disk.
    rows = np.flatnonzero(np.isfinite(drop).any(axis=1))
    if rows.size == 0:
        return zeros
    first = np.argmax(drop[rows], axis=1)
    upper, lower = medians[rows, first], medians[rows, first + 1]
    frac = np.clip((upper - level) / np.maximum(upper - lower, 1e-6), 0.0, 1.0)
    edge = np.full(n_sectors, np.nan, np.float32)
    edge[rows] = radii[first] + frac * step

    offsets = edge - float(np.nanmedian(edge))
    valid = np.isfinite(offsets)
    if not valid.all():
        around = np.arange(n_sectors)
        offsets = np.interp(around, around[valid], offsets[valid], period=n_sectors)
    offsets = np.clip(offsets, -LIMB_MAX_OFFSET_PX, LIMB_MAX_OFFSET_PX)
    offsets = np.median(np.stack([np.roll(offsets, k) for k in range(-2, 3)]), axis=0)
    offsets = (np.roll(offsets, -1) + offsets + np.roll(offsets, 1)) / 3.0
    return offsets.astype(np.float32)


def _offset_map(
    shape: tuple[int, ...], center: tuple[float, float], offsets: np.ndarray,
    sector_deg: float | None = None,
) -> np.ndarray:
    """Per-pixel limb offset, interpolated between sector centres around the circle."""
    if offsets.size == 0 or not np.any(offsets):
        return np.zeros(shape[:2], np.float32)
    width = 360.0 / offsets.size if sector_deg is None else float(sector_deg)
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]].astype(np.float32)
    theta = np.degrees(np.arctan2(yy - center[1], xx - center[0])) % 360.0
    n = offsets.size
    position = theta / width - 0.5
    base = np.floor(position)
    frac = (position - base).astype(np.float32)
    k0 = base.astype(np.int64) % n
    return ((1.0 - frac) * offsets[k0] + frac * offsets[(k0 + 1) % n]).astype(np.float32)


def glow_profile(
    image: np.ndarray, coverage: np.ndarray, center: tuple[float, float], radius: float,
    ring_px: float = GLOW_RING_PX, offsets: np.ndarray | None = None,
    sector_deg: float | None = None,
) -> np.ndarray:
    """Median brightness around the disk, one value per ring beyond the limb.

    Distance is measured from the real limb when ``offsets`` are given, otherwise from the
    fitted circle.  Prominences are local, so they barely move a median taken all the way
    round, while the scattered-light glow is the same all round and is captured exactly.
    """
    gray = image if image.ndim == 2 else image.mean(axis=2)
    distance = _radius_map(gray.shape, center) - radius
    if offsets is not None:
        distance = distance - _offset_map(gray.shape, center, offsets, sector_deg)
    outside = coverage & (distance > 0)
    if outside.sum() < 100:
        return np.zeros(1, np.float32)
    bins = np.floor(distance[outside] / ring_px).astype(np.int64)
    values = gray[outside]
    order = np.argsort(bins, kind="stable")
    bins, values = bins[order], values[order]
    n_bins = int(bins[-1]) + 1
    starts = np.searchsorted(bins, np.arange(n_bins))
    ends = np.searchsorted(bins, np.arange(n_bins), side="right")
    profile = np.full(n_bins, np.nan, np.float32)
    for k in np.flatnonzero(ends > starts):
        profile[k] = np.median(values[starts[k] : ends[k]])
    good = ~np.isnan(profile)
    if not good.any():
        return np.zeros(1, np.float32)
    profile = np.interp(np.arange(n_bins), np.flatnonzero(good), profile[good]).astype(np.float32)
    return cv2.GaussianBlur(profile.reshape(-1, 1), (0, 0), 0.7).ravel()


def evaluate_glow(profile: np.ndarray, distance: np.ndarray, ring_px: float = GLOW_RING_PX) -> np.ndarray:
    """Glow at each pixel, interpolated between ring centres so there are no steps."""
    centres = (np.arange(profile.size) + 0.5) * ring_px
    return np.interp(distance, centres, profile).astype(np.float32)


def _angle_map(shape: tuple[int, ...], center: tuple[float, float]) -> np.ndarray:
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]].astype(np.float32)
    return (np.degrees(np.arctan2(yy - center[1], xx - center[0])) % 360.0).astype(np.float32)


def glow_field(
    image: np.ndarray, coverage: np.ndarray, center: tuple[float, float], radius: float,
    offsets: np.ndarray | None = None, sector_deg: float = GLOW_SECTOR_DEG,
    ring_px: float = GLOW_FIELD_RING_PX,
) -> np.ndarray:
    """Median brightness beyond the limb per angular sector and ring, shape (sectors, rings).

    Scattered light is not the same all round the disk.  On the sample frames the sky 30 to
    50 px beyond the limb ranged from 1 226 to 1 962 ADU between 30-degree sectors, which a
    x10 boost turned into broad bright arcs; right at the limb it left dark dashes wherever
    the local glow outshone the ring average.  Measuring the glow per sector follows it.  A
    15-degree sector is still several times wider than a prominence, so the median over it
    ignores the prominence and keeps it as signal.
    """
    gray = image if image.ndim == 2 else image.mean(axis=2)
    distance = _radius_map(gray.shape, center) - radius
    if offsets is not None:
        distance = distance - _offset_map(gray.shape, center, offsets)
    ys, xs = np.nonzero(coverage & (distance > 0))
    if ys.size < 100:
        return np.zeros((1, 1), np.float32)
    step = max(1, int(np.ceil(ys.size / GLOW_SAMPLE_LIMIT)))
    ys, xs = ys[::step], xs[::step]
    d = distance[ys, xs]
    theta = np.degrees(np.arctan2(ys - center[1], xs - center[0])) % 360.0
    n_sectors = int(round(360.0 / sector_deg))
    n_rings = int(d.max() // ring_px) + 1
    sector = np.minimum((theta / sector_deg).astype(np.int64), n_sectors - 1)
    ring = np.minimum((d / ring_px).astype(np.int64), n_rings - 1)
    values = gray[ys, xs].astype(np.float32)
    key = sector * n_rings + ring
    order = np.lexsort((values, key))
    key, values = key[order], values[order]
    cells = np.arange(n_sectors * n_rings)
    starts = np.searchsorted(key, cells)
    ends = np.searchsorted(key, cells, side="right")
    enough = (ends - starts) >= 5
    field = np.full(cells.size, np.nan, np.float32)
    field[enough] = values[(starts[enough] + ends[enough] - 1) // 2]
    field = field.reshape(n_sectors, n_rings)

    # Cells with too few pixels: fill along the radius within the sector first, then, for a
    # ring a sector never reaches, from the neighbouring sectors around the circle.
    rings = np.arange(n_rings)
    for row in range(n_sectors):
        good = np.isfinite(field[row])
        if 2 <= good.sum() < n_rings:
            field[row] = np.interp(rings, rings[good], field[row, good])
    around = np.arange(n_sectors)
    for column in range(n_rings):
        good = np.isfinite(field[:, column])
        if good.any() and not good.all():
            field[:, column] = np.interp(around, around[good], field[good, column], period=n_sectors)
    if not np.isfinite(field).all():
        fallback = float(np.nanmedian(field)) if np.isfinite(field).any() else 0.0
        field = np.where(np.isfinite(field), field, fallback).astype(np.float32)
    return cv2.GaussianBlur(field, (5, 1), 0.7).astype(np.float32)


def evaluate_glow_field(
    field: np.ndarray, distance: np.ndarray, theta: np.ndarray,
    sector_deg: float = GLOW_SECTOR_DEG, ring_px: float = GLOW_FIELD_RING_PX,
) -> np.ndarray:
    """Glow at each pixel, interpolated in angle (around the circle) and in distance."""
    n_sectors, n_rings = field.shape
    if field.size == 1:
        return np.full(distance.shape, float(field[0, 0]), np.float32)
    position = theta / sector_deg - 0.5
    base = np.floor(position)
    across = (position - base).astype(np.float32)
    k0 = base.astype(np.int64) % n_sectors
    k1 = (k0 + 1) % n_sectors
    radial = np.clip(distance / ring_px - 0.5, 0.0, n_rings - 1)
    r0 = np.floor(radial).astype(np.int64)
    out = (radial - r0).astype(np.float32)
    r1 = np.minimum(r0 + 1, n_rings - 1)
    near_side = (1.0 - out) * field[k0, r0] + out * field[k0, r1]
    far_side = (1.0 - out) * field[k1, r0] + out * field[k1, r1]
    return ((1.0 - across) * near_side + across * far_side).astype(np.float32)


def limb_model(
    image: np.ndarray, coverage: np.ndarray, center: tuple[float, float], radius: float,
    with_glow: bool = True,
) -> LimbModel:
    gray = image if image.ndim == 2 else image.mean(axis=2)
    offsets = limb_offsets(gray, coverage, center, radius)
    profile = (glow_field(image, coverage, center, radius, offsets=offsets)
               if with_glow else np.zeros((1, 1), np.float32))
    return LimbModel(offsets=offsets, profile=profile, sector_deg=360.0 / max(offsets.size, 1))


def prominence_layers(
    image: np.ndarray, center: tuple[float, float], radius: float, boost: float,
    border_px: float = 1.0, feather_px: float = 4.0, remove_glow: bool = True,
    coverage: np.ndarray | None = None, sky_level: float | None = None,
    model: LimbModel | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """The boosted copy and the mask that lays the untouched disk over it.

    Returns ``(boosted, mask)``: the image to put underneath, and a weight in 0..1 for the
    untouched image on top.  Brightens everything outside the disk; keeps the disk untouched.

    The protected region follows the real limb, measured around the circumference, plus
    ``border_px``; it is never smaller than the fitted circle plus ``border_px``, so no part
    of the disk is ever cut.  A feather of ``feather_px`` joins the two layers outside it.
    """
    image = image.astype(np.float32, copy=False)
    cov = coverage if coverage is not None else np.ones(image.shape[:2], bool)
    if model is None:
        model = limb_model(image, cov, center, radius, remove_glow)
    rr = _radius_map(image.shape, center)
    offset = _offset_map(image.shape, center, model.offsets, model.sector_deg)
    distance = rr - radius - offset  # beyond the real limb
    protect = rr - np.maximum(offset, 0.0)  # never protect less than the fitted circle
    edge = radius + border_px
    if feather_px > 0:
        mask = np.clip((edge - protect) / feather_px + 1.0, 0.0, 1.0).astype(np.float32)
    else:
        mask = (protect <= edge).astype(np.float32)

    outer = image.copy()
    if remove_glow:
        if sky_level is None:
            sky_level, _ = sky_statistics(image, cov, center, radius)
        glow = evaluate_glow_field(model.profile, distance, _angle_map(image.shape, center),
                                   model.glow_sector_deg, model.ring_px)
        outside = distance > 0
        if image.ndim == 3:
            glow = glow[..., None]
            outside = outside[..., None]
        # What is left once the glow is removed is the prominence signal.  Flooring it at
        # zero means nothing can end up darker than the boosted sky, so there is no dark rim.
        excess = np.maximum(image - glow, 0.0)
        outer = np.where(outside, excess + float(sky_level), image)
    outer = np.clip(outer * float(boost), 0.0, 65535.0)

    if sky_level is None:
        sky_level, _ = sky_statistics(image, cov, center, radius)
    floor = min(float(sky_level) * float(boost), 65535.0)
    # Nothing outside the untouched disk may be darker than the boosted sky.  A plain blend
    # across the soft edge passes through the original limb falloff, which at a strong boost
    # is dimmer than the boosted sky and shows as a dark ring round the disk.  The joint must
    # come out as if the original were lifted to the boosted sky there, but the top layer has
    # to stay an exact copy of the image, so the mask carries the correction instead: where
    # the original is dimmer than that floor, the mask lets a little more of the boosted copy
    # through, by exactly the amount that gives the lifted result.  Inside the protected
    # region (weight 1) and beyond the soft edge (weight 0) the mask is left as it is.
    gray = image if image.ndim == 2 else image.mean(axis=2)
    boosted = outer if outer.ndim == 2 else outer.mean(axis=2)
    feathered = (mask > 0.0) & (mask < 1.0) & (gray < floor)
    span = gray - boosted
    solvable = feathered & (np.abs(span) > 1e-3)
    target = mask * floor + (1.0 - mask) * boosted
    adjusted = (target - boosted) / np.where(solvable, span, 1.0)
    mask = np.where(solvable, np.minimum(np.clip(adjusted, 0.0, 1.0), mask), mask)
    return outer.astype(np.float32), mask.astype(np.float32)


def prominence_composite(
    image: np.ndarray, center: tuple[float, float], radius: float, boost: float,
    border_px: float = 1.0, feather_px: float = 4.0, remove_glow: bool = True,
    coverage: np.ndarray | None = None, sky_level: float | None = None,
    model: LimbModel | None = None,
) -> np.ndarray:
    """The untouched disk laid over the boosted copy, as one image (see ``prominence_layers``).

    The protected region follows the real limb, measured around the circumference, plus
    ``border_px``; it is never smaller than the fitted circle plus ``border_px``, so no part
    of the disk is ever cut.  A feather of ``feather_px`` joins the two layers outside it.
    """
    image = image.astype(np.float32, copy=False)
    outer, mask = prominence_layers(image, center, radius, boost, border_px, feather_px,
                                    remove_glow, coverage, sky_level, model)
    return blend_layers(image, outer, mask)


def blend_layers(top: np.ndarray, bottom: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Normal blending of ``top`` over ``bottom`` through ``mask`` (0..1), as float32."""
    weight = mask[..., None] if top.ndim == 3 else mask
    return (weight * top + (1.0 - weight) * bottom).astype(np.float32)


def blend_layers_uint16(top: np.ndarray, bottom: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """The same blend on 16-bit layers and a 16-bit mask, as an editor recomposes the file."""
    weight = mask.astype(np.float64) / 65535.0
    if top.ndim == 3:
        weight = weight[..., None]
    mixed = weight * top.astype(np.float64) + (1.0 - weight) * bottom.astype(np.float64)
    return np.clip(np.rint(mixed), 0, 65535).astype(np.uint16)


# --- the whole step ------------------------------------------------------------------------


def _to_uint16(image: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(image), 0, 65535).astype(np.uint16)


BOOSTED_FILL_REACH_PX = 32


def _refill_boosted(
    composite: np.ndarray, coverage: np.ndarray, sky_level: float, sky_mad: float,
    params: FinishParams, seed: int,
) -> np.ndarray:
    """Fill the no-data areas of the boosted layer with boosted real sky taken from nearby.

    Boosting multiplies any small difference between a synthetic fill and the real sky into a
    visible outline.  A flat fill cannot win: near the canvas edges the real sky can sit just
    under the glow model, so boosting floors it almost flat, while elsewhere it keeps its
    noise.  Each empty pixel therefore copies a real boosted pixel from just inside the data,
    picked at random within ``BOOSTED_FILL_REACH_PX`` along and into the boundary.  The fill
    then has the local tone and noise by construction, and random picks cannot streak.
    ``sky_level`` and ``sky_mad`` are unused but kept so both callers pass the same facts.
    """
    del sky_level, sky_mad, params
    if bool(np.all(coverage)) or not bool(np.any(coverage)):
        return composite
    height, width = coverage.shape
    source = (~coverage).astype(np.uint8) * 255
    _, labels = cv2.distanceTransformWithLabels(
        source, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL
    )
    cov_y, cov_x = np.nonzero(coverage)
    lookup = np.zeros(int(labels.max()) + 1, np.int64)
    lookup[labels[coverage]] = np.arange(cov_y.size)
    empty_y, empty_x = np.nonzero(~coverage)
    nearest = lookup[labels[empty_y, empty_x]]
    near_y = cov_y[nearest].astype(np.float32)
    near_x = cov_x[nearest].astype(np.float32)

    # inward normal from the empty pixel to its nearest real pixel, and the tangent along the edge
    ny = near_y - empty_y
    nx = near_x - empty_x
    norm = np.maximum(np.hypot(ny, nx), 1e-6)
    ny, nx = ny / norm, nx / norm
    ty, tx = -nx, ny

    rng = np.random.default_rng(seed)
    depth = rng.uniform(0.0, BOOSTED_FILL_REACH_PX, empty_y.size).astype(np.float32)
    along = rng.uniform(-BOOSTED_FILL_REACH_PX, BOOSTED_FILL_REACH_PX, empty_y.size).astype(np.float32)
    pick_y = np.clip(np.rint(near_y + ny * depth + ty * along), 0, height - 1).astype(np.int64)
    pick_x = np.clip(np.rint(near_x + nx * depth + tx * along), 0, width - 1).astype(np.int64)
    inside = coverage[pick_y, pick_x]
    pick_y = np.where(inside, pick_y, cov_y[nearest])
    pick_x = np.where(inside, pick_x, cov_x[nearest])

    out = composite.copy()
    out[empty_y, empty_x] = composite[pick_y, pick_x]
    return out


def finish(result: MosaicResult, params: FinishParams | None = None, seed: int = 0) -> FinishedImage:
    """Apply the finishing steps to a built mosaic.  Default params return it as built."""
    params = params or FinishParams()
    warnings: list[str] = []
    image = result.mosaic.astype(np.float32)
    coverage = result.label_map >= 0
    center = (float(result.sun_center[0]), float(result.sun_center[1]))
    radius = float(result.sun_radius)
    if radius <= 0:
        height, width = image.shape[:2]
        center = ((width - 1) / 2.0, (height - 1) / 2.0)
        warnings.append(
            "the solar limb could not be measured on the mosaic; the image centre is used "
            "for turning and for the prominence mask"
        )

    sky_level, sky_mad = sky_statistics(image, coverage, center, radius)

    if params.flip:
        image, coverage, center = flip_horizontal(image, coverage, center)
    if abs(params.rotation_deg) > 1e-9 or params.square:
        image, coverage, center = rotate_and_square(
            image, coverage, center, params.rotation_deg, params.square
        )
    if params.changes_geometry:
        image = fill_background(image, coverage, sky_mad, seed)

    filled_fraction = float(1.0 - coverage.mean())
    prominences = prominence_layer = disk_mask = None
    if params.boosts:
        if radius <= 0:
            warnings.append("prominences were not boosted because the disk could not be located")
        else:
            outer, weight = prominence_layers(
                image, center, radius, params.prominence_boost,
                params.prominence_border_px, params.prominence_feather_px,
                params.remove_glow, coverage, sky_level,
            )
            if not bool(coverage.all()):
                outer = _refill_boosted(outer, coverage, sky_level, sky_mad, params, seed)
            prominence_layer = _to_uint16(outer)
            disk_mask = np.clip(np.rint(weight * 65535.0), 0, 65535).astype(np.uint16)
            # Built from the 16-bit layers themselves, so the flat image is exactly what an
            # editor shows when it opens the layers.
            prominences = blend_layers_uint16(_to_uint16(image), prominence_layer, disk_mask)

    return FinishedImage(
        image=_to_uint16(image), coverage=coverage, sun_center=center, sun_radius=radius,
        prominences=prominences, filled_fraction=filled_fraction, params=params,
        warnings=warnings, prominence_layer=prominence_layer, disk_mask=disk_mask,
    )


# --- composing orientations and live previews ---------------------------------------------


def compose_orientation(
    applied_flip: bool, applied_angle: float, pending_flip: bool, pending_angle: float,
) -> tuple[bool, float]:
    """The single flip-then-turn equivalent to an applied orientation followed by a new one.

    Every orientation is stored as "mirror if asked, then turn".  Mirroring reverses the sense
    of any turn already made, so a new flip on top negates the angle applied before it.  This
    is what lets Apply always resample the built mosaic once, however many times it is used.
    """
    angle = pending_angle + (-applied_angle if pending_flip else applied_angle)
    angle = (angle + 180.0) % 360.0 - 180.0
    if abs(angle + 180.0) < 1e-9:
        angle = 180.0
    return applied_flip != pending_flip, float(angle)


@dataclass
class PreviewSource:
    """A downsampled, still linear copy of an image, for fast on-screen rendering."""

    image: np.ndarray  # float32
    coverage: np.ndarray  # bool
    center: tuple[float, float]
    radius: float
    scale: float
    sky_level: float
    stretch: tuple[float, float]
    sky_mad: float = 1.0
    model: LimbModel | None = None


def make_preview_source(
    image: np.ndarray, coverage: np.ndarray, center: tuple[float, float], radius: float,
    max_px: int = 1600,
) -> PreviewSource:
    """Downsample once; the display stretch is fixed here from the unboosted disk."""
    from .io import stretch_limits

    full = image.astype(np.float32, copy=False)
    scale = min(1.0, max_px / max(full.shape[:2]))
    if scale < 1.0:
        size = (int(round(full.shape[1] * scale)), int(round(full.shape[0] * scale)))
        small = cv2.resize(full, size, interpolation=cv2.INTER_AREA)
        small_cov = cv2.resize(coverage.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
    else:
        small, small_cov = full, coverage.copy()
    small_center = (center[0] * scale, center[1] * scale)
    small_radius = radius * scale
    sky, mad = sky_statistics(small, small_cov, small_center, small_radius)
    stretch = stretch_limits(small, mask=small_cov.astype(np.uint8) * 255)
    return PreviewSource(small, small_cov, small_center, small_radius, scale, sky, stretch, mad)


def render_preview(source: PreviewSource, params: FinishParams) -> np.ndarray:
    """What ``finish`` would produce, at preview scale and in a fraction of the time.

    Exposed areas are shown at the sky level rather than with the full smoothed fill, which
    only the full-resolution Apply computes.  Returns linear float32 at preview scale.
    """
    image, coverage, center = source.image, source.coverage, source.center
    geometry_changed = params.changes_geometry
    if params.flip:
        image, coverage, center = flip_horizontal(image, coverage, center)
    if abs(params.rotation_deg) > 1e-9 or params.square:
        image, coverage, center = rotate_and_square(
            image, coverage, center, params.rotation_deg, params.square
        )
    if geometry_changed:
        image = np.where(coverage, image, np.float32(source.sky_level)).astype(np.float32)
    if params.boosts and source.radius > 0:
        model = None
        if not geometry_changed:
            stale = source.model is None or (params.remove_glow and source.model.profile.size <= 1)
            if stale:
                source.model = limb_model(image, coverage, center, source.radius, with_glow=True)
            model = source.model
        outer, weight = prominence_layers(
            image, center, source.radius, params.prominence_boost,
            params.prominence_border_px * source.scale,
            params.prominence_feather_px * source.scale,
            params.remove_glow, coverage, source.sky_level, model,
        )
        if not bool(coverage.all()):
            # Same treatment as the saved file, so what is on screen is what gets written.
            outer = _refill_boosted(outer, coverage, source.sky_level, source.sky_mad, params, 0)
        image = blend_layers(image, outer, weight)
    return image
