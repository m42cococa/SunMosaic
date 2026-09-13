"""The whole mosaic in one call.  No UI imports live here, on purpose."""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np

from . import blend as blend_mod
from . import io as io_mod
from . import photometric, preprocess, register, warp
from .errors import SunMosaicError
from .limb import cross_check, fit_limb
from .types import MosaicResult, Params, Tile

ProgressCallback = Callable[[str, float], None]

STEPS = ["Reading files", "Finding the limb", "Aligning frames",
         "Matching brightness", "Blending", "Finishing"]


def _noop(step: str, fraction: float) -> None:  # pragma: no cover - default callback
    pass


def stitch(
    tiles: list[Tile], params: Params | None = None, progress: ProgressCallback | None = None,
) -> MosaicResult:
    """Register, equalise and blend ``tiles`` into one full-disk mosaic."""
    params = params or Params()
    report = progress or _noop
    timings: dict[str, float] = {}
    warnings: list[str] = []
    clock = time.perf_counter()

    def mark(step: str) -> None:
        timings[step] = time.perf_counter() - clock

    io_mod.validate_tiles(tiles)
    for tile in tiles:
        warnings.extend(f"{tile.name}: {w}" for w in tile.warnings)

    report(STEPS[0], 0.05)
    for tile in tiles:
        preprocess.prepare(tile, params.highpass_sigma)
    mark(STEPS[0])

    report(STEPS[1], 0.20)
    limbs = [fit_limb(t, params.min_arc_deg, params.max_limb_resid) for t in tiles]
    warnings.extend(cross_check(limbs))
    if all(lf is None for lf in limbs):
        warnings.append(
            "no frame shows a clear solar limb; alignment relies on image texture alone"
        )
    mark(STEPS[1])

    report(STEPS[2], 0.35)
    placements, pairs, canvas_size, place_warnings = register.build_placements(tiles, limbs, params)
    warnings.extend(place_warnings)

    if params.refine_rotation and register.needs_rotation(tiles, pairs, params):
        report("Correcting rotation", 0.45)
        outcome = _refine_rotation(tiles, limbs, placements, pairs, params)
        if outcome is not None:
            tiles, limbs, placements, pairs, canvas_size, angles, rotation_warnings = outcome
            warnings = [w for w in warnings if w not in place_warnings]
            warnings.extend(rotation_warnings)
            spread = float(np.degrees(angles.max() - angles.min()))
            warnings.append(
                f"the frames were rotated relative to each other by up to {spread:.2f} "
                "degrees; they were turned into a common orientation before joining"
            )
    mark(STEPS[2])

    report(STEPS[3], 0.55)
    images, masks = warp.warp_all(tiles, placements, canvas_size, params.interp)
    disk_masks = [
        warp.warp_mask(t.disk_mask, p, canvas_size) for t, p in zip(tiles, placements)
    ]
    pedestal = float(np.median([t.sky_median for t in tiles]))
    images, gains, offsets, field, field_degree, photo_warnings = photometric.equalise(
        tiles, images, masks, disk_masks, pairs, placements, pedestal, params.equalize
    )
    warnings.extend(photo_warnings)
    for k, placement in enumerate(placements):
        placement.gain = float(gains[k])
        placement.offset = float(offsets[k])
    mark(STEPS[3])

    report(STEPS[4], 0.70)
    distances = blend_mod.distance_maps(masks)
    labels = blend_mod.seam_labels(distances)
    overlap_dims = [
        min(p.overlap_w, p.overlap_h) for p in pairs if p.accepted
    ] or [4 * 2**blend_mod.MIN_LEVELS]
    levels = blend_mod.choose_levels(overlap_dims, params.pyramid_levels)
    if params.blend == "multiband":
        mosaic = blend_mod.multiband_blend(
            images, masks, distances, labels, levels, fill_value=pedestal
        )
    elif params.blend == "feather":
        mosaic = blend_mod.feather_blend(images, distances, masks)
    else:
        mosaic = blend_mod.hard_blend(images, labels)
    mark(STEPS[4])

    report(STEPS[5], 0.90)
    covered = labels >= 0
    uncovered_px = int((~covered).sum())
    mosaic = blend_mod.fill_uncovered(mosaic, covered)
    mosaic = np.clip(np.rint(mosaic), 0, 65535).astype(np.uint16)
    if uncovered_px:
        warnings.append(
            f"{uncovered_px / mosaic[..., 0].size if mosaic.ndim == 3 else uncovered_px / mosaic.size:.1%} "
            "of the canvas corners lie outside every frame; the background there was "
            "extended from the nearest pixels and holds no real data"
        )

    sun_center, sun_radius = _refit_sun(mosaic, params)
    if sun_radius > 0:
        tile_radii = [lf.r for lf in limbs if lf is not None and lf.trusted]
        if tile_radii and abs(sun_radius - float(np.median(tile_radii))) > 0.02 * sun_radius:
            warnings.append(
                "the assembled disk is not quite circular; check the seams before publishing"
            )
    mark(STEPS[5])

    result = MosaicResult(
        mosaic=mosaic, native_dtype=np.dtype(np.uint16), native_max=65535.0,
        canvas_size=canvas_size, tile_names=[t.name for t in tiles], placements=placements,
        limbs=limbs, pairs=pairs, sun_center=sun_center, sun_radius=sun_radius,
        label_map=labels, pedestal=pedestal, uncovered_px=uncovered_px,
        field_coefficients=field, field_degree=field_degree,
        warnings=warnings, timings=timings,
    )
    report("Done", 1.0)
    return result


def _refine_rotation(tiles, limbs, placements, pairs, params):
    """Measure the angle between frames, turn them to match, and place them again.

    Returns ``None`` when no angle could be measured or when turning the frames did not
    help, in which case the translation-only placement stands.
    """
    origins = [(p.x, p.y) for p in placements]
    angles, measured = register.solve_rotations(tiles, origins, pairs, params)
    if measured < len(tiles) - 1:
        return None
    if float(np.degrees(np.abs(angles).max())) < register.MIN_USEFUL_ROTATION_DEG:
        return None

    rotated = [preprocess.prepare(t, params.highpass_sigma) for t in register.derotate(tiles, angles)]
    new_limbs = [fit_limb(t, params.min_arc_deg, params.max_limb_resid) for t in rotated]
    limb_warnings = cross_check(new_limbs)
    try:
        new_placements, new_pairs, new_canvas, place_warnings = register.build_placements(
            rotated, new_limbs, params
        )
    except SunMosaicError:
        return None

    def score(pair_list):
        accepted = [p for p in pair_list if p.accepted]
        worst = max((p.residual_px or 0.0) for p in accepted) if accepted else float("inf")
        return len(accepted), -worst

    if score(new_pairs) <= score(pairs):
        return None
    for placement, angle in zip(new_placements, angles):
        placement.rotation = float(angle)
    return (rotated, new_limbs, new_placements, new_pairs, new_canvas, angles,
            limb_warnings + place_warnings)


def _refit_sun(mosaic: np.ndarray, params: Params) -> tuple[tuple[float, float], float]:
    """Fit the limb of the finished mosaic, as a sanity check on the assembly."""
    gray = mosaic if mosaic.ndim == 2 else mosaic.mean(axis=2)
    probe = Tile(
        name="mosaic", data=gray.astype(np.float32), native_dtype=np.dtype(np.uint16),
        native_max=65535.0,
    )
    probe.disk_mask = preprocess.disk_mask(probe.data)
    fit = fit_limb(probe, params.min_arc_deg, max_resid=5.0)
    if fit is None:
        return (0.0, 0.0), 0.0
    return (fit.cx, fit.cy), fit.r


def stitch_paths(
    paths: list[str], params: Params | None = None, progress: ProgressCallback | None = None,
) -> MosaicResult:
    """Convenience wrapper used by the CLI and the folder picker."""
    return stitch([io_mod.read_tile(p) for p in paths], params, progress)
