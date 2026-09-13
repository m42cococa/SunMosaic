"""End-to-end runs against synthetic data, where the truth is known exactly."""

from __future__ import annotations

import numpy as np
import pytest

from sunmosaic.pipeline import stitch
from sunmosaic.types import Params


def _relative(placements):
    ox, oy = placements[0].x, placements[0].y
    return [(p.x - ox, p.y - oy) for p in placements]


def test_four_quadrants_are_assembled_accurately(quad_case):
    result = stitch(list(quad_case.tiles), Params())
    errors = [
        float(np.hypot(gx - tx, gy - ty))
        for (gx, gy), (tx, ty) in zip(_relative(result.placements), quad_case.relative_origins())
    ]
    assert max(errors) < 0.5, f"placement errors {errors}"
    assert result.sun_radius == pytest.approx(quad_case.radius, rel=0.02)
    assert result.mosaic.dtype == np.uint16
    assert result.worst_residual < 1.0
    assert all(p.source == "limb" for p in result.placements)


def test_fifth_central_frame_is_assembled(center_case):
    result = stitch(list(center_case.tiles), Params())
    errors = [
        float(np.hypot(gx - tx, gy - ty))
        for (gx, gy), (tx, ty) in zip(_relative(result.placements), center_case.relative_origins())
    ]
    assert max(errors) < 1.0, f"placement errors {errors}"
    assert result.placements[4].source in ("template", "phasecorr")


def test_mosaic_is_larger_than_any_frame(quad_case):
    result = stitch(list(quad_case.tiles), Params())
    width, height = result.canvas_size
    assert width > max(t.width for t in quad_case.tiles)
    assert height > max(t.height for t in quad_case.tiles)
    assert width >= 2 * quad_case.radius


def test_the_whole_disk_is_covered_by_real_data(quad_case):
    """No part of the disk may fall in the synthetic corner fill."""
    result = stitch(list(quad_case.tiles), Params())
    cx, cy = result.sun_center
    yy, xx = np.mgrid[0:result.mosaic.shape[0], 0:result.mosaic.shape[1]].astype(np.float32)
    disk = np.hypot(xx - cx, yy - cy) < result.sun_radius * 0.98
    assert (result.label_map[disk] >= 0).all()


@pytest.mark.parametrize("blend", ["multiband", "feather", "hard"])
def test_every_blend_mode_produces_a_sane_mosaic(quad_case, blend):
    result = stitch(list(quad_case.tiles), Params(blend=blend))
    assert result.mosaic.dtype == np.uint16
    assert int(result.mosaic.max()) > 10000
    assert np.isfinite(result.mosaic).all()


@pytest.mark.parametrize("interp", ["cubic", "lanczos", "linear", "integer"])
def test_every_interpolation_mode_runs(quad_case, interp):
    result = stitch(list(quad_case.tiles), Params(interp=interp))
    assert result.sun_radius == pytest.approx(quad_case.radius, rel=0.02)


def _seam_step(result) -> float:
    """Mean gradient magnitude along the seams, on the disk only."""
    import cv2

    labels = result.label_map
    edge = np.zeros(labels.shape, np.uint8)
    for index in range(int(labels.max()) + 1):
        edge |= cv2.morphologyEx(
            (labels == index).astype(np.uint8) * 255, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)
        )
    cx, cy = result.sun_center
    yy, xx = np.mgrid[0:labels.shape[0], 0:labels.shape[1]].astype(np.float32)
    disk = np.hypot(xx - cx, yy - cy) < result.sun_radius * 0.9
    band = (edge > 0) & disk
    gx, gy = np.gradient(result.mosaic.astype(np.float32))
    return float(np.hypot(gx, gy)[band].mean())


def test_illumination_field_reduces_the_seam_step(quad_case):
    """The whole point of the polynomial: a constant per frame cannot absorb a gradient."""
    constant = stitch(list(quad_case.tiles), Params(equalize="gain+offset", blend="hard"))
    linear = stitch(list(quad_case.tiles), Params(equalize="linear", blend="hard"))
    assert linear.field_degree == 1
    assert linear.field_coefficients is not None
    assert _seam_step(linear) < _seam_step(constant)


def test_illumination_field_stays_gentle(quad_case):
    result = stitch(list(quad_case.tiles), Params(equalize="quadratic"))
    assert result.field_coefficients is not None
    shape_terms = np.abs(result.field_coefficients[:, 1:])
    assert shape_terms.max() < 0.35, "the fitted field should stay a gentle correction"


def test_multiband_beats_hard_seams_on_smoothness(quad_case):
    """The blend must actually reduce the step at the seam it is there to hide.

    Measured with brightness matching switched off, so there is a real step to hide and the
    test does not silently depend on how good the photometric correction happens to be.
    """
    hard = stitch(list(quad_case.tiles), Params(blend="hard", equalize="off"))
    soft = stitch(list(quad_case.tiles), Params(blend="multiband", equalize="off"))
    labels = hard.label_map
    import cv2

    boundary = cv2.morphologyEx(
        (labels == 0).astype(np.uint8) * 255, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)
    )
    band = (boundary > 0) & (labels >= 0)
    hard_step = float(np.abs(np.gradient(hard.mosaic.astype(np.float32))[1][band]).mean())
    soft_step = float(np.abs(np.gradient(soft.mosaic.astype(np.float32))[1][band]).mean())
    assert soft_step < hard_step


def test_colour_frames_are_stitched(quad_case):
    """Colour input has a branch in every stage; run it once end to end."""
    from sunmosaic.types import Tile

    colour = [
        Tile(name=t.name, data=np.repeat(t.data[:, :, None], 3, axis=2).astype(np.float32),
             native_dtype=t.native_dtype, native_max=t.native_max)
        for t in quad_case.tiles
    ]
    result = stitch(colour, Params())
    assert result.mosaic.ndim == 3
    assert result.mosaic.shape[2] == 3
    assert result.mosaic.dtype == np.uint16
    assert result.sun_radius == pytest.approx(quad_case.radius, rel=0.02)
    # The three identical channels must stay identical through warping and blending.
    assert np.array_equal(result.mosaic[:, :, 0], result.mosaic[:, :, 1])


@pytest.mark.parametrize("spread,applied", [(0.75, (0.0, 0.25, 0.50, 0.75)),
                                            (2.40, (0.0, 0.8, 1.6, 2.4))])
def test_rotated_frames_are_straightened_end_to_end(spread, applied):
    """Field rotation is detected, measured and undone, and reported to the user."""
    from sunmosaic.synthetic import quadrant_case

    case = quadrant_case(seed=9, rotations_deg=applied)
    result = stitch(list(case.tiles), Params())
    assert sum(p.accepted for p in result.pairs) == 6
    assert result.worst_residual < 1.0
    turned = np.degrees([p.rotation for p in result.placements])
    assert turned.max() - turned.min() == pytest.approx(spread, abs=0.15)
    assert any("rotated relative to each other" in w for w in result.warnings)
    assert result.sun_radius == pytest.approx(case.radius, rel=0.02)


def test_rotation_refinement_can_be_switched_off():
    """Without it, a strongly rotated set stops matching; that is what it exists to fix."""
    from sunmosaic.synthetic import quadrant_case

    case = quadrant_case(seed=9, rotations_deg=(0.0, 0.8, 1.6, 2.4))
    result = stitch(list(case.tiles), Params(refine_rotation=False))
    assert sum(p.accepted for p in result.pairs) < 6
    assert all(p.rotation == 0.0 for p in result.placements)


def test_unrotated_frames_are_left_alone(quad_case):
    result = stitch(list(quad_case.tiles), Params())
    assert all(p.rotation == 0.0 for p in result.placements)
    assert not any("rotated relative to each other" in w for w in result.warnings)
