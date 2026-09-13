"""Finishing steps: flip, rotation to a square, background fill, and prominence boosting."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from sunmosaic.blend import fill_uncovered
from sunmosaic.finish import (
    FinishParams,
    compose_orientation,
    fill_background,
    finish,
    flip_horizontal,
    glow_profile,
    make_preview_source,
    prominence_composite,
    render_preview,
    rotate_and_square,
)
from sunmosaic.limb import fit_limb
from sunmosaic.preprocess import disk_mask
from sunmosaic.types import MosaicResult, Tile

SKY = 550.0
DISK = 30000.0
SIZE = 401  # odd, so the centre falls exactly on a pixel
CENTRE = (200.0, 200.0)
RADIUS = 150.0


def _radius(shape, centre):
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]].astype(np.float32)
    return np.hypot(xx - centre[0], yy - centre[1])


def _disk(noise: float = 0.0, seed: int = 0, mark: bool = True, halo: float = 0.0) -> np.ndarray:
    """A textured disk centred on a pixel, with an asymmetric mark so orientation is visible."""
    rng = np.random.default_rng(seed)
    rr = _radius((SIZE, SIZE), CENTRE)
    texture = cv2.GaussianBlur(rng.standard_normal((SIZE, SIZE)).astype(np.float32), (0, 0), 3.0)
    image = np.where(rr < RADIUS, DISK * (1.0 + 0.3 * texture), SKY).astype(np.float32)
    if halo:
        image += np.where(rr >= RADIUS, halo * np.exp(-(rr - RADIUS) / 10.0), 0.0).astype(np.float32)
    if mark:
        image[110:130, 170:190] = 60000.0  # upper middle, inside the disk
    if noise:
        image += rng.standard_normal(image.shape).astype(np.float32) * noise
    return image


def _result(image: np.ndarray, centre=CENTRE, radius=RADIUS, coverage=None) -> MosaicResult:
    """Wrap an image as if the pipeline had built it."""
    height, width = image.shape[:2]
    labels = np.zeros((height, width), np.int16) if coverage is None else np.where(coverage, 0, -1).astype(np.int16)
    return MosaicResult(
        mosaic=np.clip(np.rint(image), 0, 65535).astype(np.uint16), native_dtype=np.dtype(np.uint16),
        native_max=65535.0, canvas_size=(width, height), tile_names=["synthetic.tif"],
        placements=[], limbs=[], pairs=[], sun_center=centre, sun_radius=radius,
        label_map=labels, pedestal=SKY, uncovered_px=int((labels < 0).sum()),
    )


def _refit(image: np.ndarray):
    tile = Tile(name="t", data=image.astype(np.float32), native_dtype=np.dtype(np.uint16),
                native_max=65535.0)
    tile.disk_mask = disk_mask(tile.data)
    return fit_limb(tile, max_resid=5.0)


# --- defaults --------------------------------------------------------------------------------


def test_default_params_change_nothing():
    result = _result(_disk(noise=20))
    finished = finish(result)
    assert np.array_equal(finished.image, result.mosaic)
    assert finished.prominences is None
    assert not FinishParams().changes_geometry
    assert not FinishParams().boosts


# --- flip ------------------------------------------------------------------------------------


def test_flip_twice_is_the_identity():
    image = _disk(noise=20)
    coverage = np.ones(image.shape, bool)
    once = flip_horizontal(image, coverage, (180.3, 200.0))
    twice = flip_horizontal(*once)
    assert np.array_equal(twice[0], image)
    assert twice[2] == pytest.approx((180.3, 200.0))


def test_flip_mirrors_a_mark_and_the_centre():
    image = np.zeros((50, 80), np.float32)
    image[10, 5] = 1.0
    flipped, _, centre = flip_horizontal(image, np.ones(image.shape, bool), (20.0, 25.0))
    assert flipped[10, 80 - 1 - 5] == 1.0
    assert centre == pytest.approx((80 - 1 - 20.0, 25.0))


def test_flip_is_applied_before_the_turn():
    """Flip then turn is not turn then flip; the code must do the documented order."""
    image = _disk()
    coverage = np.ones(image.shape, bool)
    finished = finish(_result(image), FinishParams(flip=True, rotation_deg=90.0, square=True))
    flipped = flip_horizontal(image, coverage, CENTRE)
    expected, _, _ = rotate_and_square(*flipped, 90.0, square=True)
    assert np.array_equal(finished.image, np.clip(np.rint(expected), 0, 65535).astype(np.uint16))
    other_order, cov, c = rotate_and_square(image, coverage, CENTRE, 90.0, square=True)
    other_order = flip_horizontal(other_order, cov, c)[0]
    assert not np.array_equal(finished.image, np.clip(np.rint(other_order), 0, 65535).astype(np.uint16))


# --- rotation ----------------------------------------------------------------------------------


@pytest.mark.parametrize("quarter", [1, 2, 3, -1])
def test_right_angles_are_lossless(quarter):
    image = _disk(noise=20)
    turned, coverage, centre = rotate_and_square(
        image, np.ones(image.shape, bool), CENTRE, 90.0 * quarter, square=True
    )
    assert np.array_equal(turned, np.rot90(image, quarter % 4))
    assert coverage.all()
    assert centre == pytest.approx(CENTRE)


def test_counter_clockwise_is_positive():
    """A quarter turn counter-clockwise sends what was above the centre to its left.

    Checked against the exact mapping from the mark's own centroid: a point at (x, y) lands
    at (y, size - 1 - x).
    """
    image = _disk()
    before_y, before_x = (float(v.mean()) for v in np.nonzero(image >= 59000.0))
    turned, _, _ = rotate_and_square(image, np.ones(image.shape, bool), CENTRE, 90.0, square=True)
    after_y, after_x = (float(v.mean()) for v in np.nonzero(turned >= 59000.0))
    assert before_y < CENTRE[1] - 40, "the mark starts above the centre"
    assert after_x == pytest.approx(before_y)
    assert after_y == pytest.approx(SIZE - 1 - before_x)
    assert after_x < CENTRE[0] - 40, "and ends up on its left"


def test_turning_there_and_back_preserves_the_disk():
    image = _disk()
    coverage = np.ones(image.shape, bool)
    there = rotate_and_square(image, coverage, CENTRE, 30.0, square=True)
    back, _, _ = rotate_and_square(*there, -30.0, square=True)
    inside = _radius(image.shape, CENTRE) < RADIUS - 10
    inside &= ~((np.abs(np.arange(SIZE)[:, None] - 120) < 25) & (np.abs(np.arange(SIZE)[None, :] - 180) < 25))
    rms = float(np.sqrt(np.mean((back[inside] - image[inside]) ** 2)))
    assert rms / float(image[inside].mean()) < 0.01


@pytest.mark.parametrize("angle", [17.0, 30.0, -135.0])
def test_the_sun_lands_in_the_middle_of_the_square(angle):
    # a wide canvas with the Sun off centre, as a real mosaic has
    canvas = np.full((420, 560), SKY, np.float32)
    canvas[10 : 10 + SIZE, 60 : 60 + SIZE] = _disk(mark=False)
    centre = (60 + CENTRE[0], 10 + CENTRE[1])
    turned, _, new_centre = rotate_and_square(canvas, np.ones(canvas.shape, bool), centre, angle, True)
    side = max(canvas.shape)
    assert turned.shape == (side, side)
    assert new_centre == pytest.approx(((side - 1) / 2.0, (side - 1) / 2.0))
    fit = _refit(turned)
    assert fit is not None
    assert fit.cx == pytest.approx(new_centre[0], abs=0.5)
    assert fit.cy == pytest.approx(new_centre[1], abs=0.5)
    assert fit.r == pytest.approx(RADIUS, abs=0.75)


def test_turning_without_squaring_keeps_the_canvas():
    canvas = np.full((300, 500), SKY, np.float32)
    turned, _, centre = rotate_and_square(canvas, np.ones(canvas.shape, bool), (250.0, 150.0), 12.0, False)
    assert turned.shape == canvas.shape
    assert centre == (250.0, 150.0)


def test_no_value_escapes_the_source_range():
    image = _disk(noise=50)
    turned, _, _ = rotate_and_square(image, np.ones(image.shape, bool), CENTRE, 33.3, True)
    assert turned.max() <= image.max()
    assert turned.min() >= min(image.min(), 0.0)


# --- composing orientations ----------------------------------------------------------------------


@pytest.mark.parametrize("first,second", [
    ((False, 90.0), (False, 90.0)),
    ((True, 90.0), (False, -180.0)),
    ((False, 90.0), (True, 90.0)),
    ((True, -90.0), (True, 180.0)),
])
def test_orientations_compose_into_one(first, second):
    """Applying one orientation and then another equals the single composed one, exactly."""
    image = _disk()
    coverage = np.ones(image.shape, bool)
    staged = (image, coverage, CENTRE)
    for flip, angle in (first, second):
        if flip:
            staged = flip_horizontal(*staged)
        staged = rotate_and_square(*staged, angle, square=True)
    flip, angle = compose_orientation(first[0], first[1], second[0], second[1])
    direct = (image, coverage, CENTRE)
    if flip:
        direct = flip_horizontal(*direct)
    direct = rotate_and_square(*direct, angle, square=True)
    assert np.array_equal(staged[0], direct[0])


def test_composed_angle_stays_in_range():
    assert compose_orientation(False, 170.0, False, 30.0) == (False, pytest.approx(-160.0))
    assert compose_orientation(False, 90.0, False, 90.0) == (False, pytest.approx(180.0))
    assert compose_orientation(True, 0.0, True, 0.0) == (False, 0.0)


# --- background fill ------------------------------------------------------------------------------


def _exposed_corners(noise: float = 40.0):
    rng = np.random.default_rng(2)
    sky = (SKY + rng.standard_normal((SIZE, SIZE)) * noise).astype(np.float32)
    turned, coverage, _ = rotate_and_square(sky, np.ones(sky.shape, bool), CENTRE, 30.0, True)
    mad = float(np.median(np.abs(sky - np.median(sky))))
    return turned, coverage, mad


def test_fill_never_touches_real_pixels():
    turned, coverage, mad = _exposed_corners()
    filled = fill_background(turned, coverage, mad)
    assert (~coverage).sum() > 1000
    assert np.array_equal(filled[coverage], turned[coverage])


def test_fill_matches_the_neighbouring_sky():
    turned, coverage, mad = _exposed_corners()
    filled = fill_background(turned, coverage, mad)
    distance_out = cv2.distanceTransform((~coverage).astype(np.uint8), cv2.DIST_L2, 5)
    distance_in = cv2.distanceTransform(coverage.astype(np.uint8), cv2.DIST_L2, 5)
    near_fill = (~coverage) & (distance_out < 20)
    near_real = coverage & (distance_in < 20)
    assert abs(float(filled[near_fill].mean()) - float(turned[near_real].mean())) < mad


def test_fill_has_no_streaks():
    """Nearest-pixel extension smears single noisy pixels into lines; the new fill must not."""
    turned, coverage, mad = _exposed_corners()
    plain = fill_uncovered(turned, coverage)
    smoothed = fill_background(turned, coverage, mad)
    region = cv2.erode((~coverage).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0

    def structure(image):
        blurred = cv2.GaussianBlur(image, (0, 0), 4.0)
        gy, gx = np.gradient(blurred)
        return float(np.hypot(gx, gy)[region].mean())

    assert structure(smoothed) < 0.5 * structure(plain)


def test_fill_is_repeatable():
    turned, coverage, mad = _exposed_corners()
    assert np.array_equal(fill_background(turned, coverage, mad), fill_background(turned, coverage, mad))


# --- prominences ------------------------------------------------------------------------------------


def _with_prominence(halo: float = 0.0):
    image = _disk(mark=False, halo=halo)
    rr = _radius(image.shape, CENTRE)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
    px, py = CENTRE[0] + RADIUS + 30.0, CENTRE[1]
    blob = 1000.0 * np.exp(-((xx - px) ** 2 + (yy - py) ** 2) / (2 * 6.0**2))
    return image + np.where(rr > RADIUS + 12, blob, 0.0).astype(np.float32), (int(py), int(px))


@pytest.mark.parametrize("remove_glow", [False, True])
def test_the_disk_itself_is_never_altered(remove_glow):
    image, _ = _with_prominence(halo=5000.0)
    out = prominence_composite(image, CENTRE, RADIUS, 20.0, 1.0, 4.0, remove_glow)
    protected = _radius(image.shape, CENTRE) <= RADIUS + 1.0
    assert np.array_equal(out[protected], image[protected])


@pytest.mark.parametrize("remove_glow", [False, True])
def test_a_prominence_is_brightened_by_the_boost(remove_glow):
    image, (py, px) = _with_prominence()
    out = prominence_composite(image, CENTRE, RADIUS, 10.0, 1.0, 4.0, remove_glow)
    assert out[py, px] / image[py, px] == pytest.approx(10.0, rel=0.02)


def test_glow_removal_leaves_a_flat_sky_around_the_disk():
    image, _ = _with_prominence(halo=5000.0)
    boost = 10.0
    out = prominence_composite(image, CENTRE, RADIUS, boost, 1.0, 4.0, True)
    rr = _radius(image.shape, CENTRE)
    rings = [float(np.median(out[(rr >= r) & (rr < r + 2)])) for r in range(int(RADIUS) + 6, int(RADIUS) + 60, 2)]
    assert max(rings) == pytest.approx(SKY * boost, rel=0.02)
    assert min(rings) == pytest.approx(SKY * boost, rel=0.02)


def test_a_plain_boost_keeps_the_ring_glow_removal_is_meant_to_fix():
    image, _ = _with_prominence(halo=5000.0)
    out = prominence_composite(image, CENTRE, RADIUS, 10.0, 1.0, 4.0, False)
    rr = _radius(image.shape, CENTRE)
    near = float(np.median(out[(rr > RADIUS + 5) & (rr < RADIUS + 7)]))
    far = float(np.median(out[(rr > RADIUS + 60) & (rr < RADIUS + 62)]))
    assert near > 3 * far


def test_the_glow_profile_ignores_a_single_prominence():
    image, _ = _with_prominence(halo=0.0)
    profile = glow_profile(image, np.ones(image.shape, bool), CENTRE, RADIUS)
    assert float(np.abs(profile[5:25] - SKY).max()) < 50.0


def test_boost_one_without_glow_removal_returns_the_image():
    image, _ = _with_prominence()
    out = prominence_composite(image, CENTRE, RADIUS, 1.0, 1.0, 4.0, False)
    assert np.allclose(out, image, atol=0.05)
    assert finish(_result(image), FinishParams(prominence_boost=1.0)).prominences is None


def test_nothing_exceeds_sixteen_bits():
    image, _ = _with_prominence(halo=5000.0)
    out = prominence_composite(image, CENTRE, RADIUS, 40.0, 1.0, 4.0, False)
    assert float(out.max()) <= 65535.0


def test_finish_writes_both_layers():
    image, _ = _with_prominence(halo=2000.0)
    finished = finish(_result(image), FinishParams(rotation_deg=90.0, square=True, prominence_boost=8.0))
    assert finished.image.dtype == np.uint16
    assert finished.prominences is not None and finished.prominences.dtype == np.uint16
    assert finished.prominences.shape == finished.image.shape
    protected = _radius(finished.image.shape, finished.sun_center) <= RADIUS + 1.0
    assert np.array_equal(finished.prominences[protected], finished.image[protected])


def test_a_missing_limb_falls_back_to_the_image_centre():
    finished = finish(_result(_disk(), radius=0.0), FinishParams(rotation_deg=10.0, square=True,
                                                                  prominence_boost=5.0))
    assert finished.prominences is None
    assert any("could not be measured" in w for w in finished.warnings)


def _wobbly_limb(amplitude: float = 2.0, lobes: int = 3):
    """A disk whose edge wanders around the fitted radius, with a steep glow following it.

    Real mosaic limbs wander by a couple of pixels, and the falloff here is as steep as on
    the sample frames (about 20 000 ADU at the edge, a third of that 3 px out).
    """
    rng = np.random.default_rng(4)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
    rr = np.hypot(xx - CENTRE[0], yy - CENTRE[1])
    theta = np.arctan2(yy - CENTRE[1], xx - CENTRE[0])
    edge = RADIUS + amplitude * np.sin(lobes * theta)
    d = rr - edge
    texture = cv2.GaussianBlur(rng.standard_normal((SIZE, SIZE)).astype(np.float32), (0, 0), 3.0)
    image = np.where(d < 0, DISK * (1.0 + 0.3 * texture), SKY + 20000.0 * np.exp(-np.clip(d, 0, None) / 3.0))
    return image.astype(np.float32), rr, theta, edge


def test_limb_offsets_follow_a_wandering_edge():
    from sunmosaic.finish import limb_offsets

    image, _, _, _ = _wobbly_limb()
    offsets = limb_offsets(image, np.ones(image.shape, bool), CENTRE, RADIUS)
    width = 360.0 / offsets.size
    centres = np.radians((np.arange(offsets.size) + 0.5) * width)
    expected = 2.0 * np.sin(3 * centres)
    assert float(np.abs(offsets - expected).max()) < 0.6
    assert float(offsets.max() - offsets.min()) > 3.0


def test_sectors_widen_on_a_small_disk():
    """Each sector must hold enough limb pixels; 2 degrees on a large disk, wider on a small one."""
    from sunmosaic.finish import sector_width

    assert sector_width(797.0) == pytest.approx(2.0)
    assert sector_width(150.0) > 3.5
    assert (360.0 / sector_width(150.0)) == pytest.approx(round(360.0 / sector_width(150.0)))


def test_a_gap_in_coverage_does_not_invent_an_edge():
    """Sectors with nothing usable are filled from their neighbours, never read as an edge."""
    from sunmosaic.finish import limb_offsets

    image, rr, theta, _ = _wobbly_limb(amplitude=0.0)
    coverage = np.ones(image.shape, bool)
    coverage[(np.degrees(theta) % 360 > 40) & (np.degrees(theta) % 360 < 50) & (rr > RADIUS - 3)] = False
    offsets = limb_offsets(image, coverage, CENTRE, RADIUS)
    assert float(np.abs(offsets).max()) < 0.5


def test_limb_offsets_are_flat_on_a_round_disk():
    from sunmosaic.finish import limb_offsets

    image, _ = _with_prominence(halo=5000.0)
    offsets = limb_offsets(image, np.ones(image.shape, bool), CENTRE, RADIUS)
    assert float(np.abs(offsets).max()) < 0.5


def test_a_wandering_limb_leaves_no_bright_speckle_ring():
    """The artefact seen on the real frames: boosted limb pixels where the edge bulges out."""
    image, rr, _, edge = _wobbly_limb()
    band = (rr - edge > 0.5) & (rr < RADIUS + 12)
    fixed = prominence_composite(image, CENTRE, RADIUS, 10.0, 1.0, 4.0, True)
    plain = prominence_composite(image, CENTRE, RADIUS, 10.0, 1.0, 4.0, False)
    speckles = int((fixed[band] > 30000).sum())
    assert speckles < 0.005 * band.sum(), f"{speckles} bright limb pixels"
    assert int((plain[band] > 30000).sum()) > 20 * max(speckles, 1), "the test must be able to see the artefact"


def test_a_strong_boost_leaves_no_dark_ring():
    """Past the point where the boosted sky outshines the limb falloff, no dark band may appear.

    The soft edge used to blend the original falloff with the boosted layer.  Once the boost
    is strong enough, that falloff is dimmer than the boosted sky and the blend showed as a
    dark ring between the disk and the sky.
    """
    image, rr, _, _ = _wobbly_limb(amplitude=0.0)
    boost = 30.0
    band = (rr > RADIUS + 1.5) & (rr < RADIUS + 10.0)
    boosted_sky = SKY * boost
    assert float(image[band].min()) < 0.9 * boosted_sky, "precondition: the falloff is dimmer than the boosted sky"
    out = prominence_composite(image, CENTRE, RADIUS, boost, 1.0, 4.0, True)
    assert float(out[band].min()) >= 0.98 * boosted_sky
    assert np.array_equal(out[rr <= RADIUS + 1.0], image[rr <= RADIUS + 1.0])


def test_a_wandering_limb_is_still_protected():
    """No part of the disk is cut, even where its edge bulges past the fitted circle.

    The fitted circle plus the border is protected by construction.  Where the real edge
    bulges further, the protected region follows the measured edge, and the 1 px border is
    the margin that absorbs the measurement error, so the disk itself stays bit for bit.
    """
    image, rr, _, edge = _wobbly_limb()
    out = prominence_composite(image, CENTRE, RADIUS, 10.0, 1.0, 4.0, True)
    assert np.array_equal(out[rr <= RADIUS + 1.0], image[rr <= RADIUS + 1.0])
    bulge = (rr <= edge) & (rr > RADIUS + 1.0)
    # three lobes of 2 px on a 150 px radius put about 205 px of disk past radius + 1
    assert bulge.sum() > 150, "the test disk must actually bulge past the fitted circle plus border"
    assert np.array_equal(out[rr <= edge], image[rr <= edge]), "the bulging disk must be kept"


# --- live preview -----------------------------------------------------------------------------------


def test_preview_matches_the_full_resolution_result():
    image, _ = _with_prominence(halo=3000.0)
    params = FinishParams(flip=True, rotation_deg=25.0, square=True, prominence_boost=6.0)
    full = finish(_result(image), params)
    source = make_preview_source(image, np.ones(image.shape, bool), CENTRE, RADIUS, max_px=SIZE)
    preview = render_preview(source, params)
    assert preview.shape == full.image.shape
    disk = _radius(preview.shape, full.sun_center) < RADIUS - 5
    assert np.allclose(preview[disk], full.prominences[disk].astype(np.float32), rtol=0.01, atol=2.0)


@pytest.mark.parametrize("remove_glow", [True, False])
def test_preview_fills_no_data_areas_like_the_saved_file(remove_glow):
    """After a turn with a boost, the exposed corners on screen must match the written file."""
    image, _ = _with_prominence(halo=3000.0)
    image = image + np.random.default_rng(8).normal(0.0, 30.0, image.shape).astype(np.float32)
    params = FinishParams(rotation_deg=30.0, square=True, prominence_boost=8.0, remove_glow=remove_glow)
    full = finish(_result(image), params)
    source = make_preview_source(image, np.ones(image.shape, bool), CENTRE, RADIUS, max_px=SIZE)
    preview = render_preview(source, params)
    exposed = ~full.coverage
    assert exposed.sum() > 1000
    on_screen = float(np.median(preview[exposed]))
    written = float(np.median(full.prominences[exposed].astype(np.float32)))
    assert on_screen == pytest.approx(written, rel=0.03)


def test_preview_stretch_is_fixed_from_the_unboosted_image():
    image, _ = _with_prominence(halo=3000.0)
    source = make_preview_source(image, np.ones(image.shape, bool), CENTRE, RADIUS, max_px=200)
    before = source.stretch
    render_preview(source, FinishParams(prominence_boost=30.0))
    assert source.stretch == before


# --- the real frames -----------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_result(sample_paths):
    from sunmosaic.pipeline import stitch_paths

    return stitch_paths(sample_paths)


def test_real_mosaic_turned_into_a_square(real_result):
    finished = finish(real_result, FinishParams(rotation_deg=30.0, square=True))
    side = max(real_result.canvas_size)
    assert finished.side == (side, side)
    fit = _refit(finished.image)
    assert fit.r == pytest.approx(real_result.sun_radius, abs=2.0)
    assert fit.cx == pytest.approx((side - 1) / 2.0, abs=1.0)
    assert fit.cy == pytest.approx((side - 1) / 2.0, abs=1.0)
    assert 0.1 < finished.filled_fraction < 0.3


def test_real_quarter_turn_copies_pixels_exactly(real_result):
    """A quarter turn moves pixels without resampling them.

    Checked against the exact mapping, worked out here rather than taken from the code: a
    counter-clockwise turn about the Sun puts source row ``cy + (x - tx)``, column
    ``cx - (y - ty)`` at output ``(x, y)``.
    """
    finished = finish(real_result, FinishParams(rotation_deg=90.0, square=True))
    source = real_result.mosaic
    height, width = source.shape
    side = finished.image.shape[0]
    cx, cy = (int(round(v)) for v in real_result.sun_center)
    tx, ty = (int(round(v)) for v in finished.sun_center)

    oy, ox = np.mgrid[0:side, 0:side]
    src_row = cy + (ox - tx)
    src_col = cx - (oy - ty)
    inside = (src_row >= 0) & (src_row < height) & (src_col >= 0) & (src_col < width)
    real = np.zeros_like(inside)
    real[inside] = real_result.label_map[src_row[inside], src_col[inside]] >= 0
    assert real.sum() > 0.9 * (real_result.label_map >= 0).sum()
    assert np.array_equal(finished.image[real], source[src_row[real], src_col[real]])
    assert finished.coverage[real].all()


def test_real_quarter_turn_only_loses_far_sky(real_result):
    """The square is as wide as the canvas with the Sun centred, so a few edge rows fall off.

    Those must be sky far from the disk, never anything a prominence could occupy.
    """
    finished = finish(real_result, FinishParams(rotation_deg=90.0, square=True))
    height, width = real_result.mosaic.shape
    side = finished.image.shape[0]
    cx, cy = (int(round(v)) for v in real_result.sun_center)
    tx, ty = (int(round(v)) for v in finished.sun_center)
    yy, xx = np.mgrid[0:height, 0:width]
    dest_x = tx + (yy - cy)
    dest_y = ty - (xx - cx)
    lands = (dest_x >= 0) & (dest_x < side) & (dest_y >= 0) & (dest_y < side)
    dropped = (real_result.label_map >= 0) & ~lands
    assert dropped.mean() < 0.005
    if dropped.any():
        distance = np.hypot(xx[dropped] - real_result.sun_center[0], yy[dropped] - real_result.sun_center[1])
        assert float(distance.min()) > real_result.sun_radius + 200


def test_real_prominence_boost_has_no_ring_and_no_dark_rim(real_result):
    boost = 10.0
    finished = finish(real_result, FinishParams(prominence_boost=boost))
    image = real_result.mosaic.astype(np.float32)
    composite = finished.prominences.astype(np.float32)
    cx, cy = real_result.sun_center
    radius = real_result.sun_radius
    rr = _radius(image.shape, (cx, cy))
    covered = real_result.label_map >= 0

    disk = rr < radius * 0.9
    assert float(np.median(composite[disk])) == pytest.approx(float(np.median(image[disk])), abs=1.0)

    far = covered & (rr > radius + 200)
    boosted_sky = float(np.median(image[far])) * boost
    profile = [float(np.median(composite[covered & (rr >= radius + d - 0.5) & (rr < radius + d + 0.5)]))
               for d in range(31)]
    disc_median = float(np.median(image[disk]))
    assert min(profile) >= 0.98 * boosted_sky, "dark rim at the edge of the protected circle"
    assert max(profile[1:9]) < disc_median, "bright ring just outside the limb"


def test_real_boost_leaves_the_limb_edge_clean(real_result):
    """Limb-placement error shows up right at the edge; real spicules sit further out.

    Measured on the sample frames: every bright pixel left after boosting lies 4 px or more
    beyond the measured edge, and the count does not shrink with finer sectors, which is what
    real chromospheric structure does and an artefact would not.
    """
    from sunmosaic.finish import _offset_map, limb_offsets

    finished = finish(real_result, FinishParams(prominence_boost=10.0))
    image = real_result.mosaic.astype(np.float32)
    covered = real_result.label_map >= 0
    centre, radius = real_result.sun_center, real_result.sun_radius
    offsets = limb_offsets(image, covered, centre, radius)
    rr = _radius(image.shape, centre)
    distance = rr - radius - _offset_map(image.shape, centre, offsets)
    at_edge = covered & (rr > radius + 1.0) & (distance < 2.0)
    bright = finished.prominences.astype(np.float32) > 30000
    assert int((at_edge & bright).sum()) == 0
    assert float(np.abs(offsets).max()) < 3.0


@pytest.mark.parametrize("boost", [10.0, 25.0, 40.0])
def test_real_strong_boost_leaves_no_dark_ring(real_result, boost):
    """On the sample frames the dark ring appeared from a boost of about 13 upward."""
    from sunmosaic.finish import _offset_map, limb_model, sky_statistics

    image = real_result.mosaic.astype(np.float32)
    covered = real_result.label_map >= 0
    centre, radius = real_result.sun_center, real_result.sun_radius
    finished = finish(real_result, FinishParams(prominence_boost=boost))
    composite = finished.prominences.astype(np.float32)
    sky, _ = sky_statistics(image, covered, centre, radius)
    model = limb_model(image, covered, centre, radius)
    rr = _radius(image.shape, centre)
    offset = _offset_map(image.shape, centre, model.offsets)
    beyond_protection = covered & (rr - np.maximum(offset, 0.0) > radius + 1.0) & (rr - radius - offset < 12.0)
    floor = min(sky * boost, 65535.0)
    assert int((composite[beyond_protection] < 0.9 * floor).sum()) == 0
    assert np.array_equal(finished.prominences[rr <= radius + 1.0], real_result.mosaic[rr <= radius + 1.0])


def test_real_glow_removal_flattens_the_sky_all_round(real_result):
    """Scattered light is stronger on some sides of the disk; its removal has to follow it.

    Before the glow was measured per sector, the sky 30 to 50 px beyond the limb ranged from
    1 226 to 1 962 ADU between 30-degree sectors on the sample frames.  A x10 boost turned that
    into broad bright arcs, and left the band at the limb looking like dark dashes wherever the
    sky beside it was bright.
    """
    from sunmosaic.finish import _offset_map, limb_model, sky_statistics

    boost = 10.0
    image = real_result.mosaic.astype(np.float32)
    covered = real_result.label_map >= 0
    centre, radius = real_result.sun_center, real_result.sun_radius
    composite = finish(real_result, FinishParams(prominence_boost=boost)).prominences.astype(np.float32)
    sky, _ = sky_statistics(image, covered, centre, radius)
    model = limb_model(image, covered, centre, radius)
    distance = _radius(image.shape, centre) - radius - _offset_map(image.shape, centre, model.offsets)
    yy, xx = np.mgrid[0 : image.shape[0], 0 : image.shape[1]].astype(np.float32)
    theta = np.degrees(np.arctan2(yy - centre[1], xx - centre[0])) % 360.0
    for start in range(0, 360, 30):
        sector = covered & (theta >= start) & (theta < start + 30)
        for inner, outer in ((30, 50), (80, 120)):
            ring = sector & (distance >= inner) & (distance < outer)
            if ring.sum() < 500:
                continue
            assert float(np.median(composite[ring])) == pytest.approx(sky * boost, rel=0.03), (start, inner)
        at_limb = sector & (distance >= 3) & (distance < 6)
        beside = sector & (distance >= 20) & (distance < 40)
        assert float(np.median(composite[at_limb])) >= 0.98 * float(np.median(composite[beside])), start


def test_real_plain_boost_scales_the_annulus(real_result):
    finished = finish(real_result, FinishParams(prominence_boost=10.0, remove_glow=False))
    image = real_result.mosaic.astype(np.float32)
    composite = finished.prominences.astype(np.float32)
    rr = _radius(image.shape, real_result.sun_center)
    annulus = (real_result.label_map >= 0) & (rr > real_result.sun_radius + 10) & (rr < real_result.sun_radius + 40)
    ratio = float(np.median(composite[annulus])) / float(np.median(image[annulus]))
    assert 8.0 <= ratio <= 12.0


def test_real_boosted_square_corners_match_the_boosted_sky(real_result):
    """Areas with no data must continue the boosted sky they touch, without an outline.

    The reference is the real sky right at the boundary, the tone the fill continues.  Deeper
    into the data the sky may drift: in one corner of the sample frames it brightens by about
    1 % over the first 80 px, which is real sky, not something the fill should reproduce.
    """
    finished = finish(real_result, FinishParams(flip=True, rotation_deg=30.0, square=True,
                                                prominence_boost=10.0))
    boosted = finished.prominences.astype(np.float32)
    coverage = finished.coverage
    far = _radius(boosted.shape, finished.sun_center) > finished.sun_radius + 300
    inside = cv2.distanceTransform(coverage.astype(np.uint8), cv2.DIST_L2, 5)
    outside = cv2.distanceTransform((~coverage).astype(np.uint8), cv2.DIST_L2, 5)
    count, corners = cv2.connectedComponents((~coverage & far).astype(np.uint8))
    checked = 0
    for index in range(1, count):
        corner = corners == index
        if corner.sum() < 20000:
            continue
        beside = (cv2.dilate(corner.astype(np.uint8), np.ones((81, 81), np.uint8)) > 0) & coverage
        boundary_sky = float(np.median(boosted[beside & (inside < 4)]))
        near = corner & (outside < 40)
        deep = corner & (outside >= 80)
        assert float(np.median(boosted[near])) == pytest.approx(boundary_sky, rel=0.02)
        if deep.sum() > 1000:
            assert float(np.median(boosted[deep])) == pytest.approx(boundary_sky, rel=0.02)
        checked += 1
    assert checked >= 2, "a square turned by 30 degrees has empty corners to check"
    # and the linear file is exactly the plain turned image, untouched by any of this
    plain = finish(real_result, FinishParams(flip=True, rotation_deg=30.0, square=True))
    assert np.array_equal(finished.image, plain.image)


def test_real_boosted_fill_has_no_outline_at_display_scale(real_result):
    """What the eye sees is block averages; the step across the boundary must vanish there."""
    finished = finish(real_result, FinishParams(flip=True, rotation_deg=30.0, square=True,
                                                prominence_boost=10.0))
    boosted = finished.prominences.astype(np.float32)
    height, width = boosted.shape
    size = (width // 8, height // 8)
    small = cv2.resize(boosted, size, interpolation=cv2.INTER_AREA)
    fraction = cv2.resize(finished.coverage.astype(np.float32), size, interpolation=cv2.INTER_AREA)
    radius = cv2.resize(_radius(boosted.shape, finished.sun_center), size, interpolation=cv2.INTER_AREA)
    far = radius > finished.sun_radius + 300
    real = (fraction > 0.999) & far
    empty = (fraction < 0.001) & far
    ring = np.ones((3, 3), np.uint8)
    real_edge = real & (cv2.dilate(empty.astype(np.uint8), ring) > 0)
    empty_edge = empty & (cv2.dilate(real.astype(np.uint8), ring) > 0)
    step = abs(float(small[empty_edge].mean()) - float(small[real_edge].mean()))
    assert step < 0.25 * float(small[real].std())


@pytest.mark.parametrize("boost", [10.0, 25.0])
def test_the_saved_layers_rebuild_the_boosted_image_exactly(boost):
    from sunmosaic.finish import blend_layers_uint16

    image, _ = _with_prominence(halo=5000.0)
    finished = finish(_result(image), FinishParams(rotation_deg=30.0, square=True,
                                                   prominence_boost=boost))
    top, bottom, mask = finished.image, finished.prominence_layer, finished.disk_mask
    assert bottom.dtype == mask.dtype == np.uint16 and bottom.shape == mask.shape == top.shape
    assert np.array_equal(blend_layers_uint16(top, bottom, mask), finished.prominences)
    protected = _radius(top.shape, finished.sun_center) <= RADIUS + 1.0
    assert bool((mask[protected] == 65535).all()), "the mask must show the whole disk"
    assert bool((mask[_radius(top.shape, finished.sun_center) > RADIUS + 12] == 0).all())


def test_the_mask_only_opens_further_where_the_limb_is_dimmer_than_the_boosted_sky():
    from sunmosaic.finish import prominence_layers

    image, _ = _with_prominence(halo=5000.0)
    boost = 25.0
    outer, mask = prominence_layers(image, CENTRE, RADIUS, boost, 1.0, 4.0, True, sky_level=SKY)
    _, plain = prominence_layers(image, CENTRE, RADIUS, 1.0, 1.0, 4.0, False, sky_level=SKY)
    assert bool((mask <= plain + 1e-6).all()), "the mask never shows more than the soft edge"
    changed = mask < plain - 1e-6
    assert bool((image[changed] < SKY * boost).all())
    old_style = np.where((plain > 0) & (plain < 1), np.maximum(image, SKY * boost), image)
    expected = plain * old_style + (1.0 - plain) * outer
    got = mask * image + (1.0 - mask) * outer
    assert float(np.abs(got - expected).max()) < 0.5
