"""Regression against the real H-alpha frames.

The expected values were measured from the four sample frames of 2026-09-08. They are
recorded here so that a change in the algorithm that moves the result has to be noticed
and explained rather than silently accepted.  The whole module is skipped when the sample
frames are not installed.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from sunmosaic.io import read_tile
from sunmosaic.limb import fit_limb
from sunmosaic.pipeline import stitch
from sunmosaic.preprocess import prepare
from sunmosaic.types import Params

EXPECTED_RADIUS = 797.0
# Solar centre inside each frame, in the alphabetical order of the file names
# (2038_0 upper right, 2038_5 lower right, 2039_0 lower left, 2039_5 upper left).
EXPECTED_CENTRES = [(572.8, 944.8), (575.7, 172.4), (1008.7, 144.8), (1035.2, 973.9)]
EXPECTED_CANVAS = (2062, 1916)


@pytest.fixture(scope="module")
def sample_result(sample_paths):
    return stitch([read_tile(p) for p in sample_paths], Params())


def test_every_frame_shows_a_trusted_limb(sample_paths):
    if len(sample_paths) != 4:
        pytest.skip("regression values describe the four-frame sample set")
    for index, path in enumerate(sample_paths):
        fit = fit_limb(prepare(read_tile(path)))
        assert fit is not None and fit.trusted, f"frame {index} lost its limb"
        assert fit.r == pytest.approx(EXPECTED_RADIUS, abs=1.5)
        assert fit.arc_deg >= 125.0
        assert fit.resid_px < 1.0
        assert fit.cx == pytest.approx(EXPECTED_CENTRES[index][0], abs=2.0)
        assert fit.cy == pytest.approx(EXPECTED_CENTRES[index][1], abs=2.0)


def test_all_six_pairs_match(sample_result):
    if len(sample_result.tile_names) != 4:
        pytest.skip("regression values describe the four-frame sample set")
    assert len(sample_result.pairs) == 6
    assert all(p.accepted for p in sample_result.pairs)
    assert all(p.method == "crop-pc" for p in sample_result.pairs)
    assert min(p.response for p in sample_result.pairs) >= 0.6


def test_thin_vertical_overlaps_are_used(sample_result):
    """Roughly a quarter of the frame height, where full-frame correlation fails."""
    thin = [p for p in sample_result.pairs if p.overlap_h < 400]
    assert len(thin) >= 2
    assert all(p.accepted and p.response > 0.6 for p in thin)


def test_placement_agrees_with_the_limb_geometry(sample_result, sample_paths):
    if len(sample_paths) != 4:
        pytest.skip("regression values describe the four-frame sample set")
    origin = (sample_result.placements[0].x, sample_result.placements[0].y)
    base = EXPECTED_CENTRES[0]
    for index, centre in enumerate(EXPECTED_CENTRES):
        expected = (base[0] - centre[0], base[1] - centre[1])
        got = (sample_result.placements[index].x - origin[0],
               sample_result.placements[index].y - origin[1])
        assert np.hypot(got[0] - expected[0], got[1] - expected[1]) < 2.0


def test_frames_are_consistent_with_a_pure_shift(sample_result):
    assert sample_result.worst_residual < 1.5
    assert not any("rotated" in w for w in sample_result.warnings)


def test_canvas_and_disk_match_the_reference(sample_result):
    if len(sample_result.tile_names) != 4:
        pytest.skip("regression values describe the four-frame sample set")
    width, height = sample_result.canvas_size
    assert abs(width - EXPECTED_CANVAS[0]) <= 4
    assert abs(height - EXPECTED_CANVAS[1]) <= 4
    assert sample_result.sun_radius == pytest.approx(EXPECTED_RADIUS, abs=2.0)
    assert sample_result.mosaic.dtype == np.uint16
    assert sample_result.mosaic.shape == (height, width)


def test_the_mosaic_shows_more_than_any_single_frame(sample_result, sample_paths):
    """The point of the tool: the whole disk does not fit in one frame."""
    widest = max(read_tile(p).width for p in sample_paths)
    assert 2 * sample_result.sun_radius > widest * 0.95
    assert sample_result.canvas_size[0] > widest


def test_brightness_corrections_stay_modest(sample_result):
    gains = [p.gain for p in sample_result.placements]
    offsets = [p.offset for p in sample_result.placements]
    assert all(0.90 <= g <= 1.10 for g in gains), gains
    assert all(abs(o) < 150.0 for o in offsets), offsets


def test_the_disk_never_falls_on_filled_canvas(sample_result):
    cx, cy = sample_result.sun_center
    height, width = sample_result.mosaic.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    disk = np.hypot(xx - cx, yy - cy) < sample_result.sun_radius * 1.02
    assert (sample_result.label_map[disk] >= 0).all()


def test_a_run_is_quick(sample_paths):
    start = time.perf_counter()
    stitch([read_tile(p) for p in sample_paths], Params())
    assert time.perf_counter() - start < 30.0
