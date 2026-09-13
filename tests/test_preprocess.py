"""Disk masking and the sky-less frame guard."""

from __future__ import annotations

import numpy as np
import pytest

from sunmosaic.limb import fit_limb
from sunmosaic.preprocess import disk_mask, has_sky, highpass, prepare, sky_stats
from sunmosaic.synthetic import cut_tile, synthetic_sun
from sunmosaic.types import Tile


def test_frame_with_sky_is_recognised(quad_case):
    tile = quad_case.tiles[0]
    assert has_sky(tile.data)
    mask = disk_mask(tile.data)
    fraction = float((mask > 0).mean())
    assert 0.3 < fraction < 0.95


def test_frame_without_sky_is_all_disk(center_case):
    tile = center_case.tiles[4]
    assert not has_sky(tile.data)
    assert (disk_mask(tile.data) > 0).all(), "a frame inside the disk must not be carved up"


def test_sky_statistics_track_the_background(quad_case):
    tile = quad_case.tiles[0]
    median, mad = sky_stats(tile.data, disk_mask(tile.data))
    assert 300.0 < median < 2500.0
    assert mad > 0.0


def test_highpass_removes_a_gradient():
    ramp = np.linspace(0, 5000, 512, dtype=np.float32)[None, :].repeat(512, axis=0)
    rng = np.random.default_rng(0)
    detail = rng.standard_normal((512, 512)).astype(np.float32) * 50.0
    hp = highpass(ramp + detail, sigma=25.0)
    # Away from the borders the ramp is gone.  The outermost ~3 sigma keeps a bias, because
    # the blur reflects the image there; the Hanning window suppresses that during matching.
    left = float(hp[:, 120:170].mean())
    right = float(hp[:, 342:392].mean())
    assert abs(left - right) < 0.2
    assert float(hp.std()) == pytest.approx(1.0, abs=1e-3)


def test_frame_that_is_mostly_sky_still_counts_as_having_sky():
    """The median of such a frame is the sky level, so it must not be the yardstick."""
    sun = synthetic_sun(radius=300.0, margin=500.0, seed=1, supersample=2)
    corner = cut_tile(sun, (0, 0), (700, 700), supersample=2).astype(np.float32)
    assert float((corner > 5000).mean()) < 0.2, "this frame should be mostly sky"
    assert has_sky(corner)
    mask = disk_mask(corner)
    assert 0.0 < float((mask > 0).mean()) < 0.5
    tile = prepare(Tile(name="corner.tif", data=corner,
                        native_dtype=np.dtype(np.uint16), native_max=65535.0))
    fit = fit_limb(tile)
    assert fit is not None and fit.r == pytest.approx(300.0, rel=0.05)
