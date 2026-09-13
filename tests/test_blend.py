"""Seam placement and blending behaviour."""

from __future__ import annotations

import numpy as np
import pytest

from sunmosaic.blend import (
    choose_levels,
    distance_maps,
    feather_blend,
    fill_uncovered,
    hard_blend,
    multiband_blend,
    seam_labels,
)


def _two_tiles(height=256, width=384, split=128):
    left = np.zeros((height, width), np.uint8)
    left[:, : split + 128] = 255
    right = np.zeros((height, width), np.uint8)
    right[:, split:] = 255
    return left, right


def test_seam_runs_down_the_middle_of_the_overlap():
    left, right = _two_tiles()
    labels = seam_labels(distance_maps([left, right]))
    row = labels[128]
    boundary = int(np.argmax(row == 1))
    assert 180 < boundary < 210, f"seam at {boundary}, expected near the overlap centre"


def test_uncovered_pixels_are_labelled_minus_one():
    left, right = _two_tiles()
    left[:, 300:] = 0
    right[:, 300:] = 0
    labels = seam_labels(distance_maps([left, right]))
    assert (labels[:, 320:] == -1).all()


def test_multiband_does_not_overshoot():
    left, right = _two_tiles()
    images = [np.where(left > 0, 0.4, 0.0).astype(np.float32),
              np.where(right > 0, 0.6, 0.0).astype(np.float32)]
    distances = distance_maps([left, right])
    labels = seam_labels(distances)
    out = multiband_blend(images, [left, right], distances, labels, levels=3)
    covered = labels >= 0
    assert out[covered].min() >= 0.4 - 1e-4
    assert out[covered].max() <= 0.6 + 1e-4
    assert np.all(np.diff(out[128, 130:260]) >= -1e-4), "the transition should not reverse"


def test_multiband_is_idempotent_on_identical_frames():
    left, right = _two_tiles()
    rng = np.random.default_rng(1)
    scene = (rng.random((256, 384)).astype(np.float32) * 0.5 + 0.25)
    images = [np.where(left > 0, scene, 0.0), np.where(right > 0, scene, 0.0)]
    distances = distance_maps([left, right])
    labels = seam_labels(distances)
    out = multiband_blend(images, [left, right], distances, labels, levels=3)
    covered = labels >= 0
    assert float(np.abs(out[covered] - scene[covered]).max()) < 1e-4


def test_hard_and_feather_stay_within_the_inputs():
    left, right = _two_tiles()
    images = [np.where(left > 0, 0.4, 0.0).astype(np.float32),
              np.where(right > 0, 0.6, 0.0).astype(np.float32)]
    distances = distance_maps([left, right])
    labels = seam_labels(distances)
    for out in (hard_blend(images, labels), feather_blend(images, distances, [left, right])):
        covered = labels >= 0
        assert out[covered].min() >= 0.4 - 1e-4
        assert out[covered].max() <= 0.6 + 1e-4


@pytest.mark.parametrize("overlap,expected", [(258, 6), (120, 4), (40, 3), (2000, 6)])
def test_blend_depth_matches_the_overlap(overlap, expected):
    assert choose_levels([overlap]) == expected


def test_uncovered_corners_take_the_nearest_background():
    image = np.zeros((64, 64), np.float32)
    image[:, :32] = 700.0
    covered = np.zeros((64, 64), bool)
    covered[:, :32] = True
    filled = fill_uncovered(image, covered)
    assert np.allclose(filled, 700.0)
