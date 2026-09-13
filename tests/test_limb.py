"""Limb fitting, including the cases that must be refused."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from sunmosaic.limb import cross_check, fit_limb
from sunmosaic.preprocess import prepare
from sunmosaic.synthetic import synthetic_sun
from sunmosaic.types import LimbFit, Tile


def _tile_from(array: np.ndarray, name: str = "t.tif") -> Tile:
    return prepare(Tile(
        name=name, data=array.astype(np.float32),
        native_dtype=np.dtype(np.uint16), native_max=65535.0,
    ))


@pytest.mark.parametrize("crop_fraction,expect_trusted", [(0.75, True), (0.6, True)])
def test_limb_is_found_on_partial_views(crop_fraction, expect_trusted):
    sun = synthetic_sun(radius=400.0, margin=120.0, seed=2, supersample=2)
    full = cv2.resize(sun, (sun.shape[1] // 2, sun.shape[0] // 2), interpolation=cv2.INTER_AREA)
    height, width = full.shape
    crop = full[: int(height * crop_fraction), : int(width * crop_fraction)]
    fit = fit_limb(_tile_from(crop))
    assert fit is not None and fit.trusted is expect_trusted
    assert fit.r == pytest.approx(400.0, rel=0.02)
    centre = width / 2.0
    assert fit.cx == pytest.approx(centre, abs=2.0)
    assert fit.cy == pytest.approx(centre, abs=2.0)


def test_real_quadrants_agree_on_the_radius(quad_case):
    fits = [fit_limb(prepare(t)) for t in quad_case.tiles]
    assert all(f is not None and f.trusted for f in fits)
    radii = [f.r for f in fits]
    # The frames carry different gains, so Otsu lands at slightly different points on the
    # limb falloff and the fitted radii differ by a couple of pixels.  What matters is that
    # they agree well inside the 1 % tolerance that cross_check uses to demote an outlier.
    assert max(radii) - min(radii) < 5.0
    assert not cross_check(fits)


def test_tile_without_limb_returns_none(center_case):
    fit = fit_limb(prepare(center_case.tiles[4]))
    assert fit is None


def test_flat_field_has_no_limb():
    flat = np.full((600, 600), 12000.0, np.float32)
    assert fit_limb(_tile_from(flat)) is None


def test_cross_check_demotes_an_outlier():
    fits = [
        LimbFit(cx=0, cy=0, r=797.0, n_points=1800, arc_deg=130, resid_px=0.3, trusted=True),
        LimbFit(cx=0, cy=0, r=798.0, n_points=1800, arc_deg=130, resid_px=0.3, trusted=True),
        LimbFit(cx=0, cy=0, r=900.0, n_points=1800, arc_deg=130, resid_px=0.3, trusted=True),
    ]
    warnings = cross_check(fits)
    assert len(warnings) == 1
    assert fits[2].trusted is False
    assert fits[0].trusted is True
