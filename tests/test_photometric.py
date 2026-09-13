"""Brightness matching between frames."""

from __future__ import annotations

import numpy as np

from sunmosaic.limb import fit_limb
from sunmosaic.photometric import apply_correction, solve_gains_offsets
from sunmosaic.preprocess import prepare
from sunmosaic.register import build_placements
from sunmosaic.synthetic import quadrant_case
from sunmosaic.types import Params
from sunmosaic.warp import warp_all, warp_mask


def _solve(case, equalize="gain+offset"):
    tiles = [prepare(t) for t in case.tiles]
    limbs = [fit_limb(t) for t in tiles]
    placements, pairs, canvas, _ = build_placements(tiles, limbs, Params())
    images, masks = warp_all(tiles, placements, canvas, "cubic")
    disks = [warp_mask(t.disk_mask, p, canvas) for t, p in zip(tiles, placements)]
    pedestal = float(np.median([t.sky_median for t in tiles]))
    gains, offsets, warnings = solve_gains_offsets(
        tiles, images, masks, disks, pairs, pedestal, equalize
    )
    return gains, offsets, warnings, images, pedestal


def _normalise(values: np.ndarray) -> np.ndarray:
    return values / np.exp(np.mean(np.log(values)))


def test_gains_follow_the_applied_brightness_differences():
    case = quadrant_case(seed=13, gains=(1.0, 0.90, 1.10, 0.95))
    gains, _, warnings, _, _ = _solve(case)
    recovered = _normalise(gains)
    truth = _normalise(np.array(case.gains))
    # A constant gain cannot absorb the synthetic illumination gradient, so a few percent
    # of disagreement is expected; the multi-band blend removes what is left.
    assert np.allclose(recovered, truth, rtol=0.05), f"{recovered} vs {truth}"
    assert not [w for w in warnings if "capped" in w]


def test_equalisation_can_be_switched_off(quad_case):
    gains, offsets, _, _, _ = _solve(quad_case, equalize="off")
    assert np.allclose(gains, 1.0)
    assert np.allclose(offsets, 0.0)


def test_correction_moves_overlaps_together(quad_case):
    gains, offsets, _, images, pedestal = _solve(quad_case)
    corrected = apply_correction(images, gains, offsets, pedestal)
    assert len(corrected) == len(images)
    assert all(c.dtype == np.float32 for c in corrected)
    spread_before = np.std([float(np.median(i[i > pedestal * 2])) for i in images])
    spread_after = np.std([float(np.median(c[c > pedestal * 2])) for c in corrected])
    assert spread_after < spread_before
