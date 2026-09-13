"""Registration: the sign convention first, because getting it wrong fails silently."""

from __future__ import annotations

import numpy as np
import pytest

from sunmosaic.errors import RegistrationError
from sunmosaic.limb import fit_limb
from sunmosaic.preprocess import prepare
from sunmosaic.register import build_placements, phase_correlate_full, solve_global
from sunmosaic.types import LimbFit, PairMatch, Params, Tile


def _texture(size: int = 256, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((size, size)).astype(np.float32)


def test_phase_correlation_sign_convention():
    """``phaseCorrelate(A, B)`` must return ``t_A - t_B``.

    B holds the content of A moved down 5 rows and right 7 columns, so a feature at
    ``p`` in A sits at ``p + (7, 5)`` in B, which means ``t_A - t_B == (7, 5)``.
    """
    a = _texture()
    b = np.roll(np.roll(a, 5, axis=0), 7, axis=1)
    dx, dy, response = phase_correlate_full(a, b)
    assert response > 0.5
    assert dx == pytest.approx(7.0, abs=0.1)
    assert dy == pytest.approx(5.0, abs=0.1)


def test_pairwise_offsets_are_subpixel_accurate(quad_case):
    tiles = [prepare(t) for t in quad_case.tiles]
    limbs = [fit_limb(t) for t in tiles]
    placements, pairs, _, _ = build_placements(tiles, limbs, Params())

    truth = quad_case.relative_origins()
    origin = (placements[0].x, placements[0].y)
    for index, (px, py) in enumerate(truth):
        got = (placements[index].x - origin[0], placements[index].y - origin[1])
        assert np.hypot(got[0] - px, got[1] - py) < 0.5, f"tile {index} misplaced"
    assert all(p.accepted for p in pairs)


def test_thin_vertical_overlap_is_matched(quad_case):
    """The ~25 % vertical overlap is the case full-frame correlation cannot do."""
    tiles = [prepare(t) for t in quad_case.tiles]
    limbs = [fit_limb(t) for t in tiles]
    _, pairs, _, _ = build_placements(tiles, limbs, Params())
    thin = [p for p in pairs if p.overlap_h < 0.4 * tiles[0].height]
    assert thin, "expected at least one thin vertical overlap"
    for pair in thin:
        assert pair.accepted
        assert pair.response > 0.3


def test_loop_closure_residual_is_small(quad_case):
    tiles = [prepare(t) for t in quad_case.tiles]
    limbs = [fit_limb(t) for t in tiles]
    _, pairs, _, warnings = build_placements(tiles, limbs, Params())
    residuals = [p.residual_px for p in pairs if p.accepted]
    assert max(residuals) < 1.0
    assert not any("rotated" in w for w in warnings)


def test_limbless_tile_is_placed_by_texture(center_case):
    tiles = [prepare(t) for t in center_case.tiles]
    limbs = [fit_limb(t) for t in tiles]
    assert limbs[4] is None, "a tile wholly inside the disk must not report a limb"
    placements, _, _, _ = build_placements(tiles, limbs, Params())
    assert placements[4].source in ("template", "phasecorr")

    truth = center_case.relative_origins()
    origin = (placements[0].x, placements[0].y)
    got = (placements[4].x - origin[0], placements[4].y - origin[1])
    assert np.hypot(got[0] - truth[4][0], got[1] - truth[4][1]) < 1.0


def test_unrelated_frames_are_rejected():
    rng = np.random.default_rng(3)
    tiles = []
    for index in range(2):
        data = rng.uniform(400, 1200, size=(600, 600)).astype(np.float32)
        tiles.append(prepare(Tile(
            name=f"noise_{index}.tif", data=data,
            native_dtype=np.dtype(np.uint16), native_max=65535.0,
        )))
    with pytest.raises(RegistrationError):
        build_placements(tiles, [None, None], Params())


def test_global_solve_averages_noisy_pairs():
    """Six noisy pair measurements of four tiles must average out below the noise."""
    rng = np.random.default_rng(0)
    truth = [(0.0, 0.0), (460.0, 0.0), (460.0, 800.0), (0.0, 800.0)]
    tiles = [
        Tile(name=f"t{i}", data=np.zeros((1088, 1600), np.float32),
             native_dtype=np.dtype(np.uint16), native_max=65535.0)
        for i in range(4)
    ]
    limbs = [LimbFit(0, 0, 797, 1800, 130, 0.3, True) for _ in range(4)]
    pairs = []
    for i in range(4):
        for j in range(i + 1, 4):
            pairs.append(PairMatch(
                i=i, j=j, overlap_w=500, overlap_h=500, method="crop-pc",
                dx=truth[j][0] - truth[i][0] + rng.normal(0, 0.2),
                dy=truth[j][1] - truth[i][1] + rng.normal(0, 0.2),
                response=0.8, accepted=True,
            ))
    solved, _ = solve_global(tiles, truth, limbs, pairs, Params())
    for index, (tx, ty) in enumerate(truth):
        assert np.hypot(solved[index][0] - tx, solved[index][1] - ty) < 0.3


# --- Template matching for contained frames ---------------------------------------------


@pytest.mark.parametrize("centre_size", [(900, 700), (700, 500), (1200, 900)])
def test_contained_frame_is_found_by_template(centre_size):
    """A frame wholly inside the mosaic is located, not merely correlated against."""
    from sunmosaic.synthetic import quadrant_case

    case = quadrant_case(seed=5, with_center=True, center_size=centre_size)
    tiles = [prepare(t) for t in case.tiles]
    limbs = [fit_limb(t) for t in tiles]
    assert limbs[4] is None
    placements, _, _, _ = build_placements(tiles, limbs, Params())
    assert placements[4].source == "template"

    truth = case.relative_origins()
    origin = (placements[0].x, placements[0].y)
    got = (placements[4].x - origin[0], placements[4].y - origin[1])
    assert np.hypot(got[0] - truth[4][0], got[1] - truth[4][1]) < 1.0


def test_template_declines_a_frame_larger_than_the_mosaic(quad_case):
    from sunmosaic.register import place_by_template

    tiles = [prepare(t) for t in quad_case.tiles]
    origins: list[tuple[float, float] | None] = [(0.0, 0.0), None, None, None]
    assert place_by_template(tiles, origins, 1) is None


# --- Rotation ---------------------------------------------------------------------------


def test_rotation_sign_and_magnitude():
    """The measured angle is what must be applied to undo the frame's own rotation."""
    from sunmosaic.register import estimate_pair_rotation
    from sunmosaic.synthetic import quadrant_case

    applied = (0.0, 0.0, 0.0, 0.6)
    case = quadrant_case(seed=9, rotations_deg=applied)
    tiles = [prepare(t) for t in case.tiles]
    limbs = [fit_limb(t) for t in tiles]
    placements, _, _, _ = build_placements(tiles, limbs, Params())
    origins = [(p.x, p.y) for p in placements]
    estimate = estimate_pair_rotation(tiles, origins, 0, 3, Params())
    assert estimate is not None
    angle, score = estimate
    assert np.degrees(angle) == pytest.approx(-0.6, abs=0.1)
    assert score > 0.1


def test_rotations_are_solved_and_undone():
    from sunmosaic.register import derotate, solve_rotations
    from sunmosaic.synthetic import quadrant_case

    applied = (0.0, 0.25, 0.50, 0.75)
    case = quadrant_case(seed=9, rotations_deg=applied)
    tiles = [prepare(t) for t in case.tiles]
    limbs = [fit_limb(t) for t in tiles]
    placements, pairs, _, _ = build_placements(tiles, limbs, Params())
    angles, measured = solve_rotations(tiles, [(p.x, p.y) for p in placements], pairs, Params())
    assert measured == 6

    expected = -(np.array(applied) - np.mean(applied))
    assert np.allclose(np.degrees(angles), expected, atol=0.1)

    straightened = [prepare(t) for t in derotate(tiles, angles)]
    leftover, _ = solve_rotations(
        straightened,
        [(p.x, p.y) for p in build_placements(
            straightened, [fit_limb(t) for t in straightened], Params())[0]],
        pairs, Params(),
    )
    assert float(np.degrees(np.abs(leftover)).max()) < 0.1


def test_no_rotation_is_invented_when_there_is_none(quad_case):
    from sunmosaic.register import needs_rotation, solve_rotations

    tiles = [prepare(t) for t in quad_case.tiles]
    limbs = [fit_limb(t) for t in tiles]
    placements, pairs, _, _ = build_placements(tiles, limbs, Params())
    assert not needs_rotation(tiles, pairs, Params())
    angles, _ = solve_rotations(tiles, [(p.x, p.y) for p in placements], pairs, Params())
    assert float(np.degrees(np.abs(angles)).max()) < 0.05
