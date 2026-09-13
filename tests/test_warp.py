"""Sub-pixel placement, including the overshoot check that picks the default interpolation."""

from __future__ import annotations

import numpy as np
import pytest

from sunmosaic.types import Placement, Tile
from sunmosaic.warp import warp_tile

SKY = 550.0
DISK = 30000.0


def _tile(data: np.ndarray) -> Tile:
    return Tile(name="t.tif", data=data.astype(np.float32),
                native_dtype=np.dtype(np.uint16), native_max=65535.0)


def _limb_frame(size: int = 400, radius: float = 150.0) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    rr = np.hypot(xx - size / 2.0, yy - size / 2.0)
    return np.where(rr < radius, DISK, SKY).astype(np.float32)


def test_integer_mode_copies_pixels_exactly():
    rng = np.random.default_rng(0)
    data = rng.uniform(500, 40000, size=(200, 260)).astype(np.float32)
    canvas, mask = warp_tile(_tile(data), Placement(x=12.4, y=7.6, source="limb"),
                             (300, 260), interp="integer")
    assert np.array_equal(canvas[8 : 8 + 200, 12 : 12 + 260], data)
    assert (mask[8 : 8 + 200, 12 : 12 + 260] == 255).all()


def test_fractional_shift_lands_where_asked():
    rng = np.random.default_rng(1)
    base = rng.uniform(500, 40000, size=(160, 160)).astype(np.float32)
    canvas, mask = warp_tile(_tile(base), Placement(x=20.5, y=10.5, source="limb"), (220, 200))
    interior = canvas[40:140, 50:150]
    assert mask[40:140, 50:150].all()
    assert float(interior.mean()) == pytest.approx(float(base.mean()), rel=0.05)


@pytest.mark.parametrize("interp", ["cubic", "lanczos", "linear", "integer"])
def test_no_dark_ring_around_the_limb(interp):
    """A sharp limb must not undershoot into a dark halo when shifted by half a pixel.

    Bicubic and Lanczos both overshoot at step edges.  This is the check that keeps cubic as
    the default: if it ever rings badly on a limb, the default has to become linear.
    """
    frame = _limb_frame()
    canvas, mask = warp_tile(_tile(frame), Placement(x=10.5, y=10.5, source="limb"), (440, 440))
    valid = mask > 0
    size = frame.shape[0]
    yy, xx = np.mgrid[0:440, 0:440].astype(np.float32)
    rr = np.hypot(xx - (10.5 + size / 2.0), yy - (10.5 + size / 2.0))
    # Measured right at the edge, where ringing lives, not out where it has decayed.
    outside = valid & (rr > 151.5) & (rr < 162.0)
    inside = valid & (rr < 148.5) & (rr > 138.0)
    assert canvas[outside].min() >= SKY - 1.0, "dark ring just outside the limb"
    assert canvas[inside].max() <= DISK + 1.0, "bright overshoot just inside the limb"
