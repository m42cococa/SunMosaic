"""A synthetic Sun and tile cutter, so the pipeline can be tested against known truth.

Tiles are cut from a supersampled disk at integer positions and then downsampled, which
gives exact sub-pixel offsets without ever resampling the test data the way the code under
test would.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .types import Tile

SUPERSAMPLE = 4
SKY_ADU = 550.0
DISK_ADU = 30000.0


@dataclass
class SyntheticCase:
    """Tiles plus the truth they were generated from."""

    tiles: list[Tile]
    origins: list[tuple[float, float]]  # true tile origins, full-resolution pixels
    gains: list[float]
    radius: float
    center: tuple[float, float]

    def relative_origins(self) -> list[tuple[float, float]]:
        ox, oy = self.origins[0]
        return [(x - ox, y - oy) for x, y in self.origins]


def synthetic_sun(
    radius: float = 800.0, margin: float = 180.0, seed: int = 0,
    limb_darkening: float = 0.6, prominences: int = 5, supersample: int = SUPERSAMPLE,
) -> np.ndarray:
    """Render a limb-darkened, textured solar disk with prominences, at ``supersample`` scale."""
    rng = np.random.default_rng(seed)
    size = int(round(2 * (radius + margin))) * supersample
    big_r = radius * supersample
    centre = size / 2.0

    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    rr = np.hypot(xx - centre, yy - centre)

    inside = rr < big_r
    mu = np.zeros_like(rr)
    mu[inside] = np.sqrt(np.clip(1.0 - (rr[inside] / big_r) ** 2, 0.0, 1.0))
    disk = np.zeros_like(rr)
    disk[inside] = DISK_ADU * (1.0 - limb_darkening * (1.0 - mu[inside]))

    # Chromospheric texture: a few octaves of smoothed noise, disk only.
    texture = np.zeros_like(rr)
    for octave, weight in ((8, 0.06), (24, 0.05), (64, 0.03)):
        noise = rng.standard_normal((size // octave + 2, size // octave + 2)).astype(np.float32)
        noise = cv2.resize(noise, (size, size), interpolation=cv2.INTER_CUBIC)
        texture += weight * noise
    disk[inside] *= 1.0 + texture[inside]

    image = np.full((size, size), SKY_ADU, np.float32)
    image[inside] = disk[inside]

    # A soft halo just outside the limb, plus a few prominences.
    halo = np.exp(-np.clip(rr - big_r, 0, None) / (12.0 * supersample)) * 900.0
    image += np.where(~inside, halo, 0.0).astype(np.float32)
    for _ in range(prominences):
        angle = rng.uniform(0, 2 * np.pi)
        height = rng.uniform(8, 26) * supersample
        width = rng.uniform(10, 30) * supersample
        px = centre + (big_r + height * 0.4) * np.cos(angle)
        py = centre + (big_r + height * 0.4) * np.sin(angle)
        blob = np.exp(-(((xx - px) ** 2 + (yy - py) ** 2) / (2 * width**2)))
        image += (blob * rng.uniform(1500, 5000)).astype(np.float32)

    image += rng.standard_normal(image.shape).astype(np.float32) * 40.0
    return np.clip(image, 0, 65535).astype(np.float32)


def cut_tile(
    sun: np.ndarray, origin4: tuple[int, int], size: tuple[int, int],
    gain: float = 1.0, gradient: tuple[float, float] = (0.0, 0.0), offset: float = 0.0,
    seed: int = 0, supersample: int = SUPERSAMPLE, rotation_deg: float = 0.0,
) -> np.ndarray:
    """Cut one tile at an integer supersampled offset and downsample it to full resolution."""
    width, height = size
    x4, y4 = origin4
    patch = sun[y4 : y4 + height * supersample, x4 : x4 + width * supersample]
    if patch.shape != (height * supersample, width * supersample):
        raise ValueError("tile falls outside the rendered Sun; enlarge the margin")
    if abs(rotation_deg) > 1e-9:
        # Field rotation turns the whole frame about the optical axis, at its centre.
        centre = ((patch.shape[1] - 1) / 2.0, (patch.shape[0] - 1) / 2.0)
        matrix = cv2.getRotationMatrix2D(centre, rotation_deg, 1.0)
        patch = cv2.warpAffine(
            patch, matrix, (patch.shape[1], patch.shape[0]),
            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
        )
    tile = cv2.resize(patch, (width, height), interpolation=cv2.INTER_AREA)

    uu, vv = np.meshgrid(
        np.linspace(-1.0, 1.0, width, dtype=np.float32),
        np.linspace(-1.0, 1.0, height, dtype=np.float32),
    )
    field = gain * np.exp(gradient[0] * uu + gradient[1] * vv)
    rng = np.random.default_rng(seed)
    noisy = tile * field + offset + rng.standard_normal(tile.shape).astype(np.float32) * 25.0
    return np.clip(np.rint(noisy), 0, 65535).astype(np.uint16)


def quadrant_case(
    radius: float = 800.0, tile_size: tuple[int, int] = (1600, 1088),
    dx: int = 460, dy: int = 800, seed: int = 0, with_center: bool = False,
    center_size: tuple[int, int] | None = None, rotations_deg: tuple[float, ...] | None = None,
    gains: tuple[float, ...] = (1.0, 0.93, 1.05, 0.98, 1.02),
    jitter: bool = True, vary_sizes: bool = True, supersample: int = SUPERSAMPLE,
) -> SyntheticCase:
    """Four quadrant tiles of one synthetic Sun, laid out like a real 2x2 capture.

    ``dx``/``dy`` are the horizontal and vertical steps between neighbouring tiles; the
    defaults reproduce the wide horizontal and thin vertical overlaps of real data.  A
    fifth, central, limb-less tile can be added to exercise the texture-only fallback.
    """
    width, height = tile_size
    union_w = width + dx
    union_h = height + dy
    margin = max(180.0, (max(union_w, union_h) - 2 * radius) / 2.0 + 30.0)
    sun = synthetic_sun(radius=radius, margin=margin, seed=seed, supersample=supersample)
    centre4 = sun.shape[0] / 2.0

    base_x4 = int(round(centre4 - union_w * supersample / 2.0))
    base_y4 = int(round(centre4 - union_h * supersample / 2.0))
    # Quarter-pixel jitters so the true offsets are not whole pixels.
    shifts = [(0, 1), (2, 3), (1, 2), (3, 0), (2, 1)] if jitter else [(0, 0)] * 5
    corners = [(0, 0), (dx, 0), (dx, dy), (0, dy), (dx // 2, dy // 2)]
    count = 5 if with_center else 4

    tiles: list[Tile] = []
    origins: list[tuple[float, float]] = []
    used_gains: list[float] = []
    gradients = [(0.05, 0.03), (-0.04, 0.05), (0.03, -0.05), (-0.05, -0.02), (0.01, 0.02)]
    size_deltas = [(0, 8), (-8, 0), (0, 0), (-8, 0), (0, 0)] if vary_sizes else [(0, 0)] * 5

    for index in range(count):
        corner_x, corner_y = corners[index]
        shift_x, shift_y = shifts[index]
        origin4 = (
            base_x4 + corner_x * supersample + shift_x,
            base_y4 + corner_y * supersample + shift_y,
        )
        if index == 4 and center_size is not None:
            this_size = center_size
            origin4 = (
                int(round(centre4 - center_size[0] * supersample / 2.0)) + shift_x,
                int(round(centre4 - center_size[1] * supersample / 2.0)) + shift_y,
            )
        else:
            this_size = (width + size_deltas[index][0], height + size_deltas[index][1])
        gain = gains[index % len(gains)]
        data = cut_tile(
            sun, origin4, this_size, gain=gain, gradient=gradients[index % len(gradients)],
            seed=seed * 100 + index, supersample=supersample,
            rotation_deg=0.0 if rotations_deg is None else rotations_deg[index % len(rotations_deg)],
        )
        tiles.append(
            Tile(name=f"synthetic_{index}.tif", data=data.astype(np.float32),
                 native_dtype=np.dtype(np.uint16), native_max=65535.0)
        )
        origins.append((origin4[0] / supersample, origin4[1] / supersample))
        used_gains.append(gain)

    return SyntheticCase(
        tiles=tiles, origins=origins, gains=used_gains, radius=radius,
        center=(centre4 / supersample, centre4 / supersample),
    )
