"""Dataclasses shared across the pipeline.

Coordinate conventions used everywhere in this package:

* A point is ``(x, y)`` = ``(column, row)``.
* ``Placement.x`` and ``Placement.y`` is the canvas position of a tile's pixel ``(0, 0)``.
* A feature at canvas position ``P`` appears in tile ``i`` at pixel ``P - t_i``.
* ``cv2.phaseCorrelate(A, B)`` returns ``(dx, dy)`` such that the content of ``B``
  equals the content of ``A`` translated by ``+(dx, dy)``; equivalently
  ``t_A - t_B == (dx, dy)``.  This is pinned by ``tests/test_register.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

Interp = Literal["cubic", "lanczos", "linear", "integer"]
BlendMode = Literal["multiband", "feather", "hard"]
Equalize = Literal["off", "gain", "gain+offset", "linear", "quadratic"]


@dataclass
class Tile:
    """One input frame, converted to float32 in native ADU."""

    name: str
    data: np.ndarray  # float32, (H, W) or (H, W, C), native ADU
    native_dtype: np.dtype
    native_max: float
    disk_mask: np.ndarray | None = None  # uint8 0/255
    highpass: np.ndarray | None = None  # float32, (H, W), unit variance
    sky_median: float = 0.0
    sky_mad: float = 1.0
    warnings: list[str] = field(default_factory=list)

    @property
    def height(self) -> int:
        return int(self.data.shape[0])

    @property
    def width(self) -> int:
        return int(self.data.shape[1])

    @property
    def channels(self) -> int:
        return 1 if self.data.ndim == 2 else int(self.data.shape[2])

    @property
    def gray(self) -> np.ndarray:
        """Single-channel view used for registration and limb fitting."""
        if self.data.ndim == 2:
            return self.data
        return self.data.mean(axis=2, dtype=np.float32)


@dataclass
class LimbFit:
    """Circle fitted to the solar limb inside one tile."""

    cx: float
    cy: float
    r: float
    n_points: int
    arc_deg: float
    resid_px: float
    trusted: bool


@dataclass
class PairMatch:
    """Measured displacement between two tiles."""

    i: int
    j: int
    overlap_w: int
    overlap_h: int
    method: str  # "crop-pc" | "full-pc" | "template"
    dx: float  # t_j - t_i, x component
    dy: float  # t_j - t_i, y component
    response: float
    accepted: bool
    reject_reason: str | None = None
    correction: float = 0.0  # |refinement| applied on top of the coarse guess, px
    residual_px: float | None = None  # after the global solve


@dataclass
class Placement:
    """Where a tile sits on the canvas, and how it is scaled photometrically."""

    x: float
    y: float
    source: str  # "limb" | "phasecorr" | "template" | "prior"
    gain: float = 1.0
    offset: float = 0.0
    rotation: float = 0.0  # radians, applied about the frame centre before placement


@dataclass
class Params:
    """Tunable pipeline settings.  Defaults are the values validated on real data."""

    highpass_sigma: float = 25.0
    crop_margin: int = 40
    min_overlap_px: int = 150
    min_response: float = 0.30
    min_arc_deg: float = 60.0
    max_limb_resid: float = 1.5
    max_loop_resid: float = 1.5
    interp: Interp = "cubic"
    blend: BlendMode = "multiband"
    equalize: Equalize = "linear"
    refine_rotation: bool = True
    pyramid_levels: int | None = None
    preview_max_px: int = 1600
    preview_gamma: float = 1.0

    def key(self) -> tuple:
        """Hashable identity, used by the UI cache."""
        return (
            self.highpass_sigma, self.crop_margin, self.min_overlap_px, self.min_response,
            self.min_arc_deg, self.max_limb_resid, self.max_loop_resid, self.interp,
            self.blend, self.equalize, self.pyramid_levels, self.refine_rotation,
        )


@dataclass
class MosaicResult:
    """Everything the UI and the CLI need to report a finished mosaic."""

    mosaic: np.ndarray  # native dtype, (H, W) or (H, W, C)
    native_dtype: np.dtype
    native_max: float
    canvas_size: tuple[int, int]  # (width, height)
    tile_names: list[str]
    placements: list[Placement]
    limbs: list[LimbFit | None]
    pairs: list[PairMatch]
    sun_center: tuple[float, float]
    sun_radius: float
    label_map: np.ndarray  # int16, -1 where uncovered
    pedestal: float
    uncovered_px: int
    field_coefficients: np.ndarray | None = None
    field_degree: int = 0
    warnings: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def worst_residual(self) -> float:
        vals = [p.residual_px for p in self.pairs if p.accepted and p.residual_px is not None]
        return max(vals) if vals else 0.0
