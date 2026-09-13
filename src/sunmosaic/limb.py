"""Fit a circle to the solar limb visible inside a single tile.

The limb is the one feature every quadrant tile shares with the true disk geometry, so a
circle fit gives a coarse placement without needing any tile-to-tile texture match.
"""

from __future__ import annotations

import cv2
import numpy as np

from .types import LimbFit, Tile

BORDER_REJECT_PX = 3
MIN_POINTS = 200


def _fit_circle_algebraic(pts: np.ndarray) -> tuple[float, float, float]:
    """Kasa least-squares circle fit.  ``pts`` is (N, 2) of (x, y)."""
    x = pts[:, 0].astype(np.float64)
    y = pts[:, 1].astype(np.float64)
    a = np.c_[2.0 * x, 2.0 * y, np.ones_like(x)]
    b = x * x + y * y
    sol, *_ = np.linalg.lstsq(a, b, rcond=None)
    cx, cy = float(sol[0]), float(sol[1])
    r2 = float(sol[2]) + cx * cx + cy * cy
    return cx, cy, float(np.sqrt(max(r2, 1e-9)))


def _arc_coverage_deg(pts: np.ndarray, cx: float, cy: float, bin_deg: float = 5.0) -> float:
    """Angular extent actually covered by limb points, in degrees."""
    angles = np.degrees(np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)) % 360.0
    n_bins = int(round(360.0 / bin_deg))
    counts = np.bincount((angles / bin_deg).astype(int) % n_bins, minlength=n_bins)
    return float(np.count_nonzero(counts >= 3) * bin_deg)


def limb_points(mask: np.ndarray) -> np.ndarray:
    """Contour points of the disk that are not lying along the frame border."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.empty((0, 2), np.float64)
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    h, w = mask.shape[:2]
    b = BORDER_REJECT_PX
    keep = (
        (contour[:, 0] > b) & (contour[:, 0] < w - 1 - b)
        & (contour[:, 1] > b) & (contour[:, 1] < h - 1 - b)
    )
    return contour[keep]


def fit_limb(tile: Tile, min_arc_deg: float = 60.0, max_resid: float = 1.5) -> LimbFit | None:
    """Fit the limb of one tile.

    Returns ``None`` when the tile shows too little limb to be usable at all (a tile fully
    inside the disk, for instance), which routes it to the texture-based fallback.
    """
    if tile.disk_mask is None:
        raise ValueError("tile must be prepared before fitting the limb")
    pts = limb_points(tile.disk_mask)
    if pts.shape[0] < 50:
        return None

    cx, cy, r = _fit_circle_algebraic(pts)
    for _ in range(2):
        dist = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
        resid = np.abs(dist - r)
        mad = float(np.median(resid))
        keep = resid <= max(3.0 * mad, 1.0)
        if keep.sum() < 50:
            break
        pts = pts[keep]
        cx, cy, r = _fit_circle_algebraic(pts)

    dist = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
    resid_px = float(np.sqrt(np.mean((dist - r) ** 2)))
    arc_deg = _arc_coverage_deg(pts, cx, cy)
    if arc_deg < 30.0:
        return None

    h, w = tile.disk_mask.shape[:2]
    # A circle that fits badly, or one far too small for the frame, is not a limb: it is
    # texture.  Reject it outright so the tile is routed to the texture-based fallback
    # rather than being placed from a meaningless centre.
    if resid_px > max(3.0, 0.05 * r) or r < 0.15 * min(h, w):
        return None
    # Circularity is judged relative to the circle: a 3 px scatter around an 800 px radius
    # is a good limb, the same scatter around a 50 px blob is not a limb at all.
    trusted = (
        arc_deg >= min_arc_deg
        and resid_px <= max(max_resid, 0.005 * r)
        and pts.shape[0] >= MIN_POINTS
        and r > 0.25 * min(h, w)
    )
    return LimbFit(cx=cx, cy=cy, r=r, n_points=int(pts.shape[0]),
                   arc_deg=arc_deg, resid_px=resid_px, trusted=bool(trusted))


def cross_check(limbs: list[LimbFit | None], tol_frac: float = 0.01) -> list[str]:
    """Demote limb fits whose radius disagrees with the others; returns warnings."""
    radii = [lf.r for lf in limbs if lf is not None and lf.trusted]
    warnings: list[str] = []
    if len(radii) < 2:
        return warnings
    median_r = float(np.median(radii))
    for idx, lf in enumerate(limbs):
        if lf is None or not lf.trusted:
            continue
        if abs(lf.r - median_r) > tol_frac * median_r:
            lf.trusted = False
            warnings.append(
                f"tile {idx}: fitted solar radius {lf.r:.1f} px disagrees with the other tiles "
                f"({median_r:.1f} px); its limb fit was demoted to a guess"
            )
    return warnings
