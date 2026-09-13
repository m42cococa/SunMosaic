"""Per-tile preparation: disk mask, sky statistics, high-pass for registration."""

from __future__ import annotations

import cv2
import numpy as np

from .types import Tile

_HANNING_CACHE: dict[tuple[int, int], np.ndarray] = {}


def to_uint8(img: np.ndarray, hi_percentile: float = 99.5) -> np.ndarray:
    """Normalise to 8 bit using a high percentile, for thresholding only."""
    hi = float(np.percentile(img, hi_percentile))
    lo = float(img.min())
    if hi <= lo:
        hi = lo + 1.0
    scaled = (img - lo) * (255.0 / (hi - lo))
    return np.clip(scaled, 0, 255).astype(np.uint8)


def has_sky(gray: np.ndarray, ratio: float = 0.3) -> bool:
    """True when the frame actually contains sky as well as disk.

    Otsu needs two populations.  A frame lying entirely inside the disk has none, and
    thresholding it would carve the chromospheric texture into a meaningless "disk".

    The darkest pixels are compared against the brightest, not against the median: a frame
    that is mostly sky has a sky-level median, and comparing to that would wrongly call it
    disk everywhere.
    """
    bright = float(np.percentile(gray, 99.0))
    if bright <= 0:
        return True
    return float(np.percentile(gray, 1.0)) < ratio * bright


def disk_mask(gray: np.ndarray) -> np.ndarray:
    """Binary mask (0/255) of the solar disk: Otsu, open 7x7, largest component.

    A frame that holds no sky at all is reported as disk everywhere, which is the truth
    and keeps the limb fit and the brightness matching from chasing texture.
    """
    if not has_sky(gray):
        return np.full(gray.shape[:2], 255, np.uint8)
    u8 = to_uint8(gray)
    _, binary = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return np.zeros_like(binary)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return ((labels == largest).astype(np.uint8)) * 255


def erode_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    k = 2 * radius + 1
    return cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))


def dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    k = 2 * radius + 1
    return cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))


def sky_stats(gray: np.ndarray, mask: np.ndarray, margin: int = 60) -> tuple[float, float]:
    """Median and MAD of the sky, measured well outside the disk.

    The disk mask is dilated so prominences and the limb halo stay out of the sample.
    """
    outside = dilate_mask(mask, margin) == 0
    values = gray[outside]
    if values.size < 1000:
        median = float(np.percentile(gray, 1.0))
        mad = float(np.percentile(np.abs(gray - median), 50)) or 1.0
        return median, max(mad, 1e-3)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return median, max(mad, 1e-3)


def highpass(gray: np.ndarray, sigma: float = 25.0) -> np.ndarray:
    """Remove the illumination gradient and limb darkening, normalise to unit variance.

    Always computed on the full tile; cropping a high-passed image is safe, high-passing
    a crop is not (the Gaussian would see the crop border instead of real neighbours).
    """
    blurred = cv2.GaussianBlur(gray, (0, 0), sigma)
    hp = (gray - blurred).astype(np.float32)
    std = float(hp.std())
    if std > 1e-6:
        hp /= std
    return np.ascontiguousarray(hp)


def hanning(width: int, height: int) -> np.ndarray:
    """Cached 2-D Hanning window for phase correlation."""
    key = (width, height)
    win = _HANNING_CACHE.get(key)
    if win is None:
        win = cv2.createHanningWindow((width, height), cv2.CV_32F)
        _HANNING_CACHE[key] = win
    return win


def prepare(tile: Tile, sigma: float = 25.0) -> Tile:
    """Fill in ``disk_mask``, ``highpass`` and the sky statistics of a tile, in place."""
    gray = tile.gray
    tile.disk_mask = disk_mask(gray)
    tile.sky_median, tile.sky_mad = sky_stats(gray, tile.disk_mask)
    tile.highpass = highpass(gray, sigma)
    return tile
