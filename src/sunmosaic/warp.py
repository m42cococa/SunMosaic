"""Place tiles onto the mosaic canvas, with sub-pixel translation."""

from __future__ import annotations

import cv2
import numpy as np

from .types import Interp, Placement, Tile

_INTERP_FLAGS = {
    "cubic": cv2.INTER_CUBIC,
    "lanczos": cv2.INTER_LANCZOS4,
    "linear": cv2.INTER_LINEAR,
}
VALID_ERODE_PX = 2


def warp_tile(
    tile: Tile, placement: Placement, canvas_size: tuple[int, int], interp: Interp = "cubic",
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(canvas_image, valid_mask)`` for one tile.

    ``interp="integer"`` rounds the placement and copies pixels with no resampling at all,
    for anyone who would rather have a half-pixel misalignment than a resampled image.
    """
    canvas_w, canvas_h = canvas_size
    channels = tile.channels
    out_shape = (canvas_h, canvas_w) if channels == 1 else (canvas_h, canvas_w, channels)

    if interp == "integer":
        canvas = np.zeros(out_shape, np.float32)
        mask = np.zeros((canvas_h, canvas_w), np.uint8)
        ix, iy = int(round(placement.x)), int(round(placement.y))
        x0, y0 = max(ix, 0), max(iy, 0)
        x1 = min(ix + tile.width, canvas_w)
        y1 = min(iy + tile.height, canvas_h)
        if x1 > x0 and y1 > y0:
            canvas[y0:y1, x0:x1] = tile.data[y0 - iy : y1 - iy, x0 - ix : x1 - ix]
            mask[y0:y1, x0:x1] = 255
        return canvas, mask

    flags = _INTERP_FLAGS.get(interp, cv2.INTER_CUBIC)
    matrix = np.array([[1.0, 0.0, placement.x], [0.0, 1.0, placement.y]], np.float32)
    canvas = cv2.warpAffine(
        tile.data, matrix, (canvas_w, canvas_h), flags=flags,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
    )
    # Cubic and Lanczos overshoot at a step edge.  On a soft solar limb that is harmless,
    # but a very sharp limb would ring below the sky and show as a dark ring.  Holding each
    # frame inside its own range removes that without touching anything else.
    if interp in ("cubic", "lanczos"):
        np.clip(canvas, float(tile.data.min()), float(tile.data.max()), out=canvas)
    ones = np.ones((tile.height, tile.width), np.float32)
    warped_mask = cv2.warpAffine(
        ones, matrix, (canvas_w, canvas_h), flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
    )
    mask = (warped_mask > 0.5).astype(np.uint8) * 255
    if VALID_ERODE_PX > 0:
        k = 2 * VALID_ERODE_PX + 1
        mask = cv2.erode(mask, np.ones((k, k), np.uint8))
    canvas = np.where(
        mask[..., None] > 0 if canvas.ndim == 3 else mask > 0, canvas, 0.0
    ).astype(np.float32)
    return canvas, mask


def warp_all(
    tiles: list[Tile], placements: list[Placement], canvas_size: tuple[int, int],
    interp: Interp = "cubic",
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    images, masks = [], []
    for tile, placement in zip(tiles, placements):
        image, mask = warp_tile(tile, placement, canvas_size, interp)
        images.append(image)
        masks.append(mask)
    return images, masks


def warp_mask(
    mask: np.ndarray, placement: Placement, canvas_size: tuple[int, int],
) -> np.ndarray:
    """Move a per-tile binary mask (such as the disk mask) onto the canvas."""
    canvas_w, canvas_h = canvas_size
    matrix = np.array([[1.0, 0.0, placement.x], [0.0, 1.0, placement.y]], np.float32)
    warped = cv2.warpAffine(
        mask, matrix, (canvas_w, canvas_h), flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    return (warped > 127).astype(np.uint8) * 255
