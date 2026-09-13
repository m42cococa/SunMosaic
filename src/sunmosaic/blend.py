"""Choose seams between the tiles and blend across them.

Each canvas pixel is assigned to whichever tile it sits deepest inside, so seams run down
the middle of the overlaps.  At fine scales every pixel then comes from exactly one frame,
which keeps the chromospheric texture sharp and avoids ghosting when features have moved
between exposures.  A Laplacian pyramid blends only the coarse scales, which is where the
residual illumination mismatch lives.
"""

from __future__ import annotations

import cv2
import numpy as np

FEATHER_PX = 64.0
MIN_LEVELS = 3
MAX_LEVELS = 6


def distance_maps(masks: list[np.ndarray]) -> list[np.ndarray]:
    """Distance of every pixel from the edge of each tile's valid area."""
    return [cv2.distanceTransform(m, cv2.DIST_L2, 5) for m in masks]


def seam_labels(distances: list[np.ndarray]) -> np.ndarray:
    """Index of the owning tile per pixel; -1 where no tile covers the canvas."""
    stack = np.stack(distances, axis=0)
    labels = np.argmax(stack, axis=0).astype(np.int16)
    covered = stack.max(axis=0) > 0
    labels[~covered] = -1
    return labels


def choose_levels(overlap_dims: list[int], override: int | None = None) -> int:
    """Pyramid depth: deep enough to hide gradients, shallow enough to stay in the overlap."""
    if override is not None:
        return int(np.clip(override, 1, MAX_LEVELS))
    smallest = min(overlap_dims) if overlap_dims else 4 * 2**MIN_LEVELS
    levels = int(np.floor(np.log2(max(smallest, 8) / 4.0)))
    return int(np.clip(levels, MIN_LEVELS, MAX_LEVELS))


def _weighted(weight: np.ndarray, image: np.ndarray) -> np.ndarray:
    return weight[..., None] * image if image.ndim == 3 else weight * image


def feather_weights(distances: list[np.ndarray], masks: list[np.ndarray]) -> list[np.ndarray]:
    weights = [np.minimum(d, FEATHER_PX) / FEATHER_PX for d in distances]
    weights = [np.where(m > 0, w, 0.0).astype(np.float32) for w, m in zip(weights, masks)]
    total = np.sum(weights, axis=0)
    safe = np.maximum(total, 1e-6)
    return [(w / safe).astype(np.float32) for w in weights]


def feather_blend(
    images: list[np.ndarray], distances: list[np.ndarray], masks: list[np.ndarray],
) -> np.ndarray:
    weights = feather_weights(distances, masks)
    out = np.zeros_like(images[0])
    for weight, image in zip(weights, images):
        out += _weighted(weight, image)
    return out


def hard_blend(images: list[np.ndarray], labels: np.ndarray) -> np.ndarray:
    out = np.zeros_like(images[0])
    for k, image in enumerate(images):
        sel = labels == k
        out[sel] = image[sel]
    return out


def _laplacian_pyramid(image: np.ndarray, levels: int) -> list[np.ndarray]:
    gaussian = [image]
    for _ in range(levels):
        gaussian.append(cv2.pyrDown(gaussian[-1]))
    pyramid = []
    for i in range(levels):
        size = (gaussian[i].shape[1], gaussian[i].shape[0])
        pyramid.append(gaussian[i] - cv2.pyrUp(gaussian[i + 1], dstsize=size))
    pyramid.append(gaussian[-1])
    return pyramid


def _gaussian_pyramid(image: np.ndarray, levels: int) -> list[np.ndarray]:
    pyramid = [image]
    for _ in range(levels):
        pyramid.append(cv2.pyrDown(pyramid[-1]))
    return pyramid


def _collapse(pyramid: list[np.ndarray]) -> np.ndarray:
    image = pyramid[-1]
    for i in range(len(pyramid) - 2, -1, -1):
        size = (pyramid[i].shape[1], pyramid[i].shape[0])
        image = cv2.pyrUp(image, dstsize=size) + pyramid[i]
    return image


def multiband_blend(
    images: list[np.ndarray], masks: list[np.ndarray], distances: list[np.ndarray],
    labels: np.ndarray, levels: int, fill_value: float = 0.0,
) -> np.ndarray:
    """Laplacian-pyramid blend using the seam labels as the per-tile weights.

    Each tile is extended beyond its own frame with a feather composite of the others, so
    that blurring at the coarse levels never drags black frame borders into the seam.
    """
    height, width = labels.shape
    block = 2**levels
    pad_h = (-height) % block
    pad_w = (-width) % block

    base = feather_blend(images, distances, masks)
    covered = labels >= 0
    base = np.where(covered[..., None] if base.ndim == 3 else covered, base, fill_value)

    def pad(array: np.ndarray) -> np.ndarray:
        if pad_h == 0 and pad_w == 0:
            return np.ascontiguousarray(array)
        return cv2.copyMakeBorder(array, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)

    base = pad(base.astype(np.float32))
    accumulator: list[np.ndarray] | None = None
    weight_sum: list[np.ndarray] | None = None

    for k, image in enumerate(images):
        mask = masks[k] > 0
        filled = np.where(mask[..., None] if image.ndim == 3 else mask, image, base[:height, :width])
        laplacian = _laplacian_pyramid(pad(filled.astype(np.float32)), levels)
        weight = pad((labels == k).astype(np.float32))
        gaussian = _gaussian_pyramid(weight, levels)
        if accumulator is None:
            accumulator = [np.zeros_like(level) for level in laplacian]
            weight_sum = [np.zeros_like(level) for level in gaussian]
        for level in range(levels + 1):
            accumulator[level] += _weighted(gaussian[level], laplacian[level])
            weight_sum[level] += gaussian[level]

    assert accumulator is not None and weight_sum is not None
    blended = [
        acc / (np.maximum(w, 1e-6)[..., None] if acc.ndim == 3 else np.maximum(w, 1e-6))
        for acc, w in zip(accumulator, weight_sum)
    ]
    out = _collapse(blended)[:height, :width]
    return np.where(covered[..., None] if out.ndim == 3 else covered, out, fill_value)


def seam_overlay(preview: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Draw the seam boundaries on an 8-bit preview, one colour per tile."""
    colours = [(255, 80, 80), (80, 255, 80), (80, 160, 255), (255, 220, 80), (255, 80, 255)]
    rgb = cv2.cvtColor(preview, cv2.COLOR_GRAY2RGB) if preview.ndim == 2 else preview.copy()
    small = cv2.resize(labels, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    for k in range(int(small.max()) + 1):
        mask = (small == k).astype(np.uint8) * 255
        edge = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
        rgb[edge > 0] = colours[k % len(colours)]
    return rgb


def fill_uncovered(image: np.ndarray, covered: np.ndarray) -> np.ndarray:
    """Extend the background into canvas corners that no frame reaches.

    A 2x2 set of tiles covers a cross-shaped area, so the corners of the bounding box hold
    no data.  They are filled from the nearest covered pixel, which keeps the sky glow
    continuous instead of leaving a flat patch with a visible edge.  The pixel count is
    reported in the result and in the saved metadata, because this region is synthetic.
    """
    if bool(np.all(covered)):
        return image
    source = (~covered).astype(np.uint8) * 255  # zeros exactly where data exists
    _, labels = cv2.distanceTransformWithLabels(
        source, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL
    )
    ys, xs = np.nonzero(covered)
    lookup = np.zeros(int(labels.max()) + 1, np.int64)
    lookup[labels[covered]] = np.arange(ys.size)
    flat = lookup[labels]
    nearest = image[ys[flat], xs[flat]]
    out = np.where(covered[..., None] if image.ndim == 3 else covered, image, nearest)
    return out.astype(np.float32)
