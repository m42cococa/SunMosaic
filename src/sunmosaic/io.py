"""Reading tiles, writing the mosaic, and making 8-bit previews."""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, BinaryIO

import cv2
import numpy as np
import tifffile

from . import __version__
from .errors import InputError
from .types import MosaicResult, Tile

MAX_TILES = 5
MIN_TILES = 2
MIN_DIMENSION = 256
MAX_MEGAPIXELS = 30.0


def _squeeze_pages(array: np.ndarray, warnings: list[str]) -> np.ndarray:
    array = np.squeeze(array)
    while array.ndim > 3:
        warnings.append("file holds several images; only the first one was used")
        array = array[0]
    if array.ndim == 3 and array.shape[2] not in (3, 4):
        warnings.append("file holds several images; only the first one was used")
        array = array[0]
    if array.ndim == 3 and array.shape[2] == 4:
        array = array[:, :, :3]
    return array


def read_tile(source: str | Path | bytes | BinaryIO, name: str | None = None) -> Tile:
    """Load one TIFF into a :class:`Tile`, normalised to float32 in 16-bit ADU."""
    warnings: list[str] = []
    if isinstance(source, (str, Path)):
        path = Path(source)
        display_name = name or path.name
        try:
            raw = tifffile.imread(str(path))
        except Exception as exc:
            raise InputError(f"Could not read '{display_name}': {exc}") from exc
    else:
        display_name = name or "uploaded.tif"
        data = source if isinstance(source, bytes) else source.read()
        try:
            raw = tifffile.imread(BytesIO(data))
        except Exception as exc:
            raise InputError(f"Could not read '{display_name}': {exc}") from exc

    array = _squeeze_pages(np.asarray(raw), warnings)
    if array.ndim not in (2, 3):
        raise InputError(f"'{display_name}' is not a 2-D image (shape {array.shape})")

    original_dtype = array.dtype
    data = array.astype(np.float32)
    if original_dtype == np.uint8:
        data *= 257.0
        warnings.append("8-bit file was promoted to 16 bit")
    elif original_dtype in (np.float32, np.float64):
        peak = float(np.nanmax(data)) if data.size else 0.0
        if peak <= 1.5:
            data *= 65535.0
        warnings.append("floating point file was rescaled to the 16-bit range")
    elif original_dtype == np.int16:
        data = data - float(np.iinfo(np.int16).min)
        warnings.append("signed 16-bit file was shifted to an unsigned range")
    elif original_dtype != np.uint16:
        warnings.append(f"unusual pixel type {original_dtype}; it was read as numbers")

    if array.ndim == 3:
        warnings.append("colour file: registration uses the average of the channels")

    return Tile(
        name=display_name, data=np.ascontiguousarray(data), native_dtype=np.dtype(np.uint16),
        native_max=65535.0, warnings=warnings,
    )


def validate_tiles(tiles: list[Tile]) -> None:
    """Reject file selections that cannot produce a mosaic, with a reason the user can act on."""
    if len(tiles) < MIN_TILES:
        raise InputError(
            f"Select at least {MIN_TILES} files: a mosaic needs overlapping views to join."
        )
    if len(tiles) > MAX_TILES:
        raise InputError(f"Select at most {MAX_TILES} files (you chose {len(tiles)}).")
    for tile in tiles:
        if min(tile.height, tile.width) < MIN_DIMENSION:
            raise InputError(
                f"'{tile.name}' is only {tile.width}x{tile.height} pixels, too small to register."
            )
        if tile.height * tile.width > MAX_MEGAPIXELS * 1e6:
            raise InputError(
                f"'{tile.name}' is {tile.height * tile.width / 1e6:.0f} megapixels, "
                f"beyond the {MAX_MEGAPIXELS:.0f} megapixel limit of this tool."
            )
    channels = {tile.channels for tile in tiles}
    if len(channels) > 1:
        raise InputError("Mixing greyscale and colour files is not supported.")


def build_metadata(result: MosaicResult) -> dict[str, Any]:
    return {
        "software": f"SunMosaic {__version__}",
        "created": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "canvas": {"width": result.canvas_size[0], "height": result.canvas_size[1]},
        "sun": {
            "center_x": round(result.sun_center[0], 2),
            "center_y": round(result.sun_center[1], 2),
            "radius_px": round(result.sun_radius, 2),
        },
        "pedestal_adu": round(result.pedestal, 2),
        "tiles": [
            {
                "name": name,
                "origin_x": round(p.x, 3),
                "origin_y": round(p.y, 3),
                "gain": round(p.gain, 5),
                "sky_offset_adu": round(p.offset, 2),
                "rotation_deg": round(float(np.degrees(p.rotation)), 4),
                "placed_by": p.source,
            }
            for name, p in zip(result.tile_names, result.placements)
        ],
        "pairs": [
            {
                "tiles": [p.i, p.j], "method": p.method, "response": round(p.response, 3),
                "residual_px": None if p.residual_px is None else round(p.residual_px, 3),
                "accepted": p.accepted,
            }
            for p in result.pairs
        ],
        "warnings": result.warnings,
    }


def write_image(data: np.ndarray, metadata: dict[str, Any], dest: str | Path | BytesIO) -> None:
    """Write any 16-bit image with a metadata block in its ImageDescription."""
    if data.dtype != np.uint16:
        data = np.clip(np.rint(data), 0, 65535).astype(np.uint16)
    target = str(dest) if isinstance(dest, (str, Path)) else dest
    tifffile.imwrite(
        target, data, photometric="rgb" if data.ndim == 3 else "minisblack",
        description=json.dumps(metadata), software=f"SunMosaic {__version__}",
    )


@dataclass
class ImageLayer:
    """One layer of a layered TIFF: 16-bit pixels and an optional 16-bit layer mask."""

    name: str
    data: np.ndarray  # uint16, HxW or HxWx3
    mask: np.ndarray | None = None  # uint16, 65535 shows the layer, 0 hides it


def _layer_source_data(layers: list[ImageLayer]):
    """The layers in Adobe's layered TIFF form (the ImageSourceData tag, #37724).

    This is the one convention for layers inside a TIFF.  Affinity Photo, Photoshop and Krita
    read it; every other reader ignores the tag and shows the flat image stored as usual.
    """
    from psdtags import (
        PsdBlendMode,
        PsdChannel,
        PsdChannelId,
        PsdClippingType,
        PsdColorSpaceType,
        PsdCompressionType,
        PsdEmpty,
        PsdFilterMask,
        PsdFormat,
        PsdKey,
        PsdLayer,
        PsdLayerFlag,
        PsdLayerMask,
        PsdLayers,
        PsdRectangle,
        PsdString,
        PsdUserMask,
        TiffImageSourceData,
    )

    records = []
    for layer in layers:
        data = np.ascontiguousarray(layer.data, dtype=np.uint16)
        height, width = data.shape[:2]
        planes = [data] if data.ndim == 2 else [data[..., c] for c in range(data.shape[2])]
        channels = [PsdChannel(
            channelid=PsdChannelId.TRANSPARENCY_MASK,
            compression=PsdCompressionType.ZIP_PREDICTED,
            data=np.full((height, width), 65535, np.uint16),
        )]
        channels += [
            PsdChannel(channelid=PsdChannelId(index), compression=PsdCompressionType.ZIP_PREDICTED,
                       data=np.ascontiguousarray(plane))
            for index, plane in enumerate(planes)
        ]
        mask = PsdLayerMask()
        if layer.mask is not None:
            channels.append(PsdChannel(
                channelid=PsdChannelId.USER_LAYER_MASK,
                compression=PsdCompressionType.ZIP_PREDICTED,
                data=np.ascontiguousarray(layer.mask, dtype=np.uint16),
            ))
            mask = PsdLayerMask(default_color=0, rectangle=PsdRectangle(0, 0, height, width))
        records.append(PsdLayer(
            name=layer.name[:255],
            rectangle=PsdRectangle(0, 0, height, width),
            channels=channels,
            mask=mask,
            opacity=255,
            blendmode=PsdBlendMode.NORMAL,
            blending_ranges=(),
            clipping=PsdClippingType.BASE,
            flags=PsdLayerFlag.PHOTOSHOP5,  # the "VISIBLE" bit set would hide the layer
            info=[PsdString(PsdKey.UNICODE_LAYER_NAME, layer.name)],
        ))
    return TiffImageSourceData(
        name="SunMosaic layers",
        psdformat=PsdFormat.LE32BIT,
        layers=PsdLayers(key=PsdKey.LAYER_16, has_transparency=False, layers=records),
        usermask=PsdUserMask(colorspace=PsdColorSpaceType.RGB, components=(65535, 0, 0, 0),
                             opacity=50),
        info=[
            PsdEmpty(PsdKey.PATTERNS),
            PsdFilterMask(colorspace=PsdColorSpaceType.RGB, components=(65535, 0, 0, 0),
                          opacity=50),
        ],
    )


def write_layered_image(
    flat: np.ndarray, layers: list[ImageLayer], metadata: dict[str, Any],
    dest: str | Path | BytesIO,
) -> None:
    """A 16-bit TIFF holding the flat image as usual plus its layers for image editors.

    Layers are listed bottom first.  ``flat`` must be what the layers look like together, since
    readers without layer support show only that.
    """
    if flat.dtype != np.uint16:
        flat = np.clip(np.rint(flat), 0, 65535).astype(np.uint16)
    source_data = _layer_source_data(layers)
    target = str(dest) if isinstance(dest, (str, Path)) else dest
    tifffile.imwrite(
        target, flat, byteorder=source_data.byteorder,
        photometric="rgb" if flat.ndim == 3 else "minisblack",
        description=json.dumps(metadata), software=f"SunMosaic {__version__}", metadata=None,
        extratags=[source_data.tifftag(maxworkers=4)],
    )


def prominence_image_layers(finished) -> list[ImageLayer]:
    """Bottom: the boosted copy.  Top: the untouched image, masked to show the disk."""
    params = finished.params
    glow = ", limb glow removed" if params.remove_glow else ""
    return [
        ImageLayer(f"Prominences, boost x{params.prominence_boost:g}{glow}",
                   finished.prominence_layer),
        ImageLayer("Sun disk, linear mosaic", finished.image, finished.disk_mask),
    ]


def write_prominence_file(result: MosaicResult, finished, dest: str | Path | BytesIO) -> None:
    """The boosted result as a layered TIFF: flat image in front, both layers inside."""
    write_layered_image(finished.prominences, prominence_image_layers(finished),
                        finished_metadata(result, finished, "prominences"), dest)


def prominence_file_bytes(result: MosaicResult, finished) -> bytes:
    buffer = BytesIO()
    write_prominence_file(result, finished, buffer)
    return buffer.getvalue()


def image_bytes(data: np.ndarray, metadata: dict[str, Any]) -> bytes:
    buffer = BytesIO()
    write_image(data, metadata, buffer)
    return buffer.getvalue()


def write_tiff(result: MosaicResult, dest: str | Path | BytesIO) -> None:
    """Write the mosaic as a 16-bit TIFF, linear, with the run's parameters embedded."""
    write_image(result.mosaic, build_metadata(result), dest)


def tiff_bytes(result: MosaicResult) -> bytes:
    buffer = BytesIO()
    write_tiff(result, buffer)
    return buffer.getvalue()


def finished_metadata(result: MosaicResult, finished, layer: str) -> dict[str, Any]:
    """Metadata for an image that went through the finishing step.

    ``layer`` is ``"linear"`` for the untouched data or ``"prominences"`` for the boosted
    composite, which is non-linear and says so, so it is never mistaken for measurements.
    """
    meta = build_metadata(result)
    params = finished.params
    width, height = finished.side
    meta["canvas"] = {"width": width, "height": height}
    meta["sun"] = {
        "center_x": round(finished.sun_center[0], 2),
        "center_y": round(finished.sun_center[1], 2),
        "radius_px": round(finished.sun_radius, 2),
    }
    meta["finish"] = {
        "layer": layer,
        "linear": layer == "linear",
        "flip_left_right": bool(params.flip),
        "rotation_deg_counterclockwise": round(float(params.rotation_deg), 4),
        "square": bool(params.square),
        "filled_fraction": round(float(finished.filled_fraction), 4),
        "fill_method": (
            "copies of real boosted sky picked at random just inside the nearest boundary"
            if layer == "prominences"
            else "nearest real pixels, smoothed, noise matched to the sky"
            if params.changes_geometry
            else "nearest real pixels"
        ),
        "prominences": None if not params.boosts else {
            "boost": round(float(params.prominence_boost), 3),
            "protected_radius_px": round(finished.sun_radius + params.prominence_border_px, 2),
            "border_px": float(params.prominence_border_px),
            "feather_px": float(params.prominence_feather_px),
            "limb_glow_removed": bool(params.remove_glow),
        },
    }
    if layer == "prominences":
        meta["finish"]["layers"] = [
            {"name": item.name, "position": position,
             "content": "boosted copy" if item.mask is None else "linear mosaic, untouched",
             "mask": None if item.mask is None else
             "shows the disk plus the border; soft edge adjusted so the joint never darkens"}
            for position, item in zip(("bottom", "top"), prominence_image_layers(finished))
        ]
    meta["warnings"] = list(result.warnings) + list(finished.warnings)
    return meta


def stretch_limits(
    image: np.ndarray, low: float = 0.1, high: float = 99.9, mask: np.ndarray | None = None,
) -> tuple[float, float]:
    """Display black and white points from percentiles of the (masked) image."""
    gray = image if image.ndim == 2 else image.mean(axis=2)
    sample = gray[mask > 0] if mask is not None and np.any(mask > 0) else gray
    lo = float(np.percentile(sample, low))
    hi = float(np.percentile(sample, high))
    return lo, (hi if hi > lo else lo + 1.0)


def stretch_with_limits(
    image: np.ndarray, lo: float, hi: float, gamma: float = 1.0, max_px: int | None = None,
) -> np.ndarray:
    """Display stretch with fixed limits, so boosting one region cannot dim another."""
    scaled = np.clip((image - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    if abs(gamma - 1.0) > 1e-3:
        scaled = np.power(scaled, 1.0 / max(gamma, 1e-3))
    out = (scaled * 255.0).astype(np.uint8)
    if max_px is not None and max(out.shape[:2]) > max_px:
        factor = max_px / max(out.shape[:2])
        out = cv2.resize(out, (int(round(out.shape[1] * factor)), int(round(out.shape[0] * factor))),
                         interpolation=cv2.INTER_AREA)
    return out


def stretch_to_uint8(
    image: np.ndarray, low: float = 0.1, high: float = 99.9, gamma: float = 1.0,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Percentile stretch for display only; the saved mosaic stays linear."""
    lo, hi = stretch_limits(image, low, high, mask)
    scaled = np.clip((image - lo) / (hi - lo), 0.0, 1.0)
    if abs(gamma - 1.0) > 1e-3:
        scaled = np.power(scaled, 1.0 / max(gamma, 1e-3))
    return (scaled * 255.0).astype(np.uint8)


def make_preview(
    image: np.ndarray, max_px: int = 1600, gamma: float = 1.0, mask: np.ndarray | None = None,
) -> np.ndarray:
    """8-bit, stretched, downscaled view of the mosaic."""
    preview = stretch_to_uint8(image, gamma=gamma, mask=mask)
    long_side = max(preview.shape[0], preview.shape[1])
    if long_side > max_px:
        scale = max_px / long_side
        preview = cv2.resize(
            preview, (int(round(preview.shape[1] * scale)), int(round(preview.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return preview


def encode_png(preview: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".png", preview if preview.ndim == 2 else preview[:, :, ::-1])
    if not ok:
        raise RuntimeError("could not encode the preview as PNG")
    return buffer.tobytes()
