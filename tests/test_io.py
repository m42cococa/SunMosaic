"""Reading odd input files, and the round trip through a 16-bit TIFF."""

from __future__ import annotations

import json
from io import BytesIO

import numpy as np
import pytest
import tifffile

from sunmosaic.errors import InputError
from sunmosaic.io import make_preview, read_tile, stretch_to_uint8, tiff_bytes, validate_tiles
from sunmosaic.types import Tile


def _write(tmp_path, array, name="t.tif"):
    path = tmp_path / name
    tifffile.imwrite(str(path), array)
    return path


def test_uint16_is_read_unchanged(tmp_path):
    rng = np.random.default_rng(0)
    data = rng.integers(0, 65535, size=(300, 400), dtype=np.uint16)
    tile = read_tile(_write(tmp_path, data))
    assert tile.data.dtype == np.float32
    assert np.array_equal(tile.data, data.astype(np.float32))
    assert tile.warnings == []


def test_uint8_is_promoted_to_sixteen_bits(tmp_path):
    data = np.full((300, 400), 200, np.uint8)
    tile = read_tile(_write(tmp_path, data))
    assert float(tile.data.max()) == pytest.approx(200 * 257)
    assert any("8-bit" in w for w in tile.warnings)


def test_float_image_is_rescaled(tmp_path):
    data = np.full((300, 400), 0.5, np.float32)
    tile = read_tile(_write(tmp_path, data))
    assert float(tile.data.max()) == pytest.approx(65535 * 0.5, rel=1e-4)
    assert any("floating point" in w for w in tile.warnings)


def test_colour_image_keeps_its_channels(tmp_path):
    data = np.zeros((200, 260, 3), np.uint16)
    data[..., 1] = 1000
    tile = read_tile(_write(tmp_path, data))
    assert tile.channels == 3
    assert tile.gray.shape == (200, 260)
    assert any("olour" in w for w in tile.warnings)


def test_multipage_file_uses_the_first_page(tmp_path):
    pages = np.zeros((3, 300, 400), np.uint16)
    pages[0] = 1234
    tile = read_tile(_write(tmp_path, pages))
    assert tile.data.shape == (300, 400)
    assert float(tile.data.mean()) == pytest.approx(1234.0)
    assert any("several images" in w for w in tile.warnings)


def test_unreadable_file_is_reported(tmp_path):
    path = tmp_path / "broken.tif"
    path.write_bytes(b"not a tiff at all")
    with pytest.raises(InputError):
        read_tile(path)


def _dummy(width=400, height=300, name="t.tif") -> Tile:
    return Tile(name=name, data=np.zeros((height, width), np.float32),
                native_dtype=np.dtype(np.uint16), native_max=65535.0)


@pytest.mark.parametrize("count", [0, 1, 6])
def test_wrong_number_of_files_is_refused(count):
    with pytest.raises(InputError):
        validate_tiles([_dummy(name=f"t{i}.tif") for i in range(count)])


def test_tiny_file_is_refused():
    with pytest.raises(InputError):
        validate_tiles([_dummy(), _dummy(width=100, height=100)])


def test_mosaic_round_trips_through_tiff(quad_case):
    from sunmosaic.pipeline import stitch

    result = stitch(list(quad_case.tiles))
    data = tiff_bytes(result)
    with tifffile.TiffFile(BytesIO(data)) as handle:
        page = handle.pages[0]
        image = page.asarray()
        description = page.tags["ImageDescription"].value
    assert image.dtype == np.uint16
    assert image.shape == result.mosaic.shape
    assert np.array_equal(image, result.mosaic)
    meta = json.loads(description)
    assert meta["canvas"]["width"] == result.canvas_size[0]
    assert len(meta["tiles"]) == len(quad_case.tiles)
    assert meta["sun"]["radius_px"] > 0


def test_preview_is_eight_bit_and_bounded():
    data = np.linspace(0, 65535, 400 * 300, dtype=np.float32).reshape(300, 400)
    preview = make_preview(data, max_px=120)
    assert preview.dtype == np.uint8
    assert max(preview.shape) == 120
    assert stretch_to_uint8(data).max() <= 255


def test_layered_tiff_keeps_the_flat_image_and_the_layers(tmp_path):
    from psdtags import PsdChannelId, PsdKey, PsdLayerFlag, TiffImageSourceData

    from sunmosaic.io import ImageLayer, read_tile, write_layered_image

    rng = np.random.default_rng(3)
    bottom = rng.integers(0, 65535, (64, 80), dtype=np.uint16)
    top = rng.integers(0, 65535, (64, 80), dtype=np.uint16)
    mask = np.zeros((64, 80), np.uint16)
    mask[16:48, 20:60] = 65535
    flat = np.where(mask == 65535, top, bottom).astype(np.uint16)
    path = tmp_path / "layered.tif"
    write_layered_image(flat, [ImageLayer("Under", bottom), ImageLayer("Over", top, mask)],
                        {"note": "kept"}, path)

    # A reader that knows nothing about layers sees the flat image and the metadata.
    assert np.array_equal(tifffile.imread(str(path)), flat)
    with tifffile.TiffFile(str(path)) as tif:
        assert json.loads(tif.pages[0].tags["ImageDescription"].value) == {"note": "kept"}
    assert np.array_equal(read_tile(path).data, flat.astype(np.float32))

    # An editor finds both layers, bottom first, 16-bit, visible, with the mask on the top one.
    source = TiffImageSourceData.fromtiff(str(path))
    assert source.layers.key == PsdKey.LAYER_16
    under, over = source.layers.layers
    assert [under.name, over.name] == ["Under", "Over"]
    assert not (under.flags & PsdLayerFlag.VISIBLE) and not (over.flags & PsdLayerFlag.VISIBLE)
    assert np.array_equal(under.asarray(channelid=PsdChannelId.CHANNEL0), bottom)
    assert np.array_equal(over.asarray(channelid=PsdChannelId.CHANNEL0), top)
    assert np.array_equal(over.asarray(channelid=PsdChannelId.USER_LAYER_MASK), mask)
    assert not under.mask and tuple(over.mask.rectangle) == (0, 0, 64, 80)
