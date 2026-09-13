"""Smoke test for the browser UI, driven headlessly.

The uploader cannot be filled programmatically, so the test drives the folder picker,
which is why that second way of choosing files earns its keep.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import SAMPLE_DIR
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "src" / "sunmosaic" / "app.py"


def _app() -> AppTest:
    return AppTest.from_file(str(APP), default_timeout=180)


def test_app_starts_and_offers_an_uploader():
    at = _app().run()
    assert not at.exception
    assert at.sidebar.file_uploader
    assert at.title[0].value.endswith("SunMosaic")


def test_app_asks_for_files_when_none_are_chosen(tmp_path):
    at = _app()
    at.session_state["folder"] = str(tmp_path)  # a real but empty folder
    at.run()
    assert not at.exception
    assert any("Choose" in info.value for info in at.info)
    assert any("No TIFF files" in caption.value for caption in at.caption)


def test_app_builds_and_offers_a_download():
    if not SAMPLE_DIR.is_dir() or len(sorted(SAMPLE_DIR.glob("*.tif"))) < 2:
        pytest.skip(f"sample frames not found in {SAMPLE_DIR}")

    at = _app()
    at.session_state["folder"] = str(SAMPLE_DIR)
    at.run()
    assert not at.exception

    names = [p.name for p in sorted(SAMPLE_DIR.glob("*.tif"))][:4]
    at.multiselect(key="folder_files").set_value(names).run()
    assert not at.exception

    at.button[0].click().run()
    assert not at.exception, at.exception

    assert at.session_state["result"] is not None
    assert len(at.session_state["tiff"]) > 1000
    assert at.image, "the mosaic should be displayed"
    labels = [b.label for b in at.download_button]
    assert any("TIFF" in label for label in labels)
    assert any("PNG" in label for label in labels)
    assert not at.error


def test_app_saves_straight_to_a_folder(tmp_path):
    if not SAMPLE_DIR.is_dir() or len(sorted(SAMPLE_DIR.glob("*.tif"))) < 2:
        pytest.skip(f"sample frames not found in {SAMPLE_DIR}")
    import tifffile

    at = _app()
    at.session_state["folder"] = str(SAMPLE_DIR)
    at.run()
    names = [p.name for p in sorted(SAMPLE_DIR.glob("*.tif"))][:4]
    at.multiselect(key="folder_files").set_value(names).run()
    at.button[0].click().run()
    assert not at.exception

    at.text_input(key="save_folder").set_value(str(tmp_path)).run()
    save = next(b for b in at.button if b.label == "Save")
    save.click().run()
    assert not at.exception
    written = tmp_path / "sun_mosaic.tif"
    assert written.exists(), [str(e.value) for e in at.error]
    assert any("Saved to" in s.value for s in at.success)
    image = tifffile.imread(str(written))
    assert image.dtype == "uint16"
    assert image.shape == at.session_state["result"].mosaic.shape

    # A second save must refuse to overwrite unless the box is ticked.
    next(b for b in at.button if b.label == "Save").click().run()
    assert any("already exists" in e.value for e in at.error)
    at.checkbox(key="save_overwrite").set_value(True).run()
    next(b for b in at.button if b.label == "Save").click().run()
    assert any("Saved to" in s.value for s in at.success)


def test_app_reports_a_bad_folder():
    at = _app()
    at.session_state["folder"] = "/no/such/folder/anywhere"
    at.run()
    assert not at.exception
    assert any("does not exist" in caption.value for caption in at.caption)


def _built_app() -> AppTest:
    """The app with the four sample frames built, or skip when they are absent."""
    if not SAMPLE_DIR.is_dir() or len(sorted(SAMPLE_DIR.glob("*.tif"))) < 2:
        pytest.skip(f"sample frames not found in {SAMPLE_DIR}")
    at = _app()
    at.session_state["folder"] = str(SAMPLE_DIR)
    at.run()
    names = [p.name for p in sorted(SAMPLE_DIR.glob("*.tif"))][:4]
    at.multiselect(key="folder_files").set_value(names).run()
    at.button[0].click().run()
    assert not at.exception
    return at


def test_app_orientation_previews_applies_and_undoes():
    at = _built_app()
    built_size = at.metric[0].value

    at.checkbox(key="finish_flip").set_value(True)
    at.slider(key="finish_rotate").set_value(30.0).run()
    assert not at.exception
    assert any("Previewing the new orientation" in info.value for info in at.info)
    assert at.session_state["working"] is None, "moving the slider alone must not change the data"

    next(b for b in at.button if b.label.startswith("Apply")).click().run()
    assert not at.exception
    assert at.session_state["applied_flip"] is True
    assert at.session_state["applied_angle"] == pytest.approx(30.0)
    assert at.slider(key="finish_rotate").value == 0.0
    assert at.checkbox(key="finish_flip").value is False
    side = max(int(v) for v in built_size.split(" x "))
    assert at.metric[0].value == f"{side} x {side}"
    assert any("flipped left to right" in c.value for c in at.caption)

    # a second Apply composes with the first instead of turning an already turned image
    at.slider(key="finish_rotate").set_value(60.0).run()
    next(b for b in at.button if b.label.startswith("Apply")).click().run()
    assert at.session_state["applied_angle"] == pytest.approx(90.0)

    next(b for b in at.button if b.label == "Undo").click().run()
    assert not at.exception
    assert at.session_state["working"] is None
    assert at.metric[0].value == built_size


def test_app_prominence_boost_offers_and_saves_both_layers(tmp_path):
    import numpy as np
    import tifffile

    at = _built_app()
    at.slider(key="finish_boost").set_value(10.0).run()
    assert not at.exception
    labels = [b.label for b in at.download_button]
    assert "Download as shown, with layers (16-bit TIFF)" in labels
    assert "Download linear mosaic, without prominences" in labels

    at.text_input(key="save_folder").set_value(str(tmp_path)).run()
    next(b for b in at.button if b.label == "Save").click().run()
    assert not at.exception, [e.value for e in at.error]
    linear = tifffile.imread(str(tmp_path / "sun_mosaic.tif"))
    boosted = tifffile.imread(str(tmp_path / "sun_mosaic_prominences.tif"))
    assert linear.shape == boosted.shape and boosted.dtype == np.uint16

    result = at.session_state["result"]
    assert np.array_equal(linear, result.mosaic), "the linear file must be the untouched mosaic"
    cx, cy = result.sun_center
    yy, xx = np.mgrid[0 : linear.shape[0], 0 : linear.shape[1]]
    rr = np.hypot(xx - cx, yy - cy)
    protected = rr <= result.sun_radius + 1.0
    assert np.array_equal(boosted[protected], linear[protected])
    outside = (rr > result.sun_radius + 20) & (result.label_map >= 0)
    assert float(np.median(boosted[outside])) > 5 * float(np.median(linear[outside]))

    from psdtags import PsdChannelId, TiffImageSourceData

    layers = TiffImageSourceData.fromtiff(str(tmp_path / "sun_mosaic_prominences.tif")).layers.layers
    assert len(layers) == 2
    assert np.array_equal(layers[1].asarray(channelid=PsdChannelId.CHANNEL0), linear)
    mask = layers[1].asarray(channelid=PsdChannelId.USER_LAYER_MASK)
    assert bool((mask[protected] == 65535).all())
