"""Browser UI for SunMosaic.

Thin on purpose: it collects files and settings, calls :func:`sunmosaic.pipeline.stitch`,
and renders the result.  Everything scientific lives in the package, so the same run can be
reproduced from the command line.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import streamlit as st

from sunmosaic import __version__
from sunmosaic.blend import seam_overlay
from sunmosaic.errors import SunMosaicError
from sunmosaic.finish import (
    FinishedImage,
    FinishParams,
    compose_orientation,
    finish,
    make_preview_source,
    render_preview,
)
from sunmosaic.io import (
    MAX_TILES,
    MIN_TILES,
    encode_png,
    finished_metadata,
    image_bytes,
    make_preview,
    prominence_file_bytes,
    read_tile,
    stretch_with_limits,
    tiff_bytes,
    write_image,
    write_prominence_file,
    write_tiff,
)
from sunmosaic.pipeline import stitch
from sunmosaic.types import MosaicResult, Params

DEFAULT_FOLDER = str(Path.home() / "Documents" / "Astronomy" / "SunMosaic")
FINISH_WIDGETS = {"finish_flip": False, "finish_rotate": 0.0}

st.set_page_config(page_title="SunMosaic", page_icon="☉", layout="wide")


def signature(payload: list[tuple[str, bytes]], params: Params) -> str:
    digest = hashlib.sha256()
    for name, data in payload:
        digest.update(name.encode("utf-8"))
        digest.update(hashlib.sha256(data).digest())
    digest.update(repr(params.key()).encode("utf-8"))
    return digest.hexdigest()


def collect_inputs() -> list[tuple[str, bytes]]:
    """Files chosen either by upload or from a folder on this machine."""
    st.sidebar.header("Sun frames")
    uploaded = st.sidebar.file_uploader(
        f"Drop {MIN_TILES} to {MAX_TILES} TIFF files",
        type=["tif", "tiff"], accept_multiple_files=True,
    )
    payload = [(f.name, f.getvalue()) for f in uploaded] if uploaded else []

    with st.sidebar.expander("...or pick from a folder on this Mac", expanded=not payload):
        folder = st.text_input("Folder", value=DEFAULT_FOLDER, key="folder")
        candidates: list[Path] = []
        if folder:
            path = Path(folder).expanduser()
            if path.is_dir():
                candidates = sorted(
                    p for p in path.iterdir()
                    if p.suffix.lower() in (".tif", ".tiff") and p.is_file()
                )
                if not candidates:
                    st.caption("No TIFF files in that folder.")
            else:
                st.caption("That folder does not exist.")
        names = [p.name for p in candidates]
        chosen = st.multiselect(
            "Files", names, default=names[:4] if not payload else [], key="folder_files",
        )
        if chosen and not payload:
            lookup = {p.name: p for p in candidates}
            payload = [(name, lookup[name].read_bytes()) for name in chosen]
    return payload


def collect_params() -> tuple[Params, bool]:
    with st.sidebar.expander("Options", expanded=False):
        blend = st.selectbox(
            "Blending", ["multiband", "feather", "hard"], index=0,
            help="Multiband hides the brightness differences between frames. "
                 "Hard shows the raw seams, which is useful to judge how well they match.",
        )
        equalize = st.selectbox(
            "Brightness matching",
            ["linear", "quadratic", "gain+offset", "gain", "off"], index=0,
            help="Corrects the exposure and background differences between frames. "
                 "Linear and quadratic also correct the illumination across each frame, "
                 "which is what the etalon sweet spot causes.",
        )
        interp = st.selectbox(
            "Sub-pixel placement", ["cubic", "lanczos", "linear", "integer"], index=0,
            help="How frames are shifted by a fraction of a pixel. "
                 "Integer does not resample the image at all.",
        )
        gamma = st.slider(
            "Preview brightness", 0.4, 2.5, 1.0, 0.1,
            help="Affects the on-screen preview only. The saved file stays linear.",
        )
        show_seams = st.checkbox("Show seam lines on the preview", value=False)
        refine_rotation = st.checkbox(
            "Correct rotation between frames", value=True,
            help="On an alt-azimuth mount the field turns during the sequence. This measures "
                 "the angle between frames and straightens them. It only acts when the frames "
                 "actually disagree, so it costs nothing when they do not.",
        )
    with st.sidebar.expander("Advanced", expanded=False):
        min_response = st.slider("Minimum match confidence", 0.05, 0.9, 0.30, 0.05)
        crop_margin = st.slider("Alignment search margin (px)", 10, 120, 40, 5)
        levels = st.selectbox("Blend depth", ["automatic", 3, 4, 5, 6], index=0)
    return Params(
        blend=blend, equalize=equalize, interp=interp, preview_gamma=gamma,
        min_response=min_response, crop_margin=crop_margin,
        refine_rotation=refine_rotation,
        pyramid_levels=None if levels == "automatic" else int(levels),
    ), show_seams


def show_thumbnails(payload: list[tuple[str, bytes]]) -> None:
    columns = st.columns(len(payload))
    for column, (name, data) in zip(columns, payload):
        with column:
            try:
                tile = read_tile(data, name)
            except SunMosaicError as exc:
                st.error(str(exc))
                continue
            thumb = make_preview(tile.data, max_px=260)
            st.image(thumb, width="stretch")
            st.caption(f"**{name}**\n\n{tile.width} x {tile.height}")


def diagnostics(result: MosaicResult) -> None:
    tiles_table = [
        {
            "File": name,
            "Placed by": {"limb": "solar limb", "phasecorr": "image texture",
                          "template": "image texture"}.get(p.source, p.source),
            "Limb arc": f"{lf.arc_deg:.0f} deg" if lf else "none",
            "Turned by": f"{np.degrees(p.rotation):+.2f} deg" if p.rotation else "",
            "Radius (px)": round(lf.r, 1) if lf else None,
            "Position x": round(p.x, 2),
            "Position y": round(p.y, 2),
            "Brightness": round(p.gain, 4),
            "Background": round(p.offset, 1),
        }
        for name, p, lf in zip(result.tile_names, result.placements, result.limbs)
    ]
    pairs_table = [
        {
            "Frames": f"{p.i + 1} and {p.j + 1}",
            "Shared area": f"{p.overlap_w} x {p.overlap_h}",
            "Confidence": round(p.response, 3),
            "Mismatch (px)": None if p.residual_px is None else round(p.residual_px, 2),
            "Used": "yes" if p.accepted else "no",
            "Reason": p.reject_reason or "",
        }
        for p in result.pairs
    ]
    st.dataframe(tiles_table, width="stretch", hide_index=True)
    st.dataframe(pairs_table, width="stretch", hide_index=True)
    st.caption(
        "Time spent: "
        + ", ".join(f"{step} {seconds:.1f}s" for step, seconds in result.timings.items())
    )


# --- finishing state -------------------------------------------------------------------------


def reset_finishing() -> None:
    """Forget any applied orientation; used when a new mosaic is built or on Undo."""
    st.session_state["applied_flip"] = False
    st.session_state["applied_angle"] = 0.0
    st.session_state["working"] = None
    st.session_state.pop("preview_working", None)
    for key, value in FINISH_WIDGETS.items():
        st.session_state[key] = value


def apply_orientation() -> None:
    """Button callback: bake the flip and turn into a full-resolution square image.

    The total orientation is always applied to the built mosaic, never to an image that was
    already turned, so pressing Apply several times costs a single resampling.
    """
    result: MosaicResult = st.session_state["result"]
    flip, angle = compose_orientation(
        st.session_state.get("applied_flip", False), st.session_state.get("applied_angle", 0.0),
        bool(st.session_state.get("finish_flip", False)),
        float(st.session_state.get("finish_rotate", 0.0)),
    )
    working = finish(result, FinishParams(flip=flip, rotation_deg=angle, square=True))
    st.session_state["applied_flip"] = flip
    st.session_state["applied_angle"] = angle
    st.session_state["working"] = working
    st.session_state.pop("preview_working", None)
    for key, value in FINISH_WIDGETS.items():
        st.session_state[key] = value


def preview_sources(result: MosaicResult, working: FinishedImage | None):
    """Downsampled linear copies for live rendering, built once per image."""
    if "preview_result" not in st.session_state:
        st.session_state["preview_result"] = make_preview_source(
            result.mosaic, result.label_map >= 0, result.sun_center, result.sun_radius,
            result_params_preview_px(),
        )
    if working is not None and "preview_working" not in st.session_state:
        st.session_state["preview_working"] = make_preview_source(
            working.image, working.coverage, working.sun_center, working.sun_radius,
            result_params_preview_px(),
        )
    return st.session_state["preview_result"], st.session_state.get("preview_working")


def result_params_preview_px() -> int:
    return Params().preview_max_px


def finishing_controls(has_applied: bool) -> tuple[bool, float, FinishParams]:
    """The Orientation and Prominences panels.  Returns pending flip, angle and boost params."""
    with st.expander("Orientation", expanded=True):
        flip = st.checkbox("Flip (mirror left to right)", key="finish_flip")
        angle = st.slider(
            "Rotate (degrees, counter-clockwise)", -180.0, 180.0, step=0.1, key="finish_rotate",
            help="The disk turns about its measured centre. The preview follows the slider; "
                 "press Apply to turn the full-resolution image.",
        )
        apply_col, undo_col = st.columns(2)
        apply_col.button(
            "Apply rotation and make square", on_click=apply_orientation, width="stretch",
            help="Turns the full-resolution mosaic, pads it to a square with the Sun in the "
                 "middle, and fills the exposed corners so the sky stays continuous.",
        )
        undo_col.button("Undo", on_click=reset_finishing, disabled=not has_applied,
                        width="stretch")

    with st.expander("Prominences", expanded=True):
        boost = st.slider(
            "Boost", 1.0, 40.0, 1.0, 0.5, key="finish_boost",
            help="Brightens everything outside the disk to reveal prominences. The disk and a "
                 "margin beyond its edge are kept from the original. 1 means off.",
        )
        remove_glow = st.checkbox(
            "Remove limb glow", value=True, key="finish_glow",
            help="The light scattered just outside the limb is the same all the way round, "
                 "so it is measured ring by ring and subtracted before boosting. Without "
                 "this, boosting turns that glow into a bright ring around the disk.",
        )
        border_col, feather_col = st.columns(2)
        border = border_col.number_input(
            "Margin (px)", 0.0, 20.0, 1.0, 0.5, key="finish_border",
            help="Kept from the original beyond the measured limb, so no part of the disk is cut.",
        )
        feather = feather_col.number_input(
            "Soft edge (px)", 0.0, 30.0, 4.0, 0.5, key="finish_feather",
            help="Width of the blend outside the margin, so the two layers meet without a hard line.",
        )
    return flip, float(angle), FinishParams(
        prominence_boost=float(boost), prominence_border_px=float(border),
        prominence_feather_px=float(feather), remove_glow=bool(remove_glow),
    )


def main() -> None:
    st.title("☉ SunMosaic")
    st.caption(
        "Join partial H-alpha views of the Sun into one full-disk mosaic, and save it as a "
        "16-bit TIFF."
    )

    payload = collect_inputs()
    params, show_seams = collect_params()

    if not payload:
        st.info(
            f"Choose {MIN_TILES} to {MAX_TILES} TIFF frames in the sidebar. They should be "
            "overlapping views of the same Sun, taken minutes apart."
        )
        st.stop()
    if len(payload) > MAX_TILES:
        st.error(f"Choose at most {MAX_TILES} files. You picked {len(payload)}.")
        st.stop()

    show_thumbnails(payload)
    run = st.button(
        "Create mosaic", type="primary", disabled=len(payload) < MIN_TILES,
        help=None if len(payload) >= MIN_TILES else f"Pick at least {MIN_TILES} files",
    )

    key = signature(payload, params)
    if run:
        try:
            with st.status("Building the mosaic...", expanded=True) as status:
                def report(step: str, fraction: float) -> None:
                    status.update(label=f"{step}...")
                tiles = [read_tile(data, name) for name, data in payload]
                result = stitch(tiles, params, report)
                status.update(label="Mosaic ready", state="complete", expanded=False)
        except SunMosaicError as exc:
            st.error(str(exc))
            st.stop()
        st.session_state["result"] = result
        st.session_state["result_key"] = key
        st.session_state["tiff"] = tiff_bytes(result)
        st.session_state.pop("preview_result", None)
        reset_finishing()

    result = st.session_state.get("result")
    if result is None:
        st.stop()
    if st.session_state.get("result_key") != key:
        st.warning("The files or settings changed. Press **Create mosaic** to rebuild.")
    st.session_state.setdefault("applied_flip", False)
    st.session_state.setdefault("applied_angle", 0.0)
    st.session_state.setdefault("working", None)
    working: FinishedImage | None = st.session_state["working"]

    picture_col, controls_col = st.columns([3, 1])
    with controls_col:
        pending_flip, pending_angle, boost_params = finishing_controls(working is not None)

    source_result, source_working = preview_sources(result, working)
    pending_geometry = pending_flip or abs(pending_angle) > 1e-9
    if pending_geometry:
        total_flip, total_angle = compose_orientation(
            st.session_state["applied_flip"], st.session_state["applied_angle"],
            pending_flip, pending_angle,
        )
        render_params = FinishParams(flip=total_flip, rotation_deg=total_angle, square=True)
        source = source_result
    elif working is not None:
        render_params = FinishParams()
        source = source_working
    else:
        render_params = FinishParams()
        source = source_result
    render_params.prominence_boost = boost_params.prominence_boost
    render_params.prominence_border_px = boost_params.prominence_border_px
    render_params.prominence_feather_px = boost_params.prominence_feather_px
    render_params.remove_glow = boost_params.remove_glow

    shown = render_preview(source, render_params)
    lo, hi = source.stretch
    preview = stretch_with_limits(shown, lo, hi, params.preview_gamma)
    untouched = not pending_geometry and working is None
    with picture_col:
        if show_seams and untouched and not boost_params.boosts:
            st.image(seam_overlay(preview, result.label_map), width="stretch")
        else:
            st.image(preview, width="stretch")
            if show_seams:
                st.caption("Seam lines are shown only on the untouched mosaic.")
        if pending_geometry:
            st.info("Previewing the new orientation. Press **Apply rotation and make square** "
                    "to include it in the saved files.")

    if working is not None:
        width, height = working.side
    else:
        width, height = result.canvas_size
    columns = st.columns(4)
    columns[0].metric("Image size", f"{width} x {height}")
    columns[1].metric("Solar radius", f"{result.sun_radius:.0f} px")
    columns[2].metric("Worst mismatch", f"{result.worst_residual:.2f} px")
    columns[3].metric("Build time", f"{max(result.timings.values()):.1f} s")
    if working is not None:
        turned = st.session_state["applied_angle"]
        st.caption(
            f"Orientation applied: {'flipped left to right, then ' if st.session_state['applied_flip'] else ''}"
            f"turned {turned:+.1f} degrees, square {width} x {height} with the Sun centred, "
            f"{working.filled_fraction:.1%} of it filled."
        )

    for warning in result.warnings + (working.warnings if working is not None else []):
        st.warning(warning)

    finished_params = FinishParams(
        flip=st.session_state["applied_flip"], rotation_deg=st.session_state["applied_angle"],
        square=working is not None,
        prominence_boost=boost_params.prominence_boost,
        prominence_border_px=boost_params.prominence_border_px,
        prominence_feather_px=boost_params.prominence_feather_px,
        remove_glow=boost_params.remove_glow,
    )
    stem = f"sun_mosaic_{Path(result.tile_names[0]).stem}"

    def linear_bytes() -> bytes:
        if working is None:
            return st.session_state["tiff"]
        return image_bytes(working.image, finished_metadata(result, working, "linear"))

    def shown_bytes() -> bytes:
        return prominence_file_bytes(result, finish(result, finished_params))

    left, middle, right = st.columns(3)
    if boost_params.boosts:
        with left:
            st.download_button(
                "Download as shown, with layers (16-bit TIFF)", data=shown_bytes,
                file_name=f"{stem}_prominences.tif", mime="image/tiff", type="primary",
                width="stretch",
            )
        with middle:
            st.download_button(
                "Download linear mosaic, without prominences", data=linear_bytes,
                file_name=f"{stem}.tif", mime="image/tiff", width="stretch",
            )
    else:
        with left:
            st.download_button(
                "Download mosaic (16-bit TIFF)", data=linear_bytes,
                file_name=f"{stem}.tif", mime="image/tiff", type="primary", width="stretch",
            )
    with right:
        st.download_button(
            "Download preview (PNG)", data=encode_png(preview),
            file_name="sun_mosaic_preview.png", mime="image/png", width="stretch",
        )

    with st.expander("Save straight to a folder"):
        folder = st.text_input("Folder", value=DEFAULT_FOLDER, key="save_folder")
        filename = st.text_input("File name", value="sun_mosaic.tif", key="save_name")
        overwrite = st.checkbox("Replace the file if it already exists", value=False,
                                key="save_overwrite")
        if boost_params.boosts:
            st.caption("The linear mosaic and a second file ending in _prominences are written. "
                       "The second holds the boosted copy and the masked disk as layers.")
        if st.button("Save"):
            target = Path(folder).expanduser() / filename
            targets = [target]
            if boost_params.boosts:
                targets.append(target.with_name(f"{target.stem}_prominences{target.suffix or '.tif'}"))
            clash = [t for t in targets if t.exists()]
            if not target.parent.is_dir():
                st.error(f"{target.parent} is not a folder.")
            elif clash and not overwrite:
                st.error(f"{clash[0]} already exists. Tick the box above to replace it.")
            else:
                if working is None:
                    write_tiff(result, target)
                else:
                    write_image(working.image, finished_metadata(result, working, "linear"), target)
                if boost_params.boosts:
                    write_prominence_file(result, finish(result, finished_params), targets[1])
                st.success("Saved to " + " and ".join(str(t) for t in targets))

    with st.expander("How the frames were joined"):
        diagnostics(result)

    st.caption(f"SunMosaic {__version__}")


main()
