# SunMosaic — H-alpha Sun Mosaic Stitching Tool: Implementation Plan

> This document is the project reference. The plan as approved is below; an **As built**
> section at the end records what changed during implementation and what was left out.

## 1. Context

The full H-alpha solar disk does not fit in one camera frame at the imaging scale used
(disk diameter ≈ 1594 px vs. frames of ≈ 1600 × 1090 px). The Sun is therefore captured as
several overlapping partial views (typically 4 quadrants, sometimes a 5th tile), each stacked
with AutoStakkert into a 16-bit TIFF. Today these tiles have to be assembled by hand in an
image editor. The goal is a small local Python tool, driven from the browser, that:

1. lets the user select 2–5 TIFF tiles,
2. registers, equalizes and blends them into one seamless full-disk mosaic,
3. shows the result, and
4. saves it as a 16-bit TIFF at native pixel scale.

The tool must be simple, robust on real data (illumination gradients from the etalon "sweet
spot", small overlaps on one axis, slightly different tile sizes) and give clear diagnostics
when something does not fit.

## 2. What we know about the input (verified on the sample set)

Sample set: `~/Documents/Astronomy/SunMosaic/2026-09-08-203{8_0,8_5,9_0,9_5}-Sun_lapl5_ap*_conv.tif`

| Property | Value |
|---|---|
| Format | TIFF, little-endian, 16-bit unsigned, 1 channel (mono), ≈ 3.5 MB each, AutoStakkert output |
| Tile sizes (rows × cols) | 1096×1600, 1088×1592, 1088×1600, 1088×1592 — **not identical** |
| Pixel range | min ≈ 540 (sky), max ≈ 61 000–65 469 (one tile is near saturation), means 25 500–27 700 |
| Layout | 2×2 quadrants, shot clockwise UR → LR → LL → UL within ≈ 90 s |
| Sun radius (limb fit) | R ≈ 797.2 px, per-tile 796.8–797.7, fit residual ≈ 0.3 px, arc ≈ 130° per tile |
| Overlap, horizontal neighbours | ≈ 1130–1170 px (≈ 71 % of width) |
| Overlap, vertical neighbours | ≈ 259–324 px (≈ 24–30 % of height) — the hard case |
| Brightness mismatch | tile-to-tile gain 0.92–1.08 (median in overlap); p10–p90 of the ratio 0.82–1.18 ⇒ spatially varying illumination, not just a constant. **This, not registration, is the main seam-visibility risk.** |
| Sky | dark, noisy, ≈ 540–600 ADU (differs per tile); Otsu threshold (≈ 81/255) separates disk from sky cleanly |
| Prominences | present beyond the limb ⇒ the canvas must keep sky around the disk, never crop to the disk |
| Resulting canvas | ≈ 2062 × 1917 px |

Registration experiments already done (results the design builds on):

- Full-frame phase correlation on high-pass images works for the 71 %-overlap pairs
  (response 0.58–0.64) but **fails** on the 25 %-overlap pairs (response ≤ 0.03, wrong offset).
- Limb circle fit per tile gives a coarse placement accurate to ≈ 1 px.
- Limb-fit coarse placement + phase correlation of the **predicted overlap crops** succeeds on
  all 6 pairs (responses 0.73–0.88, residual corrections 0.07–1.3 px).

Environment: macOS, Python 3.14.5 (`/opt/homebrew/bin/python3`), `uv` available. All chosen
dependencies resolve for 3.14 (dry-run verified: numpy, opencv-python-headless 5.0, tifffile
2026.9, imagecodecs 2026.8, streamlit 1.63). Resolution is not the same as working wheels, hence
the environment gate in §7.

## 3. Decisions

| Decision | Choice | Why |
|---|---|---|
| UI framework | **Streamlit** | Least code for exactly this feature set: multi-file picker, button, status, image display, download button, sliders; `streamlit.testing.AppTest` for a headless smoke test. Gradio is comparable; Flask/FastAPI + HTML is ~3× the code. |
| Motion model | **Translation only** (per tile: x, y, sub-pixel) | Same camera/optics, minutes apart. Rotation is detected via the loop-closure residual and handled as a later enhancement. |
| Registration | Limb-circle coarse placement → overlap-crop phase correlation → weighted least-squares global placement (with a weak limb prior) | Proven on the sample data, robust to 25 % overlap, independent of texture evolution. |
| Photometry | Sky pedestal + per-tile multiplicative gain + per-tile sky offset, all solved globally from overlap statistics (MVP); ridge-regularized low-order gain field in phase 3 | Constant gain removes the 8 % steps and offsets remove sky steps that a hard stretch would reveal; multi-band blending hides the residual gradient at the seam. |
| Blending | Distance-transform seams (Voronoi-like) + Laplacian-pyramid multi-band blend in float32; feather and hard-seam as options | Standard, seam-free, keeps fine H-alpha detail (one tile per pixel at fine scales, no ghosting). |
| Resampling | Sub-pixel shift with `cv2.warpAffine`, `INTER_CUBIC` default; options `lanczos`, `linear`, `integer` (no resampling) | Cubic keeps stacked detail crisp and rings less than Lanczos at the sharp limb; integer mode for purists. Pixel scale is never changed. |
| Output | 16-bit TIFF via `tifffile`, linear data (no stretch), same dtype as input, JSON metadata in ImageDescription; 8-bit auto-stretched PNG for display only | Astronomers post-process elsewhere (ImPPG, PixInsight, Photoshop); the mosaic must stay linear. |
| File selection | Browser uploader (primary) **and** a local folder path field listing `.tif` files (convenience) | Files live on the same machine; both are cheap. |
| Saving | Download button (primary) **and** optional "save to folder" path field | Download always works; path save avoids the Downloads detour. |
| Dependencies | numpy, opencv-python-headless, tifffile, imagecodecs, streamlit; dev: pytest, ruff | No scipy/scikit-image: OpenCV covers threshold, contours, distance transform, pyramids, phase correlation; `numpy.linalg.lstsq` covers all solves. |
| Packaging | `uv` + `pyproject.toml`, `src/` layout, `requires-python >= 3.12`, hatchling; run with `uv run sunmosaic-ui` | Isolated venv, reproducible; CLI and UI share one package. Fallback `uv python pin 3.12` if any 3.14 wheel breaks. |

## 4. Architecture

```
SunMosaic/
├── plan.md                    # this document
├── README.md                  # install / run / usage / screenshot
├── pyproject.toml             # uv-managed project, deps, scripts, pytest config
├── .streamlit/config.toml     # server.maxUploadSize=500, browser.gatherUsageStats=false
├── src/sunmosaic/
│   ├── __init__.py            # __version__
│   ├── types.py               # dataclasses: Tile, LimbFit, PairMatch, Placement, Params, MosaicResult
│   ├── errors.py              # SunMosaicError, InputError, RegistrationError (actionable messages)
│   ├── io.py                  # read_tile(path|bytes) -> Tile ; write_tiff(result, path|BytesIO) ; make_preview(...)
│   ├── preprocess.py          # to_float(), disk_mask(), sky_stats(), highpass(), hanning()
│   ├── limb.py                # fit_limb(tile) -> LimbFit | None
│   ├── register.py            # coarse_place(), refine_pair(), place_limbless(), solve_global()
│   ├── warp.py                # canvas geometry, warp_tile() (cubic/lanczos/linear/integer), validity masks
│   ├── photometric.py         # pair_stats(), solve_gains_offsets(), apply(); (phase 3) gain_field()
│   ├── blend.py               # seam_labels(), multiband_blend(), feather_blend(), hard_blend(), seam_overlay()
│   ├── pipeline.py            # stitch(tiles, params, progress_cb) -> MosaicResult  (the only entry point UI/CLI use)
│   ├── synthetic.py           # synthetic Sun + tile cutter with exact ground truth (tests, and a UI demo mode later)
│   ├── cli.py                 # sunmosaic build a.tif b.tif ... -o mosaic.tif [--preview p.png --report r.json]
│   ├── app.py                 # Streamlit UI (thin: widgets + session_state + stitch())
│   └── launch.py              # `sunmosaic-ui` entry point: runs streamlit on app.py
└── tests/
    ├── conftest.py            # synthetic fixtures; sample-dir fixture (skips if absent)
    ├── test_io.py, test_preprocess.py, test_limb.py, test_register.py,
    ├── test_photometric.py, test_warp.py, test_blend.py
    ├── test_pipeline_synthetic.py
    ├── test_samples.py        # regression on the 4 real tiles
    └── test_app.py            # streamlit.testing.v1.AppTest smoke test
```

Principles: the pipeline is pure numpy/OpenCV with no Streamlit imports and is fully usable from
the CLI and tests; the UI only converts uploads to `Tile`s, calls `stitch()`, and renders
`MosaicResult`. All images are `float32` in native ADU internally; dtype is restored on output.
Progress is reported through a `progress_cb(step_name, fraction)` callback.

Key dataclasses (`types.py`):

```python
@dataclass
class Tile:        name: str; data: np.ndarray        # float32 HxW (or HxWx3), native ADU
                   native_dtype: np.dtype; native_max: float
                   disk_mask: np.ndarray; highpass: np.ndarray; sky_median: float; sky_mad: float
@dataclass
class LimbFit:     cx: float; cy: float; r: float; n_points: int; arc_deg: float; resid_px: float
                   trusted: bool                       # None (no LimbFit) = limb-less tile
@dataclass
class PairMatch:   i: int; j: int; overlap_w: int; overlap_h: int; method: str  # "crop-pc" | "full-pc" | "template"
                   dx: float; dy: float                # measured displacement of j relative to i
                   response: float; accepted: bool; reject_reason: str | None
                   residual_px: float | None           # after the global solve
@dataclass
class Placement:   x: float; y: float                  # tile origin in canvas coordinates (sub-pixel)
                   gain: float; offset: float; source: str   # "limb" | "phasecorr" | "template" | "prior"
@dataclass
class Params:      highpass_sigma=25.0; crop_margin=40; min_overlap_px=150; min_response=0.30
                   min_arc_deg=60.0; max_limb_resid=1.5; max_loop_resid=1.5
                   interp="cubic"        # "cubic" | "lanczos" | "linear" | "integer"
                   blend="multiband"     # "multiband" | "feather" | "hard"
                   equalize="gain+offset"  # "off" | "gain" | "gain+offset" ; phase 3 adds "linear" | "quadratic"
                   pyramid_levels=None; preview_max_px=1600
@dataclass
class MosaicResult: mosaic: np.ndarray; native_dtype; canvas_size: (w, h)
                   placements: list[Placement]; limbs: list[LimbFit | None]; pairs: list[PairMatch]
                   sun_center: (x, y); sun_radius: float; label_map: np.ndarray
                   warnings: list[str]; timings: dict[str, float]
```

## 5. Pipeline (step by step)

### 5.1 Load & validate (`io.py`)
- `tifffile.imread` from a path or `BytesIO` (Streamlit gives bytes). Squeeze singleton dims;
  multi-page TIFF → page 0 with a warning.
- Accept uint16 (primary), uint8 (scaled ×257, warning), float (rescaled, warning); mono or RGB.
  RGB: registration and limb fitting use the channel mean; all channels are warped and blended.
- Convert to `float32` in native ADU; remember `native_dtype` and `native_max` (65535 / 255).
- Reject with an `InputError` naming the file: fewer than 2 or more than 5 tiles, mixed dtypes,
  any dimension < 256 px, any tile > 30 Mpx (memory guard).

### 5.2 Preprocess (`preprocess.py`)
- `disk_mask`: normalize by the 99.5th percentile to 8-bit → Otsu threshold → morphological open
  7×7 → keep the largest connected component. Also an eroded version (15 px) for photometry.
- `sky_stats`: median and MAD of pixels outside the disk mask dilated by 60 px (keeps prominences
  out); fallback to the 1st percentile if too few pixels.
- `highpass`: `img − GaussianBlur(img, σ = 25)`, divided by its own std. **Computed once on the
  full tile**, then cropped later; never high-pass a crop (edge artifacts inside the thin
  vertical overlaps). Removes the etalon gradient and limb darkening so correlation is driven by
  chromospheric texture and the limb edge.
- `hanning`: `cv2.createHanningWindow` per crop size, cached.

### 5.3 Limb fit (`limb.py`)
1. Largest external contour of the disk mask (`CHAIN_APPROX_NONE`); drop points within 3 px of the
   image border (frame edges, not limb).
2. Algebraic (Kåsa) least-squares circle fit, then 2 robust iterations dropping points with
   `|residual| > max(3·MAD, 1.0 px)` and refitting.
3. `arc_deg` = number of 5° angular bins (around the fitted centre) holding ≥ 3 points × 5;
   `resid_px` = RMS of `|dist − R|`.
4. `trusted` iff `arc_deg ≥ 60`, `resid_px ≤ 1.5`, `n_points ≥ 200`, `R > 0.25·min(H, W)`.
   30–60° → untrusted (initial guess only). < 30° or no interior contour → `None` (limb-less).
5. Cross-tile check: median R over trusted tiles; a tile with `|R − median| > 1 %` is demoted to
   untrusted with a warning. Sample values: arc 129–135°, resid ≈ 0.3 px, R spread 0.9 px ⇒ all trusted.

### 5.4 Coarse placement (`register.py: coarse_place`, `place_limbless`)
- Every tile with a limb fit (trusted or untrusted): origin `t_i = (−cx_i, −cy_i)` in a frame where
  the Sun centre is (0, 0). Source `"limb"`.
- Limb-less tiles (e.g. a 5th central tile), placed after the others:
  - MVP: full-frame phase correlation (padded to a common size, Hanning) against each placed tile;
    best response wins; accept if response ≥ 0.2 and the implied overlap ≥ 30 % of the tile area.
    Source `"phasecorr"`. A central tile overlaps every quadrant tile by > 50 %, which is the regime
    where full-frame phase correlation works (0.58–0.64 on the 71 % pairs).
  - Phase 3 upgrade: template matching (`cv2.matchTemplate`, `TM_CCOEFF_NORMED`) of the ×0.25
    downsampled high-pass tile against a ×0.25 partial mosaic of the placed tiles; accept peak ≥ 0.3;
    origin = peak × 4. Source `"template"`. A tile contained in the mosaic is exactly this case.
- If no tile has a limb at all: chain tiles by full-frame phase correlation from tile 0 (maximum
  response spanning tree) with a warning that placement is texture-only.
- A tile that no method can place ⇒ `RegistrationError("Tile <name> could not be placed: best
  response 0.04. Check that the files overlap.")`.

### 5.5 Pairwise refinement (`register.py: refine_pair`)
For every pair of placed tiles whose predicted overlap is ≥ `min_overlap_px` (150) in both dimensions:
1. Overlap rectangle in the common frame, shrunk by `crop_margin` (40 px) per side; for thin
   overlaps use 15 % of the smaller dimension, minimum 16 px.
2. Crop **equal-size** windows from `highpass_i` and `highpass_j` at **integer** positions `o_i`,
   `o_j`; record `frac = (o_j − t_j) − (o_i − t_i)`, the rounding difference.
3. `cv2.phaseCorrelate(crop_i, crop_j, hanning)` → `(dx, dy), response`.
4. Measured displacement `d_ij = (t_j − t_i)_pred + sign·(dx, dy) + frac`, where `sign` is
   **pinned by a unit test** written before this code (shift a texture by a known vector with
   `np.roll`, assert the convention). This is the most common silent bug.
5. Accept iff `response ≥ min_response (0.30)` and `|correction| ≤ crop_margin`. Otherwise record
   the pair as rejected with a reason. Sample data: successes 0.73–0.88, failures ≤ 0.03.

### 5.6 Global placement (`register.py: solve_global`)
- Unknowns: origin `t_i` per tile; the tile with the most accepted pairs (tie → largest arc) is
  fixed as gauge. x and y are solved independently with `np.linalg.lstsq`.
- Equations: `t_j − t_i = d_ij` for accepted pairs, weight `clip(response, 0.3, 1.0)`; plus
  **weak limb-prior equations** `t_i − t_k = (cx_k, cy_k) − (cx_i, cy_i)` with weight 0.02 for
  trusted limb tiles. The prior keeps the graph connected and pins a tile that lost all its pairs
  without ever overriding a measured pair.
- Per-pair residual after the solve → `PairMatch.residual_px`. With a 2×2 grid the pairs form a
  cycle, so this is a real consistency check. Max residual > `max_loop_resid` (1.5 px) ⇒ warning
  "tiles are not consistent with a pure translation (field rotation?)"; placement is still used.
  Alt-az field rotation over 90 s can reach a few px at R ≈ 800, so this warning will matter.
- A tile with no accepted pair and no trusted limb ⇒ `RegistrationError`.
- Shift all origins so the minimum is (0, 0); canvas = bounding box of all tiles, rounded up.
  Sun centre and radius are reported in canvas coordinates. Sample expectation: 2062 × 1917.

### 5.7 Canvas & warp (`warp.py`)
- Canvas padded (edge-replicated) to a multiple of `2^levels` for the pyramids; cropped back at
  the end.
- Per tile: integer part of the origin is a slice placement; the fractional part goes through
  `cv2.warpAffine` (`INTER_CUBIC` default; `lanczos`/`linear` options; `integer` rounds the
  origin and skips resampling entirely).
- Validity mask = warped ones-image, eroded 2 px (drops the soft interpolation edge). Tiles are
  rectangles, so validity regions are rectangles.
- Uncovered canvas pixels (notches from the differing tile sizes) are filled with the sky pedestal
  and their count reported.

### 5.8 Photometric equalization (`photometric.py`, in canvas space)
Model per tile, in estimation order: the sky pedestal `b` (median of all tiles' sky medians) is
subtracted; a multiplicative gain `g_i` is solved on disk pixels; then a residual sky offset `o_i`
is measured on the gain-corrected image and removed. The apply formula in step 3 follows this order.
1. **Gains.** For each accepted pair, sample the overlap on a 16 px grid restricted to: both eroded
   disk masks true, value < 64 000 (saturation), value > `b + 10·sky_mad`; values pre-smoothed with
   σ = 8 to suppress noise and seeing mismatch. Observation `y_ij = median(log(I_j − b) − log(I_i − b))`.
   Solve `a_j − a_i = y_ij` (weight = response) with gauge `Σ a_i = 0` (overall brightness
   preserved); `g_i = exp(a_i)`. Clip to `[0.7, 1.4]` with a warning. Sample expectation: 0.92–1.08.
2. **Sky offsets.** After gains, per accepted pair the median difference over overlap sky pixels
   (disk mask dilated 60 px excluded, ≥ 2000 px required); solve additive `o_i` in the same
   least-squares form with `Σ o_i = 0`. Skipped with a warning when a pair has too little sky.
3. **Apply:** `I'_i = (I_i − b)/g_i + b − o_i`. Prominences scale with the disk gain (same etalon
   transmission).
4. **Phase 3 — gain field** (`equalize="linear"|"quadratic"`): `log g_i(u, v) = a_i + poly_i(u, v)`
   with (u, v) tile coordinates in [−1, 1]; ridge prior on the non-constant terms (σ_lin 0.15,
   σ_quad 0.10 in log units) because a smooth field common to all tiles is invisible to pairwise
   ratios (null space); 3 IRLS passes with Huber δ = 0.08 so evolving features do not drive the
   fit; fallback ladder quadratic → linear → constant if any coefficient exceeds 0.35 or the
   post-fit RMS does not improve. Off until validated on the samples; then `linear` becomes the default.

### 5.9 Seams & blending (`blend.py`)
- `seam_labels`: `D_i = cv2.distanceTransform(valid_i, DIST_L2, 5)`; `label(p) = argmax_i D_i(p)`
  (ties → lower index). Seams fall mid-overlap: ≈ 565 px from the horizontal tile edges and
  ≈ 130 px from the vertical ones on the sample data.
- `multiband_blend`: levels `L = clamp(floor(log2(min_overlap_dim / 4)), 3, 6)` (sample: 6, blend
  radius ≈ 64 px at the coarsest level, inside the ≈ 130 px half-overlap). Gaussian pyramid of each
  tile's hard label mask; Laplacian pyramid of each equalized tile. Before building pyramids, fill
  each tile's invalid canvas area with a feather composite of all tiles so coarse levels never blur
  zeros or sky into the seam zone. Per level `Σ_i w_i·L_i / max(Σ_i w_i, 1e-6)`; accumulate tile by
  tile (peak ≈ 3 pyramids in memory); collapse; crop padding; clip to `[0, native_max]`.
- `feather_blend`: weights `min(D_i, 64)/64` normalized; `hard_blend`: label copy. Both are
  diagnostic escape hatches for halo artifacts.
- `seam_overlay`: seam boundaries (`morphologyEx(label, GRADIENT)`) drawn in per-tile colours on
  the preview (phase 3 UI toggle).

### 5.10 Output (`io.py`)
- `write_tiff`: round and cast to the native dtype (uint16 for the samples),
  `tifffile.imwrite(..., photometric="minisblack", compression=None, software="SunMosaic <ver>",
  description=json.dumps(meta))`; meta = version, timestamp, source file names and shapes,
  per-tile origins, gains, offsets, pedestal, canvas size, Sun centre/radius from a limb refit on
  the finished mosaic, params, warnings. Works with a path or a `BytesIO` for the download button.
- `make_preview`: 8-bit PNG, linear stretch between the 0.1th and 99.9th percentile with an
  optional gamma (display only), downscaled with `INTER_AREA` so the long side ≤ `preview_max_px`.

### 5.11 `pipeline.stitch(tiles, params, progress_cb)`
Runs 5.2 → 5.10 in order (Loading → Limb fit → Registration → Photometry → Blending → Encoding),
calls `progress_cb` between steps, records timings, collects warnings, refits the limb on the
finished mosaic as a sanity check (R ≈ 797 expected on the samples), returns `MosaicResult`.
Expected runtime on the samples: a few seconds.

## 6. UI (`src/sunmosaic/app.py`, Streamlit)

Layout (single page, sidebar + main):

- **Sidebar — Input**
  - `st.file_uploader("Sun tiles (TIFF)", type=["tif", "tiff"], accept_multiple_files=True)`;
    message if fewer than 2 or more than 5 files.
  - Expander "…or pick from a folder": `st.text_input` folder path (default
    `~/Documents/Astronomy/SunMosaic`) + `st.multiselect` over the folder's `.tif`/`.tiff` files
    (sorted, defaults = first 4, max 5).
  - Expander "Options": blend mode (multiband / feather / hard seam), equalization (off / gain /
    gain+offset; phase 3 adds linear / quadratic), sub-pixel placement (cubic / lanczos / linear /
    integer), preview gamma slider; "Advanced": min response, crop margin, pyramid levels.
- **Main**
  - Row of tile thumbnails (`st.columns`, ≤ 5) with name, size, dtype, min/max/mean.
  - `st.button("Create mosaic", type="primary")` → `with st.status("Building mosaic…")` updated by
    the progress callback; errors surface as `st.error(str(e))` with the actionable message.
  - Result: `st.image(preview, width="stretch", output_format="PNG")` (the `use_container_width`
    parameter is deprecated in current Streamlit; the built-in fullscreen button gives a closer look).
  - Metrics row: canvas size, Sun radius, worst pair residual, seconds.
  - Expander "Diagnostics": tiles table (name, size, limb arc, R, trusted, origin x/y, gain, offset,
    source) and pairs table (pair, method, overlap w×h, response, correction, residual px, accepted);
    `st.warning` per pipeline warning.
  - **Save**: `st.download_button("Download TIFF (16-bit)", data=tiff_bytes,
    file_name="sun_mosaic_<first tile date>.tif", mime="image/tiff")`,
    `st.download_button("Download PNG preview")`, and an expander "Save to folder" with a path
    input, an "overwrite" checkbox and a button that writes server-side and shows the written path.
- **State/caching**: decoded tiles cached with `@st.cache_data` keyed on the file bytes; the
  `MosaicResult` and the encoded TIFF bytes live in `st.session_state` keyed on a SHA-256 of the
  input bytes + params, computed only when the button is pressed. Gamma changes re-render the
  preview from the stored result without recomputing (Streamlit reruns the whole script on every
  widget interaction, including download clicks).
- **Launch**: `uv run sunmosaic-ui` (`launch.py` runs `streamlit run <path to app.py>`) or
  `uv run streamlit run src/sunmosaic/app.py`; the browser opens at `http://localhost:8501`.
  `.streamlit/config.toml`: `[server] maxUploadSize = 500`, `[browser] gatherUsageStats = false`.

## 7. Project setup

```toml
# pyproject.toml
[project]
name = "sunmosaic"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["numpy>=2.0", "opencv-python-headless>=4.10,<6", "tifffile>=2024.1", "imagecodecs", "streamlit>=1.40"]

[project.scripts]
sunmosaic = "sunmosaic.cli:main"
sunmosaic-ui = "sunmosaic.launch:main"

[dependency-groups]
dev = ["pytest>=8", "ruff"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

```
uv sync
uv run pytest
uv run sunmosaic-ui
uv run sunmosaic build ~/Documents/Astronomy/SunMosaic/*.tif -o mosaic.tif --preview mosaic.png --report report.json
```

**Environment gate (first task of Phase 0):** after `uv sync`, smoke-test
`import cv2, streamlit, tifffile, imagecodecs` and assert `cv2.phaseCorrelate`,
`cv2.createHanningWindow`, `cv2.distanceTransform`, `cv2.pyrDown`, `cv2.pyrUp`, `cv2.warpAffine`
exist in the OpenCV 5.0 wheel. If anything fails: `uv python pin 3.12 && uv sync`, and pin
`opencv-python-headless<5` if needed.

## 8. Implementation phases

**Phase 0 — Scaffold**
1. Copy this plan to `plan.md` in the project root.
2. `pyproject.toml`, `src/sunmosaic/` skeleton, `.streamlit/config.toml`, `README.md`, `uv sync`,
   environment gate.
3. `types.py`, `errors.py`, `io.py` (read / write / preview) with tests (8-bit, RGB, multi-page,
   float inputs; TIFF round trip of dtype, shape and metadata JSON).
4. `synthetic.py`: synthetic Sun rendered at 4× (limb darkening, multi-scale texture, faint
   prominences, sky ≈ 550 + noise), tiles cut at integer 4× offsets then `INTER_AREA`-downsampled
   for exact sub-pixel ground truth, per-tile gain, gradient and sky offset, uint16 quantization.

**Phase 1 — Core pipeline + CLI (headless)**
5. `preprocess.py`, `limb.py` with tests.
6. `register.py`: sign-convention test first, then `coarse_place`, `refine_pair`, `solve_global`,
   MVP limb-less fallback.
7. `warp.py`; `photometric.py` (gains + sky offsets).
8. `blend.py`: `seam_labels`, `feather_blend`, `hard_blend`, then `multiband_blend`.
9. `pipeline.py`, `cli.py`; `test_pipeline_synthetic.py`; `test_samples.py` regression (§9).
   Deliverable: `uv run sunmosaic build` produces `mosaic.tif` from the 4 samples.

**Phase 2 — Streamlit UI (MVP complete)**
10. `app.py`, `launch.py` per §6; `test_app.py` AppTest smoke test. Current Streamlit exposes
    `file_uploader` in `AppTest` but does not document setting files on it, so the smoke test drives
    the folder picker with the sample directory (skipped when absent); use the uploader instead only
    if the installed version supports setting files programmatically.
11. README usage section with a screenshot.
    Exit criterion: seamless-looking mosaic of the 4 samples from the browser, downloadable 16-bit TIFF.

**Phase 3 — Quality and robustness (each behind a `Params` option, validated on the samples)**
12. Gain field (linear / quadratic, ridge + IRLS, fallback ladder) — the biggest visible quality gain.
13. Template-matching placement for limb-less tiles; 5-tile synthetic test with a central tile.
14. ECC Euclidean refinement (`cv2.findTransformECC`, `MOTION_EUCLIDEAN`) on the overlap crops when
    the loop residual exceeds 1.5 px.
15. UI niceties: seam overlay toggle, 100 % crop inspector, "prefer tile" seam priority, demo mode
    on synthetic data, compressed-TIFF option, CLI batch mode.

## 9. Verification

**Unit tests (synthetic, deterministic seeds, always run)**
- Sign convention: `np.roll` a textured image by (+7, −3) → the code's displacement equals that
  vector; a sub-pixel variant (+7.25, −3.5) from the 4× renderer within 0.1 px.
- Limb fit: synthetic arcs of 120°, 90°, 60° → centre and R error ≤ 0.5 px, trusted; 45° → untrusted;
  a central tile → `None`.
- Pairwise refinement: synthetic pairs with 25 % overlap and 8 % gain difference → error < 0.3 px;
  non-overlapping crops are rejected (response < 0.3).
- Global solve: 4 tiles / 6 pairs with σ = 0.2 px noise → max placement error < 0.3 px; a
  disconnected limb-less tile → `RegistrationError`; a tile with only the limb prior stays at its
  limb placement.
- Photometric: constant gains recovered within 1 %, sky offsets within 5 ADU; (phase 3) polynomial
  coefficients within 0.02 log units and a null-space test where all tiles share one smooth field.
- Blend: idempotence (two identical overlapping tiles reproduce the input within 1 ADU RMS); two
  constant tiles (0.4 and 0.6 of range) stay within that range, monotonic transition, no NaN; no
  leakage beyond validity masks; uncovered corners equal the sky pedestal.
- Warp: integer mode is a pure copy; cubic mode round-trips a ±0.5 px shift within 0.05 px.
- Limb overshoot: warp a synthetic limb (disk ≈ 30 000 ADU next to sky ≈ 550 ADU) by 0.5 px with
  each interpolation mode and assert no pixel outside the disk drops more than 20 ADU below the sky
  level and no pixel rises above the disk maximum. Run the same check on the real tiles; `cubic`
  stays the default only if it passes, otherwise the default becomes `linear`.

**Regression on the real sample tiles (`test_samples.py`, skipped if the folder is absent)**

| Check | Expected |
|---|---|
| Limb fit, all 4 tiles | trusted; R = 797.2 ± 1.0; arc ≥ 125° |
| Sun centre in tile coords (quadrant) | 2038_0 (572.8, 944.8) → UR; 2038_5 (575.7, 172.4) → LR; 2039_0 (1008.7, 144.7) → LL; 2039_5 (1035.2, 973.8) → UL; each ± 2 px |
| Pairs accepted | all 6, method crop-pc, responses ≥ 0.6 |
| Final relative placements | within 2 px of the limb-based values (differences of the centres above); loop residual ≤ 1.5 px |
| Gains / offsets | gains in [0.90, 1.10]; offsets within ± 80 ADU |
| Canvas and output | 2062 × 1917 ± 4 px (axis order pinned on the first run); dtype uint16; limb refit on the mosaic R = 797 ± 2 |
| Runtime | < 30 s on this machine |

**UI check (manual, and `test_app.py` headless)**
1. `uv run sunmosaic-ui`, pick the 4 sample files (uploader and folder picker), click Create mosaic.
2. Inspect at 100 %: continuous limb, no visible seams or brightness steps (especially the two
   horizontal seams inside the ≈ 260 px vertical overlaps and around prominences), chromospheric
   texture not blurred in overlap zones, sky uniform under a hard stretch.
3. Download the TIFF; verify with `tifffile` that it is 16-bit and 2062 × 1917; open it in a viewer.
4. Error paths: 1 file, 6 files, an unrelated image (no overlap) → clear messages, no traceback.
5. Optional: drive the same flow with the browser automation tools for a recorded smoke test.

## 10. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Python 3.14 wheels (OpenCV 5.0, Streamlit 1.63) do not import | Environment gate in Phase 0; fallback to 3.12 and OpenCV < 5. |
| Phase-correlation sign or crop-rounding bugs (silent 1–8 px errors) | Sign test written first; explicit `frac` term; residuals reported; real-data regression values. |
| Only ≈ 25 % overlap on one axis | Overlap-crop correlation instead of full-frame; `min_overlap_px` with a clear error below it; pyramid level cap keeps blending inside the overlap. |
| Tile with short or no limb (central 5th tile, tight framing) | Trusted/untrusted/none limb states; full-frame phase-correlation fallback, then template matching (phase 3); weak limb prior keeps the graph connected; clear "could not place tile" error. |
| Rotation between tiles (alt-az field rotation over minutes) | Loop-closure residual warning; ECC Euclidean refinement in phase 3. |
| Etalon sweet-spot gradients ⇒ visible seams or bands | Gain + sky offset + multi-band blend (MVP); ridge-regularized gain field with fallback ladder (phase 3); diagnostics show pre/post RMS. |
| Chromosphere/prominence evolution and seeing (blur) mismatch between tiles | Seams mid-overlap; hard per-pixel choice at fine scales (no ghosting); Huber weights and smoothing in the photometric fit; "prefer tile" priority (phase 3). |
| Dark halos in sky or around prominences | Sky offsets solved; pedestal-based gain; feather-composite fill before pyramids; level cap; feather/hard modes as escape hatches. |
| Near-saturated pixels (one sample tile peaks at 65 469) | Excluded from photometric statistics; final clip; warning if > 0.5 % of the disk is saturated. |
| 8-bit, RGB, float or multi-page inputs; differing tile sizes | Normalized in `io.py` with warnings; all geometry comes from per-tile shapes. |
| Sub-pixel resampling softens detail | Cubic default; `integer` mode gives zero resampling; documented in the UI. |
| Wrong file selection (non-overlapping tiles) | Response thresholds and connectivity check ⇒ `RegistrationError` naming the tile; never a silent bad mosaic. |
| Streamlit reruns recompute or lose the result | `session_state` keyed on input hash + params; `cache_data` for decoding; compute only on button press. |
| Memory for larger sensors | float32 throughout; tile-by-tile pyramid accumulation; 30 Mpx per-tile guard with a message. |

---

# As built (2026-09-10)

Phases 0 to 2 of §8 are complete and verified. What follows records where the implementation
departed from the plan above, and why, so the plan stays a truthful reference.

## Confirmed on real data

| Prediction in the plan | Measured |
|---|---|
| Canvas ≈ 2062 × 1917 | 2062 × 1916 |
| All 6 pairs matched by overlap-crop correlation | all 6, responses 0.73–0.88 |
| Loop-closure residual within 1.5 px | 0.21–0.37 px, so no rotation correction is needed |
| Solar radius ≈ 797 px | 797.0, refitted on the finished mosaic |
| Gains within 0.90–1.10 | 0.954–1.045 |
| Pyramid depth 6 for a 258 px overlap | 6 |
| Runtime | 0.7 s for four frames |

Sky offsets came out at −98 to +86 ADU, wider than the ±80 the plan guessed; the regression test
allows ±150. A hard stretch of the sky shows no seam and no halo, and the hard-seam mode shows
the quadrant steps the etalon gradient predicts, which is the evidence that the multi-band blend
is doing real work.

## Changes made during implementation

1. **Frames with no sky.** Otsu needs two populations. A frame lying wholly inside the disk has
   none, and thresholding carved its texture into a meaningless "disk", which then produced a
   bogus limb fit and broke the brightness comparison. `preprocess.has_sky` now detects this and
   reports the frame as disk everywhere. This is what routes a central fifth frame correctly.
2. **Implausible limb fits are refused.** `limb.fit_limb` returns `None` when the residual
   exceeds `max(3 px, 5 % of r)` or the radius is under 15 % of the frame, so texture can never
   masquerade as a limb.
3. **Trust is judged relative to the radius.** A 3 px scatter around an 800 px radius is a good
   limb; the plan's absolute 1.5 px threshold rejected sound fits on rougher data. The criterion
   is now `resid <= max(1.5 px, 0.5 % of r)`.
4. **Warped frames are clamped to their own range.** Cubic and Lanczos overshoot at a step edge.
   Measured on a synthetic hard limb, cubic undershot to −830 ADU against a sky of 550, which
   would clip to a black ring. On the real, soft limb the four interpolation modes differ by
   about 5 ADU and nothing clips, so cubic stays the default as the plan required, with the clamp
   removing the risk for anyone whose limb is sharper. `tests/test_warp.py` measures this at the
   edge, where ringing actually lives.
5. **Canvas corners are filled from the nearest real pixel**, not with a flat pedestal. Four
   frames in a square cover a cross, and the flat fill sat about 60 ADU above the neighbouring
   sky, which showed as grey blocks under a hard stretch. The amount is reported as a warning and
   a test asserts the disk never lands on filled canvas.
6. **The folder picker earns its keep twice.** Streamlit's `AppTest` cannot fill a file uploader,
   so the headless UI test drives the folder picker instead.
7. **Sky detection compares the darkest pixels to the brightest, not to the median.** A frame
   that is mostly sky has a sky-level median, so the original test would have called it disk
   everywhere. Verified against a frame that is 96 % sky.

## Exercised in the browser

The folder picker, the **Create mosaic** button, the rendered result, the metrics row, the
warning, and the diagnostics tables were all driven in Chrome against the four real frames. The
drag-and-drop uploader and the two download buttons were not clicked in the browser; the
uploader path is covered by `read_tile` tests and the download bytes by the TIFF round-trip
test, and save-to-folder is covered end to end by `tests/test_app.py`.

## Phase 3 (added after the first release)

All three enhancements of §8 are now implemented and tested.

### Illumination field (§5.8 step 4)

A polynomial in frame coordinates, fitted in log space with a ridge prior on the shape terms,
three Huber-weighted passes, and a fallback ladder from quadratic to linear to constant. Measured
on the sample frames with hard seams, so the blend cannot mask the difference:

| Brightness matching | Mean step across the seams, on the disk | Large-scale flatness |
|---|---|---|
| off | 1614 ADU | 7.91 % |
| constant gain | 1278 ADU | 9.15 % |
| linear field | 1022 ADU | 7.35 % |
| quadratic field | 923 ADU | 7.91 % |

`linear` improves both measures and is now the default, as §3 anticipated. `quadratic` flattens
the seams further but not the disk, so it stays an option rather than the default.

### Template matching for contained frames (§5.4)

A frame with no limb is now located by matching it as a template against a quarter-scale mosaic
of the frames already placed, falling back to whole-frame correlation when it is not contained.
Placement error for a central fifth frame, against synthetic truth:

| Centre frame size | Error |
|---|---|
| 900 × 700 | 0.07 px |
| 700 × 500 | 0.47 px |
| 1200 × 900 | 0.55 px |

### Rotation refinement (§8 item 14)

Triggered when the loop-closure residual exceeds `max_loop_resid`, or when a frame matched
nothing at all. Every overlapping pair is measured with ECC, the angles are solved globally with
the angles summing to zero, and the frames are turned into a common orientation; the pipeline
then repeats, and the result is kept only if it matched more pairs or closed the loop better.

Measured on synthetic frames carrying a known rotation:

| Rotation across the set | Without refinement | With refinement |
|---|---|---|
| 0.00° | 6 of 6 pairs, 0.20 px | unchanged, no rotation invented |
| 0.75° | 6 of 6 pairs, 0.20 px | 6 of 6 pairs, 0.13 px |
| 2.40° | 1 of 6 pairs, two frames unmatched | 6 of 6 pairs, 0.39 px |

The 2.40° row is the important one. Phase correlation fails outright at that angle, so without
this step the tool reports that the frames do not match. ECC still converges there, which is why
angles are measured on every overlapping pair rather than only the matched ones. Recovered
angles agree with the truth to better than 0.05°.

Two sign conventions are now pinned by tests rather than by reasoning: phase correlation's, and
ECC's, where the measured angle is what must be applied to undo the frame's own rotation.

On the sample frames nothing changes: no rotation is detected, the residual stays at 0.37 px, and
a run still takes about a second.

## Still not implemented

Nothing from the plan remains outstanding. Rotation beyond roughly 3° is detected and reported
but not corrected, because the frames stop overlapping enough to measure; a run that hits this
would need re-capture rather than better software.

---

# Phase 4 — Finishing steps: orientation and prominences

Two optional steps that act on the finished mosaic, after it is built and before it is saved.
Neither changes the mosaic pipeline; both live in a new module and the UI drives them live.

## Context

Once the disk is assembled, two things are routinely done by hand in an image editor:

1. **Orientation.** The camera angle is arbitrary, and a star diagonal mirrors the image.
   Imagers flip and turn the disk so solar north is up and east is on the left (to match GONG
   or SDO), then pad the result to a square with the Sun centred.
2. **Prominences.** They are 20 to 50 times fainter than the disk. The standard technique is a
   duplicate layer, brightened hard, shown only outside the disk, with the disk itself taken
   from the untouched original. A mask with a small margin protects the disk edge.

Both are non-mandatory: leaving the controls alone yields exactly today's output.

## What the sample data says (measured)

| Fact | Value | Consequence |
|---|---|---|
| Limb falloff width | 21 000 ADU at R, 12 500 at R+4, 4 300 at R+8, sky ≈ 750 | A plain ×10 boost outside a 1 px border saturates this halo into a bright ring |
| Prominence brightness in the R+5..R+60 annulus | median 1 800, 99.9th pct 13 400 ADU | Boost of about 10× lifts them to disk-like levels |
| Azimuthal median of the halo removed first, then ×10 | no ring, prominences clean | This is the method to use; a soft knee helps less |
| Rotate 30° into a 2062² square | 25 ms; 19.5 % of the square holds no data | Fill quality matters; live preview must not do the fill |
| Nearest-pixel fill of those corners | visible radial streaks under a hard stretch | Smooth the fill and match the sky noise |
| Smoothed + noise-matched fill | continuous with the sky under a hard stretch | Adopt for the finishing step |
| Fill cost at full resolution | 0.5 s | Fine on "Apply", not on every slider move |

## Interpretation (to confirm)

- **Prominence layer:** the disk, enlarged by the 1 px border, is kept from the original;
  everything outside comes from the brightened copy. Brightening the disk instead would not
  reveal prominences, so this is the reading taken.
- **Rotation centre:** the fitted solar centre, not the image centre. The disk then lands
  exactly in the middle of the square, which is what "assume the disk is centred" is after,
  and it is measured to about 0.3 px on the samples.
- **Square size:** the larger of the mosaic's width and height (2062 on the samples), Sun in
  the middle. Rotation exposes corners, which are filled. The alternative, a square on the full
  diagonal so no pixel is ever lost, would be 2814 wide and mostly fill; not recommended.
- **Limb glow removal (beyond the ask):** a plain multiply, which is what the request
  describes, saturates the limb halo into a bright ring on these frames because the real limb
  falls off over about 12 px, not 1. Subtracting the azimuthal median glow outside the disk
  before multiplying removes the ring and leaves prominences untouched, since they are not
  symmetric around the disk. Proposed as on by default, with a checkbox that turns it off for
  the literal behaviour.

## Decisions

| Decision | Choice | Why |
|---|---|---|
| Where it lives | New module `finish.py`, pure functions, plus `FinishParams`; the mosaic pipeline is untouched | Same rule as the rest: the UI and CLI are thin, everything is testable headless |
| Flip | A mirror left-to-right toggle, applied before the rotation; lossless | With the rotation slider it reaches every one of the eight orientations a camera can deliver; a vertical flip is this plus 180° |
| Rotation | About the fitted Sun centre, counter-clockwise positive, cubic interpolation clamped to the source range; exact `rot90` for multiples of 90° with the centre rounded to a whole pixel (so up to 0.5 px off-centre) | Matches astronomical convention; multiples of 90° stay lossless |
| Square | Side = max(width, height), Sun at the centre | Keeps the whole disk plus prominences with modest fill |
| Fill for exposed areas | Extend nearest real pixels, smooth inside the fill (σ 25 px), add noise at 0.6× the measured sky MAD; seeded | Measured: no streaks, matches the sky. Used only by the finishing step: it re-fills everything outside the real coverage, corners included, so the finished image is consistent, while the built mosaic stays exactly as it is today |
| Prominence mask | Circle of radius R + 1 px fully protected; a feather extends outward over 4 px | Disk is never cut, as specified; the feather stops a hard edge at the composite boundary |
| Boost | Multiply the outer layer by a slider value 1..40, after subtracting the azimuthal median glow (on by default) and adding back the far-field sky; clip at 65535 | Measured to be the only variant without a limb ring |
| Output | Linear mosaic stays the primary file (rotated and squared if applied); the boosted composite is written as a second file `<name>_prominences.tif` | The composite is non-linear and must not masquerade as data |
| Live preview | Slider moves act on a downsampled **linear float32** copy of the working image, then stretch; the stretch limits are fixed from the disk of the unboosted image so the disk does not dim as the boost rises. "Apply" does the full-resolution rotation and fill; downloads are prepared once per parameter set and cached | Slider stays instant, the preview is truthful, full-res work happens once |

## Design

### `finish.py`

```python
@dataclass
class FinishParams:
    flip: bool = False               # mirror left-to-right, applied before the rotation
    rotation_deg: float = 0.0        # counter-clockwise
    square: bool = False             # the UI's Apply button turns this on
    prominence_boost: float = 1.0    # 1 = off
    prominence_border_px: float = 1.0
    prominence_feather_px: float = 4.0
    remove_glow: bool = True

@dataclass
class FinishedImage:
    image: np.ndarray                # uint16, linear, rotated/squared if asked
    coverage: np.ndarray             # bool, True where real data
    sun_center: tuple[float, float]
    sun_radius: float
    prominences: np.ndarray | None   # uint16 composite, only when boost > 1
    filled_fraction: float
    params: FinishParams

def flip_horizontal(image, coverage, sun_center) -> (image, coverage, center)
def rotate_and_square(image, coverage, sun_center, angle_deg, square=True) -> (image, coverage, center)
def fill_background(image, coverage, sky_mad, seed=0) -> image
def glow_profile(image, coverage, center, radius, ring_px=2) -> (profile, radii)
def prominence_composite(image, center, radius, boost, border_px, feather_px, remove_glow, coverage, sky_level) -> image
def finish(result: MosaicResult, params: FinishParams) -> FinishedImage
```

- `flip_horizontal`: `np.fliplr` on the image and the coverage, and the Sun centre's x
  becomes `width - 1 - x`. Lossless. Always applied before the rotation, so the two compose
  in one fixed order and the metadata describes the result unambiguously.
- `rotate_and_square`: build one affine that rotates about the Sun centre and moves it to
  `((S-1)/2, (S-1)/2)`; `warpAffine` with `INTER_CUBIC`, then clamp to the source min/max
  (same policy as `warp.py`); coverage warped with nearest neighbour and eroded 2 px. If the
  angle is a multiple of 90° use `np.rot90` and slicing so nothing is resampled; the pivot is
  the Sun centre rounded to a whole pixel, so the disk can sit up to 0.5 px off the square's
  centre, which is the price of a lossless turn. `square=False` keeps the original canvas size
  (rotation only).
- `fill_background`: nearest-pixel extension (the existing `distanceTransformWithLabels`
  trick), Gaussian-smoothed with σ = 25 px, noise from a seeded generator at 0.6 × sky MAD,
  written only where coverage is False. The built mosaic keeps its present corner fill; this
  function is applied by `finish` alone, and since it refills everything outside the real
  coverage, the old corners and the newly exposed areas end up looking the same.
- `glow_profile`: azimuthal median in 2 px rings outside R over covered pixels, lightly
  smoothed along radius, extended flat beyond the last ring.
- `prominence_composite`: `mask = clip((R + border - r) / feather + 1, 0, 1)` so it is 1 for
  all r ≤ R + border and reaches 0 at R + border + feather; outer = image − glow + sky when
  `remove_glow` else image; outer ×= boost; clip 0..65535; output = mask·image + (1−mask)·outer.
- `finish`: apply the flip if asked, then the rotation and squaring if either is asked, then
  (whenever any geometry changed)
  fill, then the composite if boost > 1. The Sun centre after rotation is the square centre by
  construction; the radius is unchanged. If the mosaic's limb refit failed (radius 0) the
  image centre is used and a warning is added.

### Metadata and files (`io.py`)

The TIFF description gains a `finish` block: flip, rotation_deg, square_side,
filled_fraction and fill method, prominence boost, border, feather, glow removal. A new `write_image(array,
metadata, dest)` carries it; `write_tiff(result, dest)` keeps its signature and callers.

### CLI (`cli.py`)

`--flip`, `--rotate DEG`, `--square`, `--prominence-boost X`, `--prominence-border PX`,
`--prominence-feather PX`, `--keep-glow`. `-o out.tif` is the linear image; with a boost above
1 a second file `out_prominences.tif` is written and reported.

### UI (`app.py`)

A "Finish" section between the metrics row and the download buttons, two expanders:

- **Orientation**: a **Flip** toggle (mirror left-to-right) and a slider "Rotate (degrees,
  counter-clockwise)" −180..180, step 0.1. The main preview follows both live (rotation of the ≤1600 px linear copy, no fill, so it is
  instant). Button **Apply rotation and make square**: the total angle so far plus the slider
  is applied to the **built mosaic** in one resampling (never to an already-rotated image, so
  two Applies cost one interpolation), then fill and centring; the slider resets to 0 and this
  becomes the working image. The flip is part of the same Apply, always before the turn.
  Button **Undo** clears the flip and sets the total angle back to 0. A caption reports the
  flip, the total angle, the square size and how much of it is fill.
- **Prominences**: slider "Boost" 1..40, step 0.5, default 1 (off), live on the main preview;
  checkbox "Remove limb glow" (on); inside "Advanced": border (default 1 px) and feather
  (default 4 px).
- The main preview shows the working image with the live rotation and boost applied. It is
  computed on a downsampled linear copy kept in session state, not on the 8-bit preview, and
  the display stretch is locked to the unboosted disk so the boost changes only the outside.
- Downloads: with the boost at 1 there is one button, **Download mosaic (16-bit TIFF)**.
  With a boost above 1 the primary button becomes **Download as shown (16-bit TIFF)** and a
  second one, **Download linear mosaic, without prominences**, serves the untouched data. A
  button labelled "mosaic" never silently drops the prominences the user just tuned. "Save
  to folder" writes the same one or two files. The metrics row adds the square size and the
  applied angle.
- Session state: `applied_flip`, `applied_angle` (cumulative) and `working` (image, coverage, centre,
  radius, and its linear preview copy) alongside the build result; slider reset via
  `st.session_state` before the widget is drawn, then `st.rerun()`.

## Implementation steps

1. `finish.py` with `FinishParams`, `FinishedImage` and the five functions. The pipeline and
   `blend.fill_uncovered` are not touched.
2. `tests/test_finish.py` (below).
3. `io.py` metadata block and the two-file write; `cli.py` flags; CLI run on the samples.
4. `app.py` Finish section; `tests/test_app.py` additions.
5. README section; `plan.md` as-built notes; browser check on the real frames.

## Verification

- **Flip**: flipping twice returns the input bit for bit; a synthetic image with a mark on
  the left has it on the right afterwards, at the mirrored column; the Sun centre's x is
  mirrored; flip then rotate 90° differs from rotate then flip, and the code applies them in
  the documented order.
- **Rotation**: rotating a square synthetic image with the disk on an exact pixel centre by
  90° equals `np.rot90` bit for bit; rotating by θ then
  −θ reproduces the disk interior within 1 % RMS; after any angle the limb refit puts the
  centre at the square centre within 0.5 px and R within 0.5 px; output side equals
  max(w, h); angle 0 with `square=False` returns the input untouched.
- **Fill**: covered pixels are never modified; the filled region's mean is within one sky MAD
  of the adjacent real sky; its gradient energy is below that of the plain nearest-pixel fill
  (no streaks); the result is identical across two runs (seeded).
- **Prominences**: inside R + border the output equals the input bit for bit; a synthetic
  prominence blob outside the feather is brighter by the boost within 2 %; with glow removal
  the azimuthal median outside the mask is flat within the noise; boost 1 with glow removal
  off returns the input; nothing exceeds 65535.
- **Real frames**: rotate 30° → R = 797 ± 2 and centre at the square centre ± 1 px; boost 10
  → the disk median is unchanged and the annulus R+10..R+40 median rises by 8 to 12×; and a
  limb profile check that a glance at the screen would miss: the azimuthal median of the
  composite from R to R+30 has no dip below the boosted far-field sky (no dark rim at the
  1 px border) and no value above the disk median between R+1 and R+8 (no bright ring).
- **App**: toggle Flip and set the rotation slider, click Apply, assert the square metric,
  the flip caption and the slider back at 0; set boost 10, assert the second download button; save to a folder writes two files.
- Browser check on the four real frames, including a flip and a 90° turn (both lossless).

## Risks

| Risk | Mitigation |
|---|---|
| Large angles expose up to a fifth of the square with no data | Smooth, noise-matched fill; fraction reported in the UI and metadata |
| The boosted image is non-linear and could be mistaken for data | Written as a separate file with a suffix, never as the mosaic |
| Bright ring at the limb after boosting | Glow removal plus a 4 px feather; border stays 1 px so the disk is never cut |
| Sun centre wrong ⇒ off-centre square and misplaced mask | Centre comes from the limb refit on the finished mosaic (0.3 px on samples); a failed refit falls back to the image centre with a warning |
| Slider reruns recomputing full-resolution work | Live operations act on the preview only; full-res on Apply and on download, cached by parameter key |
| Resampling softens detail | Multiples of 90° are lossless; other angles use the same clamped cubic as placement |

## Phase 4 as built (2026-09-12)

Both finishing steps, including the Flip button, are implemented in `src/sunmosaic/finish.py`,
wired into the browser UI and the CLI, and tested. The mosaic pipeline was not touched.

### Confirmed on the sample frames

| Check | Result |
|---|---|
| Controls left alone | the saved file is identical to the built mosaic |
| Quarter turn | bit-exact on every real pixel; 3 258 edge pixels, all at least 1 032 px from the Sun centre, fall outside the 2 062 px square |
| Flip, then 30°, squared | 2 062 × 2 062 with the Sun centred to under a pixel; 19.4 % of the square filled; 346 ms |
| Boost ×10, disk | the disk plus 1 px is bit-exact |
| Boost ×10, limb | no dark rim (lowest ring 7 490 ADU, the boosted sky is 7 490); no bright ring (at most 18 657 ADU in R+1..R+8 against a disk median of 40 148) |
| Bright pixels left beyond the limb | 686 in the 10 px beyond the limb; none within 2 px of the measured edge |
| Sky around the disk, boost ×10 | every 30° sector within 1.6 % of the boosted sky; prominences keep 92 % of their brightness above it |
| Live preview | Boost slider 133 ms once the limb is measured; rotate with boost 400 ms; full-resolution boost 523 ms |

The remaining bright pixels are spicules and prominence fringe, not artefacts. They are
identical at 2°, 1° and 0.5° sectors, sit beyond the steep part of the limb rather than on it,
and are visible in the raw data under a hard stretch.

### Changes made during implementation

1. **The glow and the mask follow the measured limb, not the fitted circle.** The first
   version tied the glow model to the circle and produced a jagged bright ring: 3 360 bright
   pixels over 184° of limb. Finer, interpolated rings alone barely helped. The cause is that
   the real limb wanders from −1.0 to +1.8 px around the fitted radius, where the falloff drops
   about 2 000 ADU per pixel. The edge is now found in 2° sectors as the steepest fall through
   the midpoint between disk and sky, smoothed around the circumference so a prominence cannot
   drag it, and both the glow distance and the protective mask follow it. The mask never
   protects less than the fitted circle plus the border.
2. **The edge finder never invents an edge.** On a small test disk, empty radial cells hid the
   crossing, and the search then reported the innermost cell, 6 px inside the disk, as the
   edge. Empty cells are now filled along the radius, sectors with no usable crossing are
   filled from their neighbours, and sectors widen on a small disk so each spans at least
   10 px of limb. The sample frames keep 2°.
3. **The sky level never includes the disk.** On an image that does not reach 200 px beyond the
   limb, the sky estimate fell back to the whole-image median and returned 2 387 ADU for a sky
   of 550. It now uses the outermost fifth of the visible sky.
4. **What is left after removing the glow is floored at zero**, so nothing can end up darker
   than the boosted sky and there is no dark rim.
5. **The Finish controls sit beside the preview**, not between the metrics and the downloads as
   planned, so the image stays in view while the Rotate and Boost sliders move.
6. **The full-resolution files are generated only when a download button is clicked.**
   Streamlit accepts a callable for the data, so dragging Boost never builds a 16-bit TIFF.
7. **Widgets the tests touch have stable keys.** The new checkboxes shifted positional indexes
   and made an older save test tick Flip instead of the overwrite box.
8. **The disk guarantee is tested as stated:** the disk itself stays bit for bit even where
   it bulges past the fitted circle. The 1 px border is the margin that absorbs the error in
   measuring the edge.
9. **Areas with no data look like sky in the boosted file too.** Boosting the linear fill
   multiplied its small departures from the sky into lighter corners, about a quarter brighter
   on a square turned by 30°. A flat fill drawn from the sky's noise distribution fixed the
   median but still left an outline at display scale, because the real boosted sky is floored
   almost flat in some places and noisy in others. Each empty pixel now copies a real boosted
   pixel picked at random within 32 px along and into the nearest boundary, which keeps the
   local tone and noise by construction; random picks cannot streak. At display scale the step
   across the boundary fell from +463 ADU to −4. The linear file is not affected.
10. **The soft edge never falls below the boosted sky.** From a boost of about 13 upward, the
    original limb falloff inside the 4 px soft edge was dimmer than the boosted sky; at ×25 a
    dark band covered 325° of the limb. Within the soft edge the original is now lifted to at
    least the boosted sky level. The protected disk itself is untouched.
11. **The glow is measured per 15° sector, not per ring.** Scattered light was far from
    symmetric: 30 to 50 px beyond the limb the sky ranged from 1 226 to 1 962 ADU between 30°
    sectors, which ×10 turned into broad bright arcs, and the band at the limb read as dark
    dashes wherever the sky beside it was bright. Per-sector medians, interpolated in angle and
    distance, bring every 30° sector within 1.6 % of the boosted sky. A 15° sector is still
    several prominence widths wide, and prominences keep 92 % of their brightness above the sky.
12. **The boosted file keeps its layers** (requested after delivery, for editing in Affinity
    Photo). `_prominences.tif` is a layered TIFF: the flat composite is stored as the normal
    image, and Adobe's ImageSourceData tag (#37724, written with `psdtags`, 16-bit `Lr16`,
    ZIP with prediction) holds two layers. Bottom: the boosted copy. Top: the linear mosaic,
    bit for bit, with a 16-bit layer mask that is 65535 on the disk plus the border. Affinity
    Photo, Photoshop and Krita read the layers; PixInsight and other plain readers open the
    flat image. To keep the top layer an exact copy, the soft-edge lift of change 10 moved
    from the pixels into the mask: where the limb is dimmer than the boosted sky, the mask
    lets through exactly enough more of the boosted copy to give the lifted result (at most
    0.5 ADU from the old composite). The flat image is rebuilt from the 16-bit layers and
    mask, so recomposing the layers reproduces it exactly. On the sample frames the file
    grows from 8.5 MB to 15 to 17 MB, and the linear file stays flat.

### Notes for use

- Squaring is off in the API and CLI unless asked (`--square`); the UI's Apply button always
  squares.
- The boosted image is non-linear. It is written as a second file with a `_prominences`
  suffix and its metadata says `"linear": false`. That file carries the two layers and the
  disk mask; its metadata lists them under `finish.layers`.
- Glow removal takes out scattered light that varies smoothly around the disk. Anything
  narrower than about 15° of limb is kept as signal, which is why prominences and spicules
  survive the boost.

## Mac application, as built (2026-09-12)

Requested after phase 4: package SunMosaic as a standalone `.dmg`. The full plan and its
results are in `dmg-plan.md`.

- **Build.** `uv run python packaging/build_app.py` writes `dist/SunMosaic.app` and
  `dist/SunMosaic-0.1.0-arm64.dmg` in about a minute.
- **Bundle.** The app carries a copy of the relocatable uv CPython 3.13 with the locked
  dependencies and the project installed into it. No PyInstaller, because Streamlit needs its
  frontend files, metadata and lazy imports intact.
- **Launcher.** `src/sunmosaic/desktop.py`, also available as `sunmosaic-app`, starts the
  Streamlit server as a child process on a free local port, then shows it in a pywebview window
  with downloads enabled. pywebview is an optional `desktop` extra, so tests and
  `sunmosaic-ui` do not need it.
- **Server lifetime.** The server watches its parent process and exits once the parent is
  gone. Quit from the menu or the Dock ends the window process without running Python's exit
  handlers, which left the server running; the watch covers that, Force Quit and crashes.
- **Signing.** Ad hoc, enough for this Mac. Bytecode is precompiled and writing it at runtime
  is disabled, so a run never changes the signed bundle.

| Check | Result |
| --- | --- |
| Launch to health `ok` | 1 to 2 s |
| Dock identity | SunMosaic, bundle ID `local.sunmosaic.app` |
| Pipeline inside the bundle | demo frames built 2062 x 1916; saved layered file had both layers, top equal to the linear file |
| Copy taken out of the DMG | launches, signature valid |
| Server after Quit / after kill -9 | gone within 0 s / 1 s |
| Size | app 562 MB, DMG 228 MB |
