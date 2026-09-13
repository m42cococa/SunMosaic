# CLAUDE.md

SunMosaic joins 2 to 5 overlapping partial H-alpha TIFF frames of the Sun into one full-disk
16-bit mosaic. It has three front ends over one pipeline: a Streamlit browser UI, the
`sunmosaic build` command line, and a standalone Apple Silicon Mac app. Python 3.13, managed
with uv.

## Commands

```
uv sync                                         # set up the environment
uv run pytest                                   # full suite, about 90 s
SUNMOSAIC_SAMPLES="$PWD/demo" uv run pytest     # include the real-frame tests, using demo/
uv run ruff check src tests packaging           # lint; must stay clean
uv run sunmosaic-ui                             # browser UI on http://localhost:8501
uv run sunmosaic build demo/*.tif -o out.tif --preview out.png
uv run --extra desktop sunmosaic-app            # the app window, without building the bundle
uv run python packaging/build_app.py            # dist/SunMosaic.app and the .dmg, about 1 min
```

## Reference documents

- `plan.md` is the design reference: measured properties of the input, algorithms and
  parameters, verification criteria, and an as-built section for every phase. Read the
  relevant part before changing the pipeline or the finishing steps.
- `dmg-plan.md` holds the Mac app packaging plan and its verification results.
- `README.md` is the user documentation. Keep it in step with CLI flags, UI labels and outputs.

## Architecture

- `pipeline.stitch()` is the single entry point used by the UI, the CLI and tests. The
  pipeline never imports Streamlit.
- Order of work: `io.read_tile`, `preprocess`, `limb` (circle fit), `register` (limb placement,
  overlap phase correlation, global least squares, template matching, rotation), `warp`,
  `photometric` (gain field), `blend` (multiband), then `io` writes the TIFF.
- `finish.py` applies optional steps to a built mosaic: flip, rotate and square, and the
  prominence layers with limb-glow removal. `FinishParams()` defaults are a no-op.
- `io.py` reads and writes TIFFs with a JSON metadata block in ImageDescription, and writes the
  layered prominence file through `psdtags`.
- `app.py` is a thin UI; `cli.py`; `launch.py` starts the browser UI; `desktop.py` runs the Mac
  app window and its server process. `types.py` holds the dataclasses; `Params` defaults are
  the values validated on the real frames. `synthetic.py` renders a Sun with known truth.

## Invariants the tests protect

- Pixels are float32 in native 16-bit ADU inside the pipeline. Saved mosaics are uint16 and
  linear; stretching is for display only. Pixel scale never changes, and sky is kept around
  the disk because it holds the prominences.
- Leaving the finishing controls alone writes exactly the mosaic as built.
- In the prominence output, the disk plus the 1 px border is bit-exact from the linear image.
- The `_prominences.tif` file stores the flat composite as the normal image, for PixInsight and
  any plain reader, and the layers in Adobe's ImageSourceData tag, which Affinity Photo reads.
  Bottom layer: the boosted copy. Top layer: the linear mosaic with a 16-bit disk mask.
  Recomposing the layers must reproduce the flat image exactly.
- The linear file stays a plain, flat TIFF.

## Tests

- Real-frame tests use the `sample_paths` fixture. It reads `~/Documents/Astronomy/SunMosaic`
  by default and skips when that folder is missing; `SUNMOSAIC_SAMPLES` points it elsewhere.
  `demo/` holds identical copies of the four frames.
- `test_app.py` drives the UI with Streamlit's AppTest. Give any widget a test touches a stable
  `key=`, because positional indexes shift when widgets are added.
- `test_desktop.py` must never import pywebview; it is an optional extra.

## Mac app

- The bundle is built by hand around the relocatable uv-managed CPython, not PyInstaller, since
  Streamlit needs its frontend files, metadata and lazy imports intact.
- The launcher is a bash script. The Streamlit server runs as a child,
  `python -m sunmosaic.desktop --serve <parent pid>`, and exits when its parent disappears,
  because Quit from the Dock ends the window process without running Python's exit handlers.
- Signing is ad hoc and must be the last write to the bundle. Bytecode is precompiled and
  `PYTHONDONTWRITEBYTECODE` is set, so a run never modifies the signed bundle. arm64 only.
- The server log is `~/Library/Logs/SunMosaic.log`.
- On this Mac, screen capture and window inspection are not permitted to automation. Verify
  the app through its server port with Playwright, and its identity with `lsappinfo`.

## Working with this user

- Image editing happens in Affinity Photo and PixInsight; there is no Photoshop license, so no
  output may depend on Photoshop and nothing can be checked in it.
- For enhancements, present the plan first. Once it is confirmed, record it in `plan.md`, or in
  the separate file the user names, before executing. Afterwards add as-built notes with the
  measured results and any deviations.
- Commit and push only when asked.

## Gotchas

- zsh aborts a command when a glob matches nothing, such as `rm -f dir/*.tif` on an empty
  folder. Name the files explicitly.
- Download buttons pass a callable as `data`, so full-resolution files are built only on click.
- `build/` and `dist/` are build outputs and are ignored by git.
