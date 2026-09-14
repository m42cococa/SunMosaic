# SunMosaic as a standalone Mac application (.dmg)

> First execution step: save this plan as `dmg-plan.md` in the project root
> (separate from `plan.md`, at the user's request). Keep its status table current while executing.

## Context

The user asked to package SunMosaic, today a Streamlit app started with `uv run sunmosaic-ui`
from a terminal, as a standalone Mac application delivered in a `.dmg`. The intended result:
double-click SunMosaic in Applications, and the app opens in its own window with no terminal,
no uv and no Python installed separately. Closing the window stops everything.

Most of the build is already done and has produced a bundle, but nothing has been launched or
verified yet. This plan covers the remaining hardening, verification and documentation.

### Status: complete (2026-09-12)

| Piece | State |
| --- | --- |
| `dmg-plan.md` | saved |
| `src/sunmosaic/desktop.py`: window launcher, server child with parent watch | done |
| `pyproject.toml`: `desktop` extra (pywebview), `sunmosaic-app` script | done |
| `tests/test_desktop.py`: 8 tests, no window library imported | pass |
| `packaging/build_app.py`: bundle, icon, ad-hoc signing, DMG; no bytecode written at runtime | done |
| `dist/SunMosaic.app` (562 MB), `dist/SunMosaic-0.1.0-arm64.dmg` (228 MB) | built and verified |
| README "Build the Mac app", `plan.md` as-built note | done |
| Full suite and lint | 162 passed, clean |

## Approach (already chosen, kept)

- **Bundle layout.** The app is built by hand around python-build-standalone, not with PyInstaller
  or py2app. Streamlit needs its static frontend, package metadata and lazy imports intact, and
  freezing tools break those. The interpreter loads `libpython` relative to itself, so the bundle
  can move.
- **Main thread.** Streamlit runs as a child process, because Cocoa (pywebview) must own the main
  thread. Quitting the window, Cmd-Q or Quit from the Dock stops the child.
- **Launcher.** `Contents/MacOS/SunMosaic` is a bash script that runs `exec`s the bundled
  `python3.13 -I -m sunmosaic.desktop`.

## Remaining steps

1. **Save this plan** to `dmg-plan.md` in the project root.
2. **Keep the signed bundle unchanged at runtime.** Set `PYTHONDONTWRITEBYTECODE=1` in the
   launcher script (`build_app.py` `write_launcher`) and in `desktop.server_environment`.
   Add a test assertion for it, then rebuild with `uv run python packaging/build_app.py`.
3. **Launch and check identity.**
   - Run `open dist/SunMosaic.app` and read the port from the log.
   - Call `curl http://127.0.0.1:<port>/_stcore/health`.
   - Run `lsappinfo info -only name,bundleID,executablepath "SunMosaic"` to confirm the Dock
     shows SunMosaic with the bundle ID and icon, not `python3.13`.
   - *Contingency if the identity is wrong:* replace the bash launcher with a small C launcher
     compiled with clang. It calls `Py_BytesMain` against the bundled `libpython3.13.dylib`,
     with the Python home set to `Resources/python` and the library path relative to the
     launcher. It forwards arguments, defaulting to `-m sunmosaic.desktop`.
     `desktop.server_command` would then use the bundled `python3.13` path explicitly.
4. **Exercise the app inside the bundle** with Playwright on that port. This proves OpenCV,
   imagecodecs, tifffile and psdtags run from the bundle.
   - Type the full path of `demo/` into the folder box, press Create mosaic, and wait for
     "Image size".
   - Set Boost and check that "Download as shown, with layers" appears.
   - Use "Save straight to a folder" to write both files into the scratchpad. Read them back:
     the linear file must be 16-bit, and the `_prominences` file must carry two layers.
   - Check the log for tracebacks.
5. **Quit cleanly.**
   - Run `osascript -e 'quit app "SunMosaic"'`.
   - `pgrep -f "streamlit run"` must show no child from the bundle. Leave the dev server on port
     8501 alone.
   - `codesign --verify --deep --strict dist/SunMosaic.app` must still pass.
6. **Relocation from the DMG.**
   - Run `hdiutil attach -nobrowse -readonly` on the DMG, then copy the app with `ditto` to a
     scratch folder.
   - Launch the copy, check health, quit it, then `hdiutil detach`.
   - This proves no absolute build paths leaked into the bundle.
7. **Documentation.**
   - Add a README section, "Build the Mac app", covering:
     - the build command and outputs with their sizes;
     - Apple Silicon only;
     - the first-launch delay of several seconds while pandas and pyarrow load;
     - the log path and the `--browser` fallback (`open -a SunMosaic --args --browser`);
     - that ad-hoc signing runs on this Mac, but other Macs need Privacy & Security, then
       Open Anyway, and clean distribution needs a Developer ID and notarization;
     - that `BUNDLE_ID` in `build_app.py` should be changed before sharing.
   - Add a short as-built note to `plan.md`.
   - Update the status in `dmg-plan.md`.
8. **Final checks.** Run the full `uv run pytest` and ruff on src, tests and packaging.

## Critical files

- `src/sunmosaic/desktop.py`: launcher, uses `sunmosaic/app.py` unchanged.
- `packaging/build_app.py`: bundle and DMG build.
- `pyproject.toml`, `tests/test_desktop.py`, `README.md`, `plan.md`, `dmg-plan.md`.
- Reused, unchanged: `sunmosaic/app.py` (UI), `sunmosaic/launch.py` (dev launcher), uv.lock
  (pins exported by `uv export --frozen --no-dev --extra desktop`).

## Verification (end to end)

| Check | Pass condition |
| --- | --- |
| Unit tests + lint | full suite passes; ruff clean |
| Bundle signature | `codesign --verify --deep --strict` passes after build and after a run |
| Launch | health `ok` within 120 s; log shows "server ready" |
| App identity | `lsappinfo` name SunMosaic, bundle ID `local.sunmosaic.app` |
| Pipeline in bundle | demo frames build a 2062 x 1916 mosaic via Playwright; saved layered file has 2 layers |
| Quit | no orphaned `streamlit run` from the bundle |
| Relocation | copy taken from the mounted DMG launches and answers health |

**Cannot be verified from this session.** Screen capture is blocked, so the native window
itself, its upload open panel and its download save panel can't be checked. The recap will say
so and give the user two clicks to try. "Save straight to a folder" and the folder picker give
paths that never touch WKWebView.

**Known limits to report.**
- Apple Silicon only.
- About 556 MB installed, 233 MB DMG.
- Ad-hoc signed.
- A Force Quit can leave the server child running.
- The default folder stays `~/Documents/Astronomy/SunMosaic`.

## Results

| Check | Result |
| --- | --- |
| Launch | health `ok` 1 to 2 s after opening; log shows "server ready" |
| App identity | `lsappinfo`: SunMosaic, `local.sunmosaic.app`, foreground; WebKit helpers named after it, so the window loaded |
| Pipeline in bundle | Playwright on the app's port: demo frames built 2062 x 1916, radius 797 px, mismatch 0.37 px; Save straight to a folder wrote a 16-bit linear file and a layered file whose top layer equals it |
| Log | no tracebacks |
| Signature after a run | valid; nothing in the bundle changed after signing |
| Relocation | copy taken out of the mounted DMG launched, answered health, signature valid |
| Window closed | server stopped, logged "window closed, server stopped" |
| Quit from outside | server gone immediately |
| Window process killed with kill -9 | server gone within 1 s |

### Found and fixed during verification

**Quit left the server running.** Quitting through an Apple event, as Cmd-Q and the Dock do,
ended the window process without running Python's exit handlers. The Streamlit child kept
serving as an orphan. The child is now started as `sunmosaic.desktop --serve <parent pid>`, and
a watcher thread exits the process once that pid is no longer its parent. The step-3 fallback,
a C launcher, was not needed: the bash launcher already gives the right Dock identity.

### Not verified from this session

Screen capture is blocked, so the window itself, the upload open panel and the download save
panel were not seen. To check them, open SunMosaic and press **Upload** in the sidebar, then
build a mosaic and press a download button. **Save straight to a folder** and the folder picker
do not go through the window's panels and were verified.

## Published: v0.1.0 (2026-09-13)

The disk image is attached to the GitHub Release
[v0.1.0](https://github.com/m42cococa/SunMosaic/releases/tag/v0.1.0); `dist/` stays out of git.
Before publishing, the build was changed so no file in the app names the builder's account.
It had been found in 2,880 files:

- **Compiled bytecode** is now built with bundle-relative paths, forced for every file.
- **The interpreter's build configuration** has uv's install path replaced with `/install`,
  in text files only; editing binary files corrupted them.
- **The shared library's install name** now points inside the bundle.
- **Console scripts and the project's install-source record** are removed.
- **A new guard** stops the build if any file still contains the home folder path.

The launcher and the server now pass `-B`, because `-I` makes Python ignore
`PYTHONDONTWRITEBYTECODE`.
