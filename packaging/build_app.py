"""Build SunMosaic.app and a .dmg holding it, for Apple Silicon Macs.

Run from the project root:

    uv run python packaging/build_app.py

The application carries its own Python: a copy of the relocatable CPython that uv installed
(python-build-standalone), with SunMosaic and its locked dependencies installed into it.  The
result lands in ``dist/``.  The bundle is signed ad hoc, which is enough to run it on this Mac;
giving it to others without a Gatekeeper warning needs a Developer ID and notarization.
"""

from __future__ import annotations

import plistlib
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
BUILD = ROOT / "build"
APP_NAME = "SunMosaic"
BUNDLE_ID = "local.sunmosaic.app"  # change to your own reverse-DNS name before sharing
PY_VERSION = f"{sys.version_info.major}.{sys.version_info.minor}"

STDLIB_PRUNE = ["test", "idlelib", "ensurepip", "tkinter", "turtledemo", "lib2to3/tests"]
LIB_PRUNE_GLOBS = ["tcl8*", "tk8*", "itcl*", "thread2*"]
TEST_DIRS_IN = ["numpy", "pandas", "pyarrow"]
# python-build-standalone's own build prefix, used in place of where uv put it on this Mac.
NEUTRAL_PREFIX = b"/install"
KEPT_EXECUTABLES = {"python", "python3", f"python{PY_VERSION}"}


def run(*command: str | Path, **kwargs) -> subprocess.CompletedProcess:
    print("+", " ".join(str(c) for c in command), flush=True)
    return subprocess.run([str(c) for c in command], check=True, **kwargs)


def version() -> str:
    text = (ROOT / "src" / "sunmosaic" / "__init__.py").read_text()
    match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', text, re.MULTILINE)
    if match is None:
        sys.exit("no __version__ in src/sunmosaic/__init__.py")
    return match.group(1)


def standalone_python() -> Path:
    """The uv-managed interpreter this build runs under, checked to be relocatable."""
    base = Path(sys.base_prefix)
    lib = base / "lib" / f"libpython{PY_VERSION}.dylib"
    binary = base / "bin" / f"python{PY_VERSION}"
    if not lib.exists() or not binary.exists():
        sys.exit(f"{base} is not a relocatable python-build-standalone install; "
                 "run this script with `uv run` in the project")
    links = subprocess.run(["otool", "-L", str(binary)], capture_output=True, text=True,
                           check=False).stdout
    if "@executable_path/../lib/libpython" not in links:
        sys.exit(f"{binary} does not load libpython relative to itself, so it cannot be moved")
    return base


def copy_python(target: Path) -> Path:
    base = standalone_python()
    shutil.copytree(base, target, symlinks=True)
    stdlib = target / "lib" / f"python{PY_VERSION}"
    (stdlib / "EXTERNALLY-MANAGED").unlink(missing_ok=True)
    for name in STDLIB_PRUNE:
        shutil.rmtree(stdlib / name, ignore_errors=True)
    for extension in (stdlib / "lib-dynload").glob("_tkinter*"):
        extension.unlink()
    for pattern in LIB_PRUNE_GLOBS:
        for path in (target / "lib").glob(pattern):
            shutil.rmtree(path, ignore_errors=True)
    shutil.rmtree(target / "share", ignore_errors=True)
    scrub_interpreter(target, base)
    return target / "bin" / f"python{PY_VERSION}"


def scrub_interpreter(root: Path, original_prefix: Path) -> None:
    """Remove where uv installed Python on this Mac from the copied interpreter.

    uv records its install location, inside the builder's home folder, in the build
    configuration files and as the shared library's install name. Python does not need either
    to run, and shipping them would publish the builder's account name.
    """
    old = str(original_prefix).encode()
    for path in root.rglob("*"):
        if path.is_symlink() or not path.is_file():
            continue
        data = path.read_bytes()
        # Only text files are edited. Binary formats store string lengths, so a shorter path
        # would corrupt them: compiled bytecode is rebuilt later anyway, and the final check
        # reports any binary that still holds the path.
        if old in data and b"\x00" not in data:
            path.write_bytes(data.replace(old, NEUTRAL_PREFIX))
    library = root / "lib" / f"libpython{PY_VERSION}.dylib"
    run("install_name_tool", "-id", f"@executable_path/../lib/{library.name}", library)
    run("codesign", "--force", "--sign", "-", library)  # the edit invalidates its signature


def install_packages(python: Path) -> None:
    BUILD.mkdir(exist_ok=True)
    requirements = BUILD / "requirements.txt"
    run("uv", "export", "--frozen", "--no-dev", "--extra", "desktop", "--no-emit-project",
        "--no-hashes", "--format", "requirements-txt", "--output-file", requirements, cwd=ROOT)
    run("uv", "pip", "install", "--python", python, "--system", "--break-system-packages",
        "--requirement", requirements, cwd=ROOT)
    run("uv", "pip", "install", "--python", python, "--system", "--break-system-packages",
        "--no-deps", "--reinstall-package", "sunmosaic", ROOT, cwd=ROOT)
    root = python.parent.parent
    site = root / "lib" / f"python{PY_VERSION}" / "site-packages"
    # Console scripts carry this Mac's bundle path in their first line, and the app never uses
    # them; the launcher runs the interpreter directly.
    for entry in python.parent.iterdir():
        if entry.name not in KEPT_EXECUTABLES:
            entry.unlink()
    # The record of where each package came from names the project folder on this Mac.
    for origin in site.glob("*.dist-info/direct_url.json"):
        origin.unlink()
    for package in TEST_DIRS_IN:
        for tests in (site / package).rglob("tests"):
            if tests.is_dir():
                shutil.rmtree(tests, ignore_errors=True)
    for cache in python.parent.parent.rglob("__pycache__"):
        shutil.rmtree(cache, ignore_errors=True)
    # Compile now, so nothing writes into the signed bundle when it runs.
    # Bytecode records the path it was compiled at; record it relative to the bundle instead.
    # Python replaces that path with the real location when it loads a module anyway.
    # -f rewrites every file, and -B stops the modules compileall itself imports from being
    # cached first with this Mac's path and then skipped as already up to date.
    subprocess.run([str(python), "-I", "-B", "-m", "compileall", "-q", "-f", "-j0", "-s", str(root),
                    "-p", f"{APP_NAME}.app/Contents/Resources/python",
                    str(root / "lib" / f"python{PY_VERSION}")], check=False)


def write_launcher(macos: Path) -> None:
    launcher = macos / APP_NAME
    launcher.write_text(f"""#!/bin/bash
# Starts SunMosaic with the Python carried inside this application.
HERE="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
unset PYTHONHOME PYTHONPATH
# -B below keeps Python from writing bytecode into the signed bundle; -I ignores this variable.
exec "$HERE/../Resources/python/bin/python{PY_VERSION}" -I -B -m sunmosaic.desktop "$@"
""")
    launcher.chmod(0o755)


def write_icon(resources: Path) -> None:
    """A red H-alpha disk with a few prominences, as an .icns file."""
    import numpy as np
    from PIL import Image

    size = 1024
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    c = (size - 1) / 2.0
    rr = np.hypot(xx - c, yy - c) / (size * 0.40)
    angle = np.arctan2(yy - c, xx - c)
    disk = np.clip(1.0 - rr, 0.0, 1.0) > 0
    limb = np.sqrt(np.clip(1.0 - rr**2, 0.0, 1.0)) ** 0.45
    texture = 0.92 + 0.08 * np.sin(xx / 23.0) * np.cos(yy / 29.0)
    prominence = np.exp(-((rr - 1.04) / 0.035) ** 2) * (
        np.exp(-((angle - 0.8) / 0.06) ** 2) + np.exp(-((angle + 2.2) / 0.05) ** 2)
        + 0.8 * np.exp(-((angle - 2.7) / 0.04) ** 2)
    )
    brightness = np.where(disk, limb * texture, 0.0) + 0.9 * prominence * (~disk)
    alpha = np.clip(np.where(disk, 1.0, 0.0) + 1.6 * prominence, 0.0, 1.0)
    rgba = np.zeros((size, size, 4), np.uint8)
    rgba[..., 0] = np.clip(255 * (0.35 + 0.65 * brightness), 0, 255)
    rgba[..., 1] = np.clip(255 * (0.45 * brightness**1.6), 0, 255)
    rgba[..., 2] = np.clip(255 * (0.12 * brightness**3), 0, 255)
    rgba[..., 3] = np.clip(255 * alpha, 0, 255)
    master = Image.fromarray(rgba, "RGBA")

    with tempfile.TemporaryDirectory() as tmp:
        iconset = Path(tmp) / f"{APP_NAME}.iconset"
        iconset.mkdir()
        for points in (16, 32, 128, 256, 512):
            master.resize((points, points), Image.LANCZOS).save(iconset / f"icon_{points}x{points}.png")
            master.resize((points * 2, points * 2), Image.LANCZOS).save(
                iconset / f"icon_{points}x{points}@2x.png")
        run("iconutil", "-c", "icns", iconset, "-o", resources / f"{APP_NAME}.icns")


def write_info_plist(contents: Path, app_version: str) -> None:
    info = {
        "CFBundleName": APP_NAME,
        "CFBundleDisplayName": APP_NAME,
        "CFBundleIdentifier": BUNDLE_ID,
        "CFBundleExecutable": APP_NAME,
        "CFBundleIconFile": f"{APP_NAME}.icns",
        "CFBundlePackageType": "APPL",
        "CFBundleShortVersionString": app_version,
        "CFBundleVersion": app_version,
        "CFBundleInfoDictionaryVersion": "6.0",
        "LSMinimumSystemVersion": "12.0",
        "LSArchitecturePriority": ["arm64"],
        "NSHighResolutionCapable": True,
        "NSHumanReadableCopyright": "SunMosaic",
    }
    with (contents / "Info.plist").open("wb") as handle:
        plistlib.dump(info, handle)


def build_app(app_version: str) -> Path:
    app = DIST / f"{APP_NAME}.app"
    shutil.rmtree(app, ignore_errors=True)
    contents = app / "Contents"
    macos, resources = contents / "MacOS", contents / "Resources"
    macos.mkdir(parents=True)
    resources.mkdir()
    python = copy_python(resources / "python")
    install_packages(python)
    write_launcher(macos)
    write_icon(resources)
    write_info_plist(contents, app_version)
    check_no_home_paths(app)
    # Signing comes last: any later write would break the seal.
    run("codesign", "--force", "--deep", "--sign", "-", app)
    run("codesign", "--verify", "--deep", "--strict", app)
    return app


def check_no_home_paths(app: Path) -> None:
    """Stop the build if any file in the bundle still contains this Mac's home folder path."""
    needle = str(Path.home()).encode()
    found = [path for path in app.rglob("*")
             if path.is_file() and not path.is_symlink() and needle in path.read_bytes()]
    if found:
        listing = "\n  ".join(str(path.relative_to(app)) for path in found[:20])
        sys.exit(f"{len(found)} files in the bundle contain {needle.decode()}:\n  {listing}")
    print(f"no file in the bundle contains {needle.decode()}")


def build_dmg(app: Path, app_version: str) -> Path:
    dmg = DIST / f"{APP_NAME}-{app_version}-arm64.dmg"
    dmg.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp) / APP_NAME
        staging.mkdir()
        run("ditto", app, staging / app.name)
        (staging / "Applications").symlink_to("/Applications")
        run("hdiutil", "create", "-volname", APP_NAME, "-srcfolder", staging, "-ov",
            "-format", "UDZO", "-imagekey", "zlib-level=9", dmg)
    return dmg


def main() -> int:
    app_version = version()
    DIST.mkdir(exist_ok=True)
    app = build_app(app_version)
    dmg = build_dmg(app, app_version)
    size = sum(f.stat().st_size for f in app.rglob("*") if f.is_file() and not f.is_symlink())
    print(f"\nbuilt {app} ({size / 1e6:.0f} MB)\nbuilt {dmg} ({dmg.stat().st_size / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
