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
    return target / "bin" / f"python{PY_VERSION}"


def install_packages(python: Path) -> None:
    BUILD.mkdir(exist_ok=True)
    requirements = BUILD / "requirements.txt"
    run("uv", "export", "--frozen", "--no-dev", "--extra", "desktop", "--no-emit-project",
        "--no-hashes", "--format", "requirements-txt", "--output-file", requirements, cwd=ROOT)
    run("uv", "pip", "install", "--python", python, "--system", "--break-system-packages",
        "--requirement", requirements, cwd=ROOT)
    run("uv", "pip", "install", "--python", python, "--system", "--break-system-packages",
        "--no-deps", "--reinstall-package", "sunmosaic", ROOT, cwd=ROOT)
    site = python.parent.parent / "lib" / f"python{PY_VERSION}" / "site-packages"
    for package in TEST_DIRS_IN:
        for tests in (site / package).rglob("tests"):
            if tests.is_dir():
                shutil.rmtree(tests, ignore_errors=True)
    for cache in python.parent.parent.rglob("__pycache__"):
        shutil.rmtree(cache, ignore_errors=True)
    # Compile now, so nothing writes into the signed bundle when it runs.
    subprocess.run([str(python), "-I", "-m", "compileall", "-q", "-j0",
                    str(python.parent.parent / "lib" / f"python{PY_VERSION}")], check=False)


def write_launcher(macos: Path) -> None:
    launcher = macos / APP_NAME
    launcher.write_text(f"""#!/bin/bash
# Starts SunMosaic with the Python carried inside this application.
HERE="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
unset PYTHONHOME PYTHONPATH
export PYTHONDONTWRITEBYTECODE=1  # the bundle is signed; never write into it
exec "$HERE/../Resources/python/bin/python{PY_VERSION}" -I -m sunmosaic.desktop "$@"
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
    # Signing comes last: any later write would break the seal.
    run("codesign", "--force", "--deep", "--sign", "-", app)
    run("codesign", "--verify", "--deep", "--strict", app)
    return app


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
