"""Command line entry point, for headless use and for scripting."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .errors import SunMosaicError
from .finish import FinishParams, finish
from .io import (
    build_metadata,
    encode_png,
    finished_metadata,
    make_preview,
    read_tile,
    write_image,
    write_prominence_file,
    write_tiff,
)
from .pipeline import stitch
from .types import Params


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sunmosaic",
        description="Stitch partial H-alpha views of the Sun into one full-disk mosaic.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="build a mosaic from 2 to 5 TIFF files")
    build.add_argument("files", nargs="+", help="input TIFF tiles")
    build.add_argument("-o", "--output", required=True, help="output TIFF path")
    build.add_argument("--preview", help="also write an 8-bit PNG preview here")
    build.add_argument("--report", help="also write the diagnostics as JSON here")
    build.add_argument("--blend", choices=["multiband", "feather", "hard"], default="multiband")
    build.add_argument(
        "--equalize", choices=["off", "gain", "gain+offset", "linear", "quadratic"],
        default="linear",
        help="how much brightness matching to apply between frames; linear and quadratic "
             "also correct the illumination across each frame",
    )
    build.add_argument(
        "--interp", choices=["cubic", "lanczos", "linear", "integer"], default="cubic",
        help="sub-pixel placement; 'integer' does not resample at all",
    )
    build.add_argument(
        "--no-rotation", action="store_true",
        help="do not try to correct rotation between frames",
    )

    finishing = build.add_argument_group(
        "finishing", "optional steps applied to the built mosaic; all off by default"
    )
    finishing.add_argument("--flip", action="store_true",
                           help="mirror left to right (applied before --rotate)")
    finishing.add_argument("--rotate", type=float, default=0.0, metavar="DEG",
                           help="turn counter-clockwise about the solar centre, -180 to 180")
    finishing.add_argument("--square", action="store_true",
                           help="pad to a square with the Sun in the middle")
    finishing.add_argument(
        "--prominence-boost", type=float, default=1.0, metavar="X",
        help="brighten everything outside the disk by X (1 = off); also writes "
             "<output>_prominences.tif",
    )
    finishing.add_argument("--prominence-border", type=float, default=1.0, metavar="PX",
                           help="protected margin beyond the limb, in pixels (default 1)")
    finishing.add_argument("--prominence-feather", type=float, default=4.0, metavar="PX",
                           help="soft transition outside the protected circle (default 4)")
    finishing.add_argument("--keep-glow", action="store_true",
                           help="boost without removing the limb glow first (brighter ring)")
    build.add_argument("--quiet", action="store_true", help="do not print progress")
    return parser


def _prominence_path(output: Path) -> Path:
    return output.with_name(f"{output.stem}_prominences{output.suffix or '.tif'}")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not -180.0 <= args.rotate <= 180.0:
        parser.error("--rotate must be between -180 and 180 degrees")
    if args.prominence_boost < 1.0:
        parser.error("--prominence-boost must be 1 or more")

    params = Params(
        blend=args.blend, equalize=args.equalize, interp=args.interp,
        refine_rotation=not args.no_rotation,
    )
    finish_params = FinishParams(
        flip=args.flip, rotation_deg=args.rotate, square=args.square,
        prominence_boost=args.prominence_boost,
        prominence_border_px=args.prominence_border,
        prominence_feather_px=args.prominence_feather,
        remove_glow=not args.keep_glow,
    )

    def progress(step: str, fraction: float) -> None:
        if not args.quiet:
            print(f"[{fraction:4.0%}] {step}", file=sys.stderr)

    try:
        tiles = [read_tile(path) for path in args.files]
        result = stitch(tiles, params, progress)
    except SunMosaicError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    output = Path(args.output)
    written = [output]
    if finish_params.changes_geometry or finish_params.boosts:
        finished = finish(result, finish_params)
        linear = finished.image
        coverage = finished.coverage
        report = finished_metadata(result, finished, "linear")
        write_image(linear, report, output)
        if finished.prominences is not None:
            prominence_file = _prominence_path(output)
            write_prominence_file(result, finished, prominence_file)
            written.append(prominence_file)
        size = finished.side
        extra_warnings = finished.warnings
    else:
        write_tiff(result, output)
        linear = result.mosaic
        coverage = result.label_map >= 0
        report = build_metadata(result)
        size = result.canvas_size
        extra_warnings = []

    if args.preview:
        Path(args.preview).write_bytes(
            encode_png(make_preview(linear, params.preview_max_px, mask=coverage.astype("uint8") * 255))
        )
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2))

    if not args.quiet:
        width, height = size
        for path in written:
            print(f"wrote {path}: {width} x {height}, 16-bit", file=sys.stderr)
        print(f"solar radius {result.sun_radius:.1f} px, "
              f"worst alignment mismatch {result.worst_residual:.2f} px", file=sys.stderr)
        for warning in list(result.warnings) + list(extra_warnings):
            print(f"note: {warning}", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
