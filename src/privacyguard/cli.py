"""Minimal CLI entry point for quick usage."""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="privacyguard",
        description="Privacy-first de-identification for video streams.",
    )
    parser.add_argument("model", help="Path to the ONNX detection model.")
    parser.add_argument(
        "-s", "--source", default="0",
        help="Video source: camera index (int) or file/RTSP URL. Default: 0",
    )
    parser.add_argument(
        "-m", "--method", default="gaussian",
        choices=["gaussian", "pixelate", "solid"],
        help="Anonymization method. Default: gaussian",
    )
    parser.add_argument(
        "-c", "--conf", type=float, default=0.4,
        help="Detection confidence threshold. Default: 0.4",
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output video file path (optional).",
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Disable live preview window.",
    )

    args = parser.parse_args(argv)

    # Lazy import so --help stays fast
    from .core import PrivacyGuard

    source: int | str
    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    guard = PrivacyGuard(
        model_path=args.model,
        method=args.method,
        conf_threshold=args.conf,
    )
    guard.run(
        source=source,
        display=not args.no_display,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
