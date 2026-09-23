"""Show a lightweight 30 fps preview of the four-camera aggregate."""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path
from typing import Sequence

try:
    from .recorder import DEFAULT_CONFIG_PATH, RecorderConfig, RecordingError
except ImportError:  # Allow ``python video/live_preview.py``.
    from recorder import DEFAULT_CONFIG_PATH, RecorderConfig, RecordingError


DEFAULT_PREVIEW_CONFIG_PATH = (
    DEFAULT_CONFIG_PATH
    if DEFAULT_CONFIG_PATH.is_file()
    else Path(__file__).with_name("example_config.json")
)


def preview_command(
    config_path: str | Path = DEFAULT_PREVIEW_CONFIG_PATH,
    *,
    width: int = 1280,
    height: int = 180,
) -> list[str]:
    """Return the zero-copy native preview command."""
    if width <= 0 or height <= 0:
        raise ValueError("preview width and height must be positive")
    config = RecorderConfig.load(config_path)
    return [
        "rpicam-hello",
        "--camera",
        str(config.camera),
        "--timeout",
        "0",
        "--viewfinder-mode",
        config.sensor_mode,
        "--viewfinder-width",
        str(width),
        "--viewfinder-height",
        str(height),
        "--preview",
        f"50,50,{width},{height}",
        "--framerate",
        str(config.framerate),
        "--denoise",
        config.denoise,
        "--info-text",
        "Arducam preview | %fps fps",
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Preview all four Arducam feeds until the window is closed"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_PREVIEW_CONFIG_PATH)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    command = preview_command(args.config, width=args.width, height=args.height)
    if args.dry_run:
        print(shlex.join(command))
        return 0

    try:
        os.execvp(command[0], command)
    except OSError as exc:
        raise RecordingError(f"Could not start {command[0]}: {exc}") from exc
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RecordingError, ValueError) as exc:
        print(f"preview error: {exc}", file=sys.stderr)
        raise SystemExit(1)
