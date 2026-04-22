#!/usr/bin/env python3
"""
Offline preprocessing for experiment videos.

This script crops every input video to a target screen aspect ratio, downsamples
so the shorter output dimension is 720 pixels, strips audio, converts to BGRA,
and writes the results into a sibling `cropped_videos` directory by default.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".mpeg", ".mpg", ".m4v", ".wmv"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess videos for low-overhead playback")
    parser.add_argument("--input_dir", default="./task/resources/video", help="Directory with source videos")
    parser.add_argument(
        "--output_dir",
        default="./task/resources/cropped_videos",
        help="Directory for processed videos",
    )
    parser.add_argument(
        "--screen_size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        required=True,
        help="Target screen size used to derive the crop aspect ratio",
    )
    parser.add_argument(
        "--short_dim",
        type=int,
        default=720,
        help="Target size for the shorter output dimension after cropping",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    parser.add_argument("--ffmpeg", default="ffmpeg", help="Path to ffmpeg executable")
    parser.add_argument("--ffprobe", default="ffprobe", help="Path to ffprobe executable")
    return parser.parse_args()


def find_video_files(input_dir: Path) -> list[Path]:
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    return sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS])


def _round_even(value: float) -> int:
    rounded = int(round(float(value)))
    if rounded % 2 != 0:
        rounded += 1
    return max(2, rounded)


def compute_target_size(screen_width: int, screen_height: int, short_dim: int) -> tuple[int, int]:
    if screen_width <= 0 or screen_height <= 0:
        raise ValueError("screen_size values must be positive integers")
    if short_dim <= 0:
        raise ValueError("short_dim must be a positive integer")

    if screen_width <= screen_height:
        out_width = int(short_dim)
        out_height = _round_even(out_width * float(screen_height) / float(screen_width))
    else:
        out_height = int(short_dim)
        out_width = _round_even(out_height * float(screen_width) / float(screen_height))
    return out_width, out_height


def _run_checked(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, capture_output=True, text=True)


def probe_video(ffprobe_bin: str, path: Path) -> dict:
    cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,pix_fmt,r_frame_rate,codec_name",
        "-of",
        "json",
        str(path),
    ]
    result = _run_checked(cmd)
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    if not streams:
        raise RuntimeError(f"No video stream found in {path}")
    return streams[0]


def build_filter(screen_width: int, screen_height: int, out_width: int, out_height: int) -> str:
    aspect = float(screen_width) / float(screen_height)
    crop_w = f"if(gte(iw/ih\\,{aspect:.12f})\\,ih*{aspect:.12f}\\,iw)"
    crop_h = f"if(gte(iw/ih\\,{aspect:.12f})\\,ih\\,iw/{aspect:.12f})"
    return (
        f"crop={crop_w}:{crop_h}:(iw-ow)/2:(ih-oh)/2,"
        f"scale={out_width}:{out_height}:flags=lanczos,"
        "format=bgra"
    )


def output_path_for(video_path: Path, output_dir: Path) -> Path:
    return output_dir / f"{video_path.stem}.mov"


def preprocess_video(
    ffmpeg_bin: str,
    ffprobe_bin: str,
    input_path: Path,
    output_path: Path,
    filter_chain: str,
    overwrite: bool,
) -> None:
    if output_path.exists() and not overwrite:
        print(f"Skipping existing file: {output_path}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_bin,
        "-y" if overwrite else "-n",
        "-i",
        str(input_path),
        "-an",
        "-sn",
        "-dn",
        "-vf",
        filter_chain,
        "-pix_fmt",
        "bgra",
        "-c:v",
        "qtrle",
        str(output_path),
    ]
    _run_checked(cmd)

    stream = probe_video(ffprobe_bin, output_path)
    if stream.get("pix_fmt") != "bgra":
        raise RuntimeError(f"Processed video is not BGRA: {output_path} ({stream.get('pix_fmt')})")


def main() -> None:
    args = parse_args()
    ffmpeg_bin = shutil.which(args.ffmpeg) or args.ffmpeg
    ffprobe_bin = shutil.which(args.ffprobe) or args.ffprobe

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    screen_width, screen_height = (int(v) for v in args.screen_size)
    out_width, out_height = compute_target_size(screen_width, screen_height, int(args.short_dim))
    filter_chain = build_filter(screen_width, screen_height, out_width, out_height)

    videos = find_video_files(input_dir)
    if not videos:
        raise FileNotFoundError(f"No video files found in {input_dir}")

    print(
        f"Preprocessing {len(videos)} videos from {input_dir} to {output_dir} "
        f"with target size {out_width}x{out_height}"
    )
    for video_path in videos:
        input_stream = probe_video(ffprobe_bin, video_path)
        print(
            f"Processing {video_path.name}: "
            f"{input_stream.get('width')}x{input_stream.get('height')} "
            f"{input_stream.get('codec_name')} {input_stream.get('pix_fmt')}"
        )
        preprocess_video(
            ffmpeg_bin=ffmpeg_bin,
            ffprobe_bin=ffprobe_bin,
            input_path=video_path,
            output_path=output_path_for(video_path, output_dir),
            filter_chain=filter_chain,
            overwrite=bool(args.overwrite),
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
