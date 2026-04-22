#!/usr/bin/env python3
"""
Offline preprocessing for experiment videos.

This script crops every input video to a target screen aspect ratio, downsamples
so the shorter output dimension is 720 pixels, strips audio, re-encodes to
HEVC/H.265 yuv420p for compact and ffpyplayer-friendly playback, and writes the
results into a sibling `cropped_videos` directory by default.
"""
import argparse
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path


VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".mpeg", ".mpg", ".m4v", ".wmv"}
PREFERRED_VIDEO_STREAM_CODEC = "hevc"
HEVC_ENCODER_NAMES = {
    "libx265",
    "hevc_videotoolbox",
    "hevc_nvenc",
    "hevc_qsv",
    "hevc_amf",
    "hevc_vaapi",
    "hevc_v4l2m2m",
    "hevc_rkmpp",
    "hevc_mf",
}
HARDWARE_HEVC_ENCODER_NAMES = {
    "hevc_videotoolbox",
    "hevc_nvenc",
    "hevc_qsv",
    "hevc_amf",
    "hevc_vaapi",
    "hevc_v4l2m2m",
    "hevc_rkmpp",
    "hevc_mf",
}


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
    parser.add_argument("--codec", default="auto", help="HEVC video codec to use, or 'auto' to prefer hardware")
    parser.add_argument("--preset", default="veryfast", help="Encoder preset used by software HEVC encoders")
    parser.add_argument("--crf", type=int, default=20, help="Quality level for libx265-style encoders (lower is higher quality)")
    parser.add_argument(
        "--tune",
        default="fastdecode",
        help="Optional encoder tune for easier playback; use '' to disable",
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
    if short_dim < 64:
        raise ValueError(
            f"short_dim={short_dim} is unrealistically small. "
            "Use a real pixel size such as 720."
        )

    if screen_width <= screen_height:
        out_width = int(short_dim)
        out_height = _round_even(out_width * float(screen_height) / float(screen_width))
    else:
        out_height = int(short_dim)
        out_width = _round_even(out_height * float(screen_width) / float(screen_height))
    return out_width, out_height


def _run_checked(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, capture_output=True, text=True)


def warn(message: str) -> None:
    print(f"WARNING: {message}", file=sys.stderr)


def _list_ffmpeg_encoders(ffmpeg_bin: str) -> set[str]:
    result = _run_checked([ffmpeg_bin, "-hide_banner", "-encoders"])
    encoders: set[str] = set()
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0][0] in {"V", "A", "S", "."}:
            encoders.add(parts[1])
    return encoders


def _is_hevc_encoder_name(codec_name: str) -> bool:
    return str(codec_name).strip() in HEVC_ENCODER_NAMES


def _preferred_hevc_encoder_order() -> list[str]:
    machine = platform.machine().lower()
    if sys.platform == "darwin":
        return ["hevc_videotoolbox", "libx265"]
    if sys.platform == "linux" and machine in {"aarch64", "arm64", "armv7l", "armv6l"}:
        # Raspberry Pi 5/Bookworm has a hardware HEVC decoder, not a practical
        # FFmpeg HEVC encoder path. Prefer software HEVC encoding directly.
        return ["libx265"]
    return [
        "hevc_nvenc",
        "hevc_qsv",
        "hevc_vaapi",
        "hevc_amf",
        "hevc_mf",
        "libx265",
    ]


def resolve_hevc_encoder_candidates(requested_codec: str, available_encoders: set[str]) -> list[str]:
    requested = str(requested_codec).strip()
    if requested == "auto":
        candidates = [codec for codec in _preferred_hevc_encoder_order() if codec in available_encoders]
        if sys.platform == "linux" and platform.machine().lower() in {"aarch64", "arm64", "armv7l", "armv6l"}:
            warn("pi5_bookworm_expected_encode_path software_hevc_libx265")
        return candidates
    if not _is_hevc_encoder_name(requested):
        raise ValueError(
            f"codec={requested} is not an allowed HEVC/H.265 encoder. "
            f"Expected 'auto' or one of: {', '.join(sorted(HEVC_ENCODER_NAMES))}"
        )
    if requested not in available_encoders:
        warn(f"requested_hevc_encoder_unavailable codec={requested}")
        raise RuntimeError(f"Requested HEVC encoder is not available in ffmpeg: {requested}")
    return [requested]


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
        f"scale={out_width}:{out_height}:flags=fast_bilinear"
    )


def output_path_for(video_path: Path, output_dir: Path) -> Path:
    return output_dir / f"{video_path.stem}.mp4"


def preprocess_video(
    ffmpeg_bin: str,
    ffprobe_bin: str,
    input_path: Path,
    input_codec_name: str,
    output_path: Path,
    filter_chain: str,
    codecs: list[str],
    preset: str,
    crf: int,
    tune: str,
    overwrite: bool,
    expected_size: tuple[int, int],
) -> str:
    if output_path.exists() and not overwrite:
        print(f"Skipping existing file: {output_path}")
        return "skipped"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for codec in codecs:
        input_args: list[str] = []
        # On Pi 5/Bookworm, Raspberry Pi OS ffmpeg can use the HEVC stateless
        # decoder via `-hwaccel drm`. Only try this for HEVC inputs.
        if (
            sys.platform == "linux"
            and platform.machine().lower() in {"aarch64", "arm64", "armv7l", "armv6l"}
            and str(input_codec_name).strip().lower() == PREFERRED_VIDEO_STREAM_CODEC
        ):
            input_args = ["-hwaccel", "drm"]

        encoder_args = [
            "-c:v",
            codec,
            "-profile:v",
            "main",
        ]
        if codec == "libx265":
            encoder_args.extend([
                "-preset",
                preset,
                "-crf",
                str(int(crf)),
            ])
            if tune:
                encoder_args.extend(["-tune", tune])
        elif codec == "hevc_videotoolbox":
            encoder_args.extend([
                "-allow_sw",
                "0",
                "-realtime",
                "1",
                "-prio_speed",
                "1",
            ])

        cmd = [
            ffmpeg_bin,
            "-y" if overwrite else "-n",
            *input_args,
            "-i",
            str(input_path),
            "-an",
            "-sn",
            "-dn",
            "-vf",
            filter_chain,
            *encoder_args,
            "-tag:v",
            "hvc1",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]

        try:
            _run_checked(cmd)
            stream = probe_video(ffprobe_bin, output_path)
            if stream.get("codec_name") != PREFERRED_VIDEO_STREAM_CODEC:
                raise RuntimeError(
                    f"Processed video is not HEVC/H.265: {output_path} ({stream.get('codec_name')})"
                )
            if stream.get("pix_fmt") != "yuv420p":
                raise RuntimeError(f"Processed video is not yuv420p: {output_path} ({stream.get('pix_fmt')})")
            if int(stream.get("width", 0)) != int(expected_size[0]) or int(stream.get("height", 0)) != int(expected_size[1]):
                raise RuntimeError(
                    f"Processed video has unexpected size: {output_path} "
                    f"({stream.get('width')}x{stream.get('height')} vs expected {expected_size[0]}x{expected_size[1]})"
                )
            return codec
        except Exception as exc:
            last_error = exc
            if output_path.exists():
                output_path.unlink(missing_ok=True)
            warn(
                f"encoder_failed file={input_path.name} codec={codec} "
                f"hwaccel={'drm' if input_args else 'none'} error={exc}"
            )

    raise RuntimeError(f"All HEVC encoder attempts failed for {input_path.name}: {last_error}")


def main() -> None:
    args = parse_args()
    ffmpeg_bin = shutil.which(args.ffmpeg) or args.ffmpeg
    ffprobe_bin = shutil.which(args.ffprobe) or args.ffprobe
    available_encoders = _list_ffmpeg_encoders(ffmpeg_bin)
    codec_candidates = resolve_hevc_encoder_candidates(str(args.codec), available_encoders)
    if not codec_candidates:
        raise RuntimeError("No usable HEVC encoder is available in ffmpeg")
    print(f"Encoder candidates: {', '.join(codec_candidates)}")

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
        if str(input_stream.get("codec_name", "")).strip().lower() != PREFERRED_VIDEO_STREAM_CODEC:
            warn(
                f"input_video_codec_mismatch file={video_path.name} "
                f"codec={input_stream.get('codec_name')} expected={PREFERRED_VIDEO_STREAM_CODEC} "
                f"action=reencode_to_{PREFERRED_VIDEO_STREAM_CODEC}"
            )
        used_codec = preprocess_video(
            ffmpeg_bin=ffmpeg_bin,
            ffprobe_bin=ffprobe_bin,
            input_path=video_path,
            input_codec_name=str(input_stream.get("codec_name", "")),
            output_path=output_path_for(video_path, output_dir),
            filter_chain=filter_chain,
            codecs=codec_candidates,
            preset=str(args.preset),
            crf=int(args.crf),
            tune=str(args.tune).strip(),
            overwrite=bool(args.overwrite),
            expected_size=(out_width, out_height),
        )
        if used_codec in HARDWARE_HEVC_ENCODER_NAMES:
            print(f"Encoded {video_path.name} with hardware HEVC encoder {used_codec}")
        elif used_codec == "libx265":
            warn(f"software_hevc_encode_in_use file={video_path.name} codec={used_codec}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
