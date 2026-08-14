"""Video validation, shared-frame mirroring, and frame pulse scheduling."""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from fractions import Fraction
import math
from multiprocessing import shared_memory
from pathlib import Path
import os
import random
import struct
import time
from typing import Any, Callable, Optional, Sequence

import numpy as np


REQUIRED_VIDEO_CODEC = "hevc"
REQUIRED_VIDEO_PIXEL_FORMAT = "yuv420p"
PI5_MAX_DECODE_PIXELS = 3840 * 2160
PI5_MAX_DECODE_DIMENSION = 4096
PI5_MAX_DECODE_FPS = 60.0

_SEQUENCE = struct.Struct("<Q")
_SLOT_METADATA = struct.Struct("<QqIIIddq")
_SHARED_VIDEO_SLOT_COUNT = 4
_HEADER_SIZE = 256


def _slot_metadata_offset(slot_index: int) -> int:
    return _SEQUENCE.size + (int(slot_index) * _SLOT_METADATA.size)


def parse_frame_rate(value: Any) -> float:
    """Parse an ffprobe frame-rate value such as ``30000/1001``."""
    text = str(value or "").strip()
    if not text or text == "0/0":
        return 0.0
    try:
        return float(Fraction(text))
    except (ValueError, ZeroDivisionError):
        return 0.0


def video_duration_seconds(stream: dict[str, Any]) -> float:
    """Return a validated ffprobe stream/container duration in seconds."""
    try:
        duration = float(stream.get("duration", 0.0) or 0.0)
    except (TypeError, ValueError):
        duration = 0.0
    if not math.isfinite(duration) or duration <= 0.0:
        return 0.0
    return duration


@dataclass(frozen=True)
class VideoClipSelection:
    source_duration_s: float
    start_s: float
    end_s: float
    duration_s: float
    start_frame: Optional[int] = None


def select_random_video_clip(
    stream: dict[str, Any],
    clip_duration_s: float,
    *,
    rng: Optional[random.Random] = None,
) -> VideoClipSelection:
    """Select a uniformly random valid temporal clip from a probed stream.

    When frame rate metadata is available, the start is snapped to a source
    frame boundary while preserving a uniform choice over all valid starts.
    """
    clip_duration_s = float(clip_duration_s)
    if not math.isfinite(clip_duration_s) or clip_duration_s <= 0.0:
        raise ValueError("clip_duration_seconds must be a positive finite value")

    source_duration_s = video_duration_seconds(stream)
    if source_duration_s <= 0.0:
        raise ValueError("video duration is missing or invalid")
    if clip_duration_s > source_duration_s + 1e-9:
        raise ValueError(
            f"clip_duration_seconds={clip_duration_s:.6f} exceeds "
            f"source duration {source_duration_s:.6f}"
        )

    chooser = rng or random.Random()
    maximum_start_s = max(0.0, source_duration_s - clip_duration_s)
    frame_rate = parse_frame_rate(
        stream.get("avg_frame_rate") or stream.get("r_frame_rate")
    )
    start_frame: Optional[int] = None
    if frame_rate > 0.0:
        maximum_start_frame = max(
            0,
            int(math.floor((maximum_start_s * frame_rate) + 1e-9)),
        )
        start_frame = chooser.randint(0, maximum_start_frame)
        start_s = float(start_frame) / frame_rate
    else:
        start_s = chooser.uniform(0.0, maximum_start_s)

    end_s = start_s + clip_duration_s
    return VideoClipSelection(
        source_duration_s=source_duration_s,
        start_s=start_s,
        end_s=end_s,
        duration_s=clip_duration_s,
        start_frame=start_frame,
    )


def _wait_for_seekable_vlc_player(movie, deadline_perf_s: float) -> bool:
    player = getattr(movie, "_player", None)
    while player is not None and time.perf_counter() < deadline_perf_s:
        try:
            if bool(player.is_seekable()):
                return True
        except Exception:
            return False
        time.sleep(0.002)
    return False


def prepare_vlc_clip(
    movie,
    clip_start_s: float,
    seek_timeout_s: float,
    *,
    ready_callback: Optional[Callable[[], None]] = None,
) -> float:
    """Seek without presenting pre-seek frames, then pause on the first frame."""
    clip_start_s = float(clip_start_s)
    seek_timeout_s = float(seek_timeout_s)
    if not math.isfinite(clip_start_s) or clip_start_s < 0.0:
        raise ValueError("clip_start_s must be a finite non-negative value")
    if not math.isfinite(seek_timeout_s) or seek_timeout_s <= 0.0:
        raise ValueError("seek_timeout_s must be a positive finite value")

    if not bool(getattr(movie, "isPlaying", False)):
        movie.play(log=False)

    deadline_perf_s = time.perf_counter() + seek_timeout_s
    if not _wait_for_seekable_vlc_player(movie, deadline_perf_s):
        raise RuntimeError(
            f"VLC source did not become seekable within {seek_timeout_s:.1f}s"
        )
    counter_before_seek = int(getattr(movie, "_frameCounter", 0))
    movie.seek(clip_start_s, log=False)

    try:
        source_frame_duration_s = 1.0 / max(float(movie.getFPS()), 1.0)
    except Exception:
        source_frame_duration_s = 1.0 / 30.0
    seek_tolerance_s = max(0.050, source_frame_duration_s * 1.5)

    while time.perf_counter() < deadline_perf_s:
        frame_counter = int(getattr(movie, "_frameCounter", 0))
        try:
            source_time_s = float(movie.getCurrentFrameTime())
        except Exception:
            source_time_s = 0.0
        frame_ready = frame_counter > counter_before_seek
        at_target = clip_start_s <= 0.0 or source_time_s >= (
            clip_start_s - seek_tolerance_s
        )
        if frame_ready and at_target:
            movie.pause(log=False)
            if ready_callback is not None:
                ready_callback()
            return source_time_s
        if bool(getattr(movie, "isFinished", False)):
            break
        time.sleep(0.002)

    raise RuntimeError(
        f"VLC did not decode the requested clip start {clip_start_s:.6f}s "
        f"within {seek_timeout_s:.1f}s"
    )


def validate_hevc_stream(
    video_path: str | Path,
    stream: dict[str, Any],
    *,
    require_pi5_compatible: bool = False,
) -> None:
    """Require the HEVC Main/yuv420p format used by the Pi 5 playback path."""
    path = Path(video_path)
    codec = str(stream.get("codec_name", "")).strip().lower()
    pixel_format = str(stream.get("pix_fmt", "")).strip().lower()
    profile = str(stream.get("profile", "")).strip().lower()
    width = int(stream.get("width", 0) or 0)
    height = int(stream.get("height", 0) or 0)
    frame_rate = parse_frame_rate(
        stream.get("avg_frame_rate") or stream.get("r_frame_rate")
    )

    problems: list[str] = []
    if codec != REQUIRED_VIDEO_CODEC:
        problems.append(f"codec={codec or 'unknown'} (required hevc/H.265)")
    if pixel_format != REQUIRED_VIDEO_PIXEL_FORMAT:
        problems.append(
            f"pix_fmt={pixel_format or 'unknown'} (required yuv420p 8-bit 4:2:0)"
        )
    if profile and profile != "main":
        problems.append(f"profile={profile} (required Main)")
    if width <= 0 or height <= 0 or width % 2 or height % 2:
        problems.append(f"size={width}x{height} (required positive even dimensions)")

    if require_pi5_compatible and width > 0 and height > 0:
        if width * height > PI5_MAX_DECODE_PIXELS:
            problems.append(
                f"size={width}x{height} exceeds the Pi 5 4K decode pixel limit"
            )
        if max(width, height) > PI5_MAX_DECODE_DIMENSION:
            problems.append(
                f"size={width}x{height} exceeds the Pi 5 maximum decode dimension"
            )
        if frame_rate <= 0.0:
            problems.append("frame_rate=unknown (required for Pi 5 validation)")
        elif frame_rate > PI5_MAX_DECODE_FPS + 1e-6:
            problems.append(
                f"frame_rate={frame_rate:.6f} exceeds the Pi 5 4K60 decode target"
            )

    if problems:
        detail = "; ".join(problems)
        raise ValueError(
            f"Video is not compatible with required Pi 5 hardware HEVC playback: "
            f"{path.name}: {detail}. Run bin/preprocess_videos.py first."
        )


def raspberry_pi_model(model_path: str | Path = "/proc/device-tree/model") -> str:
    try:
        return Path(model_path).read_bytes().replace(b"\x00", b"").decode(
            "utf-8", errors="replace"
        ).strip()
    except OSError:
        return ""


def is_raspberry_pi(model_path: str | Path = "/proc/device-tree/model") -> bool:
    return "raspberry pi" in raspberry_pi_model(model_path).lower()


def find_pi_hevc_decoder_device(
    sys_video_root: str | Path = "/sys/class/video4linux",
    dev_root: str | Path = "/dev",
) -> Optional[Path]:
    """Find an accessible Raspberry Pi HEVC V4L2 decoder device."""
    sys_root = Path(sys_video_root)
    device_root = Path(dev_root)
    try:
        entries = sorted(sys_root.glob("video*"))
    except OSError:
        entries = []

    for entry in entries:
        try:
            device_name = (entry / "name").read_text(
                encoding="utf-8", errors="replace"
            ).strip().lower()
        except OSError:
            continue
        if "hevc" not in device_name and "rpivid" not in device_name:
            continue
        candidate = device_root / entry.name
        if candidate.exists() and os.access(candidate, os.R_OK | os.W_OK):
            return candidate

    fallback = device_root / "video19"
    if fallback.exists() and os.access(fallback, os.R_OK | os.W_OK):
        return fallback
    return None


@dataclass(frozen=True)
class FramePulseEdge:
    level: int
    frame_index: int
    interval_frames: Optional[int] = None


class RandomFramePulseSchedule:
    """Schedule fixed-width pulses at randomized display-frame intervals."""

    def __init__(
        self,
        minimum_interval_frames: int = 100,
        maximum_interval_frames: int = 300,
        pulse_width_frames: int = 1,
        *,
        rng: Optional[random.Random] = None,
    ):
        self.minimum_interval_frames = int(minimum_interval_frames)
        self.maximum_interval_frames = int(maximum_interval_frames)
        self.pulse_width_frames = int(pulse_width_frames)
        if self.minimum_interval_frames <= 0:
            raise ValueError("minimum sync interval must be at least one frame")
        if self.maximum_interval_frames < self.minimum_interval_frames:
            raise ValueError("maximum sync interval must be >= minimum sync interval")
        if self.pulse_width_frames <= 0:
            raise ValueError("sync pulse width must be at least one frame")
        if self.pulse_width_frames >= self.minimum_interval_frames:
            raise ValueError("sync pulse width must be shorter than the minimum interval")

        self._rng = rng or random.Random()
        self._high = False
        self._off_frame: Optional[int] = None
        first_interval = self._sample_interval()
        self._next_on_frame = first_interval
        self._next_interval = first_interval

    @property
    def high(self) -> bool:
        return self._high

    def _sample_interval(self) -> int:
        return int(
            self._rng.randint(
                self.minimum_interval_frames,
                self.maximum_interval_frames,
            )
        )

    def edges_for_frame(self, frame_index: int) -> tuple[FramePulseEdge, ...]:
        frame_index = int(frame_index)
        edges: list[FramePulseEdge] = []
        if self._high and self._off_frame == frame_index:
            self._high = False
            self._off_frame = None
            edges.append(FramePulseEdge(level=0, frame_index=frame_index))

        if (not self._high) and self._next_on_frame == frame_index:
            interval = self._next_interval
            self._high = True
            self._off_frame = frame_index + self.pulse_width_frames
            edges.append(
                FramePulseEdge(
                    level=1,
                    frame_index=frame_index,
                    interval_frames=interval,
                )
            )
            self._next_interval = self._sample_interval()
            self._next_on_frame = frame_index + self._next_interval
        return tuple(edges)

    def mark_forced_low(self, frame_index: int) -> Optional[FramePulseEdge]:
        if not self._high:
            return None
        self._high = False
        self._off_frame = None
        return FramePulseEdge(level=0, frame_index=int(frame_index))


class SharedVideoFrameBuffer:
    """Four-slot, latest-frame-wins shared memory for decoded RGBA frames."""

    def __init__(self, maximum_frame_bytes: int):
        self.maximum_frame_bytes = int(maximum_frame_bytes)
        if self.maximum_frame_bytes <= 0:
            raise ValueError("maximum_frame_bytes must be positive")
        total_bytes = _HEADER_SIZE + (
            _SHARED_VIDEO_SLOT_COUNT * self.maximum_frame_bytes
        )
        self._shm = shared_memory.SharedMemory(create=True, size=total_bytes)
        self._sequence = 0
        self._closed = False
        self._shm.buf[:_HEADER_SIZE] = b"\x00" * _HEADER_SIZE

    @property
    def name(self) -> str:
        return self._shm.name

    @property
    def sequence(self) -> int:
        return self._sequence

    @property
    def slot_count(self) -> int:
        return _SHARED_VIDEO_SLOT_COUNT

    def descriptor(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "maximum_frame_bytes": self.maximum_frame_bytes,
            "slot_count": self.slot_count,
        }

    def publish_rgba(
        self,
        frame_buffer,
        *,
        width: int,
        height: int,
        source_frame_index: int,
        source_media_time_s: Optional[float] = None,
        main_display_flip_perf_s: Optional[float] = None,
        trial_num: Optional[int] = None,
        lock=None,
    ) -> int:
        width = int(width)
        height = int(height)
        frame_bytes = width * height * 4
        if width <= 0 or height <= 0 or frame_bytes > self.maximum_frame_bytes:
            raise ValueError(
                f"RGBA frame {width}x{height} does not fit shared capacity "
                f"{self.maximum_frame_bytes} bytes"
            )

        next_sequence = self._sequence + 1
        slot_index = next_sequence % _SHARED_VIDEO_SLOT_COUNT
        slot_offset = _HEADER_SIZE + (slot_index * self.maximum_frame_bytes)
        lock_context = lock if lock is not None else nullcontext()
        with lock_context:
            source = np.ctypeslib.as_array(frame_buffer).view(np.uint8).reshape(-1)
            if source.size < frame_bytes:
                raise ValueError(
                    f"Decoded frame buffer has {source.size} bytes; expected {frame_bytes}"
                )
            target = np.ndarray(
                (frame_bytes,),
                dtype=np.uint8,
                buffer=self._shm.buf,
                offset=slot_offset,
            )
            np.copyto(target, source[:frame_bytes], casting="no")

        media_time = (
            float(source_media_time_s)
            if source_media_time_s is not None
            else math.nan
        )
        main_flip_time = (
            float(main_display_flip_perf_s)
            if main_display_flip_perf_s is not None
            else math.nan
        )
        stored_trial_num = int(trial_num) if trial_num is not None else -1

        # Metadata belongs to the same ring slot as its pixels. The per-slot
        # sequence rejects stale slots; the global sequence below is committed
        # only after both pixels and metadata are complete.
        _SLOT_METADATA.pack_into(
            self._shm.buf,
            _slot_metadata_offset(slot_index),
            next_sequence,
            int(source_frame_index),
            width,
            height,
            frame_bytes,
            media_time,
            main_flip_time,
            stored_trial_num,
        )
        _SEQUENCE.pack_into(self._shm.buf, 0, next_sequence)
        self._sequence = next_sequence
        return next_sequence

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._shm.close()
        try:
            self._shm.unlink()
        except FileNotFoundError:
            pass


@dataclass(frozen=True)
class SharedVideoFrame:
    sequence: int
    source_frame_index: int
    source_media_time_s: Optional[float]
    main_display_flip_perf_s: Optional[float]
    trial_num: Optional[int]
    width: int
    height: int
    rgba: np.ndarray


class SharedVideoFrameReader:
    """Read the newest complete frame from a :class:`SharedVideoFrameBuffer`."""

    def __init__(
        self,
        name: str,
        maximum_frame_bytes: int,
        *,
        slot_count: int = _SHARED_VIDEO_SLOT_COUNT,
        unregister_resource_tracker: bool = False,
    ):
        self.name = str(name)
        self.maximum_frame_bytes = int(maximum_frame_bytes)
        self.slot_count = int(slot_count)
        if self.slot_count != _SHARED_VIDEO_SLOT_COUNT:
            raise ValueError(
                f"Unsupported shared video slot_count={self.slot_count}; "
                f"expected {_SHARED_VIDEO_SLOT_COUNT}"
            )
        self._shm = shared_memory.SharedMemory(name=self.name, create=False)
        self._local = np.empty((self.maximum_frame_bytes,), dtype=np.uint8)
        self._closed = False
        if unregister_resource_tracker:
            try:
                from multiprocessing import resource_tracker

                resource_tracker.unregister(self._shm._name, "shared_memory")
            except Exception:
                pass

    def read_latest(
        self,
        last_sequence: int,
        *,
        minimum_sequence: int = 1,
    ) -> Optional[SharedVideoFrame]:
        sequence_before = _SEQUENCE.unpack_from(self._shm.buf, 0)[0]
        if sequence_before <= int(last_sequence) or sequence_before < int(minimum_sequence):
            return None
        slot_index = sequence_before % self.slot_count
        (
            slot_sequence,
            source_frame_index,
            width,
            height,
            frame_bytes,
            source_media_time_s,
            main_display_flip_perf_s,
            trial_num,
        ) = _SLOT_METADATA.unpack_from(
            self._shm.buf,
            _slot_metadata_offset(slot_index),
        )
        if (
            slot_sequence != sequence_before
            or width <= 0
            or height <= 0
            or frame_bytes != width * height * 4
            or frame_bytes > self.maximum_frame_bytes
        ):
            return None

        slot_offset = _HEADER_SIZE + (slot_index * self.maximum_frame_bytes)
        source = np.ndarray(
            (frame_bytes,),
            dtype=np.uint8,
            buffer=self._shm.buf,
            offset=slot_offset,
        )
        np.copyto(self._local[:frame_bytes], source, casting="no")
        sequence_after = _SEQUENCE.unpack_from(self._shm.buf, 0)[0]
        if sequence_after != sequence_before:
            return None

        rgba = self._local[:frame_bytes].reshape((height, width, 4))
        return SharedVideoFrame(
            sequence=int(sequence_before),
            source_frame_index=int(source_frame_index),
            source_media_time_s=(
                float(source_media_time_s)
                if math.isfinite(source_media_time_s)
                else None
            ),
            main_display_flip_perf_s=(
                float(main_display_flip_perf_s)
                if math.isfinite(main_display_flip_perf_s)
                else None
            ),
            trial_num=(int(trial_num) if int(trial_num) >= 0 else None),
            width=int(width),
            height=int(height),
            rgba=rgba,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._shm.close()


def center_crop_bounds(
    content_size: Sequence[int],
    target_size: Sequence[int],
) -> tuple[int, int, int, int]:
    """Return ``left, top, right, bottom`` for an aspect-cover center crop."""
    content_w, content_h = int(content_size[0]), int(content_size[1])
    target_w, target_h = int(target_size[0]), int(target_size[1])
    if min(content_w, content_h, target_w, target_h) <= 0:
        raise ValueError("content and target sizes must be positive")

    content_aspect = content_w / content_h
    target_aspect = target_w / target_h
    if content_aspect > target_aspect:
        crop_h = content_h
        crop_w = max(1, min(content_w, int(round(crop_h * target_aspect))))
    else:
        crop_w = content_w
        crop_h = max(1, min(content_h, int(round(crop_w / target_aspect))))
    left = (content_w - crop_w) // 2
    top = (content_h - crop_h) // 2
    return left, top, left + crop_w, top + crop_h
