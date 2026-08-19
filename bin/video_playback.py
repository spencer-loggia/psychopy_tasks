"""Video validation, shared-frame mirroring, and frame pulse scheduling."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import json
import math
from multiprocessing import shared_memory
from pathlib import Path
import random
import shutil
import struct
import subprocess
from typing import Any, Optional, Sequence

import numpy as np

from .frame_timing import plan_frame_duration


REQUIRED_VIDEO_CODEC = "hevc"
REQUIRED_VIDEO_PIXEL_FORMAT = "yuv420p"

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


def video_time_origin_seconds(stream: dict[str, Any]) -> float:
    """Return the non-negative media PTS origin reported by ffprobe."""
    raw_value = stream.get("start_time", 0.0)
    if raw_value in (None, "", "N/A"):
        return 0.0
    try:
        origin = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError("video start_time is invalid") from exc
    if not math.isfinite(origin) or origin < 0.0:
        raise ValueError(
            "video start_time must be finite and non-negative; "
            "rebase the source timestamps during preprocessing"
        )
    return origin


def probe_video_stream(
    video_path: str | Path,
    ffprobe_bin: str = "ffprobe",
) -> dict[str, Any]:
    """Return first-video-stream ffprobe metadata, or an empty mapping."""
    ffprobe_path = shutil.which(ffprobe_bin) or ffprobe_bin
    try:
        result = subprocess.run(
            [
                ffprobe_path,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                (
                    "stream=codec_name,profile,level,width,height,pix_fmt,"
                    "r_frame_rate,avg_frame_rate,start_time,duration,field_order,"
                    "has_b_frames,nb_frames,color_range,color_space:format=duration"
                ),
                "-of",
                "json",
                str(video_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(result.stdout)
    except Exception:
        return {}
    streams = payload.get("streams", [])
    if not streams:
        return {}
    stream = dict(streams[0])
    if video_duration_seconds(stream) <= 0.0:
        format_info = payload.get("format", {})
        if isinstance(format_info, dict):
            format_duration = format_info.get("duration")
            if format_duration not in (None, "", "N/A"):
                stream["duration"] = format_duration
    return stream


@dataclass(frozen=True)
class VideoClipSelection:
    source_duration_s: float
    source_time_origin_s: float
    start_s: float
    end_s: float
    duration_s: float
    start_frame: Optional[int] = None
    frame_count: int = 0
    frame_rate: float = 30.0
    requested_duration_s: float = 0.0


@dataclass(frozen=True)
class VideoRefreshCadence:
    """Nearest-VBL schedule for a fixed sequence of source frames."""

    video_frame_count: int
    video_frame_rate: float
    display_refresh_rate: float
    nominal_refreshes_per_video_frame: float
    frame_refresh_counts: tuple[int, ...]
    refresh_boundaries: tuple[int, ...]
    refresh_count_histogram: tuple[tuple[int, int], ...]
    total_refreshes: int
    source_duration_s: float
    scheduled_display_duration_s: float
    final_phase_error_s: float
    maximum_absolute_phase_error_s: float

    def refreshes_for_source_frames(
        self,
        first_frame_index: int,
        frame_count: int,
    ) -> int:
        """Return physical refreshes occupied by a source-frame interval."""
        first = max(0, min(self.video_frame_count, int(first_frame_index)))
        end = max(first, min(self.video_frame_count, first + int(frame_count)))
        return self.refresh_boundaries[end] - self.refresh_boundaries[first]


def plan_video_refresh_cadence(
    video_frame_count: int,
    video_frame_rate: float,
    display_refresh_rate: float,
) -> VideoRefreshCadence:
    """Map source-frame boundaries to nearest display refreshes.

    Each boundary is rounded independently from its absolute ideal position,
    using round-half-up on an exact rational ratio. Consequently, phase error
    never accumulates beyond half a display refresh. Adjacent boundary
    differences become the per-frame hold counts.
    """
    raw_video_frame_count = video_frame_count
    if isinstance(raw_video_frame_count, bool):
        raise ValueError("video_frame_count must be a positive integer")
    video_frame_count = int(raw_video_frame_count)
    if (
        video_frame_count <= 0
        or video_frame_count != raw_video_frame_count
    ):
        raise ValueError("video_frame_count must be a positive integer")
    video_frame_rate = float(video_frame_rate)
    display_refresh_rate = float(display_refresh_rate)
    if not math.isfinite(video_frame_rate) or video_frame_rate <= 0.0:
        raise ValueError("video_frame_rate must be positive and finite")
    if (
        not math.isfinite(display_refresh_rate)
        or display_refresh_rate <= 0.0
    ):
        raise ValueError("display_refresh_rate must be positive and finite")

    refresh_ratio = Fraction(str(display_refresh_rate)) / Fraction(
        str(video_frame_rate)
    )
    boundaries = [0]
    holds: list[int] = []
    histogram: dict[int, int] = {}
    maximum_absolute_phase_error_s = 0.0

    for boundary_index in range(1, video_frame_count + 1):
        ideal_boundary = boundary_index * refresh_ratio
        # floor(x + 1/2), evaluated with integers to avoid float drift and
        # Python's alternating ties-to-even behavior.
        refresh_boundary = (
            (2 * ideal_boundary.numerator) + ideal_boundary.denominator
        ) // (2 * ideal_boundary.denominator)
        hold_count = int(refresh_boundary - boundaries[-1])
        if hold_count < 1:
            raise ValueError(
                "The measured display rate cannot show every source frame "
                f"at least once: source frame {boundary_index - 1} would "
                "receive zero physical refreshes"
            )
        boundaries.append(int(refresh_boundary))
        holds.append(hold_count)
        histogram[hold_count] = histogram.get(hold_count, 0) + 1
        actual_boundary_s = float(refresh_boundary) / display_refresh_rate
        ideal_boundary_s = float(boundary_index) / video_frame_rate
        maximum_absolute_phase_error_s = max(
            maximum_absolute_phase_error_s,
            abs(actual_boundary_s - ideal_boundary_s),
        )

    total_refreshes = boundaries[-1]
    source_duration_s = video_frame_count / video_frame_rate
    scheduled_display_duration_s = total_refreshes / display_refresh_rate
    return VideoRefreshCadence(
        video_frame_count=video_frame_count,
        video_frame_rate=video_frame_rate,
        display_refresh_rate=display_refresh_rate,
        nominal_refreshes_per_video_frame=float(refresh_ratio),
        frame_refresh_counts=tuple(holds),
        refresh_boundaries=tuple(boundaries),
        refresh_count_histogram=tuple(sorted(histogram.items())),
        total_refreshes=total_refreshes,
        source_duration_s=source_duration_s,
        scheduled_display_duration_s=scheduled_display_duration_s,
        final_phase_error_s=(
            scheduled_display_duration_s - source_duration_s
        ),
        maximum_absolute_phase_error_s=maximum_absolute_phase_error_s,
    )


def select_random_video_clip(
    stream: dict[str, Any],
    clip_duration_s: float,
    *,
    rng: Optional[random.Random] = None,
    frame_rate: float = 30.0,
) -> VideoClipSelection:
    """Select a random clip on the configured video-frame timebase."""
    clip_duration_s = float(clip_duration_s)
    if not math.isfinite(clip_duration_s) or clip_duration_s <= 0.0:
        raise ValueError("clip_duration_seconds must be a positive finite value")

    source_duration_s = video_duration_seconds(stream)
    if source_duration_s <= 0.0:
        raise ValueError("video duration is missing or invalid")
    source_time_origin_s = video_time_origin_seconds(stream)
    configured_frame_rate = float(frame_rate)
    if not math.isfinite(configured_frame_rate) or configured_frame_rate <= 0.0:
        raise ValueError("frame_rate must be a positive finite value")
    duration_plan = plan_frame_duration(
        clip_duration_s,
        configured_frame_rate,
        minimum_frames=1,
    )
    scheduled_duration_s = duration_plan.scheduled_s
    if scheduled_duration_s > source_duration_s + 1e-9:
        raise ValueError(
            f"clip_duration_seconds={clip_duration_s:.6f} exceeds "
            f"source duration {source_duration_s:.6f}"
        )

    chooser = rng or random.Random()
    maximum_start_s = max(0.0, source_duration_s - scheduled_duration_s)
    maximum_start_frame = max(
        0,
        int(math.floor((maximum_start_s * configured_frame_rate) + 1e-9)),
    )
    start_frame = chooser.randint(0, maximum_start_frame)
    start_s = source_time_origin_s + (
        float(start_frame) / configured_frame_rate
    )
    end_s = start_s + scheduled_duration_s
    return VideoClipSelection(
        source_duration_s=source_duration_s,
        source_time_origin_s=source_time_origin_s,
        start_s=start_s,
        end_s=end_s,
        duration_s=scheduled_duration_s,
        start_frame=start_frame,
        frame_count=duration_plan.frame_count,
        frame_rate=configured_frame_rate,
        requested_duration_s=clip_duration_s,
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
    field_order = str(stream.get("field_order", "")).strip().lower()
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
    if profile != "main":
        problems.append(f"profile={profile or 'unknown'} (required Main)")
    if width <= 0 or height <= 0 or width % 2 or height % 2:
        problems.append(f"size={width}x{height} (required positive even dimensions)")
    if field_order not in {"", "unknown", "progressive"}:
        problems.append(
            f"field_order={field_order} (interlaced motion must be "
            "deinterlaced during preprocessing)"
        )

    if require_pi5_compatible and frame_rate <= 0.0:
        problems.append("frame_rate=unknown (required for Pi playback validation)")

    if problems:
        detail = "; ".join(problems)
        raise ValueError(
            f"Video is not compatible with the required HEVC playback format: "
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
    """Four-slot, latest-frame-wins shared memory for displayed RGB frames."""

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
        # Fault in the complete preview ring before timing-critical playback.
        # Otherwise the first post-onset preview copy can incur shared-memory
        # page allocation while the main process is racing the next VBL.
        np.ndarray(
            (total_bytes,),
            dtype=np.uint8,
            buffer=self._shm.buf,
        ).fill(0)

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

    def publish_rgb(
        self,
        rgb: np.ndarray,
        *,
        source_frame_index: int,
        source_media_time_s: Optional[float] = None,
        main_display_flip_perf_s: Optional[float] = None,
        trial_num: Optional[int] = None,
    ) -> int:
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("Shared video frames must be HxWx3 uint8 RGB")
        if not rgb.flags.c_contiguous:
            raise ValueError("Shared video RGB frames must be contiguous")
        height, width = int(rgb.shape[0]), int(rgb.shape[1])
        frame_bytes = width * height * 3
        if width <= 0 or height <= 0 or frame_bytes > self.maximum_frame_bytes:
            raise ValueError(
                f"RGB24 frame {width}x{height} does not fit shared capacity "
                f"{self.maximum_frame_bytes} bytes"
            )

        next_sequence = self._sequence + 1
        slot_index = next_sequence % _SHARED_VIDEO_SLOT_COUNT
        slot_offset = _HEADER_SIZE + (slot_index * self.maximum_frame_bytes)
        # Invalidate the slot before touching its pixels. A reader verifies this
        # per-slot sequence both before and after its copy, so a writer can never
        # make a partially overwritten frame look committed.
        _SEQUENCE.pack_into(
            self._shm.buf,
            _slot_metadata_offset(slot_index),
            0,
        )
        target = np.ndarray(
            (height, width, 3),
            dtype=np.uint8,
            buffer=self._shm.buf,
            offset=slot_offset,
        )
        np.copyto(target, rgb, casting="no")

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
    rgb: np.ndarray


class SharedVideoFrameReader:
    """Read the newest complete frame from a :class:`SharedVideoFrameBuffer`."""

    def __init__(
        self,
        name: str,
        maximum_frame_bytes: int,
        *,
        slot_count: int = _SHARED_VIDEO_SLOT_COUNT,
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
            or frame_bytes != width * height * 3
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
        slot_sequence_after = _SEQUENCE.unpack_from(
            self._shm.buf,
            _slot_metadata_offset(slot_index),
        )[0]
        if (
            sequence_after != sequence_before
            or slot_sequence_after != sequence_before
        ):
            return None

        rgb = self._local[:frame_bytes].reshape((height, width, 3))
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
            rgb=rgb,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._shm.close()


def center_crop_bounds(
    content_size: Sequence[int],
    target_size: Sequence[int],
    *,
    alignment: int = 1,
) -> tuple[int, int, int, int]:
    """Return ``left, top, right, bottom`` for an aspect-cover center crop."""
    content_w, content_h = int(content_size[0]), int(content_size[1])
    target_w, target_h = int(target_size[0]), int(target_size[1])
    if min(content_w, content_h, target_w, target_h) <= 0:
        raise ValueError("content and target sizes must be positive")
    alignment = int(alignment)
    if alignment <= 0:
        raise ValueError("crop alignment must be positive")

    content_aspect = content_w / content_h
    target_aspect = target_w / target_h
    if content_aspect > target_aspect:
        crop_h = content_h
        crop_w = max(1, min(content_w, int(round(crop_h * target_aspect))))
    else:
        crop_w = content_w
        crop_h = max(1, min(content_h, int(round(crop_w / target_aspect))))
    if alignment > 1:
        crop_w = max(alignment, crop_w - (crop_w % alignment))
        crop_h = max(alignment, crop_h - (crop_h % alignment))
        crop_w = min(crop_w, content_w - (content_w % alignment))
        crop_h = min(crop_h, content_h - (content_h % alignment))
    left = (content_w - crop_w) // 2
    top = (content_h - crop_h) // 2
    if alignment > 1:
        left -= left % alignment
        top -= top % alignment
    return left, top, left + crop_w, top + crop_h


def upload_rgb_texture(stim: Any, rgb: np.ndarray) -> None:
    """Upload packed RGB24 with explicit client-memory unpack state.

    OpenGL pixel-unpack settings and PBO bindings are context-global state. A
    stale PsychoPy/pyglet binding would reinterpret the NumPy pointer as a PBO
    byte offset or apply a previous row stride, producing partial/shifted frames
    while leaving timing apparently correct.
    """
    import ctypes

    from pyglet import gl as GL

    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Video texture data must be HxWx3 uint8 RGB")
    if not rgb.flags.c_contiguous:
        raise ValueError("Video texture data must be contiguous")
    height, width = int(rgb.shape[0]), int(rgb.shape[1])
    pixel_pointer = rgb.ctypes.data_as(ctypes.POINTER(GL.GLubyte))
    # Discard pre-existing errors so the check below is attributable to this
    # upload. Bound the loop in case a broken context repeats one error.
    for _ in range(16):
        if int(GL.glGetError()) == int(GL.GL_NO_ERROR):
            break
    if hasattr(GL, "GL_PIXEL_UNPACK_BUFFER") and hasattr(GL, "glBindBuffer"):
        GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, 0)
    unpack_defaults = (
        ("GL_UNPACK_ALIGNMENT", 1),
        ("GL_UNPACK_ROW_LENGTH", 0),
        ("GL_UNPACK_SKIP_ROWS", 0),
        ("GL_UNPACK_SKIP_PIXELS", 0),
        ("GL_UNPACK_IMAGE_HEIGHT", 0),
        ("GL_UNPACK_SKIP_IMAGES", 0),
        ("GL_UNPACK_SWAP_BYTES", 0),
        ("GL_UNPACK_LSB_FIRST", 0),
    )
    for constant_name, value in unpack_defaults:
        if hasattr(GL, constant_name):
            GL.glPixelStorei(getattr(GL, constant_name), value)
    GL.glActiveTexture(GL.GL_TEXTURE0)
    GL.glBindTexture(GL.GL_TEXTURE_2D, stim._texID)
    GL.glTexSubImage2D(
        GL.GL_TEXTURE_2D,
        0,
        0,
        0,
        width,
        height,
        GL.GL_RGB,
        GL.GL_UNSIGNED_BYTE,
        pixel_pointer,
    )
    upload_error = int(GL.glGetError())
    GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
    if upload_error != int(GL.GL_NO_ERROR):
        raise RuntimeError(
            f"OpenGL RGB texture upload failed with error 0x{upload_error:04x}"
        )
