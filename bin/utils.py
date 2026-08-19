"""
Utility helpers for PsychoPy tasks (simplified).
Robust transparency on all platforms by using RGB + alpha MASK for ImageStim.
Also supports loading SVG by rasterizing to a requested pixel size (via cairosvg).

Modularity helpers included:
- make_bg_rect: create a full-window background rect in one call.
- make_onset_cue_stim: build a checkerboard ImageStim with a centered 2D Gaussian alpha mask.
"""
from pathlib import Path
import math
import random
from typing import List, Tuple, Optional, Dict, Union, Callable, Any, Sequence
import io
import multiprocessing as mp
import queue
import traceback

import numpy as np
from PIL import Image
from psychopy import visual, event
import time
from .screen import (
    build_reward_hit_boxes,
    compute_aspect_cover_size,
    enforce_window_vsync,
    initialize_psychopy_window,
    MainDisplayVBlankSession,
    MainDisplayFrameTimingMonitor,
    oriented_size,
    resolve_task_screens,
    resolve_window_frame_rate,
    serialize_preview_image,
)
from .frame_timing import flip_with_timestamps, plan_frame_duration
from .glx_timing import query_glx_swap_interval, query_glx_sync_values
from .buffered_video import (
    BufferedVideoFrameStream,
    DEFAULT_BUFFER_BYTES,
    VideoBufferUnderrun,
    VideoPreparationAborted,
)
from .video_playback import (
    RandomFramePulseSchedule,
    SharedVideoFrameBuffer,
    VideoRefreshCadence,
    center_crop_bounds,
    plan_video_refresh_cadence,
    probe_video_stream,
    upload_rgb_texture,
    video_duration_seconds,
    video_time_origin_seconds,
)
from .stimulus_files import (
    load_color_palette as _load_color_palette,
    load_shape_definitions as _load_shape_definitions,
    split_background_from_palette as _split_background_from_palette,
)
from .touch_input import MousePressTracker, advance_release_armed_touch_gate

# Global debug flag: when True, utilities may write debug files (PNG) to logs/
# Default is False; tasks can enable it via CLI (--debug) or config.
DEBUG = False


def set_debug(value: bool):
    global DEBUG
    DEBUG = bool(value)

# File types
RASTER_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
VECTOR_EXTS = {".svg"}
IMAGE_EXTS = RASTER_EXTS | VECTOR_EXTS  # for discovery
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".mpeg", ".mpg", ".m4v", ".wmv"}

NON_ASSOCIATED_SHAPE_IDS: Tuple[int, ...] = tuple(range(14, 28))



def find_image_files(images_dir: str, recursive: bool = False) -> List[Path]:
    p = Path(images_dir)
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if recursive:
        files = [f for f in p.rglob("*") if f.suffix.lower() in IMAGE_EXTS]
    else:
        files = [f for f in p.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS]
    return sorted(files)


def find_video_files(videos_dir: str, recursive: bool = False) -> List[Path]:
    p = Path(videos_dir)
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"Video directory not found: {videos_dir}")
    if recursive:
        files = [f for f in p.rglob("*") if f.suffix.lower() in VIDEO_EXTS]
    else:
        files = [f for f in p.iterdir() if f.is_file() and f.suffix.lower() in VIDEO_EXTS]
    return sorted(files)


def sample_images(files: List[Path], n: int, seed: Optional[int] = None) -> List[Path]:
    if seed is not None:
        random.seed(seed)
    if not files:
        return []
    return random.sample(files, n) if n <= len(files) else [random.choice(files) for _ in range(n)]


def rgb255_to_psychopy(rgb_255: Tuple[int, int, int]) -> List[float]:
    arr = np.clip(np.array(rgb_255, dtype=float), 0, 255)
    return ((arr / 127.5) - 1).tolist()  # 0->-1, 127.5->0, 255->1


def setup_window(
    bg_rgb_255: Tuple[int, int, int] = (128, 128, 128),
    fullscreen: bool = True,
    size: Optional[Tuple[int, int]] = None,
    screen_info=None,
    sync_to_refresh: bool = True,
):
    from .task_lifecycle import signal_task_window_ready

    color = rgb255_to_psychopy(bg_rgb_255)
    win_kwargs = dict(
        color=color,
        colorSpace="rgb",
        units="pix",
        allowStencil=False,
        waitBlanking=bool(sync_to_refresh),
    )
    win = initialize_psychopy_window(
        visual,
        screen_info,
        fullscreen=fullscreen,
        size=size,
        sync_to_refresh=sync_to_refresh,
        # The launcher removes its main-output curtain before activation and
        # strict realized-rectangle verification.
        on_window_ready=signal_task_window_ready,
        **win_kwargs,
    )
    return win


def setup_task_window(
    screen_config=None,
    *,
    bg_rgb_255: Tuple[int, int, int] = (128, 128, 128),
    fullscreen: bool = True,
    size: Optional[Tuple[int, int]] = None,
    allow_same_screen: bool = True,
):
    """Open the one verified, VBlank-synchronized subject task window."""
    main_screen, experimenter_screen = resolve_task_screens(
        screen_config,
        allow_same_screen=allow_same_screen,
    )
    vblank_session = MainDisplayVBlankSession.acquire(main_screen)
    win = None
    try:
        win = setup_window(
            bg_rgb_255=bg_rgb_255,
            fullscreen=fullscreen,
            size=size,
            screen_info=main_screen,
            sync_to_refresh=True,
        )
        win._neuro_tasks_main_vblank_session = vblank_session
        vblank_session.validate(win, reassert=False)
    except BaseException:
        try:
            if win is not None:
                vblank_session.close(win)
            else:
                vblank_session.release_primary()
        except Exception:
            pass
        raise
    return win, main_screen, experimenter_screen


def verify_task_window_vblank(win) -> MainDisplayVBlankSession:
    """Revalidate the shared main-output contract after other windows start."""
    session = getattr(win, "_neuro_tasks_main_vblank_session", None)
    if not isinstance(session, MainDisplayVBlankSession):
        raise RuntimeError("window was not created by setup_task_window")
    return session.validate(win)


def close_task_window(win) -> str:
    """Close a shared task window and restore its previous primary output."""
    session = getattr(win, "_neuro_tasks_main_vblank_session", None)
    if not isinstance(session, MainDisplayVBlankSession):
        raise RuntimeError("window was not created by setup_task_window")
    return session.close(win)


def _log_message(msg_logger, level: str, message: str) -> None:
    if msg_logger is None:
        return
    try:
        msg_logger.log(level, message)
    except Exception:
        pass


def _flush_message_logger(msg_logger) -> None:
    """Make pre/post-playback diagnostics visible without I/O between frames."""
    if msg_logger is None:
        return
    try:
        msg_logger.flush()
    except Exception:
        pass


def _summarize_video_frame_timing(
    frame_records: Sequence[Dict[str, Any]],
    *,
    clip_offset_perf_s: Optional[float],
    expected_frame_count: int,
    video_frame_rate: float,
    display_refresh_rate: float,
    planned_refresh_counts: Sequence[int],
    timing_failure_threshold_s: float,
    aborted: bool,
    source_frames_skipped: int,
) -> Dict[str, Any]:
    """Validate realized source-frame boundaries after the critical sequence.

    The returned rows retain every source PTS and flip timestamp for an
    auditable sidecar log.  All analysis happens after the final clear flip, so
    this validation cannot perturb presentation timing.
    """
    rows = [dict(record) for record in frame_records]
    actual_frame_count = len(rows)
    complete = (
        not aborted
        and actual_frame_count == int(expected_frame_count)
        and clip_offset_perf_s is not None
    )
    first_flip_perf_s = (
        float(rows[0]["actual_flip_perf_s"]) if rows else None
    )
    boundary_errors_s: list[float] = []
    source_pts_errors_s: list[float] = []
    realized_histogram: dict[int, int] = {}
    cadence_mismatch_count = 0
    monotonic = True

    for frame_index, row in enumerate(rows):
        actual_flip_perf_s = float(row["actual_flip_perf_s"])
        expected_flip_perf_s = (
            actual_flip_perf_s
            if first_flip_perf_s is None
            else first_flip_perf_s + frame_index / video_frame_rate
        )
        timing_error_s = actual_flip_perf_s - expected_flip_perf_s
        expected_source_pts_s = float(row["expected_source_pts_s"])
        source_pts_error_s = (
            float(row["source_media_time_s"]) - expected_source_pts_s
        )
        boundary_errors_s.append(timing_error_s)
        source_pts_errors_s.append(source_pts_error_s)
        row["expected_flip_perf_s"] = expected_flip_perf_s
        row["timing_error_s"] = timing_error_s
        row["source_pts_error_s"] = source_pts_error_s
        row["boundary_status"] = (
            "MISS"
            if abs(timing_error_s) > timing_failure_threshold_s
            else "OK"
        )

        next_boundary_perf_s: Optional[float]
        if frame_index + 1 < actual_frame_count:
            next_boundary_perf_s = float(
                rows[frame_index + 1]["actual_flip_perf_s"]
            )
        elif complete:
            next_boundary_perf_s = float(clip_offset_perf_s)
        else:
            next_boundary_perf_s = None
        planned_hold = (
            int(planned_refresh_counts[frame_index])
            if frame_index < len(planned_refresh_counts)
            else None
        )
        row["planned_hold_refreshes"] = planned_hold
        row["realized_hold_refreshes"] = None
        row["realized_hold_s"] = None
        if next_boundary_perf_s is not None:
            interval_s = next_boundary_perf_s - actual_flip_perf_s
            monotonic = monotonic and interval_s > 0.0
            realized_hold = max(
                0,
                int(math.floor(interval_s * display_refresh_rate + 0.5)),
            )
            row["realized_hold_s"] = interval_s
            row["realized_hold_refreshes"] = realized_hold
            realized_histogram[realized_hold] = (
                realized_histogram.get(realized_hold, 0) + 1
            )
            if planned_hold is not None and realized_hold != planned_hold:
                cadence_mismatch_count += 1

    offset_timing_error_s = None
    if complete and first_flip_perf_s is not None:
        expected_offset_perf_s = (
            first_flip_perf_s + expected_frame_count / video_frame_rate
        )
        offset_timing_error_s = (
            float(clip_offset_perf_s) - expected_offset_perf_s
        )
        boundary_errors_s.append(offset_timing_error_s)

    absolute_errors_s = sorted(abs(value) for value in boundary_errors_s)

    def _percentile(fraction: float) -> float:
        if not absolute_errors_s:
            return math.nan
        if len(absolute_errors_s) == 1:
            return absolute_errors_s[0]
        position = fraction * (len(absolute_errors_s) - 1)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return absolute_errors_s[lower]
        weight = position - lower
        return (
            absolute_errors_s[lower] * (1.0 - weight)
            + absolute_errors_s[upper] * weight
        )

    timing_miss_count = sum(
        abs(value) > timing_failure_threshold_s
        for value in boundary_errors_s
    )
    source_pts_contiguous = bool(
        actual_frame_count > 0
        and len(source_pts_errors_s) == actual_frame_count
        and all(
            abs(value) <= max(1e-4, 0.05 / video_frame_rate)
            for value in source_pts_errors_s
        )
    )
    if (
        not complete
        or source_frames_skipped
        or not monotonic
        or not source_pts_contiguous
    ):
        status = "FAIL"
    elif timing_miss_count or cadence_mismatch_count:
        status = "WARN"
    else:
        status = "PASS"

    displayed_duration_s = (
        float(clip_offset_perf_s) - first_flip_perf_s
        if complete and first_flip_perf_s is not None
        else None
    )
    effective_frame_rate = (
        expected_frame_count / displayed_duration_s
        if displayed_duration_s is not None and displayed_duration_s > 0.0
        else math.nan
    )
    planned_histogram: dict[int, int] = {}
    for hold in planned_refresh_counts:
        hold = int(hold)
        planned_histogram[hold] = planned_histogram.get(hold, 0) + 1

    return {
        "status": status,
        "complete": bool(complete),
        "frame_count": actual_frame_count,
        "source_pts_contiguous": source_pts_contiguous,
        "maximum_source_pts_error_s": max(
            (abs(value) for value in source_pts_errors_s),
            default=0.0,
        ),
        "timing_miss_count": int(timing_miss_count),
        "cadence_mismatch_count": int(cadence_mismatch_count),
        "maximum_absolute_timing_error_s": max(
            absolute_errors_s,
            default=math.nan,
        ),
        "median_absolute_timing_error_s": _percentile(0.5),
        "p95_absolute_timing_error_s": _percentile(0.95),
        "offset_timing_error_s": offset_timing_error_s,
        "monotonic": bool(monotonic),
        "planned_hold_histogram": planned_histogram,
        "realized_hold_histogram": realized_histogram,
        "nonuniform_cadence": len(planned_histogram) > 1,
        "displayed_duration_s": displayed_duration_s,
        "effective_frame_rate": effective_frame_rate,
        "timing_failure_threshold_s": timing_failure_threshold_s,
        "rows": rows,
    }


def _set_gpio_level_on_flip(lgpio_module, chip, pin: int, level: int) -> None:
    """Set a GPIO level from PsychoPy's frame-flip callback queue."""
    if lgpio_module is None or chip is None:
        raise RuntimeError("Frame sync GPIO was not initialized")
    result = lgpio_module.gpio_write(chip, int(pin), int(level))
    if isinstance(result, int) and result < 0:
        raise RuntimeError(
            f"GPIO write failed with code {result} on pin {int(pin)}"
        )


def _capture_perf_counter_on_flip(target: Dict[str, float]) -> None:
    """Capture the flip callback time before slower hardware callbacks run."""
    target["actual_perf_s"] = time.perf_counter()


def play_video_fill_screen(
    win: visual.Window,
    video_path: Union[str, Path],
    logger=None,
    bg_rect=None,
    msg_logger=None,
    allow_escape: bool = True,
    stop_on_mouse_click: bool = False,
    mouse: Optional[event.Mouse] = None,
    ffprobe_bin: str = "ffprobe",
    external_abort_checker: Optional[Callable[[], bool]] = None,
    trial_num: Optional[int] = None,
    stream_info: Optional[Dict[str, Any]] = None,
    frame_publisher: Optional[SharedVideoFrameBuffer] = None,
    sync_schedule: Optional[RandomFramePulseSchedule] = None,
    sync_gpio_module=None,
    sync_gpio_chip=None,
    sync_pin: int = 18,
    frame_publish_interval_s: float = 0.0,
    clip_start_s: float = 0.0,
    clip_duration_s: Optional[float] = None,
    seek_timeout_s: float = 15.0,
    decoder_ready_callback: Optional[Callable[[], None]] = None,
    video_onset_callback: Optional[Callable[[float], None]] = None,
    stimulus_rotation_degrees: float = 0.0,
    native_target_size: Optional[Sequence[float]] = None,
    video_frame_rate: float = 30.0,
    video_frame_count: Optional[int] = None,
    requested_clip_duration_s: Optional[float] = None,
    display_refresh_rate: Optional[float] = None,
    refresh_cadence: Optional[VideoRefreshCadence] = None,
    video_buffer_bytes: int = DEFAULT_BUFFER_BYTES,
) -> Dict[str, Any]:
    """Present an exact frame sequence prepared by an ffpyplayer worker."""
    video_file = Path(video_path)
    if stream_info is None and not video_file.is_file():
        raise FileNotFoundError(f"Video file not found: {video_file}")

    stream = dict(stream_info or {})
    if not stream:
        stream = probe_video_stream(video_file, ffprobe_bin=ffprobe_bin)
    if sync_schedule is not None and (
        sync_gpio_module is None or sync_gpio_chip is None
    ):
        raise RuntimeError("A sync schedule requires an initialized GPIO output")

    frame_publish_interval_s = float(frame_publish_interval_s)
    if frame_publish_interval_s < 0.0:
        raise ValueError("frame_publish_interval_s cannot be negative")
    source_duration_s = video_duration_seconds(stream)
    source_time_origin_s = video_time_origin_seconds(stream)
    clip_start_s = float(clip_start_s)
    if clip_duration_s is None:
        clip_duration_s = (
            source_time_origin_s + source_duration_s - clip_start_s
        )
    clip_duration_s = float(clip_duration_s)
    if not math.isfinite(clip_start_s) or clip_start_s < 0.0:
        raise ValueError("clip_start_s must be a finite non-negative value")
    if not math.isfinite(clip_duration_s) or clip_duration_s <= 0.0:
        raise ValueError("clip_duration_s must be a positive finite value")
    source_end_s = source_time_origin_s + source_duration_s
    if source_duration_s > 0.0 and (
        clip_start_s < source_time_origin_s - 1e-6
        or clip_start_s + clip_duration_s > source_end_s + 1e-6
    ):
        raise ValueError(
            f"Requested clip {clip_start_s:.6f}-"
            f"{clip_start_s + clip_duration_s:.6f}s exceeds source PTS range "
            f"{source_time_origin_s:.6f}-{source_end_s:.6f}s"
        )

    video_frame_rate = float(video_frame_rate)
    if not math.isfinite(video_frame_rate) or video_frame_rate <= 0.0:
        raise ValueError("video_frame_rate must be a positive finite value")
    display_refresh_rate = float(
        video_frame_rate
        if display_refresh_rate is None
        else display_refresh_rate
    )
    if not math.isfinite(display_refresh_rate) or display_refresh_rate <= 0.0:
        raise ValueError(
            "display_refresh_rate must be a positive finite value"
        )
    display_frame_period_s = 1.0 / display_refresh_rate
    requested_clip_duration_s = float(
        clip_duration_s
        if requested_clip_duration_s is None
        else requested_clip_duration_s
    )
    if (
        not math.isfinite(requested_clip_duration_s)
        or requested_clip_duration_s <= 0.0
    ):
        raise ValueError("requested_clip_duration_s must be positive and finite")
    if video_frame_count is None:
        video_frame_count = plan_frame_duration(
            clip_duration_s,
            video_frame_rate,
            minimum_frames=1,
        ).frame_count
    video_frame_count = int(video_frame_count)
    if video_frame_count <= 0:
        raise ValueError("video_frame_count must be positive")
    if refresh_cadence is None:
        refresh_cadence = plan_video_refresh_cadence(
            video_frame_count,
            video_frame_rate,
            display_refresh_rate,
        )
    elif (
        refresh_cadence.video_frame_count != video_frame_count
        or not math.isclose(
            refresh_cadence.video_frame_rate,
            video_frame_rate,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        or not math.isclose(
            refresh_cadence.display_refresh_rate,
            display_refresh_rate,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ):
        raise ValueError(
            "refresh_cadence does not match this clip and display"
        )
    scheduled_display_duration_s = (
        refresh_cadence.scheduled_display_duration_s
    )
    effective_video_frame_rate = (
        video_frame_count / scheduled_display_duration_s
    )
    video_buffer_bytes = int(video_buffer_bytes)
    if video_buffer_bytes <= 0:
        raise ValueError("video_buffer_bytes must be positive")
    rotation_degrees = float(stimulus_rotation_degrees)
    if not math.isfinite(rotation_degrees):
        raise ValueError("stimulus_rotation_degrees must be finite")

    video_size = (
        int(stream.get("width", 0) or 0),
        int(stream.get("height", 0) or 0),
    )
    if min(video_size) <= 0:
        raise ValueError("video stream width and height must be positive")
    native_size = tuple(native_target_size or tuple(win.size))
    subject_target_size = oriented_size(native_size, rotation_degrees)
    crop_bounds = center_crop_bounds(
        video_size,
        subject_target_size,
        alignment=2,
    )
    prepared_size = (
        crop_bounds[2] - crop_bounds[0],
        crop_bounds[3] - crop_bounds[1],
    )
    draw_size = compute_aspect_cover_size(subject_target_size, prepared_size)
    backend_used = "ffpyplayer"

    if stop_on_mouse_click and mouse is None:
        mouse = event.Mouse(win=win)
    if stop_on_mouse_click and mouse is not None:
        try:
            mouse.clickReset()
        except Exception:
            pass
        event.clearEvents(eventType="mouse")

    first_flip_ps = None
    first_flip_perf = None
    first_flip_requested_perf = None
    last_flip_perf = None
    previous_source_flip_perf = None
    end_perf = None
    end_requested_perf = None
    dropped_frames = 0
    aborted = False
    abort_reason = ""
    next_frame_publish_perf = 0.0
    frames_presented = 0
    source_frame_holds_completed = 0
    display_refreshes_presented = 0
    sync_records: List[Dict[str, Any]] = []
    expected_duration_s = requested_clip_duration_s
    actual_source_start_s = None
    actual_source_last_frame_s = None
    scheduled_video_slots_skipped = 0
    late_frame_count = 0
    maximum_frame_lateness_s = 0.0
    clip_offset_lateness_s = 0.0
    clip_offset_timing_error_s = 0.0
    clear_flip_submitted_perf_s = None
    maximum_source_frame_interval_s = 0.0
    maximum_boundary_timing_error_s = 0.0
    # The realized source boundary is the VBL nearest its ideal media time.
    # One millisecond of timestamp tolerance keeps a slightly noisy measured
    # refresh estimate from rejecting the correct adjacent VBL.
    timing_failure_threshold_s = (
        0.5 * display_frame_period_s + 0.001
    )
    # Submit just before the nearest-VBL decision boundary. The blocking swap
    # performs the quantization; this lead is only scheduling headroom.
    flip_submit_lead_s = 0.49 * display_frame_period_s
    frame_stream = None
    prepared_frame = None
    clear_flip_completed = False
    display_warmup_flips = 0
    startup_preload_frames = 0
    startup_average_preparation_fps = 0.0
    frame_fetch_total_s = 0.0
    frame_fetch_max_s = 0.0
    texture_upload_total_s = 0.0
    texture_upload_max_s = 0.0
    frame_draw_total_s = 0.0
    frame_draw_max_s = 0.0
    preview_copy_total_s = 0.0
    preview_copy_max_s = 0.0
    preview_copy_count = 0
    minimum_flip_submission_headroom_s = math.inf
    frame_timing_records: List[Dict[str, Any]] = []
    timing_validation: Dict[str, Any] = {
        "status": "FAIL",
        "rows": [],
    }
    glx_swap_interval = None
    glx_swap_interval_detail = "not queried"
    glx_sync_start = None
    glx_sync_start_detail = "not queried"
    glx_sync_end = None
    glx_sync_end_detail = "not queried"
    glx_sync_validation: Dict[str, Any] = {
        "status": "UNAVAILABLE",
    }

    def _poll_abort_reason():
        if allow_escape and event.getKeys(["escape"]):
            return "escape_pressed"
        if external_abort_checker is not None:
            try:
                external_abort = external_abort_checker()
            except Exception:
                external_abort = False
            if external_abort:
                return (
                    str(external_abort)
                    if isinstance(external_abort, str)
                    else "external_abort"
                )
        if stop_on_mouse_click and mouse is not None:
            try:
                if any(mouse.getPressed()):
                    return "mouse_click"
            except Exception:
                pass
        return ""

    def _mark_aborted(reason: str, *, level: str = "WARN") -> None:
        nonlocal aborted, abort_reason
        reason = str(reason or "external_abort")
        if not aborted:
            _log_message(
                msg_logger,
                level,
                f"video_abort trial_num={trial_num} file={video_file.name} "
                f"reason={reason}",
            )
        aborted = True
        abort_reason = reason

    def _wait_for_flip_submission(target_perf_s: float) -> None:
        """Sleep once until the safe submission window for an absolute VBL."""
        remaining_s = (
            float(target_perf_s)
            - flip_submit_lead_s
            - time.perf_counter()
        )
        if remaining_s > 0.0:
            time.sleep(remaining_s)

    try:
        frame_stream = BufferedVideoFrameStream(
            video_path=video_file,
            source_size=video_size,
            crop_bounds=crop_bounds,
            clip_start_s=clip_start_s,
            frame_count=video_frame_count,
            frame_rate=video_frame_rate,
            seek_timeout_s=seek_timeout_s,
            memory_budget_bytes=video_buffer_bytes,
        )
        startup_preload_frames = min(
            video_frame_count,
            frame_stream.layout.preload_chunks
            * frame_stream.layout.frames_per_chunk,
        )
        _log_message(
            msg_logger,
            "INFO",
            (
                f"video_preparation_start trial_num={trial_num} "
                f"file={video_file.name} "
                f"startup_preload_frames={startup_preload_frames} "
                f"chunk_frames={frame_stream.layout.frames_per_chunk} "
                f"preload_chunks={frame_stream.layout.preload_chunks}"
            ),
        )
        try:
            frame_stream.wait_until_ready(abort_checker=_poll_abort_reason)
        except VideoPreparationAborted as exc:
            _mark_aborted(exc.reason)
        if frame_stream.startup_wait_s > 0.0:
            startup_average_preparation_fps = (
                startup_preload_frames / frame_stream.startup_wait_s
            )

        if not aborted:
            if decoder_ready_callback is not None:
                decoder_ready_callback()
            blank_image = Image.new("RGB", prepared_size, (0, 0, 0))
            video_stim = visual.ImageStim(
                win,
                image=blank_image,
                units="pix",
                size=draw_size,
                pos=(0.0, 0.0),
                ori=rotation_degrees,
                interpolate=True,
                flipVert=True,
                autoLog=False,
            )
            # Exercise the exact texture-upload/draw/swap path while the
            # stimulus is still black. This moves lazy GL allocation and the
            # first post-idle swaps out of the measured clip, so frame zero is
            # not also the display pipeline's warm-up frame.
            upload_rgb_texture(
                video_stim,
                np.zeros(
                    (prepared_size[1], prepared_size[0], 3),
                    dtype=np.uint8,
                ),
            )
            if not enforce_window_vsync(win):
                raise RuntimeError(
                    "Could not reassert refresh-synchronized swaps on the "
                    "main PsychoPy window"
                )
            display_warmup_flips = 2
            for _ in range(display_warmup_flips):
                video_stim.draw()
                win.flip()
            glx_swap_interval, glx_swap_interval_detail = (
                query_glx_swap_interval()
            )
            buffer_mode = (
                "whole_clip"
                if frame_stream.layout.slot_count == 1
                else "three_chunk_ring"
            )
            _log_message(
                msg_logger,
                "INFO",
                (
                    f"video_playback file={video_file.name} "
                    f"source_path={video_file} clip_start_s={clip_start_s:.6f} "
                    f"clip_end_s={clip_start_s + clip_duration_s:.6f} "
                    f"source_time_origin_s={source_time_origin_s:.6f} "
                    f"video_size={video_size} crop_bounds={crop_bounds} "
                    f"prepared_size={prepared_size} "
                    f"native_target_size={native_size} "
                    f"subject_target_size={subject_target_size} "
                    f"rotation_deg={rotation_degrees:g} "
                    f"win_size={tuple(win.size)} draw_size={draw_size} "
                    f"video_frame_rate={video_frame_rate:.6f} "
                    f"video_frame_count={video_frame_count} "
                    f"display_refresh_rate={display_refresh_rate:.6f} "
                    f"nominal_refreshes_per_video_frame="
                    f"{refresh_cadence.nominal_refreshes_per_video_frame:.9f} "
                    f"refresh_hold_histogram="
                    f"{dict(refresh_cadence.refresh_count_histogram)} "
                    f"maximum_cadence_phase_error_s="
                    f"{refresh_cadence.maximum_absolute_phase_error_s:.9f} "
                    f"scheduled_display_duration_s="
                    f"{scheduled_display_duration_s:.6f} "
                    f"display_warmup_flips={display_warmup_flips} "
                    f"backend={backend_used} buffer_mode={buffer_mode} "
                    f"chunk_frames={frame_stream.layout.frames_per_chunk} "
                    f"buffer_bytes={frame_stream.layout.total_bytes} "
                    f"startup_wait_s={frame_stream.startup_wait_s:.6f} "
                    f"startup_preload_frames={startup_preload_frames} "
                    f"startup_average_preparation_fps="
                    f"{startup_average_preparation_fps:.3f} "
                    f"codec={stream.get('codec_name')} "
                    f"pix_fmt={stream.get('pix_fmt')}"
                ),
            )
            # Log setup/preload diagnostics before the critical sequence. The
            # logger remains buffered between frame boundaries, then the full
            # timing report is flushed by the task after the clip.
            _flush_message_logger(msg_logger)

            for expected_frame_index in range(video_frame_count):
                reason = _poll_abort_reason()
                if reason:
                    _mark_aborted(reason)
                    break
                frame_fetch_started_s = time.perf_counter()
                try:
                    prepared_frame = frame_stream.next_frame(
                        abort_checker=_poll_abort_reason,
                    )
                except VideoPreparationAborted as exc:
                    _mark_aborted(exc.reason)
                    break
                except VideoBufferUnderrun as exc:
                    _log_message(
                        msg_logger,
                        "ERROR",
                        f"video_buffer_underrun trial_num={trial_num} "
                        f"file={video_file.name} error={exc}",
                    )
                    raise RuntimeError(
                        "Prepared-video buffer underrun; playback stopped "
                        "without skipping a source frame"
                    ) from exc
                if prepared_frame is None:
                    raise RuntimeError(
                        f"ffpyplayer ended before frame "
                        f"{expected_frame_index}"
                    )
                if prepared_frame.frame_index != expected_frame_index:
                    raise RuntimeError(
                        f"prepared frame sequence jumped from "
                        f"{expected_frame_index} to "
                        f"{prepared_frame.frame_index}"
                    )
                frame_fetch_elapsed_s = (
                    time.perf_counter() - frame_fetch_started_s
                )
                frame_fetch_total_s += frame_fetch_elapsed_s
                frame_fetch_max_s = max(
                    frame_fetch_max_s,
                    frame_fetch_elapsed_s,
                )

                # The CPU view remains leased through the GL upload and the
                # optional preview copy. A chunk slot is released only by
                # the following next_frame() call.
                texture_upload_started_s = time.perf_counter()
                upload_rgb_texture(video_stim, prepared_frame.rgb)
                texture_upload_elapsed_s = (
                    time.perf_counter() - texture_upload_started_s
                )
                texture_upload_total_s += texture_upload_elapsed_s
                texture_upload_max_s = max(
                    texture_upload_max_s,
                    texture_upload_elapsed_s,
                )
                display_frame_index = int(prepared_frame.frame_index)
                current_source_time_s = float(
                    prepared_frame.source_pts_s
                )
                sync_edges = ()
                if sync_schedule is not None:
                    sync_edges = sync_schedule.edges_for_frame(
                        display_frame_index
                    )

                target_flip_perf = None
                if first_flip_perf is not None:
                    target_flip_perf = (
                        first_flip_perf
                        + expected_frame_index / video_frame_rate
                    )
                    _wait_for_flip_submission(target_flip_perf)

                flip_perf_capture: Dict[str, float] = {}
                win.callOnFlip(
                    _capture_perf_counter_on_flip,
                    flip_perf_capture,
                )
                for edge in sync_edges:
                    win.callOnFlip(
                        _set_gpio_level_on_flip,
                        sync_gpio_module,
                        sync_gpio_chip,
                        int(sync_pin),
                        int(edge.level),
                    )

                frame_draw_started_s = time.perf_counter()
                # The aspect-cover video is opaque and covers the complete
                # main framebuffer. Drawing the full-screen background beneath
                # it only doubles fill work on the Pi GPU.
                video_stim.draw()
                flip_requested_perf = time.perf_counter()
                frame_draw_elapsed_s = (
                    flip_requested_perf - frame_draw_started_s
                )
                frame_draw_total_s += frame_draw_elapsed_s
                frame_draw_max_s = max(
                    frame_draw_max_s,
                    frame_draw_elapsed_s,
                )
                if target_flip_perf is not None:
                    minimum_flip_submission_headroom_s = min(
                        minimum_flip_submission_headroom_s,
                        target_flip_perf - flip_requested_perf,
                    )
                flip_ps = win.flip()
                flip_return_perf = time.perf_counter()
                flip_perf = flip_perf_capture.get(
                    "actual_perf_s",
                    flip_return_perf,
                )

                if target_flip_perf is not None:
                    boundary_timing_error_s = (
                        flip_perf - target_flip_perf
                    )
                    maximum_boundary_timing_error_s = max(
                        maximum_boundary_timing_error_s,
                        abs(boundary_timing_error_s),
                    )
                    maximum_frame_lateness_s = max(
                        maximum_frame_lateness_s,
                        max(0.0, boundary_timing_error_s),
                    )
                    if (
                        abs(boundary_timing_error_s)
                        > timing_failure_threshold_s
                    ):
                        late_frame_count += 1
                        _log_message(
                            msg_logger,
                            "WARN",
                            f"video_frame_boundary_missed "
                            f"trial_num={trial_num} "
                            f"file={video_file.name} "
                            f"frame_index={display_frame_index} "
                            f"timing_error_s="
                            f"{boundary_timing_error_s:.6f} "
                            f"target_perf_s={target_flip_perf:.9f} "
                            f"actual_perf_s={flip_perf:.9f} "
                            "action=continue_absolute_schedule",
                        )

                if previous_source_flip_perf is not None:
                    maximum_source_frame_interval_s = max(
                        maximum_source_frame_interval_s,
                        flip_perf - previous_source_flip_perf,
                    )
                previous_source_flip_perf = flip_perf
                last_flip_perf = flip_perf
                frames_presented += 1
                source_frame_holds_completed = expected_frame_index
                display_refreshes_presented = (
                    refresh_cadence.refresh_boundaries[
                        expected_frame_index
                    ]
                )
                actual_source_last_frame_s = current_source_time_s
                for edge in sync_edges:
                    sync_records.append(
                        {
                            "level": int(edge.level),
                            "frame_index": int(edge.frame_index),
                            "interval_frames": edge.interval_frames,
                            "timestamp_perf_s": flip_perf,
                            "requested_timestamp_perf_s": (
                                flip_requested_perf
                            ),
                        }
                    )

                if first_flip_ps is None:
                    first_flip_ps = flip_ps
                    first_flip_perf = flip_perf
                    first_flip_requested_perf = flip_requested_perf
                    actual_source_start_s = current_source_time_s
                    glx_sync_start, glx_sync_start_detail = (
                        query_glx_sync_values()
                    )
                    if video_onset_callback is not None:
                        try:
                            video_onset_callback(first_flip_perf)
                        except Exception as exc:
                            _log_message(
                                msg_logger,
                                "WARN",
                                f"video_onset_callback_failed "
                                f"trial_num={trial_num} "
                                f"error={type(exc).__name__}: {exc}",
                            )

                frame_timing_records.append(
                    {
                        "source_frame_index": display_frame_index,
                        "source_media_time_s": current_source_time_s,
                        "expected_source_pts_s": (
                            clip_start_s
                            + display_frame_index / video_frame_rate
                        ),
                        "flip_requested_perf_s": flip_requested_perf,
                        "actual_flip_perf_s": flip_perf,
                    }
                )

                if frame_publisher is not None:
                    # The preview is best-effort and begins at its
                    # configured sampling interval. It cannot delay the
                    # main display's absolute source-frame schedule.
                    if (
                        frames_presented == 1
                        and frame_publish_interval_s > 0.0
                    ):
                        next_frame_publish_perf = (
                            flip_perf + frame_publish_interval_s
                        )
                    elif flip_perf >= next_frame_publish_perf:
                        preview_copy_started_s = time.perf_counter()
                        frame_publisher.publish_rgb(
                            prepared_frame.rgb,
                            source_frame_index=int(
                                round(
                                    (
                                        current_source_time_s
                                        - source_time_origin_s
                                    )
                                    * video_frame_rate
                                )
                            ),
                            source_media_time_s=current_source_time_s,
                            main_display_flip_perf_s=flip_perf,
                            trial_num=trial_num,
                        )
                        preview_copy_elapsed_s = (
                            time.perf_counter() - preview_copy_started_s
                        )
                        preview_copy_total_s += preview_copy_elapsed_s
                        preview_copy_max_s = max(
                            preview_copy_max_s,
                            preview_copy_elapsed_s,
                        )
                        preview_copy_count += 1
                        next_frame_publish_perf = (
                            flip_perf + frame_publish_interval_s
                        )
    finally:
        final_sync_edge = None
        clear_flip_perf_capture: Dict[str, float] = {}
        try:
            win.callOnFlip(
                _capture_perf_counter_on_flip,
                clear_flip_perf_capture,
            )
            if sync_schedule is not None:
                final_sync_edge = sync_schedule.mark_forced_low(
                    frames_presented
                )
                if final_sync_edge is not None:
                    win.callOnFlip(
                        _set_gpio_level_on_flip,
                        sync_gpio_module,
                        sync_gpio_chip,
                        int(sync_pin),
                        0,
                    )
            if bg_rect is not None:
                bg_rect.draw()

            requested_clear_perf = None
            if (
                first_flip_perf is not None
                and not aborted
                and frames_presented == video_frame_count
            ):
                requested_clear_perf = (
                    first_flip_perf
                    + video_frame_count / video_frame_rate
                )
                _wait_for_flip_submission(requested_clear_perf)

            clear_flip_submitted_perf_s = time.perf_counter()
            win.flip()
            clear_flip_return_perf = time.perf_counter()
            clear_flip_completed = True
            # The requested timestamp is when Python submitted the blocking
            # flip. Keep the predicted next-VBL target separate for timing
            # validation and the nominal scheduled endpoint.
            end_requested_perf = clear_flip_submitted_perf_s
            end_perf = clear_flip_perf_capture.get(
                "actual_perf_s",
                clear_flip_return_perf,
            )
            glx_sync_end, glx_sync_end_detail = query_glx_sync_values()
            if requested_clear_perf is not None:
                clip_offset_timing_error_s = (
                    end_perf - requested_clear_perf
                )
                maximum_boundary_timing_error_s = max(
                    maximum_boundary_timing_error_s,
                    abs(clip_offset_timing_error_s),
                )
                clip_offset_lateness_s = max(
                    0.0,
                    clip_offset_timing_error_s,
                )
                source_frame_holds_completed = video_frame_count
                display_refreshes_presented = (
                    refresh_cadence.total_refreshes
                )
            if final_sync_edge is not None:
                sync_records.append(
                    {
                        "level": 0,
                        "frame_index": int(final_sync_edge.frame_index),
                        "interval_frames": None,
                        "timestamp_perf_s": end_perf,
                        "requested_timestamp_perf_s": end_requested_perf,
                    }
                )
            if abs(clip_offset_timing_error_s) > timing_failure_threshold_s:
                late_frame_count += 1
                _log_message(
                    msg_logger,
                    "WARN",
                    f"video_offset_timing_error trial_num={trial_num} "
                    f"file={video_file.name} "
                    f"timing_error_s={clip_offset_timing_error_s:.6f} "
                    "action=record_actual_offset",
                )
        finally:
            if sync_schedule is not None and not clear_flip_completed:
                try:
                    _set_gpio_level_on_flip(
                        sync_gpio_module,
                        sync_gpio_chip,
                        int(sync_pin),
                        0,
                    )
                except Exception:
                    pass
            if frame_stream is not None:
                # Drop the final NumPy export before closing its SharedMemory.
                prepared_frame = None
                frame_stream.close()

            timing_validation = _summarize_video_frame_timing(
                frame_timing_records,
                clip_offset_perf_s=end_perf,
                expected_frame_count=video_frame_count,
                video_frame_rate=video_frame_rate,
                display_refresh_rate=display_refresh_rate,
                planned_refresh_counts=(
                    refresh_cadence.frame_refresh_counts
                ),
                timing_failure_threshold_s=timing_failure_threshold_s,
                aborted=aborted,
                source_frames_skipped=scheduled_video_slots_skipped,
            )
            if glx_sync_start is not None and glx_sync_end is not None:
                delta_msc = int(glx_sync_end["msc"]) - int(
                    glx_sync_start["msc"]
                )
                delta_sbc = int(glx_sync_end["sbc"]) - int(
                    glx_sync_start["sbc"]
                )
                delta_ust = int(glx_sync_end["ust"]) - int(
                    glx_sync_start["ust"]
                )
                expected_delta_msc = (
                    refresh_cadence.total_refreshes
                    if timing_validation["complete"]
                    else None
                )
                expected_delta_sbc = (
                    video_frame_count
                    if timing_validation["complete"]
                    else None
                )
                counter_rate_hz = (
                    delta_msc * 1_000_000.0 / delta_ust
                    if delta_ust > 0
                    else math.nan
                )
                counters_match = bool(
                    expected_delta_msc is not None
                    and delta_msc == expected_delta_msc
                    and delta_sbc == expected_delta_sbc
                )
                glx_sync_validation = {
                    "status": "PASS" if counters_match else "WARN",
                    "delta_msc": delta_msc,
                    "delta_sbc": delta_sbc,
                    "delta_ust": delta_ust,
                    "expected_delta_msc": expected_delta_msc,
                    "expected_delta_sbc": expected_delta_sbc,
                    "counter_rate_hz": counter_rate_hz,
                }
            else:
                glx_sync_validation = {
                    "status": "UNAVAILABLE",
                    "detail": (
                        f"start={glx_sync_start_detail}; "
                        f"end={glx_sync_end_detail}"
                    ),
                }

            # Persist flip records only after the refresh-critical sequence.
            # Their captured timestamps remain the actual callOnFlip times;
            # deferring file I/O prevents the first event row from delaying
            # the next source-frame boundary.
            if logger is not None and first_flip_perf is not None:
                logger.log_frame_flip(
                    trial_num=trial_num,
                    event="video_clip_start",
                    timestamp_perf_s=first_flip_perf,
                    requested_timestamp_perf_s=first_flip_requested_perf,
                    requested_duration=expected_duration_s,
                )
                if end_perf is not None:
                    logger.log_frame_flip(
                        trial_num=trial_num,
                        event="video_clip_end",
                        timestamp_perf_s=end_perf,
                        requested_timestamp_perf_s=end_requested_perf,
                    )
            if first_flip_perf is not None:
                _log_message(
                    msg_logger,
                    "INFO",
                    (
                        f"video_start trial_num={trial_num} "
                        f"file={video_file.name} source_path={video_file} "
                        f"onset_perf_s={first_flip_perf:.9f} "
                        f"requested_source_start_s={clip_start_s:.6f} "
                        f"actual_source_start_s={actual_source_start_s:.6f} "
                        f"configured_video_fps={video_frame_rate:.6f} "
                        f"scheduled_video_frames={video_frame_count} "
                        f"video_size={video_size} "
                        f"prepared_size={prepared_size} "
                        f"draw_size=({draw_size[0]:.1f},"
                        f"{draw_size[1]:.1f}) backend={backend_used}"
                    ),
                )

            validation_level = (
                "INFO"
                if timing_validation["status"] == "PASS"
                else "WARN"
            )
            _log_message(
                msg_logger,
                validation_level,
                (
                    f"video_frame_validation trial_num={trial_num} "
                    f"file={video_file.name} "
                    f"status={timing_validation['status']} "
                    f"expected_frames={video_frame_count} "
                    f"presented_frames={timing_validation['frame_count']} "
                    f"source_pts_contiguous="
                    f"{int(timing_validation['source_pts_contiguous'])} "
                    f"maximum_source_pts_error_ms="
                    f"{1000.0 * timing_validation['maximum_source_pts_error_s']:.3f} "
                    f"timing_misses="
                    f"{timing_validation['timing_miss_count']} "
                    f"cadence_mismatches="
                    f"{timing_validation['cadence_mismatch_count']} "
                    f"median_absolute_error_ms="
                    f"{1000.0 * timing_validation['median_absolute_timing_error_s']:.3f} "
                    f"p95_absolute_error_ms="
                    f"{1000.0 * timing_validation['p95_absolute_timing_error_s']:.3f} "
                    f"maximum_absolute_error_ms="
                    f"{1000.0 * timing_validation['maximum_absolute_timing_error_s']:.3f} "
                    f"allowed_absolute_error_ms="
                    f"{1000.0 * timing_failure_threshold_s:.3f} "
                    f"planned_hold_histogram="
                    f"{timing_validation['planned_hold_histogram']} "
                    f"realized_hold_histogram="
                    f"{timing_validation['realized_hold_histogram']} "
                    f"effective_video_fps="
                    f"{timing_validation['effective_frame_rate']:.6f}"
                ),
            )
            _log_message(
                msg_logger,
                (
                    "INFO"
                    if glx_sync_validation["status"] == "PASS"
                    else "WARN"
                ),
                (
                    f"video_main_vblank_validation trial_num={trial_num} "
                    f"file={video_file.name} "
                    f"status={glx_sync_validation['status']} "
                    f"swap_interval={glx_swap_interval!r} "
                    f"swap_interval_detail={glx_swap_interval_detail} "
                    f"delta_msc={glx_sync_validation.get('delta_msc', 'unavailable')} "
                    f"expected_delta_msc="
                    f"{glx_sync_validation.get('expected_delta_msc', 'unavailable')} "
                    f"delta_sbc={glx_sync_validation.get('delta_sbc', 'unavailable')} "
                    f"expected_delta_sbc="
                    f"{glx_sync_validation.get('expected_delta_sbc', 'unavailable')} "
                    f"counter_rate_hz="
                    f"{glx_sync_validation.get('counter_rate_hz', 'unavailable')} "
                    f"detail={glx_sync_validation.get('detail', 'GLX_OML_sync_control')}"
                ),
            )
            if timing_validation["nonuniform_cadence"]:
                integer_refresh_candidates = [
                    multiplier * video_frame_rate
                    for multiplier in range(1, 9)
                    if 40.0
                    <= multiplier * video_frame_rate
                    <= 144.0
                ]
                _log_message(
                    msg_logger,
                    "WARN",
                    (
                        f"video_motion_cadence trial_num={trial_num} "
                        f"file={video_file.name} uniform_holds=0 "
                        f"display_fps={display_refresh_rate:.6f} "
                        f"video_fps={video_frame_rate:.6f} "
                        f"planned_hold_histogram="
                        f"{timing_validation['planned_hold_histogram']} "
                        "interpretation=expected_pulldown_judder_in_fast_motion "
                        "preferred_integer_multiple_display_rates_hz="
                        + ",".join(
                            f"{value:.6f}"
                            for value in integer_refresh_candidates
                        )
                    ),
                )

            _log_message(
                msg_logger,
                "INFO",
                (
                    f"video_main_display_timing trial_num={trial_num} "
                    f"file={video_file.name} "
                    f"frames_presented={frames_presented} "
                    f"source_frame_holds_completed="
                    f"{source_frame_holds_completed} "
                    f"display_refreshes_presented="
                    f"{display_refreshes_presented} "
                    f"source_frames_skipped="
                    f"{scheduled_video_slots_skipped} "
                    f"missed_boundaries={late_frame_count} "
                    f"maximum_boundary_lateness_s="
                    f"{maximum_frame_lateness_s:.6f} "
                    f"maximum_boundary_timing_error_s="
                    f"{maximum_boundary_timing_error_s:.6f} "
                    f"maximum_source_frame_interval_s="
                    f"{maximum_source_frame_interval_s:.6f} "
                    f"offset_lateness_s={clip_offset_lateness_s:.6f} "
                    f"presentation_flips={frames_presented}"
                ),
            )
            if frames_presented > 0:
                _log_message(
                    msg_logger,
                    "INFO",
                    (
                        f"video_pipeline_performance trial_num={trial_num} "
                        f"file={video_file.name} "
                        f"frames={frames_presented} "
                        f"mean_fetch_ms="
                        f"{1000.0 * frame_fetch_total_s / frames_presented:.3f} "
                        f"max_fetch_ms={1000.0 * frame_fetch_max_s:.3f} "
                        f"mean_upload_ms="
                        f"{1000.0 * texture_upload_total_s / frames_presented:.3f} "
                        f"max_upload_ms={1000.0 * texture_upload_max_s:.3f} "
                        f"mean_draw_ms="
                        f"{1000.0 * frame_draw_total_s / frames_presented:.3f} "
                        f"max_draw_ms={1000.0 * frame_draw_max_s:.3f} "
                        f"preview_copies={preview_copy_count} "
                        f"mean_preview_copy_ms="
                        f"{1000.0 * preview_copy_total_s / max(1, preview_copy_count):.3f} "
                        f"max_preview_copy_ms="
                        f"{1000.0 * preview_copy_max_s:.3f} "
                        f"minimum_submission_headroom_ms="
                        f"{(1000.0 * minimum_flip_submission_headroom_s if math.isfinite(minimum_flip_submission_headroom_s) else math.nan):.3f}"
                    ),
                )
            if logger is not None:
                for record in sync_records:
                    is_on = int(record["level"]) == 1
                    pulse_duration = (
                        refresh_cadence.refreshes_for_source_frames(
                            int(record["frame_index"]),
                            int(sync_schedule.pulse_width_frames),
                        )
                        * display_frame_period_s
                        if is_on and sync_schedule is not None
                        else None
                    )
                    logger.log_signal(
                        trial_num=trial_num,
                        event=(
                            "video_sync_signal_on"
                            if is_on
                            else "video_sync_signal_off"
                        ),
                        timestamp_perf_s=float(
                            record["timestamp_perf_s"]
                        ),
                        requested_timestamp_perf_s=record.get(
                            "requested_timestamp_perf_s"
                        ),
                        requested_duration=(
                            pulse_duration if is_on else None
                        ),
                    )
                    if is_on:
                        _log_message(
                            msg_logger,
                            "INFO",
                            (
                                f"video_sync_pulse "
                                f"trial_num={trial_num} "
                                f"file={video_file.name} "
                                f"display_frame="
                                f"{record['frame_index']} "
                                f"interval_frames="
                                f"{record['interval_frames']} "
                                f"pin={int(sync_pin)}"
                            ),
                        )

    # A boundary timing miss changes a hold by a VBL but does not omit the
    # source frame. Keep source-frame drops distinct from cadence quality.
    dropped_frames = int(scheduled_video_slots_skipped)
    if first_flip_perf is not None:
        _log_message(
            msg_logger,
            "INFO",
            (
                f"video_end trial_num={trial_num} "
                f"file={video_file.name} source_path={video_file} "
                f"requested_source_end_s="
                f"{clip_start_s + clip_duration_s:.6f} "
                f"actual_source_last_frame_s="
                f"{actual_source_last_frame_s} "
                f"last_frame_on_perf_s={last_flip_perf} "
                f"clip_offset_perf_s={end_perf} "
                f"frames_presented={frames_presented} "
                f"display_refreshes_presented="
                f"{display_refreshes_presented} "
                f"dropped_frames={dropped_frames} "
                f"late_frames={late_frame_count} "
                f"aborted={int(aborted)} "
                f"abort_reason={abort_reason or 'none'} "
                f"backend={backend_used}"
            ),
        )
    _flush_message_logger(msg_logger)

    buffer_mode = (
        "whole_clip"
        if frame_stream is not None
        and frame_stream.layout.slot_count == 1
        else "three_chunk_ring"
    )
    return {
        "video_name": video_file.name,
        "video_path": video_file,
        "source_duration_s": source_duration_s,
        "source_time_origin_s": source_time_origin_s,
        "clip_start_s": clip_start_s,
        "clip_end_s": clip_start_s + clip_duration_s,
        "clip_duration_s": clip_duration_s,
        "expected_duration_s": expected_duration_s,
        "start_flip_psychopy_s": first_flip_ps,
        "start_flip_perf_s": first_flip_perf,
        "start_flip_requested_perf_s": first_flip_requested_perf,
        "end_time_perf_s": end_perf,
        "clip_offset_perf_s": end_perf,
        "end_requested_perf_s": end_requested_perf,
        "clear_flip_submitted_perf_s": clear_flip_submitted_perf_s,
        "requested_end_perf_s": (
            first_flip_perf + video_frame_count / video_frame_rate
            if first_flip_perf is not None
            else None
        ),
        "last_frame_end_perf_s": (
            end_perf if first_flip_perf is not None else None
        ),
        "last_frame_on_perf_s": last_flip_perf,
        "actual_source_start_s": actual_source_start_s,
        "actual_source_last_frame_s": actual_source_last_frame_s,
        "frames_presented": int(frames_presented),
        "source_frame_holds_completed": int(
            source_frame_holds_completed
        ),
        "display_refreshes_presented": int(
            display_refreshes_presented
        ),
        "displayed_duration_s": (
            end_perf - first_flip_perf
            if end_perf is not None and first_flip_perf is not None
            else None
        ),
        "dropped_frames": int(dropped_frames),
        "aborted": bool(aborted),
        "abort_reason": abort_reason,
        "video_size": tuple(video_size),
        "draw_size": tuple(draw_size),
        "native_target_size": tuple(native_size),
        "subject_target_size": tuple(subject_target_size),
        "stimulus_rotation_degrees": rotation_degrees,
        "scheduled_duration_s": clip_duration_s,
        "scheduled_display_duration_s": scheduled_display_duration_s,
        "backend_used": backend_used,
        "scheduled_video_slots_skipped": int(
            scheduled_video_slots_skipped
        ),
        "late_frame_count": int(late_frame_count),
        "maximum_frame_lateness_s": float(
            maximum_frame_lateness_s
        ),
        "clip_offset_lateness_s": float(clip_offset_lateness_s),
        "clip_offset_timing_error_s": float(
            clip_offset_timing_error_s
        ),
        "maximum_refresh_interval_s": float(
            maximum_source_frame_interval_s
        ),
        "maximum_boundary_timing_error_s": float(
            maximum_boundary_timing_error_s
        ),
        "long_video_intervals": int(late_frame_count),
        "timing_validation_status": timing_validation["status"],
        "timing_validation_miss_count": int(
            timing_validation.get("timing_miss_count", 0)
        ),
        "cadence_mismatch_count": int(
            timing_validation.get("cadence_mismatch_count", 0)
        ),
        "timing_error_p95_s": float(
            timing_validation.get(
                "p95_absolute_timing_error_s",
                math.nan,
            )
        ),
        "timing_error_maximum_s": float(
            timing_validation.get(
                "maximum_absolute_timing_error_s",
                math.nan,
            )
        ),
        "timing_failure_threshold_s": float(
            timing_failure_threshold_s
        ),
        "realized_refresh_hold_histogram": dict(
            timing_validation.get("realized_hold_histogram", {})
        ),
        "source_pts_contiguous": bool(
            timing_validation.get("source_pts_contiguous", False)
        ),
        "frame_timing_records": list(
            timing_validation.get("rows", [])
        ),
        "glx_swap_interval": glx_swap_interval,
        "glx_swap_interval_detail": glx_swap_interval_detail,
        "main_vblank_validation_status": glx_sync_validation["status"],
        "main_vblank_delta_msc": glx_sync_validation.get("delta_msc"),
        "main_vblank_expected_delta_msc": glx_sync_validation.get(
            "expected_delta_msc"
        ),
        "main_vblank_delta_sbc": glx_sync_validation.get("delta_sbc"),
        "main_vblank_expected_delta_sbc": glx_sync_validation.get(
            "expected_delta_sbc"
        ),
        "configured_video_frame_rate": video_frame_rate,
        "display_refresh_rate": display_refresh_rate,
        "nominal_refreshes_per_video_frame": (
            refresh_cadence.nominal_refreshes_per_video_frame
        ),
        "minimum_refreshes_per_video_frame": min(
            refresh_cadence.frame_refresh_counts
        ),
        "maximum_refreshes_per_video_frame": max(
            refresh_cadence.frame_refresh_counts
        ),
        "refresh_hold_histogram": dict(
            refresh_cadence.refresh_count_histogram
        ),
        "cadence_final_phase_error_s": (
            refresh_cadence.final_phase_error_s
        ),
        "cadence_maximum_absolute_phase_error_s": (
            refresh_cadence.maximum_absolute_phase_error_s
        ),
        "effective_video_frame_rate": effective_video_frame_rate,
        "scheduled_video_frame_count": video_frame_count,
        "sync_pulses": sum(
            1
            for record in sync_records
            if int(record["level"]) == 1
        ),
        "crop_bounds": tuple(crop_bounds),
        "prepared_frame_size": tuple(prepared_size),
        "video_buffer_bytes": (
            int(frame_stream.layout.total_bytes)
            if frame_stream is not None
            else 0
        ),
        "video_buffer_mode": buffer_mode,
        "video_preparation_wait_s": (
            float(frame_stream.startup_wait_s)
            if frame_stream is not None
            else 0.0
        ),
        "display_warmup_flips": int(display_warmup_flips),
        "startup_preload_frames": int(startup_preload_frames),
        "startup_average_preparation_fps": float(
            startup_average_preparation_fps
        ),
    }

def resolve_frame_rate(
    win: visual.Window,
    configured_fps: Optional[float] = None,
    *,
    msg_logger=None,
    context: str = "task",
) -> Tuple[float, float]:
    """Measure the main window and compare an optional configured override."""
    return resolve_window_frame_rate(
        win,
        configured_fps=configured_fps,
        msg_logger=msg_logger,
        context=context,
    )


def detect_frame_rate(win: visual.Window, msg_logger=None) -> Tuple[float, float]:
    """Detect the display refresh rate and return (fps, frameDur_s).

    Uses low-overhead blank flips and falls back to 60 Hz if unavailable.
    Logs the result to the optional message logger.
    """
    return resolve_frame_rate(win, msg_logger=msg_logger)


def make_fixation_cross(
    win: visual.Window,
    size: int = 40,
    color: Tuple[int, int, int] = (0, 0, 0),
    ori: float = 0.0,
):
    # If size is zero or negative, return None to indicate no fixation should be shown.
    if size is None or size <= 0:
        return None
    return visual.TextStim(
        win,
        text="+",
        height=size,
        color=rgb255_to_psychopy(color),
        colorSpace="rgb",
        ori=float(ori),
    )


def make_bg_rect(win: visual.Window, bg_rgb_255: Tuple[int, int, int]):
    """Create a full-window background rectangle in pixel units.

    This avoids duplicating rectangle construction logic across tasks.
    """
    return visual.Rect(
        win,
        width=win.size[0],
        height=win.size[1],
        fillColor=rgb255_to_psychopy(bg_rgb_255),
        fillColorSpace="rgb",
        lineColor=None,
        units="pix",
    )


def _to_pil_rgba(obj: Union[Image.Image, Path, np.ndarray]) -> Optional[Image.Image]:
    """Return a PIL RGBA image or None if conversion fails (raster inputs only)."""
    if isinstance(obj, Image.Image):
        return obj.convert("RGBA")
    if isinstance(obj, Path):
        if obj.suffix.lower() in VECTOR_EXTS:
            # handled elsewhere; this helper is raster-only
            return None
        try:
            with Image.open(obj) as im:
                return im.convert("RGBA").copy()
        except Exception:
            return None
    # assume array-like
    try:
        arr = np.asarray(obj)
    except Exception:
        return None
    if arr.dtype.kind == "f":
        arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    if arr.ndim == 2:
        return Image.fromarray(arr, mode="L").convert("RGBA")
    if arr.ndim == 3 and arr.shape[2] in (3, 4):
        mode = "RGBA" if arr.shape[2] == 4 else "RGB"
        return Image.fromarray(arr.astype(np.uint8), mode=mode).convert("RGBA")
    try:
        return Image.fromarray(arr).convert("RGBA")
    except Exception:
        return None


def _rasterize_svg_to_rgba(
    svg_path: Path, size_px: Tuple[int, int], bg_rgb_255: Optional[Tuple[int, int, int]] = None
) -> Image.Image:
    """
    Rasterize an SVG file to a PIL RGBA image using cairosvg.
    size_px: (width, height) in pixels.
    """
    try:
        import cairosvg  # type: ignore
    except Exception as e:
        raise ImportError(
            "SVG support requires 'cairosvg'. Install with: pip install cairosvg"
        ) from e

    if not size_px or len(size_px) != 2 or size_px[0] <= 0 or size_px[1] <= 0:
        raise ValueError("svg_size must be a (width, height) tuple of positive ints")

    # Read file bytes and ask cairosvg to render with a transparent background.
    # Using an explicit transparent color avoids backend defaults that may fill
    # the canvas with an opaque color on some systems.
    svg_bytes = svg_path.read_bytes()
    # If a background color is provided, request cairosvg to rasterize with
    # that opaque background. Otherwise request a transparent background.
    if bg_rgb_255 is not None:
        r, g, b = (int(c) for c in bg_rgb_255)
        bg_token = f"rgb({r},{g},{b})"
    else:
        bg_token = "rgba(0,0,0,0)"

    png_bytes = cairosvg.svg2png(
        bytestring=svg_bytes,
        output_width=int(size_px[0]),
        output_height=int(size_px[1]),
        background_color=bg_token,
    )

    im = Image.open(io.BytesIO(png_bytes)).convert("RGBA")

    # Save a debug copy so it's easy to verify that the rasterized SVG has
    # the expected background. Don't fail if logs/ can't be written; this
    # is purely diagnostic.
    # Save debug raster only when debugging is enabled
    try:
        if DEBUG:
            logs_dir = Path("logs")
            logs_dir.mkdir(parents=True, exist_ok=True)
            debug_path = logs_dir / f"debug_rasterized_svg_{svg_path.stem}.png"
            im.save(debug_path)
    except Exception:
        # ignore failures to write debug file
        pass

    return im


def rasterize_svg(
    svg_path: Path,
    size_px: Tuple[int, int],
    bg_rgb_255: Optional[Tuple[int, int, int]] = None,
) -> Image.Image:
    """Rasterize an SVG as authored, without overriding its fill or stroke."""
    return _rasterize_svg_to_rgba(
        Path(svg_path),
        size_px=size_px,
        bg_rgb_255=bg_rgb_255,
    )


def rasterize_svg_with_color(
    svg_path: Path,
    size_px: Tuple[int, int],
    color_rgb_255: Tuple[int, int, int],
    bg_rgb_255: Optional[Tuple[int, int, int]] = None,
    stroke_rgb_255: Optional[Tuple[int, int, int]] = None,
    stroke_width_px: Optional[float] = None,
    stroke_linejoin: Optional[str] = None,
    stroke_linecap: Optional[str] = None,
    flip: bool = False,
    outline_only: bool = False,
) -> Image.Image:
    """Rasterize an SVG and force its fill color to `color_rgb_255`.

    This reads the SVG text, injects a CSS rule to set fill color on common
    shape elements, preserves or sets stroke properties, then rasterizes via
    cairosvg to a PIL RGBA image.

    Parameters:
    - stroke_rgb_255: optional RGB tuple to set the stroke color. If None,
      defaults to black (0,0,0) per project convention.
    - stroke_width_px: optional stroke width in pixels. If None the SVG's
      original stroke-width is left unchanged.
    - stroke_linejoin: optional SVG/CSS line join override such as "round".
    - stroke_linecap: optional SVG/CSS line cap override such as "round".
    """
    try:
        import cairosvg  # type: ignore
    except Exception as e:
        raise ImportError(
            "SVG support requires 'cairosvg'. Install with: pip install cairosvg"
        ) from e

    if not size_px or len(size_px) != 2 or size_px[0] <= 0 or size_px[1] <= 0:
        raise ValueError("size_px must be a (width, height) tuple of positive ints")

    svg_text = svg_path.read_text(encoding="utf-8")

    r, g, b = (int(c) for c in color_rgb_255)

    # Determine stroke color (default to black if user didn't provide one).
    if stroke_rgb_255 is None:
        sr, sg, sb = (0, 0, 0)
    else:
        sr, sg, sb = (int(c) for c in stroke_rgb_255)

    # Build CSS rules. We always set fill to the requested color. For stroke
    # we set the requested color; if stroke_width_px is provided we also set
    # stroke-width. If stroke_width_px is None, the SVG's original stroke
    # width is preserved.
    if outline_only: 
        style_rules = ["fill:none !important", f"stroke:rgb({sr},{sg},{sb}) !important"]
    else: 
        style_rules = [f"fill:rgb({r},{g},{b}) !important", f"stroke:rgb({sr},{sg},{sb}) !important"]
    if stroke_width_px is not None:
        # ensure numeric formatting
        try:
            sw = float(stroke_width_px)
            style_rules.append(f"stroke-width:{sw}px !important")
        except Exception:
            # ignore invalid stroke width and leave it unspecified
            pass
    if stroke_linejoin is not None:
        join = str(stroke_linejoin).strip().lower()
        if join in {"miter", "round", "bevel"}:
            style_rules.append(f"stroke-linejoin:{join} !important")
    if stroke_linecap is not None:
        cap = str(stroke_linecap).strip().lower()
        if cap in {"butt", "round", "square"}:
            style_rules.append(f"stroke-linecap:{cap} !important")

    style_block = f"<style>path,rect,circle,polygon,ellipse,g,polyline{{{';'.join(style_rules)}}}</style>"

    # Find the end of the <svg ...> start tag to inject style immediately after it.
    idx = svg_text.find("<svg")
    if idx == -1:
        # fallback: just prepend the style
        mod_svg = style_block + svg_text
    else:
        # find the next '>' after the <svg
        gt = svg_text.find('>', idx)
        if gt == -1:
            mod_svg = style_block + svg_text
        else:
            mod_svg = svg_text[: gt + 1] + style_block + svg_text[gt + 1 :]

    # If a background color is provided, set the background token; else
    # request transparency.
    if bg_rgb_255 is not None:
        bg_token = f"rgb({int(bg_rgb_255[0])},{int(bg_rgb_255[1])},{int(bg_rgb_255[2])})"
    else:
        bg_token = "rgba(0,0,0,0)"

    png_bytes = cairosvg.svg2png(
        bytestring=mod_svg.encode("utf-8"),
        output_width=int(size_px[0]),
        output_height=int(size_px[1]),
        background_color=bg_token,
    )

    im = Image.open(io.BytesIO(png_bytes)).convert("RGBA")

    if flip:
        im = im.transpose(Image.FLIP_TOP_BOTTOM)
    
    try:
        if DEBUG:
            logs_dir = Path("logs")
            logs_dir.mkdir(parents=True, exist_ok=True)
            debug_path = logs_dir / f"debug_rasterized_svg_colored_{svg_path.stem}_{r}_{g}_{b}.png"
            im.save(debug_path)
    except Exception:
        pass
    return im


def load_color_palette(tsv_path: Path) -> Dict[int, Tuple[int, int, int]]:
    return _load_color_palette(tsv_path)


def split_background_from_palette(
    colors: Dict[int, Tuple[int, int, int]]
) -> Tuple[Tuple[int, int, int], Dict[int, Tuple[int, int, int]]]:
    """Split first TSV row (background) from subsequent color definitions.

    `colors` must preserve file row order (as produced by `load_color_palette`).
    Returns (bg_rgb, remaining_colors).
    """
    return _split_background_from_palette(colors)


def load_shape_definitions(tsv_path: Path) -> Dict[int, Path]:
    return _load_shape_definitions(tsv_path)


def load_image_assets(
    files: List[Path],
    raster_size: Optional[Tuple[int, int]] = None,
    bg_rgb_255: Optional[Tuple[int, int, int]] = None,
) -> Dict[Path, Image.Image]:
    """
    Preload a mixed list of raster and SVG images into PIL Images.
    - Raster images are converted to RGBA and optionally resized to raster_size.
    - SVG images are rasterized to svg_size (required if any SVGs are present).
    Returns: dict Path -> PIL.Image.Image (RGBA).
    """
    images: Dict[Path, Image.Image] = {}
    has_svg = any(f.suffix.lower() in VECTOR_EXTS for f in files)
    if has_svg and not raster_size:
        raise ValueError("SVG files detected but no image_size provided. Pass --image_size W H or set in config.")

    for p in files:
        ext = p.suffix.lower()
        if ext in VECTOR_EXTS:
            # Use raster_size as the target rasterization size for SVGs.
            im = _rasterize_svg_to_rgba(p, raster_size, bg_rgb_255=bg_rgb_255)  # type: ignore[arg-type]
            images[p] = im
        else:
            with Image.open(p) as im:
                im = im.convert("RGBA")
                if raster_size is not None:
                    im = im.resize((int(raster_size[0]), int(raster_size[1])), Image.LANCZOS)
                images[p] = im.copy()
    return images


def make_image_stim_from_array(
    win: visual.Window,
    img_obj,
    size: Optional[Tuple[int, int]] = None,
    bg_rgb_255: Optional[Tuple[int, int, int]] = None,
    ori: float = 0.0,
):
    """
    Create an ImageStim from PIL/Path/ndarray.

        - If bg_rgb_255 is given, we pre-composite RGBA onto that solid color (no transparency).
        - Otherwise we pass RGB + a 2D mask in the range [-1, 1] (PsychoPy convention),
            which is robust on macOS and elsewhere.
    """
    pil = _to_pil_rgba(img_obj)
    if pil is None and isinstance(img_obj, Path) and img_obj.suffix.lower() in VECTOR_EXTS:
        # If someone accidentally passes an SVG Path directly here, rasterize on the fly
        raise ValueError("SVG Paths must be pre-rasterized via load_image_assets (provide svg_size).")

    if pil is None:
        # last resort: let PsychoPy try whatever it is
        return visual.ImageStim(
            win, image=img_obj, size=size, units="pix", ori=float(ori)
        )

    # If a background color was requested earlier during rasterization,
    # the image may already be fully opaque. In that case, pass the RGB
    # image directly to PsychoPy (no mask) and avoid interpolation which
    # can introduce edge artifacts. Also prefer the image's native pixel
    # size when no explicit `size` is provided to avoid resampling.
    if size is None:
        size = pil.size

    # If a background was requested we probably flattened the SVG; check
    # whether the alpha channel is fully opaque and, if so, send RGB only.
    a = pil.getchannel("A")
    try:
        extrema = a.getextrema()
    except Exception:
        extrema = (255, 255)

    rgb = pil.convert("RGB")
    if extrema == (255, 255):
        # fully opaque: use RGB image (no mask) and no interpolation to
        # prevent border smoothing artifacts
        return visual.ImageStim(
            win,
            image=rgb,
            size=size,
            units="pix",
            interpolate=False,
            ori=float(ori),
        )

    # Otherwise preserve transparency via RGB + mask. PsychoPy expects masks in
    # the range [-1, 1] where -1 is fully transparent and +1 is fully opaque.
    # Convert the 8-bit alpha channel to that range to avoid unintended 50% opacity.
    mask01 = np.asarray(a, dtype=np.float32) / 255.0  # H x W, 0..1
    mask_pm1 = (mask01 * 2.0) - 1.0                  # H x W, -1..1
    return visual.ImageStim(
        win,
        image=rgb,
        mask=mask_pm1,
        size=size,
        units="pix",
        interpolate=False,
        ori=float(ori),
    )


def clear_events():
    event.clearEvents()


def make_onset_cue_stim(
    win: visual.Window,
    bg_rgb_255: Tuple[int, int, int],
    size_frac: float = 0.125,
    cells: int = 8,
    sigma_frac: float = 0.22,
    zero_threshold: int = 1,
    ori: float = 0.0,
):
    """Create a checkerboard onset cue ImageStim with a centered 2D Gaussian alpha mask.

    Parameters:
    - size_frac: fraction of min(window size) for cue edge length
    - cells: number of checkerboard cells per side
    - sigma_frac: sigma expressed as a fraction of cue width
    - zero_threshold: values <= this threshold in [0..255] are set to 0 in the mask
    """
    from PIL import Image, ImageDraw

    w = int(max(4, min(win.size) * float(size_frac)))
    if w <= 0:
        w = 400

    # Build checkerboard RGB on top of background color
    cb = Image.new("RGB", (w, w), color=(int(bg_rgb_255[0]), int(bg_rgb_255[1]), int(bg_rgb_255[2])))
    draw = ImageDraw.Draw(cb)
    cell = max(2, w // int(cells))
    for y in range(0, w, cell):
        for x in range(0, w, cell):
            xi = x // cell
            yi = y // cell
            fill = (0, 0, 0) if ((xi + yi) % 2 == 0) else (255, 255, 255)
            draw.rectangle([x, y, x + cell - 1, y + cell - 1], fill=fill)

    # 2D Gaussian mask centered at cue
    cx = (w - 1) / 2.0
    cy = (w - 1) / 2.0
    sigma = max(2.0, w * float(sigma_frac))
    yy, xx = np.mgrid[0:w, 0:w]
    gauss = np.exp(-0.5 * (((xx - cx) / sigma) ** 2 + ((yy - cy) / sigma) ** 2))
    mask_arr = np.clip(gauss * 255.0, 0, 255)
    mask_u8 = mask_arr.astype(np.uint8)
    if zero_threshold is not None and zero_threshold > 0:
        mask_u8[mask_u8 <= int(zero_threshold)] = 0
    cb.putalpha(Image.fromarray(mask_u8, mode="L"))

    # Convert to ImageStim (preserve alpha; ImageStim builder will provide proper mask)
    stim = make_image_stim_from_array(
        win, cb, size=(w, w), bg_rgb_255=None, ori=ori
    )
    try:
        stim.pos = (0, 0)
    except Exception:
        pass
    return stim


def _send_led_pulse_on_flip(chip, pin: int, duration_us: int):
    """GPIO pulse callback executed by PsychoPy at flip time.
    
    This is called by PsychoPy's callOnFlip mechanism, ensuring the GPIO write
    happens at the exact moment the frame is presented, minimizing latency.
    
    Args:
        chip: lgpio chip handle
        pin: GPIO pin number (BCM numbering)
        duration_us: pulse duration in microseconds
    """
    import lgpio
    
    # Set pin HIGH immediately (called at flip time by PsychoPy)
    lgpio.gpio_write(chip, pin, 1)
    
    # Use hardware-timed pulse to turn it off after duration
    result = lgpio.tx_pulse(chip, pin, 0, duration_us, 0, 1)
    
    # If pulse fails, at least turn the pin back off
    if result < 0:
        try:
            lgpio.gpio_write(chip, pin, 0)
        except Exception:
            pass
        raise RuntimeError(f"Hardware pulse failed with code {result} during flip callback")


def present_trial_with_persistent_dots(
    win: visual.Window,
    preloaded: Dict[Union[Path, Tuple[int, int]], Image.Image],
    trial_options: List[Union[Path, Tuple[int, int]]],
    positions: List[Tuple[float, float]],
    duration: float,
    choice_time: float,
    dot_size: int,
    dot_color: Tuple[int, int, int],
    bg_rect,
    fix,
    logger,
    trial_num: int,
    isi: float = 0.0,
    init_dot_color: Optional[Tuple[int, int, int]] = None,
    bg_rgb_255: Optional[Tuple[int, int, int]] = None,
    onset_cue: Optional[visual.ImageStim] = None,
    msg_logger=None,
    fps: Optional[float] = None,
    raspi: bool = False,
    pigpio_pi=None,
    raspi_pin: int = 18,
    sequential: bool = True,
    is_memory: bool = True,
    choice_hitbox_scale: float = 1.0,
    trial_meta: Optional[Dict[str, Any]] = None,
    experimenter_preview=None,
    external_abort_checker: Optional[Callable[[], bool]] = None,
    scene_main_size: Optional[Sequence[float]] = None,
    event_profile: str = "generic_afc",
    reward_levels: Optional[Sequence[int]] = None,
    pre_options_cue_image: Optional[Any] = None,
    pre_options_cue_duration: float = 0.0,
    pre_options_delay: float = 0.0,
    pre_options_cue_event: str = "match_cue_on",
    pre_options_delay_event: str = "delay_start",
    stimulus_rotation_degrees: float = 0.0,
    detect_pre_options_cue_touch: bool = False,
    on_pre_options_cue_touch: Optional[Callable[[], bool]] = None,
    pre_options_cue_touch_event: str = "match_cue_touch",
):
    """Present stimuli one at a time, leave faint dots at their locations,
    show all dots for `choice_time`, then clear.

    Returns a tuple (aborted: bool, choice_info: Optional[dict]).
    - aborted: True if the task should stop immediately.
    - choice_info: None if no choice was made during the choice period, or a dict with keys:
        - chosen_index: int (1-based index within the trial options)
        - chosen_pos: (x,y) psychopy pixel coords
        - choice_time_perf_s: perf_counter timestamp when the choice was made
        - reaction_time_s: time from choice_start to option_touch
        - touch_x / touch_y: screen coordinates of the option touch

    When ``detect_pre_options_cue_touch`` is enabled, a fresh press on the
    pre-options cue is logged once. ``on_pre_options_cue_touch`` runs after the
    cue's configured frame duration and after it has been cleared from the main
    display. A truthy callback return value aborts the trial.
    """
    from psychopy import core as _core

    dots: List[visual.Circle] = []
    dot_records: List[Dict[str, Any]] = []
    _visual = visual
    stim_sizes: List[Tuple[float, float]] = []
    # stims to potentially keep visible during the choice period when is_memory is False
    stims_for_choice: List[visual.ImageStim] = []
    stims_for_choice_preview: List[Dict[str, Any]] = []
    preview_reward_levels: Optional[list[int]] = None
    if reward_levels is not None:
        preview_reward_levels = [int(level) for level in reward_levels]
        if len(preview_reward_levels) != len(trial_options):
            raise ValueError("reward_levels must contain one value per trial option")
    # Establish frame timing
    if fps is None:
        fps, frame_dur = detect_frame_rate(win, msg_logger=msg_logger)
    else:
        fps = float(fps)
        frame_dur = 1.0 / float(fps)
    frame_timing_monitor = MainDisplayFrameTimingMonitor(win, frame_dur)
    if trial_meta is not None:
        trial_meta["_main_display_frame_timing_monitor"] = frame_timing_monitor

    rotation_degrees = float(stimulus_rotation_degrees)
    if not math.isfinite(rotation_degrees):
        raise ValueError("stimulus_rotation_degrees must be finite")

    cue_plan = plan_frame_duration(
        pre_options_cue_duration,
        fps,
        minimum_frames=1 if pre_options_cue_image is not None else 0,
    )
    delay_plan = plan_frame_duration(pre_options_delay, fps)
    isi_plan = plan_frame_duration(isi, fps)
    choice_plan = plan_frame_duration(choice_time, fps, minimum_frames=1)
    if sequential or is_memory:
        stim_plan = plan_frame_duration(duration, fps, minimum_frames=1)
    else:
        # Simultaneous non-memory trials expose the options for choice_time;
        # duration is intentionally a no-op in this presentation mode.
        stim_plan = plan_frame_duration(0.0, fps)

    def _set_initiation_time(perf_s: Optional[float] = None):
        if trial_meta is None or "initiation_time_s" in trial_meta:
            return
        perf_now = float(perf_s) if perf_s is not None else time.perf_counter()
        trial_meta["initiation_time_s"] = logger.seconds_since_session_start(perf_now)

    from psychopy import event as _event
    mouse = _event.Mouse(win=win)
    mouse_presses = MousePressTracker(mouse)
    trial_start_signal_armed_s: Optional[float] = None
    trial_start_signal_sent = False

    def _should_abort(notes: str) -> bool:
        if external_abort_checker is None:
            return False
        try:
            if external_abort_checker():
                _log_message(msg_logger, "WARN", notes)
                return True
        except Exception:
            pass
        return False

    def _frame_event_name(kind: str, option_idx: Optional[int] = None) -> str:
        if event_profile in {"active_foraging", "match2cue"}:
            if kind == "dot":
                if sequential:
                    return f"option_{int(option_idx)}_dot"
                return "options_dot"
            if kind == "stim":
                if sequential:
                    return f"option_{int(option_idx)}_on"
                return "options_on"
            if kind == "choice_start":
                return "choice_start"
            if kind == "gray":
                return (
                    "grey_inter_trial_interval"
                    if event_profile == "active_foraging"
                    else "gray_inter_trial_interval"
                )
            if kind == "onset_cue":
                return "onset_cue_on"
        generic_names = {
            "dot": "stimulus_dot",
            "stim": "stimulus_on",
            "choice_start": "choice_start",
            "gray": "gray_inter_trial_interval",
            "onset_cue": "onset_cue_on",
        }
        return generic_names[kind]

    def _interaction_event_name(kind: str) -> str:
        if kind == "cue_touch":
            return "cue_touch"
        return "option_touch"

    def _arm_trial_start_signal() -> bool:
        nonlocal trial_start_signal_armed_s, trial_start_signal_sent
        if trial_start_signal_sent or trial_start_signal_armed_s is not None:
            return True
        if not raspi or pigpio_pi is None:
            return True
        try:
            pulse_s = 0.25
            duration_us = int(pulse_s * 1_000_000)
            win.callOnFlip(_send_led_pulse_on_flip, pigpio_pi, raspi_pin, duration_us)
            trial_start_signal_armed_s = pulse_s
            _log_message(msg_logger, "INFO", f"raspi_pulse_registered trial_num={trial_num} duration_s={pulse_s:.6f}")
            return True
        except Exception as e:
            error_msg = f"trial_start_signal_registration_failed trial_num={trial_num} error={e}"
            _log_message(msg_logger, "ERROR", error_msg)
            return False

    def _commit_trial_start_signal(
        flip_perf_s: float,
        requested_perf_s: Optional[float] = None,
    ) -> None:
        nonlocal trial_start_signal_armed_s, trial_start_signal_sent
        if trial_start_signal_armed_s is None:
            return
        logger.log_signal(
            trial_num=trial_num,
            event="trial_start_signal_on",
            timestamp_perf_s=flip_perf_s,
            requested_timestamp_perf_s=requested_perf_s,
            requested_duration=trial_start_signal_armed_s,
        )
        trial_start_signal_sent = True
        trial_start_signal_armed_s = None

    def _preview_images(items: Sequence[Dict[str, Any]]) -> list[Dict[str, Any]]:
        images = []
        for item in items:
            image = {
                "image_payload": item.get("image_payload"),
                "pos": [float(item["pos"][0]), float(item["pos"][1])],
                "size": [float(item["size"][0]), float(item["size"][1])],
                "ori": float(item.get("ori", 0.0)),
            }
            images.append(image)
        return images

    def _preview_dots() -> list[Dict[str, Any]]:
        return [
            {
                "pos": [float(item["pos"][0]), float(item["pos"][1])],
                "radius": float(item["radius"]),
                "color": list(item["color"]),
            }
            for item in dot_records
        ]

    def _show_preview(images: Optional[Sequence[Dict[str, Any]]] = None) -> None:
        if experimenter_preview is None or bg_rgb_255 is None:
            return
        preview_items = list(images or [])
        fixation_size = None
        try:
            fixation_size = float(getattr(fix, "height"))
        except Exception:
            fixation_size = None
        experimenter_preview.show_static_scene(
            bg_rgb_255=bg_rgb_255,
            main_size=tuple(scene_main_size) if scene_main_size is not None else tuple(win.size),
            images=_preview_images(preview_items),
            dots=_preview_dots(),
            hit_boxes=build_reward_hit_boxes(
                preview_items,
                hitbox_scale=choice_hitbox_scale,
            ),
            fixation_size=fixation_size,
            fixation_color=(0, 0, 0),
            main_rotation_deg=rotation_degrees,
        )

    def _make_preview_image_entry(
        image_obj,
        stim_obj: visual.ImageStim,
        reward_level: Optional[int] = None,
    ) -> Dict[str, Any]:
        entry = {
            "image_payload": serialize_preview_image(image_obj),
            "pos": tuple(float(v) for v in getattr(stim_obj, "pos", (0.0, 0.0))),
            "size": tuple(float(v) for v in getattr(stim_obj, "size", (64.0, 64.0))),
            "ori": float(getattr(stim_obj, "ori", rotation_degrees)),
        }
        if reward_level is not None:
            entry["reward_level"] = int(reward_level)
        return entry

    if onset_cue is not None:
        try:
            onset_cue.pos = (0, 0)
            onset_cue.opacity = 1.0
            onset_cue.ori = rotation_degrees
        except Exception:
            pass
        bg_rect.draw()
        onset_cue.draw()
        if fix is not None:
            fix.draw()
        try:
            _show_preview(
                [
                    {
                        "image_payload": serialize_preview_image(getattr(onset_cue, "image", None)),
                        "pos": tuple(float(v) for v in getattr(onset_cue, "pos", (0.0, 0.0))),
                        "size": tuple(float(v) for v in getattr(onset_cue, "size", (200.0, 200.0))),
                        "ori": float(getattr(onset_cue, "ori", rotation_degrees)),
                    }
                ]
            )
        except Exception:
            pass
        if not _arm_trial_start_signal():
            return True, None
        # Arm before the vsync-blocked flip so even a press and release that
        # both arrive during the flip remain observable through click timing.
        mouse_presses.reset()
        oc_timing = flip_with_timestamps(win)
        oc_perf = oc_timing.actual_perf_s
        _commit_trial_start_signal(oc_perf, oc_timing.requested_perf_s)
        logger.log_frame_flip(
            trial_num=trial_num,
            event=_frame_event_name("onset_cue"),
            timestamp_perf_s=oc_perf,
            requested_timestamp_perf_s=oc_timing.requested_perf_s,
        )
        while True:
            if _event.getKeys(["escape"]):
                _log_message(msg_logger, "WARN", f"escape_pressed trial_num={trial_num} during_onset_cue=1")
                return True, None
            if _should_abort("experimenter_exit_during_onset_cue"):
                return True, None

            touch_sample = mouse_presses.poll()
            click_pos = touch_sample.position
            if touch_sample.active:
                try:
                    oc_w, oc_h = onset_cue.size
                except Exception:
                    oc_w, oc_h = (200, 200)
                oc_x, oc_y = getattr(onset_cue, "pos", (0, 0))
                if abs(click_pos[0] - oc_x) <= oc_w / 2.0 and abs(click_pos[1] - oc_y) <= oc_h / 2.0:
                    click_perf = time.perf_counter()
                    _set_initiation_time(click_perf)
                    logger.log_interaction(
                        trial_num=trial_num,
                        event=_interaction_event_name("cue_touch"),
                        timestamp_perf_s=click_perf,
                    )
                    _show_preview([])
                    break

            _core.wait(0.01)

    cue_frames = cue_plan.frame_count
    cue_s = cue_plan.scheduled_s
    delay_frames = delay_plan.frame_count
    delay_s = delay_plan.scheduled_s
    if pre_options_cue_image is not None:
        cue_stim = make_image_stim_from_array(
            win,
            pre_options_cue_image,
            size=None,
            bg_rgb_255=bg_rgb_255,
            ori=rotation_degrees,
        )
        cue_stim.pos = (0.0, 0.0)
        cue_preview = [
            {
                "image_payload": serialize_preview_image(pre_options_cue_image),
                "pos": [0.0, 0.0],
                "size": [float(cue_stim.size[0]), float(cue_stim.size[1])],
                "ori": rotation_degrees,
            }
        ]
        _show_preview(cue_preview)
        if not _arm_trial_start_signal():
            return True, None
        pre_options_cue_touched = False
        pre_options_cue_touch_armed = True
        cue_touch_target = None
        clear_timing = None
        if detect_pre_options_cue_touch:
            # A held checkerboard-initiation press must be released before the
            # matching cue can accept a touch.
            pre_options_cue_touch_armed = not mouse_presses.reset()
            cue_touch_target = visual.Rect(
                win,
                width=max(1.0, float(cue_stim.size[0]) * float(choice_hitbox_scale)),
                height=max(1.0, float(cue_stim.size[1]) * float(choice_hitbox_scale)),
                pos=cue_stim.pos,
                units="pix",
                fillColor=None,
                lineColor=None,
                opacity=0.0,
                ori=rotation_degrees,
            )

        def _poll_pre_options_cue_touch() -> None:
            nonlocal pre_options_cue_touched, pre_options_cue_touch_armed
            if not detect_pre_options_cue_touch or pre_options_cue_touched:
                return
            touch_sample = mouse_presses.poll()
            pre_options_cue_touch_armed, touch_is_eligible = (
                advance_release_armed_touch_gate(
                    pre_options_cue_touch_armed,
                    touch_sample,
                )
            )
            if not touch_is_eligible:
                return
            click_pos = touch_sample.position
            try:
                cue_contains_touch = bool(cue_touch_target.contains(click_pos))
            except Exception:
                cue_w, cue_h = cue_stim.size
                cue_x, cue_y = cue_stim.pos
                cue_contains_touch = bool(
                    abs(click_pos[0] - cue_x)
                    <= (cue_w * float(choice_hitbox_scale)) / 2.0
                    and abs(click_pos[1] - cue_y)
                    <= (cue_h * float(choice_hitbox_scale)) / 2.0
                )
            if touch_sample.press_started:
                _log_message(
                    msg_logger,
                    "INFO",
                    (
                        f"match_cue_touch_attempt trial_num={trial_num} "
                        f"click_xy=({click_pos[0]:.1f},{click_pos[1]:.1f}) "
                        f"matched={int(cue_contains_touch)}"
                    ),
                )
            if not cue_contains_touch:
                return
            touch_perf = time.perf_counter()
            pre_options_cue_touched = True
            logger.log_interaction(
                trial_num=trial_num,
                event=pre_options_cue_touch_event,
                timestamp_perf_s=touch_perf,
            )
            if trial_meta is not None:
                trial_meta["match_cue_touched"] = True
                trial_meta["match_cue_touch_perf_s"] = touch_perf

        first_flip = True
        with frame_timing_monitor.continuous_sequence():
            for _ in range(cue_frames):
                if _event.getKeys(["escape"]):
                    _log_message(msg_logger, "WARN", f"escape_pressed trial_num={trial_num} during_match_cue=1")
                    return True, None
                if _should_abort("experimenter_exit_during_match_cue"):
                    return True, None
                bg_rect.draw()
                cue_stim.draw()
                if fix is not None:
                    fix.draw()
                flip_timing = flip_with_timestamps(win)
                if first_flip:
                    cue_perf = flip_timing.actual_perf_s
                    _commit_trial_start_signal(
                        cue_perf,
                        flip_timing.requested_perf_s,
                    )
                    _set_initiation_time(cue_perf)
                    logger.log_frame_flip(
                        trial_num=trial_num,
                        event=pre_options_cue_event,
                        timestamp_perf_s=cue_perf,
                        requested_timestamp_perf_s=flip_timing.requested_perf_s,
                        requested_duration=cue_plan.requested_s,
                    )
                    first_flip = False

                _poll_pre_options_cue_touch()

        if detect_pre_options_cue_touch:
            # Close the match-cue touch window on its offset flip, then poll
            # once more so a short tap during the final displayed frame is not
            # lost. Reward delivery stays outside the frame-counted cue loop.
            _show_preview([])
            bg_rect.draw()
            if fix is not None:
                fix.draw()
            clear_timing = flip_with_timestamps(win)
            if trial_meta is not None:
                trial_meta["match_cue_clear_flip_perf_s"] = (
                    clear_timing.actual_perf_s
                )
                trial_meta["match_cue_clear_flip_requested_perf_s"] = (
                    clear_timing.requested_perf_s
                )
            _log_message(
                msg_logger,
                "INFO",
                (
                    f"match_cue_cleared trial_num={trial_num} "
                    f"timestamp_perf_s={clear_timing.actual_perf_s:.9f} "
                    "reason=match_cue_touch_window_end"
                ),
            )
            _poll_pre_options_cue_touch()

        match_cue_reward_callback_ran = bool(
            pre_options_cue_touched and on_pre_options_cue_touch is not None
        )
        if match_cue_reward_callback_ran:
            if on_pre_options_cue_touch():
                return True, None

        delay_frames_remaining = delay_frames
        delay_already_started = False
        if (
            clear_timing is not None
            and not match_cue_reward_callback_ran
            and delay_frames_remaining > 0
        ):
            # When no match-cue reward runs, the touch-window closing flip is
            # also the first configured delay frame. Do not add an extra frame
            # merely because touch detection was enabled.
            logger.log_frame_flip(
                trial_num=trial_num,
                event=pre_options_delay_event,
                timestamp_perf_s=clear_timing.actual_perf_s,
                requested_timestamp_perf_s=clear_timing.requested_perf_s,
                requested_duration=delay_plan.requested_s,
            )
            delay_frames_remaining -= 1
            delay_already_started = True

        if delay_frames_remaining > 0:
            _show_preview([])
            first_flip = not delay_already_started
            with frame_timing_monitor.continuous_sequence():
                for _ in range(delay_frames_remaining):
                    if _event.getKeys(["escape"]):
                        _log_message(msg_logger, "WARN", f"escape_pressed trial_num={trial_num} during_match_delay=1")
                        return True, None
                    if _should_abort("experimenter_exit_during_match_delay"):
                        return True, None
                    bg_rect.draw()
                    if fix is not None:
                        fix.draw()
                    flip_timing = flip_with_timestamps(win)
                    if first_flip:
                        delay_perf = flip_timing.actual_perf_s
                        logger.log_frame_flip(
                            trial_num=trial_num,
                            event=pre_options_delay_event,
                            timestamp_perf_s=delay_perf,
                            requested_timestamp_perf_s=flip_timing.requested_perf_s,
                            requested_duration=delay_plan.requested_s,
                        )
                        first_flip = False

    # Quantize durations to frames and log rounding in message logger.
    stim_frames, stim_s = stim_plan.frame_count, stim_plan.scheduled_s
    isi_frames, isi_s = isi_plan.frame_count, isi_plan.scheduled_s
    choice_s = choice_plan.scheduled_s
    _log_message(
        msg_logger,
        "INFO",
        (
            f"timing_quantization trial_num={trial_num} "
            f"stim_duration={duration:.6f}s-> {stim_frames}fr({stim_s:.6f}s) "
            f"isi={isi:.6f}s-> {isi_frames}fr({isi_s:.6f}s) "
            f"choice_time={choice_time:.6f}s-> {choice_plan.frame_count}fr({choice_s:.6f}s) "
            f"pre_options_cue={float(pre_options_cue_duration):.6f}s-> {cue_frames}fr({cue_s:.6f}s) "
            f"pre_options_delay={float(pre_options_delay):.6f}s-> {delay_frames}fr({delay_s:.6f}s)"
        ),
    )

    chosen_info = None
    click_registered = False
    click_perf_capture = None
    click_meta = None
    poll_interval_s = 0.002
    touch_acquire_window_s = 0.050
    choice_started = False
    choice_flip = None
    choice_perf = None
    choice_window_s = float(choice_s)
    choice_deadline = None
    choice_input_armed = False
    pos_list = list(positions)
    stims: List[visual.ImageStim] = []
    names: List[str] = []
    preview_images: List[Dict[str, Any]] = []
    choice_hit_targets: List[visual.Rect] = []
 
    def _build_choice_hit_targets():
        if choice_hit_targets:
            return
        if len(stim_sizes) < len(pos_list):
            return
        for ppos, stim_size in zip(pos_list, stim_sizes):
            try:
                w = max(1.0, float(stim_size[0]) * float(choice_hitbox_scale))
                h = max(1.0, float(stim_size[1]) * float(choice_hitbox_scale))
            except Exception:
                w, h = (64.0, 64.0)
            target = visual.Rect(
                win,
                width=w,
                height=h,
                pos=ppos,
                units="pix",
                fillColor=None,
                lineColor=None,
                opacity=0.0,
                ori=rotation_degrees,
            )
            choice_hit_targets.append(target)

    def _match_choice_target(click_pos: Tuple[float, float]) -> Optional[int]:
        _build_choice_hit_targets()
        for i, target in enumerate(choice_hit_targets, start=1):
            try:
                if target.contains(click_pos):
                    return i
            except Exception:
                pass
        return None

    def _acquire_choice_target(
        click_pos: Tuple[float, float],
        chosen_idx: Optional[int],
        touch_onset_perf: float,
    ) -> Tuple[Tuple[float, float], Optional[int]]:
        if chosen_idx is not None or choice_deadline is None:
            return click_pos, chosen_idx

        acquire_deadline = min(choice_deadline, touch_onset_perf + touch_acquire_window_s)
        while time.perf_counter() < acquire_deadline:
            _event.getKeys([])
            click_pos = mouse.getPos()
            chosen_idx = _match_choice_target(click_pos)
            if chosen_idx is not None:
                break
            remaining_acquire = acquire_deadline - time.perf_counter()
            if remaining_acquire > 0:
                _core.wait(min(poll_interval_s, remaining_acquire))
        return click_pos, chosen_idx

    def _log_choice_touch_attempt(click_pos: Tuple[float, float], chosen_idx: Optional[int], origin: str) -> None:
        if msg_logger is None:
            return
        try:
            msg_logger.log(
                "INFO",
                (
                    f"choice_touch_attempt trial_num={trial_num} "
                    f"click_xy=({click_pos[0]:.1f},{click_pos[1]:.1f}) "
                    f"matched_idx={chosen_idx} origin={origin}"
                ),
            )
        except Exception:
            pass

    def _commit_choice(chosen_idx: int, click_pos: Tuple[float, float], click_perf: float) -> None:
        nonlocal click_registered, click_perf_capture, click_meta, chosen_info
        click_perf_capture = float(click_perf)
        chosen_info = {
            "chosen_index": int(chosen_idx),
            "chosen_pos": tuple(pos_list[chosen_idx - 1]),
            "choice_start_perf_s": float(choice_perf),
            "choice_time_perf_s": float(click_perf_capture),
            "reaction_time_s": float(click_perf_capture - choice_perf),
            "touch_x": float(click_pos[0]),
            "touch_y": float(click_pos[1]),
        }
        logger.log_interaction(
            trial_num=trial_num,
            event=_interaction_event_name("option_touch"),
            timestamp_perf_s=click_perf_capture,
        )
        click_meta = {"idx": chosen_idx}
        click_registered = True

    def _arm_choice_input() -> None:
        nonlocal choice_input_armed
        if choice_input_armed:
            return
        mouse_presses.reset()
        choice_input_armed = True

    def _start_choice_window(
        start_flip_ps,
        start_perf: float,
        requested_perf: Optional[float] = None,
    ) -> None:
        nonlocal choice_started, choice_flip, choice_perf, choice_window_s, choice_deadline
        if choice_started:
            return
        if not choice_input_armed:
            # Defensive fallback for future presentation branches. Existing
            # paths arm immediately before the response-opening flip.
            _arm_choice_input()
        choice_started = True
        choice_flip = start_flip_ps
        choice_perf = float(start_perf)
        choice_window_s = max(0.0, float(choice_s))
        choice_deadline = choice_perf + choice_window_s
        start_touch = mouse_presses.poll()
        start_click_pos = start_touch.position
        logger.log_frame_flip(
            trial_num=trial_num,
            event=_frame_event_name("choice_start"),
            timestamp_perf_s=choice_perf,
            requested_timestamp_perf_s=requested_perf,
            requested_duration=choice_plan.requested_s,
        )

        if start_touch.active and choice_perf is not None:
            touch_onset_perf = time.perf_counter()
            chosen_idx = _match_choice_target(start_click_pos)
            start_click_pos_acquired, chosen_idx = _acquire_choice_target(
                start_click_pos,
                chosen_idx,
                touch_onset_perf,
            )
            start_origin = (
                "choice_start_buffered_touch"
                if start_touch.buffered_press and not start_touch.down
                else "choice_start_active_touch"
            )
            _log_choice_touch_attempt(start_click_pos_acquired, chosen_idx, origin=start_origin)
            if chosen_idx is not None:
                _commit_choice(chosen_idx, start_click_pos_acquired, touch_onset_perf)

    def _abort_from_input(reason: str) -> bool:
        if _event.getKeys(["escape"]):
            _log_message(msg_logger, "WARN", f"escape_pressed trial_num={trial_num} reason={reason}")
            return True
        return _should_abort(reason)

    def _poll_choice_until(deadline_perf: float) -> bool:
        nonlocal click_registered, click_perf_capture, click_meta, chosen_info
        while time.perf_counter() < deadline_perf and not click_registered:
            if _abort_from_input("experimenter_exit_during_choice"):
                return True

            touch_sample = mouse_presses.poll()
            click_pos = touch_sample.position
            touch_started = touch_sample.press_started

            if touch_sample.active and choice_perf is not None:
                touch_onset_perf = time.perf_counter()
                chosen_idx = _match_choice_target(click_pos)

                if chosen_idx is None and touch_started:
                    click_pos, chosen_idx = _acquire_choice_target(click_pos, chosen_idx, touch_onset_perf)

                if touch_started:
                    _log_choice_touch_attempt(click_pos, chosen_idx, origin="touch_start")

                if chosen_idx is not None:
                    _commit_choice(chosen_idx, click_pos, touch_onset_perf)
                    break

            remaining = deadline_perf - time.perf_counter()
            if remaining > 0:
                _core.wait(min(poll_interval_s, remaining))
        return False

    def _choice_transition_request_deadline() -> Optional[float]:
        if choice_deadline is None:
            return None
        # Submit the clearing frame before the midpoint between the two
        # surrounding refreshes so it realizes on the refresh nearest the
        # requested end rather than one refresh after it.
        return max(float(choice_perf), choice_deadline - (frame_dur / 2.0))

    def _record_gray_flip(
        perf_s: float,
        requested_perf_s: Optional[float] = None,
    ) -> None:
        if trial_meta is None:
            return
        trial_meta["gray_flip_perf_s"] = float(perf_s)
        if requested_perf_s is not None:
            trial_meta["gray_flip_requested_perf_s"] = float(requested_perf_s)

    def _build_stimulus(p, pos):
        if isinstance(p, tuple) and len(p) == 2:
            sid, cid = p
            name = f"shape{sid}_color{cid}"
            pil_img = preloaded.get((sid, cid))
            if pil_img is None:
                pil_img = preloaded.get(p)
        else:
            name = getattr(p, "name", str(p))
            pil_img = preloaded[p]
        stim = make_image_stim_from_array(
            win,
            pil_img,
            size=None,
            bg_rgb_255=bg_rgb_255,
            ori=rotation_degrees,
        )
        stim.pos = pos
        return name, pil_img, stim

    def _make_dot(pos, color_rgb):
        dot = _visual.Circle(
            win,
            radius=dot_size / 2.0,
            fillColor=rgb255_to_psychopy(color_rgb),
            fillColorSpace="rgb",
            lineColor=None,
            units="pix",
        )
        dot.pos = pos
        return dot

    if sequential:
        for idx, (p, pos) in enumerate(zip(trial_options, positions), start=1):
            name, pil_img, stim = _build_stimulus(p, pos)
            stims_for_choice.append(stim)
            reward_level = (
                preview_reward_levels[idx - 1]
                if preview_reward_levels is not None
                else None
            )
            stims_for_choice_preview.append(
                _make_preview_image_entry(pil_img, stim, reward_level)
            )
            try:
                stim_sizes.append(tuple(stim.size))
            except Exception:
                stim_sizes.append((0.0, 0.0))

            cue_dot = None
            if isi_frames > 0:
                cue_color = init_dot_color if init_dot_color is not None else dot_color
                cue_dot = _make_dot(pos, cue_color)
                dots.append(cue_dot)
                dot_records.append({"pos": tuple(pos), "radius": float(dot_size) / 2.0, "color": tuple(cue_color)})
                first_flip = True
                _show_preview([])
                if not _arm_trial_start_signal():
                    return True, None
                with frame_timing_monitor.continuous_sequence():
                    for _ in range(isi_frames):
                        if _abort_from_input("experimenter_exit_during_isi"):
                            return True, None
                        bg_rect.draw()
                        for d in dots:
                            d.draw()
                        if fix is not None:
                            fix.draw()
                        dot_timing = flip_with_timestamps(win)
                        if first_flip:
                            dot_perf = dot_timing.actual_perf_s
                            _commit_trial_start_signal(
                                dot_perf,
                                dot_timing.requested_perf_s,
                            )
                            _set_initiation_time(dot_perf)
                            logger.log_frame_flip(
                                trial_num=trial_num,
                                event=_frame_event_name("dot", idx),
                                timestamp_perf_s=dot_perf,
                                requested_timestamp_perf_s=dot_timing.requested_perf_s,
                                requested_duration=isi_plan.requested_s,
                            )
                            first_flip = False

            first_flip = True
            current_preview_image = [
                _make_preview_image_entry(pil_img, stim, reward_level)
            ]
            _show_preview(current_preview_image)
            if not _arm_trial_start_signal():
                return True, None
            with frame_timing_monitor.continuous_sequence():
                for _ in range(stim_frames):
                    if _abort_from_input("experimenter_exit_during_stimulus"):
                        return True, None
                    bg_rect.draw()
                    for d in dots:
                        d.draw()
                    stim.draw()
                    if fix is not None:
                        fix.draw()
                    stim_timing = flip_with_timestamps(win)
                    flip_ps = stim_timing.psychopy_s
                    if first_flip:
                        flip_perf = stim_timing.actual_perf_s
                        _commit_trial_start_signal(
                            flip_perf,
                            stim_timing.requested_perf_s,
                        )
                        _set_initiation_time(flip_perf)
                        logger.log_frame_flip(
                            trial_num=trial_num,
                            event=_frame_event_name("stim", idx),
                            timestamp_perf_s=flip_perf,
                            requested_timestamp_perf_s=stim_timing.requested_perf_s,
                            requested_duration=stim_plan.requested_s,
                        )
                        first_flip = False

            if is_memory:
                if cue_dot is None:
                    cue_dot = _make_dot(pos, dot_color)
                    dots.append(cue_dot)
                    dot_records.append({"pos": tuple(pos), "radius": float(dot_size) / 2.0, "color": tuple(dot_color)})
                else:
                    cue_dot.fillColor = rgb255_to_psychopy(dot_color)
                    cue_dot.fillColorSpace = "rgb"
                    dot_records[-1]["color"] = tuple(dot_color)
            elif cue_dot is not None and dots:
                dots.pop()
                dot_records.pop()

            if (not is_memory) and idx == len(trial_options):
                bg_rect.draw()
                for s in stims_for_choice:
                    s.draw()
                if fix is not None:
                    fix.draw()
                _arm_choice_input()
                off_timing = flip_with_timestamps(win)
                off_flip = off_timing.psychopy_s
                off_perf = off_timing.actual_perf_s
                _build_choice_hit_targets()
                _start_choice_window(
                    off_flip,
                    off_perf,
                    off_timing.requested_perf_s,
                )
                _show_preview(stims_for_choice_preview)

    else:
        for item_index, (p, pos) in enumerate(zip(trial_options, positions)):
            name, pil_img, stim = _build_stimulus(p, pos)
            stims.append(stim)
            names.append(name)
            reward_level = (
                preview_reward_levels[item_index]
                if preview_reward_levels is not None
                else None
            )
            preview_images.append(
                _make_preview_image_entry(pil_img, stim, reward_level)
            )
            try:
                stim_sizes.append(tuple(stim.size))
            except Exception:
                stim_sizes.append((0.0, 0.0))

        if isi_frames > 0:
            cue_color = init_dot_color if init_dot_color is not None else dot_color
            for pos in positions:
                dot = _make_dot(pos, cue_color)
                dots.append(dot)
                dot_records.append({"pos": tuple(pos), "radius": float(dot_size) / 2.0, "color": tuple(cue_color)})
            first_flip = True
            _show_preview([])
            if not _arm_trial_start_signal():
                return True, None
            with frame_timing_monitor.continuous_sequence():
                for _ in range(isi_frames):
                    if _abort_from_input("experimenter_exit_during_isi"):
                        return True, None
                    bg_rect.draw()
                    for d in dots:
                        d.draw()
                    if fix is not None:
                        fix.draw()
                    dot_timing = flip_with_timestamps(win)
                    if first_flip:
                        dot_perf = dot_timing.actual_perf_s
                        _commit_trial_start_signal(
                            dot_perf,
                            dot_timing.requested_perf_s,
                        )
                        _set_initiation_time(dot_perf)
                        logger.log_frame_flip(
                            trial_num=trial_num,
                            event=_frame_event_name("dot"),
                            timestamp_perf_s=dot_perf,
                            requested_timestamp_perf_s=dot_timing.requested_perf_s,
                            requested_duration=isi_plan.requested_s,
                        )
                        first_flip = False

        _show_preview(preview_images)
        if not _arm_trial_start_signal():
            return True, None
        first_flip = True
        flip_ps = None
        flip_perf = None
        with frame_timing_monitor.continuous_sequence():
            for _ in range(stim_frames if is_memory else 1):
                if _abort_from_input("experimenter_exit_during_stimulus"):
                    return True, None
                bg_rect.draw()
                for d in dots:
                    d.draw()
                for s in stims:
                    s.draw()
                if fix is not None:
                    fix.draw()
                if first_flip and not is_memory:
                    _arm_choice_input()
                stim_timing = flip_with_timestamps(win)
                flip_ps = stim_timing.psychopy_s
                if first_flip:
                    flip_perf = stim_timing.actual_perf_s
                    _commit_trial_start_signal(
                        flip_perf,
                        stim_timing.requested_perf_s,
                    )
                    _set_initiation_time(flip_perf)
                    stim_request = (
                        stim_plan.requested_s
                        if is_memory
                        else choice_plan.requested_s
                    )
                    logger.log_frame_flip(
                        trial_num=trial_num,
                        event=_frame_event_name("stim"),
                        timestamp_perf_s=flip_perf,
                        requested_timestamp_perf_s=stim_timing.requested_perf_s,
                        requested_duration=stim_request,
                    )
                    first_flip = False
        if flip_perf is None:
            flip_perf = time.perf_counter()
        if not is_memory:
            _build_choice_hit_targets()
            _start_choice_window(
                flip_ps,
                flip_perf,
                stim_timing.requested_perf_s,
            )
        elif choice_started and choice_deadline is not None:
            if _poll_choice_until(min(choice_deadline, flip_perf + frame_dur)):
                return True, None

        if not is_memory:
            transition_request_deadline = _choice_transition_request_deadline()
            if (
                transition_request_deadline is not None
                and _poll_choice_until(transition_request_deadline)
            ):
                return True, None
        else:
            if not dots:
                for pos in positions:
                    dot = _make_dot(pos, dot_color)
                    dots.append(dot)
                    dot_records.append(
                        {
                            "pos": tuple(pos),
                            "radius": float(dot_size) / 2.0,
                            "color": tuple(dot_color),
                        }
                    )
            for d in dots:
                d.fillColor = rgb255_to_psychopy(dot_color)
                d.fillColorSpace = "rgb"
            for item in dot_records:
                item["color"] = tuple(dot_color)

    if not click_registered:
        if not choice_started:
            bg_rect.draw()
            if is_memory:
                for d in dots:
                    d.draw()
            else:
                for s in (stims_for_choice if sequential else stims):
                    s.draw()
            if fix is not None:
                fix.draw()
            _arm_choice_input()
            choice_timing = flip_with_timestamps(win)
            choice_flip = choice_timing.psychopy_s
            choice_perf_now = choice_timing.actual_perf_s
            _build_choice_hit_targets()
            _start_choice_window(
                choice_flip,
                choice_perf_now,
                choice_timing.requested_perf_s,
            )
            _show_preview([] if is_memory else (stims_for_choice_preview if sequential else preview_images))

        transition_request_deadline = _choice_transition_request_deadline()
        if transition_request_deadline is not None:
            if _poll_choice_until(transition_request_deadline):
                return True, None

    bg_rect.draw()
    if fix is not None:
        fix.draw()
    clear_timing = flip_with_timestamps(win)
    _show_preview([])
    _record_gray_flip(
        clear_timing.actual_perf_s,
        clear_timing.requested_perf_s,
    )
    return False, chosen_info


def sample_non_overlapping_positions(
    count: int,
    stim_size: Tuple[int, int],
    win_size: Tuple[int, int],
    max_attempts: int = 2000,
    margin: int = 50,
) -> List[Tuple[float, float]]:
    """Public helper: wrapper around the non-overlap placement algorithm.

    Returns a list of (x, y) positions in PsychoPy pixel coords (centered at 0,0).
    """
    w_win, h_win = win_size
    w_stim, h_stim = stim_size
    half_w = w_win / 2.0
    half_h = h_win / 2.0

    # Enforce margin from window edges
    min_x = -half_w + w_stim / 2.0 + margin
    max_x = half_w - w_stim / 2.0 - margin
    min_y = -half_h + h_stim / 2.0 + margin
    max_y = half_h - h_stim / 2.0 - margin

    if min_x > max_x or min_y > max_y:
        raise ValueError("Stimulus size is larger than window; cannot place stimuli")

    rects: List[Tuple[float, float, float, float]] = []
    positions: List[Tuple[float, float]] = []
    attempts = 0
    while len(positions) < count and attempts < max_attempts:
        attempts += 1
        x = random.uniform(min_x, max_x)
        y = random.uniform(min_y, max_y)
        ok = True
        for (cx, cy, ww, hh) in rects:
            if abs(x - cx) < (w_stim + ww) / 2.0 and abs(y - cy) < (h_stim + hh) / 2.0:
                ok = False
                break
        if ok:
            rects.append((x, y, w_stim, h_stim))
            positions.append((x, y))

    if len(positions) < count:
        raise RuntimeError(f"Could not place {count} non-overlapping stimuli after {max_attempts} attempts")
    return positions


def clamp_positions(
    positions: List[Tuple[float, float]],
    stim_size: Tuple[int, int],
    win_size: Tuple[int, int],
    margin: int = 50,
) -> List[Tuple[float, float]]:
    """Clamp a list of center positions so stimuli remain within window margins.

    Returns a new list of (x,y) positions.
    """
    half_w = win_size[0] / 2.0
    half_h = win_size[1] / 2.0
    w_stim, h_stim = stim_size
    min_x = -half_w + w_stim / 2.0 + margin
    max_x = half_w - w_stim / 2.0 - margin
    min_y = -half_h + h_stim / 2.0 + margin
    max_y = half_h - h_stim / 2.0 - margin

    out: List[Tuple[float, float]] = []
    for (x, y) in positions:
        cx = min(max(x, min_x), max_x)
        cy = min(max(y, min_y), max_y)
        out.append((cx, cy))
    return out


def sample_trial_options(files: List[Path], num_afc: int, n_trials: int, seed: Optional[int] = None) -> List[List[Path]]:
    """Sample the options for each trial.

    Each trial contains `num_afc` unique stimuli sampled without replacement.
    Trials are independent, so the same image may appear in different trials.
    """
    if seed is not None:
        random.seed(seed)
    if num_afc < 1:
        raise ValueError("num_afc must be >= 1")
    if num_afc > len(files):
        raise ValueError("num_afc cannot be larger than the number of available images")

    trial_option_sets: List[List[Path]] = []
    for _ in range(n_trials):
        trial_option_sets.append(random.sample(files, num_afc))
    return trial_option_sets


def make_color_gaussian_image(
    color_rgb_255: Tuple[int, int, int],
    size_px: Tuple[int, int],
    sigma_frac: float = 0.22,
    zero_threshold: int = 1,
) -> Image.Image:
    """Create an RGBA color patch with a centered 2D Gaussian alpha mask."""
    if not size_px or len(size_px) != 2 or size_px[0] <= 0 or size_px[1] <= 0:
        raise ValueError("size_px must be a (width, height) tuple of positive ints")

    w, h = int(size_px[0]), int(size_px[1])
    rgb = tuple(int(c) for c in color_rgb_255)
    im = Image.new("RGB", (w, h), color=rgb)

    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    sigma = max(2.0, min(w, h) * float(sigma_frac))
    yy, xx = np.mgrid[0:h, 0:w]
    gauss = np.exp(-0.5 * (((xx - cx) / sigma) ** 2 + ((yy - cy) / sigma) ** 2))
    mask_u8 = np.clip(gauss * 255.0, 0, 255).astype(np.uint8)
    if zero_threshold is not None and zero_threshold > 0:
        mask_u8[mask_u8 <= int(zero_threshold)] = 0
    im.putalpha(Image.fromarray(mask_u8, mode="L"))
    return im


def _csc1_feature_id(pair: Tuple[int, int], feature: str) -> int:
    sid, cid = pair
    if feature == "shape":
        return int(sid)
    if feature == "color":
        return int(cid)
    raise ValueError(f"Unknown feature type: {feature}")


def _csc1_feature_key(pair: Tuple[int, int], feature: str) -> Tuple[str, int]:
    return (f"{feature}_only", _csc1_feature_id(pair, feature))


def _csc1_choice_mapping(trial_type: str) -> Tuple[str, str]:
    if trial_type == "shape_to_color":
        return "shape", "color"
    if trial_type == "color_to_shape":
        return "color", "shape"
    raise ValueError("trial_type must be 'shape_to_color' or 'color_to_shape'")


def make_shape_to_shape_trial(
    *,
    n_choices: int,
    rng: Optional[random.Random] = None,
    cue_shape_id: Optional[int] = None,
) -> Tuple[List[Tuple[int, Optional[int]]], int]:
    """Construct one identity-matching trial from s14.svg through s27.svg.

    The cue/target is included exactly once among the shuffled choices. Every
    remaining choice is a different non-associated shape. ``target_index`` is
    returned as a 1-based index for ``present_delayed_afc_trial``.
    """
    n_choices = int(n_choices)
    if not 2 <= n_choices <= len(NON_ASSOCIATED_SHAPE_IDS):
        raise ValueError(
            f"n_choices must be between 2 and "
            f"{len(NON_ASSOCIATED_SHAPE_IDS)}, got {n_choices}"
        )

    if rng is None:
        rng = random.Random()

    if cue_shape_id is None:
        cue_shape_id = rng.choice(NON_ASSOCIATED_SHAPE_IDS)
    else:
        cue_shape_id = int(cue_shape_id)

    if cue_shape_id not in NON_ASSOCIATED_SHAPE_IDS:
        raise ValueError(
            "cue_shape_id must be between 14 and 27, "
            f"got {cue_shape_id}"
        )

    distractor_pool = [
        shape_id
        for shape_id in NON_ASSOCIATED_SHAPE_IDS
        if shape_id != cue_shape_id
    ]
    distractor_ids = rng.sample(distractor_pool, k=n_choices - 1)

    choice_shape_ids = [cue_shape_id, *distractor_ids]
    rng.shuffle(choice_shape_ids)

    target_index = choice_shape_ids.index(cue_shape_id) + 1
    trial_options: List[Tuple[int, Optional[int]]] = [
        (shape_id, None)
        for shape_id in choice_shape_ids
    ]
    return trial_options, target_index


def present_delayed_afc_trial(
    *,
    win: visual.Window,
    preloaded: Dict[Any, Image.Image],
    trial_options: List[Tuple[int, Optional[int]]],
    positions: List[Tuple[float, float]],
    cue_time: float,
    delay_time: float,
    choice_time: float,
    bg_rect,
    fix,
    logger,
    trial_num: int,
    target_index: int,
    trial_type: str = "shape_to_color",
    isi: float = 0.0,
    bg_rgb_255: Optional[Tuple[int, int, int]] = None,
    onset_cue: Optional[visual.ImageStim] = None,
    msg_logger=None,
    fps: Optional[float] = None,
    choice_hitbox_scale: float = 1.0,
    trial_meta: Optional[Dict[str, Any]] = None,
    cue_pos: Tuple[float, float] = (0.0, 0.0),
    raspi: bool = False,
    pigpio_pi=None,
    raspi_pin: int = 18,
    external_abort_checker=None,
    show_fixation: bool = False,
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """Present one delayed AFC trial with frame-locked cue/delay/choice timing.

    Sequence:
        checkerboard onset cue click/touch -> optional pre-cue ISI ->
        feature cue -> delay -> choices -> grey

    For ``shape_to_shape`` trials, every entry in ``trial_options`` must be
    ``(shape_id, None)``, where ``shape_id`` is 14 through 27. The item at the
    1-based ``target_index`` supplies both the cue and the correct choice. All
    remaining entries are different non-associated-shape distractors.

    The AFC cue sequence always begins only after the participant clicks/touches
    the ``make_onset_cue_stim`` checkerboard stimulus.
    """
    from psychopy import core as _core

    if len(trial_options) != len(positions):
        raise ValueError("trial_options and positions must have the same length")
    if not trial_options:
        raise ValueError("trial_options must contain at least one AFC choice")
    if target_index < 1 or target_index > len(trial_options):
        raise ValueError("target_index must be 1-based and within trial_options")
    if onset_cue is None:
        raise ValueError(
            "present_delayed_afc_trial requires an onset_cue made by "
            "make_onset_cue_stim"
        )

    # Normalize every stimulus to (shape_id, color_id). Shape-only stimuli use
    # color_id=None, for example (14, None) for s14.svg.
    normalized_trial_options: List[Tuple[int, Optional[int]]] = []
    for item_number, raw_pair in enumerate(trial_options, start=1):
        try:
            shape_id, color_id = raw_pair
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Each trial_options entry must contain exactly two values: "
                "(shape_id, color_id). Use None for a shape with no associated "
                f"color. Invalid entry {item_number}: {raw_pair!r}"
            ) from exc

        try:
            normalized_shape_id = int(shape_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid shape ID in trial_options entry {item_number}: "
                f"{raw_pair!r}"
            ) from exc

        if color_id is None:
            normalized_color_id: Optional[int] = None
        else:
            try:
                normalized_color_id = int(color_id)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid color ID in trial_options entry {item_number}: "
                    f"{raw_pair!r}"
                ) from exc

        normalized_trial_options.append(
            (normalized_shape_id, normalized_color_id)
        )

    trial_options = normalized_trial_options

    if trial_type == "shape_to_shape":
        if len(trial_options) < 2:
            raise ValueError("shape_to_shape requires at least two choices")

        allowed_shape_ids = frozenset(range(14, 28))
        invalid_pairs = [
            pair
            for pair in trial_options
            if pair[0] not in allowed_shape_ids or pair[1] is not None
        ]
        if invalid_pairs:
            raise ValueError(
                "shape_to_shape trials must contain only (shape_id, None) "
                "entries for s14.svg through s27.svg. "
                f"Invalid entries: {invalid_pairs!r}"
            )

        choice_shape_ids = [pair[0] for pair in trial_options]
        if len(choice_shape_ids) != len(set(choice_shape_ids)):
            raise ValueError(
                "shape_to_shape choices must be unique so that the target "
                "appears exactly once. "
                f"Got shape IDs: {choice_shape_ids!r}"
            )

    if fps is None:
        fps, frame_dur = detect_frame_rate(win, msg_logger=msg_logger)
    else:
        fps = float(fps)
        frame_dur = 1.0 / fps

    isi_plan = plan_frame_duration(isi, fps)
    cue_plan = plan_frame_duration(cue_time, fps, minimum_frames=1)
    delay_plan = plan_frame_duration(delay_time, fps)
    choice_plan = plan_frame_duration(choice_time, fps, minimum_frames=1)
    isi_frames, isi_s = isi_plan.frame_count, isi_plan.scheduled_s
    cue_frames, cue_s = cue_plan.frame_count, cue_plan.scheduled_s
    delay_frames, delay_s = delay_plan.frame_count, delay_plan.scheduled_s
    choice_frames, choice_s = choice_plan.frame_count, choice_plan.scheduled_s

    _log_message(
        msg_logger,
        "INFO",
        (
            f"timing_quantization trial_num={trial_num} "
            f"isi={float(isi):.6f}s-> {isi_frames}fr({isi_s:.6f}s) "
            f"cue_time={float(cue_time):.6f}s-> {cue_frames}fr({cue_s:.6f}s) "
            f"delay_time={float(delay_time):.6f}s-> {delay_frames}fr({delay_s:.6f}s) "
            f"choice_time={float(choice_time):.6f}s-> {choice_frames}fr({choice_s:.6f}s)"
        ),
    )

    if trial_type == "shape_to_shape":
        cue_feature, choice_feature = "shape", "shape"
    else:
        cue_feature, choice_feature = _csc1_choice_mapping(trial_type)

    target_pair = trial_options[target_index - 1]

    def _lookup_feature_image(
        pair: Tuple[int, Optional[int]],
        feature: str,
        *,
        role: str,
    ) -> Image.Image:
        """Return an image already stored in ``preloaded``."""
        if trial_type != "shape_to_shape":
            image = preloaded.get(_csc1_feature_key(pair, feature))
            if image is None:
                image = preloaded.get(pair)
            if image is None:
                raise KeyError(
                    f"Missing {role} image for {pair!r}, feature={feature!r}"
                )
            return image

        # An existing _csc1_feature_key may already support (shape_id, None).
        # If it does not, try common explicit keys for the preloaded SVG image.
        shape_id = int(pair[0])
        candidate_keys: List[Any] = []
        try:
            candidate_keys.append(_csc1_feature_key(pair, "shape"))
        except (NameError, TypeError, ValueError, KeyError, IndexError):
            pass

        candidate_keys.extend(
            [
                ("shape", shape_id),
                pair,
                (shape_id, None),
                shape_id,
                f"s{shape_id}.svg",
                f"s{shape_id}",
            ]
        )

        checked_keys: List[Any] = []
        seen_keys = set()
        for key in candidate_keys:
            try:
                if key in seen_keys:
                    continue
                seen_keys.add(key)
            except TypeError:
                continue

            checked_keys.append(key)
            image = preloaded.get(key)
            if image is not None:
                return image

        raise KeyError(
            f"Missing {role} image for s{shape_id}.svg. Preload it under "
            f"('shape', {shape_id}) or ({shape_id}, None). "
            f"Checked keys: {checked_keys!r}"
        )

    cue_img = _lookup_feature_image(target_pair, cue_feature, role="cue")

    if trial_type == "shape_to_shape":
        _log_message(
            msg_logger,
            "INFO",
            (
                f"shape_to_shape_config trial_num={trial_num} "
                f"cue_shape=s{target_pair[0]}.svg "
                f"choice_shapes={[f's{pair[0]}.svg' for pair in trial_options]} "
                f"target_index={target_index}"
            ),
        )

    cue_stim = make_image_stim_from_array(win, cue_img, size=None, bg_rgb_255=bg_rgb_255)
    cue_stim.pos = cue_pos

    choice_stims: List[visual.ImageStim] = []
    choice_hit_targets: List[visual.Rect] = []
    for pair, pos in zip(trial_options, positions):
        pair = tuple(pair)
        choice_img = _lookup_feature_image(
            pair,
            choice_feature,
            role="choice",
        )
        stim = make_image_stim_from_array(win, choice_img, size=None, bg_rgb_255=bg_rgb_255)
        stim.pos = pos
        choice_stims.append(stim)
        try:
            w, h = stim.size
        except Exception:
            w, h = (64.0, 64.0)
        choice_hit_targets.append(
            visual.Rect(
                win,
                width=max(1.0, float(w) * float(choice_hitbox_scale)),
                height=max(1.0, float(h) * float(choice_hitbox_scale)),
                pos=pos,
                units="pix",
                fillColor=None,
                lineColor=None,
                opacity=0.0,
            )
        )

    mouse = event.Mouse(win=win)
    mouse_presses = MousePressTracker(mouse)
    try:
        event.clearEvents(eventType="mouse")
    except Exception:
        pass

    trial_start_signal_armed_s: Optional[float] = None
    trial_start_signal_sent = False

    def _set_initiation_time(perf_s: Optional[float] = None) -> None:
        if trial_meta is None or "initiation_time_s" in trial_meta:
            return
        perf_now = float(perf_s) if perf_s is not None else time.perf_counter()
        try:
            trial_meta["initiation_time_s"] = logger.seconds_since_session_start(perf_now)
        except Exception:
            trial_meta["initiation_time_s"] = ""

    def _record_gray_flip(perf_s: float) -> None:
        if trial_meta is not None:
            trial_meta["gray_flip_perf_s"] = float(perf_s)

    def _abort_from_input(reason: str) -> bool:
        if event.getKeys(["escape"]):
            _log_message(msg_logger, "WARN", f"escape_pressed trial_num={trial_num} reason={reason}")
            return True
        if external_abort_checker is not None:
            try:
                if external_abort_checker():
                    _log_message(msg_logger, "WARN", f"external_abort trial_num={trial_num} reason={reason}")
                    return True
            except Exception:
                pass
        return False

    def _draw_fixation_if_requested() -> None:
        if show_fixation and fix is not None:
            fix.draw()

    def _draw_blank() -> None:
        bg_rect.draw()
        _draw_fixation_if_requested()

    def _arm_trial_start_signal() -> bool:
        nonlocal trial_start_signal_armed_s, trial_start_signal_sent
        if trial_start_signal_sent or trial_start_signal_armed_s is not None:
            return True
        if not raspi or pigpio_pi is None:
            return True
        try:
            pulse_s = 0.25
            duration_us = int(pulse_s * 1_000_000)
            win.callOnFlip(_send_led_pulse_on_flip, pigpio_pi, raspi_pin, duration_us)
            trial_start_signal_armed_s = pulse_s
            _log_message(msg_logger, "INFO", f"raspi_pulse_registered trial_num={trial_num} duration_s={pulse_s:.6f}")
            return True
        except Exception as e:
            _log_message(msg_logger, "ERROR", f"trial_start_signal_registration_failed trial_num={trial_num} error={e}")
            return False

    def _commit_trial_start_signal(
        flip_perf_s: float,
        requested_perf_s: Optional[float] = None,
    ) -> None:
        nonlocal trial_start_signal_armed_s, trial_start_signal_sent
        if trial_start_signal_armed_s is None:
            return
        logger.log_signal(
            trial_num=trial_num,
            event="trial_start_signal_on",
            timestamp_perf_s=flip_perf_s,
            requested_timestamp_perf_s=requested_perf_s,
            requested_duration=trial_start_signal_armed_s,
        )
        trial_start_signal_sent = True
        trial_start_signal_armed_s = None

    # Mandatory self-initiation cue: show the checkerboard cue and wait for a *new*
    # click/touch inside its bounds before beginning the AFC cue/delay/choice sequence.
    
    try:
        onset_cue.pos = (0, 0)
        onset_cue.opacity = 1.0
    except Exception:
        pass
    bg_rect.draw()
    onset_cue.draw()
    mouse_presses.reset()
    onset_timing = flip_with_timestamps(win)
    oc_perf = onset_timing.actual_perf_s
    logger.log_frame_flip(
        trial_num=trial_num,
        event="onset_cue_on",
        timestamp_perf_s=oc_perf,
        requested_timestamp_perf_s=onset_timing.requested_perf_s,
    )
    while True:
        if _abort_from_input("during_onset_cue"):
            return True, None
        touch_sample = mouse_presses.poll()
        click_pos = touch_sample.position
        if touch_sample.press_started:
            try:
                oc_w, oc_h = onset_cue.size
            except Exception:
                oc_w, oc_h = (200, 200)
            oc_x, oc_y = getattr(onset_cue, "pos", (0, 0))
            if abs(click_pos[0] - oc_x) <= oc_w / 2.0 and abs(click_pos[1] - oc_y) <= oc_h / 2.0:
                click_perf = time.perf_counter()
                _set_initiation_time(click_perf)
                logger.log_interaction(
                    trial_num=trial_num,
                    event="cue_touch",
                    timestamp_perf_s=click_perf,
                )
                _log_message(
                    msg_logger,
                    "INFO",
                    (
                        f"checkerboard_onset_cue_touch trial_num={trial_num} "
                        f"click_xy=({click_pos[0]:.1f},{click_pos[1]:.1f})"
                    ),
                )
                # Prevent the onset-cue press from carrying into the choice window.
                while any(mouse.getPressed()):
                    if _abort_from_input("waiting_for_onset_cue_release"):
                        return True, None
                    _core.wait(0.005)
                mouse_presses.reset()
                break
        _core.wait(0.01)

    if isi_frames > 0:
        first_flip = True
        for _ in range(isi_frames):
            if _abort_from_input("during_pre_cue_isi"):
                return True, None
            _draw_blank()
            flip_with_timestamps(win)
            if first_flip:
                _log_message(msg_logger, "INFO", f"pre_cue_interval trial_num={trial_num} duration_s={isi_s:.6f}")
                first_flip = False

    first_flip = True
    if not _arm_trial_start_signal():
        return True, None
    for _ in range(cue_frames):
        if _abort_from_input("during_cue"):
            return True, None
        bg_rect.draw()
        cue_stim.draw()
        _draw_fixation_if_requested()
        cue_timing = flip_with_timestamps(win)
        if first_flip:
            cue_perf = cue_timing.actual_perf_s
            _commit_trial_start_signal(
                cue_perf,
                cue_timing.requested_perf_s,
            )
            _set_initiation_time(cue_perf)
            if trial_meta is not None:
                trial_meta["cue_flip_perf_s"] = float(cue_perf)
                trial_meta["cue_feature"] = cue_feature
                trial_meta["choice_feature"] = choice_feature
                trial_meta["target_pair"] = target_pair
                trial_meta["target_shape_id"] = int(target_pair[0])
                trial_meta["target_color_id"] = target_pair[1]
                trial_meta["choice_pairs"] = list(trial_options)
            logger.log_frame_flip(
                trial_num=trial_num,
                event="feature_cue_on",
                timestamp_perf_s=cue_perf,
                requested_timestamp_perf_s=cue_timing.requested_perf_s,
                requested_duration=cue_plan.requested_s,
            )
            first_flip = False

    if delay_frames > 0:
        first_flip = True
        for _ in range(delay_frames):
            if _abort_from_input("during_delay"):
                return True, None
            _draw_blank()
            delay_timing = flip_with_timestamps(win)
            if first_flip:
                delay_perf = delay_timing.actual_perf_s
                if trial_meta is not None:
                    trial_meta["delay_flip_perf_s"] = float(delay_perf)
                logger.log_frame_flip(
                    trial_num=trial_num,
                    event="delay_start",
                    timestamp_perf_s=delay_perf,
                    requested_timestamp_perf_s=delay_timing.requested_perf_s,
                    requested_duration=delay_plan.requested_s,
                )
                first_flip = False

    def _match_choice(click_pos: Tuple[float, float]) -> Optional[int]:
        for idx, target in enumerate(choice_hit_targets, start=1):
            try:
                if target.contains(click_pos):
                    return idx
            except Exception:
                pass
        return None

    bg_rect.draw()
    for stim in choice_stims:
        stim.draw()
    _draw_fixation_if_requested()
    mouse_presses.reset()
    choice_timing = flip_with_timestamps(win)
    choice_perf = choice_timing.actual_perf_s
    if trial_meta is not None:
        trial_meta["choice_start_perf_s"] = float(choice_perf)
    logger.log_frame_flip(
        trial_num=trial_num,
        event="options_on",
        timestamp_perf_s=choice_perf,
        requested_timestamp_perf_s=choice_timing.requested_perf_s,
        requested_duration=choice_plan.requested_s,
    )
    logger.log_frame_flip(
        trial_num=trial_num,
        event="choice_start",
        timestamp_perf_s=choice_perf,
        requested_timestamp_perf_s=choice_timing.requested_perf_s,
        requested_duration=choice_plan.requested_s,
    )

    choice_deadline = choice_perf + float(choice_s)
    choice_transition_request_deadline = max(
        choice_perf,
        choice_deadline - (frame_dur / 2.0),
    )
    choice_info: Optional[Dict[str, Any]] = None
    start_touch = mouse_presses.poll()

    if start_touch.active:
        start_pos = start_touch.position
        immediate_idx = _match_choice(start_pos)
        if immediate_idx is not None:
            click_perf = time.perf_counter()
            correct = immediate_idx == int(target_index)
            logger.log_interaction(
                trial_num=trial_num,
                event="option_touch",
                timestamp_perf_s=click_perf,
            )
            choice_info = {
                "chosen_index": int(immediate_idx),
                "chosen_pos": tuple(positions[immediate_idx - 1]),
                "chosen_pair": trial_options[immediate_idx - 1],
                "chosen_shape_id": int(trial_options[immediate_idx - 1][0]),
                "chosen_color_id": trial_options[immediate_idx - 1][1],
                "choice_start_perf_s": float(choice_perf),
                "choice_time_perf_s": float(click_perf),
                "reaction_time_s": float(click_perf - choice_perf),
                "touch_x": float(start_pos[0]),
                "touch_y": float(start_pos[1]),
                "is_correct": bool(correct),
                "target_index": int(target_index),
                "target_pair": target_pair,
                "target_shape_id": int(target_pair[0]),
                "target_color_id": target_pair[1],
                "trial_type": trial_type,
                "cue_feature": cue_feature,
                "choice_feature": choice_feature,
            }

    poll_interval_s = 0.002
    while (
        time.perf_counter() < choice_transition_request_deadline
        and choice_info is None
    ):
        if _abort_from_input("during_choice"):
            return True, None
        touch_sample = mouse_presses.poll()
        click_pos = touch_sample.position
        if touch_sample.press_started:
            chosen_idx = _match_choice(click_pos)
            _log_message(
                msg_logger,
                "INFO",
                (
                    f"choice_touch_attempt trial_num={trial_num} "
                    f"click_xy=({click_pos[0]:.1f},{click_pos[1]:.1f}) matched_idx={chosen_idx}"
                ),
            )
            if chosen_idx is not None:
                click_perf = time.perf_counter()
                correct = chosen_idx == int(target_index)
                logger.log_interaction(
                    trial_num=trial_num,
                    event="option_touch",
                    timestamp_perf_s=click_perf,
                )
                choice_info = {
                    "chosen_index": int(chosen_idx),
                    "chosen_pos": tuple(positions[chosen_idx - 1]),
                    "chosen_pair": trial_options[chosen_idx - 1],
                    "chosen_shape_id": int(trial_options[chosen_idx - 1][0]),
                    "chosen_color_id": trial_options[chosen_idx - 1][1],
                    "choice_start_perf_s": float(choice_perf),
                    "choice_time_perf_s": float(click_perf),
                    "reaction_time_s": float(click_perf - choice_perf),
                    "touch_x": float(click_pos[0]),
                    "touch_y": float(click_pos[1]),
                    "is_correct": bool(correct),
                    "target_index": int(target_index),
                    "target_pair": target_pair,
                    "target_shape_id": int(target_pair[0]),
                    "target_color_id": target_pair[1],
                    "trial_type": trial_type,
                    "cue_feature": cue_feature,
                    "choice_feature": choice_feature,
                }
                break
        remaining = choice_transition_request_deadline - time.perf_counter()
        if remaining > 0:
            _core.wait(min(poll_interval_s, remaining))

    _draw_blank()
    gray_timing = flip_with_timestamps(win)
    gray_perf = gray_timing.actual_perf_s
    _record_gray_flip(gray_perf)
    if trial_meta is not None:
        trial_meta["gray_flip_requested_perf_s"] = float(
            gray_timing.requested_perf_s
        )

    if choice_info is None:
        _log_message(
            msg_logger,
            "INFO",
            (
                f"choice_timeout trial_num={trial_num} trial_type={trial_type} "
                f"target_index={target_index} target_pair={target_pair}"
            ),
        )
    else:
        _log_message(
            msg_logger,
            "INFO",
            (
                f"choice_registered trial_num={trial_num} idx={choice_info['chosen_index']} "
                f"chosen_pair={choice_info['chosen_pair']} "
                f"correct={int(choice_info['is_correct'])} trial_type={trial_type} "
                f"target_index={target_index} target_pair={target_pair}"
            ),
        )

    return False, choice_info

    
# -----------------------------------------------------------------------------------------
# Trial Buffer Manager (for background trial generation with multiprocessing)
# -----------------------------------------------------------------------------------------

def _trial_buffer_worker_generic(
    trial_generator_func: Callable[[int, dict], dict],
    config: dict,
    trial_queue: Any,
    stop_event: Any,
    start_idx: int = 0,
):
    """
    Generic worker process that generates trials in the background.
    
    Args:
        trial_generator_func: Callable that takes (trial_idx, config) and returns trial dict
        config: Configuration dictionary to pass to the generator
        trial_queue: Queue to push generated trials into
        stop_event: Event to signal worker to stop
        start_idx: Starting trial index
    """
    trial_idx = start_idx
    while not stop_event.is_set():
        try:
            trial_data = trial_generator_func(trial_idx, config)
            while not stop_event.is_set():
                try:
                    trial_queue.put(trial_data, timeout=0.1)
                    break
                except queue.Full:
                    continue
            trial_idx += 1
        except Exception as e:
            # Put error into queue for main process to handle
            error_text = str(e) or repr(e)
            error_trace = traceback.format_exc()
            while not stop_event.is_set():
                try:
                    trial_queue.put(
                        {
                            "type": "error",
                            "error": error_text,
                            "traceback": error_trace,
                            "trial_idx": trial_idx,
                        },
                        timeout=0.1,
                    )
                    break
                except queue.Full:
                    continue
            break


class TrialBufferManager:
    """
    Generic trial buffer manager that uses multiprocessing to pre-generate trials
    on a separate core. Works with any task paradigm by taking a user-defined
    trial generation callable.
    
    The trial_generator_func should be a function that takes:
        - trial_idx: int (the index of the trial to generate)
        - config: dict (containing any parameters needed for generation)
    
    And returns a dictionary representing the trial data.
    
    Example usage:
        def my_trial_generator(trial_idx: int, config: dict) -> dict:
            # Generate trial based on config
            return {"trial_idx": trial_idx, "stimuli": [...], ...}
        
        buffer_mgr = TrialBufferManager(
            trial_generator_func=my_trial_generator,
            config={"param1": value1, "param2": value2},
            buffer_size=5
        )
        
        # In your task loop:
        trial_data = buffer_mgr.get_next_trial()
        # Use trial_data...
        
        # When done:
        buffer_mgr.close()
    """
    
    def __init__(
        self, 
        trial_generator_func: Callable[[int, dict], dict],
        config: dict,
        buffer_size: int = 5,
        start_idx: int = 0,
    ):
        """
        Initialize the trial buffer manager.
        
        Args:
            trial_generator_func: A callable that generates trial data.
                                  Must take (trial_idx: int, config: dict) -> dict
            config: Dictionary of configuration parameters to pass to generator
            buffer_size: Maximum number of trials to buffer ahead (default: 5)
            start_idx: Starting trial index (default: 0)
        """
        self.trial_generator_func = trial_generator_func
        self.config = config
        self.buffer_size = buffer_size
        self.start_idx = start_idx
        self.next_trial_idx = start_idx
        
        # Set up multiprocessing with spawn context (required for some libraries like PsychoPy)
        ctx = mp.get_context('spawn')
        self.trial_queue = ctx.Queue(maxsize=buffer_size)
        self.stop_event = ctx.Event()
        
        # Start the worker process
        self.worker = ctx.Process(
            target=_trial_buffer_worker_generic,
            args=(trial_generator_func, config, self.trial_queue, self.stop_event, start_idx)
        )
        self.worker.start()
        self.is_closed = False
    
    def get_next_trial(self) -> dict:
        """
        Get the next pre-generated trial from the buffer.
        
        Returns:
            Dictionary containing the trial data generated by trial_generator_func
            
        Raises:
            RuntimeError: If the buffer manager has been closed or worker encountered an error
        """
        if self.is_closed:
            raise RuntimeError("TrialBufferManager has been closed")
        
        try:
            trial_data = self.trial_queue.get(timeout=30.0)
            
            # Check if worker sent an error
            if isinstance(trial_data, dict) and trial_data.get("type") == "error":
                error_text = trial_data.get("error", "Unknown worker error")
                trial_idx = trial_data.get("trial_idx")
                trace_text = trial_data.get("traceback")
                details = f"Trial generation error at trial_idx={trial_idx}: {error_text}"
                if trace_text:
                    details = f"{details}\n{trace_text}"
                raise RuntimeError(details)
            
            self.next_trial_idx += 1
            return trial_data
            
        except Exception as e:
            self.close()
            raise RuntimeError(f"Failed to get next trial: {e}")
    
    def close(self):
        """
        Clean up the worker process and release resources.
        Should be called when done using the buffer manager.
        """
        if self.is_closed:
            return
            
        self.is_closed = True
        self.stop_event.set()
        
        # Give worker time to finish cleanly
        self.worker.join(timeout=2.0)
        
        # Force terminate if still alive
        if self.worker.is_alive():
            self.worker.terminate()
            self.worker.join(timeout=1.0)
    
    def __del__(self):
        """Destructor to ensure cleanup happens even if close() not called."""
        self.close()
