"""Play random fixed-duration clips from explicitly configured video files."""
import argparse
import math
import random
import sys
import time
from pathlib import Path
from typing import Optional, Sequence, Tuple

from psychopy import event, logging as pylogging

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from bin import utils
from bin.affinity import (
    build_main_and_worker_affinity_plan,
    describe_cpu_set,
    set_process_cpu_affinity,
)
from bin.config import load_config, validate_config
from bin.daqc2_outputs import DAQC2DigitalOutputs, PeriodicDOUTPulseController
from bin.buffered_video import DEFAULT_BUFFER_BYTES
from bin.frame_timing import plan_frame_duration
from bin.logger import SessionLogBundle
from bin.task_lifecycle import USER_EXIT_CODE
from bin.video_playback import (
    RandomFramePulseSchedule,
    SharedVideoFrameBuffer,
    center_crop_bounds,
    is_raspberry_pi,
    parse_frame_rate,
    plan_video_refresh_cadence,
    probe_video_stream,
    select_random_video_clip,
    validate_hevc_stream,
    video_duration_seconds,
    video_time_origin_seconds,
)
from bin.screen import (
    ExperimenterPreview,
    describe_screen,
    load_screen_config,
    oriented_size,
    resolve_scene_size,
    software_stimulus_rotation,
)


def _resolve_video_files(video_files: Sequence[str]) -> list[Path]:
    if isinstance(video_files, (str, bytes)) or not isinstance(video_files, Sequence):
        raise ValueError("video_files must be a JSON list of video file paths")
    if not video_files:
        raise ValueError("video_files must contain at least one video file path")

    resolved: list[Path] = []
    for index, value in enumerate(video_files):
        if not isinstance(value, (str, Path)) or not str(value).strip():
            raise ValueError(f"video_files[{index}] must be a non-empty path")
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = path.absolute()
        if not path.is_file():
            raise FileNotFoundError(f"Video file not found: {path}")
        resolved.append(path)
    return resolved


def parse_args():
    parser = argparse.ArgumentParser(description="Play a random video stimulus")
    parser.add_argument("--config", help="Path to JSON config file. CLI overrides config keys.")
    parser.add_argument("--video_files", nargs="+", default=None, help="Explicit paths to source video files")
    parser.add_argument("--clip_duration_seconds", type=float, default=None, help="Duration of each randomly selected clip/trial")
    parser.add_argument("--num_clips", type=int, default=None, help="Number of clips/trials to play")
    parser.add_argument("--seek_timeout_seconds", type=float, default=None, help="Maximum wait for a network source seek and first decoded frame")
    parser.add_argument("--output_dir", default=None, help="Directory to save logs")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--fullscreen", action="store_true", default=None, help="Run fullscreen")
    parser.add_argument("--win_size", type=int, nargs=2, default=None, help="Window size when not fullscreen")
    parser.add_argument("--bg", type=int, nargs=3, default=None, help="Background RGB color")
    parser.add_argument("--refresh_rate", type=float, default=None, help="Override detected display refresh rate (Hz)")
    parser.add_argument("--frame_rate", type=float, default=None, help="Authoritative video presentation and clip-selection rate (Hz)")
    parser.add_argument("--video_buffer_megabytes", type=float, default=None, help="Maximum shared-memory budget for prepared RGB video chunks")
    parser.add_argument("--ffprobe", default=None, help="Path to ffprobe for codec probing")
    parser.add_argument("--raspi", action="store_true", default=None, help="Enable Raspberry Pi frame-sync GPIO and pump behavior")
    parser.add_argument("--no_raspi", action="store_false", dest="raspi", help="Disable Raspberry Pi hardware behavior")
    parser.add_argument("--sync_pin", type=int, default=None, help="BCM GPIO pin for frame-locked video sync pulses")
    parser.add_argument("--sync_interval_frames", type=int, nargs=2, default=None, metavar=("MIN", "MAX"), help="Inclusive randomized interval between sync-pulse onsets, in video presentation frames")
    parser.add_argument("--sync_pulse_frames", type=int, default=None, help="Sync pulse width in video presentation frames")
    parser.add_argument("--daq_address", type=int, default=None, help="Pi-Plates DAQC2plate address (0-7)")
    parser.add_argument("--daq_module", default=None, help="Python module for the DAQC2plate driver")
    parser.add_argument("--pump_pin", type=int, default=None, help="DAQC2 DOUT bit for the periodic pump pulse (0-7)")
    parser.add_argument("--pump_pulse_time_seconds", type=float, default=None, help="Duration of each pump pulse")
    parser.add_argument("--pump_interval", type=float, default=None, help="Seconds between periodic pump onsets")
    parser.add_argument("--main_screen", default=None, help="Main task screen index or output name")
    parser.add_argument("--experimenter_screen", default=None, help="Experimenter screen index or output name")
    return parser.parse_args()


def run_task(
    video_files: Sequence[str],
    clip_duration_seconds: Optional[float],
    output_dir: str,
    num_clips: int,
    seed: Optional[int] = None,
    fullscreen: bool = True,
    win_size: Optional[Tuple[int, int]] = None,
    bg: Tuple[int, int, int] = (0, 0, 0),
    refresh_rate: Optional[float] = None,
    config_name: Optional[str] = None,
    ffprobe_bin: str = "ffprobe",
    screen_config=None,
    raspi: bool = False,
    sync_pin: int = 18,
    sync_interval_frames: Tuple[int, int] = (100, 300),
    sync_pulse_frames: int = 1,
    seek_timeout_seconds: float = 30.0,
    daq_address: int = 0,
    daq_module_name: str = "piplates.DAQC2plate",
    pump_pin: int = 0,
    pump_pulse_time_seconds: Optional[float] = None,
    pump_interval: Optional[float] = None,
    frame_rate: float = 30.0,
    video_buffer_megabytes: float = DEFAULT_BUFFER_BYTES / (1024 * 1024),
):
    if isinstance(num_clips, bool) or not isinstance(num_clips, int) or num_clips <= 0:
        raise ValueError("num_clips must be a positive integer")
    if clip_duration_seconds is None:
        raise ValueError("clip_duration_seconds is required")
    clip_duration_seconds = float(clip_duration_seconds)
    if not math.isfinite(clip_duration_seconds) or clip_duration_seconds <= 0.0:
        raise ValueError("clip_duration_seconds must be a positive finite value")
    seek_timeout_seconds = float(seek_timeout_seconds)
    if not math.isfinite(seek_timeout_seconds) or seek_timeout_seconds <= 0.0:
        raise ValueError("seek_timeout_seconds must be a positive finite value")
    frame_rate = float(frame_rate)
    if not math.isfinite(frame_rate) or frame_rate <= 0.0:
        raise ValueError("frame_rate must be a positive finite value")
    video_buffer_megabytes = float(video_buffer_megabytes)
    if (
        not math.isfinite(video_buffer_megabytes)
        or video_buffer_megabytes <= 0.0
    ):
        raise ValueError("video_buffer_megabytes must be positive and finite")
    video_buffer_bytes = int(round(video_buffer_megabytes * 1024 * 1024))
    selection_rng = random.Random(seed)
    sync_rng = random.Random(None if seed is None else int(seed) + 0x51A7)
    if len(sync_interval_frames) != 2:
        raise ValueError("sync_interval_frames must contain exactly MIN and MAX")
    sync_interval_min = int(sync_interval_frames[0])
    sync_interval_max = int(sync_interval_frames[1])
    sync_pulse_frames = int(sync_pulse_frames)
    sync_pin = int(sync_pin)
    daq_address = DAQC2DigitalOutputs.validate_address(int(daq_address))
    pump_pin = DAQC2DigitalOutputs.validate_bit(int(pump_pin))
    if raspi:
        # Validate all timing parameters before opening windows or GPIO.
        RandomFramePulseSchedule(
            sync_interval_min,
            sync_interval_max,
            pulse_width_frames=sync_pulse_frames,
            rng=random.Random(0),
        )
        if sync_pin < 0:
            raise ValueError("sync_pin must be a non-negative BCM GPIO number")
        if pump_interval is None or pump_pulse_time_seconds is None:
            raise ValueError(
                "pump_interval and pump_pulse_time_seconds are required when raspi=true"
            )

    resolved_video_files = _resolve_video_files(video_files)

    video_streams = {}
    for video_path in dict.fromkeys(resolved_video_files):
        stream = probe_video_stream(video_path, ffprobe_bin=ffprobe_bin)
        if not stream:
            raise RuntimeError(f"Could not probe video stream: {video_path}")
        validate_hevc_stream(
            video_path,
            stream,
            require_pi5_compatible=bool(raspi),
        )
        source_duration_s = video_duration_seconds(stream)
        source_time_origin_s = video_time_origin_seconds(stream)
        if source_duration_s <= 0.0:
            raise ValueError(f"Video duration is missing or invalid: {video_path}")
        if source_duration_s + 1e-9 < clip_duration_seconds:
            raise ValueError(
                f"Video is shorter than clip_duration_seconds: {video_path} "
                f"({source_duration_s:.6f}s < {clip_duration_seconds:.6f}s)"
            )
        stream["start_time"] = source_time_origin_s
        probed_rate = parse_frame_rate(
            stream.get("avg_frame_rate") or stream.get("r_frame_rate")
        )
        if probed_rate <= 0.0:
            raise ValueError(f"Video frame rate is missing or invalid: {video_path}")
        if abs(probed_rate - frame_rate) > 0.001:
            raise ValueError(
                f"Video frame rate {probed_rate:.6f} does not match configured "
                f"frame_rate {frame_rate:.6f}: {video_path}. Normalize the "
                "source offline or use its exact configured rate."
            )
        average_rate = parse_frame_rate(stream.get("avg_frame_rate"))
        nominal_rate = parse_frame_rate(stream.get("r_frame_rate"))
        if (
            average_rate > 0.0
            and nominal_rate > 0.0
            and abs(average_rate - nominal_rate) > 0.001
        ):
            raise ValueError(
                f"Video reports inconsistent average/nominal frame rates "
                f"({average_rate:.6f}/{nominal_rate:.6f}): {video_path}. "
                "Normalize it to exact CFR during preprocessing."
            )
        video_streams[video_path] = stream

    win, main_screen, experimenter_screen = utils.setup_task_window(
        screen_config,
        bg_rgb_255=bg,
        fullscreen=fullscreen,
        size=win_size,
        allow_same_screen=True,
    )
    bg_rect = utils.make_bg_rect(win, bg)
    mouse = event.Mouse(win=win)
    experimenter_preview = None
    frame_publisher = None
    sync_lgpio = None
    sync_gpio_chip = None
    pump_outputs = None
    pump_controller = None
    native_main_size = resolve_scene_size(
        main_screen,
        fullscreen=bool(fullscreen),
        requested_size=win_size,
        realized_size=tuple(win.size),
    )
    main_rotation_deg = software_stimulus_rotation(main_screen.rotation)
    subject_main_size = oriented_size(native_main_size, main_rotation_deg)
    maximum_frame_bytes = 0
    minimum_runtime_crop_fraction = 1.0
    for stream in video_streams.values():
        source_size = (int(stream["width"]), int(stream["height"]))
        crop_bounds = center_crop_bounds(
            source_size,
            subject_main_size,
            alignment=2,
        )
        cropped_width = crop_bounds[2] - crop_bounds[0]
        cropped_height = crop_bounds[3] - crop_bounds[1]
        maximum_frame_bytes = max(
            maximum_frame_bytes,
            cropped_width * cropped_height * 3,
        )
        minimum_runtime_crop_fraction = min(
            minimum_runtime_crop_fraction,
            (cropped_width * cropped_height)
            / float(source_size[0] * source_size[1]),
        )

    resolved_config_name = str(config_name).strip() if config_name else "play_video"
    session_logs = SessionLogBundle(
        output_root=output_dir,
        task_name="play_video",
        config_name=resolved_config_name,
        behavior_fieldnames=[
            "trial_num",
            "source_video_path",
            "source_video_name",
            "source_duration_seconds",
            "source_clip_start_seconds",
            "source_clip_end_seconds",
            "requested_clip_duration_seconds",
            "scheduled_clip_duration_seconds",
            "configured_video_frame_rate",
            "scheduled_video_frames",
            "actual_source_start_seconds",
            "actual_source_last_frame_seconds",
            "first_frame_time_since_session_start",
            "last_frame_end_time_since_session_start",
            "displayed_duration_seconds",
            "display_frames",
            "aborted",
            "stop_reason",
            "dropped_frames",
            "timing_misses",
            "timing_validation_status",
            "timing_error_p95_milliseconds",
            "timing_error_maximum_milliseconds",
            "cadence_mismatches",
            "source_pts_contiguous",
            "realized_refresh_hold_histogram",
            "main_vblank_validation_status",
            "glx_swap_interval",
            "main_vblank_delta_msc",
            "main_vblank_expected_delta_msc",
            "main_vblank_delta_sbc",
            "main_vblank_expected_delta_sbc",
            "scheduled_video_slots_skipped",
            "sync_pulses",
        ],
        additional_table_fieldnames={
            "video_frame_timing": [
                "trial_num",
                "source_frame_index",
                "source_media_time_seconds",
                "expected_source_time_seconds",
                "source_pts_error_seconds",
                "expected_time_since_video_start_seconds",
                "requested_time_since_session_start",
                "expected_time_since_session_start",
                "actual_time_since_session_start",
                "timing_error_seconds",
                "planned_hold_refreshes",
                "realized_hold_refreshes",
                "realized_hold_seconds",
                "boundary_status",
            ]
        },
        auto_flush=False,
    )
    logger = session_logs.event_logger
    msg_logger = session_logs.message_logger
    behavior_logger = session_logs.behavior_logger
    frame_timing_logger = getattr(
        session_logs,
        "table_loggers",
        {},
    ).get("video_frame_timing")
    if behavior_logger is None:
        raise RuntimeError("play_video requires a behavior logger")
    pylogging.console.setLevel(pylogging.CRITICAL)

    pump_failure_logged = False

    def _drain_pump_edges() -> None:
        nonlocal pump_failure_logged
        if pump_controller is None:
            return
        for edge in pump_controller.drain_edges():
            logger.log_signal(
                trial_num=None,
                event="pump_on" if edge.active else "pump_off",
                timestamp_perf_s=edge.actual_perf_s,
                requested_timestamp_perf_s=edge.requested_perf_s,
                requested_duration=(
                    float(pump_pulse_time_seconds) if edge.active else None
                ),
            )
        if pump_controller.failure is not None and not pump_failure_logged:
            pump_failure_logged = True
            msg_logger.log(
                "ERROR",
                f"periodic_pump_failure error={pump_controller.failure}",
            )

    def _write_video_behavior_row(
        *,
        trial_num: int,
        chosen_video: Path,
        selected_clip,
        playback_info=None,
        failure_reason: str = "",
    ) -> None:
        playback = dict(playback_info or {})
        first_frame_time = (
            logger.seconds_since_session_start(playback["start_flip_perf_s"])
            if playback.get("start_flip_perf_s") is not None
            else None
        )
        last_frame_end_time = (
            logger.seconds_since_session_start(
                playback["last_frame_end_perf_s"]
            )
            if playback.get("last_frame_end_perf_s") is not None
            else None
        )
        behavior_logger.writerow(
            {
                "trial_num": int(trial_num),
                "source_video_path": str(
                    playback.get("video_path", chosen_video)
                ),
                "source_video_name": str(
                    playback.get("video_name", chosen_video.name)
                ),
                "source_duration_seconds": (
                    f"{float(playback.get('source_duration_s', selected_clip.source_duration_s)):.9f}"
                ),
                "source_clip_start_seconds": (
                    f"{float(playback.get('clip_start_s', selected_clip.start_s)):.9f}"
                ),
                "source_clip_end_seconds": (
                    f"{float(playback.get('clip_end_s', selected_clip.end_s)):.9f}"
                ),
                "requested_clip_duration_seconds": (
                    f"{float(selected_clip.requested_duration_s):.9f}"
                ),
                "scheduled_clip_duration_seconds": (
                    f"{float(selected_clip.duration_s):.9f}"
                ),
                "configured_video_frame_rate": (
                    f"{float(selected_clip.frame_rate):.9f}"
                ),
                "scheduled_video_frames": int(selected_clip.frame_count),
                "actual_source_start_seconds": (
                    f"{float(playback['actual_source_start_s']):.9f}"
                    if playback.get("actual_source_start_s") is not None
                    else ""
                ),
                "actual_source_last_frame_seconds": (
                    f"{float(playback['actual_source_last_frame_s']):.9f}"
                    if playback.get("actual_source_last_frame_s") is not None
                    else ""
                ),
                "first_frame_time_since_session_start": (
                    f"{float(first_frame_time):.9f}"
                    if first_frame_time is not None
                    else ""
                ),
                "last_frame_end_time_since_session_start": (
                    f"{float(last_frame_end_time):.9f}"
                    if last_frame_end_time is not None
                    else ""
                ),
                "displayed_duration_seconds": (
                    f"{float(playback['displayed_duration_s']):.9f}"
                    if playback.get("displayed_duration_s") is not None
                    else ""
                ),
                "display_frames": playback.get("frames_presented", ""),
                "aborted": int(
                    True if failure_reason else playback.get("aborted", False)
                ),
                "stop_reason": (
                    failure_reason
                    or playback.get("abort_reason")
                    or "completed"
                ),
                "dropped_frames": playback.get("dropped_frames", ""),
                "timing_misses": playback.get("late_frame_count", ""),
                "timing_validation_status": playback.get(
                    "timing_validation_status",
                    "",
                ),
                "timing_error_p95_milliseconds": (
                    f"{1000.0 * float(playback['timing_error_p95_s']):.6f}"
                    if playback.get("timing_error_p95_s") is not None
                    else ""
                ),
                "timing_error_maximum_milliseconds": (
                    f"{1000.0 * float(playback['timing_error_maximum_s']):.6f}"
                    if playback.get("timing_error_maximum_s") is not None
                    else ""
                ),
                "cadence_mismatches": playback.get(
                    "cadence_mismatch_count",
                    "",
                ),
                "source_pts_contiguous": (
                    int(bool(playback["source_pts_contiguous"]))
                    if "source_pts_contiguous" in playback
                    else ""
                ),
                "realized_refresh_hold_histogram": str(
                    playback.get("realized_refresh_hold_histogram", "")
                ),
                "main_vblank_validation_status": playback.get(
                    "main_vblank_validation_status",
                    "",
                ),
                "glx_swap_interval": (
                    playback.get("glx_swap_interval")
                    if playback.get("glx_swap_interval") is not None
                    else ""
                ),
                "main_vblank_delta_msc": (
                    playback.get("main_vblank_delta_msc")
                    if playback.get("main_vblank_delta_msc") is not None
                    else ""
                ),
                "main_vblank_expected_delta_msc": (
                    playback.get("main_vblank_expected_delta_msc")
                    if playback.get("main_vblank_expected_delta_msc")
                    is not None
                    else ""
                ),
                "main_vblank_delta_sbc": (
                    playback.get("main_vblank_delta_sbc")
                    if playback.get("main_vblank_delta_sbc") is not None
                    else ""
                ),
                "main_vblank_expected_delta_sbc": (
                    playback.get("main_vblank_expected_delta_sbc")
                    if playback.get("main_vblank_expected_delta_sbc")
                    is not None
                    else ""
                ),
                "scheduled_video_slots_skipped": playback.get(
                    "scheduled_video_slots_skipped",
                    "",
                ),
                "sync_pulses": playback.get("sync_pulses", ""),
            }
        )

    def _write_video_frame_timing_rows(
        *,
        trial_num: int,
        playback_info,
    ) -> None:
        if frame_timing_logger is None:
            return
        playback = dict(playback_info or {})
        video_fps = float(
            playback.get("configured_video_frame_rate", 0.0) or 0.0
        )
        for record in playback.get("frame_timing_records", ()):
            source_frame_index = int(record["source_frame_index"])
            frame_timing_logger.writerow(
                {
                    "trial_num": int(trial_num),
                    "source_frame_index": source_frame_index,
                    "source_media_time_seconds": (
                        f"{float(record['source_media_time_s']):.9f}"
                    ),
                    "expected_source_time_seconds": (
                        f"{float(record['expected_source_pts_s']):.9f}"
                    ),
                    "source_pts_error_seconds": (
                        f"{float(record['source_pts_error_s']):.9f}"
                    ),
                    "expected_time_since_video_start_seconds": (
                        f"{source_frame_index / video_fps:.9f}"
                        if video_fps > 0.0
                        else ""
                    ),
                    "requested_time_since_session_start": (
                        f"{logger.seconds_since_session_start(float(record['flip_requested_perf_s'])):.9f}"
                    ),
                    "expected_time_since_session_start": (
                        f"{logger.seconds_since_session_start(float(record['expected_flip_perf_s'])):.9f}"
                    ),
                    "actual_time_since_session_start": (
                        f"{logger.seconds_since_session_start(float(record['actual_flip_perf_s'])):.9f}"
                    ),
                    "timing_error_seconds": (
                        f"{float(record['timing_error_s']):.9f}"
                    ),
                    "planned_hold_refreshes": record.get(
                        "planned_hold_refreshes",
                        "",
                    ),
                    "realized_hold_refreshes": (
                        record.get("realized_hold_refreshes")
                        if record.get("realized_hold_refreshes") is not None
                        else ""
                    ),
                    "realized_hold_seconds": (
                        f"{float(record['realized_hold_s']):.9f}"
                        if record.get("realized_hold_s") is not None
                        else ""
                    ),
                    "boundary_status": record.get(
                        "boundary_status",
                        "",
                    ),
                }
            )

    try:
        if raspi:
            import lgpio

            sync_lgpio = lgpio
            sync_gpio_chip = lgpio.gpiochip_open(0)
            if isinstance(sync_gpio_chip, int) and sync_gpio_chip < 0:
                raise RuntimeError(
                    f"Could not open GPIO chip 0; lgpio={sync_gpio_chip}"
                )
            claim_result = lgpio.gpio_claim_output(sync_gpio_chip, sync_pin)
            if isinstance(claim_result, int) and claim_result < 0:
                raise RuntimeError(
                    f"Could not claim sync GPIO {sync_pin}; lgpio={claim_result}"
                )
            write_result = lgpio.gpio_write(sync_gpio_chip, sync_pin, 0)
            if isinstance(write_result, int) and write_result < 0:
                raise RuntimeError(
                    f"Could not initialize sync GPIO {sync_pin} low; lgpio={write_result}"
                )
            pump_outputs = DAQC2DigitalOutputs(
                address=daq_address,
                module_name=daq_module_name,
            )
            pump_controller = PeriodicDOUTPulseController(
                pump_outputs,
                bit=pump_pin,
                interval_s=float(pump_interval),
                pulse_duration_s=float(pump_pulse_time_seconds),
            )

        msg_logger.log(
            "INFO",
            f"session_start task=play_video config_name={resolved_config_name} session_dir={session_logs.session_dir}",
        )
        # Keep CPU 0 free while multiprocessing children and decoder threads
        # are created. The preview and ffpyplayer worker inherit this worker-only
        # mask. Only the main presentation thread is later moved to CPU 0.
        affinity_plan = build_main_and_worker_affinity_plan(main_core=0)
        main_cpu_affinity = affinity_plan.get("main_cpu_affinity")
        worker_cpu_affinity = affinity_plan.get("worker_cpu_affinity")
        parent_staged_off_main_core = False
        if affinity_plan.get("supported"):
            msg_logger.log(
                "INFO",
                (
                    "cpu_affinity_plan "
                    f"current=[{describe_cpu_set(affinity_plan['current_affinity'])}] "
                    f"main=[{describe_cpu_set(main_cpu_affinity)}] "
                    f"workers=[{describe_cpu_set(worker_cpu_affinity) if worker_cpu_affinity else ''}]"
                ),
            )
            if affinity_plan.get("warning"):
                msg_logger.log("WARN", str(affinity_plan["warning"]))
            if worker_cpu_affinity:
                staged_ok, staged_detail = set_process_cpu_affinity(worker_cpu_affinity)
                if staged_ok:
                    parent_staged_off_main_core = True
                    msg_logger.log("INFO", f"cpu_affinity_spawn_phase {staged_detail}")
                else:
                    msg_logger.log("WARN", f"cpu_affinity_spawn_phase_failed {staged_detail}")
        else:
            msg_logger.log("WARN", f"cpu_affinity_unavailable {affinity_plan.get('reason')}")

        if pump_controller is not None:
            pump_controller.start(anchor_perf_s=time.perf_counter())
            msg_logger.log(
                "INFO",
                (
                    f"periodic_pump_started daq_module={daq_module_name} "
                    f"daq_address={daq_address} pump_dout={pump_pin} "
                    f"pump_interval_s={float(pump_interval):.6f} "
                    f"pulse_duration_s={float(pump_pulse_time_seconds):.6f} "
                    "schedule=absolute_onset skip_missed_intervals=1"
                ),
            )

        msg_logger.log(
            "INFO",
            f"resolved_screens main={describe_screen(main_screen)} experimenter={describe_screen(experimenter_screen)}",
        )
        for video_path, stream in video_streams.items():
            msg_logger.log(
                "INFO",
                (
                    f"video_source_validation file={video_path.name} "
                    f"codec={stream.get('codec_name')} "
                    f"profile={stream.get('profile')} "
                    f"pix_fmt={stream.get('pix_fmt')} "
                    f"size={stream.get('width')}x{stream.get('height')} "
                    f"avg_frame_rate={stream.get('avg_frame_rate')} "
                    f"r_frame_rate={stream.get('r_frame_rate')} "
                    f"field_order={stream.get('field_order', 'unknown')} "
                    f"has_b_frames={stream.get('has_b_frames', 'unknown')} "
                    f"start_time={stream.get('start_time')} "
                    f"status=accepted"
                ),
            )
        msg_logger.log(
            "INFO",
            (
                f"video_requirements codec=hevc profile=Main pix_fmt=yuv420p "
                f"probed_once=1 n_video_paths={len(resolved_video_files)} "
                f"n_unique_videos={len(video_streams)} clip_duration_s={clip_duration_seconds:.6f} "
                f"num_clips={num_clips} "
                f"configured_video_fps={frame_rate:.6f} "
                f"video_buffer_megabytes={video_buffer_megabytes:.3f} "
                f"minimum_runtime_crop_fraction="
                f"{minimum_runtime_crop_fraction:.6f} "
                f"seek_timeout_s={seek_timeout_seconds:.3f}"
            ),
        )
        if minimum_runtime_crop_fraction < 0.8:
            msg_logger.log(
                "WARN",
                (
                    "video_decode_efficiency runtime_crop_discards_pixels "
                    f"minimum_retained_fraction="
                    f"{minimum_runtime_crop_fraction:.6f} "
                    "recommendation=preprocess_to_subject_screen_aspect "
                    f"subject_size={int(subject_main_size[0])}x"
                    f"{int(subject_main_size[1])}"
                ),
            )
        if raspi:
            msg_logger.log(
                "INFO",
                (
                    "pi5_video_mode decoder=ffpyplayer "
                    f"sync_pin={sync_pin} interval_frames={sync_interval_min}-{sync_interval_max} "
                    f"pulse_width_frames={sync_pulse_frames} pump_dout={pump_pin}"
                ),
            )
        msg_logger.log(
            "INFO",
            (
                "resolved_main_scene_size "
                f"native_size={native_main_size[0]}x{native_main_size[1]} "
                f"subject_size={subject_main_size[0]}x{subject_main_size[1]} "
                f"output_rotation={main_screen.rotation} "
                f"stimulus_rotation_deg={main_rotation_deg} "
                f"fullscreen={int(bool(fullscreen))} requested_win_size={win_size} "
                f"realized_win_size={tuple(win.size)}"
            ),
        )
        fps, _ = utils.resolve_frame_rate(
            win,
            refresh_rate,
            msg_logger=msg_logger,
            context="play_video",
        )
        cadence_frame_count = plan_frame_duration(
            clip_duration_seconds,
            frame_rate,
            minimum_frames=1,
        ).frame_count
        try:
            refresh_cadence = plan_video_refresh_cadence(
                cadence_frame_count,
                frame_rate,
                fps,
            )
        except ValueError as exc:
            msg_logger.log(
                "ERROR",
                f"video_monitor_cadence_unusable "
                f"configured_video_fps={frame_rate:.6f} "
                f"monitor_fps={fps:.6f} error={exc}",
            )
            raise
        effective_video_fps = (
            cadence_frame_count
            / refresh_cadence.scheduled_display_duration_s
        )
        cadence_error_hz = effective_video_fps - frame_rate
        msg_logger.log(
            "INFO",
            (
                f"video_cadence configured_video_fps={frame_rate:.6f} "
                f"monitor_fps={fps:.6f} nominal_refreshes_per_video_frame="
                f"{refresh_cadence.nominal_refreshes_per_video_frame:.9f} "
                f"refresh_hold_histogram="
                f"{dict(refresh_cadence.refresh_count_histogram)} "
                f"total_refreshes={refresh_cadence.total_refreshes} "
                f"maximum_phase_error_s="
                f"{refresh_cadence.maximum_absolute_phase_error_s:.9f} "
                f"final_phase_error_s="
                f"{refresh_cadence.final_phase_error_s:.9f} "
                f"effective_video_fps={effective_video_fps:.9f} "
                f"cadence_error_hz={cadence_error_hz:.9f} "
                "schedule=absolute_source_time_nearest_vbl "
                "source_frames=never_skip"
            ),
        )
        if experimenter_screen is not None:
            frame_publisher = SharedVideoFrameBuffer(maximum_frame_bytes)
            experimenter_preview = ExperimenterPreview(
                experimenter_screen,
                task_label=resolved_config_name,
                start_perf_s=time.perf_counter(),
                update_interval_s=0.1,
            )
            experimenter_preview.clear_scene(
                bg_rgb_255=bg,
                main_size=native_main_size,
                main_rotation_deg=main_rotation_deg,
            )
            msg_logger.log(
                "INFO",
                (
                    f"experimenter_video_mirror mode=single_decode_latest_frame_wins "
                    f"shared_memory={frame_publisher.name} capacity_bytes={maximum_frame_bytes}"
                    f" slots={frame_publisher.slot_count} publish_interval_s=0.1 "
                    "vsync=secondary_output_independent "
                    "main_swap_reference_unchanged=1 "
                    "frame_pacing=new_source_frame"
                ),
            )

        main_vblank = utils.verify_task_window_vblank(win)
        primary_lease = main_vblank.primary_lease
        main_vblank_status = (
            "confirmed"
            if main_vblank.swap_interval == 1
            else "platform_not_applicable"
        )
        msg_logger.log(
            "INFO",
            (
                "main_vblank_reference phase=post_preview "
                f"output={main_screen.name} "
                f"refresh_sync_requested="
                f"{int(main_vblank.refresh_sync_requested)} "
                f"xrandr_primary_changed={int(primary_lease.changed)} "
                f"xrandr_detail={main_vblank.primary_verification_detail} "
                f"swap_interval={main_vblank.swap_interval!r} "
                f"swap_interval_detail={main_vblank.swap_interval_detail} "
                "secondary_drawable_independent=1 "
                f"status={main_vblank_status}"
            ),
        )

        def _pin_main_for_playback() -> None:
            if not main_cpu_affinity:
                return
            main_ok, main_detail = set_process_cpu_affinity(main_cpu_affinity)
            if main_ok:
                msg_logger.log("INFO", f"cpu_affinity_main_phase {main_detail}")
                return

            msg_logger.log("WARN", f"cpu_affinity_main_phase_failed {main_detail}")
            restore_affinity = affinity_plan.get("current_affinity")
            if parent_staged_off_main_core and restore_affinity:
                restore_ok, restore_detail = set_process_cpu_affinity(restore_affinity)
                if restore_ok:
                    msg_logger.log("WARN", f"cpu_affinity_restore_after_failure {restore_detail}")
                else:
                    msg_logger.log("WARN", f"cpu_affinity_restore_failed {restore_detail}")

        def _stage_main_for_decoder() -> None:
            if not worker_cpu_affinity:
                return
            staged_ok, staged_detail = set_process_cpu_affinity(worker_cpu_affinity)
            if staged_ok:
                msg_logger.log("INFO", f"cpu_affinity_decoder_phase {staged_detail}")
                return

            msg_logger.log("WARN", f"cpu_affinity_decoder_phase_failed {staged_detail}")
            restore_affinity = affinity_plan.get("current_affinity")
            if restore_affinity:
                restore_ok, restore_detail = set_process_cpu_affinity(restore_affinity)
                if restore_ok:
                    msg_logger.log("WARN", f"cpu_affinity_restore_after_decoder_failure {restore_detail}")
                else:
                    msg_logger.log("WARN", f"cpu_affinity_restore_failed {restore_detail}")

        msg_logger.log(
            "INFO",
            f"task_ready monitor_fps={fps:.6f} video_fps={frame_rate:.6f} n_video_paths={len(resolved_video_files)} clip_duration_s={clip_duration_seconds:.6f} num_clips={num_clips}",
        )

        try:
            event.clearEvents(eventType="mouse")
            mouse.clickReset()
        except Exception:
            pass
        playback_info = None
        played_videos = 0
        stop_reason = "done"

        def _external_abort_reason():
            if pump_controller is not None and pump_controller.failed:
                return "pump_failure"
            if experimenter_preview is not None and experimenter_preview.poll():
                return "experimenter_exit"
            return False

        while played_videos < num_clips:
            _drain_pump_edges()
            if pump_controller is not None and pump_controller.failed:
                raise RuntimeError(
                    f"Periodic pump output failed: {pump_controller.failure}"
                )
            if experimenter_preview is not None and experimenter_preview.poll():
                stop_reason = "experimenter_exit"
                msg_logger.log("WARN", "experimenter_exit_before_video")
                break
            chosen_video = selection_rng.choice(resolved_video_files)
            chosen_stream = video_streams[chosen_video]
            selected_clip = select_random_video_clip(
                chosen_stream,
                clip_duration_seconds,
                rng=selection_rng,
                frame_rate=frame_rate,
            )
            if experimenter_preview is not None and frame_publisher is not None:
                preview_crop_bounds = center_crop_bounds(
                    (
                        int(chosen_stream["width"]),
                        int(chosen_stream["height"]),
                    ),
                    subject_main_size,
                    alignment=2,
                )
                preview_frame_size = (
                    preview_crop_bounds[2] - preview_crop_bounds[0],
                    preview_crop_bounds[3] - preview_crop_bounds[1],
                )
                experimenter_preview.play_shared_video(
                    shared_frame_buffer=frame_publisher.descriptor(),
                    minimum_sequence=frame_publisher.sequence + 1,
                    video_size=preview_frame_size,
                    bg_rgb_255=bg,
                    main_size=native_main_size,
                    main_rotation_deg=main_rotation_deg,
                )
            sync_schedule = (
                RandomFramePulseSchedule(
                    sync_interval_min,
                    sync_interval_max,
                    pulse_width_frames=sync_pulse_frames,
                    rng=sync_rng,
                )
                if raspi
                else None
            )
            try:
                try:
                    playback_info = utils.play_video_fill_screen(
                        win=win,
                        video_path=chosen_video,
                        logger=logger,
                        bg_rect=bg_rect,
                        msg_logger=msg_logger,
                        allow_escape=True,
                        stop_on_mouse_click=False,
                        mouse=mouse,
                        ffprobe_bin=ffprobe_bin,
                        external_abort_checker=_external_abort_reason,
                        trial_num=played_videos + 1,
                        stream_info=chosen_stream,
                        frame_publisher=frame_publisher,
                        sync_schedule=sync_schedule,
                        sync_gpio_module=sync_lgpio,
                        sync_gpio_chip=sync_gpio_chip,
                        sync_pin=sync_pin,
                        frame_publish_interval_s=0.1,
                        clip_start_s=selected_clip.start_s,
                        clip_duration_s=selected_clip.duration_s,
                        requested_clip_duration_s=(
                            selected_clip.requested_duration_s
                        ),
                        video_frame_rate=selected_clip.frame_rate,
                        video_frame_count=selected_clip.frame_count,
                        display_refresh_rate=fps,
                        refresh_cadence=refresh_cadence,
                        video_buffer_bytes=video_buffer_bytes,
                        seek_timeout_s=seek_timeout_seconds,
                        decoder_ready_callback=_pin_main_for_playback,
                        video_onset_callback=(
                            experimenter_preview.mark_video_started
                            if experimenter_preview is not None
                            else None
                        ),
                        stimulus_rotation_degrees=main_rotation_deg,
                        native_target_size=native_main_size,
                    )
                finally:
                    # Keep future children off the reserved presentation core.
                    _stage_main_for_decoder()
            except Exception as exc:
                failure_reason = f"playback_error:{type(exc).__name__}"
                msg_logger.log(
                    "ERROR",
                    f"video_trial_failed trial_num={played_videos + 1} "
                    f"file={chosen_video.name} reason={failure_reason} "
                    f"error={exc}",
                )
                _write_video_behavior_row(
                    trial_num=played_videos + 1,
                    chosen_video=chosen_video,
                    selected_clip=selected_clip,
                    failure_reason=failure_reason,
                )
                session_logs.flush()
                raise
            _write_video_frame_timing_rows(
                trial_num=played_videos + 1,
                playback_info=playback_info,
            )
            _drain_pump_edges()
            if playback_info.get("abort_reason") == "pump_failure":
                _write_video_behavior_row(
                    trial_num=played_videos + 1,
                    chosen_video=chosen_video,
                    selected_clip=selected_clip,
                    playback_info=playback_info,
                )
                session_logs.flush()
                raise RuntimeError(
                    f"Periodic pump output failed: {pump_controller.failure if pump_controller is not None else 'unknown error'}"
                )
            played_videos += 1
            _write_video_behavior_row(
                trial_num=played_videos,
                chosen_video=chosen_video,
                selected_clip=selected_clip,
                playback_info=playback_info,
            )
            if experimenter_preview is not None:
                experimenter_preview.clear_scene(
                    bg_rgb_255=bg,
                    main_size=native_main_size,
                    main_rotation_deg=main_rotation_deg,
                )
            session_logs.flush()
            if playback_info["aborted"]:
                stop_reason = playback_info.get("abort_reason") or "aborted"
                break

        msg_logger.log(
            "INFO",
            (
                f"session_end status={stop_reason} played_videos={played_videos} "
                f"stop_reason={stop_reason} "
                f"last_video={playback_info['video_name'] if playback_info is not None else ''}"
            ),
        )
        return stop_reason
    finally:
        if pump_controller is not None:
            try:
                pump_controller.stop()
            except Exception as exc:
                msg_logger.log("ERROR", f"periodic_pump_cleanup_failed error={exc}")
            _drain_pump_edges()
        if sync_lgpio is not None and sync_gpio_chip is not None:
            try:
                sync_lgpio.gpio_write(sync_gpio_chip, sync_pin, 0)
            except Exception:
                pass
            try:
                sync_lgpio.gpiochip_close(sync_gpio_chip)
            except Exception:
                pass
        if experimenter_preview is not None:
            try:
                experimenter_preview.close()
            except Exception:
                pass
        if frame_publisher is not None:
            try:
                frame_publisher.close()
            except Exception:
                pass
        try:
            restore_detail = utils.close_task_window(win)
            msg_logger.log(
                "INFO",
                f"main_vblank_reference_restored detail={restore_detail}",
            )
        except Exception as exc:
            msg_logger.log(
                "WARN",
                "main_vblank_reference_restore_failed "
                f"error={type(exc).__name__}: {exc}",
            )
        try:
            session_logs.close()
        except Exception:
            pass


def main():
    args = parse_args()
    cfg = {}
    if args.config:
        cfg = load_config(args.config)
        validate_config(
            cfg,
            required=[
                "config_name",
                "output_dir",
                "video_files",
                "clip_duration_seconds",
                "num_clips",
            ],
        )

    def _get(name, default=None):
        val = getattr(args, name, None)
        if val is not None:
            return val
        return cfg.get(name, default)

    def _optional_float(name):
        value = _get(name, None)
        return None if value is None else float(value)

    screen_config = load_screen_config(
        cfg,
        cli_main=args.main_screen,
        cli_experimenter=args.experimenter_screen,
    )
    daq_cfg = cfg.get("daq", {})

    try:
        stop_reason = run_task(
            video_files=_get("video_files", []),
            clip_duration_seconds=_get("clip_duration_seconds", None),
            output_dir=_get("output_dir", "./logs"),
            num_clips=_get("num_clips", None),
            seed=_get("seed", None),
            fullscreen=bool(_get("fullscreen", cfg.get("fullscreen", True))),
            win_size=tuple(_get("win_size", cfg.get("win_size", None))) if _get("win_size", None) else None,
            bg=tuple(_get("bg", cfg.get("bg", (0, 0, 0)))),
            refresh_rate=_get("refresh_rate", cfg.get("refresh_rate", cfg.get("refrech_rate", None))),
            frame_rate=float(_get("frame_rate", cfg.get("frame_rate", 30.0))),
            video_buffer_megabytes=float(
                _get(
                    "video_buffer_megabytes",
                    cfg.get(
                        "video_buffer_megabytes",
                        DEFAULT_BUFFER_BYTES / (1024 * 1024),
                    ),
                )
            ),
            config_name=_get("config_name", cfg.get("config_name", "play_video")),
            ffprobe_bin=_get("ffprobe", cfg.get("ffprobe", "ffprobe")),
            screen_config=screen_config,
            raspi=bool(_get("raspi", is_raspberry_pi())),
            sync_pin=int(_get("sync_pin", cfg.get("sync_pin", 18))),
            sync_interval_frames=tuple(
                int(value)
                for value in _get(
                    "sync_interval_frames",
                    cfg.get("sync_interval_frames", (100, 300)),
                )
            ),
            sync_pulse_frames=int(
                _get("sync_pulse_frames", cfg.get("sync_pulse_frames", 1))
            ),
            seek_timeout_seconds=float(
                _get("seek_timeout_seconds", cfg.get("seek_timeout_seconds", 30.0))
            ),
            daq_address=int(
                _get("daq_address", daq_cfg.get("address", cfg.get("daq_address", 0)))
            ),
            daq_module_name=str(
                _get(
                    "daq_module",
                    daq_cfg.get(
                        "module_name",
                        cfg.get("daq_module_name", "piplates.DAQC2plate"),
                    ),
                )
            ),
            pump_pin=int(_get("pump_pin", cfg.get("pump_pin", 0))),
            pump_pulse_time_seconds=_optional_float("pump_pulse_time_seconds"),
            pump_interval=_optional_float("pump_interval"),
        )
        if stop_reason != "done":
            sys.exit(USER_EXIT_CODE)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
