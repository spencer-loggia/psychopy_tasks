#!/usr/bin/env python3
"""Calibrate Eye 0 from the Pi 5 UART with a separate experimenter display."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import time

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from psychopy import core, event, logging as pylogging
from bin import utils
from bin.config import load_config
from bin.daqc2_outputs import DAQC2DigitalOutputs
from bin.eye_calibration_keys import CALIBRATION_KEYS, fixation_key_steps
from bin.eye_calibration_preview import _make_fixation_cross, _draw_fixation_cross
from bin.eye_calibration_reward import CalibrationAutoReward
from bin.eye_calibration_video import CalibrationVideoClips
from bin.buffered_video import DEFAULT_BUFFER_BYTES
from bin.video_playback import SharedVideoFrameBuffer, make_rgb_video_stim, upload_rgb_texture
from bin.eye_tracking_config import load_eye_tracking_settings
from bin.eye_tracking import (
    UARTEyeTracker,
    UARTEyeTrackerConfig,
    EyeCalibration,
    EyeFilterConfig,
    calibration_payload,
    fraction_position_within_diameter,
    fraction_to_pixels,
)
from bin.logger import SessionLogBundle
from bin.screen import (
    ExperimenterPreview,
    describe_screen,
    load_screen_config,
    oriented_size,
    resolve_scene_size,
    rotate_centered_point,
    set_window_mouse_visible,
    software_stimulus_rotation,
)
from bin.task_lifecycle import USER_EXIT_CODE


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Path to JSON config file")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--fullscreen", action="store_true", default=None)
    parser.add_argument("--win_size", type=int, nargs=2, default=None)
    parser.add_argument("--main_screen", default=None)
    parser.add_argument("--experimenter_screen", default=None)
    parser.add_argument("--video_files", nargs="+", default=None)
    parser.add_argument("--video_size", type=int, nargs=2, default=None)
    parser.add_argument("--clip_duration_seconds", type=float, default=None)
    parser.add_argument("--frame_rate", type=float, default=None)
    parser.add_argument("--seek_timeout_seconds", type=float, default=None)
    parser.add_argument("--video_buffer_megabytes", type=float, default=None)
    parser.add_argument("--ffprobe", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--uart_device", default=None, help="Pi 5 3-pin UART: /dev/ttyAMA10"
    )
    parser.add_argument(
        "--simulate_eye",
        action="store_true",
        default=None,
        help="Simulate centered eye data without UART",
    )
    parser.add_argument(
        "--raspi", action="store_true", default=None, help="Enable DAQC2 reward output"
    )
    return parser.parse_args()


def _deliver_reward_pulse(
    *, outputs, pump_pin, pulse_duration_s, logger, should_exit=lambda: False,
    on_wait=None,
):
    """Use DAQC2 logical on/off semantics and log actual software-timed edges.

    Clear the output even if logging or the wait raises. Poll the separate UI
    during the pulse so Exit can shorten a manual or fixation reward.
    """
    duration_s = float(pulse_duration_s)
    if not math.isfinite(duration_s) or duration_s < 0:
        raise ValueError("pump_pulse_time_seconds must be finite and nonnegative")
    if duration_s == 0:
        return time.perf_counter()
    requested_s = time.perf_counter()
    try:
        outputs.write(pump_pin, True)
        start_s = time.perf_counter()
        logger.log_signal(
            trial_num=None,
            event="pump_on",
            timestamp_perf_s=start_s,
            requested_timestamp_perf_s=requested_s,
            requested_duration=duration_s,
        )
        deadline = start_s + duration_s
        while time.perf_counter() < deadline:
            if should_exit():
                break
            if on_wait is not None:
                on_wait()
            core.wait(min(0.01, max(0, deadline - time.perf_counter())))
    finally:
        requested_off_s = time.perf_counter()
        outputs.write(pump_pin, False)
        logger.log_signal(
            trial_num=None,
            event="pump_off",
            timestamp_perf_s=time.perf_counter(),
            requested_timestamp_perf_s=requested_off_s,
        )
    return time.perf_counter()


def run_task(
    *,
    output_dir="logs",
    config_name="calibrate_eye_tracker",
    fullscreen=True,
    win_size=None,
    screen_config=None,
    bg=(128, 128, 128),
    fixation_color=(0, 0, 0),
    fixation_cross_size=40.0,
    fixation_cross_weight=6.0,
    uart_config=None,
    filter_config=None,
    initial_calibration=None,
    initial_gain_anchor_px=None,
    x_scale_limits=(-0.02, 0.02),
    y_scale_limits=(-0.02, 0.02),
    fix_diameter=0.05,
    reward_interval=2.0,
    raspi=False,
    daq_address=0,
    daq_module_name="piplates.DAQC2plate",
    pump_pin=0,
    pump_pulse_time_seconds=0.25,
    video_files=(),
    video_size=(100, 100),
    clip_duration_seconds=5.0,
    frame_rate=30.0,
    seek_timeout_seconds=30.0,
    video_buffer_megabytes=DEFAULT_BUFFER_BYTES / (1024 * 1024),
    ffprobe_bin="ffprobe",
    seed=None,
) -> Path:
    uart_config = uart_config or UARTEyeTrackerConfig()
    filter_config = filter_config or EyeFilterConfig()
    calibration = initial_calibration or EyeCalibration()
    for name, value in (
        ("fix_diameter", fix_diameter),
        ("pump_pulse_time_seconds", pump_pulse_time_seconds),
    ):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if not math.isfinite(reward_interval) or reward_interval <= 0:
        raise ValueError("reward_interval must be positive and finite")
    if reward_interval <= pump_pulse_time_seconds:
        raise ValueError("reward_interval must be longer than pump_pulse_time_seconds")
    for limits, value in (
        (x_scale_limits, calibration.x_scale),
        (y_scale_limits, calibration.y_scale),
    ):
        if (
            len(limits) != 2
            or not all(math.isfinite(v) for v in limits)
            or not limits[0] < limits[1]
        ):
            raise ValueError(
                "Scale limits must contain two finite numbers in increasing order"
            )
        if not limits[0] <= value <= limits[1]:
            raise ValueError("Initial scale is outside its slider limits")
    if not all(math.isfinite(value) for value in asdict(calibration).values()):
        raise ValueError("Calibration values must be finite")
    videos = CalibrationVideoClips(
        video_files=video_files, video_size=video_size,
        clip_duration_seconds=clip_duration_seconds, frame_rate=frame_rate,
        seek_timeout_seconds=seek_timeout_seconds, video_buffer_megabytes=video_buffer_megabytes,
        ffprobe_bin=ffprobe_bin, seed=seed, raspi=raspi,
    )

    # Validate DOUT addressing even in simulation, so a legacy GPIO pin (17)
    # cannot silently become an invalid plate output when the rig is enabled.
    outputs = DAQC2DigitalOutputs(
        address=daq_address, module_name=daq_module_name, enabled=raspi
    )
    pump_pin = outputs.validate_bit(pump_pin)
    session_logs = SessionLogBundle(
        output_root=Path(output_dir),
        task_name="calibrate_eye_tracker",
        config_name=config_name,
        auto_flush=False,
    )
    logger, msg_logger = session_logs.event_logger, session_logs.message_logger
    main_win = preview = tracker = None
    output_opened = False
    fixation = (0.0, 0.0)
    auto_reward = None
    last_revision = last_zero_count = 0
    last_attention_count = 0
    last_auto_reward_count = 0
    last_video_toggle_count = 0
    video_publisher = video_stim = None
    video_sequence = video_clip_number = 0
    last_video_error = None
    status = ""
    attention_started_s = None
    last_error = None
    latest_state = None
    gain_anchor_px = initial_gain_anchor_px
    exit_requested = False
    interrupted_error = None

    def save_calibration(path, *, error=None):
        # This is called only after setup has produced a sample, or on normal
        # exit. Geometry and calibration both belong to the main process, so
        # an aborted preview cannot prevent recovering the applied settings.
        payload = calibration_payload(
            calibration,
            uart_config=uart_config,
            filter_config=filter_config,
            main_screen_size=subject_size,
            fixation_fraction=fixation,
            latest_state=latest_state,
        )
        payload["eye_tracker_calibration"].update(
            {
                "saved_at": datetime.now().isoformat(timespec="seconds"),
                "main_native_size_px": list(main_size),
                "main_rotation_deg": rotation,
                "gain_anchor_px": gain_anchor_px,
                "completion_status": "interrupted" if error else "complete",
                "fixation_acceptance": {
                    "fix_diameter": fix_diameter,
                    "reward_interval": reward_interval,
                    "recent_duration_s": CalibrationAutoReward.RECENT_SECONDS,
                    "recent_fraction_threshold": CalibrationAutoReward.RECENT_THRESHOLD,
                    "interval_fraction_threshold": CalibrationAutoReward.INTERVAL_THRESHOLD,
                    "threshold_comparison": "strictly_greater",
                    "fraction_basis": "time",
                    "diameter_units": "fraction_of_shorter_subject_scene_dimension",
                },
            }
        )
        if error:
            payload["eye_tracker_calibration"]["interruption_error"] = error
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(path)
        msg_logger.log("INFO", f"eye_calibration_saved path={path}")

    try:
        outputs.open()
        output_opened = True
        outputs.write(pump_pin, False)
        msg_logger.log(
            "INFO",
            f"DAQC2 reward address={daq_address} pump_dout={pump_pin} enabled={int(raspi)}",
        )
        main_win, main_screen, experimenter_screen = utils.setup_task_window(
            screen_config,
            bg_rgb_255=bg,
            fullscreen=fullscreen,
            size=win_size,
            allow_same_screen=False,
        )
        if experimenter_screen is None:
            raise RuntimeError(
                "Eye calibration requires a separate experimenter screen for its controls"
            )
        msg_logger.log(
            "INFO",
            f"resolved_screens main={describe_screen(main_screen)} experimenter={describe_screen(experimenter_screen)}",
        )
        set_window_mouse_visible(main_win, False)
        # Use the same realized scene and software rotation as active_foraging,
        # including windowed mode. Fractions refer to what the subject sees.
        main_size = resolve_scene_size(
            main_screen,
            fullscreen=fullscreen,
            requested_size=win_size,
            realized_size=tuple(main_win.size),
        )
        rotation = software_stimulus_rotation(main_screen.rotation)
        subject_size = oriented_size(main_size, rotation)
        preview = ExperimenterPreview(
            experimenter_screen,
            task_label=config_name,
            update_interval_s=0.05,
            calibration_config={
                "subject_size": subject_size,
                "bg": bg,
                "fixation_color": fixation_color,
                "fixation_cross_size": fixation_cross_size,
                "fixation_cross_weight": fixation_cross_weight,
                "calibration": asdict(calibration),
                "x_scale_limits": x_scale_limits,
                "y_scale_limits": y_scale_limits,
                "video_size": video_size,
            },
        )
        utils.verify_task_window_vblank(main_win)
        pylogging.console.setLevel(pylogging.CRITICAL)
        msg_logger.log(
            "INFO",
            f"calibration_geometry native={main_size} subject={subject_size} rotation={rotation}",
        )
        auto_reward = CalibrationAutoReward(
            reward_interval=reward_interval, calibration=calibration,
            fixation=fixation, screen_size=subject_size, fix_diameter=fix_diameter,
            max_sample_age_s=uart_config.max_sample_age_s,
        )
        tracker = UARTEyeTracker(
            uart_config=uart_config, filter_config=filter_config,
            on_state=auto_reward.observe,
        )
        tracker.start()
        msg_logger.log(
            "INFO",
            f"eye_uart_started eye_id=0 config={asdict(uart_config)} filter={asdict(filter_config)}",
        )
        main_bg = utils.make_bg_rect(main_win, bg)
        cross = _make_fixation_cross(
            main_win,
            size=fixation_cross_size,
            weight=fixation_cross_weight,
            color_rgb=fixation_color,
        )
        for part in cross:
            part.ori = rotation

        def should_exit():
            nonlocal exit_requested
            exit_requested = (
                exit_requested or preview.poll() or bool(event.getKeys(["escape"]))
            )
            return exit_requested

        def draw_scene():
            nonlocal video_stim, video_publisher, video_sequence, video_clip_number
            position = rotate_centered_point(fraction_to_pixels(fixation, subject_size), rotation)
            frame = videos.frame()
            main_bg.draw()
            if videos.enabled:
                if video_stim is None:
                    video_stim = make_rgb_video_stim(main_win, video_size, rotation=rotation)
                    video_publisher = SharedVideoFrameBuffer(video_size[0] * video_size[1] * 3)
                if frame is not None and frame.sequence != video_sequence:
                    upload_rgb_texture(video_stim, frame.rgb)
                    video_publisher.publish_rgb(
                        frame.rgb, source_frame_index=frame.source_frame_index,
                        source_media_time_s=frame.source_pts_s,
                    )
                    video_sequence = frame.sequence
                    if frame.clip_number != video_clip_number:
                        video_clip_number = frame.clip_number
                        msg_logger.log(
                            "INFO", f"calibration_video clip={frame.clip_number} file={frame.path} "
                            f"source_pts_s={frame.source_pts_s:.6f} duration_s={clip_duration_seconds}",
                        )
                video_stim.pos = position
                video_stim.draw()
            else:
                _draw_fixation_cross(cross, position, attention_started_s=attention_started_s)
            main_win.flip()
            preview.show_calibration_state(
                eye_fraction=eye_fraction, eye_valid=latest_state.valid,
                status_text=status, attention_started_s=attention_started_s,
                video_visible=videos.enabled,
                video_descriptor=video_publisher.descriptor() if video_publisher else None,
                video_sequence=video_publisher.sequence if video_publisher else 0,
            )

        def reward():
            auto_reward.reward_delivered(now_s=time.perf_counter())
            _deliver_reward_pulse(
                outputs=outputs,
                pump_pin=pump_pin,
                pulse_duration_s=pump_pulse_time_seconds,
                logger=logger,
                should_exit=should_exit,
                # Keep the attention movie moving during the software-timed
                # pulse; the UART observation worker is already independent.
                on_wait=draw_scene if videos.enabled else None,
            )
            # The UART callback continues both observation windows during the
            # pulse. Do not discard that history or require a fixation break.

        while not should_exit():
            latest_state = tracker.get_state()
            if gain_anchor_px is None and latest_state.valid:
                # Hold one reference throughout slider drags. Zero explicitly
                # replaces it; moving the fixation target must not replace it.
                gain_anchor_px = latest_state.smoothed_position
            if latest_state.error != last_error:
                msg_logger.log(
                    "WARN" if latest_state.error else "INFO",
                    f"eye_uart_error={latest_state.error or 'cleared'}",
                )
                last_error = latest_state.error
            controls = preview.get_calibration_controls()
            # Each window owns its keyboard events. Merge subject-window keys
            # with the preview's requests before applying any action, so both
            # focus paths share zeroing, attention and reward behavior. Gain and
            # target steps go to the UI owner and return as a normal revision.
            main_keys = event.getKeys(CALIBRATION_KEYS)
            x_steps = main_keys.count("right") - main_keys.count("left")
            y_steps = main_keys.count("up") - main_keys.count("down")
            if x_steps or y_steps:
                preview.request_calibration_scale_steps(
                    x_steps=x_steps, y_steps=y_steps
                )
            fix_x_steps, fix_y_steps = fixation_key_steps(main_keys)
            if fix_x_steps or fix_y_steps:
                preview.request_calibration_fixation_steps(
                    x_steps=fix_x_steps, y_steps=fix_y_steps
                )
            video_count = int(controls.get("video_toggle_count", 0))
            toggles = max(main_keys.count("v"), video_count - last_video_toggle_count)
            last_video_toggle_count = video_count
            if toggles % 2:
                enabled = videos.toggle()
                msg_logger.log("INFO", f"calibration_video enabled={int(enabled)}")
            attention_count = controls.get("attention_count", 0)
            attention_requested = (
                "f" in main_keys or attention_count != last_attention_count
            )
            last_attention_count = attention_count
            controls_changed = controls["revision"] != last_revision
            zero_requested = (
                "z" in main_keys or controls["zero_count"] != last_zero_count
            )
            if controls_changed or zero_requested:
                if controls_changed:
                    if gain_anchor_px is not None:
                        calibration.set_gains(
                            x_scale=controls["x_scale"],
                            y_scale=controls["y_scale"],
                            anchor_px=gain_anchor_px,
                        )
                    else:
                        # No observed position exists to anchor yet. This only
                        # initializes sensitivity; invalid data never sets offsets.
                        calibration.x_scale = controls["x_scale"]
                        calibration.y_scale = controls["y_scale"]
                    fixation = (controls["fix_x"], controls["fix_y"])
                if zero_requested:
                    if latest_state.valid:
                        gain_anchor_px = latest_state.smoothed_position
                        calibration.set_offsets_for_fixation(
                            fixation_fraction=fixation,
                            x_px=latest_state.x_smooth_px,
                            y_px=latest_state.y_smooth_px,
                        )
                    else:
                        msg_logger.log(
                            "WARN", "offset_set_ignored: Eye 0 sample is invalid"
                        )
                last_revision, last_zero_count = (
                    controls["revision"],
                    controls["zero_count"],
                )
                auto_reward.set_mapping(calibration, fixation, now_s=time.perf_counter())
                msg_logger.log(
                    "INFO",
                    f"calibration_controls fixation={fixation} calibration={asdict(calibration)}",
                )
            eye_fraction = calibration.map_position(*latest_state.smoothed_position)
            in_window = latest_state.valid and fraction_position_within_diameter(
                eye_fraction,
                fixation,
                diameter_fraction=fix_diameter,
                screen_size=subject_size,
            )
            auto_count = int(controls.get("auto_reward_count", 0))
            toggle_count = max(main_keys.count("a"), auto_count - last_auto_reward_count)
            last_auto_reward_count = auto_count
            if toggle_count % 2:
                enabled = auto_reward.toggle(now_s=time.perf_counter())
                msg_logger.log(
                    "INFO", f"auto_reward enabled={int(enabled)} reward_interval={reward_interval}"
                )
            auto_status = auto_reward.status(now_s=time.perf_counter())
            # Drain both sources every frame; simultaneous requests produce
            # one pulse rather than leaving a second queued keyboard reward.
            manual_reward = preview.consume_manual_reward_request() or "r" in main_keys
            if manual_reward or (auto_status.due and in_window):
                msg_logger.log(
                    "INFO",
                    f"reward source={'manual' if manual_reward else 'auto'} "
                    f"fixation_recent={auto_status.recent_fraction:.6f} "
                    f"fixation_interval={auto_status.interval_fraction:.6f}",
                )
                reward()
            if attention_requested:
                # Start with the next draw (after any reward pulse). A fresh F
                # request restarts the one-second cue rather than queuing it.
                attention_started_s = time.perf_counter()
                msg_logger.log(
                    "INFO", "fixation_attention duration_s=1 frequency_hz=5 scale=2"
                )
            status = (
                f"Eye 0 ({latest_state.x_smooth_px:.2f}, {latest_state.y_smooth_px:.2f}) px  "
                f"auto {'ON' if auto_status.enabled else 'OFF'} (A)\n"
                f"fix 100ms: {auto_status.recent_fraction * 100:.1f}%  "
                f"{reward_interval:g}s: {auto_status.interval_fraction * 100:.1f}%"
            )
            if not latest_state.valid:
                status += f"\n{latest_state.error or latest_state.rejection_reason}"
            if not raspi:
                status += "\nReward output disabled (raspi: false)"
            status += f"\nVideo {'ON' if videos.enabled else 'OFF'} (V)"
            if videos.error:
                status += f": {videos.error}"
            if videos.error != last_video_error:
                last_video_error = videos.error
                if last_video_error:
                    msg_logger.log("WARN", f"calibration_video_error: {last_video_error}")
            draw_scene()

        save_calibration(session_logs.calibration_path)
        return session_logs.calibration_path
    except Exception as exc:
        interrupted_error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        # Reward is off before potentially slow camera/recording shutdown. Each
        # cleanup is independent so a UART failure cannot strand either window.
        if output_opened:
            try:
                outputs.write(pump_pin, False)
            except Exception as exc:
                msg_logger.log("ERROR", f"pump_cleanup_failed: {exc}")
        if interrupted_error is not None and latest_state is not None:
            # Preserve work without importing an interrupted run as a completed
            # calibration. It can be explicitly reloaded via calibration_file.
            try:
                recovery_path = session_logs.calibration_path.with_name(
                    "calibration_recovery.json"
                )
                save_calibration(recovery_path, error=interrupted_error)
            except Exception as exc:
                msg_logger.log("ERROR", f"eye_calibration_recovery_failed: {exc}")
        if tracker is not None:
            try:
                tracker.stop()
            except Exception as exc:
                msg_logger.log("ERROR", f"eye_uart_cleanup_failed: {exc}")
        try:
            videos.close()
        except Exception as exc:
            msg_logger.log("ERROR", f"calibration_video_cleanup_failed: {exc}")
        try:
            if preview is not None:
                preview.close()
        finally:
            try:
                try:
                    if video_publisher is not None:
                        video_publisher.close()
                finally:
                    if main_win is not None:
                        utils.close_task_window(main_win)
            finally:
                session_logs.close()


def main():
    args = parse_args()
    cfg = load_config(args.config) if args.config else {}
    daq = cfg.get("daq", {})
    if any(
        key in daq
        for key in (
            "x_channel",
            "y_channel",
            "voltage_min",
            "voltage_max",
            "sample_rate_hz",
        )
    ):
        raise ValueError(
            "Analog eye input config is obsolete; use uart and recalibrate in crop-local pixels"
        )
    settings = load_eye_tracking_settings(cfg, for_calibration=True)
    # Apply CLI overrides after loading, including inherited UART settings.
    overrides = {}
    if args.uart_device is not None:
        overrides["device"] = args.uart_device
    if args.simulate_eye is not None:
        overrides["simulate"] = args.simulate_eye
    uart_config = replace(settings.uart, **overrides)
    video_options = {}
    for name, default in (
        ("video_files", ()), ("video_size", (100, 100)), ("clip_duration_seconds", 5.0),
        ("frame_rate", 30.0), ("seek_timeout_seconds", 30.0),
        ("video_buffer_megabytes", DEFAULT_BUFFER_BYTES / (1024 * 1024)),
        ("ffprobe", "ffprobe"), ("seed", None),
    ):
        value = getattr(args, name, None)
        video_options["ffprobe_bin" if name == "ffprobe" else name] = (
            value if value is not None else cfg.get(name, default)
        )
    path = run_task(
        output_dir=args.output_dir or cfg.get("output_dir", "logs"),
        config_name=cfg.get("config_name", "calibrate_eye_tracker"),
        fullscreen=(
            args.fullscreen
            if args.fullscreen is not None
            else cfg.get("fullscreen", True)
        ),
        win_size=args.win_size or cfg.get("win_size"),
        screen_config=load_screen_config(
            cfg, cli_main=args.main_screen, cli_experimenter=args.experimenter_screen
        ),
        bg=tuple(cfg.get("bg", (128, 128, 128))),
        fixation_color=tuple(cfg.get("fixation_color", (0, 0, 0))),
        fixation_cross_size=float(
            cfg.get("fixation_cross_size", cfg.get("fixation_size", 40))
        ),
        fixation_cross_weight=float(cfg.get("fixation_cross_weight", 6)),
        uart_config=uart_config,
        filter_config=settings.filter,
        initial_calibration=settings.calibration,
        initial_gain_anchor_px=settings.gain_anchor_px,
        x_scale_limits=cfg.get("x_scale_limits", (-0.02, 0.02)),
        y_scale_limits=cfg.get("y_scale_limits", (-0.02, 0.02)),
        fix_diameter=float(cfg.get("fix_diameter", 0.05)),
        reward_interval=float(cfg.get("reward_interval", cfg.get("fix_accept_time") or 2)),
        raspi=args.raspi if args.raspi is not None else cfg.get("raspi", False),
        daq_address=int(daq.get("address", cfg.get("daq_address", 0))),
        daq_module_name=daq.get(
            "module_name", cfg.get("daq_module_name", "piplates.DAQC2plate")
        ),
        pump_pin=int(cfg.get("pump_pin", 0)),
        pump_pulse_time_seconds=float(cfg.get("pump_pulse_time_seconds", 0.25)),
        **video_options,
    )
    print(f"Saved eye calibration to {path}")
    sys.exit(USER_EXIT_CODE)


if __name__ == "__main__":
    main()
