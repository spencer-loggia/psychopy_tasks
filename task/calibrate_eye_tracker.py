#!/usr/bin/env python3
"""Calibrate Eye 0 from the Pi 5 UART with a separate experimenter display."""
from __future__ import annotations

import argparse
from dataclasses import asdict
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
from bin.eye_calibration_preview import _make_fixation_cross, _draw_fixation_cross
from bin.eye_tracking import (
    UARTEyeTracker,
    UARTEyeTrackerConfig,
    EyeCalibration,
    EyeFilterConfig,
    FixationWindow,
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
    *, outputs, pump_pin, pulse_duration_s, logger, should_exit=lambda: False
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
    x_scale_limits=(-0.02, 0.02),
    y_scale_limits=(-0.02, 0.02),
    fix_diameter=0.05,
    fix_accept_percent=0.95,
    fix_accept_time=2.0,
    raspi=False,
    daq_address=0,
    daq_module_name="piplates.DAQC2plate",
    pump_pin=0,
    pump_pulse_time_seconds=0.25,
) -> Path:
    uart_config = uart_config or UARTEyeTrackerConfig()
    filter_config = filter_config or EyeFilterConfig()
    calibration = initial_calibration or EyeCalibration()
    for name, value in (
        ("fix_diameter", fix_diameter),
        ("fix_accept_time", fix_accept_time),
        ("pump_pulse_time_seconds", pump_pulse_time_seconds),
    ):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    accept_fraction = (
        fix_accept_percent / 100 if fix_accept_percent > 1 else fix_accept_percent
    )
    if not math.isfinite(accept_fraction) or not 0 <= accept_fraction <= 1:
        raise ValueError("fix_accept_percent must be in [0, 1] or [0, 100]")
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
    window = FixationWindow(
        fix_accept_time, accept_fraction, max_gap_s=uart_config.max_sample_age_s
    )
    auto_reward_armed = True
    last_revision = last_zero_count = 0
    last_error = None
    latest_state = None
    exit_requested = False
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
            },
        )
        utils.verify_task_window_vblank(main_win)
        pylogging.console.setLevel(pylogging.CRITICAL)
        msg_logger.log(
            "INFO",
            f"calibration_geometry native={main_size} subject={subject_size} rotation={rotation}",
        )
        tracker = UARTEyeTracker(uart_config=uart_config, filter_config=filter_config)
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

        def reward():
            _deliver_reward_pulse(
                outputs=outputs,
                pump_pin=pump_pin,
                pulse_duration_s=pump_pulse_time_seconds,
                logger=logger,
                should_exit=should_exit,
            )
            # The presentation loop cannot count samples while waiting for the
            # pump. Require a fresh full observation window after each pulse.
            window.clear()

        while not should_exit():
            latest_state = tracker.get_state()
            if latest_state.error != last_error:
                msg_logger.log(
                    "WARN" if latest_state.error else "INFO",
                    f"eye_uart_error={latest_state.error or 'cleared'}",
                )
                last_error = latest_state.error
            controls = preview.get_calibration_controls()
            if controls["revision"] != last_revision:
                calibration.x_scale, calibration.y_scale = (
                    controls["x_scale"],
                    controls["y_scale"],
                )
                fixation = (controls["fix_x"], controls["fix_y"])
                if controls["zero_count"] != last_zero_count:
                    if latest_state.valid:
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
                window.clear()
                auto_reward_armed = True
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
            accepted = window.update(latest_state, in_window)
            manual_reward = preview.consume_manual_reward_request() or bool(
                event.getKeys(["r"])
            )
            if manual_reward or (accepted and auto_reward_armed):
                msg_logger.log(
                    "INFO",
                    f"reward source={'manual' if manual_reward else 'fixation'} acceptance={window.acceptance:.6f}",
                )
                reward()
                auto_reward_armed = False
            elif not in_window:
                auto_reward_armed = True
            status = (
                f"Eye 0 ({latest_state.x_smooth_px:.2f}, {latest_state.y_smooth_px:.2f}) px  "
                f"fix {window.acceptance * 100:.1f}%"
            )
            if not latest_state.valid:
                status += f"\n{latest_state.error or latest_state.rejection_reason}"
            preview.show_calibration_state(
                eye_fraction=eye_fraction,
                eye_valid=latest_state.valid,
                status_text=status,
            )
            main_bg.draw()
            _draw_fixation_cross(
                cross,
                rotate_centered_point(
                    fraction_to_pixels(fixation, subject_size), rotation
                ),
            )
            main_win.flip()

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
                "fixation_acceptance": {
                    "fix_diameter": fix_diameter,
                    "fix_accept_percent": accept_fraction,
                    "fix_accept_time": fix_accept_time,
                    "diameter_units": "fraction_of_shorter_subject_scene_dimension",
                },
            }
        )
        session_logs.calibration_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        msg_logger.log(
            "INFO", f"eye_calibration_saved path={session_logs.calibration_path}"
        )
        return session_logs.calibration_path
    finally:
        # Reward is off before potentially slow camera/recording shutdown. Each
        # cleanup is independent so a UART failure cannot strand either window.
        if output_opened:
            try:
                outputs.write(pump_pin, False)
            except Exception as exc:
                msg_logger.log("ERROR", f"pump_cleanup_failed: {exc}")
        if tracker is not None:
            try:
                tracker.stop()
            except Exception as exc:
                msg_logger.log("ERROR", f"eye_uart_cleanup_failed: {exc}")
        try:
            if preview is not None:
                preview.close()
        finally:
            try:
                if main_win is not None:
                    utils.close_task_window(main_win)
            finally:
                session_logs.close()


def main():
    args = parse_args()
    cfg = load_config(args.config) if args.config else {}
    uart = dict(cfg.get("uart", {}))
    if args.uart_device is not None:
        uart["device"] = args.uart_device
    if args.simulate_eye is not None:
        uart["simulate"] = args.simulate_eye
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
    # The experiment manager flattens unset history values as JSON null. Those
    # must fall back to the task's initial settings, not become float(None).
    values = {}
    for key, default in asdict(EyeCalibration()).items():
        value = cfg.get(key)
        if value is None:
            value = cfg.get(f"initial_{key}", default)
        values[key] = float(default if value is None else value)
    calibration = EyeCalibration(**values)
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
        uart_config=UARTEyeTrackerConfig(**uart),
        filter_config=EyeFilterConfig(**cfg.get("eye_filter", {})),
        initial_calibration=calibration,
        x_scale_limits=cfg.get("x_scale_limits", (-0.02, 0.02)),
        y_scale_limits=cfg.get("y_scale_limits", (-0.02, 0.02)),
        fix_diameter=float(cfg.get("fix_diameter", 0.05)),
        fix_accept_percent=float(cfg.get("fix_accept_percent", 0.95)),
        fix_accept_time=float(cfg.get("fix_accept_time", 2)),
        raspi=args.raspi if args.raspi is not None else cfg.get("raspi", False),
        daq_address=int(daq.get("address", cfg.get("daq_address", 0))),
        daq_module_name=daq.get(
            "module_name", cfg.get("daq_module_name", "piplates.DAQC2plate")
        ),
        pump_pin=int(cfg.get("pump_pin", 0)),
        pump_pulse_time_seconds=float(cfg.get("pump_pulse_time_seconds", 0.25)),
    )
    print(f"Saved eye calibration to {path}")
    sys.exit(USER_EXIT_CODE)


if __name__ == "__main__":
    main()
