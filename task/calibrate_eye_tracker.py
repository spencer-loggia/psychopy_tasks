#!/usr/bin/env python3
"""
Eye-tracker calibration task.

Reads two analog channels from a Pi-Plates DAQC2plate, maps voltage to
main-screen centered screen-fraction coordinates, and lets the experimenter
adjust x/y scale and offsets interactively.
"""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, Optional, Sequence, Tuple

from psychopy import core, event, logging as pylogging, visual

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from bin import utils
from bin.config import load_config
from bin.eye_tracking import (
    AnalogEyeTracker,
    DAQC2AnalogConfig,
    EyeCalibration,
    EyeFilterConfig,
    calibration_payload,
    clamp_fraction,
    fraction_position_within_diameter,
    fraction_to_pixels,
)
from bin.logger import SessionLogBundle
from bin.screen import describe_screen, load_screen_config, resolve_scene_size, resolve_task_screens


class Slider:
    def __init__(
        self,
        win,
        *,
        orientation: str,
        pos: Tuple[float, float],
        length: float,
        value_min: float,
        value_max: float,
        value: float,
        label: str,
    ):
        self.win = win
        self.orientation = orientation
        self.pos = (float(pos[0]), float(pos[1]))
        self.length = max(1.0, float(length))
        self.value_min = float(value_min)
        self.value_max = float(value_max)
        self.value = float(value)
        self.label = str(label)
        self.track = visual.Rect(
            win,
            width=self.length if self.orientation == "horizontal" else 6.0,
            height=6.0 if self.orientation == "horizontal" else self.length,
            pos=self.pos,
            fillColor=utils.rgb255_to_psychopy((70, 70, 70)),
            fillColorSpace="rgb",
            lineColor=utils.rgb255_to_psychopy((210, 210, 210)),
            lineColorSpace="rgb",
            units="pix",
        )
        self.knob = visual.Rect(
            win,
            width=24.0,
            height=24.0,
            pos=self._knob_pos(),
            fillColor=utils.rgb255_to_psychopy((245, 245, 245)),
            fillColorSpace="rgb",
            lineColor=utils.rgb255_to_psychopy((30, 30, 30)),
            lineColorSpace="rgb",
            units="pix",
        )
        self.text = visual.TextStim(
            win,
            text="",
            units="pix",
            height=18.0,
            color=utils.rgb255_to_psychopy((245, 245, 245)),
            colorSpace="rgb",
            pos=(0, 0),
        )

    def _fraction(self) -> float:
        span = self.value_max - self.value_min
        if span == 0:
            return 0.5
        return max(0.0, min(1.0, (float(self.value) - self.value_min) / span))

    def _knob_pos(self) -> Tuple[float, float]:
        frac = self._fraction()
        if self.orientation == "horizontal":
            return (self.pos[0] - (self.length * 0.5) + (frac * self.length), self.pos[1])
        return (self.pos[0], self.pos[1] - (self.length * 0.5) + (frac * self.length))

    def contains(self, point: Sequence[float]) -> bool:
        px, py = float(point[0]), float(point[1])
        kx, ky = self._knob_pos()
        if abs(px - kx) <= 24.0 and abs(py - ky) <= 24.0:
            return True
        if self.orientation == "horizontal":
            return abs(py - self.pos[1]) <= 18.0 and abs(px - self.pos[0]) <= self.length * 0.5
        return abs(px - self.pos[0]) <= 18.0 and abs(py - self.pos[1]) <= self.length * 0.5

    def set_from_pos(self, point: Sequence[float]) -> None:
        if self.orientation == "horizontal":
            frac = (float(point[0]) - (self.pos[0] - self.length * 0.5)) / self.length
        else:
            frac = (float(point[1]) - (self.pos[1] - self.length * 0.5)) / self.length
        frac = max(0.0, min(1.0, frac))
        self.value = self.value_min + (frac * (self.value_max - self.value_min))

    def draw(self) -> None:
        self.knob.pos = self._knob_pos()
        self.track.draw()
        self.knob.draw()
        self.text.text = f"{self.label} {self.value:.4f}"
        if self.orientation == "horizontal":
            self.text.pos = (self.pos[0], self.pos[1] + 30.0)
        else:
            self.text.pos = (self.pos[0] + 54.0, self.pos[1])
        self.text.draw()


def parse_args():
    parser = argparse.ArgumentParser(description="Calibrate eye-tracker analog voltage mapping")
    parser.add_argument("--config", help="Path to JSON config file")
    parser.add_argument("--output_dir", default=None, help="Output directory for logs/calibration JSON")
    parser.add_argument("--fullscreen", action="store_true", default=None, help="Use fullscreen PsychoPy windows")
    parser.add_argument("--win_size", type=int, nargs=2, default=None, help="Main window size when not fullscreen")
    parser.add_argument("--main_screen", default=None, help="Main task screen index or output name")
    parser.add_argument("--experimenter_screen", default=None, help="Experimenter screen index or output name")
    parser.add_argument("--simulate_eye", action="store_true", default=None, help="Use zero-voltage simulated DAQ input")
    parser.add_argument("--raspi", action="store_true", default=None, help="Enable Raspberry Pi GPIO pump output")
    return parser.parse_args()


def _get_nested(cfg: Dict[str, Any], section: str, key: str, default: Any = None) -> Any:
    nested = cfg.get(section, {})
    if isinstance(nested, dict) and key in nested:
        return nested[key]
    return cfg.get(key, default)


def _as_rgb(value: Any, default: Tuple[int, int, int]) -> Tuple[int, int, int]:
    if value is None:
        return default
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return default
    return tuple(max(0, min(255, int(v))) for v in value)


def _make_fixation_cross(win, *, size: float, weight: float, color_rgb: Sequence[int]):
    size = max(1.0, float(size))
    weight = max(1.0, min(size, float(weight)))
    color = utils.rgb255_to_psychopy(tuple(int(v) for v in color_rgb))
    horizontal = visual.Rect(
        win,
        width=size,
        height=weight,
        fillColor=color,
        fillColorSpace="rgb",
        lineColor=None,
        units="pix",
    )
    vertical = visual.Rect(
        win,
        width=weight,
        height=size,
        fillColor=color,
        fillColorSpace="rgb",
        lineColor=None,
        units="pix",
    )
    return (horizontal, vertical)


def _draw_fixation_cross(parts, pos: Sequence[float]) -> None:
    for part in parts:
        part.pos = (float(pos[0]), float(pos[1]))
        part.draw()


def _make_button(win, *, text: str, pos: Tuple[float, float], size: Tuple[float, float], color_rgb):
    rect = visual.Rect(
        win,
        width=float(size[0]),
        height=float(size[1]),
        pos=pos,
        fillColor=utils.rgb255_to_psychopy(color_rgb),
        fillColorSpace="rgb",
        lineColor=utils.rgb255_to_psychopy((20, 20, 20)),
        lineColorSpace="rgb",
        units="pix",
    )
    label = visual.TextStim(
        win,
        text=text,
        units="pix",
        height=max(18.0, min(float(size[0]), float(size[1])) * 0.45),
        pos=pos,
        color=utils.rgb255_to_psychopy((255, 255, 255)),
        colorSpace="rgb",
    )
    return rect, label


def _draw_button(button) -> None:
    rect, label = button
    rect.draw()
    label.draw()


def _compute_preview_layout(exp_size: Sequence[int], main_size: Sequence[int]) -> dict:
    exp_w, exp_h = max(float(exp_size[0]), 1.0), max(float(exp_size[1]), 1.0)
    main_w, main_h = max(float(main_size[0]), 1.0), max(float(main_size[1]), 1.0)

    left_controls = max(86.0, exp_w * 0.10)
    right_pad = max(22.0, exp_w * 0.025)
    top_controls = max(74.0, exp_h * 0.11)
    bottom_controls = max(86.0, exp_h * 0.13)
    usable_w = max(100.0, exp_w - left_controls - right_pad)
    usable_h = max(100.0, exp_h - top_controls - bottom_controls)

    box_aspect = main_w / main_h
    box_w = usable_w
    box_h = box_w / box_aspect
    if box_h > usable_h:
        box_h = usable_h
        box_w = box_h * box_aspect

    usable_center_x = -exp_w * 0.5 + left_controls + (usable_w * 0.5)
    usable_center_y = -exp_h * 0.5 + bottom_controls + (usable_h * 0.5)
    return {
        "box_center": (usable_center_x, usable_center_y),
        "box_size": (box_w, box_h),
        "left_controls": left_controls,
        "right_pad": right_pad,
        "top_controls": top_controls,
        "bottom_controls": bottom_controls,
    }


def _real_screen_size(screen_info, fallback_size: Sequence[int]) -> Tuple[int, int]:
    if screen_info is not None and int(screen_info.width) > 0 and int(screen_info.height) > 0:
        return (int(screen_info.width), int(screen_info.height))
    return (max(int(fallback_size[0]), 1), max(int(fallback_size[1]), 1))


def _fraction_to_preview(
    fraction_pos: Sequence[float],
    *,
    box_center: Sequence[float],
    box_size: Sequence[float],
    clamp: bool = True,
) -> Tuple[float, float]:
    fx = float(fraction_pos[0])
    fy = float(fraction_pos[1])
    if clamp:
        fx = clamp_fraction(fx)
        fy = clamp_fraction(fy)
    return (
        float(box_center[0]) + (fx * float(box_size[0])),
        float(box_center[1]) + (fy * float(box_size[1])),
    )


def _preview_to_fraction(
    point: Sequence[float],
    *,
    box_center: Sequence[float],
    box_size: Sequence[float],
) -> Tuple[float, float]:
    return (
        clamp_fraction((float(point[0]) - float(box_center[0])) / max(float(box_size[0]), 1.0)),
        clamp_fraction((float(point[1]) - float(box_center[1])) / max(float(box_size[1]), 1.0)),
    )


def _open_pump_gpio(raspi: bool, pump_pin: int, msg_logger):
    if not raspi:
        msg_logger.log("INFO", "raspi=False; reward button will log pump events without GPIO output")
        return None, None
    try:
        import lgpio

        chip = lgpio.gpiochip_open(0)
        lgpio.gpio_claim_output(chip, int(pump_pin))
        msg_logger.log("INFO", f"lgpio initialized on chip 0, pump_pin={int(pump_pin)}")
        return lgpio, chip
    except Exception as exc:
        msg_logger.log("WARN", f"lgpio not available or failed to initialize pump pin: {exc}; reward events logged only")
        return None, None


def _deliver_reward_pulse(
    *,
    lgpio_module,
    gpio_chip,
    pump_pin: int,
    pulse_duration_s: float,
    logger,
    msg_logger,
) -> float:
    duration_s = max(0.0, float(pulse_duration_s))
    start_perf = time.perf_counter()
    if lgpio_module is not None and gpio_chip is not None:
        try:
            lgpio_module.gpio_write(gpio_chip, int(pump_pin), 1)
        except Exception as exc:
            msg_logger.log("ERROR", f"Failed to set pump_pin high: {exc}")
    logger.log_signal(
        trial_num=None,
        event="pump_on",
        timestamp_perf_s=start_perf,
        requested_duration=duration_s,
    )

    used_hardware_timing = False
    if lgpio_module is not None and gpio_chip is not None and duration_s > 0:
        try:
            result = lgpio_module.tx_pulse(gpio_chip, int(pump_pin), 0, int(duration_s * 1_000_000), 0, 1)
            if result < 0:
                raise RuntimeError(f"tx_pulse failed with code {result}")
            used_hardware_timing = True
        except Exception as exc:
            msg_logger.log("WARN", f"hardware_timed_pump_pulse_failed error={exc}; falling back to blocking wait")

    if used_hardware_timing:
        logger.log_signal(
            trial_num=None,
            event="pump_off",
            timestamp_perf_s=start_perf + duration_s,
        )
    else:
        core.wait(duration_s)
        if lgpio_module is not None and gpio_chip is not None:
            try:
                lgpio_module.gpio_write(gpio_chip, int(pump_pin), 0)
            except Exception as exc:
                msg_logger.log("ERROR", f"Failed to set pump_pin low: {exc}")
        logger.log_signal(
            trial_num=None,
            event="pump_off",
            timestamp_perf_s=time.perf_counter(),
        )
    return start_perf + duration_s


def run_task(
    *,
    output_dir: str = "logs",
    config_name: str = "calibrate_eye_tracker",
    fullscreen: bool = False,
    win_size: Optional[Tuple[int, int]] = None,
    experimenter_win_size: Optional[Tuple[int, int]] = None,
    screen_config: Optional[Dict[str, Any]] = None,
    bg: Tuple[int, int, int] = (128, 128, 128),
    fixation_color: Tuple[int, int, int] = (0, 0, 0),
    fixation_cross_size: float = 40.0,
    fixation_cross_weight: float = 6.0,
    daq_config: Optional[DAQC2AnalogConfig] = None,
    filter_config: Optional[EyeFilterConfig] = None,
    initial_calibration: Optional[EyeCalibration] = None,
    x_scale_limits: Tuple[float, float] = (-0.20, 0.20),
    y_scale_limits: Tuple[float, float] = (-0.20, 0.20),
    fix_diameter: float = 0.05,
    fix_accept_percent: float = 0.95,
    fix_accept_time: float = 2.0,
    raspi: bool = False,
    pump_pin: int = 17,
    pump_pulse_time_seconds: float = 0.25,
) -> Path:
    resolved_config_name = str(config_name or "calibrate_eye_tracker")
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    session_logs = SessionLogBundle(
        output_root=output_root,
        task_name="calibrate_eye_tracker",
        config_name=resolved_config_name,
        auto_flush=False,
    )
    logger = session_logs.event_logger
    msg_logger = session_logs.message_logger
    msg_logger.log("INFO", f"session_start task=calibrate_eye_tracker config_name={resolved_config_name}")

    main_win = None
    exp_win = None
    tracker = None
    gpio_chip = None
    lgpio_module = None
    calibration_path = output_root / f"{datetime.now().strftime('%Y%m%d%H%M%S')}_eye_calibration.json"

    calibration = initial_calibration or EyeCalibration()
    daq_config = daq_config or DAQC2AnalogConfig()
    filter_config = filter_config or EyeFilterConfig()
    fixation_window_diameter = max(0.0, float(fix_diameter))
    fixation_accept_fraction = float(fix_accept_percent)
    if fixation_accept_fraction > 1.0:
        fixation_accept_fraction = fixation_accept_fraction / 100.0
    fixation_accept_fraction = max(0.0, min(1.0, fixation_accept_fraction))
    fixation_accept_time_s = max(0.0, float(fix_accept_time))
    fixation_fraction = [0.0, 0.0]
    latest_state = None

    try:
        main_screen, experimenter_screen = resolve_task_screens(screen_config)
        if experimenter_screen is None:
            raise RuntimeError("calibrate_eye_tracker requires a configured or detected experimenter screen")
        msg_logger.log(
            "INFO",
            f"resolved_screens main={describe_screen(main_screen)} experimenter={describe_screen(experimenter_screen)}",
        )

        main_win = utils.setup_window(bg_rgb_255=bg, fullscreen=fullscreen, size=win_size, screen_info=main_screen)
        exp_win = utils.setup_window(
            bg_rgb_255=(30, 30, 30),
            fullscreen=fullscreen,
            size=experimenter_win_size,
            screen_info=experimenter_screen,
        )
        pylogging.console.setLevel(pylogging.CRITICAL)
        try:
            main_win.mouseVisible = False
        except Exception:
            pass

        main_size = _real_screen_size(main_screen, tuple(main_win.size))
        exp_size = resolve_scene_size(
            experimenter_screen,
            fullscreen=bool(fullscreen),
            requested_size=experimenter_win_size,
            realized_size=tuple(exp_win.size),
        )
        layout = _compute_preview_layout(exp_size, main_size)
        box_center = layout["box_center"]
        box_size = layout["box_size"]
        msg_logger.log(
            "INFO",
            (
                f"eye_calibration_geometry main_size={main_size[0]}x{main_size[1]} "
                f"experimenter_size={exp_size[0]}x{exp_size[1]} "
                f"preview_box_center=({box_center[0]:.1f},{box_center[1]:.1f}) "
                f"preview_box_size=({box_size[0]:.1f},{box_size[1]:.1f})"
            ),
        )
        msg_logger.log(
            "INFO",
            (
                f"fixation_acceptance fix_diameter={fixation_window_diameter:.9f} "
                f"fix_accept_percent={fixation_accept_fraction:.6f} "
                f"fix_accept_time={fixation_accept_time_s:.6f}"
            ),
        )

        tracker = AnalogEyeTracker(
            daq_config=daq_config,
            calibration=calibration,
            filter_config=filter_config,
        )
        tracker.start()
        msg_logger.log(
            "INFO",
            (
                f"daq_sampler_started address={daq_config.address} "
                f"x_channel={daq_config.x_channel} y_channel={daq_config.y_channel} "
                f"sample_rate_hz={float(daq_config.sample_rate_hz):.3f} "
                f"ema_gamma={float(filter_config.ema_gamma):.6f} "
                f"simulate={int(daq_config.simulate)}"
            ),
        )

        lgpio_module, gpio_chip = _open_pump_gpio(bool(raspi), int(pump_pin), msg_logger)

        main_bg = utils.make_bg_rect(main_win, bg)
        exp_bg = utils.make_bg_rect(exp_win, (30, 30, 30))
        preview_box = visual.Rect(
            exp_win,
            width=box_size[0],
            height=box_size[1],
            pos=box_center,
            fillColor=utils.rgb255_to_psychopy(bg),
            fillColorSpace="rgb",
            lineColor=utils.rgb255_to_psychopy((230, 230, 230)),
            lineColorSpace="rgb",
            lineWidth=2,
            units="pix",
        )
        main_cross = _make_fixation_cross(
            main_win,
            size=fixation_cross_size,
            weight=fixation_cross_weight,
            color_rgb=fixation_color,
        )
        exp_cross = _make_fixation_cross(
            exp_win,
            size=max(6.0, float(fixation_cross_size) * (float(box_size[0]) / max(float(main_size[0]), 1.0))),
            weight=max(2.0, float(fixation_cross_weight) * (float(box_size[0]) / max(float(main_size[0]), 1.0))),
            color_rgb=fixation_color,
        )
        eye_dot = visual.Circle(
            exp_win,
            radius=max(4.0, min(float(exp_size[0]), float(exp_size[1])) * 0.008),
            fillColor=utils.rgb255_to_psychopy((40, 120, 255)),
            fillColorSpace="rgb",
            lineColor=utils.rgb255_to_psychopy((255, 255, 255)),
            lineColorSpace="rgb",
            units="pix",
        )

        button_w = max(68.0, float(exp_size[0]) * 0.08)
        button_h = max(42.0, float(exp_size[1]) * 0.07)
        reward_button = _make_button(
            exp_win,
            text="reward",
            pos=(-float(exp_size[0]) * 0.5 + button_w * 0.5 + 18.0, float(exp_size[1]) * 0.5 - button_h * 0.5 - 18.0),
            size=(button_w, button_h),
            color_rgb=(50, 150, 80),
        )
        exit_button = _make_button(
            exp_win,
            text="exit",
            pos=(float(exp_size[0]) * 0.5 - button_w * 0.5 - 18.0, float(exp_size[1]) * 0.5 - button_h * 0.5 - 18.0),
            size=(button_w, button_h),
            color_rgb=(200, 55, 55),
        )
        zero_button = _make_button(
            exp_win,
            text="x",
            pos=(-float(exp_size[0]) * 0.5 + 44.0, -float(exp_size[1]) * 0.5 + 44.0),
            size=(52.0, 52.0),
            color_rgb=(75, 75, 75),
        )

        x_slider = Slider(
            exp_win,
            orientation="horizontal",
            pos=(box_center[0], -float(exp_size[1]) * 0.5 + max(34.0, layout["bottom_controls"] * 0.42)),
            length=box_size[0],
            value_min=x_scale_limits[0],
            value_max=x_scale_limits[1],
            value=calibration.x_scale,
            label="x scale",
        )
        y_slider = Slider(
            exp_win,
            orientation="vertical",
            pos=(-float(exp_size[0]) * 0.5 + max(38.0, layout["left_controls"] * 0.42), box_center[1]),
            length=box_size[1],
            value_min=y_scale_limits[0],
            value_max=y_scale_limits[1],
            value=calibration.y_scale,
            label="y scale",
        )
        status_text = visual.TextStim(
            exp_win,
            text="",
            units="pix",
            height=18.0,
            pos=(box_center[0], float(exp_size[1]) * 0.5 - button_h - 28.0),
            color=utils.rgb255_to_psychopy((245, 245, 245)),
            colorSpace="rgb",
        )

        mouse = event.Mouse(win=exp_win)
        last_mouse_down = False
        dragging_slider: Optional[Slider] = None
        reward_block_until_s = 0.0
        last_sample_error = None
        exit_requested = False
        fixation_history = deque()
        auto_reward_armed = True
        last_fix_acceptance = 0.0

        while not exit_requested:
            if event.getKeys(["escape"]):
                msg_logger.log("INFO", "exit_requested_by_escape")
                break

            latest_state = tracker.get_state()
            if latest_state.error and latest_state.error != last_sample_error:
                msg_logger.log("WARN", f"daq_sample_error sample_index={latest_state.sample_index} error={latest_state.error}")
                last_sample_error = latest_state.error
            elif latest_state.error is None and last_sample_error is not None:
                msg_logger.log("INFO", "daq_sample_error_cleared")
                last_sample_error = None

            calibration = tracker.update_calibration(
                x_scale=float(x_slider.value),
                y_scale=float(y_slider.value),
            )
            eye_fraction = calibration.map_voltages(
                latest_state.x_smooth_voltage,
                latest_state.y_smooth_voltage,
            )
            main_fix_pos = fraction_to_pixels(fixation_fraction, main_size)
            exp_fix_pos = _fraction_to_preview(fixation_fraction, box_center=box_center, box_size=box_size)
            exp_eye_pos = _fraction_to_preview(eye_fraction, box_center=box_center, box_size=box_size)
            now_s = time.perf_counter()
            eye_in_fix_window = fraction_position_within_diameter(
                eye_fraction,
                fixation_fraction,
                diameter_fraction=fixation_window_diameter,
                screen_size=main_size,
            )
            fixation_history.append((now_s, bool(eye_in_fix_window)))
            cutoff_s = now_s - fixation_accept_time_s
            while len(fixation_history) > 1 and fixation_history[1][0] <= cutoff_s:
                fixation_history.popleft()
            if fixation_history and fixation_accept_time_s > 0.0:
                window_age_s = fixation_history[-1][0] - fixation_history[0][0]
                fixation_sample_count = len(fixation_history)
                fixation_hit_count = sum(1 for _sample_s, in_window in fixation_history if in_window)
                last_fix_acceptance = fixation_hit_count / float(fixation_sample_count)
                fixation_window_ready = window_age_s >= fixation_accept_time_s
            else:
                fixation_sample_count = 0
                fixation_hit_count = 0
                last_fix_acceptance = 0.0
                fixation_window_ready = False
            fixation_accepted = fixation_window_ready and last_fix_acceptance >= fixation_accept_fraction
            if fixation_accepted and auto_reward_armed and now_s >= reward_block_until_s:
                reward_block_until_s = _deliver_reward_pulse(
                    lgpio_module=lgpio_module,
                    gpio_chip=gpio_chip,
                    pump_pin=int(pump_pin),
                    pulse_duration_s=float(pump_pulse_time_seconds),
                    logger=logger,
                    msg_logger=msg_logger,
                )
                auto_reward_armed = False
                msg_logger.log(
                    "INFO",
                    (
                        f"auto_fixation_reward acceptance={last_fix_acceptance:.6f} "
                        f"hits={fixation_hit_count} samples={fixation_sample_count} "
                        f"fix_x_fraction={fixation_fraction[0]:.9f} fix_y_fraction={fixation_fraction[1]:.9f}"
                    ),
                )
            elif not eye_in_fix_window and not fixation_accepted:
                auto_reward_armed = True

            try:
                mouse_down = bool(mouse.getPressed()[0])
            except Exception:
                mouse_down = False
            mouse_pos = mouse.getPos()

            if mouse_down:
                if not last_mouse_down:
                    if exit_button[0].contains(mouse_pos):
                        msg_logger.log("INFO", "exit_requested_by_button")
                        exit_requested = True
                    elif reward_button[0].contains(mouse_pos):
                        if now_s >= reward_block_until_s:
                            reward_block_until_s = _deliver_reward_pulse(
                                lgpio_module=lgpio_module,
                                gpio_chip=gpio_chip,
                                pump_pin=int(pump_pin),
                                pulse_duration_s=float(pump_pulse_time_seconds),
                                logger=logger,
                                msg_logger=msg_logger,
                            )
                    elif zero_button[0].contains(mouse_pos):
                        calibration = tracker.set_offsets_for_fixation(fixation_fraction, state=latest_state)
                        fixation_history.clear()
                        auto_reward_armed = True
                        last_fix_acceptance = 0.0
                        msg_logger.log(
                            "INFO",
                            (
                                f"eye_calibration_offset_set x_offset={calibration.x_offset:.9f} "
                                f"y_offset={calibration.y_offset:.9f} "
                                f"x_smooth_voltage={latest_state.x_smooth_voltage:.6f} "
                                f"y_smooth_voltage={latest_state.y_smooth_voltage:.6f}"
                            ),
                        )
                    elif x_slider.contains(mouse_pos):
                        dragging_slider = x_slider
                        dragging_slider.set_from_pos(mouse_pos)
                        fixation_history.clear()
                        auto_reward_armed = True
                        last_fix_acceptance = 0.0
                    elif y_slider.contains(mouse_pos):
                        dragging_slider = y_slider
                        dragging_slider.set_from_pos(mouse_pos)
                        fixation_history.clear()
                        auto_reward_armed = True
                        last_fix_acceptance = 0.0
                    elif preview_box.contains(mouse_pos):
                        fx, fy = _preview_to_fraction(mouse_pos, box_center=box_center, box_size=box_size)
                        fixation_fraction = [fx, fy]
                        fixation_history.clear()
                        auto_reward_armed = True
                        last_fix_acceptance = 0.0
                        msg_logger.log("INFO", f"fixation_position_set x_fraction={fx:.9f} y_fraction={fy:.9f}")
                elif dragging_slider is not None:
                    dragging_slider.set_from_pos(mouse_pos)
                    fixation_history.clear()
                    auto_reward_armed = True
                    last_fix_acceptance = 0.0
            else:
                dragging_slider = None
            last_mouse_down = mouse_down

            main_bg.draw()
            _draw_fixation_cross(main_cross, main_fix_pos)

            exp_bg.draw()
            preview_box.draw()
            _draw_fixation_cross(exp_cross, exp_fix_pos)
            eye_dot.pos = exp_eye_pos
            eye_dot.draw()
            x_slider.draw()
            y_slider.draw()
            _draw_button(reward_button)
            _draw_button(exit_button)
            _draw_button(zero_button)
            status_text.text = (
                f"eye ({latest_state.x_smooth_voltage:.3f} V, {latest_state.y_smooth_voltage:.3f} V)  "
                f"mapped ({eye_fraction[0]:.4f}, {eye_fraction[1]:.4f})  "
                f"fix {last_fix_acceptance * 100.0:.1f}%"
            )
            if latest_state.last_rejected:
                status_text.text += f"  rejected {latest_state.rejection_reason}"
            elif latest_state.error:
                status_text.text += "  daq error"
            status_text.draw()

            main_win.flip()
            exp_win.flip()

        calibration = tracker.get_calibration()
        payload = calibration_payload(
            calibration,
            daq_config=daq_config,
            filter_config=filter_config,
            main_screen_size=main_size,
            fixation_fraction=fixation_fraction,
            latest_state=latest_state,
        )
        payload["saved_at"] = datetime.now().isoformat(timespec="seconds")
        payload["fixation_acceptance"] = {
            "fix_diameter": fixation_window_diameter,
            "fix_accept_percent": fixation_accept_fraction,
            "fix_accept_time": fixation_accept_time_s,
            "diameter_units": "fraction_of_shorter_main_screen_dimension",
        }
        calibration_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        msg_logger.log("INFO", f"eye_calibration_saved path={calibration_path}")
        return calibration_path
    finally:
        if tracker is not None:
            tracker.stop()
        if lgpio_module is not None and gpio_chip is not None:
            try:
                lgpio_module.gpio_write(gpio_chip, int(pump_pin), 0)
                lgpio_module.gpiochip_close(gpio_chip)
            except Exception:
                pass
        try:
            session_logs.flush()
            session_logs.close()
        except Exception:
            pass
        for win in (main_win, exp_win):
            if win is not None:
                try:
                    win.close()
                except Exception:
                    pass


def main():
    args = parse_args()
    cfg: Dict[str, Any] = {}
    if args.config:
        cfg = load_config(args.config)

    screen_config = load_screen_config(
        cfg,
        cli_main=args.main_screen,
        cli_experimenter=args.experimenter_screen,
    )

    output_dir = args.output_dir if args.output_dir is not None else cfg.get("output_dir", "logs")
    fullscreen = bool(args.fullscreen if args.fullscreen is not None else cfg.get("fullscreen", False))
    win_size = args.win_size if args.win_size is not None else cfg.get("win_size")
    experimenter_win_size = cfg.get("experimenter_win_size")
    simulate_eye = bool(args.simulate_eye if args.simulate_eye is not None else _get_nested(cfg, "daq", "simulate", False))
    raspi = bool(args.raspi if args.raspi is not None else cfg.get("raspi", False))

    daq_config = DAQC2AnalogConfig(
        address=int(_get_nested(cfg, "daq", "address", cfg.get("daq_address", 0))),
        x_channel=int(
            _get_nested(
                cfg,
                "daq",
                "x_channel",
                _get_nested(cfg, "daq", "eye_pos_x_channel", cfg.get("eye_pos_x_channel", 0)),
            )
        ),
        y_channel=int(
            _get_nested(
                cfg,
                "daq",
                "y_channel",
                _get_nested(cfg, "daq", "eye_pos_y_channel", cfg.get("eye_pos_y_channel", 1)),
            )
        ),
        sample_rate_hz=float(_get_nested(cfg, "daq", "sample_rate_hz", 240.0)),
        voltage_min=float(_get_nested(cfg, "daq", "voltage_min", -10.0)),
        voltage_max=float(_get_nested(cfg, "daq", "voltage_max", 10.0)),
        module_name=str(_get_nested(cfg, "daq", "module_name", "piplates.DAQC2plate")),
        simulate=simulate_eye,
    )
    max_voltage_step = _get_nested(
        cfg,
        "eye_filter",
        "max_voltage_step",
        _get_nested(cfg, "filter", "max_voltage_step", cfg.get("blink_max_voltage_step", 8.0)),
    )
    filter_config = EyeFilterConfig(
        ema_gamma=float(
            _get_nested(
                cfg,
                "eye_filter",
                "ema_gamma",
                _get_nested(cfg, "filter", "ema_gamma", cfg.get("ema_gamma", 0.98)),
            )
        ),
        reject_blink_artifacts=bool(
            _get_nested(
                cfg,
                "eye_filter",
                "reject_blink_artifacts",
                _get_nested(cfg, "filter", "reject_blink_artifacts", cfg.get("reject_blink_artifacts", True)),
            )
        ),
        max_voltage_step=None if max_voltage_step is None else float(max_voltage_step),
    )

    calibration = EyeCalibration(
        x_scale=float(cfg.get("x_scale", cfg.get("initial_x_scale", 0.05))),
        y_scale=float(cfg.get("y_scale", cfg.get("initial_y_scale", 0.05))),
        x_offset=float(cfg.get("x_offset", cfg.get("initial_x_offset", 0.0))),
        y_offset=float(cfg.get("y_offset", cfg.get("initial_y_offset", 0.0))),
    )

    x_scale_limits = tuple(cfg.get("x_scale_limits", [-0.20, 0.20]))
    y_scale_limits = tuple(cfg.get("y_scale_limits", [-0.20, 0.20]))
    if len(x_scale_limits) != 2 or len(y_scale_limits) != 2:
        raise ValueError("x_scale_limits and y_scale_limits must each contain two numbers")

    path = run_task(
        output_dir=str(output_dir),
        config_name=str(cfg.get("config_name", "calibrate_eye_tracker")),
        fullscreen=fullscreen,
        win_size=tuple(win_size) if win_size is not None else None,
        experimenter_win_size=tuple(experimenter_win_size) if experimenter_win_size is not None else None,
        screen_config=screen_config,
        bg=_as_rgb(cfg.get("bg"), (128, 128, 128)),
        fixation_color=_as_rgb(cfg.get("fixation_color"), (0, 0, 0)),
        fixation_cross_size=float(cfg.get("fixation_cross_size", cfg.get("fixation_size", 40.0))),
        fixation_cross_weight=float(cfg.get("fixation_cross_weight", 6.0)),
        daq_config=daq_config,
        filter_config=filter_config,
        initial_calibration=calibration,
        x_scale_limits=(float(x_scale_limits[0]), float(x_scale_limits[1])),
        y_scale_limits=(float(y_scale_limits[0]), float(y_scale_limits[1])),
        fix_diameter=float(cfg.get("fix_diameter", 0.05)),
        fix_accept_percent=float(cfg.get("fix_accept_percent", 0.95)),
        fix_accept_time=float(cfg.get("fix_accept_time", 2.0)),
        raspi=raspi,
        pump_pin=int(cfg.get("pump_pin", 17)),
        pump_pulse_time_seconds=float(cfg.get("pump_pulse_time_seconds", 0.25)),
    )
    print(f"Saved eye calibration to {path}")


if __name__ == "__main__":
    main()
