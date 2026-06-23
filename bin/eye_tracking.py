"""
Reusable analog eye-tracker sampling and calibration helpers.

Coordinates are represented as screen fractions relative to the real main
screen dimensions, centered at (0, 0). A value of x=-0.5 is the left edge,
x=0.5 is the right edge, y=-0.5 is the bottom edge, and y=0.5 is the top edge.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib
import math
import multiprocessing as mp
import queue
import time
from typing import Optional, Sequence, Tuple


_STATE_FIELDS = (
    "timestamp_perf_s",
    "x_raw_voltage",
    "y_raw_voltage",
    "x_voltage",
    "y_voltage",
    "x_smooth_voltage",
    "y_smooth_voltage",
    "x_fraction",
    "y_fraction",
    "sample_index",
    "accepted_count",
    "rejected_count",
    "consecutive_rejected",
    "last_rejection_code",
)
_REJECTION_REASONS = {
    0: None,
    1: "nonfinite",
    2: "out_of_range",
    3: "voltage_step",
}


@dataclass(frozen=True)
class EyeTrackerState:
    timestamp_perf_s: float
    x_raw_voltage: float
    y_raw_voltage: float
    x_voltage: float
    y_voltage: float
    x_smooth_voltage: float
    y_smooth_voltage: float
    x_fraction: float
    y_fraction: float
    sample_index: int
    accepted_count: int
    rejected_count: int
    consecutive_rejected: int
    last_rejected: bool = False
    rejection_reason: Optional[str] = None
    error: Optional[str] = None

    @property
    def position(self) -> Tuple[float, float]:
        return (float(self.x_fraction), float(self.y_fraction))

    @property
    def smoothed_voltage(self) -> Tuple[float, float]:
        return (float(self.x_smooth_voltage), float(self.y_smooth_voltage))


@dataclass
class DAQC2AnalogConfig:
    address: int = 0
    x_channel: int = 0
    y_channel: int = 1
    sample_rate_hz: float = 240.0
    voltage_min: float = -10.0
    voltage_max: float = 10.0
    module_name: str = "piplates.DAQC2plate"
    simulate: bool = False


@dataclass
class EyeFilterConfig:
    ema_gamma: float = 0.98
    reject_blink_artifacts: bool = True
    max_voltage_step: Optional[float] = 8.0

    def normalized_gamma(self) -> float:
        return max(0.0, min(0.999999, float(self.ema_gamma)))


@dataclass
class EyeCalibration:
    x_scale: float = 0.05
    y_scale: float = 0.05
    x_offset: float = 0.0
    y_offset: float = 0.0

    def map_voltages(self, x_voltage: float, y_voltage: float) -> Tuple[float, float]:
        return (
            (float(self.x_scale) * float(x_voltage)) + float(self.x_offset),
            (float(self.y_scale) * float(y_voltage)) + float(self.y_offset),
        )

    def set_offsets_for_fixation(
        self,
        *,
        fixation_fraction: Sequence[float],
        x_voltage: float,
        y_voltage: float,
    ) -> None:
        self.x_offset = float(fixation_fraction[0]) - (float(self.x_scale) * float(x_voltage))
        self.y_offset = float(fixation_fraction[1]) - (float(self.y_scale) * float(y_voltage))

    def to_json_dict(self) -> dict:
        return {
            "x_scale": float(self.x_scale),
            "y_scale": float(self.y_scale),
            "x_offset": float(self.x_offset),
            "y_offset": float(self.y_offset),
            "coordinate_units": "screen_fraction_centered",
            "coordinate_definition": {
                "x": "screen_fraction * main_screen_width_px, center origin; visible range [-0.5, 0.5]",
                "y": "screen_fraction * main_screen_height_px, center origin; visible range [-0.5, 0.5]",
            },
        }


class DAQC2AnalogReader:
    """Small adapter around piplates.DAQC2plate analog reads."""

    def __init__(self, config: DAQC2AnalogConfig):
        self.config = config
        self._module = None

    def open(self) -> None:
        if self.config.simulate:
            return
        self._validate_channel(self.config.x_channel, "x_channel")
        self._validate_channel(self.config.y_channel, "y_channel")
        self._module = importlib.import_module(str(self.config.module_name))
        if not hasattr(self._module, "getADC"):
            raise RuntimeError(f"{self.config.module_name} does not expose getADC(address, channel)")

    def read_pair(self) -> Tuple[float, float]:
        if self.config.simulate:
            return (0.0, 0.0)
        if self._module is None:
            self.open()
        return (
            float(self._module.getADC(int(self.config.address), int(self.config.x_channel))),
            float(self._module.getADC(int(self.config.address), int(self.config.y_channel))),
        )

    @staticmethod
    def _validate_channel(value: int, name: str) -> None:
        channel = int(value)
        if channel < 0 or channel > 7:
            raise ValueError(f"{name} must be an analog input channel in the range 0 through 7")


class EyePositionFilter:
    """Maintains blink-rejected exponential moving average eye voltage."""

    def __init__(
        self,
        *,
        calibration: EyeCalibration,
        daq_config: DAQC2AnalogConfig,
        filter_config: EyeFilterConfig,
    ):
        self.calibration = calibration
        self.daq_config = daq_config
        self.filter_config = filter_config
        self._smooth_x: Optional[float] = None
        self._smooth_y: Optional[float] = None
        self._last_accepted_x: Optional[float] = None
        self._last_accepted_y: Optional[float] = None
        self._sample_index = 0
        self._accepted_count = 0
        self._rejected_count = 0
        self._consecutive_rejected = 0
        self._last_state = EyeTrackerState(
            timestamp_perf_s=time.perf_counter(),
            x_raw_voltage=0.0,
            y_raw_voltage=0.0,
            x_voltage=0.0,
            y_voltage=0.0,
            x_smooth_voltage=0.0,
            y_smooth_voltage=0.0,
            x_fraction=0.0,
            y_fraction=0.0,
            sample_index=0,
            accepted_count=0,
            rejected_count=0,
            consecutive_rejected=0,
        )

    def update_calibration(self, calibration: EyeCalibration) -> None:
        self.calibration = calibration
        self._last_state = self._state_from_current(
            timestamp_perf_s=self._last_state.timestamp_perf_s,
            x_raw_voltage=self._last_state.x_raw_voltage,
            y_raw_voltage=self._last_state.y_raw_voltage,
            last_rejected=self._last_state.last_rejected,
            rejection_reason=self._last_state.rejection_reason,
        )

    def update(self, x_voltage: float, y_voltage: float, *, timestamp_perf_s: Optional[float] = None) -> EyeTrackerState:
        self._sample_index += 1
        now_s = time.perf_counter() if timestamp_perf_s is None else float(timestamp_perf_s)
        raw_x = float(x_voltage)
        raw_y = float(y_voltage)
        rejection_reason = self._rejection_reason(raw_x, raw_y)

        if rejection_reason is not None:
            self._rejected_count += 1
            self._consecutive_rejected += 1
            self._last_state = self._state_from_current(
                timestamp_perf_s=now_s,
                x_raw_voltage=raw_x,
                y_raw_voltage=raw_y,
                last_rejected=True,
                rejection_reason=rejection_reason,
            )
            return self._last_state

        self._accepted_count += 1
        self._consecutive_rejected = 0
        gamma = self.filter_config.normalized_gamma()
        if self._smooth_x is None or self._smooth_y is None:
            self._smooth_x = raw_x
            self._smooth_y = raw_y
        else:
            self._smooth_x = (gamma * self._smooth_x) + ((1.0 - gamma) * raw_x)
            self._smooth_y = (gamma * self._smooth_y) + ((1.0 - gamma) * raw_y)
        self._last_accepted_x = raw_x
        self._last_accepted_y = raw_y
        self._last_state = self._state_from_current(
            timestamp_perf_s=now_s,
            x_raw_voltage=raw_x,
            y_raw_voltage=raw_y,
            last_rejected=False,
            rejection_reason=None,
        )
        return self._last_state

    def _rejection_reason(self, x_voltage: float, y_voltage: float) -> Optional[str]:
        if not self.filter_config.reject_blink_artifacts:
            return None
        if not math.isfinite(x_voltage) or not math.isfinite(y_voltage):
            return "nonfinite"

        lo = min(float(self.daq_config.voltage_min), float(self.daq_config.voltage_max))
        hi = max(float(self.daq_config.voltage_min), float(self.daq_config.voltage_max))
        if x_voltage < lo or x_voltage > hi or y_voltage < lo or y_voltage > hi:
            return "out_of_range"

        max_step = self.filter_config.max_voltage_step
        if max_step is not None and float(max_step) > 0 and self._last_accepted_x is not None and self._last_accepted_y is not None:
            if max(abs(x_voltage - self._last_accepted_x), abs(y_voltage - self._last_accepted_y)) > float(max_step):
                return "voltage_step"
        return None

    def _state_from_current(
        self,
        *,
        timestamp_perf_s: float,
        x_raw_voltage: float,
        y_raw_voltage: float,
        last_rejected: bool,
        rejection_reason: Optional[str],
    ) -> EyeTrackerState:
        smooth_x = float(self._smooth_x) if self._smooth_x is not None else 0.0
        smooth_y = float(self._smooth_y) if self._smooth_y is not None else 0.0
        mapped_x, mapped_y = self.calibration.map_voltages(smooth_x, smooth_y)
        return EyeTrackerState(
            timestamp_perf_s=float(timestamp_perf_s),
            x_raw_voltage=float(x_raw_voltage),
            y_raw_voltage=float(y_raw_voltage),
            x_voltage=float(self._last_accepted_x) if self._last_accepted_x is not None else 0.0,
            y_voltage=float(self._last_accepted_y) if self._last_accepted_y is not None else 0.0,
            x_smooth_voltage=smooth_x,
            y_smooth_voltage=smooth_y,
            x_fraction=float(mapped_x),
            y_fraction=float(mapped_y),
            sample_index=int(self._sample_index),
            accepted_count=int(self._accepted_count),
            rejected_count=int(self._rejected_count),
            consecutive_rejected=int(self._consecutive_rejected),
            last_rejected=bool(last_rejected),
            rejection_reason=rejection_reason,
        )


class AnalogEyeTracker:
    """Process-backed analog eye tracker exposing smoothed screen-fraction position."""

    def __init__(
        self,
        *,
        daq_config: Optional[DAQC2AnalogConfig] = None,
        calibration: Optional[EyeCalibration] = None,
        filter_config: Optional[EyeFilterConfig] = None,
    ):
        self.daq_config = daq_config or DAQC2AnalogConfig()
        self.calibration = calibration or EyeCalibration()
        self.filter_config = filter_config or EyeFilterConfig()
        self._ctx = mp.get_context("spawn")
        self._state = self._ctx.Array("d", [0.0] * len(_STATE_FIELDS), lock=True)
        self._calibration = self._ctx.Array(
            "d",
            [
                float(self.calibration.x_scale),
                float(self.calibration.y_scale),
                float(self.calibration.x_offset),
                float(self.calibration.y_offset),
            ],
            lock=True,
        )
        self._stop_event = self._ctx.Event()
        self._status_queue = self._ctx.Queue(maxsize=16)
        self._process: Optional[mp.Process] = None
        self._last_error: Optional[str] = None

    def start(self) -> None:
        if self._process is not None and self._process.is_alive():
            return
        self._stop_event.clear()
        self._process = self._ctx.Process(
            target=_eye_tracker_worker,
            name="AnalogEyeTracker",
            args=(
                self.daq_config,
                self.filter_config,
                self._calibration,
                self._state,
                self._stop_event,
                self._status_queue,
            ),
            daemon=True,
        )
        self._process.start()

    def stop(self, *, timeout_s: float = 1.0) -> None:
        self._stop_event.set()
        if self._process is not None:
            self._process.join(timeout=max(0.0, float(timeout_s)))
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=0.5)

    def update_calibration(
        self,
        *,
        x_scale: Optional[float] = None,
        y_scale: Optional[float] = None,
        x_offset: Optional[float] = None,
        y_offset: Optional[float] = None,
    ) -> EyeCalibration:
        with self._calibration.get_lock():
            values = list(self._calibration[:])
            if x_scale is not None:
                values[0] = float(x_scale)
            if y_scale is not None:
                values[1] = float(y_scale)
            if x_offset is not None:
                values[2] = float(x_offset)
            if y_offset is not None:
                values[3] = float(y_offset)
            self._calibration[:] = values
        self.calibration = EyeCalibration(*values)
        return self.calibration

    def set_offsets_for_fixation(self, fixation_fraction: Sequence[float], state: Optional[EyeTrackerState] = None) -> EyeCalibration:
        current_state = state or self.get_state()
        calibration = self.get_calibration()
        calibration.set_offsets_for_fixation(
            fixation_fraction=fixation_fraction,
            x_voltage=current_state.x_smooth_voltage,
            y_voltage=current_state.y_smooth_voltage,
        )
        return self.update_calibration(x_offset=calibration.x_offset, y_offset=calibration.y_offset)

    def get_calibration(self) -> EyeCalibration:
        with self._calibration.get_lock():
            values = list(self._calibration[:])
        self.calibration = EyeCalibration(*values)
        return self.calibration

    def get_state(self) -> EyeTrackerState:
        self._drain_status_queue()
        with self._state.get_lock():
            values = list(self._state[:])
        payload = dict(zip(_STATE_FIELDS, values))
        code = int(payload["last_rejection_code"])
        return EyeTrackerState(
            timestamp_perf_s=float(payload["timestamp_perf_s"]),
            x_raw_voltage=float(payload["x_raw_voltage"]),
            y_raw_voltage=float(payload["y_raw_voltage"]),
            x_voltage=float(payload["x_voltage"]),
            y_voltage=float(payload["y_voltage"]),
            x_smooth_voltage=float(payload["x_smooth_voltage"]),
            y_smooth_voltage=float(payload["y_smooth_voltage"]),
            x_fraction=float(payload["x_fraction"]),
            y_fraction=float(payload["y_fraction"]),
            sample_index=int(payload["sample_index"]),
            accepted_count=int(payload["accepted_count"]),
            rejected_count=int(payload["rejected_count"]),
            consecutive_rejected=int(payload["consecutive_rejected"]),
            last_rejected=code != 0,
            rejection_reason=_REJECTION_REASONS.get(code),
            error=self._last_error,
        )

    def get_position(self, *, clamp: bool = False) -> Tuple[float, float]:
        state = self.get_state()
        if not clamp:
            return state.position
        return (clamp_fraction(state.x_fraction), clamp_fraction(state.y_fraction))

    def _drain_status_queue(self) -> None:
        while True:
            try:
                message = self._status_queue.get_nowait()
            except queue.Empty:
                break
            except Exception:
                break
            if not isinstance(message, dict):
                continue
            if message.get("type") == "error":
                self._last_error = str(message.get("message", ""))
            elif message.get("type") == "ok":
                self._last_error = None


def _eye_tracker_worker(
    daq_config: DAQC2AnalogConfig,
    filter_config: EyeFilterConfig,
    calibration_shared,
    state_shared,
    stop_event,
    status_queue,
) -> None:
    reader = DAQC2AnalogReader(daq_config)
    eye_filter = EyePositionFilter(
        calibration=_read_calibration(calibration_shared),
        daq_config=daq_config,
        filter_config=filter_config,
    )
    sample_rate_hz = max(1.0, float(daq_config.sample_rate_hz))
    period_s = 1.0 / sample_rate_hz
    next_sample_s = time.perf_counter()
    opened = False
    last_error_message: Optional[str] = None

    while not stop_event.is_set():
        now_s = time.perf_counter()
        if now_s < next_sample_s:
            time.sleep(min(period_s, max(0.0, next_sample_s - now_s)))
            continue

        if not opened:
            try:
                reader.open()
                opened = True
                last_error_message = None
                _put_status(status_queue, {"type": "ok"})
            except Exception as exc:
                message = str(exc)
                if message != last_error_message:
                    _put_status(status_queue, {"type": "error", "message": message})
                    last_error_message = message
                next_sample_s = time.perf_counter() + 1.0
                continue

        try:
            raw_x, raw_y = reader.read_pair()
            eye_filter.update_calibration(_read_calibration(calibration_shared))
            state = eye_filter.update(raw_x, raw_y, timestamp_perf_s=time.perf_counter())
            _write_state(state_shared, state)
        except Exception as exc:
            message = str(exc)
            if message != last_error_message:
                _put_status(status_queue, {"type": "error", "message": message})
                last_error_message = message
            opened = False

        next_sample_s += period_s
        if next_sample_s < time.perf_counter() - period_s:
            next_sample_s = time.perf_counter() + period_s


def _read_calibration(calibration_shared) -> EyeCalibration:
    with calibration_shared.get_lock():
        values = list(calibration_shared[:])
    return EyeCalibration(
        x_scale=float(values[0]),
        y_scale=float(values[1]),
        x_offset=float(values[2]),
        y_offset=float(values[3]),
    )


def _write_state(state_shared, state: EyeTrackerState) -> None:
    rejection_code = 0
    for code, reason in _REJECTION_REASONS.items():
        if reason == state.rejection_reason:
            rejection_code = int(code)
            break
    values = [
        float(state.timestamp_perf_s),
        float(state.x_raw_voltage),
        float(state.y_raw_voltage),
        float(state.x_voltage),
        float(state.y_voltage),
        float(state.x_smooth_voltage),
        float(state.y_smooth_voltage),
        float(state.x_fraction),
        float(state.y_fraction),
        float(state.sample_index),
        float(state.accepted_count),
        float(state.rejected_count),
        float(state.consecutive_rejected),
        float(rejection_code),
    ]
    with state_shared.get_lock():
        state_shared[:] = values


def _put_status(status_queue, message: dict) -> None:
    try:
        status_queue.put_nowait(message)
    except Exception:
        pass


def fraction_to_pixels(fraction_pos: Sequence[float], screen_size: Sequence[float]) -> Tuple[float, float]:
    return (
        float(fraction_pos[0]) * max(float(screen_size[0]), 1.0),
        float(fraction_pos[1]) * max(float(screen_size[1]), 1.0),
    )


def pixels_to_fraction(pixel_pos: Sequence[float], screen_size: Sequence[float]) -> Tuple[float, float]:
    return (
        float(pixel_pos[0]) / max(float(screen_size[0]), 1.0),
        float(pixel_pos[1]) / max(float(screen_size[1]), 1.0),
    )


def clamp_fraction(value: float, *, limit: float = 0.5) -> float:
    if not math.isfinite(float(value)):
        return 0.0
    return max(-float(limit), min(float(limit), float(value)))


def fraction_distance_px(
    pos_a: Sequence[float],
    pos_b: Sequence[float],
    screen_size: Sequence[float],
) -> float:
    dx_px = (float(pos_a[0]) - float(pos_b[0])) * max(float(screen_size[0]), 1.0)
    dy_px = (float(pos_a[1]) - float(pos_b[1])) * max(float(screen_size[1]), 1.0)
    return math.hypot(dx_px, dy_px)


def fraction_position_within_diameter(
    position_fraction: Sequence[float],
    target_fraction: Sequence[float],
    *,
    diameter_fraction: float,
    screen_size: Sequence[float],
) -> bool:
    if not math.isfinite(float(diameter_fraction)) or float(diameter_fraction) <= 0:
        return False
    radius_px = 0.5 * float(diameter_fraction) * min(
        max(float(screen_size[0]), 1.0),
        max(float(screen_size[1]), 1.0),
    )
    return fraction_distance_px(position_fraction, target_fraction, screen_size) <= radius_px


def calibration_payload(
    calibration: EyeCalibration,
    *,
    daq_config: DAQC2AnalogConfig,
    filter_config: EyeFilterConfig,
    main_screen_size: Sequence[int],
    fixation_fraction: Sequence[float],
    latest_state: Optional[EyeTrackerState] = None,
) -> dict:
    payload = calibration.to_json_dict()
    payload.update(
        {
            "daq": asdict(daq_config),
            "filter": asdict(filter_config),
            "main_screen_size_px": [int(main_screen_size[0]), int(main_screen_size[1])],
            "fixation_fraction": [float(fixation_fraction[0]), float(fixation_fraction[1])],
        }
    )
    if latest_state is not None:
        payload["latest_state"] = asdict(latest_state)
    return payload
