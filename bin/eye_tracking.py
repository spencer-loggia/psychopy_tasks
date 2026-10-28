"""Eye-0 UART acquisition and crop-pixel to subject-screen calibration.

The camera reports crop-local pixels (x right, y down). Calibration produces
centered fractions of the *oriented subject scene*, with y increasing upward.
Serial I/O runs off the rendering thread; only new, valid camera frames enter
the moving average. See eye_tracker_wire/SOURCE.md for protocol provenance.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, replace
import logging
import math
import os
from pathlib import Path
import threading
import time
from typing import Optional, Sequence, Tuple

from .task_clock import TaskClock
from .eye_tracker_wire.controller import EyeTrackerController, ResponseTimeout, SerialPacketChannel
from .eye_tracker_warnings import EyeTrackerWarnings
from .eye_tracker_wire.protocol import (
    CommandCode,
    DecodedStatus,
    ProtocolError,
    StatusCode,
    StatusFlag,
    validate_run_number,
)

EXPERIMENT_UART_PENDING_ENV = "NEURO_EYE_TRACKER_EXPERIMENT_PENDING"
EXPERIMENT_UART_READY_ENV = "NEURO_EYE_TRACKER_EXPERIMENT_READY"
EXPERIMENT_BLOCK_NUMBER_ENV = "NEURO_EYE_TRACKER_BLOCK_NUMBER"
_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class UARTEyeTrackerConfig:
    device: str = "/dev/ttyAMA10"
    baud: int = 460800
    poll_rate_hz: float = 100.0
    max_sample_age_s: float = 0.15
    response_timeout_s: float = 0.25
    state_timeout_s: float = 10.0
    state_retries: int = 1
    manage_tracking: bool = True
    expect_recording: bool = True
    simulate: bool = False

    def __post_init__(self):
        for name in ("manage_tracking", "expect_recording", "simulate"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"uart.{name} must be true or false")
        for name in (
            "poll_rate_hz",
            "max_sample_age_s",
            "response_timeout_s",
            "state_timeout_s",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"uart.{name} must be positive and finite")
        if self.baud <= 0 or self.poll_rate_hz >= self.baud / 1280:
            raise ValueError(
                "uart.poll_rate_hz must fit a 128-byte 8-N-1 exchange at uart.baud"
            )
        if self.state_retries < 0:
            raise ValueError("uart.state_retries must be nonnegative")


@dataclass(frozen=True)
class EyeFilterConfig:
    # The source is about 50 camera frames/s. No extra smoothing
    # by default; 0.98 would add seconds of lag at the camera's frame rate.
    ema_gamma: float = 0.0

    def __post_init__(self):
        if not math.isfinite(self.ema_gamma) or not 0 <= self.ema_gamma < 1:
            raise ValueError("eye_filter.ema_gamma must be finite and in [0, 1)")


@dataclass
class EyeCalibration:
    x_scale: float = 0.005
    y_scale: float = -0.005
    x_offset: float = 0.0
    y_offset: float = 0.0
    x_cross: float = 0.0
    y_cross: float = 0.0

    def map_position(self, x_px: float, y_px: float) -> Tuple[float, float]:
        return (
            self.x_scale * x_px + self.x_cross * y_px + self.x_offset,
            self.y_cross * x_px + self.y_scale * y_px + self.y_offset,
        )

    def set_offsets_for_fixation(self, *, fixation_fraction, x_px, y_px) -> None:
        self.x_offset = (
            float(fixation_fraction[0]) - self.x_scale * x_px - self.x_cross * y_px
        )
        self.y_offset = (
            float(fixation_fraction[1]) - self.y_cross * x_px - self.y_scale * y_px
        )

    @staticmethod
    def _signed_row_gain(diagonal, cross):
        """Signed row length, with the diagonal breaking equal-size ties."""
        dominant = diagonal if abs(diagonal) >= abs(cross) else cross
        return math.copysign(math.hypot(diagonal, cross), dominant)

    @property
    def x_gain(self) -> float:
        """Sensitivity along subject x, including a rotated camera's cross term."""
        return self._signed_row_gain(self.x_scale, self.x_cross)

    @property
    def y_gain(self) -> float:
        """Sensitivity along subject y, including a rotated camera's cross term."""
        return self._signed_row_gain(self.y_scale, self.y_cross)

    def _gain_direction(self, axis):
        diagonal = getattr(self, f"{axis}_scale")
        cross = getattr(self, f"{axis}_cross")
        gain = self._signed_row_gain(diagonal, cross)
        if gain:
            return (diagonal / gain, cross / gain)
        return getattr(self, f"_{axis}_gain_direction", (1.0, 0.0))

    def restore_gain_directions(self, directions) -> None:
        """Restore optional zero-gain directions from saved calibration metadata.

        Serialized vectors use raw-input order ``[raw_x, raw_y]`` for each
        subject axis. Their unit length and canonical positive dominant entry
        keep slider gain values consistent after reloading a zero-gain row.
        Nonzero rows always derive their direction from the actual coefficients.
        """
        if not isinstance(directions, dict) or set(directions) != {"x", "y"}:
            raise ValueError("gain_directions must contain x and y direction vectors")
        validated = {}
        for axis in ("x", "y"):
            vector = directions[axis]
            if not isinstance(vector, (list, tuple)) or len(vector) != 2:
                raise ValueError(f"gain_directions.{axis} must contain two coordinates")
            try:
                if any(isinstance(value, bool) for value in vector):
                    raise ValueError
                raw_x, raw_y = map(float, vector)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"gain_directions.{axis} must be a finite unit vector") from exc
            norm = math.hypot(raw_x, raw_y)
            if not math.isfinite(norm) or not math.isclose(norm, 1.0, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError(f"gain_directions.{axis} must be a finite unit vector")
            diagonal, cross = (raw_x, raw_y) if axis == "x" else (raw_y, raw_x)
            if self._signed_row_gain(diagonal, cross) < 0:
                raise ValueError(f"gain_directions.{axis} must have a positive dominant component")
            validated[axis] = (diagonal / norm, cross / norm)
        for axis, direction in validated.items():
            setattr(self, f"_{axis}_gain_direction", direction)

    def set_gains(self, *, x_scale: float, y_scale: float, anchor_px) -> None:
        """Set signed subject-axis row gains, preserving a fixed eye mapping.

        The argument names preserve the original diagonal-calibration API. For
        affine maps, use ``x_gain`` / ``y_gain`` as the slider readout: each gain
        scales its entire affine row, preserving its direction. Keep that row's
        direction through zero so dragging back out of zero still works, even
        when a row consists entirely of a cross term. These direction caches
        are separate metadata, excluded from the six calibration coefficients.
        """
        target = self.map_position(*anchor_px)
        for axis, diagonal_name, cross_name, gain in (
            ("x", "x_scale", "x_cross", float(x_scale)),
            ("y", "y_scale", "y_cross", float(y_scale)),
        ):
            direction = self._gain_direction(axis)
            cache_name = f"_{axis}_gain_direction"
            setattr(self, cache_name, direction)
            setattr(self, diagonal_name, gain * direction[0])
            setattr(self, cross_name, gain * direction[1])
        self.set_offsets_for_fixation(
            fixation_fraction=target, x_px=anchor_px[0], y_px=anchor_px[1]
        )

    def to_json_dict(self) -> dict:
        x_direction = self._gain_direction("x")
        y_direction = self._gain_direction("y")
        return {
            **asdict(self),
            "gain_directions": {
                "x": list(x_direction),
                "y": [y_direction[1], y_direction[0]],
            },
            "input_units": "crop_local_pixels",
            "eye_id": 0,
            "coordinate_units": "screen_fraction_centered",
            "coordinate_definition": {
                "x": "fraction of oriented subject scene width; right positive; edges [-0.5, 0.5]",
                "y": "fraction of oriented subject scene height; up positive; edges [-0.5, 0.5]",
            },
        }


@dataclass(frozen=True)
class EyeTrackerState:
    # Internal freshness estimate only: receive time minus result age and RTT.
    # Logged times below preserve the local acquisition and remote camera clocks.
    timestamp_perf_s: float = 0.0
    task_time_ns: int = 0
    session_id: int = 0
    frame_sequence: int = 0
    camera_time_ns: int = 0
    x_raw_px: float = 0.0
    y_raw_px: float = 0.0
    x_smooth_px: float = 0.0
    y_smooth_px: float = 0.0
    pupil_diameter_px: float = 0.0
    confidence: float = 0.0
    valid: bool = False
    rejection_reason: Optional[str] = "waiting_for_sample"
    error: Optional[str] = None

    @property
    def smoothed_position(self) -> Tuple[float, float]:
        return (self.x_smooth_px, self.y_smooth_px)


class EyePositionFilter:
    """Validate status before deduplication, then smooth only Eye 0.

    A duplicate can still announce a blink, a stopped camera or stale data.
    Restarted services are latched unusable until the task restarts:
    their ROI configuration may have changed along with the service-start ID.
    """

    def __init__(self, *, uart_config=None, filter_config=None):
        self.config = uart_config or UARTEyeTrackerConfig()
        self.filter_config = filter_config or EyeFilterConfig()
        self.state = EyeTrackerState()
        self._session_id = None
        self._last_key = None
        self._smooth = None
        self._restarted = False

    def invalidate(self, reason, *, error=None):
        self._smooth = None
        self.state = replace(
            self.state,
            valid=False,
            rejection_reason=reason,
            error=error,
            pupil_diameter_px=0.0,
            confidence=0.0,
        )
        return self.state

    def consume(
        self, status: DecodedStatus, *, now_s: float, round_trip_s: float = 0.0,
        task_time_ns: int = 0,
    ):
        key = (status.session_id, status.frame_sequence, status.sensor_timestamp_ns)
        previous_key = self._last_key
        is_new = status.sensor_timestamp_ns != 0 and key != previous_key
        if (
            is_new and previous_key is not None
            and status.session_id == previous_key[0]
            and status.sensor_timestamp_ns <= previous_key[2]
        ):
            # Do not move the camera watermark backwards: replaying an older
            # response must not make a previously recorded frame new again.
            return self.invalidate("out_of_order")
        age_s = status.result_age_ms / 1000.0 + max(0.0, round_trip_s)
        if now_s - self.state.timestamp_perf_s > self.config.max_sample_age_s:
            self._smooth = None
        if is_new:
            self._last_key = key
            self.state = replace(
                self.state,
                task_time_ns=task_time_ns,
                session_id=status.session_id,
                frame_sequence=status.frame_sequence,
                camera_time_ns=status.sensor_timestamp_ns,
                timestamp_perf_s=now_s - age_s,
            )
        if self._session_id is None:
            self._session_id = status.session_id
        elif status.session_id != self._session_id:
            self._restarted = True
        if self._restarted:
            return self.invalidate(
                "service_restarted",
                error="Eye tracker restarted; restart task and check calibration",
            )
        if status.status != StatusCode.OK or status.flags & StatusFlag.ERROR:
            return self.invalidate(
                "tracker_error", error=f"Tracker status: {status.status.name}"
            )
        required = StatusFlag.TRACKING | StatusFlag.CONFIGURED
        if (
            status.flags & required != required
            or status.configured_eye_count < 1
            or status.flags & (StatusFlag.CALIBRATING | StatusFlag.SHUTTING_DOWN)
        ):
            return self.invalidate("not_tracking_or_configured")
        if (
            status.flags & StatusFlag.RESULT_STALE
            or age_s > self.config.max_sample_age_s
        ):
            return self.invalidate("stale")
        if status.sensor_timestamp_ns == 0:
            return self.invalidate("waiting_for_sample")

        # Eye 0 is the only experimental signal, even with two eyes configured.
        eye = status.eyes[0]
        if not eye.valid:
            return self.invalidate("blink" if eye.blink else "lost")
        if not is_new:
            return self.get_state(now_s=now_s)
        if self._smooth is None:
            self._smooth = (eye.x, eye.y)
        else:
            gamma = self.filter_config.ema_gamma
            self._smooth = tuple(
                gamma * old + (1 - gamma) * new
                for old, new in zip(self._smooth, (eye.x, eye.y))
            )
        self.state = replace(
            self.state,
            x_raw_px=eye.x,
            y_raw_px=eye.y,
            x_smooth_px=self._smooth[0],
            y_smooth_px=self._smooth[1],
            pupil_diameter_px=eye.pupil_diameter,
            confidence=eye.confidence,
            valid=True,
            rejection_reason=None,
            error=None,
        )
        return self.state

    def get_state(self, *, now_s):
        if (
            self.state.valid
            and now_s - self.state.timestamp_perf_s > self.config.max_sample_age_s
        ):
            return self.invalidate("stale")
        return self.state


class UARTEyeTracker:
    """Own the UART in one worker thread, leaving both screen loops responsive."""

    def __init__(self, *, uart_config=None, filter_config=None, on_state=None,
                 on_measurement=None, task_clock=None):
        self.config = uart_config or UARTEyeTrackerConfig()
        self.filter = EyePositionFilter(
            uart_config=self.config, filter_config=filter_config
        )
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self.task_clock = task_clock or TaskClock()
        self._on_measurement = on_measurement
        self._on_state = on_state
        self._callback_error = None
        self.warnings = EyeTrackerWarnings()

    def start(self):
        if self._thread is not None:
            raise RuntimeError("Eye tracker already started")
        self._thread = threading.Thread(target=self._run, name="eye-uart", daemon=True)
        self._thread.start()

    def get_state(self):
        with self._lock:
            return self.filter.get_state(now_s=time.perf_counter())

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            # A synchronous START already in progress may need its matching ACK
            # before STOP can be sent. Both state commands have bounded retries.
            timeout = (
                2 * self.config.state_timeout_s * (self.config.state_retries + 1) + 2
            )
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                self.warnings.warn("stop", "Eye UART worker did not stop; tracker shutdown is unconfirmed.")
        if getattr(self, "_cleanup_error", None):
            self.warnings.warn(
                "stop", f"Eye tracker STOP_TRACKING failed: {self._cleanup_error}"
            )
        if self._callback_error is not None:
            self.warnings.warn("consumer", f"Eye sample consumer failed: {self._callback_error}")

    def _check_status(self, status):
        problems = {}
        if status.status != StatusCode.OK or status.flags & StatusFlag.ERROR:
            problems["status"] = (
                "UART reply received: eye tracker reports a health error "
                f"(status={status.status.name}, flags={int(status.flags):#x}); "
                "gaze may be unavailable. Check the tracker display/logs."
            )
        if status.flags & StatusFlag.CALIBRATING:
            problems["calibrating"] = "Eye tracker is in calibration mode; tracking/recording may not have started."
        if status.flags & StatusFlag.SHUTTING_DOWN:
            problems["shutdown"] = "Eye tracker is shutting down; eye measurements are unavailable."
        if not status.flags & StatusFlag.TRACKING:
            problems["tracking"] = "Eye tracker did not start tracking, or tracking has stopped."
        if not status.flags & StatusFlag.CONFIGURED or status.configured_eye_count < 1:
            problems["configuration"] = "Eye tracker is not configured; eye measurements are unavailable."
        if self.config.expect_recording and not status.flags & StatusFlag.RECORDING:
            problems["recording"] = "Eye tracker recording did not start, or has stopped. Check the tracker Pi's recording settings and storage."
        if self.get_state().rejection_reason == "service_restarted":
            problems["restart"] = "Eye tracker restarted; gaze will remain unavailable for this task. Check calibration before the next task."
        self.warnings.warn_many(problems)

    def _wait_for_experiment_start(self):
        pending = os.environ.get(EXPERIMENT_UART_PENDING_ENV)
        if not pending:
            return
        deadline = time.monotonic() + 2 * self.config.state_timeout_s * (self.config.state_retries + 1) + 2
        while Path(pending).exists() and not self._stop.is_set():
            if time.monotonic() >= deadline:
                raise TimeoutError("Experiment START_EXPERIMENT is still pending; task continues without eye tracking")
            self._stop.wait(0.05)
        ready = os.environ.get(EXPERIMENT_UART_READY_ENV)
        if ready and not self._stop.is_set() and not Path(ready).is_file():
            raise RuntimeError(
                "Eye tracker experiment start was not confirmed; task continues without eye tracking"
            )

    def _publish_state(self):
        """Optional worker-side consumer; never perform serial/file I/O on flip."""
        if self._on_state is None or self._callback_error is not None:
            return
        try:
            self._on_state(self.get_state())
        except Exception as exc:
            self._callback_error = str(exc)
            self.warnings.warn("consumer", f"Eye sample consumer failed: {exc}")
            with self._lock:
                self.filter.invalidate("consumer_error", error=str(exc))
            self._stop.set()

    def _consume_measurement(self, status, *, received_perf_ns, round_trip_s=0.0):
        with self._lock:
            previous_key = self.filter._last_key
            state = self.filter.consume(
                status, now_s=received_perf_ns / 1_000_000_000,
                round_trip_s=round_trip_s,
                task_time_ns=self.task_clock.task_time_ns(timestamp_perf_ns=received_perf_ns),
            )
            is_new = self.filter._last_key != previous_key
        # Pass the exact acquisition snapshot, never a later get_state() read.
        if is_new and self._on_measurement is not None:
            self._on_measurement(state)

    def _run(self):
        channel = None
        controller = None
        start_attempted = False
        try:
            if not self.config.simulate:
                self._wait_for_experiment_start()
                if self._stop.is_set():
                    return
                channel = SerialPacketChannel(self.config.device, baud=self.config.baud)
                controller = EyeTrackerController(
                    channel,
                    response_timeout_s=self.config.response_timeout_s,
                    state_timeout_s=self.config.state_timeout_s,
                    retries=self.config.state_retries,
                )
                if self.config.manage_tracking:
                    # The launcher supplies the actual task block, including
                    # gaps from tasks that did not start eye tracking.
                    number = os.environ.get(EXPERIMENT_BLOCK_NUMBER_ENV)
                    block_number = validate_run_number(int(number)) if number is not None else None
                    start_attempted = True
                    status = controller.start_tracking(block_number=block_number)
                    received_perf_ns = time.perf_counter_ns()
                    self._consume_measurement(status, received_perf_ns=received_perf_ns)
                    self._check_status(status)
                    self._publish_state()
            next_poll = time.perf_counter()
            consecutive_link_errors = 0
            while not self._stop.is_set():
                try:
                    sent_ns = time.perf_counter_ns()
                    sent_s = sent_ns / 1_000_000_000
                    if self.config.simulate:
                        with self._lock:
                            state = self.filter.state
                            self.filter.state = replace(
                                state,
                                timestamp_perf_s=sent_s,
                                task_time_ns=self.task_clock.task_time_ns(timestamp_perf_ns=sent_ns),
                                camera_time_ns=sent_ns,
                                frame_sequence=state.frame_sequence + 1,
                                valid=True,
                                rejection_reason=None,
                                error=None,
                            )
                        if self._on_measurement is not None:
                            self._on_measurement(self.filter.state)
                    else:
                        status = controller.poll()
                        received_perf_ns = time.perf_counter_ns()
                        self._consume_measurement(
                            status, received_perf_ns=received_perf_ns,
                            round_trip_s=(received_perf_ns - sent_ns) / 1_000_000_000,
                        )
                        self._check_status(status)
                    consecutive_link_errors = 0
                except Exception as exc:
                    with self._lock:
                        self.filter.invalidate("uart_error", error=str(exc))
                    consecutive_link_errors += 1
                    _LOG.info("Eye UART poll failed (%d consecutive): %s", consecutive_link_errors, exc)
                    # A single lost telemetry reply does not mean START failed.
                    # Invalidate gaze immediately, but surface a dialog only
                    # for a sustained link failure or a non-transient error.
                    if consecutive_link_errors >= 3 or not isinstance(exc, (ResponseTimeout, ProtocolError)):
                        self.warnings.warn(
                            "link", f"Eye tracker telemetry unavailable after {consecutive_link_errors} "
                            f"consecutive poll failures; acquisition will keep retrying: {exc}"
                        )
                self._publish_state()
                next_poll = max(
                    next_poll + 1 / self.config.poll_rate_hz, time.perf_counter()
                )
                self._stop.wait(max(0, next_poll - time.perf_counter()))
        except Exception as exc:
            with self._lock:
                self.filter.invalidate("uart_error", error=str(exc))
            self.warnings.warn("link", f"Eye tracker is unavailable or tracking/recording start could not be confirmed: {exc}")
            self._publish_state()
        finally:
            try:
                if start_attempted and controller is not None:
                    status = controller.send_state_command(CommandCode.STOP_TRACKING)
                    if status.flags & (StatusFlag.TRACKING | StatusFlag.RECORDING):
                        self.warnings.warn("stop", "Eye tracker did not confirm tracking/recording stopped cleanly.")
                    elif status.flags & StatusFlag.ERROR:
                        self.warnings.warn(
                            "status", "Eye tracker confirmed tracking/recording stopped, "
                            "but still reports a tracker health error."
                        )
            except Exception as exc:
                self._cleanup_error = str(exc)
            finally:
                if channel is not None:
                    try:
                        channel.close()
                    except Exception as exc:
                        self._cleanup_error = str(exc)
            with self._lock:
                # Preserve startup/link failures for the main task to inspect
                # even after the worker has closed its serial channel.
                previous = self.filter.state
                self.filter.invalidate(
                    previous.rejection_reason if previous.error else "stopped",
                    error=getattr(self, "_cleanup_error", None) or previous.error,
                )
            self._publish_state()
            if getattr(self, "_cleanup_error", None):
                self.warnings.warn("stop", f"Eye tracker shutdown failed: {self._cleanup_error}")


class FixationWindow:
    """Measure time spent fixating over a complete rolling observation window.

    Each distinct camera result holds until the next result. Clip those spans
    to the exact window edges, so irregular frame timing cannot overweight a
    burst of samples. Missing/stale observations never earn fixation credit.
    The caller serializes access when acquisition and decisions use threads.
    """

    def __init__(self, duration_s, accept_fraction, *, max_gap_s=0.15, strict=False):
        self.duration_s = float(duration_s)
        self.accept_fraction = float(accept_fraction)
        self.max_gap_s = float(max_gap_s)
        self.strict = strict
        self.clear()

    def clear(self):
        self.history = deque()
        self._last_key = None
        self.acceptance = 0.0

    def update(self, state, in_window):
        if not state.valid and state.rejection_reason not in ("blink", "lost"):
            self.clear()
            return False
        key = (state.session_id, state.frame_sequence, state.camera_time_ns)
        if key != self._last_key:
            if (
                self.history
                and (
                    state.timestamp_perf_s - self.history[-1][0] > self.max_gap_s
                    or state.timestamp_perf_s <= self.history[-1][0]
                )
            ):
                # Missing observations cannot be evidence of sustained fixation.
                self.clear()
            self._last_key = key
            self.history.append(
                (state.timestamp_perf_s, state.valid and bool(in_window))
            )
        elif self.history and not state.valid:
            # A duplicate can invalidate the most recent result (blink/loss).
            self.history[-1] = (self.history[-1][0], False)
        return self.accepts(now_s=state.timestamp_perf_s)

    def accepts(self, *, now_s):
        """Evaluate at decision time, including expiry if the producer stops."""
        self.acceptance = 0.0
        if not self.history or self.duration_s <= 0:
            return False
        age = now_s - self.history[-1][0]
        if age < 0 or age > self.max_gap_s:
            return False
        cutoff = now_s - self.duration_s
        while len(self.history) > 1 and self.history[1][0] <= cutoff:
            self.history.popleft()
        fixated_s = 0.0
        end_s = now_s
        for start_s, hit in reversed(self.history):
            if hit:
                fixated_s += max(0.0, end_s - max(start_s, cutoff))
            end_s = start_s
        self.acceptance = min(1.0, fixated_s / self.duration_s)
        ready = self.history[0][0] <= cutoff + 1e-9
        passes = (
            self.acceptance - self.accept_fraction > 1e-9
            if self.strict else self.acceptance >= self.accept_fraction
        )
        return (
            self.history[-1][1]
            and ready
            and passes
        )


def fraction_to_pixels(
    fraction_pos: Sequence[float], screen_size: Sequence[float]
) -> Tuple[float, float]:
    return (
        float(fraction_pos[0]) * max(float(screen_size[0]), 1.0),
        float(fraction_pos[1]) * max(float(screen_size[1]), 1.0),
    )


def pixels_to_fraction(
    pixel_pos: Sequence[float], screen_size: Sequence[float]
) -> Tuple[float, float]:
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
    radius_px = (
        0.5
        * float(diameter_fraction)
        * min(
            max(float(screen_size[0]), 1.0),
            max(float(screen_size[1]), 1.0),
        )
    )
    return (
        fraction_distance_px(position_fraction, target_fraction, screen_size)
        <= radius_px
    )


def calibration_payload(
    calibration: EyeCalibration,
    *,
    uart_config: UARTEyeTrackerConfig,
    filter_config: EyeFilterConfig,
    main_screen_size: Sequence[int],
    fixation_fraction: Sequence[float],
    latest_state: Optional[EyeTrackerState] = None,
) -> dict:
    eye_tracker_calibration = calibration.to_json_dict()
    eye_tracker_calibration.update(
        {
            "uart": asdict(uart_config),
            "filter": asdict(filter_config),
            "main_screen_size_px": [int(main_screen_size[0]), int(main_screen_size[1])],
            "fixation_fraction": [
                float(fixation_fraction[0]),
                float(fixation_fraction[1]),
            ],
        }
    )
    if latest_state is not None:
        eye_tracker_calibration["latest_state"] = {
            key: value for key, value in asdict(latest_state).items()
            if key != "timestamp_perf_s"
        }
    return {"eye_tracker_calibration": eye_tracker_calibration}
