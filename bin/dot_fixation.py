"""Acquire fixation histories off the presentation thread for option-dot gates."""

from dataclasses import replace
import math
import threading

from .eye_tracking import FixationWindow, fraction_distance_px, pixels_to_fraction
from .screen import oriented_size, rotate_centered_point


def _positive_setting(value, name):
    try:
        if isinstance(value, bool):
            raise ValueError
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be positive and finite") from None
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def validate_dot_fixation_config(
    wait_for_saccade_on_dot, *, isi, eye_tracking_enabled,
    fixation_criteria_s, fixation_window_radius, saccade_wait_timeout_s,
    require_saccade_to_all_options=True, sequential=True,
):
    """Reject an enabled gate without visible dots, tracking, or valid limits."""
    if not isinstance(wait_for_saccade_on_dot, bool):
        raise ValueError("wait_for_saccade_on_dot must be true or false")
    if not isinstance(require_saccade_to_all_options, bool):
        raise ValueError("require_saccade_to_all_options must be true or false")
    if not wait_for_saccade_on_dot:
        return
    if not eye_tracking_enabled:
        raise ValueError("wait_for_saccade_on_dot requires eye_tracking.enabled=true")
    _positive_setting(isi, "isi when wait_for_saccade_on_dot=true")
    _positive_setting(fixation_criteria_s, "fixation_criteria_s")
    _positive_setting(fixation_window_radius, "fixation_window_radius")
    if saccade_wait_timeout_s is not None:
        _positive_setting(saccade_wait_timeout_s, "saccade_wait_timeout_s")
    elif sequential and not require_saccade_to_all_options:
        raise ValueError("saccade_wait_timeout_s must be finite when require_saccade_to_all_options=false")


class DotFixationGate:
    """Require a complete continuous fixation on one displayed dot.

    Targets and the radius use subject-display pixels. The caller supplies
    native PsychoPy target positions; calibration already maps eye samples to
    the oriented subject scene. UART acquisition calls ``observe`` and the
    presentation thread reads ``status`` without waiting for the worker.
    """

    def __init__(self, settings, *, fixation_criteria_s, fixation_window_radius):
        self.fixation_criteria_s = _positive_setting(fixation_criteria_s, "fixation_criteria_s")
        self.fixation_window_radius = _positive_setting(fixation_window_radius, "fixation_window_radius")
        self._calibration = replace(settings.calibration)
        self._max_gap_s = settings.uart.max_sample_age_s
        self._lock = threading.Lock()
        self._targets = ()
        self._windows = ()
        self._screen_size = (1, 1)
        self._onset_perf_s = math.inf
        self._first_entry_perf_s = None

    def start(self, positions, *, native_size, rotation_degrees, onset_perf_s):
        """Begin fresh histories at the actual dot flip, excluding older frames."""
        if not math.isfinite(onset_perf_s):
            raise ValueError("onset_perf_s must be finite")
        screen_size = oriented_size(native_size, rotation_degrees)
        targets = tuple(
            pixels_to_fraction(rotate_centered_point(pos, -rotation_degrees), screen_size)
            for pos in positions
        )
        windows = tuple(
            # History resets on every miss below. Allow only rounding error
            # in the fraction: subtracting large perf-counter times can make
            # a completely fixated interval fractionally smaller than 1.0.
            FixationWindow(self.fixation_criteria_s, 1.0 - 1e-6, max_gap_s=self._max_gap_s)
            for _ in targets
        )
        with self._lock:
            self._screen_size = screen_size
            self._targets = targets
            self._windows = windows
            self._onset_perf_s = onset_perf_s
            self._first_entry_perf_s = None

    def observe(self, state):
        """Consume each camera observation through the shared fixation detector."""
        with self._lock:
            if not self._windows or state.timestamp_perf_s < self._onset_perf_s:
                return
            position = (
                self._calibration.map_position(*state.smoothed_position)
                if state.valid else None
            )
            for target, window in zip(self._targets, self._windows):
                in_window = position is not None and fraction_distance_px(
                    position, target, self._screen_size,
                ) <= self.fixation_window_radius + 1e-9
                if in_window:
                    if self._first_entry_perf_s is None:
                        self._first_entry_perf_s = state.timestamp_perf_s
                    window.update(state, True)
                else:
                    window.clear()

    def reached_dot_before(self, deadline_perf_s):
        """Read the first valid gaze arrival since dot onset, independently of dwell.

        Acquisition publishes this immutable timestamp once per start; reading
        it needs no lock and cannot stall presentation or miss a brief saccade.
        """
        first_entry = self._first_entry_perf_s
        return first_entry is not None and first_entry <= deadline_perf_s

    def status(self, *, now_s):
        """Return the first accepted target's one-based index, or None."""
        if not self._lock.acquire(blocking=False):
            return None
        try:
            for index, window in enumerate(self._windows, start=1):
                if window.accepts(now_s=now_s):
                    return index
            return None
        finally:
            self._lock.release()

    def stop(self):
        """Stop collecting so intervening stimuli cannot seed the next dot."""
        with self._lock:
            self._targets = ()
            self._windows = ()
            self._onset_perf_s = math.inf
            self._first_entry_perf_s = None
