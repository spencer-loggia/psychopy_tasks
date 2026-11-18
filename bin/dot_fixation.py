"""Acquire fixation histories off the presentation thread for option-dot gates."""

from dataclasses import replace
import math
import threading

from .eye_tracking import FixationWindow, fraction_to_pixels
from .automatic_recalibration import OffsetRecalibration


# A busy acquisition thread has not reported a fixation loss. Consumers must
# preserve their previous state for this result, but still service deadlines.
FIXATION_UNAVAILABLE = object()
FIXATION_ACCEPT_FRACTION = 1.0 - 1e-6
FIXATION_RADIUS_EPSILON_PX = 1e-9


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

    Targets, gaze, screen size and radius use participant-view pixels. UART
    acquisition calls ``observe``, which maps camera samples into this space.
    The presentation thread reads ``status`` without waiting for the worker.
    """

    def __init__(self, settings, *, fixation_criteria_s, fixation_window_radius):
        self.fixation_criteria_s = _positive_setting(fixation_criteria_s, "fixation_criteria_s")
        self.fixation_window_radius = _positive_setting(fixation_window_radius, "fixation_window_radius")
        self._calibration = replace(settings.calibration)
        self._max_gap_s = settings.uart.max_sample_age_s
        self.recalibration = (
            OffsetRecalibration(settings.calibration, max_gap_s=self._max_gap_s)
            if settings.automatic_recalibration else None
        )
        self._recalibrated_targets = set()
        self._lock = threading.Lock()
        self._targets = ()
        self._windows = ()
        self._screen_size = (1, 1)
        self._onset_perf_s = math.inf
        self._first_entry_perf_s = None
        self._last_accepted_sample_s = None
        self._stable_window = None
        self._stable_anchor = None
        self._stable_session = None

    def start(self, positions, *, screen_size, onset_perf_s,
              allow_unaccepted_recalibration=False):
        """Begin fresh histories at the actual dot flip, excluding older frames."""
        if not math.isfinite(onset_perf_s):
            raise ValueError("onset_perf_s must be finite")
        targets = tuple(tuple(map(float, pos)) for pos in positions)
        windows = tuple(
            # History resets on every miss below. Allow only rounding error
            # in the fraction: subtracting large perf-counter times can make
            # a completely fixated interval fractionally smaller than 1.0.
            FixationWindow(self.fixation_criteria_s, FIXATION_ACCEPT_FRACTION, max_gap_s=self._max_gap_s)
            for _ in targets
        )
        with self._lock:
            self._screen_size = screen_size
            self._targets = targets
            self._windows = windows
            self._onset_perf_s = onset_perf_s
            self._first_entry_perf_s = None
            self._last_accepted_sample_s = None
            self._recalibrated_targets.clear()
            self._stable_anchor = None
            self._stable_session = None
            self._stable_window = (
                FixationWindow(self.fixation_criteria_s, FIXATION_ACCEPT_FRACTION, max_gap_s=self._max_gap_s)
                if self.recalibration is not None and allow_unaccepted_recalibration and len(targets) == 1
                else None
            )
            if self.recalibration is not None:
                self.recalibration.reset_history()

    def observe(self, state):
        """Consume each camera observation through the shared fixation detector."""
        with self._lock:
            if not self._windows or state.timestamp_perf_s < self._onset_perf_s:
                return
            if self.recalibration is not None:
                self.recalibration.observe(state)
                self._calibration = self.recalibration.calibration
            position = (
                fraction_to_pixels(
                    self._calibration.map_position(*state.smoothed_position), self._screen_size,
                )
                if state.valid else None
            )
            if self._stable_window is not None:
                stable = self._stable_window
                if position is None or not all(math.isfinite(v) for v in (*position, state.timestamp_perf_s)):
                    stable.clear()
                    self._stable_anchor = None
                else:
                    # The same radius/duration used by the target gate defines
                    # stability around a measured anchor, independent of offset.
                    reset = (
                        self._stable_anchor is None
                        or state.session_id != self._stable_session
                        or (stable.history and not 0 <= state.timestamp_perf_s - stable.history[-1][0] <= self._max_gap_s)
                        or (self._stable_anchor is not None and math.dist(
                            position, self._stable_anchor,
                        ) > self.fixation_window_radius + FIXATION_RADIUS_EPSILON_PX)
                    )
                    if reset:
                        stable.clear()
                        self._stable_anchor = position
                        self._stable_session = state.session_id
                    stable.update(state, True)
            for target, window in zip(self._targets, self._windows):
                in_window = position is not None and math.dist(
                    position, target,
                ) <= self.fixation_window_radius + FIXATION_RADIUS_EPSILON_PX
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
        """Return an accepted index, None for loss/expiry, or unavailable.

        Lock contention cannot invalidate a fresh acceptance. Its freshness
        budget remains anchored to the camera sample that supported the last
        successful read, never to a retry or a newer, unexamined observation.
        Even persistent contention therefore cannot keep stale fixation alive.
        """
        if not self._lock.acquire(blocking=False):
            sample_s = self._last_accepted_sample_s
            if sample_s is not None and not 0 <= now_s - sample_s <= self._max_gap_s:
                return None
            return FIXATION_UNAVAILABLE
        try:
            for index, window in enumerate(self._windows, start=1):
                if window.accepts(now_s=now_s):
                    self._last_accepted_sample_s = window.history[-1][0]
                    if self.recalibration is not None and index not in self._recalibrated_targets:
                        # Only the first acceptance precedes this dot's reveal.
                        # Polls and later refixations must not keep nudging it.
                        self._recalibrated_targets.add(index)
                        self.recalibration.accept(
                            self._targets[index - 1], now_s=now_s,
                            screen_size=self._screen_size,
                            fixation_start_s=window.history[0][0],
                        )
                    return index
            if (self._stable_window is not None and 1 not in self._recalibrated_targets
                    and self._stable_window.accepts(now_s=now_s)):
                adjusted = self.recalibration.accept(
                    self._targets[0], now_s=now_s,
                    screen_size=self._screen_size,
                    fixation_start_s=self._stable_window.history[0][0],
                )
                if adjusted:
                    self._recalibrated_targets.add(1)
                # A calibration correction is not a fixation acceptance. The
                # usual target window still controls reveal and task timing.
            self._last_accepted_sample_s = None
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
            self._last_accepted_sample_s = None
            self._stable_window = None
            self._stable_anchor = None
            self._stable_session = None
            if self.recalibration is not None:
                self.recalibration.reset_history()
