"""Fixation-gated calibration rewards, independent of screen and pump waits."""

from dataclasses import dataclass, replace
import math
import threading

from .eye_tracking import FixationWindow, fraction_position_within_diameter


@dataclass(frozen=True)
class AutoRewardStatus:
    enabled: bool
    due: bool
    recent_fraction: float
    interval_fraction: float


class CalibrationAutoReward:
    """Collect UART observations in its worker; decide/deliver on the main thread.

    The two windows share an atomic mapping snapshot. Only a mapping change or
    mode toggle resets history: samples collected during pump pulses still count.
    """

    RECENT_SECONDS = 0.1
    RECENT_THRESHOLD = 0.95
    INTERVAL_THRESHOLD = 0.85

    def __init__(
        self, *, reward_interval, calibration, fixation, screen_size,
        fix_diameter, max_sample_age_s,
    ):
        if not math.isfinite(reward_interval) or reward_interval <= 0:
            raise ValueError("reward_interval must be positive and finite")
        self.reward_interval = reward_interval
        self._lock = threading.Lock()
        self._calibration = replace(calibration)
        self._fixation = tuple(fixation)
        self._screen_size = tuple(screen_size)
        self._fix_diameter = fix_diameter
        self._enabled = False
        self._next_reward_s = math.inf
        self._recent = FixationWindow(
            self.RECENT_SECONDS, self.RECENT_THRESHOLD,
            max_gap_s=min(max_sample_age_s, self.RECENT_SECONDS), strict=True,
        )
        self._interval = FixationWindow(
            reward_interval, self.INTERVAL_THRESHOLD,
            max_gap_s=max_sample_age_s, strict=True,
        )

    def _reset(self, now_s):
        self._recent.clear()
        self._interval.clear()
        self._next_reward_s = now_s + self.reward_interval

    def toggle(self, *, now_s):
        with self._lock:
            self._enabled = not self._enabled
            self._reset(now_s)
            return self._enabled

    def set_mapping(self, calibration, fixation, *, now_s):
        with self._lock:
            # Never expose half-applied gain/offset edits to the UART callback.
            self._calibration = replace(calibration)
            self._fixation = tuple(fixation)
            self._reset(now_s)

    def observe(self, state):
        with self._lock:
            in_window = state.valid and fraction_position_within_diameter(
                self._calibration.map_position(*state.smoothed_position),
                self._fixation, diameter_fraction=self._fix_diameter,
                screen_size=self._screen_size,
            )
            self._recent.update(state, in_window)
            self._interval.update(state, in_window)

    def status(self, *, now_s):
        with self._lock:
            recent_ok = self._recent.accepts(now_s=now_s)
            interval_ok = self._interval.accepts(now_s=now_s)
            return AutoRewardStatus(
                enabled=self._enabled,
                due=(self._enabled and now_s >= self._next_reward_s
                     and recent_ok and interval_ok),
                recent_fraction=self._recent.acceptance,
                interval_fraction=self._interval.acceptance,
            )

    def reward_delivered(self, *, now_s):
        with self._lock:
            # Manual rewards also postpone the next automatic pulse. Missed
            # opportunities never queue up multiple catch-up rewards.
            self._next_reward_s = now_s + self.reward_interval
