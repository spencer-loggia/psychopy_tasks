"""Shared fixation acceptance and scheduling for timed gaze-based rewards."""

from dataclasses import dataclass, replace
import math
import threading

from .eye_tracking import FixationWindow


def _positive_finite_setting(value, name):
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be positive and finite") from None
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return value


@dataclass(frozen=True)
class AutoRewardStatus:
    enabled: bool
    due: bool
    recent_fraction: float
    interval_fraction: float


class FixationAutoReward:
    """Collect UART observations and make reward decisions on separate threads.

    Subclasses define the accepted area in calibrated subject-screen fractions.
    Both rolling windows share an atomic mapping snapshot and reset at each
    reward onset, settings/mapping change or mode toggle. New samples collected
    during pump pulses count toward the next full interval.
    """

    RECENT_SECONDS = 0.1
    RECENT_THRESHOLD = 0.95
    INTERVAL_THRESHOLD = 0.80

    def __init__(self, *, reward_interval, calibration, max_sample_age_s):
        self.reward_interval = _positive_finite_setting(
            reward_interval, "reward_interval",
        )
        self._lock = threading.Lock()
        self._calibration = replace(calibration)
        self._enabled = False
        self._window_started_s = -math.inf
        self._recent = FixationWindow(
            self.RECENT_SECONDS, self.RECENT_THRESHOLD,
            max_gap_s=min(max_sample_age_s, self.RECENT_SECONDS), strict=True,
        )
        self._interval = FixationWindow(
            self.reward_interval, self.INTERVAL_THRESHOLD,
            max_gap_s=max_sample_age_s, strict=True,
        )

    def _set_reward_interval(self, reward_interval):
        """Update the interval while the caller owns ``_lock``."""
        if reward_interval is None or reward_interval == self.reward_interval:
            return False
        self.reward_interval = reward_interval
        self._interval.duration_s = reward_interval
        return True

    def _update_settings(self, settings):
        """Update validated settings while the caller owns ``_lock``."""
        return self._set_reward_interval(settings.get("reward_interval"))

    def _configure(self, settings, *, now_s, blocking, before_update=None):
        settings = {
            name: _positive_finite_setting(value, name)
            for name, value in settings.items() if value is not None
        }
        if not math.isfinite(now_s):
            raise ValueError("now_s must be finite")
        if not self._lock.acquire(blocking):
            return None
        try:
            if before_update is not None:
                before_update()
            changed = self._update_settings(settings)
            if changed:
                self._reset(now_s)
            return changed
        finally:
            self._lock.release()

    def configure(self, *, reward_interval=None, now_s):
        """Apply new rules atomically, waiting for an in-flight observation.

        ``now_s`` is the effective change time on the eye tracker's performance
        clock. Timing-sensitive callers should use :meth:`try_configure`.
        Return whether a setting changed.
        """
        return self._configure(
            {"reward_interval": reward_interval}, now_s=now_s, blocking=True,
        )

    def try_configure(self, *, reward_interval=None, now_s, before_update=None):
        """Apply once without waiting, or return ``None`` while acquisition is busy.

        Optional ``before_update`` runs under the policy lock before any policy
        mutation, including history reset. It must not wait, perform I/O, or
        call a locking policy method. Callback errors leave policy unchanged.
        Return ``True`` for changed rules and ``False`` for unchanged rules.
        """
        return self._configure(
            {"reward_interval": reward_interval}, now_s=now_s, blocking=False,
            before_update=before_update,
        )

    def set_reward_interval(self, reward_interval, *, now_s):
        return self.configure(
            reward_interval=_positive_finite_setting(reward_interval, "reward_interval"),
            now_s=now_s,
        )

    def _reset(self, now_s):
        self._recent.clear()
        self._interval.clear()
        self._window_started_s = now_s

    def toggle(self, *, now_s):
        with self._lock:
            self._enabled = not self._enabled
            self._reset(now_s)
            return self._enabled

    def _in_window(self, position):
        raise NotImplementedError

    def observe(self, state):
        with self._lock:
            # A repeated UART reply or in-flight callback from before the
            # reset must not seed the new window with pre-reward history.
            if state.timestamp_perf_s < self._window_started_s:
                return
            in_window = state.valid and self._in_window(
                self._calibration.map_position(*state.smoothed_position)
            )
            self._recent.update(state, in_window)
            self._interval.update(state, in_window)

    def status(self, *, now_s):
        with self._lock:
            recent_ok = self._recent.accepts(now_s=now_s)
            interval_ok = self._interval.accepts(now_s=now_s)
            return AutoRewardStatus(
                enabled=self._enabled,
                due=(self._enabled and recent_ok and interval_ok),
                recent_fraction=self._recent.acceptance,
                interval_fraction=self._interval.acceptance,
            )

    def reward_delivered(self, *, now_s):
        with self._lock:
            # Both automatic and manual rewards require a full new window.
            self._reset(now_s)


class ScreenAutoReward(FixationAutoReward):
    """Accept the entire subject screen, including its edges and corners."""

    def _in_window(self, position):
        x, y = position
        return -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5
