"""Reliable PsychoPy mouse/touch press sampling.

PsychoPy stores both the current button state and a timestamp for the most
recent press since ``Mouse.clickReset()``.  A touchscreen tap can be shorter
than one refresh interval, so both its press and release may be dispatched
together after a blocking ``Window.flip()``.  In that case the current button
state is already up, but the timestamp still records the press.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Callable, Optional, Sequence


@dataclass(frozen=True)
class MousePressSample:
    """One sampled pointer position and button state."""

    position: tuple[float, float]
    buttons: tuple[bool, ...]
    press_started: bool
    buffered_press: bool

    @property
    def down(self) -> bool:
        return any(self.buttons)

    @property
    def active(self) -> bool:
        """Whether this sample represents a held or newly buffered press."""
        return self.down or self.press_started


@dataclass(frozen=True)
class TouchHoldSample:
    """State of a continuous touch hold after one tracker update."""

    qualified: bool
    released: bool
    missing_since_s: Optional[float]


class TouchHoldTracker:
    """Track a held touch while tolerating short input-registration gaps.

    ``touch_registered`` should be true only while a touch is both down and
    within its required target. Tolerated gaps remain part of the elapsed hold
    because they represent presumed registration errors rather than releases.
    """

    def __init__(
        self,
        required_hold_s: float,
        *,
        max_break_s: float = 0.100,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        required_hold_s = float(required_hold_s)
        max_break_s = float(max_break_s)
        if not math.isfinite(required_hold_s) or required_hold_s < 0.0:
            raise ValueError("required_hold_s must be finite and non-negative")
        if not math.isfinite(max_break_s) or max_break_s < 0.0:
            raise ValueError("max_break_s must be finite and non-negative")
        self.required_hold_s = required_hold_s
        self.max_break_s = max_break_s
        self._clock = clock
        self.reset()

    def reset(self) -> None:
        self._started_s: Optional[float] = None
        self._missing_since_s: Optional[float] = None
        self._released = False

    def start(self, timestamp_s: Optional[float] = None) -> None:
        now = self._resolve_time(timestamp_s)
        self._started_s = now
        self._missing_since_s = None
        self._released = False

    def update(
        self,
        touch_registered: bool,
        timestamp_s: Optional[float] = None,
    ) -> TouchHoldSample:
        now = self._resolve_time(timestamp_s)
        if self._started_s is None:
            if touch_registered:
                self._started_s = now
            return self._sample(now)

        if not self._released:
            if touch_registered:
                self._missing_since_s = None
            else:
                if self._missing_since_s is None:
                    self._missing_since_s = now
                if now - self._missing_since_s > self.max_break_s:
                    self._released = True
        return self._sample(now)

    def _resolve_time(self, timestamp_s: Optional[float]) -> float:
        now = self._clock() if timestamp_s is None else float(timestamp_s)
        if not math.isfinite(now):
            raise ValueError("touch-hold timestamps must be finite")
        return now

    def _sample(self, now: float) -> TouchHoldSample:
        started = self._started_s is not None
        elapsed_s = max(0.0, now - self._started_s) if started else 0.0
        return TouchHoldSample(
            qualified=bool(started and not self._released and elapsed_s >= self.required_hold_s),
            released=self._released,
            missing_since_s=self._missing_since_s,
        )


def _as_bool_tuple(value: Any) -> tuple[bool, ...]:
    try:
        return tuple(bool(item) for item in value)
    except TypeError:
        return ()


def _has_press_timestamp(values: Sequence[Any]) -> bool:
    for value in values:
        try:
            if float(value) > 0.0:
                return True
        except (TypeError, ValueError):
            continue
    return False


class MousePressTracker:
    """Combine current button state with PsychoPy's buffered press times."""

    def __init__(self, mouse) -> None:
        self.mouse = mouse
        self._previous_down = False

    def _read_buttons_and_times(self) -> tuple[tuple[bool, ...], tuple[Any, ...]]:
        try:
            result = self.mouse.getPressed(getTime=True)
        except Exception:
            result = self.mouse.getPressed()

        if isinstance(result, tuple) and len(result) == 2:
            buttons = _as_bool_tuple(result[0])
            try:
                times = tuple(result[1])
            except TypeError:
                times = ()
            if buttons:
                return buttons, times
        return _as_bool_tuple(result), ()

    def _reset_click_times(self) -> None:
        try:
            self.mouse.clickReset()
        except Exception:
            pass

    def reset(self) -> None:
        """Discard earlier presses and begin a fresh response window.

        This should be called immediately before the flip that opens a response
        window. Capturing the current state prevents an already-held press from
        becoming a new edge; resetting its clock preserves presses that arrive
        during the blocking flip.
        """
        buttons, _ = self._read_buttons_and_times()
        self._previous_down = any(buttons)
        self._reset_click_times()

    def poll(self) -> MousePressSample:
        """Pump events and return held, edge, and short-tap information."""
        buttons, press_times = self._read_buttons_and_times()
        down = any(buttons)
        buffered_press = _has_press_timestamp(press_times)
        press_started = (down and not self._previous_down) or buffered_press
        self._previous_down = down

        # A timestamp remains set after release until clickReset(). Consume it
        # exactly once so a short tap cannot be returned on every poll.
        if buffered_press:
            self._reset_click_times()

        position = self.mouse.getPos()
        return MousePressSample(
            position=(float(position[0]), float(position[1])),
            buttons=buttons,
            press_started=bool(press_started),
            buffered_press=bool(buffered_press),
        )
