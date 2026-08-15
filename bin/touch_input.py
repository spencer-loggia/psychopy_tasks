"""Reliable PsychoPy mouse/touch press sampling.

PsychoPy stores both the current button state and a timestamp for the most
recent press since ``Mouse.clickReset()``.  A touchscreen tap can be shorter
than one refresh interval, so both its press and release may be dispatched
together after a blocking ``Window.flip()``.  In that case the current button
state is already up, but the timestamp still records the press.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


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

    def reset(self) -> bool:
        """Discard earlier presses and return whether a button is held now.

        This should be called immediately before the flip that opens a response
        window.  Reading first pumps pending window-system events; resetting
        second makes any press arriving during the blocking flip observable.
        """
        buttons, _ = self._read_buttons_and_times()
        self._previous_down = any(buttons)
        self._reset_click_times()
        return self._previous_down

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
