"""Refresh-locked visual synchronization with an independent random schedule.

The presenter owns display swaps. Call ``prepare_flip`` last when drawing a
frame, using its predicted display time, and ``flush_events`` after the swap.
Static presenters also use ``next_flip_perf_s`` to wake for bar-only swaps.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import random
import time
from typing import Any, Mapping


def validate_sync_pin(pin):
    if pin is not None and (isinstance(pin, bool) or not isinstance(pin, int) or pin < 0):
        raise ValueError("sync_pin must be a non-negative BCM GPIO integer or null")
    return pin


@dataclass(frozen=True)
class SyncBarConfig:
    enabled: bool = False
    height: int = 20
    interval_mean_seconds: float = 1.0
    interval_variance_seconds: float = 0.2

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("sync_bar.enabled must be true or false")
        if isinstance(self.height, bool) or not isinstance(self.height, int) or self.height <= 0:
            raise ValueError("sync_bar.height must be a positive integer in pixels")
        for name in ("interval_mean_seconds", "interval_variance_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"sync_bar.{name} must be a finite number")
        if self.interval_mean_seconds <= 0:
            raise ValueError("sync_bar.interval_mean_seconds must be positive")
        if not 0 <= self.interval_variance_seconds < self.interval_mean_seconds:
            raise ValueError(
                "sync_bar.interval_variance_seconds must be non-negative and less than interval_mean_seconds"
            )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None = None) -> "SyncBarConfig":
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise ValueError("sync_bar must be a JSON object")
        unknown = set(value) - cls.__dataclass_fields__.keys()
        if unknown:
            raise ValueError(f"Unknown sync_bar options: {', '.join(sorted(unknown))}")
        return cls(**value)


def sync_bar_rectangles(native_size, height: int, rotation_degrees=0) -> list[dict]:
    """Return five native-coordinate rectangles at the subject's visible top."""
    from .screen import oriented_size, rotate_centered_point

    width, screen_height = oriented_size(native_size, rotation_degrees)
    cell_width = int(width) // 5
    if cell_width < 1 or height > screen_height:
        raise ValueError("sync_bar must fit the oriented display")
    return [
        {
            "pos": rotate_centered_point(
                (-width / 2 + (index + 0.5) * cell_width, (screen_height - height) / 2),
                rotation_degrees,
            ),
            "width": cell_width,
            "height": height,
            "ori": rotation_degrees,
        }
        for index in range(5)
    ]


@dataclass(frozen=True)
class SyncFlipRecord:
    requested_perf_ns: int
    actual_perf_ns: int
    phase: int
    transition_num: int
    gpio_perf_ns: int | None = None
    gpio_requested_perf_ns: int | None = None


class SyncBar:
    """Cached five-rectangle overlay and a GPIO edge on each color inversion.

    GPIO ownership stays with the caller: provide an already claimed, initially
    low output. The pin follows the bar phase, so every inversion produces one
    edge without a pulse timer or extra display swap.
    """

    def __init__(
        self,
        win,
        config: SyncBarConfig | Mapping[str, Any] | None = None,
        *,
        rotation_degrees=0,
        native_size=None,
        event_logger=None,
        gpio_module=None,
        gpio_chip=None,
        sync_pin: int | None = None,
        preview_callback=None,
        clock=None,
        rng=None,
        rect_factory=None,
    ) -> None:
        self.config = config if isinstance(config, SyncBarConfig) else SyncBarConfig.from_mapping(config)
        self.win = win
        self.event_logger = event_logger
        self.preview_callback = preview_callback
        self._clock = time.perf_counter_ns if clock is None else clock
        self._rng = random.Random() if rng is None else rng
        self._gpio = gpio_module
        self._chip = gpio_chip
        self._pin = sync_pin
        validate_sync_pin(sync_pin)
        if gpio_module is not None and (gpio_chip is None or sync_pin is None):
            raise ValueError("sync GPIO requires a chip handle and sync_pin")
        self.phase = 0
        self.transition_num = 0
        self._drawn_phase = None
        self._next_perf_ns = None
        self._pending = False
        self._records: list[SyncFlipRecord] = []
        self._gpio_error = None
        self._closed = False
        self.rectangles = []
        self.geometry = []
        if self.config.enabled:
            if not callable(getattr(win, "callOnFlip", None)):
                raise ValueError("sync_bar requires a window with callOnFlip support")
            self.geometry = sync_bar_rectangles(
                win.size if native_size is None else native_size,
                self.config.height,
                rotation_degrees,
            )
            if rect_factory is None:
                from psychopy.visual import Rect

                rect_factory = Rect
            self.rectangles = [
                rect_factory(win, units="pix", colorSpace="rgb255", lineWidth=0,
                             fillColor=(0, 0, 0), lineColor=None, autoLog=False, **geometry)
                for geometry in self.geometry
            ]

    def _interval_ns(self) -> int:
        mean = self.config.interval_mean_seconds
        spread = self.config.interval_variance_seconds
        return max(1, round(self._rng.uniform(mean - spread, mean + spread) * 1_000_000_000))

    def start(self, now_perf_s: float | None = None) -> None:
        """Start once, after startup measurements and before the first scene."""
        if self.config.enabled and self._next_perf_ns is None and not self._closed:
            now_ns = self._clock() if now_perf_s is None else round(now_perf_s * 1_000_000_000)
            self._next_perf_ns = now_ns + self._interval_ns()

    @property
    def next_flip_perf_s(self) -> float:
        return math.inf if self._next_perf_ns is None else self._next_perf_ns / 1_000_000_000

    def prepare_flip(self, target_perf_s: float | None = None) -> bool:
        """Draw the overlay last and arm at most one inversion for this swap.

        ``target_perf_s`` is the expected time of the imminent display refresh.
        Missed deadlines produce one inversion, never a burst of catch-up edges.
        """
        if not self.config.enabled or self._closed:
            return False
        self.start()
        target_ns = self._clock() if target_perf_s is None else round(target_perf_s * 1_000_000_000)
        if not self._pending and target_ns >= self._next_perf_ns:
            self._pending = True
            # Draw-time work stays out of the display/GPIO callback.
            next_interval_ns = self._interval_ns()
            # The GPIO edge belongs to this realized swap, including when the
            # original requested deadline was deferred for a task transition.
            self.win.callOnFlip(self._on_flip, self._next_perf_ns, next_interval_ns)
        draw_phase = self.phase ^ int(self._pending)
        if draw_phase != self._drawn_phase:
            for index, rectangle in enumerate(self.rectangles):
                level = 255 * ((index + draw_phase) % 2)
                rectangle.fillColor = (level, level, level)
            self._drawn_phase = draw_phase
        for rectangle in self.rectangles:
            rectangle.draw()
        return self._pending

    def _on_flip(self, requested_ns: int, next_interval_ns: int) -> None:
        if self._closed:
            return
        actual_ns = self._clock()
        self.phase ^= 1
        gpio_ns = None
        gpio_requested_ns = None
        if self._gpio is not None:
            try:
                gpio_requested_ns = self._clock()
                result = self._gpio.gpio_write(self._chip, self._pin, self.phase)
                if isinstance(result, int) and result < 0:
                    raise RuntimeError(f"sync GPIO write failed: lgpio={result}")
                gpio_ns = self._clock()
            except Exception as error:
                # Preserve the displayed event, then report outside the callback.
                self._gpio_error = error
        self.transition_num += 1
        self._records.append(SyncFlipRecord(
            requested_ns, actual_ns, self.phase, self.transition_num, gpio_ns, gpio_requested_ns,
        ))
        self._next_perf_ns = actual_ns + next_interval_ns
        self._pending = False

    def flush_events(self) -> list[SyncFlipRecord]:
        """Publish completed transitions outside the timing-critical callback."""
        records, self._records = self._records, []
        for record in records:
            if self.event_logger is not None:
                self.event_logger.log_event(
                    trial_num=None,
                    event="sync_bar_flip",
                    event_type="frame_flip",
                    requested_timestamp_perf_ns=record.requested_perf_ns,
                    timestamp_perf_ns=record.actual_perf_ns,
                    sync_transition_num=record.transition_num,
                    sync_phase=record.phase,
                )
                if record.gpio_perf_ns is not None:
                    self.event_logger.log_event(
                        trial_num=None,
                        event="sync_signal_on" if record.phase else "sync_signal_off",
                        event_type="signal",
                        requested_timestamp_perf_ns=record.gpio_requested_perf_ns,
                        timestamp_perf_ns=record.gpio_perf_ns,
                        sync_transition_num=record.transition_num,
                        sync_phase=record.phase,
                    )
            if self.preview_callback is not None:
                self.preview_callback(record.phase)
        if self._gpio_error is not None:
            error, self._gpio_error = self._gpio_error, None
            raise RuntimeError(f"Synchronization GPIO failed: {error}") from error
        return records

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.flush_events()
        finally:
            if self._gpio is not None:
                result = self._gpio.gpio_write(self._chip, self._pin, 0)
                if isinstance(result, int) and result < 0:
                    raise RuntimeError(f"Could not clear sync GPIO: lgpio={result}")
