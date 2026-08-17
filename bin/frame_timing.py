"""Small, display-independent timing primitives shared by task presenters."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class FrameDurationPlan:
    """Nearest refresh-locked representation of a requested duration."""

    requested_s: float
    frame_count: int
    scheduled_s: float

    @property
    def error_s(self) -> float:
        return self.scheduled_s - self.requested_s


def plan_frame_duration(
    requested_s: float,
    fps: float,
    *,
    minimum_frames: int = 0,
) -> FrameDurationPlan:
    """Return the nearest display-frame plan while preserving the exact request.

    Half-frame ties round up. This is deliberate and consistent across Python
    versions (unlike :func:`round`, which uses ties-to-even).
    """

    requested = float(requested_s)
    refresh_hz = float(fps)
    if not math.isfinite(refresh_hz) or refresh_hz <= 0.0:
        raise ValueError(f"fps must be a positive finite value, got {fps!r}")
    if not math.isfinite(requested) or requested < 0.0:
        raise ValueError(
            f"requested_s must be a finite non-negative value, got {requested_s!r}"
        )
    if isinstance(minimum_frames, bool):
        raise ValueError("minimum_frames must be a non-negative integer")
    minimum = int(minimum_frames)
    if minimum < 0 or minimum != minimum_frames:
        raise ValueError("minimum_frames must be a non-negative integer")

    frames = max(minimum, int(math.floor(requested * refresh_hz + 0.5)))
    return FrameDurationPlan(
        requested_s=requested,
        frame_count=frames,
        scheduled_s=frames / refresh_hz,
    )


def validate_requested_durations(
    timings_s: Mapping[str, float],
    *,
    positive: Iterable[str] = (),
    context: str = "task",
) -> None:
    """Validate duration semantics without imposing refresh alignment."""

    positive_names = frozenset(str(name) for name in positive)
    for name, raw_value in timings_s.items():
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError(
                f"Invalid {context} timing config: {name} must be finite."
            )
        if name in positive_names:
            if value <= 0.0:
                raise ValueError(
                    f"Invalid {context} timing config: {name} must be positive."
                )
        elif value < 0.0:
            raise ValueError(
                f"Invalid {context} timing config: {name} cannot be negative."
            )


@dataclass(frozen=True)
class FlipTimestamps:
    """Times immediately around one refresh-synchronized ``Window.flip``."""

    psychopy_s: Any
    requested_perf_s: float
    actual_perf_s: float


def flip_with_timestamps(win: Any) -> FlipTimestamps:
    """Request a flip and capture both submission and realized timestamps."""

    requested_perf_s = time.perf_counter()
    psychopy_s = win.flip()
    actual_perf_s = time.perf_counter()
    return FlipTimestamps(
        psychopy_s=psychopy_s,
        requested_perf_s=requested_perf_s,
        actual_perf_s=actual_perf_s,
    )
