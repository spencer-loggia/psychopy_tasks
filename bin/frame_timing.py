"""Small, display-independent timing primitives shared by task presenters."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Callable, Iterable, Mapping, Optional


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


def capture_perf_counter_on_flip(target: dict[str, float]) -> None:
    """Capture a PsychoPy flip callback time on the performance clock."""

    target["actual_perf_s"] = time.perf_counter()


def flip_with_timestamps(win: Any) -> FlipTimestamps:
    """Request a flip and capture submission and realized swap timestamps.

    PsychoPy's ``callOnFlip`` is used when available so ``actual_perf_s`` is
    sampled at the swap rather than after the blocking ``flip`` call returns.
    """

    flip_capture: dict[str, float] = {}
    call_on_flip = getattr(win, "callOnFlip", None)
    if callable(call_on_flip):
        try:
            call_on_flip(capture_perf_counter_on_flip, flip_capture)
        except Exception:
            pass
    requested_perf_s = time.perf_counter()
    psychopy_s = win.flip()
    flip_return_perf_s = time.perf_counter()
    actual_perf_s = flip_capture.get("actual_perf_s", flip_return_perf_s)
    return FlipTimestamps(
        psychopy_s=psychopy_s,
        requested_perf_s=requested_perf_s,
        actual_perf_s=actual_perf_s,
    )


DEFAULT_FLIP_SUBMISSION_LEAD_FRACTION = 0.49
DEFAULT_TIMING_TOLERANCE_S = 0.001


def flip_submission_time(
    target_perf_s: float,
    frame_duration_s: float,
    *,
    lead_fraction: float = DEFAULT_FLIP_SUBMISSION_LEAD_FRACTION,
) -> float:
    """Return when a blocking flip should be submitted for ``target_perf_s``.

    A normal vsync-enabled ``Window.flip`` presents on the next vertical blank.
    Submitting just before the midpoint between adjacent refreshes gives the
    blocking swap enough headroom while selecting the refresh nearest the
    requested target.
    """

    frame_duration = float(frame_duration_s)
    fraction = float(lead_fraction)
    if not math.isfinite(frame_duration) or frame_duration <= 0.0:
        raise ValueError("frame_duration_s must be a positive finite value")
    if not math.isfinite(fraction) or not 0.0 <= fraction < 1.0:
        raise ValueError("lead_fraction must be finite and in [0, 1)")
    return float(target_perf_s) - fraction * frame_duration


def wait_until(
    deadline_perf_s: float,
    *,
    poll_callback: Optional[Callable[[], bool]] = None,
    poll_interval_s: float = 0.002,
    wait_fn: Optional[Callable[[float], None]] = None,
) -> bool:
    """Wait efficiently until a monotonic deadline.

    ``poll_callback`` is run while waiting and may cancel by returning true.
    The return value is false only when cancellation was requested.
    """

    deadline = float(deadline_perf_s)
    interval = float(poll_interval_s)
    if not math.isfinite(deadline):
        raise ValueError("deadline_perf_s must be finite")
    if not math.isfinite(interval) or interval <= 0.0:
        raise ValueError("poll_interval_s must be a positive finite value")
    sleeper = time.sleep if wait_fn is None else wait_fn

    while True:
        if poll_callback is not None and poll_callback():
            return False
        remaining_s = deadline - time.perf_counter()
        if remaining_s <= 0.0:
            return True
        sleeper(
            remaining_s
            if poll_callback is None
            else min(interval, remaining_s)
        )


def wait_until_flip_submission(
    target_perf_s: float,
    frame_duration_s: float,
    *,
    poll_callback: Optional[Callable[[], bool]] = None,
    poll_interval_s: float = 0.002,
    wait_fn: Optional[Callable[[float], None]] = None,
) -> bool:
    """Wait until the submission window for the refresh nearest a target."""

    return wait_until(
        flip_submission_time(target_perf_s, frame_duration_s),
        poll_callback=poll_callback,
        poll_interval_s=poll_interval_s,
        wait_fn=wait_fn,
    )


class FrameTransitionScheduler:
    """Schedule and assess sparse visual transitions on a fixed-rate display.

    Static scenes remain in the front buffer. Callers draw the next scene once,
    then use :meth:`flip_at` instead of redrawing and swapping unchanged content
    on every refresh.
    """

    def __init__(
        self,
        frame_duration_s: float,
        *,
        timing_tolerance_s: float = DEFAULT_TIMING_TOLERANCE_S,
    ) -> None:
        self.frame_duration_s = float(frame_duration_s)
        self.timing_tolerance_s = float(timing_tolerance_s)
        if not math.isfinite(self.frame_duration_s) or self.frame_duration_s <= 0.0:
            raise ValueError("frame_duration_s must be a positive finite value")
        if not math.isfinite(self.timing_tolerance_s) or self.timing_tolerance_s < 0.0:
            raise ValueError("timing_tolerance_s must be finite and non-negative")
        self.transition_count = 0
        self.missed_transition_count = 0
        self.maximum_absolute_error_s = 0.0
        self.maximum_lateness_s = 0.0

    @property
    def failure_threshold_s(self) -> float:
        return 0.5 * self.frame_duration_s + self.timing_tolerance_s

    @staticmethod
    def target_after(
        onset_perf_s: float,
        plan: FrameDurationPlan,
    ) -> float:
        """Return the ideal offset time for a quantized static phase."""

        return float(onset_perf_s) + float(plan.scheduled_s)

    def record(self, target_perf_s: float, actual_perf_s: float) -> float:
        """Record one realized transition and return its signed timing error."""

        error_s = float(actual_perf_s) - float(target_perf_s)
        self.transition_count += 1
        self.maximum_absolute_error_s = max(
            self.maximum_absolute_error_s,
            abs(error_s),
        )
        self.maximum_lateness_s = max(self.maximum_lateness_s, error_s, 0.0)
        if abs(error_s) > self.failure_threshold_s:
            self.missed_transition_count += 1
        return error_s

    def flip_at(
        self,
        win: Any,
        target_perf_s: float,
        *,
        poll_callback: Optional[Callable[[], bool]] = None,
        poll_interval_s: float = 0.002,
        before_flip_callback: Optional[Callable[[], None]] = None,
    ) -> Optional[FlipTimestamps]:
        """Present the already-drawn back buffer at the target refresh.

        Returns ``None`` if ``poll_callback`` cancels while waiting.
        """

        if not wait_until_flip_submission(
            target_perf_s,
            self.frame_duration_s,
            poll_callback=poll_callback,
            poll_interval_s=poll_interval_s,
        ):
            return None
        if before_flip_callback is not None:
            before_flip_callback()
        timing = flip_with_timestamps(win)
        self.record(target_perf_s, timing.actual_perf_s)
        return timing
