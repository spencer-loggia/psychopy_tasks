"""Timing validation helpers for active_foraging presentation modes."""

import math


def duration_requires_visible_phase(*, sequential: bool, is_memory: bool) -> bool:
    """Return whether duration represents a visible stimulus phase."""
    return bool(sequential) or bool(is_memory)


def validate_duration_for_presentation_mode(
    duration: float,
    *,
    sequential: bool,
    is_memory: bool,
    context: str = "active_foraging",
) -> None:
    """Validate duration semantics independently of display refresh rate."""
    duration_s = float(duration)
    if not math.isfinite(duration_s):
        raise ValueError(f"Invalid {context} timing config: duration must be finite.")
    if duration_requires_visible_phase(sequential=sequential, is_memory=is_memory):
        if duration_s <= 0.0:
            raise ValueError(
                f"Invalid {context} timing config: duration must be positive when "
                "sequential=true or is_memory=true. In memory modes, duration is the "
                "stimulus display time before the dot-only choice period."
            )
        return

    if duration_s != 0.0:
        raise ValueError(
            f"Invalid {context} timing config: when sequential=false and "
            "is_memory=false, duration must be exactly 0. In this mode the stimuli "
            "appear on the first choice frame and remain visible for choice_time only."
        )
