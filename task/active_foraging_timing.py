"""Timing validation helpers for active_foraging presentation modes."""


def duration_requires_positive_frames(*, sequential: bool, is_memory: bool) -> bool:
    """Return whether duration represents a visible stimulus phase."""
    return bool(sequential) or bool(is_memory)


def validate_duration_for_presentation_mode(
    duration: float,
    *,
    sequential: bool,
    is_memory: bool,
) -> None:
    """Validate duration semantics before frame-alignment validation."""
    duration_s = float(duration)
    if duration_requires_positive_frames(sequential=sequential, is_memory=is_memory):
        if duration_s <= 0.0:
            raise ValueError(
                "Invalid active_foraging timing config: duration must be positive when "
                "sequential=true or is_memory=true. In memory modes, duration is the "
                "stimulus display time before the dot-only choice period."
            )
        return

    if duration_s != 0.0:
        raise ValueError(
            "Invalid active_foraging timing config: when sequential=false and "
            "is_memory=false, duration must be exactly 0. In this mode the stimuli "
            "appear on the first choice frame and remain visible for choice_time only."
        )
