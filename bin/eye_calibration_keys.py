"""Shared key names for the two independently focused calibration windows."""

# PsychoPy's pyglet backend reports num_8 with Num Lock and num_up without
# it. Also accept pygame's [8] and plain digits from backends that normalize
# keypad keys. Ordinary arrow keys remain reserved for gain adjustments.
_FIXATION_DIRECTIONS = {
    key: direction
    for aliases, direction in (
        (("num_8", "num_up", "[8]", "8"), (0, 1)),
        (("num_6", "num_right", "[6]", "6"), (1, 0)),
        (("num_2", "num_down", "[2]", "2"), (0, -1)),
        (("num_4", "num_left", "[4]", "4"), (-1, 0)),
    )
    for key in aliases
}
CALIBRATION_KEYS = ("f", "z", "r", "a", "v", "up", "down", "right", "left", *_FIXATION_DIRECTIONS)
FIXATION_STEP_FRACTION = 0.025


def fixation_key_steps(keys):
    """Accumulate target movement, including repeats between UI polls."""
    x_steps = y_steps = 0
    for key in keys:
        dx, dy = _FIXATION_DIRECTIONS.get(key, (0, 0))
        x_steps += dx
        y_steps += dy
    return x_steps, y_steps
