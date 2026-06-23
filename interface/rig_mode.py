from pathlib import Path
from typing import Optional


IS_RIG_ENV_VAR = "IS_RIG"
PORTABLE_MODE_VALUE = "0"
RIG_MODE_VALUE = "1"
SWITCH_TO_PORTABLE_SCRIPT = Path("~/Desktop/switch_to_portable_mode.sh").expanduser()
SWITCH_TO_RIG_SCRIPT = Path("~/Desktop/switch_to_rig_mode.sh").expanduser()


def normalize_is_rig(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized = str(value).strip()
    if normalized in {PORTABLE_MODE_VALUE, RIG_MODE_VALUE}:
        return normalized
    return None


def is_rig_mode(value: Optional[str]) -> bool:
    return normalize_is_rig(value) == RIG_MODE_VALUE


def experimenter_cursor_visible_for_touchscreen(touchscreen: bool, is_rig: Optional[str]) -> bool:
    if not touchscreen:
        return True
    return is_rig_mode(is_rig)


def mode_button_label(is_rig: str) -> str:
    if is_rig == RIG_MODE_VALUE:
        return "portable mode"
    return "rig mode"


def target_mode_for_current_mode(is_rig: str) -> str:
    if is_rig == RIG_MODE_VALUE:
        return PORTABLE_MODE_VALUE
    return RIG_MODE_VALUE


def mode_script_for_target_mode(target_mode: str) -> Path:
    if target_mode == RIG_MODE_VALUE:
        return SWITCH_TO_RIG_SCRIPT
    if target_mode == PORTABLE_MODE_VALUE:
        return SWITCH_TO_PORTABLE_SCRIPT
    raise ValueError(f"Unknown rig mode value: {target_mode}")


def mode_command_for_target_mode(target_mode: str) -> list[str]:
    return ["sudo", "-n", "bash", str(mode_script_for_target_mode(target_mode))]
