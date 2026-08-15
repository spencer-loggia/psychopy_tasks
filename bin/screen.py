"""
Shared helpers for resolving monitor selectors and managing experimenter displays.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import io
import multiprocessing as mp
import os
import queue
import re
import subprocess
import sys
import time
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
from PIL import Image

try:
    from screeninfo import get_monitors
except ImportError:
    get_monitors = None


ScreenSelector = Optional[Union[int, str]]
_UNSET = object()
MAIN_SCREEN_ENV = "MAIN_SCREEN"
SECONDARY_SCREEN_ENV = "SECONDARY_SCREEN"


def set_window_mouse_visible(win, visible: bool) -> bool:
    """Set cursor visibility across PsychoPy backend variants."""
    applied = False
    visible_bool = bool(visible)

    try:
        win.mouseVisible = visible_bool
        applied = True
    except Exception:
        pass

    for method_name in ("setMouseVisible", "setMouseVisibility"):
        try:
            method = getattr(win, method_name, None)
            if callable(method):
                method(visible_bool)
                applied = True
        except Exception:
            pass

    handle = getattr(win, "winHandle", None)
    try:
        method = getattr(handle, "set_mouse_visible", None)
        if callable(method):
            method(visible_bool)
            applied = True
    except Exception:
        pass

    return applied


def configure_window_vsync(win, enabled: bool) -> bool:
    """Configure blanking waits and the native swap interval when available."""
    applied = False
    enabled = bool(enabled)
    try:
        win.waitBlanking = enabled
        applied = True
    except Exception:
        pass

    targets = [getattr(win, "winHandle", None), getattr(win, "backend", None)]
    for target in targets:
        if target is None:
            continue
        for method_name in ("set_vsync", "setVSync"):
            try:
                method = getattr(target, method_name, None)
                if callable(method):
                    method(enabled)
                    applied = True
                    break
            except Exception:
                continue
    return applied


def enforce_window_vsync(win) -> bool:
    """Request blocking, refresh-synchronized swaps for a PsychoPy window."""
    return configure_window_vsync(win, True)


def resolve_window_frame_rate(
    win,
    *,
    configured_fps: Optional[float] = None,
    msg_logger=None,
    context: str = "task",
    fallback_fps: float = 60.0,
) -> tuple[float, float]:
    """Measure main-window flips and compare them with an optional override."""
    measured_fps = None
    try:
        measured_fps = win.getActualFrameRate(
            nIdentical=20,
            nMaxFrames=120,
            nWarmUpFrames=10,
            threshold=1,
        )
        if measured_fps is not None:
            measured_fps = float(measured_fps)
            if not np.isfinite(measured_fps) or measured_fps <= 0.0:
                measured_fps = None
    except Exception:
        measured_fps = None

    configured = None
    if configured_fps is not None:
        configured = float(configured_fps)
        if not np.isfinite(configured) or configured <= 0.0:
            raise ValueError("configured refresh_rate must be a positive finite value")

    if configured is not None:
        used_fps = configured
        if measured_fps is None:
            _safe_log_message(
                msg_logger,
                "WARN",
                (
                    f"refresh_rate_comparison context={context} "
                    f"configured_fps={configured:.6f} measured_fps=unavailable "
                    "status=unverified"
                ),
            )
        else:
            difference_hz = abs(measured_fps - configured)
            tolerance_hz = max(0.5, configured * 0.01)
            status = "match" if difference_hz <= tolerance_hz else "mismatch"
            _safe_log_message(
                msg_logger,
                "INFO" if status == "match" else "WARN",
                (
                    f"refresh_rate_comparison context={context} "
                    f"configured_fps={configured:.6f} "
                    f"measured_fps={measured_fps:.6f} "
                    f"difference_hz={difference_hz:.6f} "
                    f"tolerance_hz={tolerance_hz:.6f} status={status}"
                ),
            )
    elif measured_fps is not None:
        used_fps = measured_fps
    else:
        used_fps = float(fallback_fps)
        _safe_log_message(
            msg_logger,
            "WARN",
            (
                f"frame_rate_detection_failed context={context} "
                f"fallback_fps={used_fps:.6f}"
            ),
        )

    frame_duration_s = 1.0 / used_fps
    _safe_log_message(
        msg_logger,
        "INFO",
        (
            f"frame_timing context={context} fps={used_fps:.6f} "
            f"frame_dur_s={frame_duration_s:.9f} "
            f"measured_fps={measured_fps if measured_fps is not None else 'unavailable'} "
            f"configured_fps={configured if configured is not None else 'none'}"
        ),
    )
    return float(used_fps), float(frame_duration_s)


def _safe_log_message(msg_logger, level: str, message: str) -> None:
    if msg_logger is None:
        return
    try:
        msg_logger.log(level, message)
    except Exception:
        pass


class MainDisplayFrameTimingMonitor:
    """Count missed refreshes only inside continuous flip sequences."""

    def __init__(self, win, frame_duration_s: float):
        self.win = win
        self.frame_duration_s = float(frame_duration_s)
        if not np.isfinite(self.frame_duration_s) or self.frame_duration_s <= 0.0:
            raise ValueError("frame_duration_s must be a positive finite value")
        self.missed_refreshes = 0

    @contextmanager
    def continuous_sequence(self):
        previous_recording = bool(
            getattr(self.win, "recordFrameIntervals", False)
        )
        count_before = None
        try:
            self.win.recordFrameIntervals = False
            self.win.refreshThreshold = self.frame_duration_s * 1.5
            count_before = int(getattr(self.win, "nDroppedFrames", 0))
            self.win.recordFrameIntervals = True
        except Exception:
            count_before = None

        try:
            yield
        finally:
            if count_before is not None:
                try:
                    self.missed_refreshes += max(
                        0,
                        int(getattr(self.win, "nDroppedFrames", 0))
                        - count_before,
                    )
                except Exception:
                    pass
            try:
                self.win.recordFrameIntervals = False
                if previous_recording:
                    self.win.recordFrameIntervals = True
            except Exception:
                pass


@dataclass(frozen=True)
class ScreenGeometry:
    index: int
    x: int
    y: int
    width: int
    height: int
    name: str = ""
    rotation: str = "normal"


def parse_screen_selector(value: Any, name: str) -> ScreenSelector:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError(f"Screen field '{name}' must be a non-negative integer or output name")
    if isinstance(value, int):
        if value < 0:
            raise ValueError(f"Screen field '{name}' must be >= 0")
        return value

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        if stripped.isdigit():
            parsed = int(stripped)
            if parsed < 0:
                raise ValueError(f"Screen field '{name}' must be >= 0")
            return parsed
        return stripped

    raise ValueError(f"Screen field '{name}' must be a non-negative integer or output name")


def load_screen_config(
    cfg: Dict[str, Any],
    *,
    cli_main: ScreenSelector = None,
    cli_experimenter: ScreenSelector = None,
) -> Dict[str, ScreenSelector]:
    screens_cfg = cfg.get("screens", {})
    if screens_cfg is None:
        screens_cfg = {}
    if not isinstance(screens_cfg, dict):
        raise ValueError("Config field 'screens' must be a JSON object")

    main_value = cli_main
    main_is_null = False
    if main_value is None:
        main_is_null = "main" in screens_cfg and screens_cfg["main"] is None
        main_value = screens_cfg.get("main", cfg.get("main_screen"))
    if main_value is None:
        main_value = os.environ.get(MAIN_SCREEN_ENV)
        if main_is_null and (main_value is None or not str(main_value).strip()):
            raise ValueError(f"screens.main is null, but {MAIN_SCREEN_ENV} is not set")

    experimenter_value = cli_experimenter
    experimenter_is_null = False
    if experimenter_value is None:
        experimenter_key = "experimenter" if "experimenter" in screens_cfg else "secondary"
        experimenter_is_null = experimenter_key in screens_cfg and screens_cfg[experimenter_key] is None
        experimenter_value = screens_cfg.get(
            experimenter_key,
            cfg.get("experimenter_screen", cfg.get("secondary_screen")),
        )
    if experimenter_value is None:
        experimenter_value = os.environ.get(SECONDARY_SCREEN_ENV)
        if experimenter_is_null and (experimenter_value is None or not str(experimenter_value).strip()):
            raise ValueError(f"screens.experimenter is null, but {SECONDARY_SCREEN_ENV} is not set")

    return {
        "main": parse_screen_selector(main_value, "screens.main"),
        "experimenter": parse_screen_selector(experimenter_value, "screens.experimenter"),
    }


def _normalize_screen_name(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _screen_name_aliases(name: str) -> set[str]:
    raw = name.strip().lower()
    aliases = {raw, _normalize_screen_name(raw)}

    def _add(text: str) -> None:
        aliases.add(text)
        aliases.add(_normalize_screen_name(text))

    if raw.startswith("hdmi-a-"):
        suffix = raw[len("hdmi-a-") :]
        _add(f"hdmi-{suffix}")
        _add(f"hdmi{suffix}")
    elif raw.startswith("hdmi-"):
        suffix = raw[len("hdmi-") :]
        _add(f"hdmi-a-{suffix}")
        _add(f"hdmi{suffix}")
    elif raw.startswith("hdmi") and raw[len("hdmi") :].isdigit():
        suffix = raw[len("hdmi") :]
        _add(f"hdmi-{suffix}")
        _add(f"hdmi-a-{suffix}")

    if raw.startswith("dsi-"):
        suffix = raw[len("dsi-") :]
        _add(f"dsi{suffix}")
    elif raw.startswith("dsi") and raw[len("dsi") :].isdigit():
        suffix = raw[len("dsi") :]
        _add(f"dsi-{suffix}")

    return aliases


def _run_monitor_query(cmd: Sequence[str]) -> str:
    try:
        result = subprocess.run(
            list(cmd),
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return ""
    return str(result.stdout or "").strip()


def _parse_xrandr_listactivemonitors(output: str) -> list[ScreenGeometry]:
    screens: list[ScreenGeometry] = []
    pattern = re.compile(
        r"^\s*(\d+):\s+\S+\s+(\d+)(?:/\d+)?x(\d+)(?:/\d+)?([+-]\d+)([+-]\d+)\s+(\S+)\s*$"
    )
    for line in output.splitlines():
        match = pattern.match(line)
        if not match:
            continue
        index, width, height, x, y, name = match.groups()
        screens.append(
            ScreenGeometry(
                index=int(index),
                x=int(x),
                y=int(y),
                width=max(int(width), 1),
                height=max(int(height), 1),
                name=str(name or ""),
            )
        )
    return screens


def _parse_xrandr_query(output: str) -> list[ScreenGeometry]:
    screens: list[ScreenGeometry] = []
    pattern = re.compile(
        r"^(\S+)\s+connected(?:\s+primary)?(?:\s+(\d+)x(\d+)\+(-?\d+)\+(-?\d+))?(?:\s+(normal|left|right|inverted))?"
    )
    for line in output.splitlines():
        match = pattern.match(line.strip())
        if not match:
            continue
        name, width, height, x, y, rotation = match.groups()
        if not width or not height:
            continue
        width_i = max(int(width), 1)
        height_i = max(int(height), 1)
        rotation_i = str(rotation or "normal").lower()
        screens.append(
            ScreenGeometry(
                index=len(screens),
                x=int(x),
                y=int(y),
                width=width_i,
                height=height_i,
                name=str(name or ""),
                rotation=rotation_i,
            )
        )
    return screens


def _get_linux_active_screens() -> list[ScreenGeometry]:
    if not sys.platform.startswith("linux"):
        return []
    active_output = _run_monitor_query(["xrandr", "--listactivemonitors"])
    active_screens = _parse_xrandr_listactivemonitors(active_output)
    query_output = _run_monitor_query(["xrandr", "--query"])
    query_screens = _parse_xrandr_query(query_output)
    return _merge_screen_lists(active_screens, query_screens)


def _merge_screen_lists(
    base_screens: list[ScreenGeometry],
    override_screens: list[ScreenGeometry],
) -> list[ScreenGeometry]:
    if not override_screens:
        return base_screens
    if not base_screens:
        return [
            ScreenGeometry(
                index=index,
                x=screen.x,
                y=screen.y,
                width=screen.width,
                height=screen.height,
                name=screen.name,
                rotation=screen.rotation,
            )
            for index, screen in enumerate(override_screens)
        ]

    unmatched_override = list(override_screens)
    matched_override_ids: set[int] = set()
    merged: list[ScreenGeometry] = []

    def _take_override(base: ScreenGeometry, ordinal: int) -> Optional[ScreenGeometry]:
        if base.name:
            base_aliases = _screen_name_aliases(base.name)
            for candidate in unmatched_override:
                if id(candidate) in matched_override_ids:
                    continue
                if candidate.name and base_aliases & _screen_name_aliases(candidate.name):
                    matched_override_ids.add(id(candidate))
                    return candidate
        for candidate in unmatched_override:
            if id(candidate) in matched_override_ids:
                continue
            if candidate.x == base.x and candidate.y == base.y:
                matched_override_ids.add(id(candidate))
                return candidate
        remaining = [candidate for candidate in unmatched_override if id(candidate) not in matched_override_ids]
        if len(remaining) == len(base_screens) - ordinal:
            candidate = remaining[0]
            matched_override_ids.add(id(candidate))
            return candidate
        return None

    for ordinal, base in enumerate(base_screens):
        override = _take_override(base, ordinal)
        if override is None:
            merged.append(base)
            continue
        merged.append(
            ScreenGeometry(
                index=base.index,
                x=override.x,
                y=override.y,
                width=override.width if override.width > 0 else base.width,
                height=override.height if override.height > 0 else base.height,
                name=base.name or override.name,
                rotation=override.rotation or base.rotation,
            )
        )

    next_index = max((screen.index for screen in merged), default=-1) + 1
    for candidate in unmatched_override:
        if id(candidate) in matched_override_ids:
            continue
        merged.append(
            ScreenGeometry(
                index=next_index,
                x=candidate.x,
                y=candidate.y,
                width=candidate.width,
                height=candidate.height,
                name=candidate.name,
                rotation=candidate.rotation,
            )
        )
        next_index += 1

    return merged


def get_monitor_screens() -> list[ScreenGeometry]:
    base_screens: list[ScreenGeometry] = []
    if get_monitors is not None:
        try:
            monitors = list(get_monitors())
        except Exception:
            monitors = []
        base_screens = [
            ScreenGeometry(
                index=index,
                x=int(getattr(monitor, "x", 0)),
                y=int(getattr(monitor, "y", 0)),
                width=max(int(getattr(monitor, "width", 0)), 1),
                height=max(int(getattr(monitor, "height", 0)), 1),
                name=str(getattr(monitor, "name", "") or ""),
            )
            for index, monitor in enumerate(monitors)
        ]

    linux_active_screens = _get_linux_active_screens()
    return _merge_screen_lists(base_screens, linux_active_screens)


def get_tk_screens(root) -> list[ScreenGeometry]:
    screens = get_monitor_screens()
    if screens:
        return screens
    return [
        ScreenGeometry(
            index=0,
            x=0,
            y=0,
            width=max(int(root.winfo_screenwidth()), 1),
            height=max(int(root.winfo_screenheight()), 1),
            name="primary",
        )
    ]


def select_screen(
    screens: list[ScreenGeometry],
    requested_selector: ScreenSelector,
    *,
    role: str,
    default_index: Optional[int] = None,
    allow_unvalidated_index: bool = False,
) -> Optional[ScreenGeometry]:
    if requested_selector is None:
        if default_index is None:
            return None
        if 0 <= default_index < len(screens):
            return screens[default_index]
        if allow_unvalidated_index:
            return ScreenGeometry(index=default_index, x=0, y=0, width=0, height=0, name=f"screen{default_index}")
        return None

    if isinstance(requested_selector, int):
        if 0 <= requested_selector < len(screens):
            return screens[requested_selector]
        if allow_unvalidated_index and get_monitors is None:
            return ScreenGeometry(
                index=requested_selector,
                x=0,
                y=0,
                width=0,
                height=0,
                name=f"screen{requested_selector}",
            )
        available = ", ".join(str(screen.index) for screen in screens)
        raise ValueError(
            f"Requested {role} screen {requested_selector}, but detected only {len(screens)} screen(s) "
            f"(available indices: {available})."
        )

    requested_name = str(requested_selector).strip()
    requested_normalized = _normalize_screen_name(requested_name)
    for screen in screens:
        screen_name = str(screen.name or "").strip()
        if screen_name and (
            screen_name.lower() == requested_name.lower()
            or _normalize_screen_name(screen_name) == requested_normalized
        ):
            return screen

    requested_aliases = _screen_name_aliases(requested_name)
    for screen in screens:
        if screen.name and requested_aliases & _screen_name_aliases(screen.name):
            return screen

    detected_names = [screen.name for screen in screens if screen.name]
    if not detected_names:
        raise RuntimeError(
            f"Named screen selection for {role} requires detected output names. "
            f"Requested '{requested_selector}', but no screen names were available."
        )
    raise ValueError(
        f"Requested {role} screen '{requested_selector}', but detected outputs were: "
        f"{', '.join(detected_names) if detected_names else 'none'}."
    )


def resolve_task_screens(
    screen_config: Optional[Dict[str, ScreenSelector]] = None,
    *,
    allow_same_screen: bool = False,
) -> tuple[ScreenGeometry, Optional[ScreenGeometry]]:
    cfg = screen_config or {}
    screens = get_monitor_screens()
    if not screens:
        screens = [ScreenGeometry(index=0, x=0, y=0, width=0, height=0, name="primary")]

    main_screen = select_screen(
        screens,
        cfg.get("main"),
        role="main",
        default_index=0,
        allow_unvalidated_index=True,
    )
    if main_screen is None:
        raise RuntimeError("Unable to resolve a main task screen")

    default_experimenter_index = None
    for candidate in screens:
        if candidate.index != main_screen.index:
            default_experimenter_index = candidate.index
            break

    experimenter_screen = select_screen(
        screens,
        cfg.get("experimenter"),
        role="experimenter",
        default_index=default_experimenter_index,
        allow_unvalidated_index=True,
    )
    if experimenter_screen is not None and experimenter_screen.index == main_screen.index:
        if allow_same_screen:
            # Same-screen mode intentionally suppresses the secondary preview window.
            return main_screen, None
        raise ValueError("Main and experimenter screens must resolve to different displays")

    return main_screen, experimenter_screen


def resolve_interface_screen(
    root,
    screen_config: Optional[Dict[str, ScreenSelector]] = None,
) -> ScreenGeometry:
    cfg = screen_config or {}
    screens = get_tk_screens(root)
    default_index = 1 if len(screens) > 1 else 0
    screen_info = select_screen(
        screens,
        cfg.get("experimenter"),
        role="experimenter",
        default_index=default_index,
        allow_unvalidated_index=True,
    )
    if screen_info is None:
        raise RuntimeError("Unable to resolve an experimenter interface screen")
    return screen_info


def place_tk_window_on_screen(
    root,
    screen_info: ScreenGeometry,
    *,
    min_width: int = 800,
    min_height: int = 600,
    margin_x: int = 20,
    margin_y: int = 20,
) -> tuple[int, int]:
    screen_width = max(int(screen_info.width), 1)
    screen_height = max(int(screen_info.height), 1)
    usable_width = max(1, screen_width - (2 * int(margin_x)))
    usable_height = max(1, screen_height - (2 * int(margin_y)) - 40)
    window_width = min(screen_width, max(int(min_width), usable_width))
    window_height = min(screen_height, max(int(min_height), usable_height))
    window_x = int(screen_info.x) + max(0, (screen_width - window_width) // 2)
    window_y = int(screen_info.y) + max(0, (screen_height - window_height) // 2)
    root.geometry(_format_geometry(window_width, window_height, window_x, window_y))
    return window_width, window_height


def get_psychopy_window_kwargs(
    screen_info: Optional[ScreenGeometry],
    *,
    fullscreen: bool,
    size: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {}
    has_geometry = (
        screen_info is not None
        and int(screen_info.width) > 0
        and int(screen_info.height) > 0
    )
    use_virtual_position = bool(has_geometry and sys.platform.startswith("linux"))

    if fullscreen:
        if screen_info is not None:
            kwargs["screen"] = (
                _get_pyglet_screen_index(screen_info)
                if has_geometry
                else int(screen_info.index)
            )
        kwargs["winType"] = "pyglet"
        kwargs["fullscr"] = True
        return kwargs

    if screen_info is not None and not use_virtual_position:
        kwargs["screen"] = int(screen_info.index)

    if size is not None:
        resolved_size = (int(size[0]), int(size[1]))
    elif has_geometry:
        resolved_size = (int(screen_info.width), int(screen_info.height))
    else:
        resolved_size = (1024, 768)

    kwargs["size"] = resolved_size
    kwargs["fullscr"] = False
    if has_geometry:
        x = max(0, (int(screen_info.width) - int(resolved_size[0])) // 2)
        y = max(0, (int(screen_info.height) - int(resolved_size[1])) // 2)
        if use_virtual_position:
            x += int(screen_info.x)
            y += int(screen_info.y)
        kwargs["pos"] = (x, y)
    return kwargs


def _get_pyglet_screens() -> list[Any]:
    return list(_get_pyglet_display().get_screens())


def _get_pyglet_display() -> Any:
    from pyglet import canvas

    return canvas.get_display()


def _get_pyglet_options() -> Dict[str, Any]:
    import pyglet

    return pyglet.options


def _screen_rect(screen: Any) -> tuple[int, int, int, int]:
    return (
        int(screen.x),
        int(screen.y),
        int(screen.width),
        int(screen.height),
    )


def _get_pyglet_screen_index(screen_info: ScreenGeometry) -> int:
    """Map an xrandr output to PsychoPy's screen list by geometry."""
    target = _screen_rect(screen_info)
    try:
        screens = _get_pyglet_screens()
    except Exception as exc:
        raise RuntimeError(
            f"PsychoPy could not enumerate displays while selecting {screen_info.name or target}: {exc}"
        ) from exc

    matches = [index for index, screen in enumerate(screens) if _screen_rect(screen) == target]
    if len(matches) == 1:
        return matches[0]

    available = ", ".join(str(_screen_rect(screen)) for screen in screens) or "none"
    raise RuntimeError(
        f"PsychoPy could not uniquely match main output {screen_info.name or '<unnamed>'} "
        f"at {target}; pyglet displays: {available}"
    )


@contextmanager
def _bind_linux_pyglet_fullscreen(screen_info: ScreenGeometry):
    """Make PsychoPy screen 0 mean the requested X11 monitor during creation."""
    display_class = type(_get_pyglet_display())
    original_get_screens = display_class.get_screens
    target = _screen_rect(screen_info)

    def target_first(display) -> list[Any]:
        screens = list(original_get_screens(display))
        matches = [screen for screen in screens if _screen_rect(screen) == target]
        if len(matches) != 1:
            available = ", ".join(str(_screen_rect(screen)) for screen in screens) or "none"
            raise RuntimeError(
                f"Pyglet lost main output {screen_info.name or '<unnamed>'} at {target}; "
                f"available displays: {available}"
            )
        return matches + [screen for screen in screens if screen is not matches[0]]

    options = _get_pyglet_options()
    option_name = "xlib_fullscreen_override_redirect"
    previous_option = options.get(option_name, _UNSET)
    display_class.get_screens = target_first
    options[option_name] = True
    try:
        yield
    finally:
        display_class.get_screens = original_get_screens
        if previous_option is _UNSET:
            options.pop(option_name, None)
        else:
            options[option_name] = previous_option


def verify_psychopy_window_screen(win: Any, screen_info: ScreenGeometry) -> str:
    """Confirm that a realized pyglet fullscreen window covers one output."""
    handle = getattr(win, "winHandle", None)
    get_location = getattr(handle, "get_location", None)
    get_size = getattr(handle, "get_size", None)
    if not callable(get_location) or not callable(get_size):
        raise RuntimeError("PsychoPy's pyglet window does not expose its realized geometry")

    actual = (*map(int, get_location()), *map(int, get_size()))
    expected = _screen_rect(screen_info)
    if actual != expected:
        raise RuntimeError(
            f"PsychoPy window realized at {actual}, not main output "
            f"{screen_info.name or '<unnamed>'} at {expected}"
        )
    return f"{screen_info.name or 'main output'} at {expected}"


def open_psychopy_window(
    visual_module: Any,
    screen_info: Optional[ScreenGeometry],
    *,
    fullscreen: bool,
    size: Optional[Sequence[int]] = None,
    **kwargs: Any,
) -> Any:
    """Open and verify a PsychoPy window on one resolved physical display."""
    window_kwargs = dict(kwargs)
    window_kwargs.update(
        get_psychopy_window_kwargs(
            screen_info,
            fullscreen=fullscreen,
            size=size,
        )
    )

    if fullscreen and screen_info is not None and sys.platform.startswith("linux"):
        # PsychoPy 2025.1 uses `screen` as both an X screen number and a
        # pyglet monitor index. Bind the target to index 0 to remove that ambiguity.
        with _bind_linux_pyglet_fullscreen(screen_info):
            window_kwargs["screen"] = 0
            win = visual_module.Window(**window_kwargs)
    else:
        win = visual_module.Window(**window_kwargs)

    if fullscreen and screen_info is not None:
        try:
            placement = verify_psychopy_window_screen(win, screen_info)
        except Exception:
            win.close()
            raise
        win._neuro_tasks_screen_placement = placement
    return win


def set_tk_window_fullscreen(window, screen_info: ScreenGeometry) -> None:
    """Place a Tk window on one output before requesting real fullscreen."""
    window.geometry(
        _format_geometry(
            max(int(screen_info.width), 1),
            max(int(screen_info.height), 1),
            int(screen_info.x),
            int(screen_info.y),
        )
    )
    window.update_idletasks()
    window.attributes("-fullscreen", True)


def resolve_scene_size(
    screen_info: Optional[ScreenGeometry],
    *,
    fullscreen: bool,
    requested_size: Optional[Sequence[int]] = None,
    realized_size: Optional[Sequence[int]] = None,
) -> tuple[int, int]:
    if fullscreen and screen_info is not None and int(screen_info.width) > 0 and int(screen_info.height) > 0:
        return (int(screen_info.width), int(screen_info.height))
    if (not fullscreen) and requested_size is not None:
        return (int(requested_size[0]), int(requested_size[1]))
    if realized_size is not None:
        return (int(realized_size[0]), int(realized_size[1]))
    if screen_info is not None and int(screen_info.width) > 0 and int(screen_info.height) > 0:
        return (int(screen_info.width), int(screen_info.height))
    return (1024, 768)


def resolve_screen_canvas_size(
    screen_info: Optional[ScreenGeometry],
    *,
    fallback: Sequence[int] = (1024, 768),
) -> tuple[int, int]:
    if screen_info is not None and int(screen_info.width) > 0 and int(screen_info.height) > 0:
        return (int(screen_info.width), int(screen_info.height))
    return (max(int(fallback[0]), 1), max(int(fallback[1]), 1))


def _preview_to_pil_rgba(image_obj) -> Optional[Image.Image]:
    if image_obj is None:
        return None
    if isinstance(image_obj, Image.Image):
        return image_obj.convert("RGBA")
    if isinstance(image_obj, (str, os.PathLike)):
        try:
            with Image.open(image_obj) as im:
                return im.convert("RGBA").copy()
        except Exception:
            return None
    try:
        arr = np.asarray(image_obj)
    except Exception:
        return None
    if arr.dtype.kind == "f":
        arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        return Image.fromarray(arr, mode="L").convert("RGBA")
    if arr.ndim == 3 and arr.shape[2] == 3:
        return Image.fromarray(arr, mode="RGB").convert("RGBA")
    if arr.ndim == 3 and arr.shape[2] == 4:
        return Image.fromarray(arr, mode="RGBA")
    return None


def serialize_preview_image(image_obj) -> Optional[Dict[str, Any]]:
    pil = _preview_to_pil_rgba(image_obj)
    if pil is None:
        return None
    buffer = io.BytesIO()
    pil.save(buffer, format="PNG")
    return {
        "kind": "png",
        "png_bytes": buffer.getvalue(),
        "size": [int(pil.size[0]), int(pil.size[1])],
    }


def compute_centered_aspect_fit(
    container_size: Sequence[float],
    content_size: Sequence[float],
) -> Dict[str, Any]:
    container_w = max(float(container_size[0]), 1.0)
    container_h = max(float(container_size[1]), 1.0)
    content_w = max(float(content_size[0]), 1.0)
    content_h = max(float(content_size[1]), 1.0)
    scale = min(container_w / content_w, container_h / content_h)
    box_w = content_w * scale
    box_h = content_h * scale
    return {
        "box_center": (0.0, 0.0),
        "box_size": (box_w, box_h),
        "scale": scale,
        "left_margin": max(0.0, (container_w - box_w) * 0.5),
        "right_margin": max(0.0, (container_w - box_w) * 0.5),
        "top_margin": max(0.0, (container_h - box_h) * 0.5),
        "bottom_margin": max(0.0, (container_h - box_h) * 0.5),
    }


def compute_aspect_cover_size(
    container_size: Sequence[float],
    content_size: Sequence[float],
) -> tuple[float, float]:
    """Uniformly scale content so it covers a container in both dimensions."""
    container_w = float(container_size[0])
    container_h = float(container_size[1])
    content_w = float(content_size[0])
    content_h = float(content_size[1])
    if container_w <= 0 or container_h <= 0:
        raise ValueError(f"Invalid container size: {tuple(container_size)}")
    if content_w <= 0 or content_h <= 0:
        raise ValueError(f"Invalid content size: {tuple(content_size)}")

    scale = max(container_w / content_w, container_h / content_h)
    return (content_w * scale, content_h * scale)


def scale_scene_length(value: float, main_size: Sequence[float], preview_size: Sequence[float]) -> float:
    main_w = max(float(main_size[0]), 1.0)
    main_h = max(float(main_size[1]), 1.0)
    preview_w = max(float(preview_size[0]), 1.0)
    preview_h = max(float(preview_size[1]), 1.0)
    scale = min(preview_w / main_w, preview_h / main_h)
    return float(value) * scale


def scale_scene_point(
    pos: Sequence[float],
    main_size: Sequence[float],
    preview_size: Sequence[float],
) -> tuple[float, float]:
    scale = scale_scene_length(1.0, main_size, preview_size)
    return (float(pos[0]) * scale, float(pos[1]) * scale)


def scale_scene_size(
    size: Sequence[float],
    main_size: Sequence[float],
    preview_size: Sequence[float],
) -> tuple[float, float]:
    scale = scale_scene_length(1.0, main_size, preview_size)
    return (max(1.0, float(size[0]) * scale), max(1.0, float(size[1]) * scale))


def format_elapsed_hms(elapsed_s: float) -> str:
    total_seconds = max(0, int(elapsed_s))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def format_experimenter_label(
    task_label: str,
    *,
    subject: Optional[str] = None,
    current_trial_num: Optional[int] = None,
    total_trials: Optional[int] = None,
) -> str:
    """Build the task/subject/trial label shown on an experimenter preview."""
    lines = []
    if str(task_label or "").strip():
        lines.append(str(task_label).strip())
    if subject is not None and str(subject).strip():
        lines.append(f"Subject: {str(subject).strip()}")
    if current_trial_num is not None:
        total_label = str(int(total_trials)) if total_trials is not None and int(total_trials) > 0 else "∞"
        lines.append(f"Trial: {int(current_trial_num)} / {total_label}")
    return "\n".join(lines)


def describe_screen(screen_info: Optional[ScreenGeometry]) -> str:
    if screen_info is None:
        return "none"
    label = screen_info.name or f"screen{screen_info.index}"
    rotation = "" if screen_info.rotation == "normal" else f" rotation={screen_info.rotation}"
    return (
        f"{label}(index={int(screen_info.index)} "
        f"size={int(screen_info.width)}x{int(screen_info.height)} "
        f"pos={int(screen_info.x)},{int(screen_info.y)}{rotation})"
    )


def _format_geometry(width: int, height: int, x: int, y: int) -> str:
    x_part = f"+{x}" if x >= 0 else f"-{abs(x)}"
    y_part = f"+{y}" if y >= 0 else f"-{abs(y)}"
    return f"{width}x{height}{x_part}{y_part}"


def _experimenter_panel_process(
    screen_info: ScreenGeometry,
    task_label: str,
    start_perf_s: float,
    update_interval_ms: int,
    exit_event,
    stop_event,
) -> None:
    import tkinter as tk

    root = tk.Tk()
    root.title("Experimenter")
    root.configure(bg="#e9ecef")
    set_tk_window_fullscreen(root, screen_info)
    try:
        root.attributes("-topmost", True)
    except Exception:
        pass

    time_var = tk.StringVar(value="00:00:00")
    task_var = tk.StringVar(value=task_label or "")

    container = tk.Frame(root, bg="#e9ecef")
    container.pack(fill="both", expand=True)

    task_label_widget = tk.Label(
        container,
        textvariable=task_var,
        font=("Helvetica", 20),
        bg="#e9ecef",
        fg="#4a4a4a",
    )
    task_label_widget.pack(pady=(70, 20))

    timer_label = tk.Label(
        container,
        textvariable=time_var,
        font=("Helvetica", 56, "bold"),
        bg="#e9ecef",
        fg="#111111",
    )
    timer_label.pack(pady=(20, 60))

    exit_button = tk.Button(
        container,
        text="exit",
        command=exit_event.set,
        font=("Helvetica", 28, "bold"),
        width=10,
        height=2,
        bg="#c94b4b",
        activebackground="#a63a3a",
        fg="#ffffff",
    )
    exit_button.pack()

    def _tick() -> None:
        if stop_event.is_set():
            root.destroy()
            return
        elapsed = time.perf_counter() - float(start_perf_s)
        time_var.set(format_elapsed_hms(elapsed))
        root.after(update_interval_ms, _tick)

    root.protocol("WM_DELETE_WINDOW", exit_event.set)
    root.after(0, _tick)
    root.mainloop()


class ExperimenterControlPanel:
    def __init__(
        self,
        screen_info: ScreenGeometry,
        *,
        task_label: str = "",
        start_perf_s: Optional[float] = None,
        update_interval_s: float = 0.2,
    ):
        self.screen_info = screen_info
        self.task_label = task_label
        self.start_perf_s = time.perf_counter() if start_perf_s is None else float(start_perf_s)
        self.update_interval_s = max(0.1, float(update_interval_s))
        self.exit_requested = False
        self._ctx = mp.get_context("spawn")
        self._exit_event = self._ctx.Event()
        self._stop_event = self._ctx.Event()
        self._process = self._ctx.Process(
            target=_experimenter_panel_process,
            args=(
                screen_info,
                task_label,
                self.start_perf_s,
                int(round(self.update_interval_s * 1000.0)),
                self._exit_event,
                self._stop_event,
            ),
            daemon=True,
        )
        self._process.start()

    def elapsed_seconds(self) -> float:
        return max(0.0, time.perf_counter() - self.start_perf_s)

    def poll(self) -> bool:
        if self.exit_requested:
            return True
        self.exit_requested = bool(self._exit_event.is_set())
        return self.exit_requested

    def wait(self, duration_s: float, *, step_s: float = 0.05) -> bool:
        deadline = time.perf_counter() + max(0.0, float(duration_s))
        while time.perf_counter() < deadline:
            if self.poll():
                return True
            remaining = deadline - time.perf_counter()
            if remaining > 0:
                time.sleep(min(max(0.01, step_s), remaining))
        return self.poll()

    def close(self) -> None:
        try:
            self._stop_event.set()
        except Exception:
            pass
        try:
            if self._process.is_alive():
                self._process.join(timeout=1.0)
        except Exception:
            pass
        try:
            if self._process.is_alive():
                self._process.terminate()
        except Exception:
            pass


def _preview_rgb255_to_psychopy(rgb_255: Sequence[int]) -> list[float]:
    return [max(-1.0, min(1.0, (float(v) / 127.5) - 1.0)) for v in rgb_255]


def _build_preview_image_stim(win, payload: Dict[str, Any], *, pos, size):
    from psychopy import visual

    image_payload = payload.get("image_payload")
    pil = None
    if isinstance(image_payload, dict) and image_payload.get("kind") == "png":
        try:
            pil = Image.open(io.BytesIO(image_payload["png_bytes"])).convert("RGBA")
        except Exception:
            pil = None
    if pil is None:
        pil = _preview_to_pil_rgba(payload.get("image"))
    if pil is None:
        return None

    alpha = np.asarray(pil.getchannel("A"), dtype=np.float32) / 255.0
    rgb = pil.convert("RGB")
    mask_pm1 = (alpha * 2.0) - 1.0
    return visual.ImageStim(
        win,
        image=rgb,
        mask=mask_pm1,
        units="pix",
        pos=pos,
        size=size,
        interpolate=False,
    )


def _normalize_reward_counts(value: Any) -> Optional[dict[int, int]]:
    if value is None:
        return None
    out = {0: 0, 1: 0, 2: 0, 3: 0}
    if isinstance(value, dict):
        items = value.items()
    else:
        try:
            items = enumerate(list(value))
        except Exception:
            return None
    for key, count in items:
        try:
            idx = int(key)
            if idx in out:
                out[idx] = max(0, int(count))
        except Exception:
            continue
    return out


def _normalize_status_counts(value: Any) -> Optional[dict[str, int]]:
    if value is None:
        return None
    if not isinstance(value, dict):
        return None
    out: dict[str, int] = {}
    for label, count in value.items():
        clean_label = str(label).strip()
        if not clean_label:
            continue
        try:
            out[clean_label] = max(0, int(count))
        except Exception:
            continue
    return out


def reward_level_color(level: int) -> tuple[int, int, int]:
    palette = {
        0: (220, 60, 60),
        1: (140, 140, 140),
        2: (230, 200, 40),
        3: (60, 180, 75),
    }
    return palette.get(int(level), (255, 255, 255))


def build_reward_hit_boxes(
    items: Sequence[Dict[str, Any]],
    *,
    hitbox_scale: float = 1.0,
    line_width: float = 6.0,
) -> list[Dict[str, Any]]:
    """Build preview outlines from stimulus entries carrying reward levels."""
    scale = max(0.0, float(hitbox_scale))
    boxes: list[Dict[str, Any]] = []
    for item in items:
        if item.get("reward_level") is None:
            continue
        pos = item.get("pos", (0.0, 0.0))
        size = item.get("size", (64.0, 64.0))
        boxes.append(
            {
                "pos": [float(pos[0]), float(pos[1])],
                "size": [float(size[0]) * scale, float(size[1]) * scale],
                "color": list(reward_level_color(int(item["reward_level"]))),
                "line_width": float(line_width),
            }
        )
    return boxes


def _experimenter_preview_process(
    screen_info: ScreenGeometry,
    task_label: str,
    start_perf_s: float,
    update_interval_ms: int,
    command_queue,
    reward_event,
    exit_event,
    mouse_visible: Optional[bool],
    initial_subject: Optional[str],
    initial_current_trial_num: Optional[int],
    initial_total_trials: Optional[int],
    initial_status_counts: Optional[Dict[str, int]],
    stop_event,
) -> None:
    import ctypes
    from psychopy import core, event, visual
    from pyglet import gl as GL
    from .video_playback import (
        SharedVideoFrameReader,
        center_crop_bounds,
    )
    preview_canvas_size = resolve_screen_canvas_size(screen_info)
    outside_bg_rgb = (30, 30, 30)
    preview_outline_rgb = (150, 150, 150)

    def _upload_rgba_texture(stim, rgba: np.ndarray) -> None:
        """Update an ImageStim texture as raw RGBA bytes without color remapping."""
        if rgba.dtype != np.uint8 or rgba.ndim != 3 or rgba.shape[2] != 4:
            raise ValueError("Shared video frames must be contiguous uint8 RGBA")
        if not rgba.flags.c_contiguous:
            raise ValueError("Shared video upload buffer must be contiguous")
        height, width = int(rgba.shape[0]), int(rgba.shape[1])
        pixel_pointer = rgba.ctypes.data_as(ctypes.POINTER(GL.GLubyte))
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, stim._texID)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        GL.glTexSubImage2D(
            GL.GL_TEXTURE_2D,
            0,
            0,
            0,
            width,
            height,
            GL.GL_RGBA,
            GL.GL_UNSIGNED_BYTE,
            pixel_pointer,
        )
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

    def _make_bg_rect(bg_rgb_255: Sequence[int]):
        return visual.Rect(
            win,
            width=preview_canvas_size[0],
            height=preview_canvas_size[1],
            fillColor=_preview_rgb255_to_psychopy(bg_rgb_255),
            fillColorSpace="rgb",
            lineColor=None,
            units="pix",
        )

    def _release_movie() -> None:
        nonlocal movie, movie_bg_rect, movie_outline_rect, movie_layout
        nonlocal shared_movie_active, shared_movie_stim, shared_movie_sequence
        nonlocal shared_movie_minimum_sequence, shared_movie_crop_bounds
        nonlocal shared_movie_upload_buffer
        nonlocal last_bg_rgb, static_scene
        if movie is not None:
            try:
                movie.stop(log=False)
            except Exception:
                pass
            try:
                if hasattr(movie, "unload"):
                    movie.unload(log=False)
            except Exception:
                pass
        movie = None
        shared_movie_active = False
        shared_movie_stim = None
        shared_movie_sequence = 0
        shared_movie_minimum_sequence = 1
        shared_movie_crop_bounds = None
        shared_movie_upload_buffer = None
        movie_bg_rect = None
        movie_outline_rect = None
        movie_layout = None
        static_scene = _build_static_scene(
            {
                "bg_rgb_255": last_bg_rgb,
                "main_size": preview_canvas_size,
                "subject": current_subject,
                "current_trial_num": current_trial_num,
                "total_trials": current_total_trials,
                "status_counts": current_status_counts,
            }
        )

    def _build_static_scene(payload: Dict[str, Any]) -> Dict[str, Any]:
        bg_rgb_255 = tuple(payload.get("bg_rgb_255", (0, 0, 0)))
        main_size = tuple(payload.get("main_size") or preview_canvas_size)
        layout = compute_centered_aspect_fit(preview_canvas_size, main_size)
        preview_size = layout["box_size"]
        box_center = layout["box_center"]
        canvas_bg_rect = _make_bg_rect(outside_bg_rgb)
        preview_bg_rect = visual.Rect(
            win,
            width=preview_size[0],
            height=preview_size[1],
            pos=box_center,
            fillColor=_preview_rgb255_to_psychopy(bg_rgb_255),
            fillColorSpace="rgb",
            lineColor=None,
            units="pix",
        )
        preview_outline_rect = visual.Rect(
            win,
            width=preview_size[0],
            height=preview_size[1],
            pos=box_center,
            fillColor=None,
            lineColor=_preview_rgb255_to_psychopy(preview_outline_rgb),
            lineColorSpace="rgb",
            lineWidth=2,
            units="pix",
        )

        def _map_pos(pos: Sequence[float]) -> tuple[float, float]:
            scaled = scale_scene_point(pos, main_size, preview_size)
            return (float(box_center[0]) + scaled[0], float(box_center[1]) + scaled[1])

        images = []
        for item in payload.get("images", []) or []:
            stim = _build_preview_image_stim(
                win,
                item,
                pos=_map_pos(item.get("pos", (0, 0))),
                size=scale_scene_size(item.get("size", (64, 64)), main_size, preview_size),
            )
            if stim is not None:
                images.append(stim)

        dots = []
        for item in payload.get("dots", []) or []:
            radius = max(1.0, scale_scene_length(float(item.get("radius", 4.0)), main_size, preview_size))
            dot = visual.Circle(
                win,
                radius=radius,
                fillColor=_preview_rgb255_to_psychopy(item.get("color", (255, 255, 255))),
                fillColorSpace="rgb",
                lineColor=None,
                units="pix",
                pos=_map_pos(item.get("pos", (0, 0))),
            )
            dots.append(dot)

        hit_boxes = []
        for item in payload.get("hit_boxes", []) or []:
            scaled_size = scale_scene_size(item.get("size", (64, 64)), main_size, preview_size)
            hit_boxes.append(
                visual.Rect(
                    win,
                    width=max(4.0, scaled_size[0]),
                    height=max(4.0, scaled_size[1]),
                    pos=_map_pos(item.get("pos", (0, 0))),
                    lineColor=_preview_rgb255_to_psychopy(item.get("color", (255, 255, 255))),
                    lineColorSpace="rgb",
                    lineWidth=max(
                        2.0,
                        scale_scene_length(
                            float(item.get("line_width", 4.0)),
                            main_size,
                            preview_size,
                        ),
                    ),
                    fillColor=None,
                    units="pix",
                )
            )

        fixation = None
        fixation_size = payload.get("fixation_size", None)
        if fixation_size is not None and float(fixation_size) > 0:
            fixation = visual.TextStim(
                win,
                text="+",
                units="pix",
                height=max(1.0, scale_scene_length(float(fixation_size), main_size, preview_size)),
                color=_preview_rgb255_to_psychopy(payload.get("fixation_color", (0, 0, 0))),
                colorSpace="rgb",
                pos=_map_pos((0, 0)),
            )

        highlight_box = None
        highlight_payload = payload.get("highlight_box")
        if isinstance(highlight_payload, dict):
            line_color = highlight_payload.get("color", (255, 255, 255))
            line_width = max(2.0, scale_scene_length(float(highlight_payload.get("line_width", 4.0)), main_size, preview_size))
            highlight_box = visual.Rect(
                win,
                width=max(4.0, scale_scene_size(highlight_payload.get("size", (64, 64)), main_size, preview_size)[0]),
                height=max(4.0, scale_scene_size(highlight_payload.get("size", (64, 64)), main_size, preview_size)[1]),
                pos=_map_pos(highlight_payload.get("pos", (0, 0))),
                lineColor=_preview_rgb255_to_psychopy(line_color),
                lineColorSpace="rgb",
                lineWidth=line_width,
                fillColor=None,
                units="pix",
            )

        return {
            "bg_rgb_255": bg_rgb_255,
            "canvas_bg_rect": canvas_bg_rect,
            "preview_bg_rect": preview_bg_rect,
            "preview_outline_rect": preview_outline_rect,
            "images": images,
            "dots": dots,
            "hit_boxes": hit_boxes,
            "fixation": fixation,
            "highlight_box": highlight_box,
            "reward_counts": _normalize_reward_counts(payload.get("reward_counts")),
            "status_counts": _normalize_status_counts(payload.get("status_counts")),
            "subject": payload.get("subject"),
            "current_trial_num": payload.get("current_trial_num"),
            "total_trials": payload.get("total_trials"),
            "layout": layout,
        }

    def _place_overlay_controls(layout: Optional[Dict[str, Any]]) -> None:
        if not layout:
            layout = compute_centered_aspect_fit(preview_canvas_size, preview_canvas_size)
        canvas_w = float(preview_canvas_size[0])
        canvas_h = float(preview_canvas_size[1])
        canvas_left = -canvas_w * 0.5
        canvas_right = canvas_w * 0.5
        canvas_top = canvas_h * 0.5
        canvas_bottom = -canvas_h * 0.5
        margin = max(10.0, min(canvas_w, canvas_h) * 0.018)
        box_center = layout.get("box_center", (0.0, 0.0))
        box_size = layout.get("box_size", preview_canvas_size)
        box_left = float(box_center[0]) - (float(box_size[0]) * 0.5)
        box_right = float(box_center[0]) + (float(box_size[0]) * 0.5)
        box_top = float(box_center[1]) + (float(box_size[1]) * 0.5)
        box_bottom = float(box_center[1]) - (float(box_size[1]) * 0.5)
        left_space = max(0.0, box_left - canvas_left)
        right_space = max(0.0, canvas_right - box_right)
        top_space = max(0.0, canvas_top - box_top)
        bottom_space = max(0.0, box_bottom - canvas_bottom)
        button_h = max(float(reward_button_height), float(exit_button_height))

        if left_space >= reward_button_width + (2.0 * margin) and right_space >= exit_button_width + (2.0 * margin):
            reward_pos = (canvas_left + (left_space * 0.5), canvas_top - margin - (reward_button_height * 0.5))
            exit_pos = (canvas_right - (right_space * 0.5), canvas_top - margin - (exit_button_height * 0.5))
            clock_pos = (
                canvas_left + margin,
                reward_pos[1] - (reward_button_height * 0.5) - system_time_text_height,
            )
            timer_pos = (
                clock_pos[0],
                clock_pos[1] - max(system_time_text_height, timer_text_height) * 1.15,
            )
            counts_pos = (canvas_left + margin, timer_pos[1] - max(56.0, reward_counts_text_height * 3.2))
            label_pos = (float(box_center[0]), box_bottom + margin + (task_label_height * 0.5))
        elif top_space >= button_h + (2.0 * margin):
            y = box_top + (top_space * 0.5)
            reward_pos = (canvas_left + margin + (reward_button_width * 0.5), y)
            exit_pos = (canvas_right - margin - (exit_button_width * 0.5), y)
            clock_pos = (
                float(box_center[0]) - (timer_text_height * 2.2),
                y + (system_time_text_height * 0.65),
            )
            timer_pos = (clock_pos[0], y - (timer_text_height * 0.65))
            counts_pos = (canvas_left + margin, y - max(40.0, reward_counts_text_height * 2.0))
            label_pos = (float(box_center[0]), box_bottom + margin + (task_label_height * 0.5))
        elif bottom_space >= button_h + (2.0 * margin):
            y = canvas_bottom + (bottom_space * 0.5)
            reward_pos = (canvas_left + margin + (reward_button_width * 0.5), y)
            exit_pos = (canvas_right - margin - (exit_button_width * 0.5), y)
            clock_pos = (canvas_left + margin, canvas_top - margin - system_time_text_height)
            timer_pos = (
                clock_pos[0],
                clock_pos[1] - max(system_time_text_height, timer_text_height) * 1.15,
            )
            counts_pos = (canvas_left + margin, timer_pos[1] - max(56.0, reward_counts_text_height * 3.2))
            label_pos = (float(box_center[0]), y)
        else:
            reward_pos = (canvas_left + margin + (reward_button_width * 0.5), canvas_top - margin - (reward_button_height * 0.5))
            exit_pos = (canvas_right - margin - (exit_button_width * 0.5), canvas_top - margin - (exit_button_height * 0.5))
            clock_pos = (
                canvas_left + margin,
                reward_pos[1] - (reward_button_height * 0.5) - system_time_text_height,
            )
            timer_pos = (
                clock_pos[0],
                clock_pos[1] - max(system_time_text_height, timer_text_height) * 1.15,
            )
            counts_pos = (canvas_left + margin, timer_pos[1] - max(56.0, reward_counts_text_height * 3.2))
            label_pos = (float(box_center[0]), box_bottom + margin + (task_label_height * 0.5))

        reward_button_rect.pos = reward_pos
        reward_button_text.pos = reward_pos
        exit_button_rect.pos = exit_pos
        exit_button_text.pos = exit_pos
        system_time_text.pos = clock_pos
        timer_text.pos = timer_pos
        reward_counts_text.pos = counts_pos
        if task_label_text is not None:
            task_label_text.pos = label_pos

    def _draw_overlay(layout: Optional[Dict[str, Any]] = None) -> None:
        _place_overlay_controls(layout or static_scene.get("layout"))
        elapsed = time.perf_counter() - float(start_perf_s)
        system_time_text.text = time.strftime("%H:%M:%S")
        system_time_text.draw()
        timer_text.text = format_elapsed_hms(elapsed)
        timer_text.draw()
        status_counts = static_scene.get("status_counts")
        reward_counts = static_scene.get("reward_counts")
        if status_counts is not None:
            reward_counts_text.text = "\n".join(
                f"{label}: {count}" for label, count in status_counts.items()
            )
            reward_counts_text.draw()
        elif reward_counts is not None:
            reward_counts_text.text = (
                f"R0: {reward_counts.get(0, 0)}\n"
                f"R1: {reward_counts.get(1, 0)}\n"
                f"R2: {reward_counts.get(2, 0)}\n"
                f"R3: {reward_counts.get(3, 0)}"
            )
            reward_counts_text.draw()
        if task_label_text is not None:
            task_label_text.text = format_experimenter_label(
                task_label,
                subject=static_scene.get("subject"),
                current_trial_num=static_scene.get("current_trial_num"),
                total_trials=static_scene.get("total_trials"),
            )
            task_label_text.draw()
        reward_button_rect.draw()
        reward_button_text.draw()
        exit_button_rect.draw()
        exit_button_text.draw()

    win = open_psychopy_window(
        visual,
        screen_info,
        fullscreen=True,
        units="pix",
        colorSpace="rgb",
        color=_preview_rgb255_to_psychopy((0, 0, 0)),
        allowStencil=False,
        allowGUI=False,
        waitBlanking=False,
    )
    configure_window_vsync(win, False)
    last_cursor_apply_s = 0.0
    if mouse_visible is not None:
        set_window_mouse_visible(win, bool(mouse_visible))
    mouse = event.Mouse(win=win)
    last_mouse_down = False
    last_bg_rgb = (0, 0, 0)
    current_reward_counts = None
    current_status_counts = _normalize_status_counts(initial_status_counts)
    current_highlight_box = None
    current_subject = initial_subject
    current_trial_num = initial_current_trial_num
    current_total_trials = initial_total_trials
    static_scene = _build_static_scene(
        {
            "bg_rgb_255": last_bg_rgb,
            "main_size": preview_canvas_size,
            "subject": current_subject,
            "current_trial_num": current_trial_num,
            "total_trials": current_total_trials,
            "status_counts": current_status_counts,
        }
    )
    movie = None
    movie_bg_rect = None
    movie_outline_rect = None
    movie_layout = None
    shared_movie_reader = None
    shared_movie_active = False
    shared_movie_stim = None
    shared_movie_sequence = 0
    shared_movie_minimum_sequence = 1
    shared_movie_crop_bounds = None
    shared_movie_upload_buffer = None
    task_label_text = None

    try:
        if task_label:
            task_label_text = visual.TextStim(
                win,
                text=task_label,
                units="pix",
                height=max(18.0, min(float(preview_canvas_size[0]), float(preview_canvas_size[1])) * 0.032),
                pos=(0.0, -float(preview_canvas_size[1]) * 0.44),
                color=_preview_rgb255_to_psychopy((230, 230, 230)),
                colorSpace="rgb",
            )

        task_label_height = max(18.0, min(float(preview_canvas_size[0]), float(preview_canvas_size[1])) * 0.032)
        timer_text_height = max(22.0, min(float(preview_canvas_size[0]), float(preview_canvas_size[1])) * 0.04)
        system_time_text_height = max(
            18.0,
            min(float(preview_canvas_size[0]), float(preview_canvas_size[1])) * 0.028,
        )
        reward_counts_text_height = max(16.0, min(float(preview_canvas_size[0]), float(preview_canvas_size[1])) * 0.028)
        system_time_text = visual.TextStim(
            win,
            text="00:00:00",
            units="pix",
            height=system_time_text_height,
            pos=(-float(preview_canvas_size[0]) * 0.35, float(preview_canvas_size[1]) * 0.47),
            alignText="left",
            anchorHoriz="left",
            color=_preview_rgb255_to_psychopy((210, 210, 210)),
            colorSpace="rgb",
        )
        timer_text = visual.TextStim(
            win,
            text="00:00:00",
            units="pix",
            height=timer_text_height,
            pos=(-float(preview_canvas_size[0]) * 0.35, float(preview_canvas_size[1]) * 0.44),
            alignText="left",
            anchorHoriz="left",
            color=_preview_rgb255_to_psychopy((255, 255, 255)),
            colorSpace="rgb",
        )
        reward_counts_text = visual.TextStim(
            win,
            text="",
            units="pix",
            height=reward_counts_text_height,
            pos=(-float(preview_canvas_size[0]) * 0.35, float(preview_canvas_size[1]) * 0.30),
            alignText="left",
            anchorHoriz="left",
            color=_preview_rgb255_to_psychopy((255, 255, 255)),
            colorSpace="rgb",
        )
        reward_button_width = max(84.0, min(140.0, float(preview_canvas_size[0]) * 0.08))
        reward_button_height = max(44.0, min(64.0, float(preview_canvas_size[1]) * 0.065))
        exit_button_width = max(96.0, min(150.0, float(preview_canvas_size[0]) * 0.10))
        exit_button_height = reward_button_height
        reward_button_rect = visual.Rect(
            win,
            width=reward_button_width,
            height=reward_button_height,
            pos=(
                -float(preview_canvas_size[0]) * 0.5 + reward_button_width * 0.5 + 18.0,
                float(preview_canvas_size[1]) * 0.5 - reward_button_height * 0.5 - 18.0,
            ),
            fillColor=_preview_rgb255_to_psychopy((68, 128, 88)),
            fillColorSpace="rgb",
            lineColor=None,
            units="pix",
        )
        reward_button_text = visual.TextStim(
            win,
            text="rew.",
            units="pix",
            height=max(18.0, reward_button_height * 0.42),
            pos=reward_button_rect.pos,
            color=_preview_rgb255_to_psychopy((255, 255, 255)),
            colorSpace="rgb",
        )
        exit_button_rect = visual.Rect(
            win,
            width=exit_button_width,
            height=exit_button_height,
            pos=(float(preview_canvas_size[0]) * 0.39, float(preview_canvas_size[1]) * 0.43),
            fillColor=_preview_rgb255_to_psychopy((201, 75, 75)),
            fillColorSpace="rgb",
            lineColor=None,
            units="pix",
        )
        exit_button_text = visual.TextStim(
            win,
            text="exit",
            units="pix",
            height=max(18.0, exit_button_height * 0.42),
            pos=exit_button_rect.pos,
            color=_preview_rgb255_to_psychopy((255, 255, 255)),
            colorSpace="rgb",
        )

        while not stop_event.is_set():
            redraw_requested = False
            if mouse_visible is not None and time.perf_counter() - last_cursor_apply_s >= 0.5:
                set_window_mouse_visible(win, bool(mouse_visible))
                last_cursor_apply_s = time.perf_counter()

            while True:
                try:
                    payload = command_queue.get_nowait()
                except queue.Empty:
                    break

                try:
                    command_type = str(payload.get("type", "")).strip().lower()
                    redraw_requested = True
                    if "reward_counts" in payload:
                        current_reward_counts = _normalize_reward_counts(payload.get("reward_counts"))
                    if "status_counts" in payload:
                        current_status_counts = _normalize_status_counts(payload.get("status_counts"))
                    if "highlight_box" in payload:
                        current_highlight_box = payload.get("highlight_box")
                    if "subject" in payload:
                        current_subject = payload.get("subject")
                    if "current_trial_num" in payload:
                        current_trial_num = payload.get("current_trial_num")
                    if "total_trials" in payload:
                        current_total_trials = payload.get("total_trials")
                    scene_payload = dict(payload)
                    scene_payload["reward_counts"] = current_reward_counts
                    scene_payload["status_counts"] = current_status_counts
                    scene_payload["highlight_box"] = current_highlight_box
                    scene_payload["subject"] = current_subject
                    scene_payload["current_trial_num"] = current_trial_num
                    scene_payload["total_trials"] = current_total_trials
                    if command_type == "static_scene":
                        _release_movie()
                        last_bg_rgb = tuple(payload.get("bg_rgb_255", last_bg_rgb))
                        static_scene = _build_static_scene(scene_payload)
                    elif command_type == "play_video":
                        _release_movie()
                        last_bg_rgb = tuple(payload.get("bg_rgb_255", last_bg_rgb))
                        movie_layout = compute_centered_aspect_fit(
                            preview_canvas_size,
                            tuple(payload.get("main_size") or preview_canvas_size),
                        )
                        movie_bg_rect = _make_bg_rect(outside_bg_rgb)
                        movie_outline_rect = visual.Rect(
                            win,
                            width=movie_layout["box_size"][0],
                            height=movie_layout["box_size"][1],
                            pos=movie_layout["box_center"],
                            fillColor=None,
                            lineColor=_preview_rgb255_to_psychopy(preview_outline_rgb),
                            lineColorSpace="rgb",
                            lineWidth=2,
                            units="pix",
                        )
                        from psychopy.visual.vlcmoviestim import VlcMovieStim

                        movie = VlcMovieStim(
                            win,
                            filename=str(payload["video_path"]),
                            units="pix",
                            size=movie_layout["box_size"],
                            pos=movie_layout["box_center"],
                            loop=False,
                            autoStart=False,
                            noAudio=True,
                        )
                        movie.size = movie_layout["box_size"]
                        movie.pos = movie_layout["box_center"]
                        movie.play(log=False)
                    elif command_type == "play_shared_video":
                        _release_movie()
                        last_bg_rgb = tuple(payload.get("bg_rgb_255", last_bg_rgb))
                        movie_layout = compute_centered_aspect_fit(
                            preview_canvas_size,
                            tuple(payload.get("main_size") or preview_canvas_size),
                        )
                        movie_bg_rect = _make_bg_rect(outside_bg_rgb)
                        movie_outline_rect = visual.Rect(
                            win,
                            width=movie_layout["box_size"][0],
                            height=movie_layout["box_size"][1],
                            pos=movie_layout["box_center"],
                            fillColor=None,
                            lineColor=_preview_rgb255_to_psychopy(preview_outline_rgb),
                            lineColorSpace="rgb",
                            lineWidth=2,
                            units="pix",
                        )
                        descriptor = dict(payload["shared_frame_buffer"])
                        if (
                            shared_movie_reader is None
                            or shared_movie_reader.name != str(descriptor["name"])
                        ):
                            if shared_movie_reader is not None:
                                shared_movie_reader.close()
                            shared_movie_reader = SharedVideoFrameReader(
                                str(descriptor["name"]),
                                int(descriptor["maximum_frame_bytes"]),
                                slot_count=int(descriptor.get("slot_count", 4)),
                                unregister_resource_tracker=True,
                            )
                        shared_movie_minimum_sequence = int(
                            payload.get("minimum_sequence", 1)
                        )
                        shared_movie_sequence = shared_movie_minimum_sequence - 1
                        shared_movie_crop_bounds = center_crop_bounds(
                            tuple(payload["video_size"]),
                            tuple(payload.get("main_size") or preview_canvas_size),
                        )
                        shared_movie_active = True
                    elif command_type == "clear_scene":
                        _release_movie()
                        last_bg_rgb = tuple(payload.get("bg_rgb_255", last_bg_rgb))
                        static_scene = _build_static_scene(scene_payload if scene_payload else {"bg_rgb_255": last_bg_rgb, "main_size": preview_canvas_size})
                except Exception:
                    continue

            try:
                if event.getKeys(keyList=["r"]):
                    reward_event.set()
            except Exception:
                pass

            try:
                mouse_down = any(mouse.getPressed())
            except Exception:
                mouse_down = False
            if mouse_down and (not last_mouse_down):
                try:
                    _place_overlay_controls(
                        movie_layout
                        if movie is not None or shared_movie_active
                        else static_scene.get("layout")
                    )
                    mouse_pos = mouse.getPos()
                    if reward_button_rect.contains(mouse_pos):
                        reward_event.set()
                    elif exit_button_rect.contains(mouse_pos):
                        exit_event.set()
                except Exception:
                    pass
            last_mouse_down = mouse_down

            try:
                if movie is not None:
                    if movie_bg_rect is not None:
                        movie_bg_rect.draw()
                    movie.draw()
                    if movie_outline_rect is not None:
                        movie_outline_rect.draw()
                    _draw_overlay(movie_layout)
                    win.flip()
                    if bool(getattr(movie, "isFinished", False)):
                        _release_movie()
                    core.wait(0.005)
                    continue

                if shared_movie_active:
                    shared_frame_updated = False
                    if shared_movie_reader is not None:
                        shared_frame = shared_movie_reader.read_latest(
                            shared_movie_sequence,
                            minimum_sequence=shared_movie_minimum_sequence,
                        )
                        if shared_frame is not None:
                            shared_frame_updated = True
                            shared_movie_sequence = shared_frame.sequence
                            left, top, right, bottom = shared_movie_crop_bounds
                            cropped_view = shared_frame.rgba[top:bottom, left:right]
                            expected_shape = (
                                int(bottom - top),
                                int(right - left),
                                4,
                            )
                            if (
                                shared_movie_upload_buffer is None
                                or shared_movie_upload_buffer.shape != expected_shape
                            ):
                                shared_movie_upload_buffer = np.empty(
                                    expected_shape,
                                    dtype=np.uint8,
                                )
                            np.copyto(shared_movie_upload_buffer, cropped_view)
                            if shared_movie_stim is None:
                                blank_image = Image.new(
                                    "RGBA",
                                    (expected_shape[1], expected_shape[0]),
                                    (0, 0, 0, 255),
                                )
                                shared_movie_stim = visual.ImageStim(
                                    win,
                                    image=blank_image,
                                    units="pix",
                                    size=movie_layout["box_size"],
                                    pos=movie_layout["box_center"],
                                    interpolate=True,
                                    flipVert=True,
                                    autoLog=False,
                                )
                            _upload_rgba_texture(
                                shared_movie_stim,
                                shared_movie_upload_buffer,
                            )
                    if not shared_frame_updated and not redraw_requested:
                        core.wait(0.002)
                        continue
                    if movie_bg_rect is not None:
                        movie_bg_rect.draw()
                    if shared_movie_stim is not None:
                        shared_movie_stim.draw()
                    if movie_outline_rect is not None:
                        movie_outline_rect.draw()
                    _draw_overlay(movie_layout)
                    win.flip()
                    continue

                static_scene["canvas_bg_rect"].draw()
                static_scene["preview_bg_rect"].draw()
                for stim in static_scene["dots"]:
                    stim.draw()
                for stim in static_scene["images"]:
                    stim.draw()
                for hit_box in static_scene["hit_boxes"]:
                    hit_box.draw()
                if static_scene["fixation"] is not None:
                    static_scene["fixation"].draw()
                if static_scene["highlight_box"] is not None:
                    static_scene["highlight_box"].draw()
                static_scene["preview_outline_rect"].draw()
                _draw_overlay()
                win.flip()
            except Exception:
                static_scene = _build_static_scene(
                    {
                        "bg_rgb_255": last_bg_rgb,
                        "main_size": preview_canvas_size,
                        "reward_counts": current_reward_counts,
                        "status_counts": current_status_counts,
                        "highlight_box": current_highlight_box,
                        "subject": current_subject,
                        "current_trial_num": current_trial_num,
                        "total_trials": current_total_trials,
                    }
                )
            core.wait(max(0.02, float(update_interval_ms) / 1000.0))
    finally:
        _release_movie()
        if shared_movie_reader is not None:
            try:
                shared_movie_reader.close()
            except Exception:
                pass
        try:
            win.close()
        except Exception:
            pass


class ExperimenterPreview:
    def __init__(
        self,
        screen_info: ScreenGeometry,
        *,
        task_label: str = "",
        subject: Optional[str] = None,
        current_trial_num: Optional[int] = None,
        total_trials: Optional[int] = None,
        status_counts: Optional[Dict[str, int]] = None,
        start_perf_s: Optional[float] = None,
        update_interval_s: float = 0.1,
        mouse_visible: Optional[bool] = True,
    ):
        self.screen_info = screen_info
        self.task_label = task_label
        self.subject = subject
        self.current_trial_num = current_trial_num
        self.total_trials = total_trials
        self.status_counts = dict(status_counts) if status_counts is not None else None
        self.start_perf_s = time.perf_counter() if start_perf_s is None else float(start_perf_s)
        self.update_interval_s = max(0.05, float(update_interval_s))
        self.exit_requested = False
        self._ctx = mp.get_context("spawn")
        self._queue = self._ctx.Queue(maxsize=4)
        self._reward_event = self._ctx.Event()
        self._exit_event = self._ctx.Event()
        self._stop_event = self._ctx.Event()
        self._process = self._ctx.Process(
            target=_experimenter_preview_process,
            args=(
                screen_info,
                task_label,
                self.start_perf_s,
                int(round(self.update_interval_s * 1000.0)),
                self._queue,
                self._reward_event,
                self._exit_event,
                mouse_visible,
                subject,
                current_trial_num,
                total_trials,
                self.status_counts,
                self._stop_event,
            ),
            daemon=True,
        )
        self._process.start()

    def poll(self) -> bool:
        if self.exit_requested:
            return True
        self.exit_requested = bool(self._exit_event.is_set())
        return self.exit_requested

    def wait(self, duration_s: float, *, step_s: float = 0.05) -> bool:
        deadline = time.perf_counter() + max(0.0, float(duration_s))
        while time.perf_counter() < deadline:
            if self.poll():
                return True
            remaining = deadline - time.perf_counter()
            if remaining > 0:
                time.sleep(min(max(0.01, step_s), remaining))
        return self.poll()

    def consume_manual_reward_request(self) -> bool:
        if not self._reward_event.is_set():
            return False
        self._reward_event.clear()
        return True

    def set_trial_progress(self, current_trial_num: int, total_trials: Optional[int]) -> None:
        self.current_trial_num = int(current_trial_num)
        self.total_trials = None if total_trials is None else int(total_trials)

    def set_status_counts(self, status_counts: Optional[Dict[str, int]]) -> None:
        self.status_counts = dict(status_counts) if status_counts is not None else None

    def _send(self, payload: Dict[str, Any]) -> None:
        payload = dict(payload)
        subject = getattr(self, "subject", None)
        current_trial_num = getattr(self, "current_trial_num", None)
        total_trials = getattr(self, "total_trials", None)
        status_counts = getattr(self, "status_counts", None)
        if subject is not None:
            payload.setdefault("subject", str(subject))
        if current_trial_num is not None:
            payload.setdefault("current_trial_num", int(current_trial_num))
        if total_trials is not None:
            payload.setdefault("total_trials", int(total_trials))
        if status_counts is not None:
            payload.setdefault("status_counts", dict(status_counts))
        if self.poll() or not self._process.is_alive():
            return
        try:
            self._queue.put_nowait(payload)
            return
        except queue.Full:
            try:
                self._queue.get_nowait()
            except (queue.Empty, OSError, ValueError):
                return
        except (OSError, ValueError):
            return
        try:
            self._queue.put_nowait(payload)
        except (queue.Full, OSError, ValueError):
            pass

    def show_static_scene(
        self,
        *,
        bg_rgb_255: Sequence[int],
        main_size: Sequence[int],
        images: Optional[list[Dict[str, Any]]] = None,
        dots: Optional[list[Dict[str, Any]]] = None,
        hit_boxes: Optional[list[Dict[str, Any]]] = None,
        fixation_size: Optional[float] = None,
        fixation_color: Sequence[int] = (0, 0, 0),
        reward_counts: Any = _UNSET,
        status_counts: Any = _UNSET,
        highlight_box: Any = _UNSET,
    ) -> None:
        payload: Dict[str, Any] = {
            "type": "static_scene",
            "bg_rgb_255": list(bg_rgb_255),
            "main_size": [int(main_size[0]), int(main_size[1])],
            "images": list(images or []),
            "dots": list(dots or []),
            "hit_boxes": list(hit_boxes or []),
            "fixation_size": fixation_size,
            "fixation_color": list(fixation_color),
        }
        if reward_counts is not _UNSET:
            payload["reward_counts"] = dict(reward_counts) if reward_counts is not None else None
        if status_counts is not _UNSET:
            payload["status_counts"] = dict(status_counts) if status_counts is not None else None
        if highlight_box is not _UNSET:
            payload["highlight_box"] = dict(highlight_box) if highlight_box is not None else None
        self._send(payload)

    def clear_scene(
        self,
        *,
        bg_rgb_255: Sequence[int],
        main_size: Optional[Sequence[int]] = None,
        reward_counts: Any = _UNSET,
        status_counts: Any = _UNSET,
        highlight_box: Any = _UNSET,
    ) -> None:
        payload: Dict[str, Any] = {
            "type": "clear_scene",
            "bg_rgb_255": list(bg_rgb_255),
        }
        if main_size is not None:
            payload["main_size"] = [int(main_size[0]), int(main_size[1])]
        if reward_counts is not _UNSET:
            payload["reward_counts"] = dict(reward_counts) if reward_counts is not None else None
        if status_counts is not _UNSET:
            payload["status_counts"] = dict(status_counts) if status_counts is not None else None
        if highlight_box is not _UNSET:
            payload["highlight_box"] = dict(highlight_box) if highlight_box is not None else None
        self._send(payload)

    def play_video(
        self,
        video_path: str,
        *,
        bg_rgb_255: Sequence[int],
        main_size: Optional[Sequence[int]] = None,
    ) -> None:
        payload: Dict[str, Any] = {
            "type": "play_video",
            "video_path": str(video_path),
            "bg_rgb_255": list(bg_rgb_255),
        }
        if main_size is not None:
            payload["main_size"] = [int(main_size[0]), int(main_size[1])]
        self._send(payload)

    def play_shared_video(
        self,
        *,
        shared_frame_buffer: Dict[str, Any],
        minimum_sequence: int,
        video_size: Sequence[int],
        bg_rgb_255: Sequence[int],
        main_size: Optional[Sequence[int]] = None,
    ) -> None:
        """Mirror frames published by the main process without decoding again."""
        payload: Dict[str, Any] = {
            "type": "play_shared_video",
            "shared_frame_buffer": dict(shared_frame_buffer),
            "minimum_sequence": int(minimum_sequence),
            "video_size": [int(video_size[0]), int(video_size[1])],
            "bg_rgb_255": list(bg_rgb_255),
        }
        if main_size is not None:
            payload["main_size"] = [int(main_size[0]), int(main_size[1])]
        self._send(payload)

    def close(self) -> None:
        try:
            self._stop_event.set()
        except Exception:
            pass
        try:
            if self._process.is_alive():
                self._process.join(timeout=1.0)
        except Exception:
            pass
        try:
            if self._process.is_alive():
                self._process.terminate()
        except Exception:
            pass
