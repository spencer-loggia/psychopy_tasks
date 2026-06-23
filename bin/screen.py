"""
Shared helpers for resolving monitor selectors and managing experimenter displays.
"""
from __future__ import annotations

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


@dataclass(frozen=True)
class ScreenGeometry:
    index: int
    x: int
    y: int
    width: int
    height: int
    name: str = ""


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
        if str(rotation or "").lower() in {"left", "right"} and width_i > height_i:
            width_i, height_i = height_i, width_i
        screens.append(
            ScreenGeometry(
                index=len(screens),
                x=int(x),
                y=int(y),
                width=width_i,
                height=height_i,
                name=str(name or ""),
            )
        )
    return screens


def _get_linux_active_screens() -> list[ScreenGeometry]:
    if not sys.platform.startswith("linux"):
        return []
    output = _run_monitor_query(["xrandr", "--listactivemonitors"])
    screens = _parse_xrandr_listactivemonitors(output)
    if screens:
        return screens
    output = _run_monitor_query(["xrandr", "--query"])
    return _parse_xrandr_query(output)


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
    if screen_info is not None:
        kwargs["screen"] = int(screen_info.index)

    if fullscreen:
        kwargs["fullscr"] = True
        return kwargs

    if size is not None:
        resolved_size = (int(size[0]), int(size[1]))
    elif screen_info is not None and int(screen_info.width) > 0 and int(screen_info.height) > 0:
        resolved_size = (int(screen_info.width), int(screen_info.height))
    else:
        resolved_size = (1024, 768)

    kwargs["size"] = resolved_size
    kwargs["fullscr"] = False
    if screen_info is not None and int(screen_info.width) > 0 and int(screen_info.height) > 0:
        # When `screen` is set, PsychoPy places the window on that physical display.
        # `pos` should therefore be local to that display, not the virtual desktop.
        x = max(0, (int(screen_info.width) - int(resolved_size[0])) // 2)
        y = max(0, (int(screen_info.height) - int(resolved_size[1])) // 2)
        kwargs["pos"] = (x, y)
    return kwargs


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


def describe_screen(screen_info: Optional[ScreenGeometry]) -> str:
    if screen_info is None:
        return "none"
    label = screen_info.name or f"screen{screen_info.index}"
    return (
        f"{label}(index={int(screen_info.index)} "
        f"size={int(screen_info.width)}x{int(screen_info.height)} "
        f"pos={int(screen_info.x)},{int(screen_info.y)})"
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
    root.overrideredirect(True)
    root.geometry(
        _format_geometry(
            max(int(screen_info.width), 800),
            max(int(screen_info.height), 600),
            int(screen_info.x),
            int(screen_info.y),
        )
    )
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


def _reward_level_color(level: int) -> tuple[int, int, int]:
    palette = {
        0: (220, 60, 60),
        1: (0, 0, 0),
        2: (230, 200, 40),
        3: (60, 180, 75),
    }
    return palette.get(int(level), (255, 255, 255))


def _experimenter_preview_process(
    screen_info: ScreenGeometry,
    task_label: str,
    start_perf_s: float,
    update_interval_ms: int,
    command_queue,
    reward_event,
    exit_event,
    stop_event,
) -> None:
    from psychopy import core, event, visual
    preview_canvas_size = resolve_screen_canvas_size(screen_info)
    preview_pos = (0, 0)
    outside_bg_rgb = (30, 30, 30)
    preview_outline_rgb = (150, 150, 150)

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
        nonlocal movie, movie_bg_rect, movie_outline_rect, movie_layout, last_bg_rgb, static_scene
        if movie is None:
            return
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
        movie_bg_rect = None
        movie_outline_rect = None
        movie_layout = None
        static_scene = _build_static_scene({"bg_rgb_255": last_bg_rgb, "main_size": preview_canvas_size})

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
            "fixation": fixation,
            "highlight_box": highlight_box,
            "reward_counts": _normalize_reward_counts(payload.get("reward_counts")),
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
            timer_pos = (canvas_left + margin, reward_pos[1] - (reward_button_height * 0.5) - timer_text_height)
            counts_pos = (canvas_left + margin, timer_pos[1] - max(56.0, reward_counts_text_height * 3.2))
            label_pos = (float(box_center[0]), box_bottom + margin + (task_label_height * 0.5))
        elif top_space >= button_h + (2.0 * margin):
            y = box_top + (top_space * 0.5)
            reward_pos = (canvas_left + margin + (reward_button_width * 0.5), y)
            exit_pos = (canvas_right - margin - (exit_button_width * 0.5), y)
            timer_pos = (float(box_center[0]) - (timer_text_height * 2.2), y)
            counts_pos = (canvas_left + margin, y - max(40.0, reward_counts_text_height * 2.0))
            label_pos = (float(box_center[0]), box_bottom + margin + (task_label_height * 0.5))
        elif bottom_space >= button_h + (2.0 * margin):
            y = canvas_bottom + (bottom_space * 0.5)
            reward_pos = (canvas_left + margin + (reward_button_width * 0.5), y)
            exit_pos = (canvas_right - margin - (exit_button_width * 0.5), y)
            timer_pos = (canvas_left + margin, canvas_top - margin - timer_text_height)
            counts_pos = (canvas_left + margin, timer_pos[1] - max(56.0, reward_counts_text_height * 3.2))
            label_pos = (float(box_center[0]), y)
        else:
            reward_pos = (canvas_left + margin + (reward_button_width * 0.5), canvas_top - margin - (reward_button_height * 0.5))
            exit_pos = (canvas_right - margin - (exit_button_width * 0.5), canvas_top - margin - (exit_button_height * 0.5))
            timer_pos = (canvas_left + margin, reward_pos[1] - (reward_button_height * 0.5) - timer_text_height)
            counts_pos = (canvas_left + margin, timer_pos[1] - max(56.0, reward_counts_text_height * 3.2))
            label_pos = (float(box_center[0]), box_bottom + margin + (task_label_height * 0.5))

        reward_button_rect.pos = reward_pos
        reward_button_text.pos = reward_pos
        exit_button_rect.pos = exit_pos
        exit_button_text.pos = exit_pos
        timer_text.pos = timer_pos
        reward_counts_text.pos = counts_pos
        if task_label_text is not None:
            task_label_text.pos = label_pos

    def _draw_overlay(layout: Optional[Dict[str, Any]] = None) -> None:
        _place_overlay_controls(layout or static_scene.get("layout"))
        elapsed = time.perf_counter() - float(start_perf_s)
        timer_text.text = format_elapsed_hms(elapsed)
        timer_text.draw()
        reward_counts = static_scene.get("reward_counts")
        if reward_counts is not None:
            reward_counts_text.text = (
                f"R0: {reward_counts.get(0, 0)}\n"
                f"R1: {reward_counts.get(1, 0)}\n"
                f"R2: {reward_counts.get(2, 0)}\n"
                f"R3: {reward_counts.get(3, 0)}"
            )
            reward_counts_text.draw()
        if task_label_text is not None:
            task_label_text.draw()
        reward_button_rect.draw()
        reward_button_text.draw()
        exit_button_rect.draw()
        exit_button_text.draw()

    win = visual.Window(
        size=preview_canvas_size,
        pos=preview_pos,
        fullscr=False,
        screen=int(screen_info.index),
        units="pix",
        colorSpace="rgb",
        color=_preview_rgb255_to_psychopy((0, 0, 0)),
        allowStencil=False,
        allowGUI=False,
    )
    mouse = event.Mouse(win=win)
    last_mouse_down = False
    last_bg_rgb = (0, 0, 0)
    static_scene = _build_static_scene({"bg_rgb_255": last_bg_rgb, "main_size": preview_canvas_size})
    movie = None
    movie_bg_rect = None
    movie_outline_rect = None
    movie_layout = None
    task_label_text = None
    current_reward_counts = None
    current_highlight_box = None

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
        reward_counts_text_height = max(16.0, min(float(preview_canvas_size[0]), float(preview_canvas_size[1])) * 0.028)
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
            while True:
                try:
                    payload = command_queue.get_nowait()
                except queue.Empty:
                    break

                try:
                    command_type = str(payload.get("type", "")).strip().lower()
                    if "reward_counts" in payload:
                        current_reward_counts = _normalize_reward_counts(payload.get("reward_counts"))
                    if "highlight_box" in payload:
                        current_highlight_box = payload.get("highlight_box")
                    scene_payload = dict(payload)
                    scene_payload["reward_counts"] = current_reward_counts
                    scene_payload["highlight_box"] = current_highlight_box
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
                    elif command_type == "clear_scene":
                        _release_movie()
                        last_bg_rgb = tuple(payload.get("bg_rgb_255", last_bg_rgb))
                        static_scene = _build_static_scene(scene_payload if scene_payload else {"bg_rgb_255": last_bg_rgb, "main_size": preview_canvas_size})
                except Exception:
                    continue

            try:
                mouse_down = any(mouse.getPressed())
            except Exception:
                mouse_down = False
            if mouse_down and (not last_mouse_down):
                try:
                    _place_overlay_controls(movie_layout if movie is not None else static_scene.get("layout"))
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
                    continue

                static_scene["canvas_bg_rect"].draw()
                static_scene["preview_bg_rect"].draw()
                for stim in static_scene["dots"]:
                    stim.draw()
                for stim in static_scene["images"]:
                    stim.draw()
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
                        "highlight_box": current_highlight_box,
                    }
                )
            core.wait(max(0.02, float(update_interval_ms) / 1000.0))
    finally:
        _release_movie()
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
        start_perf_s: Optional[float] = None,
        update_interval_s: float = 0.1,
    ):
        self.screen_info = screen_info
        self.task_label = task_label
        self.start_perf_s = time.perf_counter() if start_perf_s is None else float(start_perf_s)
        self.update_interval_s = max(0.05, float(update_interval_s))
        self.exit_requested = False
        self._ctx = mp.get_context("spawn")
        self._queue = self._ctx.Queue()
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

    def _send(self, payload: Dict[str, Any]) -> None:
        if self.poll():
            return
        try:
            self._queue.put_nowait(payload)
        except Exception:
            pass

    def show_static_scene(
        self,
        *,
        bg_rgb_255: Sequence[int],
        main_size: Sequence[int],
        images: Optional[list[Dict[str, Any]]] = None,
        dots: Optional[list[Dict[str, Any]]] = None,
        fixation_size: Optional[float] = None,
        fixation_color: Sequence[int] = (0, 0, 0),
        reward_counts: Any = _UNSET,
        highlight_box: Any = _UNSET,
    ) -> None:
        payload: Dict[str, Any] = {
            "type": "static_scene",
            "bg_rgb_255": list(bg_rgb_255),
            "main_size": [int(main_size[0]), int(main_size[1])],
            "images": list(images or []),
            "dots": list(dots or []),
            "fixation_size": fixation_size,
            "fixation_color": list(fixation_color),
        }
        if reward_counts is not _UNSET:
            payload["reward_counts"] = dict(reward_counts) if reward_counts is not None else None
        if highlight_box is not _UNSET:
            payload["highlight_box"] = dict(highlight_box) if highlight_box is not None else None
        self._send(payload)

    def clear_scene(
        self,
        *,
        bg_rgb_255: Sequence[int],
        main_size: Optional[Sequence[int]] = None,
        reward_counts: Any = _UNSET,
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
