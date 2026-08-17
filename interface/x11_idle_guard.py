"""X11 input and blackout lifecycle for the experiment manager."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from bin.screen import ScreenGeometry, set_tk_window_fullscreen


class XInputControlError(RuntimeError):
    """Raised when the configured touchscreen cannot be controlled safely."""


def mask_main_inputs_enabled(cfg: Mapping[str, Any]) -> bool:
    """Read the opt-in main-screen input masking flag."""
    value = cfg.get("mask_main_inputs", False)
    if not isinstance(value, bool):
        raise ValueError("Config field 'mask_main_inputs' must be true or false")
    return value


def should_manage_main_screen(
    main_screen: ScreenGeometry,
    experimenter_screen: Optional[ScreenGeometry],
    *,
    platform: Optional[str] = None,
    display: Optional[str] = None,
) -> bool:
    """Return whether this process should apply the dual-screen X11 policy."""
    active_platform = sys.platform if platform is None else str(platform)
    active_display = os.environ.get("DISPLAY") if display is None else display
    return bool(
        active_platform.startswith("linux")
        and active_display
        and experimenter_screen is not None
        and int(main_screen.index) != int(experimenter_screen.index)
    )


def configured_main_touchscreen(cfg: Mapping[str, Any], *, required: bool) -> Optional[str]:
    """Read the exact XInput main-touchscreen name from a launch config."""
    value = cfg.get("main_touchscreen_xinput")
    if value is None or not str(value).strip():
        if required:
            raise ValueError(
                "Config field 'main_touchscreen_xinput' must contain the exact "
                "XInput device name when 'mask_main_inputs' is true"
            )
        return None
    device_name = str(value).strip()
    if device_name.isdigit():
        raise ValueError(
            "Config field 'main_touchscreen_xinput' must be a stable device name, not a numeric XInput ID"
        )
    return device_name


@dataclass
class XInputTouchscreen:
    """Enable, disable, and output-map one exact XInput device."""

    device_name: str
    output_name: str
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run

    def __post_init__(self) -> None:
        self.device_name = str(self.device_name).strip()
        self.output_name = str(self.output_name).strip()
        if not self.device_name:
            raise ValueError("XInput touchscreen device name must not be empty")
        if not self.output_name:
            raise ValueError("Main display output name must not be empty")

    def _run(self, args: list[str]) -> subprocess.CompletedProcess:
        try:
            return self.runner(
                args,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except FileNotFoundError as exc:
            raise XInputControlError("The 'xinput' command is not installed") from exc
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or str(exc)).strip()
            raise XInputControlError(
                f"XInput command failed ({' '.join(args)}): {detail}"
            ) from exc

    def resolve_device_id(self) -> str:
        result = self._run(["xinput", "list", "--id-only", self.device_name])
        device_ids = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if len(device_ids) != 1 or not device_ids[0].isdigit():
            found = ", ".join(device_ids) if device_ids else "none"
            raise XInputControlError(
                f"Expected exactly one XInput device named {self.device_name!r}; found: {found}"
            )
        return device_ids[0]

    def validate(self) -> None:
        self.resolve_device_id()

    def disable(self) -> None:
        device_id = self.resolve_device_id()
        self._run(["xinput", "disable", device_id])

    def enable_for_task(self) -> None:
        device_id = self.resolve_device_id()
        # Map while disabled so no touch can land on the wrong output between commands.
        self._run(["xinput", "map-to-output", device_id, self.output_name])
        self._run(["xinput", "enable", device_id])

    def enable_for_desktop(self) -> None:
        device_id = self.resolve_device_id()
        self._run(["xinput", "enable", device_id])


class MainScreenCurtain:
    """Borderless black Tk window covering the main subject display."""

    def __init__(self, root, screen_info: ScreenGeometry, *, tk_module) -> None:
        self.root = root
        self.screen_info = screen_info
        self.window = tk_module.Toplevel(root)
        self.window.withdraw()
        # The curtain is also the launch-time X11 focus anchor.  Keeping focus
        # on the experimenter window until it is withdrawn leaves the window
        # manager's active monitor undefined and can send a new managed
        # fullscreen window to either output.
        self.window.configure(bg="black", cursor="none", takefocus=True)
        set_tk_window_fullscreen(self.window, self.screen_info)
        try:
            self.window.attributes("-topmost", True)
        except Exception:
            pass

    def show(self) -> None:
        self.window.deiconify()
        self.window.lift()
        self.window.update_idletasks()

    def hide(self) -> None:
        self.window.withdraw()
        self.window.update_idletasks()

    def focus_for_task_launch(self, *, timeout_s: float = 1.0) -> None:
        """Make the subject output the verified active X11 display."""
        self.show()
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        while True:
            self.window.focus_force()
            # Unlike update_idletasks(), update() processes the FocusIn event
            # before the task subprocess is allowed to create its window.
            self.window.update()
            focused = self.window.focus_displayof()
            if focused is self.window or (
                focused is not None and str(focused) == str(self.window)
            ):
                return
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "Could not focus the main-screen launch curtain; refusing "
                    "to create a fullscreen task window with ambiguous placement"
                )
            time.sleep(0.01)

    def close(self) -> None:
        try:
            self.window.destroy()
        except Exception:
            pass


class ExperimentIdleGuard:
    """Coordinate the main touchscreen and blackout window around tasks."""

    def __init__(self, root, touchscreen: XInputTouchscreen, curtain: MainScreenCurtain):
        self.root = root
        self.touchscreen = touchscreen
        self.curtain = curtain
        self.released = False
        self.idle = False

    def restore_interface_focus(self) -> None:
        try:
            self.root.deiconify()
            self.root.lift()
            self.root.focus_force()
            self.root.update_idletasks()
        except Exception:
            pass

    def enter_idle(self) -> None:
        if self.released:
            return
        if not self.idle:
            try:
                self.touchscreen.disable()
                self.curtain.show()
            except Exception:
                try:
                    self.touchscreen.enable_for_desktop()
                except Exception:
                    pass
                raise
            self.idle = True
        else:
            self.curtain.show()
        self.restore_interface_focus()

    def prepare_task_launch(self) -> None:
        """Transfer focus to the covered subject output before child startup."""
        if self.released:
            raise RuntimeError("The main-screen guard has already been released")
        if not self.idle:
            raise RuntimeError("The main-screen guard is not in its covered idle state")
        self.curtain.focus_for_task_launch()

    def task_window_ready(self) -> None:
        if self.released:
            return
        self.curtain.hide()
        try:
            self.touchscreen.enable_for_task()
        except Exception:
            self.curtain.show()
            self.restore_interface_focus()
            raise
        self.idle = False

    def release_for_desktop(self) -> None:
        if self.released:
            return
        self.touchscreen.enable_for_desktop()
        self.curtain.close()
        self.released = True
        self.idle = False


def create_experiment_idle_guard(
    root,
    cfg: Mapping[str, Any],
    main_screen: ScreenGeometry,
    experimenter_screen: Optional[ScreenGeometry],
    *,
    tk_module,
) -> Optional[ExperimentIdleGuard]:
    """Build the configured guard, or return ``None`` when masking is off/inapplicable."""
    if not mask_main_inputs_enabled(cfg):
        return None

    # A single/same-screen setup is the explicit debugging exception: masking
    # it would disable the interface's own touch input and cover its window.
    if (
        experimenter_screen is None
        or int(main_screen.index) == int(experimenter_screen.index)
    ):
        return None

    if not should_manage_main_screen(main_screen, experimenter_screen):
        if not sys.platform.startswith("linux"):
            raise ValueError("'mask_main_inputs' requires Linux/X11")
        raise ValueError("'mask_main_inputs' requires an active X11 DISPLAY")
    if os.environ.get("XDG_SESSION_TYPE", "").strip().lower() == "wayland":
        raise ValueError("'mask_main_inputs' requires an X11 session, not Wayland/Xwayland")

    device_name = configured_main_touchscreen(cfg, required=True)
    output_name = str(main_screen.name or "").strip()
    if not output_name:
        raise ValueError(
            "Could not determine the X11 output name for the configured main screen"
        )

    touchscreen = XInputTouchscreen(
        device_name=device_name or "",
        output_name=output_name,
    )
    touchscreen.validate()
    curtain = MainScreenCurtain(root, main_screen, tk_module=tk_module)
    return ExperimentIdleGuard(root, touchscreen, curtain)


def wait_for_task_process(
    process: subprocess.Popen,
    *,
    ready_path: Optional[Path] = None,
    release_path: Optional[Path] = None,
    on_window_ready: Optional[Callable[[], None]] = None,
    poll_interval_s: float = 0.025,
) -> int:
    """Wait for a task, releasing guarded input once its main window is ready."""
    window_released = ready_path is None or on_window_ready is None
    while True:
        if window_released:
            return int(process.wait())
        if (
            not window_released
            and ready_path is not None
            and ready_path.is_file()
        ):
            on_window_ready()
            if release_path is not None:
                release_path.write_text("released\n", encoding="utf-8")
            window_released = True
            continue
        try:
            return int(process.wait(timeout=max(0.01, float(poll_interval_s))))
        except subprocess.TimeoutExpired:
            continue


def stop_task_process(process: subprocess.Popen, *, timeout_s: float = 5.0) -> None:
    """Stop a child that cannot continue because guarded input setup failed."""
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=max(0.1, float(timeout_s)))
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
