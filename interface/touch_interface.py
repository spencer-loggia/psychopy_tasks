#!/usr/bin/env python3
"""
Touch-friendly Tk launcher for experiment tasks and simple utilities.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional
import tkinter as tk
from tkinter import messagebox
import urllib.request
from email.utils import parsedate_to_datetime

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from bin.screen import (
    MAIN_SCREEN_ENV,
    SECONDARY_SCREEN_ENV,
    load_screen_config,
    resolve_interface_screen,
    resolve_task_screens,
    set_tk_window_fullscreen,
)
from interface.rig_mode import (
    IS_RIG_ENV_VAR,
    PORTABLE_MODE_VALUE,
    SWITCH_TO_PORTABLE_SCRIPT,
    mode_button_label,
    mode_command_for_target_mode,
    mode_script_for_target_mode,
    normalize_is_rig,
    target_mode_for_current_mode,
)
from interface.experiment_manager import (
    ExperimentManager,
    PreparedBlock,
    task_run_sequence,
)
from interface.experiment_quiet_mode import create_experiment_quiet_mode
from bin.task_lifecycle import TASK_WINDOW_READY_ENV, USER_EXIT_CODE
from interface.x11_idle_guard import (
    ExperimentIdleGuard,
    create_experiment_idle_guard,
    stop_task_process,
    wait_for_task_process,
)


IDLE_CLEANUP_MS = 30 * 60 * 1000
BUTTON_BG = "#f7f7f7"
BUTTON_ACTIVE_BG = "#d9d9d9"
SHUTDOWN_BUTTON_BG = "#b91c1c"
SHUTDOWN_BUTTON_ACTIVE_BG = "#7f1d1d"
TOUCH_SCROLL_THRESHOLD_PX = 12
WHEEL_SCROLL_UNITS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Touch launcher for neuro_tasks")
    parser.add_argument("--config", required=True, help="Path to launcher config JSON")
    parser.add_argument("--main_screen", default=None, help="Main task screen index or output name")
    parser.add_argument("--experimenter_screen", default=None, help="Experimenter screen index or output name")
    return parser.parse_args()


def _expect_dict(value: Any, name: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Config field '{name}' must be a JSON object")
    return value


def load_launcher_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as config_fh:
        cfg = json.load(config_fh)

    if not isinstance(cfg, dict):
        raise ValueError("Launcher config must contain a top-level JSON object")
    return cfg


def format_diagnostic_report(result: Mapping[str, Any]) -> str:
    """Format a complete, touch-dialog-friendly system diagnostic report."""
    success = bool(result.get("success", False))
    refresh_rate = result.get("refresh_rate_hz")
    try:
        refresh_text = f"{float(refresh_rate):.3f} Hz" if refresh_rate is not None else "unavailable"
    except (TypeError, ValueError):
        refresh_text = "unavailable"

    lines = [
        "SYSTEM DIAGNOSTIC PASSED" if success else "SYSTEM DIAGNOSTIC FOUND ERRORS",
        f"Main monitor refresh rate: {refresh_text}",
        "",
        "Checks:",
    ]
    checks = result.get("checks", [])
    if not isinstance(checks, list):
        checks = []
    failures = []
    for check in checks:
        if not isinstance(check, dict):
            continue
        status = str(check.get("status", "fail")).strip().lower()
        label = {"pass": "PASS", "fail": "FAIL", "skip": "SKIP"}.get(
            status,
            status.upper() or "FAIL",
        )
        name = str(check.get("name", "Unnamed check"))
        detail = str(check.get("detail", "")).strip()
        lines.append(f"[{label}] {name}: {detail}" if detail else f"[{label}] {name}")
        if status == "fail":
            error = str(check.get("error", "")).strip() or detail or "Unknown error"
            failures.append(f"- {name}: {error}")

    if not checks:
        failures.append("- Diagnostic runner: no check results were produced")
    if failures:
        lines.extend(("", "Errors:", *failures))
    return "\n".join(lines)


def _resolve_candidate(path_value: str, search_roots: Iterable[Path]) -> Path:
    raw_path = Path(path_value).expanduser()
    if raw_path.is_absolute():
        return raw_path

    for root in search_roots:
        candidate = (root / raw_path).resolve()
        if candidate.exists():
            return candidate

    first_root = next(iter(search_roots), Path.cwd())
    return (first_root / raw_path).resolve()


def _get_working_directory(environment_cfg: Dict[str, Any], config_dir: Path) -> Path:
    working_dir_value = environment_cfg.get("working_directory", environment_cfg.get("working_dir"))
    if not working_dir_value:
        raise KeyError("Config must define environment.working_directory or environment.working_dir")
    working_dir = _resolve_candidate(str(working_dir_value), (config_dir, Path.cwd()))
    if not working_dir.exists() or not working_dir.is_dir():
        raise ValueError(f"Working directory does not exist or is not a directory: {working_dir}")
    return working_dir


def wheel_scroll_units(*, delta: int = 0, button_num: Optional[int] = None) -> int:
    """Normalize Windows/macOS wheel deltas and X11 Button-4/5 events."""
    if button_num == 4:
        return -WHEEL_SCROLL_UNITS
    if button_num == 5:
        return WHEEL_SCROLL_UNITS
    delta = int(delta)
    if delta == 0:
        return 0
    notches = max(1, int(round(abs(delta) / 120.0)))
    direction = -1 if delta > 0 else 1
    return direction * notches * WHEEL_SCROLL_UNITS


def touch_drag_exceeds_threshold(start_y: float, current_y: float) -> bool:
    return abs(float(current_y) - float(start_y)) >= TOUCH_SCROLL_THRESHOLD_PX


def touch_drag_scroll_fraction(
    *,
    initial_first: float,
    start_y: float,
    current_y: float,
    content_height: float,
    visible_fraction: float,
) -> float:
    """Translate a direct-manipulation vertical drag into a canvas yview value."""
    if content_height <= 0.0 or visible_fraction >= 1.0:
        return 0.0
    max_first = max(0.0, 1.0 - float(visible_fraction))
    requested = float(initial_first) + (
        (float(start_y) - float(current_y)) / float(content_height)
    )
    return min(max(requested, 0.0), max_first)


class ScrollableButtonFrame(tk.Frame):
    def __init__(self, master: tk.Misc, **kwargs: Any):
        super().__init__(master, **kwargs)
        self.canvas = tk.Canvas(self, highlightthickness=0, yscrollincrement=20)
        self.scrollbar = tk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = tk.Frame(self.canvas)
        self._touch_start_y: Optional[float] = None
        self._touch_initial_first = 0.0
        self._touch_moved = False
        self._pressed_button: Optional[tk.Button] = None
        self._pressed_button_relief: Optional[str] = None

        self.inner.bind("<Configure>", self._on_inner_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas_window = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        self._bind_scroll_surface(self.canvas)
        self._bind_scroll_surface(self.inner)

    def _on_inner_configure(self, _event: tk.Event) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event: tk.Event) -> None:
        self.canvas.itemconfigure(self.canvas_window, width=event.width)

    def _bind_wheel(self, widget: tk.Misc) -> None:
        widget.bind("<MouseWheel>", self._on_mousewheel, add="+")
        widget.bind("<Button-4>", self._on_mousewheel, add="+")
        widget.bind("<Button-5>", self._on_mousewheel, add="+")

    def _bind_scroll_surface(self, widget: tk.Misc) -> None:
        self._bind_wheel(widget)
        widget.bind("<ButtonPress-1>", self._on_touch_press, add="+")
        widget.bind("<B1-Motion>", self._on_touch_motion, add="+")
        widget.bind("<ButtonRelease-1>", self._on_touch_release, add="+")

    def register_button(self, button: tk.Button) -> None:
        """Give a task button tap-to-activate and drag-to-scroll behavior."""
        self._bind_wheel(button)
        button.bind(
            "<ButtonPress-1>",
            lambda event, target=button: self._on_touch_press(event, target),
            add="+",
        )
        button.bind("<B1-Motion>", self._on_touch_motion, add="+")
        button.bind("<ButtonRelease-1>", self._on_touch_release, add="+")

    def reset_scroll(self) -> None:
        self._reset_touch_gesture()
        self.after_idle(lambda: self.canvas.yview_moveto(0.0))

    def _on_mousewheel(self, event: tk.Event) -> str:
        units = wheel_scroll_units(
            delta=getattr(event, "delta", 0),
            button_num=getattr(event, "num", None),
        )
        if units:
            self.canvas.yview_scroll(units, "units")
        return "break"

    def _on_touch_press(
        self,
        event: tk.Event,
        button: Optional[tk.Button] = None,
    ) -> str:
        self._reset_touch_gesture()
        self._touch_start_y = float(event.y_root)
        self._touch_initial_first = float(self.canvas.yview()[0])
        self._pressed_button = button
        if button is not None and str(button.cget("state")) != "disabled":
            self._pressed_button_relief = str(button.cget("relief"))
            button.configure(relief="sunken")
        return "break"

    def _on_touch_motion(self, event: tk.Event) -> str:
        if self._touch_start_y is None:
            return "break"
        current_y = float(event.y_root)
        if not self._touch_moved:
            self._touch_moved = touch_drag_exceeds_threshold(
                self._touch_start_y,
                current_y,
            )
            if self._touch_moved:
                self._restore_pressed_button_relief()
        if not self._touch_moved:
            return "break"

        bbox = self.canvas.bbox("all")
        content_height = float(bbox[3] - bbox[1]) if bbox is not None else 0.0
        first, last = self.canvas.yview()
        visible_fraction = float(last) - float(first)
        self.canvas.yview_moveto(
            touch_drag_scroll_fraction(
                initial_first=self._touch_initial_first,
                start_y=self._touch_start_y,
                current_y=current_y,
                content_height=content_height,
                visible_fraction=visible_fraction,
            )
        )
        return "break"

    def _on_touch_release(self, event: tk.Event) -> str:
        button = self._pressed_button
        should_invoke = bool(
            button is not None
            and not self._touch_moved
            and str(button.cget("state")) != "disabled"
            and self._event_is_inside_button(event, button)
        )
        self._reset_touch_gesture()
        if should_invoke and button is not None:
            button.invoke()
        return "break"

    @staticmethod
    def _event_is_inside_button(event: tk.Event, button: tk.Button) -> bool:
        x = float(event.x_root) - float(button.winfo_rootx())
        y = float(event.y_root) - float(button.winfo_rooty())
        return 0.0 <= x < float(button.winfo_width()) and 0.0 <= y < float(
            button.winfo_height()
        )

    def _restore_pressed_button_relief(self) -> None:
        if self._pressed_button is None or self._pressed_button_relief is None:
            return
        try:
            self._pressed_button.configure(relief=self._pressed_button_relief)
        except tk.TclError:
            pass
        self._pressed_button_relief = None

    def _reset_touch_gesture(self) -> None:
        self._restore_pressed_button_relief()
        self._touch_start_y = None
        self._touch_initial_first = 0.0
        self._touch_moved = False
        self._pressed_button = None


class TouchInterfaceApp:
    def __init__(
        self,
        root: tk.Tk,
        config_path: Path,
        cfg: Dict[str, Any],
        *,
        screen_info,
        idle_guard: Optional[ExperimentIdleGuard] = None,
    ):
        self.root = root
        self.config_path = config_path.resolve()
        self.config_dir = self.config_path.parent
        self.cfg = cfg
        self.screen_info = screen_info
        self.idle_guard = idle_guard
        self.environment_cfg = _expect_dict(cfg.get("environment"), "environment")
        self.tasks_cfg = _expect_dict(cfg.get("tasks"), "tasks")
        self.subjects_cfg = _expect_dict(cfg.get("subjects"), "subjects")
        if not self.subjects_cfg:
            raise ValueError("Config field 'subjects' must include at least one subject")
        for subject_name, subject_code in self.subjects_cfg.items():
            if not str(subject_name).strip() or not str(subject_code).strip():
                raise ValueError("Subject names and codes must be non-empty strings")
        self.working_dir = _get_working_directory(self.environment_cfg, self.config_dir)
        self.python_cmd = str(self.environment_cfg.get("python", "")).strip()
        if not self.python_cmd:
            raise KeyError("Config must define environment.python")

        self.task_active = False
        self.status_var = tk.StringVar(value="Ready")
        self.page_title_var = tk.StringVar(value="Task Launcher")
        self.page_stack: list[tuple[str, Dict[str, Any]]] = []
        self.experiment: Optional[ExperimentManager] = None
        self.quiet_mode = create_experiment_quiet_mode(cfg)
        self.is_rig = self._initialize_is_rig_mode()

        if self.idle_guard is not None:
            self.idle_guard.enter_idle()
        self.startup()
        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._exit_to_desktop)
        self._schedule_idle_cleanup()

    def _is_launchable_task(self, task_cfg: Dict[str, Any]) -> bool:
        return "launch" in task_cfg

    def _enter_page(self, page_name: str, page_cfg: Dict[str, Any]) -> None:
        self.page_stack.append((page_name, page_cfg))
        self._render_current_page()

    def _go_back(self) -> None:
        if len(self.page_stack) <= 1:
            return
        self.page_stack.pop()
        self._render_current_page()

    def _current_page(self) -> tuple[str, Dict[str, Any]]:
        return self.page_stack[-1]

    def _current_page_label(self) -> str:
        labels = [label for label, _page in self.page_stack]
        return " / ".join(labels)
    
    def attempt_rectify_timezone(self) -> None:
        try:
            with urllib.request.urlopen("https://www.google.com", timeout=5) as r:
                dt = parsedate_to_datetime(r.headers["Date"])  # UTC from HTTP Date header

            # Force system clock to that UTC time
            subprocess.run(
                ["sudo", "date", "-u", "-s", dt.strftime("%Y-%m-%d %H:%M:%S")],
                check=True,
            )

        except Exception as e:
            print(f"Could not sync time: {e}")

    def pull_latest_code(self) -> None:
        try:
            subprocess.run(
                ["git", "pull"],
                cwd=self.working_dir,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except Exception as e:
            print(f"Could not pull latest code: {e}")

    def startup(self) -> None:
        os.chdir(self.working_dir)
        # attempt to rectify system timezone
        self.attempt_rectify_timezone()
        self.pull_latest_code()

    def cleanup(self) -> None:
        quiet_mode = getattr(self, "quiet_mode", None)
        if self.experiment is not None or (
            quiet_mode is not None and quiet_mode.active
        ):
            return
        self.attempt_rectify_timezone()
        self.pull_latest_code()

    def _initialize_is_rig_mode(self) -> str:
        raw_mode = os.environ.get(IS_RIG_ENV_VAR)
        current_mode = normalize_is_rig(raw_mode)
        if current_mode is not None:
            return current_mode

        mode_problem = "not set" if raw_mode is None else f"set to invalid value {raw_mode!r}"
        self._run_mode_script(
            PORTABLE_MODE_VALUE,
            missing_message=(
                f"{IS_RIG_ENV_VAR} is {mode_problem}, and {SWITCH_TO_PORTABLE_SCRIPT} does not exist."
            ),
            failure_message=(
                f"Could not initialize {IS_RIG_ENV_VAR} from {mode_problem} with {SWITCH_TO_PORTABLE_SCRIPT}."
            ),
        )
        return normalize_is_rig(os.environ.get(IS_RIG_ENV_VAR)) or PORTABLE_MODE_VALUE

    def _run_mode_script(
        self,
        target_mode: str,
        *,
        missing_message: Optional[str] = None,
        failure_message: Optional[str] = None,
    ) -> bool:
        script_path = mode_script_for_target_mode(target_mode)
        if not script_path.exists():
            self.status_var.set("Mode switch unavailable")
            messagebox.showwarning(
                "Mode Switch Warning",
                missing_message or f"Script does not exist: {script_path}",
            )
            return False

        try:
            subprocess.run(
                mode_command_for_target_mode(target_mode),
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or str(exc)).strip()
            self.status_var.set("Mode switch failed")
            messagebox.showwarning(
                "Mode Switch Warning",
                f"{failure_message or 'Mode switch script failed.'}\n\n{detail}",
            )
            return False
        except Exception as exc:
            self.status_var.set("Mode switch failed")
            messagebox.showwarning(
                "Mode Switch Warning",
                f"{failure_message or 'Mode switch script failed.'}\n\n{exc}",
            )
            return False

        os.environ[IS_RIG_ENV_VAR] = target_mode
        self.is_rig = target_mode
        self.status_var.set(f"{IS_RIG_ENV_VAR}={target_mode}")
        return True

    def _switch_rig_mode(self) -> None:
        if self.task_active:
            self.status_var.set("Cannot switch modes while a task is running")
            return

        target_mode = target_mode_for_current_mode(self.is_rig)
        if self._run_mode_script(target_mode):
            self._render_root_menu()

    def _schedule_idle_cleanup(self) -> None:
        self.root.after(IDLE_CLEANUP_MS, self._run_idle_cleanup_if_needed)

    def _run_idle_cleanup_if_needed(self) -> None:
        if not self.task_active and self.experiment is None:
            self.cleanup()
        self._schedule_idle_cleanup()

    def _build_ui(self) -> None:
        self.root.title("Experiment Manager")
        set_tk_window_fullscreen(self.root, self.screen_info)

        self.root.configure(bg="#e9ecef")
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        header = tk.Label(
            self.root,
            textvariable=self.page_title_var,
            font=("Helvetica", 24, "bold"),
            bg="#e9ecef",
            anchor="w",
            padx=24,
            pady=16,
        )
        header.grid(row=0, column=0, sticky="ew")

        self.button_frame = ScrollableButtonFrame(self.root, bg="#e9ecef")
        self.button_frame.grid(row=1, column=0, sticky="nsew", padx=18, pady=(0, 12))
        self.button_frame.inner.configure(bg="#e9ecef")
        self.button_container = self.button_frame.inner

        self._render_root_menu()

        footer = tk.Label(
            self.root,
            textvariable=self.status_var,
            font=("Helvetica", 16),
            bg="#e9ecef",
            anchor="w",
            padx=24,
            pady=12,
        )
        footer.grid(row=2, column=0, sticky="ew")

    def _button_kwargs(self) -> Dict[str, Any]:
        return {
            "font": ("Helvetica", 22, "bold"),
            "height": 3,
            "padx": 16,
            "pady": 12,
            "wraplength": 900,
            "bg": BUTTON_BG,
            "activebackground": BUTTON_ACTIVE_BG,
            "relief": "raised",
            "overrelief": "raised",
            "bd": 2,
        }

    def _clear_buttons(self) -> None:
        self.button_frame.reset_scroll()
        for child in self.button_container.winfo_children():
            child.destroy()

    def _place_button(self, button: tk.Button, row_idx: int) -> None:
        self.button_frame.register_button(button)
        button.grid(row=row_idx, column=0, sticky="ew", pady=10, padx=10)
        self.button_container.grid_columnconfigure(0, weight=1)

    def _render_root_menu(self) -> None:
        self._clear_buttons()
        self.page_stack = []
        self.page_title_var.set("Experiment Manager")
        self.status_var.set("Choose an action")

        self._create_start_experiment_button(0)
        self._create_diagnostic_button(1)
        self._create_rig_mode_button(2)
        self._create_desktop_button(3)
        self._create_shutdown_button(4)

    def _create_start_experiment_button(self, row_idx: int) -> None:
        button = tk.Button(
            self.button_container,
            text="Start Experiment",
            command=self._render_subject_selection,
            **self._button_kwargs(),
        )
        self._place_button(button, row_idx)

    def _create_diagnostic_button(self, row_idx: int) -> None:
        button = tk.Button(
            self.button_container,
            text="Run System Diagnostic",
            command=self._run_system_diagnostic,
            **self._button_kwargs(),
        )
        self._place_button(button, row_idx)

    def _run_system_diagnostic(self) -> None:
        if self.task_active:
            self.status_var.set("Cannot run diagnostic while a task is running")
            return

        diagnostic_path = self.working_dir / "task" / "system_diagnostic.py"
        if not diagnostic_path.is_file():
            self.status_var.set("Diagnostic unavailable")
            messagebox.showerror(
                "System Diagnostic",
                f"Diagnostic script not found: {diagnostic_path}",
            )
            return

        self.task_active = True
        self.status_var.set("Running system diagnostic on the main monitor...")
        self.root.update_idletasks()
        result: Dict[str, Any]
        try:
            with tempfile.TemporaryDirectory(prefix="neuro_tasks_diagnostic_") as temp_dir:
                temp_path = Path(temp_dir)
                result_path = temp_path / "result.json"
                ready_path = temp_path / ".task_window_ready"
                cmd = [
                    self.python_cmd,
                    str(diagnostic_path),
                    "--config",
                    str(self.config_path),
                    "--result",
                    str(result_path),
                ]
                env = os.environ.copy()
                if self.idle_guard is not None:
                    env[TASK_WINDOW_READY_ENV] = str(ready_path)

                process: Optional[subprocess.Popen] = None
                try:
                    process = subprocess.Popen(cmd, cwd=self.working_dir, env=env)
                    try:
                        returncode = wait_for_task_process(
                            process,
                            ready_path=(
                                ready_path if self.idle_guard is not None else None
                            ),
                            on_window_ready=(
                                self.idle_guard.task_window_ready
                                if self.idle_guard is not None
                                else None
                            ),
                        )
                    except BaseException:
                        stop_task_process(process)
                        raise
                finally:
                    if self.idle_guard is not None:
                        self.idle_guard.enter_idle()

                if not result_path.is_file():
                    raise RuntimeError(
                        f"Diagnostic exited with status {returncode} without producing a result"
                    )
                loaded_result = json.loads(result_path.read_text(encoding="utf-8"))
                if not isinstance(loaded_result, dict):
                    raise ValueError("Diagnostic result must contain a JSON object")
                result = loaded_result
        except Exception as exc:
            result = {
                "success": False,
                "refresh_rate_hz": None,
                "checks": [
                    {
                        "name": "Diagnostic runner",
                        "status": "fail",
                        "detail": "The system diagnostic could not be launched or read",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                ],
            }
        finally:
            self.task_active = False

        refresh_rate = result.get("refresh_rate_hz")
        try:
            refresh_status = (
                f"{float(refresh_rate):.3f} Hz"
                if refresh_rate is not None
                else "refresh unavailable"
            )
        except (TypeError, ValueError):
            refresh_status = "refresh unavailable"
        report = format_diagnostic_report(result)
        if bool(result.get("success", False)):
            self.status_var.set(f"Diagnostic passed — {refresh_status}")
            messagebox.showinfo("System Diagnostic", report)
        else:
            self.status_var.set(f"Diagnostic found errors — {refresh_status}")
            messagebox.showerror("System Diagnostic", report)

    def _render_subject_selection(self) -> None:
        self._clear_buttons()
        self.page_title_var.set("Select Subject")
        self.status_var.set("Choose the subject to start a new experiment")
        for row_idx, (subject_name, subject_code) in enumerate(self.subjects_cfg.items()):
            button = tk.Button(
                self.button_container,
                text=str(subject_name),
                command=lambda n=str(subject_name), c=str(subject_code): self._select_subject(n, c),
                **self._button_kwargs(),
            )
            self._place_button(button, row_idx)
        self._create_root_menu_button(len(self.subjects_cfg))

    def _select_subject(self, subject_name: str, subject_code: str) -> None:
        if self.experiment is not None:
            return
        quiet_mode = getattr(self, "quiet_mode", None)
        try:
            if quiet_mode is not None:
                quiet_mode.enter()
            self.experiment = ExperimentManager(
                working_dir=self.working_dir,
                launch_config_path=self.config_path,
                launch_config=self.cfg,
                subject_name=subject_name,
                subject_code=subject_code,
            )
        except Exception as exc:
            if quiet_mode is not None:
                try:
                    quiet_mode.exit()
                except Exception:
                    pass
            messagebox.showerror("Experiment Error", str(exc))
            self.status_var.set("Could not start experiment")
            return
        self.page_stack = [("Tasks", self.tasks_cfg)]
        self.status_var.set(
            f"Subject: {subject_name} — {self.experiment.experiment_dir.name}"
        )
        self._render_current_page()

    def _render_current_page(self) -> None:
        self._clear_buttons()

        page_name, page_cfg = self._current_page()
        self.page_title_var.set(self._current_page_label())

        row_idx = 0
        for task_name, task_cfg in page_cfg.items():
            self._create_task_button(row_idx, task_name, task_cfg)
            row_idx += 1

        if len(self.page_stack) == 1:
            self._create_end_experiment_button(row_idx)
        else:
            self._create_back_button(row_idx)

    def _create_task_button(self, row_idx: int, task_name: str, task_cfg: Any) -> None:
        if not isinstance(task_cfg, dict):
            raise ValueError(f"Task '{task_name}' must be a JSON object")

        if self._is_launchable_task(task_cfg):
            command = lambda n=task_name, c=task_cfg: self._run_task(n, c)
        else:
            command = lambda n=task_name, c=task_cfg: self._enter_page(n, c)

        button = tk.Button(
            self.button_container,
            text=task_name,
            command=command,
            **self._button_kwargs(),
        )
        self._place_button(button, row_idx)

    def _create_rig_mode_button(self, row_idx: int) -> None:
        button = tk.Button(
            self.button_container,
            text=mode_button_label(self.is_rig),
            command=self._switch_rig_mode,
            **self._button_kwargs(),
        )
        self._place_button(button, row_idx)

    def _create_desktop_button(self, row_idx: int) -> None:
        button = tk.Button(
            self.button_container,
            text="Desktop",
            command=self._exit_to_desktop,
            **self._button_kwargs(),
        )
        self._place_button(button, row_idx)

    def _exit_to_desktop(self) -> None:
        if self.task_active:
            self.status_var.set("Cannot exit to desktop while a task is running")
            return
        quiet_mode = getattr(self, "quiet_mode", None)
        try:
            if quiet_mode is not None:
                quiet_mode.exit()
        except Exception as exc:
            self.status_var.set("Could not restore quiet mode")
            messagebox.showerror("Desktop Error", str(exc))
            return
        try:
            if self.idle_guard is not None:
                self.idle_guard.release_for_desktop()
        except Exception as exc:
            self.status_var.set("Could not restore main touchscreen")
            messagebox.showerror("Desktop Error", str(exc))
            return
        self.root.destroy()

    def _create_shutdown_button(self, row_idx: int) -> None:
        button_kwargs = self._button_kwargs()
        button_kwargs.update(
            {
                "bg": SHUTDOWN_BUTTON_BG,
                "fg": "white",
                "activebackground": SHUTDOWN_BUTTON_ACTIVE_BG,
                "activeforeground": "white",
            }
        )
        button = tk.Button(
            self.button_container,
            text="Shutdown",
            command=self._shutdown_system,
            **button_kwargs,
        )
        self._place_button(button, row_idx)

    def _create_back_button(self, row_idx: int) -> None:
        button = tk.Button(
            self.button_container,
            text="Back",
            command=self._go_back,
            **self._button_kwargs(),
        )
        self._place_button(button, row_idx)

    def _create_root_menu_button(self, row_idx: int) -> None:
        button = tk.Button(
            self.button_container,
            text="Back",
            command=self._render_root_menu,
            **self._button_kwargs(),
        )
        self._place_button(button, row_idx)

    def _create_end_experiment_button(self, row_idx: int) -> None:
        button = tk.Button(
            self.button_container,
            text="End Experiment",
            command=self._end_experiment,
            **self._button_kwargs(),
        )
        self._place_button(button, row_idx)

    def _end_experiment(self) -> None:
        if self.task_active:
            self.status_var.set("Cannot end experiment while a task is running")
            return
        quiet_mode = getattr(self, "quiet_mode", None)
        try:
            if quiet_mode is not None:
                quiet_mode.exit()
        except Exception as exc:
            messagebox.showerror("Experiment Cleanup Error", str(exc))
            self.status_var.set("Could not restore quiet mode")
            return
        self.experiment = None
        self._render_root_menu()

    def _shutdown_command(self) -> list[str]:
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            return ["shutdown", "-h", "now"]
        return ["sudo", "-n", "shutdown", "-h", "now"]

    def _shutdown_system(self) -> None:
        if self.task_active:
            self.status_var.set("Cannot shut down while a task is running")
            return

        self.status_var.set("Cleaning up before shutdown...")
        self.root.update_idletasks()

        try:
            self.cleanup()
            subprocess.run(
                self._shutdown_command(),
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or str(exc)).strip()
            self.status_var.set("Shutdown failed")
            messagebox.showerror("Shutdown Error", detail)
            return
        except Exception as exc:
            self.status_var.set("Shutdown failed")
            messagebox.showerror("Shutdown Error", str(exc))
            return

        self.status_var.set("Shutdown requested")
        self.root.after(1000, self.root.destroy)

    def _run_block(self, block: PreparedBlock) -> subprocess.CompletedProcess:
        if self.experiment is None:
            raise RuntimeError("Select a subject before launching a task")
        cmd = [self.python_cmd, str(block.launch_path), "--config", str(block.config_path)]
        self.status_var.set(f"Running block {block.block_num}: {block.block_name}")
        self.root.update_idletasks()
        self.root.withdraw()
        ready_path = block.output_dir / ".task_window_ready"
        env = self.experiment.subprocess_environment(block)
        if self.idle_guard is not None:
            env[TASK_WINDOW_READY_ENV] = str(ready_path)
        process: Optional[subprocess.Popen] = None
        try:
            process = subprocess.Popen(
                cmd,
                cwd=self.working_dir,
                env=env,
            )
            try:
                returncode = wait_for_task_process(
                    process,
                    ready_path=ready_path if self.idle_guard is not None else None,
                    on_window_ready=(
                        self.idle_guard.task_window_ready
                        if self.idle_guard is not None
                        else None
                    ),
                )
            except BaseException:
                stop_task_process(process)
                raise
            return subprocess.CompletedProcess(cmd, returncode)
        finally:
            try:
                if self.idle_guard is not None:
                    self.idle_guard.enter_idle()
                else:
                    self.root.deiconify()
                    self.root.lift()
            finally:
                try:
                    ready_path.unlink(missing_ok=True)
                finally:
                    self.experiment.finish_block(block)

    def _run_task(self, task_name: str, task_cfg: Dict[str, Any]) -> None:
        if self.task_active:
            return
        if self.experiment is None:
            messagebox.showerror("Launch Error", "Select a subject before launching a task")
            return

        self.task_active = True
        blocks_run = 0
        try:
            for launch_value, config_value in task_run_sequence(task_name, task_cfg):
                block = self.experiment.prepare_block(
                    task_name=task_name,
                    launch_value=launch_value,
                    config_value=config_value,
                )
                result = self._run_block(block)
                blocks_run += 1
                if result.returncode == USER_EXIT_CODE:
                    self.status_var.set(
                        f"Stopped: {task_name} after {blocks_run} block(s)"
                    )
                    break
                if result.returncode != 0:
                    self.status_var.set(
                        f"Failed: {block.block_name} (exit {result.returncode})"
                    )
                    messagebox.showwarning(
                        "Task Finished",
                        f"Block '{block.block_name}' exited with status {result.returncode}; its loop was stopped.",
                    )
                    break
            else:
                self.status_var.set(f"Finished: {task_name} ({blocks_run} block(s))")
        except Exception as exc:
            messagebox.showerror("Launch Error", str(exc))
            self.status_var.set(f"Launch failed: {task_name}")
        finally:
            self.task_active = False


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    cfg = load_launcher_config(config_path)

    root = tk.Tk()
    idle_guard = None
    app = None
    try:
        screen_cfg = load_screen_config(
            cfg,
            cli_main=args.main_screen,
            cli_experimenter=args.experimenter_screen,
        )
        main_screen, experimenter_screen = resolve_task_screens(
            screen_cfg,
            allow_same_screen=True,
        )
        screen_info = resolve_interface_screen(root, screen_cfg)
        if screen_cfg["main"] is not None:
            os.environ[MAIN_SCREEN_ENV] = str(screen_cfg["main"])
        if screen_cfg["experimenter"] is not None:
            os.environ[SECONDARY_SCREEN_ENV] = str(screen_cfg["experimenter"])
        idle_guard = create_experiment_idle_guard(
            root,
            cfg,
            main_screen,
            experimenter_screen,
            tk_module=tk,
        )
        app = TouchInterfaceApp(
            root,
            config_path,
            cfg,
            screen_info=screen_info,
            idle_guard=idle_guard,
        )
        root.mainloop()
    finally:
        if app is not None and app.quiet_mode is not None:
            try:
                app.quiet_mode.exit()
            except Exception as exc:
                print(f"Could not restore quiet-mode services: {exc}", file=sys.stderr)
        if idle_guard is not None:
            try:
                idle_guard.release_for_desktop()
            except Exception as exc:
                print(f"Could not restore main touchscreen: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
