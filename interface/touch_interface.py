#!/usr/bin/env python3
"""
Touch-friendly Tk launcher for experiment tasks and simple utilities.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
import tkinter as tk
from tkinter import messagebox
import urllib.request, subprocess
from email.utils import parsedate_to_datetime


IDLE_CLEANUP_MS = 30 * 60 * 1000
BUTTON_BG = "#f7f7f7"
BUTTON_ACTIVE_BG = "#d9d9d9"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Touch launcher for neuro_tasks")
    parser.add_argument("--config", required=True, help="Path to launcher config JSON")
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


class ScrollableButtonFrame(tk.Frame):
    def __init__(self, master: tk.Misc, **kwargs: Any):
        super().__init__(master, **kwargs)
        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.scrollbar = tk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = tk.Frame(self.canvas)

        self.inner.bind("<Configure>", self._on_inner_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas_window = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

    def _on_inner_configure(self, _event: tk.Event) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event: tk.Event) -> None:
        self.canvas.itemconfigure(self.canvas_window, width=event.width)


class TouchInterfaceApp:
    def __init__(self, root: tk.Tk, config_path: Path, cfg: Dict[str, Any]):
        self.root = root
        self.config_path = config_path.resolve()
        self.config_dir = self.config_path.parent
        self.cfg = cfg
        self.environment_cfg = _expect_dict(cfg.get("environment"), "environment")
        self.tasks_cfg = _expect_dict(cfg.get("tasks"), "tasks")
        self.working_dir = _get_working_directory(self.environment_cfg, self.config_dir)
        self.python_cmd = str(self.environment_cfg.get("python", "")).strip()
        if not self.python_cmd:
            raise KeyError("Config must define environment.python")

        self.task_active = False
        self.status_var = tk.StringVar(value="Ready")
        self.page_title_var = tk.StringVar(value="Task Launcher")
        self.page_stack: list[tuple[str, Dict[str, Any]]] = []

        self.startup()
        self._build_ui()
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
                dt = parsedate_to_datetime(r.headers["Date"])
            subprocess.run(["sudo", "timedatectl", "set-ntp", "false"], check=False)
            subprocess.run(["sudo", "date", "-u", "-s", dt.strftime("%Y-%m-%d %H:%M:%S")], check=True)
        except Exception as e:
            print(f"Could not sync time: {e}")

    def startup(self) -> None:
        os.chdir(self.working_dir)
        # attempt to rectify system timezone
        self.attempt_rectify_timezone()

    def cleanup(self) -> None:
        self.attempt_rectify_timezone()

    def _schedule_idle_cleanup(self) -> None:
        self.root.after(IDLE_CLEANUP_MS, self._run_idle_cleanup_if_needed)

    def _run_idle_cleanup_if_needed(self) -> None:
        if not self.task_active:
            self.cleanup()
        self._schedule_idle_cleanup()

    def _build_ui(self) -> None:
        self.root.title("Task Launcher")
        screen_width = int(self.root.winfo_screenwidth())
        screen_height = int(self.root.winfo_screenheight())
        window_width = max(800, screen_width - 40)
        window_height = max(600, screen_height - 80)
        self.root.geometry(f"{window_width}x{window_height}+20+20")
        self.root.minsize(800, 600)

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

        button_frame = ScrollableButtonFrame(self.root, bg="#e9ecef")
        button_frame.grid(row=1, column=0, sticky="nsew", padx=18, pady=(0, 12))
        button_frame.inner.configure(bg="#e9ecef")
        self.button_container = button_frame.inner

        self.page_stack = [("Task Launcher", self.tasks_cfg)]
        self._render_current_page()

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

    def _render_current_page(self) -> None:
        for child in self.button_container.winfo_children():
            child.destroy()

        page_name, page_cfg = self._current_page()
        self.page_title_var.set(self._current_page_label())

        row_idx = 0
        for task_name, task_cfg in page_cfg.items():
            self._create_task_button(row_idx, task_name, task_cfg)
            row_idx += 1

        if len(self.page_stack) == 1:
            self._create_desktop_button(row_idx)
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
        button.grid(row=row_idx, column=0, sticky="ew", pady=10, padx=10)
        self.button_container.grid_columnconfigure(0, weight=1)

    def _create_desktop_button(self, row_idx: int) -> None:
        button = tk.Button(
            self.button_container,
            text="Desktop",
            command=self.root.destroy,
            **self._button_kwargs(),
        )
        button.grid(row=row_idx, column=0, sticky="ew", pady=10, padx=10)

    def _create_back_button(self, row_idx: int) -> None:
        button = tk.Button(
            self.button_container,
            text="Back",
            command=self._go_back,
            **self._button_kwargs(),
        )
        button.grid(row=row_idx, column=0, sticky="ew", pady=10, padx=10)

    def _build_command(self, task_name: str, task_cfg: Dict[str, Any]) -> subprocess.CompletedProcess:
        launch_value = task_cfg.get("launch")
        if not launch_value:
            raise KeyError(f"Task '{task_name}' is missing required field 'launch'")

        launch_path = _resolve_candidate(
            str(launch_value),
            (self.working_dir, self.config_dir),
        )

        cmd = [self.python_cmd, str(launch_path)]

        task_config_value = task_cfg.get("config")
        if task_config_value:
            task_config_path = _resolve_candidate(
                str(task_config_value),
                (self.working_dir, self.config_dir),
            )
            cmd.extend(["--config", str(task_config_path)])

        self.status_var.set(f"Running: {task_name}")
        self.root.update_idletasks()
        return subprocess.run(cmd, cwd=self.working_dir, check=False)

    def _run_task(self, task_name: str, task_cfg: Dict[str, Any]) -> None:
        if self.task_active:
            return

        self.task_active = True
        try:
            result = self._build_command(task_name, task_cfg)
        except Exception as exc:
            messagebox.showerror("Launch Error", str(exc))
            self.status_var.set(f"Launch failed: {task_name}")
        else:
            self.status_var.set(f"Finished: {task_name} (exit {result.returncode})")
            if result.returncode != 0:
                messagebox.showwarning(
                    "Task Finished",
                    f"Task '{task_name}' exited with status {result.returncode}.",
                )
        finally:
            self.task_active = False
            self.cleanup()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    cfg = load_launcher_config(config_path)

    root = tk.Tk()
    app = TouchInterfaceApp(root, config_path, cfg)
    root.mainloop()


if __name__ == "__main__":
    main()
