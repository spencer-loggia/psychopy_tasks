"""Shared lifecycle contract for experiment-managed task subprocesses."""
from __future__ import annotations

import os
from pathlib import Path


USER_EXIT_CODE = 130
TASK_WINDOW_READY_ENV = "NEURO_TASK_WINDOW_READY_PATH"


def signal_task_window_ready() -> bool:
    """Tell the experiment manager that the task's main window is ready."""
    raw_path = os.environ.get(TASK_WINDOW_READY_ENV)
    if not raw_path:
        return False
    ready_path = Path(raw_path)
    ready_path.write_text("ready\n", encoding="utf-8")
    return True
