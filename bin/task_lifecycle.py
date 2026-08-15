"""Shared lifecycle contract for experiment-managed task subprocesses."""
from __future__ import annotations

import os
import time
from pathlib import Path


USER_EXIT_CODE = 130
TASK_WINDOW_READY_ENV = "NEURO_TASK_WINDOW_READY_PATH"
TASK_WINDOW_RELEASE_ENV = "NEURO_TASK_WINDOW_RELEASE_PATH"
TASK_WINDOW_RELEASE_TIMEOUT_S = 10.0


def signal_task_window_ready() -> bool:
    """Tell the manager the main window exists, then await guarded release."""
    raw_path = os.environ.get(TASK_WINDOW_READY_ENV)
    if not raw_path:
        return False
    ready_path = Path(raw_path)
    raw_release_path = os.environ.get(TASK_WINDOW_RELEASE_ENV)
    release_path = Path(raw_release_path) if raw_release_path else None
    if release_path is not None:
        release_path.unlink(missing_ok=True)
    ready_path.write_text("ready\n", encoding="utf-8")

    if release_path is not None:
        deadline = time.monotonic() + TASK_WINDOW_RELEASE_TIMEOUT_S
        while not release_path.is_file():
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "Timed out waiting for the launcher to uncover the task window"
                )
            time.sleep(0.01)
    return True
