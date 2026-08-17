"""Shared lifecycle contract for experiment-managed task subprocesses."""
from __future__ import annotations

import os
import time
from pathlib import Path


USER_EXIT_CODE = 130
TASK_WINDOW_READY_ENV = "NEURO_TASK_WINDOW_READY_PATH"
TASK_WINDOW_RELEASE_ENV = "NEURO_TASK_WINDOW_RELEASE_PATH"
TASK_WINDOW_RELEASE_TIMEOUT_S = 10.0
_completed_window_ready_signals: set[tuple[Path, Path | None]] = set()


def signal_task_window_ready() -> bool:
    """Tell the manager once that the main window exists, then await release."""
    raw_path = os.environ.get(TASK_WINDOW_READY_ENV)
    if not raw_path:
        return False
    ready_path = Path(raw_path)
    raw_release_path = os.environ.get(TASK_WINDOW_RELEASE_ENV)
    release_path = Path(raw_release_path) if raw_release_path else None
    signal_key = (ready_path, release_path)
    if signal_key in _completed_window_ready_signals:
        return True

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
    _completed_window_ready_signals.add(signal_key)
    return True
