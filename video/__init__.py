"""Low-overhead recording for the Arducam quad-camera module."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .recorder import Recording

__all__ = ["record"]


def record(
    destination: str | Path,
    duration_seconds: float | None = None,
    *,
    config_path: str | Path | None = None,
    overwrite: bool = False,
) -> "Recording":
    """Start the recorder without importing camera code at package import time."""
    from .recorder import DEFAULT_CONFIG_PATH, record as start_recording

    return start_recording(
        destination,
        duration_seconds,
        config_path=DEFAULT_CONFIG_PATH if config_path is None else config_path,
        overwrite=overwrite,
    )
