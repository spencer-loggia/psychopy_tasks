"""One UART/calibration config contract for calibration and presentation tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Optional

from .eye_tracking import EyeCalibration, EyeFilterConfig, UARTEyeTrackerConfig

CALIBRATION_KEYS = ("x_scale", "y_scale", "x_offset", "y_offset", "x_cross", "y_cross")
_OPTIONAL_CALIBRATION_KEYS = ("x_cross", "y_cross")


@dataclass(frozen=True)
class EyeTrackingSettings:
    uart: UARTEyeTrackerConfig = field(default_factory=UARTEyeTrackerConfig)
    filter: EyeFilterConfig = field(default_factory=EyeFilterConfig)
    calibration: EyeCalibration = field(default_factory=EyeCalibration)
    show_preview: bool = True
    gain_anchor_px: Optional[tuple[float, float]] = None
    record_rate_hz: float = 60.0

    def __post_init__(self):
        if (
            isinstance(self.record_rate_hz, bool)
            or not math.isfinite(self.record_rate_hz)
            or self.record_rate_hz <= 0
        ):
            raise ValueError("eye_tracking.record_rate_hz must be positive and finite")


def _object(value, name):
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _boolean(value, name):
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be true or false")
    return value


def resolve_eye_tracking_paths(cfg, base_dir):
    """Resolve before a source config is copied into an experiment block."""
    section = cfg.get("eye_tracking")
    if isinstance(section, dict) and section.get("calibration_file") is not None:
        value = section["calibration_file"]
        if not isinstance(value, str) or not value.strip():
            raise ValueError("eye_tracking.calibration_file must be a path string")
        path = Path(value).expanduser()
        section["calibration_file"] = str((Path(base_dir) / path).resolve())


def _calibration_object(value):
    # Accept a saved calibration.json object or a state.json history. The
    # experiment manager's flat coefficients are handled separately below.
    if isinstance(value, list):
        value = value[-1] if value else None
    return _object(value, "eye_tracker_calibration")


def load_eye_tracking_settings(cfg, *, for_calibration=False):
    """Return None when disabled; otherwise validate and resolve all settings.

    Precedence: saved calibration file, embedded calibration/history, explicit
    task UART/filter settings, then non-null flattened runtime coefficients.
    Defaults are allowed for the calibration task, but presentation tasks require
    all four legacy measured coefficients; missing affine cross terms default
    to zero. Merely having state fields never opens UART.
    """
    section = _object(cfg.get("eye_tracking"), "eye_tracking")
    enabled = _boolean(section.get("enabled", False), "eye_tracking.enabled")
    if not for_calibration and not enabled:
        return None

    saved = {}
    if section.get("calibration_file") is not None:
        path = Path(section["calibration_file"]).expanduser()
        payload = _object(json.loads(path.read_text(encoding="utf-8")), str(path))
        saved.update(
            _calibration_object(payload.get("eye_tracker_calibration", payload))
        )
    embedded = _calibration_object(cfg.get("eye_tracker_calibration"))
    saved.update({key: value for key, value in embedded.items() if value is not None})
    saved.update(_object(section.get("calibration"), "eye_tracking.calibration"))
    units = saved.get("input_units", cfg.get("input_units"))
    if units is not None and units != "crop_local_pixels":
        raise ValueError(
            "Eye calibration must use crop_local_pixels, not analog voltage"
        )
    eye_id = saved.get("eye_id", cfg.get("eye_id", 0))
    if eye_id not in (None, 0):
        raise ValueError("Only Eye 0 (first UART slot) is supported")
    coordinates = saved.get("coordinate_units", cfg.get("coordinate_units"))
    if coordinates not in (None, "screen_fraction_centered"):
        raise ValueError(
            "Eye calibration must use screen_fraction_centered coordinates"
        )

    values = {}
    defaults = EyeCalibration()
    for key in CALIBRATION_KEYS:
        value = cfg.get(key)
        if value is None:
            value = saved.get(key)
        if value is None and for_calibration:
            value = cfg.get(f"initial_{key}")
            if value is None:
                value = getattr(defaults, key)
        if value is None and key in _OPTIONAL_CALIBRATION_KEYS:
            value = 0.0
        if value is None:
            raise ValueError(
                f"Eye tracking is enabled but calibration {key} is missing; calibrate first"
            )
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"Eye calibration {key} must be finite")
        values[key] = value

    uart = dict(_object(saved.get("uart"), "saved uart"))
    uart.update(_object(cfg.get("uart"), "uart"))
    uart.update(_object(section.get("uart"), "eye_tracking.uart"))
    for key in ("simulate", "manage_tracking", "expect_recording"):
        if key in uart:
            _boolean(uart[key], f"uart.{key}")
    filter_values = dict(_object(saved.get("filter"), "saved filter"))
    filter_values.update(_object(cfg.get("eye_filter"), "eye_filter"))
    filter_values.update(_object(section.get("filter"), "eye_tracking.filter"))
    anchor = saved.get("gain_anchor_px")
    if anchor is not None:
        if not isinstance(anchor, (list, tuple)) or len(anchor) != 2:
            raise ValueError("gain_anchor_px must contain two crop-pixel coordinates")
        anchor = tuple(float(v) for v in anchor)
        if not all(math.isfinite(v) for v in anchor):
            raise ValueError("gain_anchor_px must be finite")
    calibration = EyeCalibration(**values)
    gain_directions = cfg.get("gain_directions")
    if gain_directions is None:
        gain_directions = saved.get("gain_directions")
    if gain_directions is not None:
        calibration.restore_gain_directions(gain_directions)
    try:
        return EyeTrackingSettings(
            uart=UARTEyeTrackerConfig(**uart),
            filter=EyeFilterConfig(**filter_values),
            calibration=calibration,
            show_preview=_boolean(
                section.get("show_preview", True), "eye_tracking.show_preview"
            ),
            gain_anchor_px=anchor,
            record_rate_hz=section.get("record_rate_hz", 60.0),
        )
    except TypeError as exc:
        raise ValueError(f"Invalid eye-tracking configuration: {exc}") from exc
