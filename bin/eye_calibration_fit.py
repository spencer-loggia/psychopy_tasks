"""Incremental affine calibration from raw eye pixels to subject fractions.

Camera pixels stay in their original UART coordinate system. The fitted output
uses subject-relative axes; the screen transform handles monitor orientation.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from .eye_tracking import EyeCalibration


def _coordinate_pair(values, name):
    if len(values) != 2:
        raise ValueError(f"{name} must contain two coordinates")
    pair = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in pair):
        raise ValueError(f"{name} must contain finite coordinates")
    return pair


@dataclass(frozen=True)
class CalibrationPoint:
    raw_px: tuple[float, float]
    target_fraction: tuple[float, float]

    def __post_init__(self):
        object.__setattr__(self, "raw_px", _coordinate_pair(self.raw_px, "raw_px"))
        object.__setattr__(
            self,
            "target_fraction",
            _coordinate_pair(self.target_fraction, "target_fraction"),
        )


def fit_affine_calibration(
    points: Sequence[CalibrationPoint], *, previous: EyeCalibration
) -> EyeCalibration:
    """Return an affine least-squares fit closest to the previous linear map.

    Centering the samples separates offsets from sensitivity. The pseudoinverse
    changes only the linear directions constrained by the data. A single point
    therefore changes only offsets, two points preserve the unconstrained
    direction, and three noncollinear points determine a complete affine map.
    Collinear or repeated points remain valid; inconsistent targets are fitted
    in least squares. Nearly singular directions are left unchanged rather
    than amplifying tiny sample differences into extreme gains.
    """
    result = deepcopy(previous)
    if not points:
        return result
    raw = np.asarray([point.raw_px for point in points], dtype=float)
    targets = np.asarray([point.target_fraction for point in points], dtype=float)
    matrix = np.asarray(
        [[previous.x_scale, previous.x_cross], [previous.y_cross, previous.y_scale]],
        dtype=float,
    )
    if not np.isfinite(matrix).all():
        raise ValueError("Eye calibration slopes must be finite")
    # Center displacements from a real sample, rather than absolute camera
    # coordinates. In particular, repeated decimal-valued pixels must produce
    # exact zero displacements: roundoff in mean(raw) can otherwise manufacture
    # a tiny nonzero direction whose pseudoinverse implies enormous gains.
    raw_delta = raw - raw[0]
    target_delta = targets - targets[0]
    raw_delta_center = raw_delta.mean(axis=0)
    target_delta_center = target_delta.mean(axis=0)
    raw_center = raw[0] + raw_delta_center
    target_center = targets[0] + target_delta_center
    centered_raw = raw_delta - raw_delta_center
    centered_targets = target_delta - target_delta_center
    residual = centered_targets - centered_raw @ matrix.T
    correction = np.linalg.lstsq(centered_raw, residual, rcond=1e-10)[0].T
    matrix += correction
    offset = target_center - matrix @ raw_center
    result.x_scale, result.x_cross = map(float, matrix[0])
    result.y_cross, result.y_scale = map(float, matrix[1])
    result.x_offset, result.y_offset = map(float, offset)
    return result


class AffineCalibrationFit:
    """Collect observations while retaining exact undo snapshots.

    ``calibration`` remains the same mutable object, so trackers, preview and
    recording can share it. Manual gain changes affect the current map; the
    next observation refits all collected points. Undo restores the calibration
    immediately before the removed point, including any prior manual changes.
    Removing the final point restores the original pre-collection calibration.
    """

    def __init__(self, calibration: EyeCalibration):
        self.calibration = calibration
        self._points = []
        self._snapshots = []

    @property
    def points(self) -> tuple[CalibrationPoint, ...]:
        return tuple(self._points)

    def _restore(self, calibration):
        # Include runtime gain-direction caches, needed when undoing through a
        # zero-gain drag, while preserving references to the shared object.
        self.calibration.__dict__.clear()
        self.calibration.__dict__.update(deepcopy(calibration.__dict__))

    def add_point(self, *, raw_px, target_fraction) -> CalibrationPoint:
        point = CalibrationPoint(raw_px, target_fraction)
        before = deepcopy(self.calibration)
        fitted = fit_affine_calibration([*self._points, point], previous=before)
        self._snapshots.append(before)
        self._points.append(point)
        self._restore(fitted)
        return point

    def remove_last_point(self) -> CalibrationPoint | None:
        if not self._points:
            return None
        point = self._points.pop()
        self._restore(self._snapshots.pop())
        return point
