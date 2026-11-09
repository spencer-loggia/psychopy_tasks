"""Incremental polynomial calibration from raw eye pixels to subject fractions.

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


def _pseudoinverse(matrix, tolerance):
    """Use an absolute cutoff so a projected roundoff residual has rank zero."""
    u, singular, vt = np.linalg.svd(matrix, full_matrices=False)
    keep = singular > tolerance
    return (vt[keep].T / singular[keep]) @ u[:, keep].T, vt[keep]


def _transport_unmeasured_direction(before, after, measured):
    """Retain transverse sensitivity and handedness when only one axis is known.

    A minimum-norm additive update can align both output directions and collapse
    a nonsingular map to a line (e.g. a 90-degree camera rotation). Transport the
    free direction through the measured direction's rotation, retaining its
    length and its angle to that direction. No extra fit parameter is needed.
    """
    old = before @ measured
    new = after @ measured
    old_length, new_length = np.linalg.norm(old), np.linalg.norm(new)
    if old_length == 0 or new_length == 0:
        # Coincident targets can genuinely require zero measured sensitivity.
        return after
    old, new = old / old_length, new / new_length
    cosine = old @ new
    sine = old[0] * new[1] - old[1] * new[0]
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    transverse = np.array([-measured[1], measured[0]])
    return after + np.outer(rotation @ (before @ transverse) - after @ transverse, transverse)


def _fit_calibration(points, *, previous, quadratic):
    result = deepcopy(previous)
    if not points:
        return result
    raw = np.asarray([point.raw_px for point in points], dtype=float)
    targets = np.asarray([point.target_fraction for point in points], dtype=float)
    matrix = np.array([[previous.x_scale, previous.x_cross],
                       [previous.y_cross, previous.y_scale]], dtype=float)
    curve = np.array([[getattr(previous, f"{axis}_quad_{term}")
                       for term in ("xx", "xy", "yy")] for axis in ("x", "y")], dtype=float)
    if not np.isfinite(matrix).all() or not np.isfinite(curve).all():
        raise ValueError("Eye calibration coefficients must be finite")

    # Center displacements from a real sample. Repeated decimal-valued pixels
    # must give exact zero spread rather than a tiny roundoff-only direction.
    raw_delta = raw - raw[0]
    target_delta = targets - targets[0]
    raw_mean = raw_delta.mean(axis=0)
    center = raw[0] + raw_mean
    target_mean = target_delta.mean(axis=0)
    target_center = targets[0] + target_mean
    centered = raw_delta - raw_mean
    radius = np.linalg.norm(centered, axis=1).max()
    if radius == 0:
        result.set_offsets_for_fixation(
            fixation_fraction=target_center, x_px=raw[0, 0], y_px=raw[0, 1],
        )
        return result

    # Center/scale only for solving; saved coefficients still use raw pixels.
    # sqrt(2)*xy makes the curvature norm equal to the symmetric quadratic
    # matrix's Frobenius norm, independent of rotation of the camera axes.
    linear = centered / radius
    x, y = linear.T
    square = np.column_stack((x * x, math.sqrt(2) * x * y, y * y))
    square_mean = square.mean(axis=0)
    square -= square_mean
    cx, cy = center
    local_matrix = radius * (matrix + np.column_stack((
        2 * curve[:, 0] * cx + curve[:, 1] * cy,
        curve[:, 1] * cx + 2 * curve[:, 2] * cy,
    )))
    local_curve = curve * (radius ** 2) / [1, math.sqrt(2), 1]
    residual = target_delta - target_mean - linear @ local_matrix.T - square @ local_curve.T
    tolerance = 1e-10 * np.linalg.norm(np.column_stack((linear, square)), ord=2)
    linear_inverse, directions = _pseudoinverse(linear, tolerance)
    curve_delta = np.zeros((3, 2))
    if quadratic:
        # Lexicographic minimum change: retain curvature whenever an affine
        # correction suffices. Otherwise change only the curvature directions
        # that remain constrained after eliminating translation and linear terms.
        projected_square = square - linear @ (linear_inverse @ square)
        projected_residual = residual - linear @ (linear_inverse @ residual)
        curve_inverse, _ = _pseudoinverse(projected_square, tolerance)
        curve_delta = curve_inverse @ projected_residual
    linear_delta = (linear_inverse @ (residual - square @ curve_delta)).T
    if len(directions) == 1:
        linear_delta = _transport_unmeasured_direction(
            local_matrix, local_matrix + linear_delta, directions[0],
        ) - local_matrix

    raw_curve_delta = curve_delta.T * [1, math.sqrt(2), 1] / (radius ** 2)
    matrix += linear_delta / radius - np.column_stack((
        2 * raw_curve_delta[:, 0] * cx + raw_curve_delta[:, 1] * cy,
        raw_curve_delta[:, 1] * cx + 2 * raw_curve_delta[:, 2] * cy,
    ))
    curve += raw_curve_delta
    result.x_scale, result.x_cross = map(float, matrix[0])
    result.y_cross, result.y_scale = map(float, matrix[1])
    for axis, row in zip(("x", "y"), curve):
        for term, value in zip(("xx", "xy", "yy"), row):
            setattr(result, f"{axis}_quad_{term}", float(value))
    result.set_offsets_for_fixation(
        fixation_fraction=target_center - square_mean @ (local_curve + curve_delta.T).T,
        x_px=cx, y_px=cy,
    )
    return result


def fit_affine_calibration(
    points: Sequence[CalibrationPoint], *, previous: EyeCalibration
) -> EyeCalibration:
    """Fit the affine part, retaining prior curvature and unmeasured sensitivity.

    One point changes only offsets; two rotate the unmeasured direction with
    the measured direction to avoid line collapse. Three noncollinear points
    determine the affine part. Extra/inconsistent points use least squares.
    """
    return _fit_calibration(points, previous=previous, quadratic=False)


def fit_quadratic_calibration(
    points: Sequence[CalibrationPoint], *, previous: EyeCalibration
) -> EyeCalibration:
    """Fit the least change to the existing map, introducing curvature as needed.

    Translation is free, then affine changes are preferred over curvature
    changes. Among curvature changes fitting the data, minimize the Frobenius
    norm of the quadratic matrices; among remaining affine solutions, minimize
    the change in the Jacobian at the sample center. The one-direction case
    transports the free direction instead of permitting accidental collapse.

    Six points with independent [1, x, y, x², xy, y²] rows determine a complete
    quadratic map. More points give least squares; rank-deficient collections
    preserve unconstrained curvature. Numerically singular directions are not
    inverted. No origin, normalization, regularization, or model setting is saved.
    """
    return _fit_calibration(points, previous=previous, quadratic=True)


class AffineCalibrationFit:
    """Collect observations while retaining exact undo snapshots.

    ``calibration`` remains the same mutable object, so trackers, preview and
    recording can share it. Manual gain changes affect the current map; the
    next observation refits all collected points. Undo restores the calibration
    immediately before the removed point, including any prior manual changes.
    Removing the final point restores the original pre-collection calibration.
    """

    _fit = staticmethod(fit_affine_calibration)

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
        fitted = self._fit([*self._points, point], previous=before)
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


class QuadraticCalibrationFit(AffineCalibrationFit):
    """Incremental quadratic fitting with the same shared object and undo contract."""

    _fit = staticmethod(fit_quadratic_calibration)
