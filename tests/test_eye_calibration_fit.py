"""Numerical calibration behavior in constrained and underconstrained cases."""

from copy import deepcopy
from dataclasses import asdict
import unittest

import numpy as np

from bin.eye_calibration_fit import (
    AffineCalibrationFit,
    CalibrationPoint,
    fit_affine_calibration,
)
from bin.eye_tracking import EyeCalibration


class EyeAffineCalibrationTests(unittest.TestCase):
    def assertPosition(self, calibration, raw, expected):
        np.testing.assert_allclose(
            calibration.map_position(*raw), expected, rtol=1e-10, atol=1e-12
        )

    def test_cross_terms_map_raw_camera_axes_to_subject_axes(self):
        calibration = EyeCalibration(
            x_scale=0, y_scale=0, x_cross=0.01, y_cross=-0.02,
            x_offset=-0.5, y_offset=2,
        )
        self.assertPosition(calibration, (100, 50), (0, 0))
        self.assertPosition(calibration, (105, 60), (0.1, -0.1))
        calibration.set_offsets_for_fixation(
            fixation_fraction=(0.2, -0.3), x_px=100, y_px=50
        )
        self.assertPosition(calibration, (100, 50), (0.2, -0.3))

    def test_signed_row_gains_include_cross_terms(self):
        calibration = EyeCalibration(
            x_scale=0.003, x_cross=0.004, y_scale=-0.004, y_cross=0.003,
        )
        self.assertAlmostEqual(calibration.x_gain, 0.005)
        self.assertAlmostEqual(calibration.y_gain, -0.005)
        # A dominant cross coefficient controls the sign even near a zero
        # diagonal; the diagonal breaks ties in a deterministic way.
        calibration.x_scale, calibration.x_cross = 1e-15, -0.01
        self.assertAlmostEqual(calibration.x_gain, -0.01)
        calibration.x_scale, calibration.x_cross = 0.01, -0.01
        self.assertGreater(calibration.x_gain, 0)

    def test_gains_preserve_affine_direction_and_anchor_through_zero(self):
        calibration = EyeCalibration(
            x_scale=0.003, x_cross=0.004, y_cross=0.006, y_scale=-0.008,
            x_offset=-0.5, y_offset=0.2,
        )
        anchor = (100, 50)
        mapped_anchor = calibration.map_position(*anchor)
        initial_displacement = (
            np.asarray(calibration.map_position(110, 60)) - mapped_anchor
        )
        for gains, factors in (((0.01, -0.005), (2, 0.5)), ((0, 0), (0, 0)),
                               ((-0.005, 0.02), (-1, -2))):
            calibration.set_gains(x_scale=gains[0], y_scale=gains[1], anchor_px=anchor)
            self.assertPosition(calibration, anchor, mapped_anchor)
            self.assertPosition(
                calibration, (110, 60),
                np.asarray(mapped_anchor) + initial_displacement * factors,
            )
            np.testing.assert_allclose((calibration.x_gain, calibration.y_gain), gains)
        self.assertNotIn("_x_gain_direction", calibration.to_json_dict())

    def test_pure_cross_axis_gain_survives_zero_and_sign_reversal(self):
        calibration = EyeCalibration(x_scale=0, y_scale=0, x_cross=0.01, y_cross=-0.02)
        anchor = (20, 30)
        expected = calibration.map_position(*anchor)
        calibration.set_gains(x_scale=0, y_scale=0, anchor_px=anchor)
        calibration.set_gains(x_scale=-0.03, y_scale=0.01, anchor_px=anchor)
        self.assertEqual((calibration.x_scale, calibration.y_scale), (0, 0))
        self.assertAlmostEqual(calibration.x_cross, -0.03)
        self.assertAlmostEqual(calibration.y_cross, 0.01)
        self.assertPosition(calibration, anchor, expected)

    def test_single_observation_changes_offsets_only(self):
        calibration = EyeCalibration(
            x_scale=0.002, x_cross=0.004, y_cross=-0.003, y_scale=0.001,
            x_offset=1, y_offset=-2,
        )
        old = deepcopy(calibration)
        fit = AffineCalibrationFit(calibration)
        point = fit.add_point(raw_px=(100, 50), target_fraction=(0.2, -0.1))
        self.assertEqual(fit.points, (point,))
        self.assertEqual(
            (calibration.x_scale, calibration.x_cross, calibration.y_cross, calibration.y_scale),
            (old.x_scale, old.x_cross, old.y_cross, old.y_scale),
        )
        self.assertPosition(calibration, (100, 50), (0.2, -0.1))
        self.assertIs(fit.calibration, calibration)

    def test_two_observations_preserve_unconstrained_direction(self):
        calibration = EyeCalibration(x_scale=0.003, x_cross=0.004, y_cross=-0.002)
        old_matrix = np.array([
            [calibration.x_scale, calibration.x_cross],
            [calibration.y_cross, calibration.y_scale],
        ])
        fit = AffineCalibrationFit(calibration)
        fit.add_point(raw_px=(100, 50), target_fraction=(-0.2, -0.1))
        fit.add_point(raw_px=(120, 60), target_fraction=(0.3, 0.4))
        self.assertPosition(calibration, (100, 50), (-0.2, -0.1))
        self.assertPosition(calibration, (120, 60), (0.3, 0.4))
        new_matrix = np.array([
            [calibration.x_scale, calibration.x_cross],
            [calibration.y_cross, calibration.y_scale],
        ])
        # (-1, 2) is perpendicular to the sampled displacement (20, 10).
        np.testing.assert_allclose(new_matrix @ [-1, 2], old_matrix @ [-1, 2])

    def test_three_noncollinear_points_recover_complete_affine_map(self):
        truth = EyeCalibration(
            x_scale=0.001, y_scale=-0.003, x_cross=0.004, y_cross=0.002,
            x_offset=-0.7, y_offset=0.2,
        )
        calibration = EyeCalibration()
        fit = AffineCalibrationFit(calibration)
        for raw in ((100, 50), (140, 55), (110, 90)):
            fit.add_point(raw_px=raw, target_fraction=truth.map_position(*raw))
        for raw in ((100, 50), (150, 30), (10, 90), (-20, -10)):
            self.assertPosition(calibration, raw, truth.map_position(*raw))

    def test_full_rank_refit_ignores_manual_gain_changes(self):
        truth = EyeCalibration(x_scale=0.002, y_scale=-0.004, x_cross=0.001)
        calibration = EyeCalibration()
        fit = AffineCalibrationFit(calibration)
        for raw in ((100, 50), (120, 50), (100, 70)):
            fit.add_point(raw_px=raw, target_fraction=truth.map_position(*raw))
        calibration.set_gains(x_scale=0.02, y_scale=0.03, anchor_px=(100, 50))
        fit.add_point(raw_px=(120, 70), target_fraction=truth.map_position(120, 70))
        self.assertPosition(calibration, (130, 90), truth.map_position(130, 90))

    def test_overdetermined_fit_averages_inconsistent_observations(self):
        truth = EyeCalibration(x_cross=0.002, y_cross=0.003, x_offset=-0.4)
        points = []
        for raw in ((100, 50), (130, 50), (100, 80), (130, 80)):
            target = np.asarray(truth.map_position(*raw))
            for noise in (-0.03, 0.03):
                points.append(CalibrationPoint(raw, target + noise))
        result = fit_affine_calibration(points, previous=EyeCalibration())
        self.assertPosition(result, (110, 70), truth.map_position(110, 70))

    def test_repeated_points_average_targets_without_changing_slopes(self):
        calibration = EyeCalibration(x_cross=0.001, y_cross=0.003)
        old = asdict(calibration)
        fit = AffineCalibrationFit(calibration)
        for target in ((-0.2, 0.1), (0.4, 0.3), (0.1, -0.1)):
            fit.add_point(raw_px=(100, 50), target_fraction=target)
        self.assertPosition(calibration, (100, 50), (0.1, 0.1))
        for key in ("x_scale", "y_scale", "x_cross", "y_cross"):
            self.assertEqual(getattr(calibration, key), old[key])

    def test_collinear_and_nearly_collinear_points_do_not_explode(self):
        calibration = EyeCalibration(x_cross=0.001, y_cross=0.003)
        points = [
            CalibrationPoint((100 + n * 10, 50 + n * 20 + (n % 2) * 1e-12),
                             (-0.2 + n * 0.1, 0.2 - n * 0.05))
            for n in range(5)
        ]
        result = fit_affine_calibration(points, previous=calibration)
        for point in points:
            self.assertPosition(result, point.raw_px, point.target_fraction)
        self.assertLess(max(abs(value) for value in asdict(result).values()), 1)

    def test_repeated_decimal_pixels_do_not_create_roundoff_slopes(self):
        original = EyeCalibration(x_cross=0.001, y_cross=0.003)
        for count in range(3, 11):
            points = [
                CalibrationPoint((0.1, 0.3), (0.1 + i * 0.2, -0.1 + i * 0.1))
                for i in range(count)
            ]
            result = fit_affine_calibration(points, previous=original)
            for key in ("x_scale", "y_scale", "x_cross", "y_cross"):
                with self.subTest(count=count, key=key):
                    self.assertEqual(getattr(result, key), getattr(original, key))
            self.assertPosition(
                result, (0.1, 0.3),
                np.mean([point.target_fraction for point in points], axis=0),
            )

    def test_undo_restores_exact_snapshots_and_empty_collection(self):
        calibration = EyeCalibration(x_cross=0.001)
        original = asdict(calibration)
        fit = AffineCalibrationFit(calibration)
        self.assertIsNone(fit.remove_last_point())
        first = fit.add_point(raw_px=(100, 50), target_fraction=(0.1, 0.2))
        calibration.set_gains(x_scale=0.01, y_scale=-0.01, anchor_px=(100, 50))
        before_second = deepcopy(calibration.__dict__)
        second = fit.add_point(raw_px=(150, 70), target_fraction=(-0.1, -0.2))
        self.assertEqual(fit.remove_last_point(), second)
        self.assertEqual(calibration.__dict__, before_second)
        self.assertEqual(fit.points, (first,))
        self.assertEqual(fit.remove_last_point(), first)
        self.assertEqual(asdict(calibration), original)
        self.assertEqual(fit.points, ())
        self.assertIsNone(fit.remove_last_point())

    def test_invalid_point_is_rejected_without_mutation(self):
        calibration = EyeCalibration()
        original = asdict(calibration)
        fit = AffineCalibrationFit(calibration)
        for raw, target in (((1,), (0, 0)), ((1, 2), (float("nan"), 0)),
                            ((1, float("inf")), (0, 0))):
            with self.subTest(raw=raw, target=target), self.assertRaises(ValueError):
                fit.add_point(raw_px=raw, target_fraction=target)
            self.assertEqual(asdict(calibration), original)
            self.assertEqual(fit.points, ())


if __name__ == "__main__":
    unittest.main()
