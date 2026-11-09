"""Numerical calibration behavior in constrained and underconstrained cases."""

from copy import deepcopy
from dataclasses import asdict
import unittest

import numpy as np

from bin.eye_calibration_fit import (
    AffineCalibrationFit,
    CalibrationPoint,
    QuadraticCalibrationFit,
    fit_affine_calibration,
    fit_quadratic_calibration,
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

    def test_two_observations_transport_unconstrained_sensitivity(self):
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
        old_free, new_free = old_matrix @ [-1, 2], new_matrix @ [-1, 2]
        self.assertAlmostEqual(np.linalg.norm(new_free), np.linalg.norm(old_free))
        # Its angle to the measured direction, including handedness, survives.
        old_axis, new_axis = old_matrix @ [2, 1], new_matrix @ [2, 1]
        self.assertAlmostEqual(new_axis @ new_free / np.linalg.norm(new_axis),
                               old_axis @ old_free / np.linalg.norm(old_axis))
        self.assertGreater(np.linalg.det(new_matrix) * np.linalg.det(old_matrix), 0)

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


class EyeQuadraticCalibrationTests(unittest.TestCase):
    RAW = ((80, 30), (120, 30), (120, 70), (80, 70), (100, 50), (100, 70))

    def setUp(self):
        self.prior = EyeCalibration(
            x_scale=0.003, x_cross=0.001, y_cross=-0.001, y_scale=-0.004,
            x_offset=-0.4, y_offset=0.1,
            x_quad_xx=2e-6, x_quad_xy=-3e-6, x_quad_yy=1e-6,
            y_quad_xx=-1e-6, y_quad_xy=2e-6, y_quad_yy=-2e-6,
        )
        self.truth = EyeCalibration(
            x_scale=0.004, x_cross=-0.002, y_cross=0.003, y_scale=-0.002,
            x_offset=-0.5, y_offset=0.2,
            x_quad_xx=1e-5, x_quad_xy=4e-6, x_quad_yy=-8e-6,
            y_quad_xx=-7e-6, y_quad_xy=-5e-6, y_quad_yy=9e-6,
        )

    def assertMapping(self, actual, expected, raw):
        np.testing.assert_allclose(actual.map_position(*raw), expected.map_position(*raw),
                                   rtol=1e-9, atol=1e-11)

    def test_polynomial_terms_and_one_point_translation(self):
        calibration = EyeCalibration(
            x_scale=0, y_scale=0, x_quad_xx=1, x_quad_xy=2, x_quad_yy=3,
            y_quad_xx=4, y_quad_xy=5, y_quad_yy=6,
        )
        self.assertEqual(calibration.map_position(2, 3), (43, 100))
        original = deepcopy(self.prior)
        fit = QuadraticCalibrationFit(self.prior)
        target = (0.2, -0.3)
        shift = np.array(target) - original.map_position(100, 50)
        fit.add_point(raw_px=(100, 50), target_fraction=target)
        for key, value in asdict(original).items():
            if not key.endswith('offset'):
                self.assertEqual(getattr(self.prior, key), value)
        for raw in self.RAW:
            np.testing.assert_allclose(self.prior.map_position(*raw),
                                       np.array(original.map_position(*raw)) + shift)

    def test_affine_initialization_and_two_point_rotations_never_collapse(self):
        for fit_type in (AffineCalibrationFit, QuadraticCalibrationFit):
            for degrees in (0, 30, 89.999, 90, 135, 180, 270):
                with self.subTest(fit=fit_type, degrees=degrees):
                    calibration = EyeCalibration()
                    fit = fit_type(calibration)
                    angle = np.radians(degrees)
                    target = np.array([np.cos(angle), np.sin(angle)]) * 0.3
                    fit.add_point(raw_px=(100, 50), target_fraction=(0, 0))
                    fit.add_point(raw_px=(140, 50), target_fraction=target)
                    np.testing.assert_allclose(calibration.map_position(140, 50), target, atol=1e-12)
                    free = np.array(calibration.map_position(100, 90))
                    self.assertAlmostEqual(np.linalg.norm(free), 0.2)
                    self.assertAlmostEqual(target @ free, 0, places=12)
                    matrix = [[calibration.x_scale, calibration.x_cross],
                              [calibration.y_cross, calibration.y_scale]]
                    self.assertLess(np.linalg.det(matrix), -1e-6)
                    self.assertTrue(all(value == 0 for key, value in asdict(calibration).items()
                                        if 'quad' in key))

    def test_three_points_preserve_existing_curvature(self):
        fit = QuadraticCalibrationFit(self.prior)
        curve = {key: value for key, value in asdict(self.prior).items() if 'quad' in key}
        for raw in self.RAW[:3]:
            fit.add_point(raw_px=raw, target_fraction=self.truth.map_position(*raw))
            for key, value in curve.items():
                self.assertEqual(getattr(self.prior, key), value)
        for raw in self.RAW[:3]:
            self.assertMapping(self.prior, self.truth, raw)

    def test_four_and_five_points_make_minimal_curvature_changes_then_six_recover_map(self):
        original = deepcopy(self.prior)
        fit = QuadraticCalibrationFit(self.prior)
        for count, raw in enumerate(self.RAW, 1):
            fit.add_point(raw_px=raw, target_fraction=self.truth.map_position(*raw))
            for captured in self.RAW[:count]:
                self.assertMapping(self.prior, self.truth, captured)
            for axis in ('x', 'y'):
                if count == 4:
                    # Four rectangle corners constrain xy; xx and yy remain free.
                    for term in ('xx', 'yy'):
                        key = f'{axis}_quad_{term}'
                        self.assertAlmostEqual(getattr(self.prior, key), getattr(original, key), places=13)
                    self.assertAlmostEqual(getattr(self.prior, f'{axis}_quad_xy'),
                                           getattr(self.truth, f'{axis}_quad_xy'), places=13)
                if count == 5:
                    # Adding the center constrains xx+yy, preserving xx-yy.
                    self.assertAlmostEqual(
                        getattr(self.prior, f'{axis}_quad_xx') - getattr(self.prior, f'{axis}_quad_yy'),
                        getattr(original, f'{axis}_quad_xx') - getattr(original, f'{axis}_quad_yy'), places=13)
        for key, value in asdict(self.truth).items():
            self.assertAlmostEqual(getattr(self.prior, key), value, places=11)
        for raw in ((105, 58), (62, 88), (150, 20)):
            self.assertMapping(self.prior, self.truth, raw)

    def test_overdetermined_fit_averages_noise_and_recovers_after_manual_edits(self):
        fit = QuadraticCalibrationFit(self.prior)
        for raw in self.RAW:
            for noise in (-0.02, 0.02):
                fit.add_point(raw_px=raw, target_fraction=np.array(self.truth.map_position(*raw)) + noise)
        self.prior.set_gains(x_scale=0.02, y_scale=-0.03, anchor_px=(100, 50))
        fit.add_point(raw_px=(110, 60), target_fraction=self.truth.map_position(110, 60))
        for raw in (*self.RAW, (130, 40)):
            self.assertMapping(self.prior, self.truth, raw)

    def test_repeated_and_nearly_collinear_points_remain_stable(self):
        original = deepcopy(self.prior)
        points = [CalibrationPoint((0.1, 0.3), (i * 0.1, -i * 0.1)) for i in range(7)]
        result = fit_quadratic_calibration(points, previous=original)
        for key, value in asdict(original).items():
            if not key.endswith('offset'):
                self.assertEqual(getattr(result, key), value)
        np.testing.assert_allclose(result.map_position(0.1, 0.3), (0.3, -0.3))
        points = [CalibrationPoint((100 + n * 10, 50 + n * 20 + (n % 2) * 1e-12),
                                   (0.01 * n * n, -0.1 * n)) for n in range(7)]
        result = fit_quadratic_calibration(points, previous=original)
        for point in points:
            np.testing.assert_allclose(result.map_position(*point.raw_px), point.target_fraction, atol=1e-10)
        self.assertLess(max(abs(value) for value in asdict(result).values()), 10)

    def test_large_camera_origin_does_not_ill_condition_the_fit(self):
        shift = np.array([3000, 2000])
        points = [CalibrationPoint(np.array(raw) + shift, self.truth.map_position(*raw))
                  for raw in self.RAW]
        result = fit_quadratic_calibration(points, previous=EyeCalibration())
        for raw in (*self.RAW, (105, 58), (150, 20)):
            np.testing.assert_allclose(result.map_position(*(np.array(raw) + shift)),
                                       self.truth.map_position(*raw), atol=1e-10)

    def test_integer_valued_initial_coefficients_can_be_refitted(self):
        previous = EyeCalibration(**{key: 0 for key in asdict(self.prior)})
        points = [CalibrationPoint(raw, self.truth.map_position(*raw)) for raw in self.RAW]
        for fit in (fit_affine_calibration, fit_quadratic_calibration):
            result = fit(points, previous=previous)
            self.assertTrue(all(np.isfinite(value) for value in asdict(result).values()))
        self.assertMapping(result, self.truth, (110, 60))

    def test_undo_restores_curvature_and_zero_gain_metadata_exactly(self):
        fit = QuadraticCalibrationFit(self.prior)
        original = deepcopy(self.prior.__dict__)
        snapshots = []
        for raw in self.RAW:
            self.prior.set_gains(x_scale=0, y_scale=0, anchor_px=raw)
            snapshots.append(deepcopy(self.prior.__dict__))
            fit.add_point(raw_px=raw, target_fraction=self.truth.map_position(*raw))
        for snapshot in reversed(snapshots):
            fit.remove_last_point()
            self.assertEqual(self.prior.__dict__, snapshot)
        self.assertEqual(fit.points, ())
        # A fresh capture with no intervening edits restores the initial map.
        self.prior.__dict__.clear()
        self.prior.__dict__.update(original)
        fit.add_point(raw_px=self.RAW[0], target_fraction=(0, 0))
        fit.remove_last_point()
        self.assertEqual(self.prior.__dict__, original)

    def test_gain_changes_scale_entire_curve_around_anchor_through_zero(self):
        anchor = (100, 50)
        original = deepcopy(self.prior)
        target = np.array(original.map_position(*anchor))
        gains = self.prior.x_gain, self.prior.y_gain
        for factors in ((2, 0.5), (0, 0), (-1, -2)):
            self.prior.set_gains(x_scale=gains[0] * factors[0], y_scale=gains[1] * factors[1], anchor_px=anchor)
            for raw in self.RAW:
                expected = target + (np.array(original.map_position(*raw)) - target) * factors
                np.testing.assert_allclose(self.prior.map_position(*raw), expected, atol=1e-12)

    def test_invalid_capture_does_not_mutate_points_or_mapping(self):
        fit = QuadraticCalibrationFit(self.prior)
        fit.add_point(raw_px=self.RAW[0], target_fraction=(0, 0))
        original, points = deepcopy(self.prior.__dict__), fit.points
        with self.assertRaises(ValueError):
            fit.add_point(raw_px=(float('nan'), 0), target_fraction=(0, 0))
        self.assertEqual(self.prior.__dict__, original)
        self.assertEqual(fit.points, points)


if __name__ == "__main__":
    unittest.main()
