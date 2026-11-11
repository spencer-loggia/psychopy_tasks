"""Dot fixation shares the timestamped eye detector and calibrated geometry."""

import unittest

from bin.dot_fixation import DotFixationGate, validate_dot_fixation_config
from bin.eye_tracking import EyeCalibration, EyeTrackerState, UARTEyeTrackerConfig
from bin.eye_tracking_config import EyeTrackingSettings
from bin.screen import oriented_size, rotate_centered_point


class DotFixationConfigTests(unittest.TestCase):
    def settings(self, **updates):
        values = dict(
            wait_for_saccade_on_dot=True, isi=0.1, eye_tracking_enabled=True,
            fixation_criteria_s=0.1, fixation_window_radius=150,
            saccade_wait_timeout_s=1.0,
        )
        values.update(updates)
        return values

    def test_enabled_requires_eyes_and_positive_isi(self):
        validate_dot_fixation_config(**self.settings())
        for field, value in (("wait_for_saccade_on_dot", 1), ("isi", 0),
                             ("eye_tracking_enabled", False)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_dot_fixation_config(**self.settings(**{field: value}))

    def test_invalid_limits_rejected_only_when_enabled(self):
        for field in ("fixation_criteria_s", "fixation_window_radius", "saccade_wait_timeout_s"):
            for value in (None, 0, -1, float("inf"), float("nan"), "invalid", True):
                if field == "saccade_wait_timeout_s" and value is None:
                    continue
                with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, field):
                    validate_dot_fixation_config(**self.settings(**{field: value}))
        validate_dot_fixation_config(**self.settings(
            wait_for_saccade_on_dot=False, isi=0, eye_tracking_enabled=False,
            fixation_criteria_s=None, fixation_window_radius=None, saccade_wait_timeout_s=None,
        ))

    def test_null_timeout_disables_timeout_but_keeps_fixation_validation(self):
        validate_dot_fixation_config(**self.settings(saccade_wait_timeout_s=None))
        with self.assertRaisesRegex(ValueError, "fixation_criteria_s"):
            validate_dot_fixation_config(**self.settings(
                saccade_wait_timeout_s=None, fixation_criteria_s=None,
            ))

    def test_optional_sequential_options_require_boolean_policy_and_finite_timeout(self):
        for required in (False, True):
            validate_dot_fixation_config(**self.settings(require_saccade_to_all_options=required))
        for required in (None, 0, 1, "false"):
            with self.subTest(required=required), self.assertRaisesRegex(ValueError, "require_saccade_to_all_options"):
                validate_dot_fixation_config(**self.settings(require_saccade_to_all_options=required))
        with self.assertRaisesRegex(ValueError, "saccade_wait_timeout_s must be finite"):
            validate_dot_fixation_config(**self.settings(
                require_saccade_to_all_options=False, saccade_wait_timeout_s=None,
            ))
        for options in ({"sequential": False}, {"wait_for_saccade_on_dot": False}):
            validate_dot_fixation_config(**self.settings(
                require_saccade_to_all_options=False, saccade_wait_timeout_s=None, **options,
            ))


class DotFixationGateTests(unittest.TestCase):
    def gate(self, positions=((0, 0),), *, size=(1000, 600), rotation=0,
             onset=1.0, calibration=None, criteria=0.1, radius=50):
        gate = DotFixationGate(
            EyeTrackingSettings(
                calibration=calibration or EyeCalibration(x_scale=1, y_scale=1),
                uart=UARTEyeTrackerConfig(max_sample_age_s=0.04),
            ),
            fixation_criteria_s=criteria, fixation_window_radius=radius,
        )
        gate.start(positions, native_size=size, rotation_degrees=rotation, onset_perf_s=onset)
        return gate

    def feed(self, gate, start_ms=1000, end_ms=1110, *, position=(0, 0)):
        for ms in range(start_ms, end_ms + 1, 10):
            gate.observe(EyeTrackerState(
                timestamp_perf_s=ms / 1000, frame_sequence=ms, camera_time_ns=ms * 1_000_000,
                valid=True, x_smooth_px=position[0], y_smooth_px=position[1],
            ))

    def test_full_duration_required_and_start_excludes_pre_dot_samples(self):
        gate = self.gate()
        self.feed(gate, 900, 990)
        self.assertIsNone(gate.status(now_s=1.1))
        self.feed(gate, 1000, 1090)
        self.assertIsNone(gate.status(now_s=1.09))
        self.feed(gate, 1100, 1110)
        self.assertEqual(gate.status(now_s=1.11), 1)

    def test_continuous_fixation_accepts_at_realistic_performance_clock_origins(self):
        for origin in (10, 100, 1000, 100_000, 1_000_000, 10_000_000):
            with self.subTest(origin=origin):
                gate = self.gate(onset=origin)
                for index in range(20):
                    now = origin + index * 0.01
                    gate.observe(EyeTrackerState(
                        timestamp_perf_s=now, frame_sequence=index, valid=True,
                    ))
                self.assertEqual(gate.status(now_s=now), 1)

    def test_breaking_window_requires_another_continuous_interval(self):
        gate = self.gate()
        self.feed(gate, 1000, 1060)
        self.feed(gate, 1070, 1070, position=(0.1, 0))
        self.feed(gate, 1080, 1160)
        self.assertIsNone(gate.status(now_s=1.16))
        self.feed(gate, 1170, 1190)
        self.assertEqual(gate.status(now_s=1.19), 1)

    def test_saccade_arrival_is_latched_independently_of_fixation_duration(self):
        gate = self.gate()
        self.feed(gate, 990, 990)
        gate.observe(EyeTrackerState(timestamp_perf_s=1.01, valid=False))
        self.feed(gate, 1020, 1020, position=(0.2, 0))
        self.assertFalse(gate.reached_dot_before(1.05))
        self.feed(gate, 1040, 1040)
        self.assertIsNone(gate.status(now_s=1.04))
        self.feed(gate, 1050, 1050, position=(0.2, 0))
        self.assertFalse(gate.reached_dot_before(1.03))
        with gate._lock:
            self.assertTrue(gate.reached_dot_before(1.05))
        self.assertIsNone(gate.status(now_s=1.05))

    def test_new_dot_and_stop_clear_the_previous_saccade_arrival(self):
        gate = self.gate()
        self.feed(gate)
        self.assertTrue(gate.reached_dot_before(1.1))
        gate.start(((200, 0),), native_size=(1000, 600), rotation_degrees=0, onset_perf_s=1.2)
        self.feed(gate, 1190, 1190, position=(0.2, 0))
        self.feed(gate, 1200, 1200)
        self.assertFalse(gate.reached_dot_before(1.3))
        self.feed(gate, 1250, 1250, position=(0.2, 0))
        self.assertTrue(gate.reached_dot_before(1.3))
        gate.stop()
        self.assertFalse(gate.reached_dot_before(float("inf")))

    def test_invalid_and_stale_samples_cannot_accept(self):
        for reason in ("blink", "lost", "stale", "uart_error", "service_restarted"):
            with self.subTest(reason=reason):
                gate = self.gate()
                self.feed(gate)
                gate.observe(EyeTrackerState(
                    timestamp_perf_s=1.11, frame_sequence=1110, camera_time_ns=1_110_000_000,
                    valid=False, rejection_reason=reason,
                ))
                self.assertIsNone(gate.status(now_s=1.11))
                self.feed(gate, 1120, 1190)
                self.assertIsNone(gate.status(now_s=1.19))
                self.feed(gate, 1200, 1230)
                self.assertEqual(gate.status(now_s=1.23), 1)
        gate = self.gate()
        self.feed(gate)
        self.assertIsNone(gate.status(now_s=1.151))
        self.feed(gate, 1160, 1180)
        self.assertIsNone(gate.status(now_s=1.18))

    def test_replayed_camera_frame_cannot_refresh_stale_history(self):
        gate = self.gate()
        state = EyeTrackerState(timestamp_perf_s=1, frame_sequence=1, valid=True)
        for _ in range(100):
            gate.observe(state)
        self.assertIsNone(gate.status(now_s=1.1))

    def test_stop_and_new_target_discard_previous_history(self):
        gate = self.gate()
        self.feed(gate)
        self.assertEqual(gate.status(now_s=1.11), 1)
        gate.stop()
        self.feed(gate, 1120, 1200)
        self.assertIsNone(gate.status(now_s=1.2))
        gate.start(((0, 0),), native_size=(1000, 600), rotation_degrees=0, onset_perf_s=1.2)
        self.feed(gate, 1100, 1190)
        self.assertIsNone(gate.status(now_s=1.2))
        self.feed(gate, 1200, 1310)
        self.assertEqual(gate.status(now_s=1.31), 1)

    def test_each_dot_has_its_own_continuous_history(self):
        gate = self.gate(((-200, 0), (200, 0)))
        self.feed(gate, 1000, 1060, position=(-0.2, 0))
        self.feed(gate, 1070, 1130, position=(0.2, 0))
        self.assertIsNone(gate.status(now_s=1.13))
        self.feed(gate, 1140, 1180, position=(0.2, 0))
        self.assertEqual(gate.status(now_s=1.18), 2)

    def test_rotated_native_targets_use_oriented_subject_pixel_radius(self):
        for rotation in (0, 90, 180, 270):
            size = (1000, 600)
            subject_size = oriented_size(size, rotation)
            target = (150, -100)
            native_target = rotate_centered_point(target, rotation)
            for offset, accepted in ((49, True), (50, True), (51, False)):
                with self.subTest(rotation=rotation, offset=offset):
                    gate = self.gate((native_target,), size=size, rotation=rotation)
                    self.feed(gate, position=(
                        (target[0] + offset) / subject_size[0], target[1] / subject_size[1],
                    ))
                    self.assertEqual(gate.status(now_s=1.11), 1 if accepted else None)

    def test_affine_calibration_and_nonfinite_positions(self):
        calibration = EyeCalibration(
            x_scale=0, y_scale=0, x_cross=0.01, y_cross=-0.01,
            x_offset=-1, y_offset=1,
        )
        for raw, accepted in (((100, 100), True), ((100, 110), False),
                              ((float("nan"), 100), False), ((100, float("inf")), False)):
            with self.subTest(raw=raw):
                gate = self.gate(calibration=calibration)
                self.feed(gate, position=raw)
                self.assertEqual(gate.status(now_s=1.11), 1 if accepted else None)

    def test_status_does_not_wait_for_acquisition_lock(self):
        gate = self.gate()
        self.feed(gate)
        with gate._lock:
            self.assertIsNone(gate.status(now_s=1.11))
        self.assertEqual(gate.status(now_s=1.11), 1)


if __name__ == "__main__":
    unittest.main()
