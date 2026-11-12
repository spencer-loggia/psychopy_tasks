"""Offset drift correction must not alter fixation timing or the fitted map."""

from dataclasses import asdict, replace
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from bin.dot_fixation import DotFixationGate, FIXATION_UNAVAILABLE
from bin.eye_tracking import EyeCalibration, EyeTrackerState, UARTEyeTrackerConfig
from bin.eye_tracking_config import EyeTrackingSettings, load_eye_tracking_settings
from bin.logger import SessionLogBundle
from bin.screen import rotate_centered_point
from bin.task_eye_tracking import read_preview_gaze


class AutomaticRecalibrationTests(unittest.TestCase):
    def make_gate(self, *, enabled=True, criteria=0.1, calibration=None, rotation=0):
        settings = EyeTrackingSettings(
            calibration=calibration or EyeCalibration(x_scale=1, y_scale=1),
            uart=UARTEyeTrackerConfig(max_sample_age_s=0.04),
            automatic_recalibration=enabled,
        )
        gate = DotFixationGate(settings, fixation_criteria_s=criteria, fixation_window_radius=100)
        # Use a square scene to isolate rotation from the sample arithmetic.
        gate.start((rotate_centered_point((200, -100), rotation),), native_size=(1000, 1000),
                   rotation_degrees=rotation, onset_perf_s=1)
        return settings, gate

    def state(self, ms, position=(0.22, -0.12), **kwargs):
        return EyeTrackerState(
            timestamp_perf_s=ms / 1000, frame_sequence=ms,
            camera_time_ns=ms * 1_000_000, valid=True,
            x_smooth_px=position[0], y_smooth_px=position[1], **kwargs,
        )

    def feed(self, gate, start=1000, end=1100, **kwargs):
        for ms in range(start, end + 1, 10):
            gate.observe(self.state(ms, **kwargs))

    def test_disabled_by_default_and_requires_enabled_eye_tracking(self):
        cfg = {"eye_tracking": {"enabled": True}, **asdict(EyeCalibration())}
        self.assertFalse(load_eye_tracking_settings(cfg).automatic_recalibration)
        cfg["eye_tracking"]["automatic_recalibration"] = True
        self.assertTrue(load_eye_tracking_settings(cfg).automatic_recalibration)
        cfg["eye_tracking"]["enabled"] = False
        with self.assertRaisesRegex(ValueError, "requires eye_tracking.enabled=true"):
            load_eye_tracking_settings(cfg)
        for value in (1, 0, "true", "false", None):
            cfg["eye_tracking"]["automatic_recalibration"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "must be true or false"):
                load_eye_tracking_settings(cfg)
        settings, gate = self.make_gate(enabled=False)
        self.feed(gate)
        self.assertEqual(gate.status(now_s=1.1), 1)
        self.assertIsNone(gate.recalibration)
        self.assertEqual(settings.calibration.x_offset, 0)

    def test_uses_sample_50_ms_before_actual_acceptance_with_rotated_targets(self):
        for rotation in (0, 90, 180, 270):
            with self.subTest(rotation=rotation):
                settings, gate = self.make_gate(rotation=rotation)
                self.feed(gate, end=1050)
                self.feed(gate, start=1060, end=1120, position=(0.24, -0.14))
                # Actual acceptance is later than the earliest possible dwell.
                self.assertEqual(gate.status(now_s=1.103), None)  # future sample
                self.assertEqual(gate.status(now_s=1.123), 1)
                updated = gate.recalibration.calibration
                self.assertAlmostEqual(updated.x_offset, -0.002)
                self.assertAlmostEqual(updated.y_offset, 0.002)
                self.assertEqual(settings.calibration.x_offset, 0)
                self.assertIsNot(updated, settings.calibration)

    def test_earlier_sample_not_latest_sample_is_used_without_changing_other_terms(self):
        calibration = EyeCalibration(
            x_scale=1, y_scale=1, x_cross=.01, y_cross=.02,
            x_quad_xx=.01, x_quad_xy=.02, x_quad_yy=.03,
            y_quad_xx=.04, y_quad_xy=.05, y_quad_yy=.06,
        )
        settings, gate = self.make_gate(calibration=calibration)
        self.feed(gate, end=1050)
        self.feed(gate, start=1060, position=(0.24, -0.14))
        self.assertEqual(gate.status(now_s=1.103), 1)
        updated = gate.recalibration.calibration
        measured = calibration.map_position(.22, -.12)
        self.assertAlmostEqual(updated.x_offset, .05 * (.2 - measured[0]))
        self.assertAlmostEqual(updated.y_offset, .05 * (-.1 - measured[1]))
        for key, value in asdict(calibration).items():
            if key not in ("x_offset", "y_offset"):
                self.assertEqual(getattr(updated, key), value)

    def test_one_correction_per_dot_even_with_polling_and_refixation(self):
        _, gate = self.make_gate()
        self.feed(gate)
        original = gate.recalibration.calibration
        for _ in range(100):
            self.assertEqual(gate.status(now_s=1.1), 1)
        updated = gate.recalibration.calibration
        self.assertEqual(gate.recalibration.update_count, 1)
        self.assertEqual(original.x_offset, 0)
        self.assertAlmostEqual(updated.x_offset, -.001)
        gate.observe(replace(self.state(1110), valid=False))
        self.feed(gate, start=1120, end=1230)
        self.assertEqual(gate.status(now_s=1.23), 1)
        self.assertIs(gate.recalibration.calibration, updated)
        gate.stop()
        gate.start(((200, -100),), native_size=(1000, 1000), rotation_degrees=0, onset_perf_s=2)
        self.feed(gate, start=2000, end=2110)
        self.assertEqual(gate.status(now_s=2.11), 1)
        self.assertEqual(gate.recalibration.update_count, 2)
        self.assertAlmostEqual(gate.recalibration.calibration.x_offset, -.00195)
        self.assertAlmostEqual(updated.x_offset, -.001)

    def test_invalid_gapped_short_stale_and_contended_fixations_cannot_adjust(self):
        for fault in ("invalid", "nonfinite", "gap", "short", "stale", "contended", "duplicate"):
            with self.subTest(fault=fault):
                _, gate = self.make_gate(criteria=.03 if fault == "short" else .1)
                if fault == "duplicate":
                    for _ in range(100):
                        gate.observe(self.state(1000))
                else:
                    self.feed(gate, end=1030 if fault == "short" else 1100)
                now = 1.03 if fault == "short" else 1.11
                if fault == "invalid":
                    gate.observe(replace(self.state(1110), valid=False))
                elif fault == "nonfinite":
                    gate.observe(self.state(1110, position=(float("nan"), 0)))
                elif fault == "gap":
                    gate.observe(self.state(1200))
                    now = 1.2
                elif fault == "stale":
                    now = 1.2
                if fault == "contended":
                    with gate._lock:
                        self.assertIs(gate.status(now_s=now), FIXATION_UNAVAILABLE)
                else:
                    gate.status(now_s=now)
                self.assertEqual(gate.recalibration.update_count, 0)

    def test_history_is_bounded_and_never_reuses_an_earlier_dot(self):
        _, gate = self.make_gate()
        for index in range(5000):
            gate.observe(replace(self.state(index), timestamp_perf_s=1 + index / 100_000))
        self.assertLessEqual(len(gate.recalibration._history), 256)
        gate.stop()
        self.assertFalse(gate.recalibration._history)
        gate.start(((200, -100),), native_size=(1000, 1000), rotation_degrees=0, onset_perf_s=2)
        self.feed(gate)
        self.assertFalse(gate.recalibration._history)

    def test_close_saves_once_and_gaze_preview_csv_share_updated_mapping(self):
        for stop_error in (None, OSError("STOP failed")):
            with self.subTest(stop_error=stop_error), tempfile.TemporaryDirectory() as tmp:
                settings, gate = self.make_gate()
                settings = replace(settings, calibration_metadata={"custom_metadata": "preserved"})
                tracker = Mock()
                tracker.stop.side_effect = stop_error
                with patch("bin.task_eye_tracking.UARTEyeTracker", return_value=tracker), patch(
                    "bin.eye_tracker_warnings.show_eye_tracker_warning"
                ):
                    logs = SessionLogBundle(
                        output_root=tmp, task_name="active_foraging", config_name="recalibration",
                        eye_tracking_config=settings, on_eye_state=gate.observe,
                        eye_recalibration=gate.recalibration,
                    )
                    session = logs.eye_tracking
                    try:
                        logs.set_eye_tracking_screen((1000, 1000))
                        for ms in range(1000, 1101, 10):
                            state = self.state(ms)
                            session._record_measurement(state)
                            session._update_preview(state)
                        self.assertEqual(gate.status(now_s=1.1), 1)
                        self.assertFalse(logs.calibration_path.exists())
                        state = self.state(1110)
                        tracker.get_state.return_value = state
                        session._record_measurement(state)
                        session._update_preview(state)
                        for actual, expected in zip(session.get_position(), (.219, -.119)):
                            self.assertAlmostEqual(actual, expected)
                        self.assertEqual(read_preview_gaze(session.preview_state, now_s=1.11),
                                         session.get_position())
                        gate.stop()
                    finally:
                        logs.close()
                    saved_bytes = logs.calibration_path.read_bytes()
                    session.close()
                    logs.close()
                    self.assertEqual(logs.calibration_path.read_bytes(), saved_bytes)
                    saved = json.loads(saved_bytes)["eye_tracker_calibration"]
                    self.assertEqual(saved["custom_metadata"], "preserved")
                    self.assertEqual(saved["automatic_recalibration"]["update_count"], 1)
                    loaded = load_eye_tracking_settings({"eye_tracking": {
                        "enabled": True, "calibration_file": str(logs.calibration_path),
                    }})
                    self.assertFalse(loaded.automatic_recalibration)
                    self.assertEqual(loaded.calibration, gate.recalibration.calibration)
                    with (logs.session_dir / "eye_tracking.csv").open() as handle:
                        rows = list(csv.DictReader(handle))
                    self.assertEqual(float(rows[0]["x_screen_px"]), 220)
                    self.assertEqual(float(rows[-1]["x_screen_px"]), 219)
                    tracker.stop.assert_called_once()


if __name__ == "__main__":
    unittest.main()
