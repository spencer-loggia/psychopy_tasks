"""Verify recording time, calibration/rotation, invalid rows and writer failures."""

import csv
from dataclasses import replace
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from bin.eye_tracking import EyeCalibration, EyeTrackerState
from bin.eye_tracking_log import EyeDataRecorder
from bin.screen import software_stimulus_rotation


class EyeDataRecorderTests(unittest.TestCase):
    def record(
        self, states, *, geometry=((800, 600), 0),
        writer_error=None, calibration=None,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "eye_tracking.csv"
            recorder = EyeDataRecorder(
                path, calibration=calibration or EyeCalibration(
                    x_scale=0.01, y_scale=-0.02, x_offset=-1, y_offset=1
                ),
            )
            if geometry is not None:
                recorder.set_screen_geometry(*geometry)
            if writer_error is not None:
                recorder._writer = Mock()
                recorder._writer.writerow.side_effect = writer_error
            for state in states:
                recorder.submit(state)
            try:
                recorder.close()
            except RuntimeError:
                if writer_error is None:
                    raise
            with path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
        self.assertTrue(recorder._file.closed)
        return rows, recorder, None

    @staticmethod
    def valid_state():
        return EyeTrackerState(
            timestamp_perf_s=99.9, task_time_ns=900_000_001,
            camera_time_ns=1234567890123456789, frame_sequence=7,
            x_smooth_px=120, y_smooth_px=60, pupil_diameter_px=12.5,
            valid=True, rejection_reason=None,
        )

    def test_calibrated_screen_pixels_match_stimulus_coordinates_for_all_rotations(self):
        # Calibration maps (120, 60) camera pixels to (0.2, -0.2) of the
        # subject-oriented screen. Expected coordinates are native stimulus
        # coordinates, including the rig's normal-output clockwise turn.
        for rotation, expected in (
            (0, (160, -120)), (90, (-160, -120)),
            (180, (-160, 120)), (270, (160, 120)),
        ):
            with self.subTest(rotation=rotation):
                rows, recorder, _ = self.record(
                    [self.valid_state()], geometry=((800, 600), rotation)
                )
                recorder.check_error()
                row = rows[0]
                self.assertEqual((float(row["x_screen_px"]), float(row["y_screen_px"])), expected)
                self.assertEqual(float(row["pupil_diameter_px"]), 12.5)
                self.assertEqual(int(row["task_time_ns"]), 900_000_001)
                self.assertEqual(int(row["camera_time_ns"]), 1234567890123456789)
                self.assertEqual(row["screen_ready"], "1")

    def test_normal_screen_subject_up_and_left_use_native_psychopy_axes(self):
        rotation = software_stimulus_rotation("normal")
        self.assertEqual(rotation, 90)
        # The default test calibration centers camera (100, 50). Moving the
        # subject's gaze up decreases camera y; left decreases camera x.
        center = replace(self.valid_state(), x_smooth_px=100, y_smooth_px=50)
        up = replace(center, y_smooth_px=45, frame_sequence=8)
        left = replace(center, x_smooth_px=90, frame_sequence=9)
        rows, _, _ = self.record([center, up, left], geometry=((800, 600), rotation))
        expected_screen_positions = ((400, 300), (480, 300), (400, 240))
        for row, (screen_x, screen_y) in zip(rows, expected_screen_positions):
            self.assertAlmostEqual(float(row["x_screen_px"]), screen_x - 800 / 2)
            self.assertAlmostEqual(float(row["y_screen_px"]), 600 / 2 - screen_y)

    def test_full_affine_calibration_is_saved_in_native_psychopy_coordinates(self):
        calibration = EyeCalibration(
            x_scale=0.002, y_scale=-0.004, x_cross=0.003, y_cross=-0.001,
            x_offset=-0.2, y_offset=0.5,
        )
        # Camera (120,60) maps to subject fractions (.22,.14). The native
        # 800x600 screen's subject scene is 600x800, giving native screen pixel
        # (512,432), or PsychoPy (112,-132), after the one clockwise turn.
        rows, _, _ = self.record(
            [self.valid_state()],
            geometry=((800, 600), software_stimulus_rotation("normal")),
            calibration=calibration,
        )
        self.assertAlmostEqual(float(rows[0]["x_screen_px"]), 512 - 800 / 2)
        self.assertAlmostEqual(float(rows[0]["y_screen_px"]), 600 / 2 - 432)

    def test_invalid_rows_use_nan_instead_of_last_valid_gaze_or_zero_pupil(self):
        valid = self.valid_state()
        reasons = ("blink", "lost", "stale", "uart_error", "service_restarted")
        states = [valid] + [replace(valid, valid=False, rejection_reason=r, frame_sequence=i + 8,
                          task_time_ns=900_000_001 + (i+1)*20_000_000) for i, r in enumerate(reasons)]
        rows, recorder, _ = self.record(states)
        recorder.check_error()
        self.assertEqual(len(rows), len(states))
        for row, reason in zip(rows[1:], reasons):
            for field in ("x_screen_px", "y_screen_px", "pupil_diameter_px"):
                self.assertTrue(math.isnan(float(row[field])))
            self.assertEqual(row["eye_valid"], "0")
            self.assertEqual(row["rejection_reason"], reason)
        self.assertEqual([int(row["task_time_ns"]) for row in rows],
                         [900_000_001 + i*20_000_000 for i in range(len(states))])

    def test_repeated_source_frames_are_written_once_with_original_times(self):
        rows, _, _ = self.record([self.valid_state()] * 4)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["task_time_ns"], "900000001")
        self.assertEqual(rows[0]["camera_time_ns"], "1234567890123456789")
        self.assertNotIn("sample_index", rows[0])
        self.assertNotIn("timestamp_ms", rows[0])

    def test_startup_without_a_camera_measurement_does_not_create_rows(self):
        rows, _, _ = self.record([EyeTrackerState(), self.valid_state()], geometry=None)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["screen_ready"], "0")
        self.assertTrue(math.isnan(float(rows[0]["x_screen_px"])))
        self.assertTrue(math.isnan(float(rows[0]["y_screen_px"])))
        self.assertEqual(float(rows[0]["pupil_diameter_px"]), 12.5)

    def test_queue_drains_acquisition_snapshots_and_rejects_post_close_samples(self):
        with tempfile.TemporaryDirectory() as tmp, patch("threading.Thread.start"):
            path = Path(tmp) / "eye.csv"
            calibration = EyeCalibration(x_scale=0.01, y_scale=-0.02, x_offset=-1, y_offset=1)
            recorder = EyeDataRecorder(path, calibration=calibration)
            recorder.set_screen_geometry((800, 600))
            first = self.valid_state()
            second = replace(first, frame_sequence=8, camera_time_ns=first.camera_time_ns + 20_000_000,
                             task_time_ns=first.task_time_ns + 20_000_000)
            recorder.submit(first)
            recorder.submit(second)
            # Simulate disk backlog while both geometry and calibration change.
            recorder.set_screen_geometry((1600, 1200))
            calibration.x_offset = 99
            recorder.stop_accepting()
            recorder.submit(replace(second, frame_sequence=9))
            recorder._run()
            with path.open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual([int(row["task_time_ns"]) for row in rows],
                             [first.task_time_ns, second.task_time_ns])
            self.assertEqual([float(row["x_screen_px"]) for row in rows], [160, 160])

    def test_full_queue_reports_incomplete_recording_without_blocking_producer(self):
        with tempfile.TemporaryDirectory() as tmp, patch("threading.Thread.start"):
            recorder = EyeDataRecorder(Path(tmp) / "eye.csv", calibration=EyeCalibration(), queue_size=1)
            recorder.submit(self.valid_state())
            recorder.submit(replace(self.valid_state(), frame_sequence=8))
            recorder.stop_accepting()
            recorder._run()
            with self.assertRaisesRegex(RuntimeError, "queue overflow"):
                recorder.check_error()

    def test_offscreen_gaze_is_recorded_without_clamping(self):
        rows, _, _ = self.record([replace(self.valid_state(), x_smooth_px=200)])
        self.assertEqual(float(rows[0]["x_screen_px"]), 800)

    def test_disk_error_is_reported_and_file_closed(self):
        _, recorder, _ = self.record([self.valid_state()], writer_error=OSError("disk full"))
        with self.assertRaisesRegex(RuntimeError, "Eye CSV recording failed: disk full"):
            recorder.check_error()


if __name__ == "__main__":
    unittest.main()
