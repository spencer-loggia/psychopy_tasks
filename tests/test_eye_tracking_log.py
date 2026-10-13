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


class ScheduledStop:
    """Advance a test clock at each recording deadline without real sleeps."""

    def __init__(self, clock):
        self.clock = clock
        self.stopped = False
        self.waits = []

    def is_set(self):
        return self.stopped

    def set(self):
        self.stopped = True

    def wait(self, duration):
        self.waits.append(duration)
        self.clock[0] += duration


class EyeDataRecorderTests(unittest.TestCase):
    def record(self, states, *, geometry=((800, 600), 0), delays=None, writer_error=None):
        clock = [100.0]
        stop = ScheduledStop(clock)
        index = 0

        def get_state():
            nonlocal index
            if delays:
                clock[0] += delays[index]
            state = states[index]
            index += 1
            if index == len(states):
                stop.set()
            return state

        with tempfile.TemporaryDirectory() as tmp, patch("threading.Thread.start"):
            path = Path(tmp) / "eye_tracking.csv"
            recorder = EyeDataRecorder(
                path, get_state=get_state,
                calibration=EyeCalibration(
                    x_scale=0.01, y_scale=-0.02, x_offset=-1, y_offset=1
                ),
                start_perf_s=99.0, rate_hz=50, clock=lambda: clock[0],
            )
            recorder._stop = stop
            if geometry is not None:
                recorder.set_screen_geometry(*geometry)
            if writer_error is not None:
                recorder._writer = Mock()
                recorder._writer.writerow.side_effect = writer_error
            recorder._run()
            with path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
        self.assertTrue(recorder._file.closed)
        return rows, recorder, stop

    @staticmethod
    def valid_state():
        return EyeTrackerState(
            timestamp_perf_s=99.9, sample_index=42, frame_sequence=7,
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
                self.assertEqual(float(row["timestamp_ms"]), 1000)
                self.assertEqual(float(row["sample_timestamp_ms"]), 900)
                self.assertEqual(row["screen_ready"], "1")

    def test_invalid_rows_use_nan_instead_of_last_valid_gaze_or_zero_pupil(self):
        valid = self.valid_state()
        reasons = ("blink", "lost", "stale", "uart_error", "service_restarted")
        states = [valid] + [replace(valid, valid=False, rejection_reason=r) for r in reasons]
        rows, recorder, _ = self.record(states)
        recorder.check_error()
        self.assertEqual(len(rows), len(states))
        for row, reason in zip(rows[1:], reasons):
            for field in ("x_screen_px", "y_screen_px", "pupil_diameter_px"):
                self.assertTrue(math.isnan(float(row[field])))
            self.assertEqual(row["eye_valid"], "0")
            self.assertEqual(row["rejection_reason"], reason)
        self.assertEqual([float(row["timestamp_ms"]) for row in rows], [1000, 1020, 1040, 1060, 1080, 1100])

    def test_repeated_source_frames_are_retained_with_distinct_observation_times(self):
        rows, _, _ = self.record([self.valid_state()] * 4)
        self.assertEqual([float(row["timestamp_ms"]) for row in rows], [1000, 1020, 1040, 1060])
        self.assertEqual({row["frame_sequence"] for row in rows}, {"7"})
        self.assertEqual({row["sample_index"] for row in rows}, {"42"})
        self.assertEqual({float(row["sample_timestamp_ms"]) for row in rows}, {900})

    def test_slow_observation_skips_missed_ticks_without_fabricated_catchup_rows(self):
        rows, _, _ = self.record([self.valid_state()] * 3, delays=[0, 0.08, 0])
        self.assertEqual([float(row["timestamp_ms"]) for row in rows], [1000, 1100, 1120])

    def test_startup_and_missing_geometry_are_explicit(self):
        rows, _, _ = self.record([EyeTrackerState(), self.valid_state()], geometry=None)
        self.assertEqual(rows[0]["rejection_reason"], "waiting_for_sample")
        self.assertTrue(math.isnan(float(rows[0]["sample_timestamp_ms"])))
        for row in rows:
            self.assertEqual(row["screen_ready"], "0")
            self.assertTrue(math.isnan(float(row["x_screen_px"])))
            self.assertTrue(math.isnan(float(row["y_screen_px"])))
        self.assertEqual(float(rows[1]["pupil_diameter_px"]), 12.5)

    def test_offscreen_gaze_is_recorded_without_clamping(self):
        rows, _, _ = self.record([replace(self.valid_state(), x_smooth_px=200)])
        self.assertEqual(float(rows[0]["x_screen_px"]), 800)

    def test_disk_error_is_reported_and_file_closed(self):
        _, recorder, _ = self.record([self.valid_state()], writer_error=OSError("disk full"))
        with self.assertRaisesRegex(RuntimeError, "Eye CSV recording failed: disk full"):
            recorder.check_error()


if __name__ == "__main__":
    unittest.main()
