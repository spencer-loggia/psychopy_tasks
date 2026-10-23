"""Exercise the shared task API and actual preview loop without rig hardware."""

from dataclasses import replace
import csv
import math
import multiprocessing as mp
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest.mock import Mock, patch

from bin.eye_tracking import (
    EyeCalibration,
    EyeTrackerState,
    UARTEyeTracker,
    UARTEyeTrackerConfig,
)
from bin.eye_tracking_config import EyeTrackingSettings
from bin.logger import EXACT_SESSION_DIR_ENV, SessionLogBundle
from bin import screen
from bin.task_eye_tracking import TaskEyeTrackingSession, read_preview_gaze


class TaskEyeTrackingTests(unittest.TestCase):
    def setUp(self):
        patcher = patch("bin.eye_tracker_warnings.show_eye_tracker_warning")
        self.warning = patcher.start()
        self.addCleanup(patcher.stop)

    def test_main_thread_and_preview_share_calibration_and_reject_invalid_data(self):
        settings = EyeTrackingSettings(
            calibration=EyeCalibration(
                x_scale=0.01, y_scale=-0.02, x_offset=-1, y_offset=1
            )
        )
        tracker = UARTEyeTracker()
        now = [100.0]
        with patch(
            "bin.task_eye_tracking.UARTEyeTracker", return_value=tracker
        ), patch.object(tracker, "start"), patch.object(tracker, "stop"), patch(
            "bin.eye_tracking.time.perf_counter", side_effect=lambda: now[0]
        ):
            session = TaskEyeTrackingSession(settings)
            try:
                self.assertIsNone(session.get_position())

                def publish(state):
                    tracker.filter.state = state
                    session._update_preview(tracker.get_state())

                valid = EyeTrackerState(
                    timestamp_perf_s=100,
                    valid=True,
                    rejection_reason=None,
                    x_smooth_px=120,
                    y_smooth_px=60,
                )
                publish(valid)
                for actual, expected in zip(session.get_position(), (0.2, -0.2)):
                    self.assertAlmostEqual(actual, expected)
                self.assertEqual(
                    read_preview_gaze(session.preview_state), session.get_position()
                )
                self.assertEqual(session.get_state().smoothed_position, (120, 60))
                # Keep out-of-scene coordinates available to future task logic,
                # without drawing a misleading marker on the preview controls.
                publish(replace(valid, x_smooth_px=170))
                self.assertAlmostEqual(session.get_position()[0], 0.7)
                self.assertIsNone(read_preview_gaze(session.preview_state))
                for reason in ("blink", "lost", "uart_error", "service_restarted"):
                    publish(replace(valid, valid=False, rejection_reason=reason))
                    self.assertIsNone(session.get_position())
                    self.assertIsNone(read_preview_gaze(session.preview_state))
                publish(valid)
                # A stalled publisher must expire independently in both readers.
                now[0] += 0.2
                self.assertIsNone(read_preview_gaze(session.preview_state))
                self.assertIsNone(session.get_position())
            finally:
                session.close()
            self.assertIsNone(session.get_position())
            self.assertIsNone(read_preview_gaze(session.preview_state))
            tracker.stop.assert_called_once()

    def test_preview_can_be_disabled_while_main_thread_keeps_gaze(self):
        tracker = Mock()
        tracker.get_state.return_value = EyeTrackerState(valid=True)
        with patch("bin.task_eye_tracking.UARTEyeTracker", return_value=tracker):
            session = TaskEyeTrackingSession(EyeTrackingSettings(show_preview=False))
            try:
                self.assertIsNone(session.preview_state)
                self.assertEqual(session.get_position(), (0, 0))
            finally:
                session.close()

    def test_state_observer_receives_valid_and_invalid_gaze_without_preview(self):
        tracker = Mock()
        observed = []
        with patch("bin.task_eye_tracking.UARTEyeTracker", return_value=tracker) as factory:
            session = TaskEyeTrackingSession(
                EyeTrackingSettings(show_preview=False), on_state=observed.append,
            )
            callback = factory.call_args.kwargs["on_state"]
            valid = EyeTrackerState(timestamp_perf_s=100, frame_sequence=1, valid=True)
            lost = replace(valid, valid=False, rejection_reason="lost")
            try:
                callback(valid)
                callback(lost)
                self.assertEqual(observed, [valid, lost])
            finally:
                session.close()
            callback(valid)
            self.assertEqual(observed, [valid, lost])

    def test_shared_session_lifecycle_for_all_presentation_tasks(self):
        for task in (
            "active_foraging",
            "play_video",
            "match2cue",
            "afc_csc1",
            "afc_trial_sequence",
            "random_image_sequence",
        ):
            with self.subTest(task=task), tempfile.TemporaryDirectory() as tmp:
                logs = SessionLogBundle(
                    output_root=tmp,
                    task_name=task,
                    config_name="eye-test",
                    eye_tracking_config=EyeTrackingSettings(
                        uart=UARTEyeTrackerConfig(simulate=True)
                    ),
                )
                session = logs.eye_tracking
                logs.set_eye_tracking_screen((800, 600), 90)
                try:
                    deadline = time.monotonic() + 1
                    while (
                        read_preview_gaze(logs.eye_tracking_preview) is None
                        and time.monotonic() < deadline
                    ):
                        time.sleep(0.005)
                    self.assertEqual(session.get_position(), (0, 0))
                    self.assertEqual(
                        read_preview_gaze(logs.eye_tracking_preview), (0, 0)
                    )
                finally:
                    logs.close()
                self.assertIsNone(session.get_position())
                self.assertFalse(session.tracker._thread.is_alive())
                self.assertFalse(session.get_state().valid)
                with (logs.session_dir / "eye_tracking.csv").open(newline="") as handle:
                    self.assertTrue(list(csv.DictReader(handle)))
                self.assertFalse(session.recorder._thread.is_alive())

    def test_recording_follows_acquisition_without_flips_in_exact_experiment_block_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            observed = []
            block_dir = Path(tmp).resolve() / "experiment" / "blocks" / "1_active_foraging"
            with patch.dict(os.environ, {EXACT_SESSION_DIR_ENV: str(block_dir)}):
                logs = SessionLogBundle(
                    output_root=block_dir, task_name="active_foraging", config_name="static",
                    eye_tracking_config=EyeTrackingSettings(
                        uart=UARTEyeTrackerConfig(simulate=True), show_preview=False
                    ),
                    on_eye_state=observed.append,
                )
            session = logs.eye_tracking
            try:
                logs.set_eye_tracking_screen((800, 600), 90)
                self.assertEqual(session.recorder.path, block_dir / "eye_tracking.csv")
                # No task loop, window flips, or preview process runs here.
                # Wait across a periodic flush to verify on-disk streaming too.
                time.sleep(1.15)
                with session.recorder.path.open(newline="") as handle:
                    live_rows = list(csv.DictReader(handle))
                self.assertGreater(len(live_rows), 35)
                self.assertGreater(sum(state.valid for state in observed), 35)
            finally:
                logs.close()
            with session.recorder.path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            timestamps = [int(row["task_time_ns"]) for row in rows]
            self.assertEqual(timestamps, sorted(timestamps))
            with (block_dir / "event_log.tsv").open() as handle:
                events = list(csv.DictReader(handle, delimiter="\t"))
            self.assertGreaterEqual(timestamps[0], int(events[0]["task_time_ns"]))
            self.assertLessEqual(timestamps[-1], int(events[-1]["task_time_ns"]))
            self.assertGreaterEqual(timestamps[0], 0)
            self.assertGreater(timestamps[-1] - timestamps[0], 900_000_000)
            actual_hz = (len(rows) - 1) * 1_000_000_000 / (timestamps[-1] - timestamps[0])
            self.assertGreater(actual_hz, 75)
            self.assertLess(actual_hz, 115)
            self.assertTrue(any(
                row["screen_ready"] == "1" and not math.isnan(float(row["x_screen_px"]))
                for row in rows
            ))
            self.assertFalse(session.recorder._thread.is_alive())
            self.assertFalse(session.tracker._thread.is_alive())

    def test_disabled_tracking_creates_no_eye_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            logs = SessionLogBundle(
                output_root=tmp, task_name="active_foraging", config_name="disabled"
            )
            logs.set_eye_tracking_screen((800, 600), 90)
            logs.close()
            self.assertFalse((logs.session_dir / "eye_tracking.csv").exists())

    def test_recording_failure_warns_and_task_continues_through_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            logs = SessionLogBundle(
                output_root=tmp, task_name="active_foraging", config_name="disk-error",
                eye_tracking_config=EyeTrackingSettings(
                    uart=UARTEyeTrackerConfig(simulate=True), show_preview=False
                ),
            )
            session = logs.eye_tracking
            session.recorder._error = OSError("disk full")
            logs.flush()
            logs.flush()
            logs.close()
            self.warning.assert_called_once()
            self.assertIn("disk full", self.warning.call_args.args[0])
            self.assertFalse(session.tracker._thread.is_alive())
            self.assertTrue(session.recorder._file.closed)

    def test_startup_error_remains_available_after_worker_exits(self):
        tracker = UARTEyeTracker()
        with patch(
            "bin.eye_tracking.SerialPacketChannel", side_effect=OSError("no UART")
        ):
            tracker.start()
            tracker.stop()
        self.assertFalse(tracker.get_state().valid)
        self.assertEqual(tracker.get_state().error, "no UART")
        self.warning.assert_called_once()

    def test_csv_start_failure_does_not_abort_task_or_eye_acquisition(self):
        tracker = Mock()
        tracker.get_state.return_value = EyeTrackerState(valid=True)
        with patch("bin.task_eye_tracking.UARTEyeTracker", return_value=tracker), patch(
            "bin.task_eye_tracking.EyeDataRecorder", side_effect=OSError("disk full")
        ):
            session = TaskEyeTrackingSession(EyeTrackingSettings(show_preview=False), output_dir="unused")
            self.assertEqual(session.get_position(), (0, 0))
            session.close()
        self.warning.assert_called_once()
        tracker.stop.assert_called_once()

    def test_static_and_frozen_movie_preview_draw_and_expire_oriented_gaze(self):
        for command_type in ("static_scene", "play_shared_video"):
            with self.subTest(command_type=command_type):
                self._exercise_preview(command_type)

    def _exercise_preview(self, command_type):
        now = [100.0]
        stop = threading.Event()
        shared = mp.get_context("spawn").Array("d", [100, 0.25, -0.25, 1, 0.15])
        dot = Mock(pos=(0, 0))
        marker_positions = []
        dot.draw.side_effect = lambda: marker_positions.append(dot.pos)
        win = Mock()
        flip_times = []

        def flip():
            flip_times.append(now[0])
            if len(flip_times) == 3:
                stop.set()

        win.flip.side_effect = flip
        win._neuro_tasks_screen_placement = "test"

        def wait(seconds):
            now[0] += seconds
            # Fail boundedly if the preview stops drawing after its first frame.
            if now[0] > 101:
                stop.set()

        psychopy = types.ModuleType("psychopy")
        psychopy.core = types.SimpleNamespace(wait=wait)
        mouse = Mock()
        mouse.getPressed.return_value = (False, False, False)
        psychopy.event = types.SimpleNamespace(
            Mouse=Mock(return_value=mouse), getKeys=lambda **_: []
        )

        def stimulus(*args, **kwargs):
            return Mock(pos=kwargs.get("pos", (0, 0)))

        psychopy.visual = types.SimpleNamespace(
            Rect=stimulus,
            TextStim=stimulus,
            ImageStim=stimulus,
            Circle=Mock(return_value=dot),
        )
        payload = {
            "type": command_type,
            "main_size": (800, 600),
            "main_rotation_deg": 90,
            "shared_frame_buffer": {"name": "fake", "maximum_frame_bytes": 100},
            "video_size": (2, 2),
        }
        commands = iter([payload])
        reader = Mock()
        reader.read_latest.return_value = None
        startup = Mock()
        with patch.dict(sys.modules, {"psychopy": psychopy}), patch.object(
            screen, "open_psychopy_window", return_value=win
        ), patch.object(screen, "configure_window_vsync"), patch.object(
            screen,
            "_get_latest_preview_command",
            side_effect=lambda _: next(commands, None),
        ), patch.object(
            screen.time, "perf_counter", side_effect=lambda: now[0]
        ), patch(
            "bin.video_playback.SharedVideoFrameReader", return_value=reader
        ):
            screen._experimenter_preview_process(
                screen_info=screen.ScreenGeometry(
                    index=1, name="test", width=1280, height=1024, x=0, y=0
                ),
                task_label="test",
                start_perf_s=100,
                update_interval_ms=200,
                command_queue=None,
                reward_event=threading.Event(),
                exit_event=threading.Event(),
                mouse_visible=None,
                initial_subject=None,
                initial_current_trial_num=None,
                initial_total_trials=None,
                initial_status_counts=None,
                video_start_perf_shared=types.SimpleNamespace(value=0),
                stop_event=stop,
                startup_sender=startup,
                eye_tracking_shared=shared,
            )
        self.assertTrue(startup.send.call_args.args[0]["ok"])
        self.assertEqual(len(flip_times), 3)
        layout = screen.compute_centered_aspect_fit((1280, 1024), (600, 800))
        self.assertEqual(
            marker_positions,
            [(layout["box_size"][0] * 0.25, layout["box_size"][1] * -0.25)],
        )
        win.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
