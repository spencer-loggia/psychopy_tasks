import csv
import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from bin.logger import EventCodeLibrary, MessageLogger, SessionClock, SessionLogBundle, load_task_event_definitions
from bin.task_lifecycle import (
    TASK_WINDOW_READY_ENV,
    TASK_WINDOW_RELEASE_ENV,
    signal_task_window_ready,
)


class LoggingSpecTests(unittest.TestCase):
    def test_task_window_ready_signal_is_manager_opt_in(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(signal_task_window_ready())

        with tempfile.TemporaryDirectory() as tmpdir:
            ready_path = Path(tmpdir) / "task-ready"
            with patch.dict(os.environ, {TASK_WINDOW_READY_ENV: str(ready_path)}):
                self.assertTrue(signal_task_window_ready())
            self.assertEqual(ready_path.read_text(encoding="utf-8"), "ready\n")

    def test_task_window_signal_waits_for_a_fresh_launcher_release(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ready_path = Path(tmpdir) / "task-ready"
            release_path = Path(tmpdir) / "task-released"
            release_path.write_text("stale\n", encoding="utf-8")
            finished = threading.Event()

            def signal() -> None:
                signal_task_window_ready()
                finished.set()

            with patch.dict(
                os.environ,
                {
                    TASK_WINDOW_READY_ENV: str(ready_path),
                    TASK_WINDOW_RELEASE_ENV: str(release_path),
                },
            ):
                thread = threading.Thread(target=signal)
                thread.start()
                deadline = time.monotonic() + 1.0
                while not ready_path.is_file() and time.monotonic() < deadline:
                    time.sleep(0.001)
                self.assertTrue(ready_path.is_file())
                self.assertFalse(finished.is_set())
                release_path.write_text("released\n", encoding="utf-8")
                thread.join(timeout=1.0)

            self.assertTrue(finished.is_set())

    def test_task_window_signal_times_out_without_launcher_release(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ready_path = Path(tmpdir) / "task-ready"
            release_path = Path(tmpdir) / "task-released"
            with (
                patch.dict(
                    os.environ,
                    {
                        TASK_WINDOW_READY_ENV: str(ready_path),
                        TASK_WINDOW_RELEASE_ENV: str(release_path),
                    },
                ),
                patch("bin.task_lifecycle.time.monotonic", side_effect=[0.0, 11.0]),
            ):
                with self.assertRaisesRegex(RuntimeError, "launcher to uncover"):
                    signal_task_window_ready()

            self.assertEqual(ready_path.read_text(encoding="utf-8"), "ready\n")

    def test_session_bundle_exposes_optional_calibration_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle = SessionLogBundle(
                output_root=tmpdir,
                task_name="active_foraging",
                config_name="test",
            )
            try:
                self.assertEqual(bundle.calibration_path, bundle.session_dir / "calibration.json")
                self.assertFalse(bundle.calibration_path.exists())
            finally:
                bundle.close()

    def test_message_logger_rejects_unknown_levels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MessageLogger(tmpdir, session_clock=SessionClock())
            try:
                with self.assertRaises(ValueError):
                    logger.log("DEBUG", "not allowed")
            finally:
                logger.close()

    def test_task_backed_event_logger_rejects_unknown_events(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle = SessionLogBundle(
                output_root=tmpdir,
                task_name="active_foraging",
                config_name="test",
            )
            try:
                with self.assertRaises(KeyError):
                    bundle.event_logger.log_frame_flip(
                        trial_num=1,
                        event="options_onn",
                        timestamp_perf_s=bundle.session_clock.start_perf_s,
                    )
            finally:
                bundle.close()

    def test_event_code_library_exports_only_used_events(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle = SessionLogBundle(
                output_root=tmpdir,
                task_name="active_foraging",
                config_name="test",
            )
            try:
                bundle.event_logger.log_frame_flip(
                    trial_num=1,
                    event="options_on",
                    timestamp_perf_s=bundle.session_clock.start_perf_s,
                    requested_duration=0.5,
                )
                bundle.message_logger.log("INFO", "session test")
            finally:
                bundle.close()

            payload = json.loads((bundle.session_dir / "event_code_library.json").read_text(encoding="utf-8"))
            self.assertEqual(
                payload,
                {
                    "111": {
                        "description": "Stimuli for all options became visible simultaneously.",
                        "event": "options_on",
                        "event_type": "frame_flip",
                    }
                },
            )

    def test_shared_library_expands_active_foraging_option_templates(self):
        definitions, event_patterns = load_task_event_definitions("active_foraging")
        library = EventCodeLibrary(definitions, event_patterns=event_patterns)

        option_dot = library.ensure("option_2_dot", "frame_flip")
        option_on = library.ensure("option_2_on", "frame_flip")

        self.assertEqual(option_dot.code, 1002)
        self.assertEqual(option_dot.description, "Dot cue for option 2 became visible.")
        self.assertEqual(option_on.code, 1102)
        self.assertEqual(option_on.description, "Stimulus for option 2 became visible.")

    def test_event_log_uses_trial_num_without_stimulus_counter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle = SessionLogBundle(
                output_root=tmpdir,
                task_name="random_image_sequence",
                config_name="test",
            )
            try:
                bundle.event_logger.log_frame_flip(
                    trial_num=1,
                    event="stimulus_on",
                    timestamp_perf_s=bundle.session_clock.start_perf_s,
                    requested_duration=0.5,
                )
            finally:
                bundle.close()

            with (bundle.session_dir / "event_log.tsv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))

            self.assertEqual(len(rows), 1)
            self.assertEqual(
                list(rows[0]),
                [
                    "trial_num",
                    "time_since_session_start",
                    "event",
                    "event_code",
                    "event_type",
                    "requested_duration",
                ],
            )
            self.assertEqual(rows[0]["trial_num"], "1")

    def test_trial_sequence_uses_trial_cue_event(self):
        definitions, _ = load_task_event_definitions("afc_trial_sequence")

        self.assertIn("trial_cue", definitions)
        self.assertEqual(definitions["trial_cue"].code, 104)

    def test_play_video_registers_frame_locked_sync_edges(self):
        definitions, _ = load_task_event_definitions("play_video")

        self.assertEqual(definitions["video_sync_signal_on"].code, 302)
        self.assertEqual(definitions["video_sync_signal_off"].code, 303)
        self.assertEqual(definitions["video_sync_signal_on"].default_type, "signal")


if __name__ == "__main__":
    unittest.main()
