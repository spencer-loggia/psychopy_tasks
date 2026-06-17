import json
import tempfile
import unittest

from bin.logger import EventCodeLibrary, MessageLogger, SessionClock, SessionLogBundle, load_task_event_definitions


class LoggingSpecTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
