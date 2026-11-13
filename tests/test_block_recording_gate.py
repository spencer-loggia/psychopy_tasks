"""Launcher ordering: acquisition precedes framing and outlives the task footer."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from interface.experiment_manager import PreparedBlock
from interface.touch_interface import TouchInterfaceApp
from video.recorder import RecordingError


class BlockRecordingGateTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        (self.directory / "video.json").write_text("{}")
        self.app = object.__new__(TouchInterfaceApp)
        self.app.cfg = {"record_experiment_video": True, "video_config": "video.json"}
        self.app.config_dir = self.directory
        self.app.working_dir = self.directory
        self.app.python_cmd = "python"
        self.app.status_var = Mock()
        self.app.idle_guard = None
        self.app.experiment = Mock()
        self.app.experiment.subprocess_environment.return_value = {}
        self.app._hide_interface_for_process = Mock()
        self.app._restore_interface_after_process = Mock()
        self.app._confirm_external_recording_start = Mock(return_value=True)
        self.app.task_active = False
        self.app.cleanup_active = False

    def block(self, number=1, *, sync=True):
        directory = self.directory / str(number)
        directory.mkdir()
        config = directory / "config.json"
        config.write_text(json.dumps({"sync_bar": {"enabled": sync}}))
        return PreparedBlock(number, "demo", self.directory / "demo.py", config,
                             directory, f"b0000000-0000-4000-8000-{number:012d}")

    def test_acquisition_precedes_task_and_video_outlives_footer(self):
        block = self.block()
        events = []
        recording = Mock()
        recording.wait_until_ready.side_effect = lambda: events.append("video ready")
        recording.stop.side_effect = lambda: events.append("video stopped")
        self.app.experiment.finish_block.side_effect = lambda _: events.append("block finished")
        with patch("interface.touch_interface.record_video", return_value=recording), \
             patch("interface.touch_interface.subprocess.Popen", side_effect=lambda *a, **k: events.append("task started") or Mock()), \
             patch("interface.touch_interface.wait_for_task_process", side_effect=lambda *a, **k: events.append("task/footer completed") or 0):
            self.app._run_block(block)
        self.assertEqual(events, ["video ready", "task started", "task/footer completed",
                                  "video stopped", "block finished"])

    def test_unready_camera_prevents_task_and_still_stops_recording(self):
        block = self.block()
        recording = Mock()
        recording.wait_until_ready.side_effect = RecordingError("No encoded camera picture")
        with patch("interface.touch_interface.record_video", return_value=recording), \
             patch("interface.touch_interface.subprocess.Popen") as popen:
            with self.assertRaisesRegex(RecordingError, "No encoded"):
                self.app._run_block(block)
        popen.assert_not_called()
        recording.stop.assert_called_once_with()
        self.app._restore_interface_after_process.assert_called_once_with()
        self.app.experiment.finish_block.assert_called_once_with(block)

    def test_disabled_sync_preserves_existing_nonblocking_camera_start(self):
        block = self.block(sync=False)
        recording = Mock()
        with patch("interface.touch_interface.record_video", return_value=recording), \
             patch("interface.touch_interface.subprocess.Popen"), \
             patch("interface.touch_interface.wait_for_task_process", return_value=0):
            self.app._run_block(block)
        recording.wait_until_ready.assert_not_called()
        recording.stop.assert_called_once_with()

    def test_loop_stays_locked_through_footer_and_recording_cleanup(self):
        blocks = [self.block(1), self.block(2)]
        self.app.experiment.prepare_block.side_effect = blocks
        events = []
        recording = Mock()
        recording.wait_until_ready.side_effect = lambda: events.append("ready")
        recording.stop.side_effect = lambda: events.append("video stopped")
        self.app.experiment.finish_block.side_effect = lambda block: events.append(f"finished {block.block_num}")

        def await_footer(*args, **kwargs):
            self.assertTrue(self.app.task_active)
            prepared = self.app.experiment.prepare_block.call_count
            # A rapid button press during the footer must not start another block.
            self.app._run_task("rapid selection", {"launch": "other.py"})
            self.assertEqual(self.app.experiment.prepare_block.call_count, prepared)
            events.append(f"footer {prepared}")
            return 0

        with patch("interface.touch_interface.record_video", return_value=recording), \
             patch("interface.touch_interface.subprocess.Popen", side_effect=lambda *a, **k: events.append("task started") or Mock()), \
             patch("interface.touch_interface.wait_for_task_process", side_effect=await_footer), \
             patch("interface.touch_interface.messagebox.showerror") as error:
            self.app._run_task("loop", {"launch": ["demo.py"], "config": ["demo.json"],
                                        "order_mode": "sequential", "n_iters": 2})
        error.assert_not_called()
        self.assertEqual(events, ["ready", "task started", "footer 1", "video stopped", "finished 1",
                                  "ready", "task started", "footer 2", "video stopped", "finished 2"])
        self.assertFalse(self.app.task_active)

    def test_incomplete_footer_stops_loop_even_if_task_returns_success(self):
        first, second = self.block(1), self.block(2)
        self.app.experiment.prepare_block.side_effect = [first, second]
        (first.output_dir / "sync_protocol.json").write_text(json.dumps({"end_complete": False}))
        recording = Mock()
        with patch("interface.touch_interface.record_video", return_value=recording), \
             patch("interface.touch_interface.subprocess.Popen") as popen, \
             patch("interface.touch_interface.wait_for_task_process", return_value=0), \
             patch("interface.touch_interface.messagebox.showerror") as error:
            self.app._run_task("loop", {"launch": ["demo.py"], "config": ["demo.json"],
                                        "order_mode": "sequential", "n_iters": 2})
        self.assertEqual(self.app.experiment.prepare_block.call_count, 1)
        popen.assert_called_once()
        recording.stop.assert_called_once_with()
        error.assert_called_once()
        self.assertIn("footer did not complete", error.call_args.args[1])
        self.assertFalse(self.app.task_active)


if __name__ == "__main__":
    unittest.main()
