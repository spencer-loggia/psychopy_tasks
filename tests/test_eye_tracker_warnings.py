import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from bin.eye_tracker_warnings import EyeTrackerWarnings, _main, show_eye_tracker_warning
from bin.screen import ScreenGeometry


class EyeTrackerWarningTests(unittest.TestCase):
    def test_status_problems_are_combined_and_repeated_dialogs_suppressed(self):
        warnings = EyeTrackerWarnings()
        with patch("bin.eye_tracker_warnings.show_eye_tracker_warning") as show:
            warnings.warn_many({"tracking": "No tracking", "recording": "No recording"})
            warnings.warn("recording", "No recording")
            self.assertEqual(show.call_count, 1)
            self.assertEqual(show.call_args.args[0], "No tracking\n\nNo recording")
            warnings.warn("stop", "Stop unconfirmed")
            self.assertEqual(show.call_count, 2)

    def test_warning_delivery_does_not_wait_for_dialog_process(self):
        entered, release, finished = threading.Event(), threading.Event(), threading.Event()

        def slow_spawn(*args, **kwargs):
            entered.set()
            release.wait(2)
            finished.set()

        with patch("bin.eye_tracker_warnings.subprocess.Popen", side_effect=slow_spawn) as popen, patch("builtins.print"):
            try:
                show_eye_tracker_warning("No eye tracker")
                self.assertTrue(entered.wait(1))
                self.assertFalse(finished.is_set())
                self.assertTrue(popen.call_args.kwargs["start_new_session"])
            finally:
                release.set()
                self.assertTrue(finished.wait(1))

    def test_dialog_is_nonmodal_and_placed_on_experimenter_screen(self):
        root = Mock()
        with patch("tkinter.Tk", return_value=root), patch("tkinter.Label"), patch("tkinter.Button"), patch(
            "bin.screen.resolve_task_screens",
            return_value=(ScreenGeometry(0, 0, 0, 1920, 1080), ScreenGeometry(1, 1920, 0, 800, 480)),
        ):
            _main("No recording")
        root.geometry.assert_called_once_with("+1960+40")
        root.grab_set.assert_not_called()
        root.wait_window.assert_not_called()
        root.mainloop.assert_called_once_with()

    def test_warning_can_be_dismissed_by_enter_or_escape_without_blocking(self):
        for key in ("Return", "KP_Enter", "Escape"):
            with self.subTest(key=key):
                root, button = Mock(), Mock()
                button.invoke.side_effect = root.destroy
                with patch("tkinter.Tk", return_value=root), patch("tkinter.Label"), patch(
                    "tkinter.Button", return_value=button
                ), patch("bin.screen.resolve_task_screens", side_effect=RuntimeError("no display")):
                    _main("No recording")
                bindings = {call.args[0]: call.args[1] for call in button.bind.call_args_list}
                self.assertEqual(bindings[f"<{key}>"](SimpleNamespace(keysym=key)), "break")
                root.destroy.assert_called_once_with()
                root.grab_set.assert_not_called()
                root.wait_window.assert_not_called()


if __name__ == "__main__":
    unittest.main()
