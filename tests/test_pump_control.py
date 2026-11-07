"""Exercise the common numeric editor and its process-safe pending setting."""
import multiprocessing as mp
import types
import unittest
from unittest.mock import Mock

from bin.pump_control import PumpPulseControl, PumpPulseEditor
from tests.test_eye_calibration_preview import Stimulus, TextStim


def request_from_preview(control):
    control.request(0.375)


class PumpControlTests(unittest.TestCase):
    def editor(self, **kwargs):
        control = PumpPulseControl(0.25, **kwargs)
        editor = PumpPulseEditor(Mock(), control, visual=types.SimpleNamespace(Rect=Stimulus, TextStim=TextStim))
        editor.draw((100, 100))
        self.assertFalse(editor.click((0, 0)))
        self.assertTrue(editor.click((100, 100)))
        return control, editor

    def test_enter_queues_new_seconds_and_applies_only_at_boundary(self):
        control, editor = self.editor()
        self.assertEqual(editor.handle_keys(["0", "period", "3", "7", "5", "return"]), [])
        self.assertFalse(editor.editing)
        self.assertEqual(control.snapshot(), (0.25, 0.375))
        editor.draw((100, 100))
        self.assertIn("Pump: 0.25 s", editor.label.text)
        self.assertIn("Next: 0.375 s", editor.label.text)
        self.assertEqual(control.apply_pending(), 0.375)
        editor.draw((100, 100))
        self.assertEqual(editor.label.text, "Pump: 0.375 s")

    def test_fixation_pixel_radius_uses_shared_button_and_keyboard_editor(self):
        control = PumpPulseControl(0.25, fixation_window_radius=150)
        self.addCleanup(control.settings.close, Mock())
        editor = PumpPulseEditor(
            Mock(), control, visual=types.SimpleNamespace(Rect=Stimulus, TextStim=TextStim),
        )
        editor.layout_strip((800, 600))
        editor.draw()
        self.assertEqual(editor.labels["fixation_window_radius"].text, "Fix radius: 150 px")
        editor.handle_keys(["tab", "tab", "enter", "2", "2", "5", "enter"])
        self.assertEqual(control.settings.snapshot()["fixation_window_radius"], (150, 225))
        editor.draw()
        self.assertIn("Next: 225 px", editor.labels["fixation_window_radius"].text)
        control.apply_pending()
        editor.draw()
        self.assertEqual(editor.labels["fixation_window_radius"].text, "Fix radius: 225 px")
        self.assertTrue(editor.click(editor.buttons["fixation_window_radius"].pos))
        editor.handle_keys(["0", "enter"])
        self.assertTrue(editor.editing)
        self.assertTrue(editor.error)
        self.assertEqual(control.settings.snapshot()["fixation_window_radius"], (225, 225))
        editor.handle_keys(["escape"])
        self.assertEqual(control.snapshot(), (0.25, 0.25))
        for invalid in (-1, float("nan"), float("inf")):
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                control.settings.request("fixation_window_radius", invalid)

    def test_escape_cancels_edit_without_exiting_or_changing_pending_value(self):
        control, editor = self.editor()
        control.request(0.3)
        self.assertEqual(editor.handle_keys(["9", "escape", "r"]), [])
        self.assertFalse(editor.editing)
        self.assertEqual(control.snapshot(), (0.25, 0.3))
        self.assertEqual(editor.handle_keys(["escape", "r"]), ["escape", "r"])

    def test_edit_swallows_reward_keys_and_clicks_outside_popup(self):
        control, editor = self.editor()
        self.assertEqual(editor.handle_keys(["r", "z", "a", "v"]), [])
        self.assertTrue(editor.click((-500, -500)))
        self.assertEqual(control.snapshot(), (0.25, 0.25))

    def test_keypad_backspace_and_invalid_value_correction(self):
        control, editor = self.editor(allow_zero=False, maximum_s=1)
        editor.handle_keys(["backspace", "num_enter"])
        self.assertTrue(editor.editing)
        self.assertTrue(editor.error)
        editor.handle_keys(["num_0", "num_decimal", "num_5", "num_6", "backspace", "num_enter"])
        self.assertFalse(editor.editing)
        self.assertEqual(control.snapshot(), (0.25, 0.5))

    def test_invalid_requests_leave_current_and_pending_unchanged(self):
        control = PumpPulseControl(0.25, allow_zero=False, maximum_s=1)
        for invalid in ("", "no", -0.1, 0, 1, 1.5, float("nan"), float("inf")):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                control.request(invalid)
            self.assertEqual(control.snapshot(), (0.25, 0.25))
        control = PumpPulseControl(0.25)
        control.request(0)
        self.assertEqual(control.apply_pending(), 0)

    def test_editor_keeps_invalid_entry_open_with_error(self):
        control, editor = self.editor(maximum_s=1)
        editor.handle_keys(["1", "enter"])
        self.assertTrue(editor.editing)
        self.assertIn("less than 1", editor.error)
        self.assertEqual(control.snapshot(), (0.25, 0.25))
        editor.handle_keys(["escape"])
        self.assertFalse(editor.editing)

    def test_confirming_unchanged_value_preserves_full_precision(self):
        control, editor = self.editor()
        editor.handle_keys(["escape"])
        control.request(0.123456789)
        control.apply_pending()
        editor.click((100, 100))
        editor.handle_keys(["return"])
        self.assertEqual(control.snapshot(), (0.123456789, 0.123456789))

    def test_preview_process_request_survives_without_scene_messages(self):
        control = PumpPulseControl(0.25)
        child = mp.get_context("spawn").Process(target=request_from_preview, args=(control,))
        child.start()
        child.join(timeout=10)
        if child.is_alive():
            child.terminate()
            child.join()
            self.fail("preview request did not complete")
        self.assertEqual(child.exitcode, 0)
        self.assertEqual(control.snapshot(), (0.25, 0.375))
        self.assertEqual(control.apply_pending(), 0.375)
