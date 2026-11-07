"""Shared preview editors remain compact and consume all edit-only input."""
import math
import queue
import sys
import types
import unittest
from unittest.mock import Mock, patch

from bin.live_settings_ui import SETTINGS_STRIP_HEIGHT, SettingsPanel
from bin.screen import ExperimenterPreview, compute_preview_aspect_fit
from tests.test_eye_calibration_preview import NativeLabel, Stimulus, TextStim


class SettingsModel:
    boundary = "trial"
    specs = tuple(
        types.SimpleNamespace(name=name, label=label, units=units, title=title)
        for name, label, units, title in (
            ("pump_pulse_time", "Pump", "s", "Pump pulse duration (seconds)"),
            ("fixation_reward_interval", "Fix interval", "s", "Fixation reward interval (seconds)"),
            ("fixation_radius", "Fix radius", "fraction", "Fixation radius (screen fraction)"),
        )
    )

    def __init__(self):
        self.values = {spec.name: (0.25, 0.25) for spec in self.specs}

    def snapshot(self):
        return dict(self.values)

    def request(self, name, value):
        value = float(value)
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Enter a finite positive number")
        self.values[name] = self.values[name][0], value


class LiveSettingsUITests(unittest.TestCase):
    def make_panel(self):
        settings = SettingsModel()
        panel = SettingsPanel(
            Mock(), settings, visual=types.SimpleNamespace(Rect=Stimulus, TextStim=TextStim),
        )
        return settings, panel

    def assert_disjoint(self, rect, scene):
        self.assertTrue(
            abs(rect.pos[0] - scene["box_center"][0]) >= (rect.width + scene["box_size"][0]) / 2
            or abs(rect.pos[1] - scene["box_center"][1]) >= (rect.height + scene["box_size"][1]) / 2
        )

    def test_buttons_and_open_popup_fit_strip_without_covering_subject_scene(self):
        for canvas in ((800, 600), (1920, 1080), (1080, 1920)):
            for subject in ((1920, 1080), (1080, 1920), canvas):
                _, panel = self.make_panel()
                panel.layout_strip(canvas)
                scene = compute_preview_aspect_fit(canvas, subject, settings_height=SETTINGS_STRIP_HEIGHT)
                self.assertAlmostEqual(scene["box_size"][0] / scene["box_size"][1], subject[0] / subject[1])
                for rect in (*panel.buttons.values(), panel.panel):
                    self.assert_disjoint(rect, scene)
                    self.assertLessEqual(abs(rect.pos[0]) + rect.width / 2, canvas[0] / 2)
                    self.assertLessEqual(abs(rect.pos[1]) + rect.height / 2, canvas[1] / 2)
                buttons = list(panel.buttons.values())
                for first, second in zip(buttons, buttons[1:]):
                    self.assertLess(first.pos[0] + first.width / 2, second.pos[0] - second.width / 2)

    def test_one_editor_routes_each_value_and_consumes_entire_keyboard_batch(self):
        settings, panel = self.make_panel()
        panel.layout_strip((800, 600))
        for spec in settings.specs:
            self.assertTrue(panel.click(panel.buttons[spec.name].pos))
            self.assertTrue(panel.click((-400, 300)))
            self.assertEqual(panel.handle_keys(["num_0", "num_decimal", "num_5", "return", "r", "escape"]), [])
            self.assertFalse(panel.editing)
            self.assertEqual(settings.snapshot()[spec.name], (0.25, 0.5))
        self.assertEqual(panel.handle_keys(["escape", "r"]), ["escape", "r"])

    def test_tab_cycles_only_settings_and_enter_opens_focused_value(self):
        settings, panel = self.make_panel()
        self.assertEqual(panel.handle_keys(["enter", "r", "escape"]), ["enter", "r", "escape"])
        for index in (0, 1, 2, 0):
            self.assertEqual(panel.handle_keys(["tab"]), [])
            panel.draw()
            self.assertEqual(panel.focused_index, index)
            self.assertEqual(panel.buttons[panel.specs[index].name].lineWidth, 3)
        self.assertEqual(panel.handle_keys(["enter", "0", "period", "5", "num_enter", "r"]), [])
        self.assertEqual(settings.snapshot()[panel.specs[0].name], (0.25, 0.5))
        self.assertFalse(panel.editing)
        panel.handle_keys(["tab", "return"])
        self.assertEqual(panel.active_spec, panel.specs[1])

    def make_keyboard_bridge(self):
        preview = object.__new__(ExperimenterPreview)
        preview._settings_keys_queue = queue.Queue()
        preview._settings_keyboard_state = [0, 0]
        preview._settings_keyboard_sequence = 0
        preview._settings_pending_edit = False
        pending = []

        def get_keys(keyList=None, timeStamped=False):
            selected = [item for item in pending if keyList is None or item[0] in keyList]
            pending[:] = [item for item in pending if item not in selected]
            return selected if timeStamped else [item[0] for item in selected]

        psychopy = types.ModuleType("psychopy")
        psychopy.event = types.SimpleNamespace(getKeys=get_keys)
        return preview, pending, patch.dict(sys.modules, {"psychopy": psychopy})

    def test_subject_window_forwards_ordered_complete_editor_batch(self):
        settings, panel = self.make_panel()
        preview, pending, installed = self.make_keyboard_bridge()
        pending.extend((key, index) for index, key in enumerate(
            ["tab", "return", "num_0", "num_decimal", "num_5", "return", "r", "escape"]
        ))
        with installed:
            preview.poll_settings_keyboard()
        sequence, keys = preview._settings_keys_queue.get_nowait()
        self.assertEqual(sequence, 1)
        self.assertEqual(panel.handle_keys(keys), [])
        self.assertEqual(settings.snapshot()[panel.specs[0].name], (0.25, 0.5))
        self.assertEqual(pending, [])

    def test_subject_window_routes_pending_editor_input_before_preview_acknowledges(self):
        _, panel = self.make_panel()
        preview, pending, installed = self.make_keyboard_bridge()
        with installed:
            pending.extend([("tab", 0), ("enter", 1)])
            preview.poll_settings_keyboard()
            pending.extend([("num_5", 2), ("escape", 3)])
            preview.poll_settings_keyboard()
            for _ in range(2):
                sequence, keys = preview._settings_keys_queue.get_nowait()
                self.assertEqual(panel.handle_keys(keys), [])
                preview._settings_keyboard_state[:] = [sequence, int(panel.editing)]
            self.assertFalse(panel.editing)
            pending.extend([("r", 4), ("escape", 5), ("num_8", 6), ("tab", 7)])
            preview.poll_settings_keyboard()
            self.assertEqual(pending, [("r", 4), ("escape", 5), ("num_8", 6)])
            self.assertEqual(preview._settings_keys_queue.get_nowait()[1], ["tab"])

    def test_cancel_invalid_and_calibration_keys_do_not_change_settings(self):
        settings, panel = self.make_panel()
        panel.layout_strip((800, 600))
        before = settings.snapshot()
        panel.click(panel.button.pos)
        self.assertEqual(panel.handle_keys(["r", "z", "a", "v", "up", "left"]), [])
        panel.handle_keys(["0", "enter"])
        self.assertTrue(panel.editing)
        self.assertIn("positive", panel.error)
        self.assertEqual(panel.handle_keys(["escape", "r"]), [])
        self.assertFalse(panel.editing)
        self.assertEqual(settings.snapshot(), before)

    def test_pending_readouts_reuse_existing_native_text_layouts(self):
        settings, panel = self.make_panel()
        panel.layout_strip((800, 600))
        native_labels = [stim._pygletTextObj for stim in panel.labels.values()]
        allocations = NativeLabel.allocations
        for number in range(100):
            settings.request("fixation_radius", (number + 1) / 1000)
            panel.draw()
        self.assertEqual(NativeLabel.allocations, allocations)
        self.assertEqual([stim._pygletTextObj for stim in panel.labels.values()], native_labels)
        self.assertIn("Next: 0.1 fraction", panel.labels["fixation_radius"].text)

    def test_long_numeric_text_stays_inside_gutter_without_changing_full_value(self):
        settings, panel = self.make_panel()
        panel.layout_gutter(left=275, right=395, top=218, bottom=-290)
        panel.click(panel.buttons["fixation_radius"].pos)
        panel.handle_keys(list("0.025"))
        panel.draw()
        self.assertEqual(panel.input.text, "0.025|")
        panel.handle_keys(["delete"])
        original = "0.12345678912345678"
        panel.handle_keys(list(original))
        panel.draw()
        self.assertEqual(panel.text, original)
        self.assertTrue(panel.input.text.startswith("<"))
        self.assertTrue(panel.input.text.endswith(original[-4:] + "|"))
        conservative_text_width = len(panel.input.text) * panel.input.height
        self.assertLessEqual(conservative_text_width, panel.field.width - 12)
        self.assertGreaterEqual(panel.input.pos[0] - conservative_text_width / 2, panel.panel.pos[0] - panel.panel.width / 2)
        panel.handle_keys(["enter"])
        self.assertEqual(settings.snapshot()["fixation_radius"], (0.25, float(original)))

    def test_layout_is_unchanged_without_settings(self):
        scene = compute_preview_aspect_fit((1920, 1080), (1920, 1080))
        self.assertEqual(scene["box_center"], (0, 0))
        self.assertEqual(scene["box_size"], (1920, 1080))
