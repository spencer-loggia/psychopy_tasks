"""Render experimenter controls with a fake window and real scene routing."""
import queue
import sys
import threading
import types
import unittest
from unittest.mock import Mock, patch

from bin.screen import ScreenGeometry, _experimenter_preview_process
from tests.test_eye_calibration_preview import Stimulus, TextStim
from tests.test_live_settings_ui import SettingsModel


class ExperimenterPreviewControlsTests(unittest.TestCase):
    def render(self, *, sync_enabled, settings=None, key_batches=()):
        primitives, drawn = [], []

        def factory(kind, cls=Stimulus):
            def build(win, **kwargs):
                stim = cls(win, **kwargs)
                primitives.append((kind, stim))
                stim.draw = lambda: drawn.append((kind, getattr(stim, "text", None)))
                return stim
            return build

        visual = types.SimpleNamespace(
            Rect=factory("rect"), Circle=factory("circle"),
            TextStim=factory("text", TextStim),
        )
        mouse = Mock()
        mouse.getPressed.return_value = (False, False, False)
        psychopy = types.ModuleType("psychopy")
        psychopy.visual = visual
        psychopy.event = types.SimpleNamespace(getKeys=lambda: [], Mouse=lambda **kwargs: mouse)
        psychopy.core = types.SimpleNamespace(wait=lambda duration: None)
        sync = [20 if sync_enabled else 0, 800, 600, 0]
        win = Mock()
        win.flip.side_effect = lambda: sync.__setitem__(3, 1)
        commands = queue.Queue()
        commands.put({
            "type": "static_scene", "main_size": (600, 800), "main_rotation_deg": 90,
            "circles": [{"pos": (60, 80), "radius": 20, "color": (200, 200, 200)}],
        })
        stop = Mock()
        stop.is_set.side_effect = [False, True]
        startup = Mock()
        settings_keys = queue.Queue()
        for index, batch in enumerate(key_batches, start=1):
            settings_keys.put((index, batch))
        self.keyboard_state = [0, 0]
        self.reward_event, self.exit_event = threading.Event(), threading.Event()
        with patch.dict(sys.modules, {"psychopy": psychopy}), patch(
            "bin.screen.open_psychopy_window", return_value=win,
        ), patch("bin.screen.configure_window_vsync"):
            _experimenter_preview_process(
                ScreenGeometry(index=1, x=0, y=0, width=800, height=600),
                "", 0, 100, commands, self.reward_event, self.exit_event,
                None, None, None, None, None, types.SimpleNamespace(value=float("nan")),
                stop, startup, sync_bar_shared=sync, live_settings=settings,
                settings_keys_queue=settings_keys, settings_keyboard_state=self.keyboard_state,
            )
        self.assertTrue(startup.send.call_args.args[0]["ok"])
        self.assertEqual(win.flip.call_count, 2)
        return primitives, drawn

    def test_sync_uses_phase_text_without_adding_bar_rectangles(self):
        enabled, drawn = self.render(sync_enabled=True)
        disabled, disabled_drawn = self.render(sync_enabled=False)
        self.assertIn(("text", "sync OFF"), drawn)
        self.assertIn(("text", "sync ON"), drawn)
        self.assertNotIn(("text", "sync OFF"), disabled_drawn)
        self.assertEqual(
            [(s.pos, s.width, s.height) for kind, s in enabled if kind == "rect"],
            [(s.pos, s.width, s.height) for kind, s in disabled if kind == "rect"],
        )

    def test_fixation_circle_is_unfilled_and_rotates_with_subject_scene(self):
        primitives, drawn = self.render(sync_enabled=False)
        circles = [stim for kind, stim in primitives if kind == "circle"]
        self.assertEqual(len(circles), 1)
        self.assertIsNone(circles[0].fillColor)
        self.assertEqual(circles[0].radius, 20)
        self.assertEqual(circles[0].pos, (-80, 60))
        self.assertEqual(circles[0].lineColor, [200 / 127.5 - 1] * 3)
        self.assertIn(("circle", None), drawn)

    def test_forwarded_edits_keep_batch_boundaries_and_acknowledge_editor_state(self):
        settings = SettingsModel()
        self.render(sync_enabled=False, settings=settings, key_batches=[
            ["tab", "enter", "0", "period", "5", "enter", "escape", "r"],
            ["tab", "enter", "0", "period", "7", "5", "enter"],
            ["r"],
        ])
        self.assertEqual(settings.snapshot()["pump_pulse_time"], (0.25, 0.5))
        self.assertEqual(settings.snapshot()["fixation_reward_interval"], (0.25, 0.75))
        self.assertEqual(self.keyboard_state, [3, 0])
        self.assertTrue(self.reward_event.is_set())
        self.assertFalse(self.exit_event.is_set())
