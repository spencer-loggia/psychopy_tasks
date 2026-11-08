"""Render experimenter controls with a fake window and real scene routing."""
import queue
import sys
import threading
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

from bin.screen import ScreenGeometry, _experimenter_preview_process, serialize_preview_image
from tests.test_eye_calibration_preview import Stimulus, TextStim
from tests.test_live_settings_ui import SettingsModel


class ExperimenterPreviewControlsTests(unittest.TestCase):
    def render(self, *, sync_enabled, settings=None, key_batches=(), scenes=None,
               preserve_scene_transitions=False, render_cycles=1, local_keys=()):
        primitives, drawn = [], []
        pending, self.frames = [], []

        def factory(kind, cls=Stimulus):
            def build(win, **kwargs):
                stim = cls(win, **kwargs)
                primitives.append((kind, stim))
                def draw():
                    drawn.append((kind, getattr(stim, "text", None)))
                    pending.append((kind, stim))
                stim.draw = draw
                return stim
            return build

        visual = types.SimpleNamespace(
            Rect=factory("rect"), Circle=factory("circle"),
            TextStim=factory("text", TextStim),
            ImageStim=factory("image"),
        )
        mouse = Mock()
        mouse.getPressed.return_value = (False, False, False)
        psychopy = types.ModuleType("psychopy")
        psychopy.visual = visual
        psychopy.event = types.SimpleNamespace(getKeys=lambda: list(local_keys), Mouse=lambda **kwargs: mouse)
        self.wait = Mock()
        psychopy.core = types.SimpleNamespace(wait=self.wait)
        sync = [20 if sync_enabled else 0, 800, 600, 0]
        win = Mock()
        self.rendered_sequence = types.SimpleNamespace(value=0)
        self.acknowledged_at_flip = []
        def flip():
            self.frames.append(tuple(pending))
            pending.clear()
            self.acknowledged_at_flip.append(self.rendered_sequence.value)
            sync[3] = 1
        win.flip.side_effect = flip
        commands = queue.Queue()
        for scene in scenes if scenes is not None else [{
            "type": "static_scene", "main_size": (600, 800), "main_rotation_deg": 90,
            "circles": [{"pos": (60, 80), "radius": 20, "color": (200, 200, 200)}],
        }]:
            commands.put(scene)
        stop = Mock()
        stop.is_set.side_effect = [False] * render_cycles + [True]
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
                preserve_scene_transitions=preserve_scene_transitions,
                rendered_sequence_shared=self.rendered_sequence,
            )
        self.assertTrue(startup.send.call_args.args[0]["ok"])
        self.assertEqual(win.flip.call_count, render_cycles + 1)
        return primitives, drawn

    def test_brief_reveals_and_chosen_status_each_get_a_separate_low_rate_flip(self):
        image = serialize_preview_image(np.zeros((4, 4, 3), dtype=np.uint8))
        positions = [(-100, 20), (100, 20)]
        hidden_sets = [{0, 1}, {1}, {0, 1}, {0}, {0}]
        scenes = []
        for sequence, hidden in enumerate(hidden_sets, start=1):
            scenes.append({
                "type": "static_scene", "_preview_sequence": sequence, "main_size": (800, 600),
                "images": [{"image_payload": image, "pos": pos, "size": (40, 40)} for pos in positions],
                "overlay_dots": [{"pos": positions[i], "radius": 5} for i in sorted(hidden)],
                "hit_boxes": [{"pos": pos, "size": (50, 50), "color": (230, 200, 40)} for pos in positions],
                "status_boxes": ([{"pos": positions[1], "size": (60, 60), "color": (180, 80, 220)}]
                                 if sequence == 5 else []),
            })
        self.render(sync_enabled=False, scenes=scenes, preserve_scene_transitions=True,
                    render_cycles=len(scenes), local_keys=("r",))
        visible_sets = []
        for frame in self.frames[1:]:
            images = {stim.pos for kind, stim in frame if kind == "image"}
            overlays = {stim.pos for kind, stim in frame if kind == "circle"}
            visible_sets.append(images - overlays)
            image_indices = [i for i, (kind, _) in enumerate(frame) if kind == "image"]
            dot_indices = [i for i, (kind, _) in enumerate(frame) if kind == "circle"]
            self.assertLess(max(image_indices), min(dot_indices), "Hidden dots must overlay images")
        self.assertEqual(visible_sets, [set(), {positions[0]}, set(), {positions[1]}, {positions[1]}])
        chosen = [stim for kind, stim in self.frames[-1] if kind == "rect" and getattr(stim, "width", None) == 60]
        self.assertEqual(len(chosen), 1)
        self.assertEqual(chosen[0].lineColor, [180/127.5-1, 80/127.5-1, 220/127.5-1])
        self.assertEqual(self.acknowledged_at_flip, [0, 0, 1, 2, 3, 4])
        self.assertEqual(self.rendered_sequence.value, 5)
        self.assertEqual([call.args for call in self.wait.call_args_list], [(0.1,)] * 5)
        self.assertTrue(self.reward_event.is_set(), "Queued scenes must not starve preview controls")

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
