"""Verify target/slider interaction and preview geometry without a display."""

from contextlib import nullcontext
import importlib.util
from pathlib import Path
import sys
import threading
import types
import unittest
from unittest.mock import Mock, patch

import bin


class Stimulus:
    def __init__(self, win, **kwargs):
        self.pos = (0, 0)
        self.__dict__.update(kwargs)
        self.draw = Mock()

    def contains(self, pos):
        return (
            abs(pos[0] - self.pos[0]) <= self.width / 2
            and abs(pos[1] - self.pos[1]) <= self.height / 2
        )


class SharedControls(list):
    def get_lock(self):
        return nullcontext()


def load_renderer():
    psychopy = types.ModuleType("psychopy")
    mouse = Mock()
    mouse.getPressed.return_value = (False, False, False)
    mouse.getPos.return_value = (0, 0)
    psychopy.event = types.SimpleNamespace(
        Mouse=Mock(return_value=mouse), getKeys=Mock(return_value=[])
    )
    psychopy.visual = types.SimpleNamespace(
        Rect=Stimulus, Circle=Stimulus, TextStim=Stimulus
    )
    utils = types.ModuleType("bin.utils")
    utils.rgb255_to_psychopy = lambda value: [2 * v / 255 - 1 for v in value]
    utils.make_bg_rect = lambda win, bg: Stimulus(win)
    path = Path(__file__).resolve().parents[1] / "bin/eye_calibration_preview.py"
    spec = importlib.util.spec_from_file_location(
        "_calibration_renderer_under_test", path
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict(
        sys.modules, {"psychopy": psychopy, "bin.utils": utils}
    ), patch.object(bin, "utils", utils, create=True):
        spec.loader.exec_module(module)
    return module, mouse


class EyeCalibrationPreviewTests(unittest.TestCase):
    def make_controls(self, canvas=(1280, 1024), subject=(1080, 1920)):
        module, mouse = load_renderer()
        shared = SharedControls([0.005, -0.005, 0, 0, 0, 0])
        controls = module.EyeCalibrationControls(
            Mock(),
            canvas,
            {
                "subject_size": subject,
                "bg": (128, 128, 128),
                "fixation_color": (0, 0, 0),
                "fixation_cross_size": 40,
                "fixation_cross_weight": 6,
                "calibration": {"x_scale": 0.005, "y_scale": -0.005},
                "x_scale_limits": (-0.02, 0.02),
                "y_scale_limits": (-0.02, 0.02),
            },
            shared,
            threading.Event(),
            threading.Event(),
        )
        return controls, shared, mouse

    def click(self, controls, mouse, pos):
        mouse.getPressed.return_value = (False, False, False)
        controls.poll_controls()
        mouse.getPos.return_value = pos
        mouse.getPressed.return_value = (True, False, False)
        controls.poll_controls()

    def test_layout_preserves_subject_aspect_and_keeps_controls_outside(self):
        for canvas, subject in (
            ((1280, 1024), (1080, 1920)),
            ((1920, 1080), (1920, 1080)),
            ((800, 600), (600, 800)),
        ):
            controls, _, _ = self.make_controls(canvas, subject)
            self.assertAlmostEqual(
                controls.box_size[0] / controls.box_size[1], subject[0] / subject[1]
            )
            for button in (
                controls.reward_button,
                controls.exit_button,
                controls.zero_button,
            ):
                self.assertFalse(controls.preview_box.contains(button[0].pos))
            self.assertFalse(controls.preview_box.contains(controls.x_slider.pos))
            self.assertFalse(controls.preview_box.contains(controls.y_slider.pos))

    def test_click_maps_target_then_slider_and_zero_preserve_target(self):
        controls, shared, mouse = self.make_controls()
        point = (0.2 * controls.box_size[0], -0.1 * controls.box_size[1])
        self.click(controls, mouse, point)
        self.assertAlmostEqual(shared[2], 0.2)
        self.assertAlmostEqual(shared[3], -0.1)
        self.click(controls, mouse, controls.x_slider.pos)
        self.assertAlmostEqual(shared[0], 0)
        self.assertAlmostEqual(shared[2], 0.2)
        self.click(controls, mouse, controls.zero_button[0].pos)
        self.assertEqual(shared[5], 1)
        self.assertAlmostEqual(shared[3], -0.1)

    def test_reward_exit_events_and_invalid_gaze_visibility(self):
        controls, _, mouse = self.make_controls()
        self.click(controls, mouse, controls.reward_button[0].pos)
        self.assertTrue(controls.reward_event.is_set())
        self.click(controls, mouse, controls.exit_button[0].pos)
        self.assertTrue(controls.exit_event.is_set())
        controls.draw({"eye_valid": True, "eye_fraction": (0.1, -0.2)})
        controls.eye_dot.draw.assert_called_once()
        controls.draw({"eye_valid": False, "status_text": "blink"})
        controls.eye_dot.draw.assert_called_once()
        self.assertEqual(controls.status_text.text, "blink")


if __name__ == "__main__":
    unittest.main()
