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
from bin.preview_text import set_preview_text
from bin.screen import ExperimenterPreview


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


class NativeLabel:
    """Count layout allocations independently of Python garbage collection."""

    allocations = 0

    def __init__(self, text):
        NativeLabel.allocations += 1
        self.text = text
        self.width, self.height = 300, 20
        self.delete = Mock()


class TextStim(Stimulus):
    """Model the pinned TextStim setter's replacement of its native layout."""

    def __init__(self, win, **kwargs):
        super().__init__(win, **kwargs)
        self.languageStyle = "LTR"
        self._needSetText = False
        self._pygletTextObj = NativeLabel(self.text)
        self.text_assignments = 0

    @property
    def text(self):
        return self.__dict__.get("text", "")

    @text.setter
    def text(self, value):
        if self.text != value:
            self.text_assignments += 1
            self.__dict__["text"] = value
            self._pygletTextObj = NativeLabel(value)
            self._needSetText = False


def load_renderer():
    psychopy = types.ModuleType("psychopy")
    mouse = Mock()
    mouse.getPressed.return_value = (False, False, False)
    mouse.getPos.return_value = (0, 0)
    psychopy.event = types.SimpleNamespace(
        Mouse=Mock(return_value=mouse), getKeys=Mock(return_value=[])
    )
    psychopy.visual = types.SimpleNamespace(
        Rect=Stimulus, Circle=Stimulus, TextStim=TextStim, ImageStim=Stimulus
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
        self.renderer = module
        shared = SharedControls([0.005, -0.005] + [0] * 11)
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

    def test_f_requests_attention_without_changing_calibration(self):
        controls, shared, _ = self.make_controls()
        calibration_before = shared[:6]
        self.renderer.event.getKeys.return_value = ["f"]
        controls.poll_controls()
        preview = object.__new__(ExperimenterPreview)
        preview._calibration_controls = shared
        self.assertEqual(preview.get_calibration_controls()["attention_count"], 1)
        self.assertEqual(shared[:6], calibration_before)
        controls.poll_controls()
        self.assertEqual(preview.get_calibration_controls()["attention_count"], 2)
        self.assertEqual(shared[:6], calibration_before)

    def test_r_key_requests_one_manual_reward(self):
        controls, shared, _ = self.make_controls()
        preview = object.__new__(ExperimenterPreview)
        preview._reward_event = controls.reward_event
        calibration_before = shared[:7]
        self.renderer.event.getKeys.return_value = ["r"]
        controls.poll_controls()
        self.assertTrue(preview.consume_manual_reward_request())
        self.assertFalse(preview.consume_manual_reward_request())
        self.assertEqual(shared[:7], calibration_before)

    def test_a_requests_mode_toggle_without_changing_mapping_or_arrow_requests(self):
        controls, shared, _ = self.make_controls()
        preview = object.__new__(ExperimenterPreview)
        preview._calibration_controls = shared
        original = shared[:7]
        self.renderer.event.getKeys.return_value = ["a"]
        controls.poll_controls()
        self.assertEqual(preview.get_calibration_controls()["auto_reward_count"], 1)
        self.assertEqual(shared[:7], original)
        controls.poll_controls()
        self.assertEqual(preview.get_calibration_controls()["auto_reward_count"], 2)
        preview.request_calibration_scale_steps(x_steps=1)
        controls._publish_controls(calibration_changed=False)
        self.assertEqual(shared[7:9], [1, 0])
        self.assertEqual(shared[:7], original)

    def test_v_toggle_preserves_mapping_and_pending_target_and_gain_steps(self):
        controls, shared, _ = self.make_controls()
        preview = object.__new__(ExperimenterPreview)
        preview._calibration_controls = shared
        original = shared[:7]
        self.renderer.event.getKeys.return_value = ["v", "v", "v"]
        controls.poll_controls()
        self.assertEqual(preview.get_calibration_controls()["video_toggle_count"], 3)
        self.assertEqual(shared[:7], original)
        preview.request_calibration_fixation_steps(x_steps=1)
        preview.request_calibration_scale_steps(y_steps=2)
        controls._publish_controls(calibration_changed=False)
        self.assertEqual(shared[7:9], [0, 2])
        self.assertEqual(shared[10:12], [1, 0])

    def test_video_mirror_follows_target_and_replaces_cross_until_disabled(self):
        controls, _, mouse = self.make_controls()
        self.click(controls, mouse, (0.2 * controls.box_size[0], -0.1 * controls.box_size[1]))
        reader = Mock()
        reader.read_latest.return_value = types.SimpleNamespace(sequence=5, rgb=object())
        payload = {"video_visible": True, "video_descriptor": {}, "video_sequence": 5}
        with patch.object(self.renderer, "SharedVideoFrameReader", return_value=reader), patch.object(
            self.renderer, "upload_rgb_texture",
        ) as upload:
            controls.draw(payload)
            video_stim = controls.video_stim
            self.assertEqual(video_stim.pos, mouse.getPos.return_value)
            self.assertAlmostEqual(video_stim.size[0], 100 * controls.box_size[0] / 1080)
            self.assertAlmostEqual(video_stim.size[1], 100 * controls.box_size[1] / 1920)
            for part in controls.cross:
                part.draw.assert_not_called()
            reader.read_latest.return_value = None
            controls.draw(payload)
            self.assertIs(controls.video_stim, video_stim)
            upload.assert_called_once()
            controls.draw({"video_visible": False})
            for part in controls.cross:
                part.draw.assert_called_once()
            self.assertEqual(video_stim.draw.call_count, 2)
            controls.close()
            reader.close.assert_called_once()

    def test_arrows_update_values_readouts_and_knobs_from_either_window(self):
        for source in ("main", "preview"):
            with self.subTest(source=source):
                controls, shared, _ = self.make_controls()
                preview = object.__new__(ExperimenterPreview)
                preview._calibration_controls = shared
                original_y = controls.y_slider._knob_pos()[1]
                for keys, expected, direction in (
                    (["right", "up", "up"], (0.0051, -0.0048), 1),
                    (["left", "down"], (0.0050, -0.0049), -1),
                ):
                    previous = (controls.x_slider.value, controls.y_slider.value)
                    if source == "main":
                        # Multiple requests before a preview frame must add up.
                        for key in keys:
                            preview.request_calibration_scale_steps(
                                x_steps=int(key == "right") - int(key == "left"),
                                y_steps=int(key == "up") - int(key == "down"),
                            )
                        # Publishing an unrelated UI action must not erase them.
                        controls._publish_controls(calibration_changed=False)
                    else:
                        self.renderer.event.getKeys.return_value = keys
                    controls.poll_controls()
                    controls.draw({})
                    snapshot = preview.get_calibration_controls()
                    for axis, slider, value, old in zip(
                        ("x", "y"), (controls.x_slider, controls.y_slider), expected, previous
                    ):
                        self.assertAlmostEqual(slider.value, value)
                        self.assertAlmostEqual(snapshot[f"{axis}_scale"], value)
                        self.assertGreater((slider.value - old) * direction, 0)
                        self.assertIn(f"{value:.4f}", slider.text.text)
                        self.assertEqual(slider.knob.pos, slider._knob_pos())
                    self.assertEqual(shared[2:4], [0, 0])
                    self.assertEqual(shared[7:9], [0, 0])
                    self.renderer.event.getKeys.return_value = []
                    controls.poll_controls()
                    self.assertEqual(preview.get_calibration_controls(), snapshot)
                self.assertGreater(controls.y_slider.knob.pos[1], original_y)

    def test_arrow_steps_clamp_at_slider_limits_without_repeated_revisions(self):
        controls, shared, _ = self.make_controls()
        preview = object.__new__(ExperimenterPreview)
        preview._calibration_controls = shared
        for steps, expected in ((1000, 0.02), (-1000, -0.02)):
            preview.request_calibration_scale_steps(x_steps=steps, y_steps=steps)
            controls.poll_controls()
            self.assertEqual(shared[:2], [expected, expected])
            revision = shared[4]
            preview.request_calibration_scale_steps(x_steps=steps, y_steps=steps)
            controls.poll_controls()
            self.assertEqual(shared[4], revision)

    def test_keypad_moves_clicked_target_from_either_window_without_changing_gain(self):
        from bin.eye_calibration_keys import fixation_key_steps

        # Num Lock on/off, pygame names, and normalized digits share directions.
        for up, right, down, left in (
            ("num_8", "num_6", "num_2", "num_4"),
            ("num_up", "num_right", "num_down", "num_left"),
            ("[8]", "[6]", "[2]", "[4]"),
            ("8", "6", "2", "4"),
        ):
            for source in ("main", "preview"):
                with self.subTest(source=source, up=up):
                    controls, shared, mouse = self.make_controls()
                    preview = object.__new__(ExperimenterPreview)
                    preview._calibration_controls = shared
                    self.click(controls, mouse, (
                        0.2 * controls.box_size[0], -0.1 * controls.box_size[1]
                    ))
                    for keys, expected in (
                        ([up, up, right], (0.225, -0.05)),
                        ([down, left], (0.2, -0.075)),
                    ):
                        revision = shared[4]
                        if source == "main":
                            for key in keys:
                                x_steps, y_steps = fixation_key_steps([key])
                                preview.request_calibration_fixation_steps(
                                    x_steps=x_steps, y_steps=y_steps
                                )
                            # UI updates must not overwrite queued target steps.
                            controls._publish_controls(calibration_changed=False)
                        else:
                            self.renderer.event.getKeys.return_value = keys
                        controls.poll_controls()
                        controls.draw({})
                        self.assertTrue(all(
                            key in self.renderer.event.getKeys.call_args.kwargs["keyList"]
                            for key in keys
                        ))
                        snapshot = preview.get_calibration_controls()
                        for axis, fraction, size, center, pixel in zip(
                            ("x", "y"), expected, controls.box_size,
                            controls.box_center, controls.cross[0].pos,
                        ):
                            self.assertAlmostEqual(snapshot[f"fix_{axis}"], fraction)
                            self.assertAlmostEqual(pixel, center + fraction * size)
                        self.assertGreater(snapshot["revision"], revision)
                        self.assertEqual(shared[:2], [0.005, -0.005])
                        self.assertEqual(shared[5:7], [0, 0])
                        self.assertEqual(shared[10:12], [0, 0])
                        self.renderer.event.getKeys.return_value = []
                        controls.poll_controls()
                        self.assertEqual(preview.get_calibration_controls(), snapshot)

    def test_keypad_stops_at_screen_edges_without_resetting_history_again(self):
        controls, shared, _ = self.make_controls()
        preview = object.__new__(ExperimenterPreview)
        preview._calibration_controls = shared
        for steps, expected in ((100, 0.5), (-100, -0.5)):
            preview.request_calibration_fixation_steps(x_steps=steps, y_steps=steps)
            controls.poll_controls()
            self.assertEqual(shared[2:4], [expected, expected])
            revision = shared[4]
            preview.request_calibration_fixation_steps(x_steps=steps, y_steps=steps)
            controls.poll_controls()
            self.assertEqual(shared[4], revision)

    def test_attention_cross_doubles_flashes_and_restores_after_one_second(self):
        controls, _, _ = self.make_controls()
        sizes = [(part.width, part.height) for part in controls.cross]
        payload = {"attention_started_s": 100.0}
        # The actual renderer is shared with the subject window. Sample both
        # phases and the end boundary; dimensions must never compound.
        for now, scale, draws in (
            (100.0, 2, 1),
            (100.15, 2, 1),
            (100.25, 2, 2),
            (100.95, 2, 2),
            (101.0, 1, 3),
        ):
            with self.subTest(now=now), patch.object(
                self.renderer.time, "perf_counter", return_value=now
            ):
                controls.draw(payload)
                for part, base_size in zip(controls.cross, sizes):
                    self.assertEqual(part.size, tuple(v * scale for v in base_size))
                    self.assertEqual(part.draw.call_count, draws)
                    self.assertEqual(part.pos, (0, 0))

    def test_changing_readouts_do_not_allocate_layouts_over_long_run(self):
        controls, _, _ = self.make_controls()
        # Avoid storing draw-call histories in the test itself. 100,000 frames
        # represents >80 minutes at the calibration preview's 20 Hz cadence.
        for owner in (controls, controls.x_slider, controls.y_slider):
            for value in vars(owner).values():
                for stim in value if isinstance(value, tuple) else (value,):
                    if isinstance(stim, Stimulus):
                        stim.draw = lambda: None
        text_stims = (
            controls.x_slider.text,
            controls.y_slider.text,
            controls.status_text,
        )
        labels = [stim._pygletTextObj for stim in text_stims]
        allocations = NativeLabel.allocations
        for frame in range(100_000):
            controls.draw(
                {
                    "eye_valid": True,
                    "eye_fraction": (0.1, -0.2),
                    "status_text": f"Eye 0 ({frame / 100:.2f}, 25.00) px\nfix 0.0%",
                }
            )
        self.assertEqual(NativeLabel.allocations, allocations)
        for stim, label in zip(text_stims, labels):
            self.assertIs(stim._pygletTextObj, label)
            self.assertEqual(label.text, stim.text)
            self.assertEqual(stim.text_assignments, 0)
            self.assertFalse(stim.autoLog)
        self.assertEqual(controls.y_slider.text.text, "y scale\n-0.0050")
        self.assertIn("999.99", controls.status_text.text)

    def test_preview_text_releases_layout_when_styling_requires_replacement(self):
        stim = TextStim(None, text="before")
        old_label = stim._pygletTextObj
        stim._needSetText = True
        set_preview_text(stim, "after\nsecond line")
        self.assertEqual(stim.text, "after\nsecond line")
        self.assertIsNot(stim._pygletTextObj, old_label)
        old_label.delete.assert_called_once()

    def test_preview_text_updates_metrics_and_skips_unchanged_values(self):
        stim = TextStim(None, text="before")
        label = stim._pygletTextObj
        set_preview_text(stim, "after")
        self.assertEqual(stim.width, label.width)
        self.assertEqual(stim._fontHeightPix, label.height)
        self.assertTrue(stim._needUpdate)
        stim._needUpdate = False
        set_preview_text(stim, "after")
        self.assertFalse(stim._needUpdate)
        self.assertEqual(stim.text_assignments, 0)

    def test_preview_text_without_native_label_uses_standard_setter(self):
        stim = Stimulus(None, text="before")
        set_preview_text(stim, "after")
        self.assertEqual(stim.text, "after")


if __name__ == "__main__":
    unittest.main()
