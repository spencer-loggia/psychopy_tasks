import os
import unittest
from unittest.mock import patch

from bin.screen import (
    MAIN_SCREEN_ENV,
    SECONDARY_SCREEN_ENV,
    compute_centered_aspect_fit,
    load_screen_config,
    scale_scene_point,
)


class ScreenConfigTests(unittest.TestCase):
    def test_none_screen_values_inherit_environment(self):
        cfg = {"screens": {"main": None, "experimenter": None}}
        with patch.dict(
            os.environ,
            {
                MAIN_SCREEN_ENV: "HDMI-1",
                SECONDARY_SCREEN_ENV: "DSI-1",
            },
        ):
            self.assertEqual(
                load_screen_config(cfg),
                {"main": "HDMI-1", "experimenter": "DSI-1"},
            )

    def test_configured_screen_values_override_environment(self):
        cfg = {"screens": {"main": 1, "experimenter": "DSI-2"}}
        with patch.dict(
            os.environ,
            {
                MAIN_SCREEN_ENV: "HDMI-1",
                SECONDARY_SCREEN_ENV: "DSI-1",
            },
        ):
            self.assertEqual(load_screen_config(cfg), {"main": 1, "experimenter": "DSI-2"})

    def test_null_screen_values_require_environment(self):
        cfg = {"screens": {"main": None, "experimenter": None}}
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                load_screen_config(cfg)

    def test_centered_aspect_fit_preserves_main_aspect_ratio(self):
        layout = compute_centered_aspect_fit((1920, 1080), (1000, 500))

        self.assertEqual(layout["box_center"], (0.0, 0.0))
        self.assertAlmostEqual(layout["box_size"][0], 1920.0)
        self.assertAlmostEqual(layout["box_size"][1], 960.0)
        self.assertAlmostEqual(layout["top_margin"], 60.0)
        self.assertAlmostEqual(layout["bottom_margin"], 60.0)

    def test_preview_mapping_places_main_corners_on_fitted_box_corners(self):
        main_size = (1000, 500)
        layout = compute_centered_aspect_fit((1920, 1080), main_size)
        box_w, box_h = layout["box_size"]

        upper_left = scale_scene_point((-500, 250), main_size, layout["box_size"])
        lower_right = scale_scene_point((500, -250), main_size, layout["box_size"])

        self.assertAlmostEqual(upper_left[0], -box_w * 0.5)
        self.assertAlmostEqual(upper_left[1], box_h * 0.5)
        self.assertAlmostEqual(lower_right[0], box_w * 0.5)
        self.assertAlmostEqual(lower_right[1], -box_h * 0.5)


if __name__ == "__main__":
    unittest.main()
