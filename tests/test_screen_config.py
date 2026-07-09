import os
import unittest
from unittest.mock import patch

from bin.screen import (
    MAIN_SCREEN_ENV,
    SECONDARY_SCREEN_ENV,
    ScreenGeometry,
    _parse_xrandr_query,
    compute_centered_aspect_fit,
    get_psychopy_window_kwargs,
    load_screen_config,
    resolve_task_screens,
    resolve_scene_size,
    scale_scene_point,
    select_screen,
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

    def test_secondary_alias_inherits_secondary_screen_environment(self):
        cfg = {"screens": {"main": None, "secondary": None}}
        with patch.dict(
            os.environ,
            {
                MAIN_SCREEN_ENV: "HDMI-2",
                SECONDARY_SCREEN_ENV: "HDMI-1",
            },
        ):
            self.assertEqual(
                load_screen_config(cfg),
                {"main": "HDMI-2", "experimenter": "HDMI-1"},
            )

    def test_top_level_secondary_screen_alias_matches_experimenter_screen(self):
        cfg = {"main_screen": "HDMI-2", "secondary_screen": "HDMI-1"}

        self.assertEqual(
            load_screen_config(cfg),
            {"main": "HDMI-2", "experimenter": "HDMI-1"},
        )

    def test_resolve_task_screens_rejects_same_display_by_default(self):
        screens = [
            ScreenGeometry(index=0, x=0, y=0, width=800, height=480, name="HDMI-1"),
            ScreenGeometry(index=1, x=800, y=0, width=800, height=480, name="HDMI-2"),
        ]

        with patch("bin.screen.get_monitor_screens", return_value=screens):
            with self.assertRaises(ValueError):
                resolve_task_screens({"main": 0, "experimenter": 0})

    def test_resolve_task_screens_can_collapse_same_display_to_main_only(self):
        screens = [
            ScreenGeometry(index=0, x=0, y=0, width=800, height=480, name="HDMI-1"),
            ScreenGeometry(index=1, x=800, y=0, width=800, height=480, name="HDMI-2"),
        ]

        with patch("bin.screen.get_monitor_screens", return_value=screens):
            main_screen, experimenter_screen = resolve_task_screens(
                {"main": "HDMI-1", "experimenter": "HDMI-1"},
                allow_same_screen=True,
            )

        self.assertEqual(main_screen.index, 0)
        self.assertIsNone(experimenter_screen)

    def test_hdmi_names_do_not_cross_match_by_off_by_one_alias(self):
        screens = [
            ScreenGeometry(index=0, x=0, y=0, width=800, height=480, name="HDMI-A-1"),
            ScreenGeometry(index=1, x=800, y=0, width=800, height=480, name="HDMI-A-2"),
        ]

        self.assertEqual(select_screen(screens, "HDMI-1", role="main").index, 0)
        self.assertEqual(select_screen(screens, "HDMI-2", role="experimenter").index, 1)

    def test_linux_psychopy_window_uses_virtual_desktop_position(self):
        screen = ScreenGeometry(index=1, x=800, y=0, width=2560, height=1600, name="HDMI-A-2")

        with patch("bin.screen.sys.platform", "linux"):
            kwargs = get_psychopy_window_kwargs(screen, fullscreen=True)

        self.assertNotIn("screen", kwargs)
        self.assertEqual(kwargs["fullscr"], False)
        self.assertEqual(kwargs["size"], (2560, 1600))
        self.assertEqual(kwargs["pos"], (800, 0))
        self.assertEqual(kwargs["allowGUI"], False)

    def test_non_linux_psychopy_fullscreen_uses_screen_index(self):
        screen = ScreenGeometry(index=1, x=800, y=0, width=2560, height=1600, name="HDMI-A-2")

        with patch("bin.screen.sys.platform", "darwin"):
            kwargs = get_psychopy_window_kwargs(screen, fullscreen=True)

        self.assertEqual(kwargs, {"screen": 1, "fullscr": True})

    def test_xrandr_query_uses_rotated_framebuffer_size(self):
        screens = _parse_xrandr_query(
            "HDMI-2 connected primary 1600x2560+0+0 right "
            "(normal left inverted right x axis y axis) 256mm x 160mm\n"
            "HDMI-1 connected 1920x1080+1600+0 "
            "(normal left inverted right x axis y axis) 531mm x 299mm\n"
        )

        main = select_screen(screens, "HDMI-2", role="main")
        self.assertEqual((main.width, main.height), (1600, 2560))
        self.assertEqual(main.rotation, "right")

    def test_fullscreen_scene_size_uses_rotated_framebuffer_size(self):
        screen = ScreenGeometry(
            index=0,
            x=0,
            y=0,
            width=1600,
            height=2560,
            name="HDMI-2",
            rotation="right",
        )

        self.assertEqual(
            resolve_scene_size(screen, fullscreen=True, requested_size=(2560, 1600)),
            (1600, 2560),
        )

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
