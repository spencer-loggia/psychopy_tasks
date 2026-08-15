import os
import queue
import types
import unittest
from contextlib import nullcontext
from unittest.mock import Mock, patch

from bin.screen import (
    ExperimenterPreview,
    MAIN_SCREEN_ENV,
    MainDisplayFrameTimingMonitor,
    SECONDARY_SCREEN_ENV,
    ScreenGeometry,
    _bind_linux_pyglet_display,
    _parse_xrandr_listactivemonitors,
    _parse_xrandr_query,
    _request_x11_fullscreen_monitor,
    build_reward_hit_boxes,
    compute_aspect_cover_size,
    compute_centered_aspect_fit,
    configure_window_vsync,
    enforce_window_vsync,
    format_experimenter_label,
    load_screen_config,
    open_psychopy_window,
    reward_level_color,
    resolve_window_frame_rate,
    resolve_task_screens,
    resolve_scene_size,
    scale_scene_point,
    select_screen,
    set_tk_window_fullscreen,
    verify_psychopy_window_screen,
)


class ScreenConfigTests(unittest.TestCase):
    def test_psychopy_display_constructor_enumerates_target_first(self):
        target_info = ScreenGeometry(
            index=1, x=0, y=0, width=1600, height=2560, name="HDMI-2"
        )
        other = types.SimpleNamespace(x=1600, y=0, width=1920, height=1080)
        target = types.SimpleNamespace(x=0, y=0, width=1600, height=2560)

        class FakeDisplay:
            def __init__(self, *args, **kwargs):
                self._screens = None

            def get_screens(self):
                return self._screens or [other, target]

        pyglet = types.ModuleType("pyglet")
        pyglet.canvas = types.SimpleNamespace(Display=FakeDisplay)

        with patch.dict("sys.modules", {"pyglet": pyglet}):
            with _bind_linux_pyglet_display(target_info) as selection:
                display = pyglet.canvas.Display(x_screen=0)
                self.assertIs(display.get_screens()[0], target)

        self.assertEqual(selection["monitor_index"], 1)
        self.assertEqual(selection["selected_rect"], (0, 0, 1600, 2560))
        self.assertEqual(
            selection["available_rects"],
            [(1600, 0, 1920, 1080), (0, 0, 1600, 2560)],
        )
        self.assertIs(pyglet.canvas.Display, FakeDisplay)

    def test_psychopy_display_constructor_can_fall_back_for_diagnostic(self):
        target_info = ScreenGeometry(
            index=0, x=0, y=0, width=1600, height=2560, name="HDMI-2"
        )
        other = types.SimpleNamespace(x=1600, y=0, width=1920, height=1080)

        class FakeDisplay:
            def __init__(self, *args, **kwargs):
                pass

            def get_screens(self):
                return [other]

        pyglet = types.ModuleType("pyglet")
        pyglet.canvas = types.SimpleNamespace(Display=FakeDisplay)

        with patch.dict("sys.modules", {"pyglet": pyglet}):
            with _bind_linux_pyglet_display(target_info, strict=False):
                display = pyglet.canvas.Display(x_screen=0)

        self.assertEqual(display.get_screens(), [other])

    def test_configure_window_vsync_can_disable_experimenter_blocking(self):
        class WindowHandle:
            def __init__(self):
                self.vsync_calls = []

            def set_vsync(self, enabled):
                self.vsync_calls.append(enabled)

        win = Mock()
        win.waitBlanking = True
        win.winHandle = WindowHandle()
        win.backend = None

        self.assertTrue(configure_window_vsync(win, False))
        self.assertFalse(win.waitBlanking)
        self.assertEqual(win.winHandle.vsync_calls, [False])

    def test_enforce_window_vsync_enables_blanking_and_native_swap_interval(self):
        class WindowHandle:
            def __init__(self):
                self.vsync_calls = []

            def set_vsync(self, enabled):
                self.vsync_calls.append(enabled)

        win = Mock()
        win.waitBlanking = False
        win.winHandle = WindowHandle()
        win.backend = None

        self.assertTrue(enforce_window_vsync(win))
        self.assertTrue(win.waitBlanking)
        self.assertEqual(win.winHandle.vsync_calls, [True])

    def test_refresh_override_is_measured_and_compared(self):
        win = Mock()
        win.getActualFrameRate.return_value = 59.94
        logger = Mock()

        fps, frame_duration = resolve_window_frame_rate(
            win,
            configured_fps=60.0,
            msg_logger=logger,
            context="match2cue",
        )

        self.assertEqual(fps, 60.0)
        self.assertAlmostEqual(frame_duration, 1.0 / 60.0)
        messages = [call.args[1] for call in logger.log.call_args_list]
        self.assertTrue(any("status=match" in message for message in messages))

    def test_refresh_override_logs_mismatch_but_remains_authoritative(self):
        win = Mock()
        win.getActualFrameRate.return_value = 28.3
        logger = Mock()

        fps, _ = resolve_window_frame_rate(
            win,
            configured_fps=60.0,
            msg_logger=logger,
            context="active_foraging",
        )

        self.assertEqual(fps, 60.0)
        logger.log.assert_any_call(
            "WARN",
            (
                "refresh_rate_comparison context=active_foraging "
                "configured_fps=60.000000 measured_fps=28.300000 "
                "difference_hz=31.700000 tolerance_hz=0.600000 status=mismatch"
            ),
        )

    def test_frame_timing_monitor_excludes_time_between_sequences(self):
        win = Mock()
        win.recordFrameIntervals = False
        win.nDroppedFrames = 3
        monitor = MainDisplayFrameTimingMonitor(win, 1.0 / 60.0)

        with monitor.continuous_sequence():
            win.nDroppedFrames = 5
        win.nDroppedFrames = 20
        with monitor.continuous_sequence():
            win.nDroppedFrames = 21

        self.assertEqual(monitor.missed_refreshes, 3)
        self.assertFalse(win.recordFrameIntervals)
        self.assertAlmostEqual(win.refreshThreshold, 0.025)

    def test_reward_level_colors_match_active_foraging_legend(self):
        self.assertEqual(reward_level_color(0), (220, 60, 60))
        self.assertEqual(reward_level_color(1), (140, 140, 140))
        self.assertEqual(reward_level_color(2), (230, 200, 40))
        self.assertEqual(reward_level_color(3), (60, 180, 75))

    def test_reward_hit_boxes_use_choice_target_scale(self):
        boxes = build_reward_hit_boxes(
            [
                {
                    "pos": (12.0, -8.0),
                    "size": (80.0, 60.0),
                    "reward_level": 2,
                },
                {
                    "pos": (0.0, 0.0),
                    "size": (20.0, 20.0),
                },
            ],
            hitbox_scale=1.25,
        )

        self.assertEqual(
            boxes,
            [
                {
                    "pos": [12.0, -8.0],
                    "size": [100.0, 75.0],
                    "color": [230, 200, 40],
                    "line_width": 6.0,
                }
            ],
        )

    def test_preview_queue_drops_old_scene_for_latest_scene(self):
        class OneItemQueue:
            def __init__(self):
                self.items = [{"type": "old"}]

            def put_nowait(self, payload):
                if self.items:
                    raise queue.Full
                self.items.append(payload)

            def get_nowait(self):
                if not self.items:
                    raise queue.Empty
                return self.items.pop(0)

        preview = object.__new__(ExperimenterPreview)
        preview.poll = lambda: False
        preview._process = Mock()
        preview._process.is_alive.return_value = True
        preview._queue = OneItemQueue()

        preview._send({"type": "latest"})

        self.assertEqual(preview._queue.items, [{"type": "latest"}])

    def test_preview_queue_includes_subject_and_trial_progress(self):
        preview = object.__new__(ExperimenterPreview)
        preview.poll = lambda: False
        preview._process = Mock()
        preview._process.is_alive.return_value = True
        preview._queue = queue.Queue()
        preview.subject = "Yuri"
        preview.current_trial_num = 17
        preview.total_trials = 2000

        preview._send({"type": "static_scene"})

        self.assertEqual(
            preview._queue.get_nowait(),
            {
                "type": "static_scene",
                "subject": "Yuri",
                "current_trial_num": 17,
                "total_trials": 2000,
            },
        )

    def test_preview_queue_includes_generic_status_counts(self):
        preview = object.__new__(ExperimenterPreview)
        preview.poll = lambda: False
        preview._process = Mock()
        preview._process.is_alive.return_value = True
        preview._queue = queue.Queue()
        preview.status_counts = {
            "Correct": 4,
            "Incorrect": 2,
            "Rewards delivered": 3,
        }

        preview._send({"type": "static_scene"})

        self.assertEqual(
            preview._queue.get_nowait()["status_counts"],
            {
                "Correct": 4,
                "Incorrect": 2,
                "Rewards delivered": 3,
            },
        )

    def test_experimenter_label_shows_subject_and_trial_total(self):
        self.assertEqual(
            format_experimenter_label(
                "csc2_foraging_classic",
                subject="Yuri",
                current_trial_num=17,
                total_trials=2000,
            ),
            "csc2_foraging_classic\nSubject: Yuri\nTrial: 17 / 2000",
        )

    def test_experimenter_label_marks_indefinite_trial_total(self):
        self.assertEqual(
            format_experimenter_label(
                "csc2_foraging_classic",
                subject="Yuri",
                current_trial_num=17,
                total_trials=0,
            ),
            "csc2_foraging_classic\nSubject: Yuri\nTrial: 17 / ∞",
        )

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

    def test_main_screen_uses_os_geometry(self):
        screens = [
            ScreenGeometry(index=0, x=0, y=0, width=1440, height=900, name="DP-1"),
            ScreenGeometry(index=1, x=1440, y=0, width=1920, height=1080, name="HDMI-1"),
        ]

        with patch("bin.screen.get_monitor_screens", return_value=screens):
            main, secondary = resolve_task_screens(
                {"main": "DP-1", "experimenter": "HDMI-1"}
            )

        self.assertEqual((main.width, main.height), (1440, 900))
        self.assertEqual((secondary.width, secondary.height), (1920, 1080))

    def test_main_screen_size_falls_back_only_when_os_query_fails(self):
        with patch("bin.screen.get_monitor_screens", return_value=[]):
            main, secondary = resolve_task_screens(
                {"main": "HDMI-2", "experimenter": "HDMI-1"}
            )

        self.assertEqual((main.width, main.height), (1600, 2560))
        self.assertEqual(main.name, "HDMI-2")
        self.assertIsNone(secondary)

    def test_hdmi_names_do_not_cross_match_by_off_by_one_alias(self):
        screens = [
            ScreenGeometry(index=0, x=0, y=0, width=800, height=480, name="HDMI-A-1"),
            ScreenGeometry(index=1, x=800, y=0, width=800, height=480, name="HDMI-A-2"),
        ]

        self.assertEqual(select_screen(screens, "HDMI-1", role="main").index, 0)
        self.assertEqual(select_screen(screens, "HDMI-2", role="experimenter").index, 1)

    def test_psychopy_screen_order_is_resolved_by_geometry(self):
        screen = ScreenGeometry(index=0, x=1920, y=0, width=800, height=480, name="HDMI-2")
        screens = [
            types.SimpleNamespace(x=0, y=0, width=1920, height=1080),
            types.SimpleNamespace(x=1920, y=0, width=800, height=480),
        ]
        display = types.SimpleNamespace(get_screens=lambda: screens)
        captured = {}
        win = types.SimpleNamespace(
            winHandle=types.SimpleNamespace(
                get_location=lambda: (1920, 0),
                get_size=lambda: (800, 480),
            ),
            close=Mock(),
        )
        visual = types.SimpleNamespace(
            Window=lambda **kwargs: captured.update(kwargs) or win
        )

        with (
            patch("bin.screen.sys.platform", "darwin"),
            patch("bin.screen._get_pyglet_display", return_value=display),
        ):
            open_psychopy_window(visual, screen, fullscreen=True)

        self.assertEqual(captured["screen"], 1)
        self.assertTrue(captured["fullscr"])

    def test_psychopy_window_resolves_os_screens_when_not_supplied(self):
        main = ScreenGeometry(
            index=0,
            x=0,
            y=0,
            width=1440,
            height=900,
            name="DP-1",
        )
        secondary = ScreenGeometry(
            index=1,
            x=1440,
            y=0,
            width=1920,
            height=1080,
            name="HDMI-1",
        )
        display = types.SimpleNamespace(
            get_screens=lambda: [
                types.SimpleNamespace(x=0, y=0, width=1440, height=900),
                types.SimpleNamespace(x=1440, y=0, width=1920, height=1080),
            ]
        )
        captured = {}
        win = types.SimpleNamespace(
            winHandle=types.SimpleNamespace(
                get_location=lambda: (0, 0),
                get_size=lambda: (1440, 900),
            ),
            close=Mock(),
        )
        visual = types.SimpleNamespace(
            Window=lambda **kwargs: captured.update(kwargs) or win
        )

        with (
            patch("bin.screen.sys.platform", "darwin"),
            patch("bin.screen.get_monitor_screens", return_value=[main, secondary]),
            patch("bin.screen._get_pyglet_display", return_value=display),
            patch.dict(os.environ, {}, clear=True),
        ):
            open_psychopy_window(visual, None, fullscreen=True)

        self.assertEqual(captured["size"], (1440, 900))
        self.assertEqual(captured["screen"], 0)

    def test_linux_fullscreen_uses_native_psychopy_path_and_original_monitor_index(self):
        screen = ScreenGeometry(
            index=0,
            x=0,
            y=0,
            width=1600,
            height=2560,
            name="HDMI-2",
        )
        captured = {}
        win = types.SimpleNamespace(
            winHandle=types.SimpleNamespace(
                _window=42,
                _x_screen_id=0,
                get_location=lambda: (0, 0),
                get_size=lambda: (1600, 2560),
            ),
            close=Mock(),
        )
        visual = types.SimpleNamespace(
            Window=lambda **kwargs: captured.update(kwargs) or win
        )

        with (
            patch("bin.screen.sys.platform", "linux"),
            patch(
                "bin.screen._bind_linux_pyglet_display",
                return_value=nullcontext(
                    {
                        "monitor_index": 1,
                        "selected_rect": (0, 0, 1600, 2560),
                        "available_rects": [
                            (1600, 0, 1920, 1080),
                            (0, 0, 1600, 2560),
                        ],
                    }
                ),
            ),
            patch(
                "bin.screen._place_native_x11_fullscreen",
                return_value="HDMI-2 at (0, 0, 1600, 2560)",
            ) as place_fullscreen,
        ):
            opened = open_psychopy_window(
                visual,
                screen,
                fullscreen=True,
            )

        self.assertTrue(captured["fullscr"])
        self.assertEqual(captured["screen"], 0)
        self.assertNotIn("size", captured)
        self.assertNotIn("pos", captured)
        self.assertNotIn("checkTiming", captured)
        self.assertEqual(
            opened._neuro_tasks_fullscreen_path,
            "native managed PsychoPy fullscreen",
        )
        self.assertIn("Xinerama monitor 1", opened._neuro_tasks_pyglet_selection)
        self.assertIsNone(win._neuro_tasks_screen_placement_error)
        place_fullscreen.assert_called_once_with(win, screen, 1)

    def test_x11_fullscreen_monitor_request_uses_original_xinerama_index(self):
        root = Mock()
        connection = Mock()
        connection.screen.return_value.root = root
        connection.intern_atom.return_value = 10
        event = object()
        client_message = Mock(return_value=event)
        xlib = types.ModuleType("Xlib")
        xlib.X = types.SimpleNamespace(
            SubstructureRedirectMask=1,
            SubstructureNotifyMask=2,
        )
        xlib.display = types.SimpleNamespace(Display=Mock(return_value=connection))
        xlib.protocol = types.SimpleNamespace(
            event=types.SimpleNamespace(ClientMessage=client_message)
        )
        win = types.SimpleNamespace(
            winHandle=types.SimpleNamespace(_window=42, _x_screen_id=0)
        )

        with patch.dict("sys.modules", {"Xlib": xlib}):
            _request_x11_fullscreen_monitor(win, 1)

        client_message.assert_called_once_with(
            window=42,
            client_type=10,
            data=(32, [1, 1, 1, 1, 1]),
        )
        root.send_event.assert_called_once_with(event, event_mask=3)
        connection.sync.assert_called_once_with()
        connection.close.assert_called_once_with()

    def test_linux_placement_uses_native_geometry_not_pyglet_cache(self):
        screen = ScreenGeometry(
            index=0,
            x=0,
            y=0,
            width=1600,
            height=2560,
            name="HDMI-2",
        )
        win = types.SimpleNamespace(
            winHandle=types.SimpleNamespace(
                get_location=lambda: (0, 0),
                get_size=lambda: (1920, 1080),
            )
        )

        with (
            patch("bin.screen.sys.platform", "linux"),
            patch("bin.screen._x11_window_rect", return_value=(0, 0, 1600, 2560)),
        ):
            placement = verify_psychopy_window_screen(win, screen)

        self.assertEqual(placement, "HDMI-2 at (0, 0, 1600, 2560)")

    def test_psychopy_screen_requires_an_exact_geometry_match(self):
        screen = ScreenGeometry(index=0, x=1920, y=0, width=800, height=480, name="HDMI-2")
        display = types.SimpleNamespace(
            get_screens=lambda: [
                types.SimpleNamespace(x=0, y=0, width=1920, height=1080)
            ]
        )
        visual = types.SimpleNamespace(Window=Mock())

        with (
            patch("bin.screen.sys.platform", "darwin"),
            patch("bin.screen._get_pyglet_display", return_value=display),
            self.assertRaisesRegex(RuntimeError, "could not uniquely match output HDMI-2"),
        ):
            open_psychopy_window(visual, screen, fullscreen=True)

        visual.Window.assert_not_called()

    def test_realized_psychopy_window_must_cover_requested_screen(self):
        screen = ScreenGeometry(index=0, x=1920, y=0, width=800, height=480, name="HDMI-2")
        win = types.SimpleNamespace(
            winHandle=types.SimpleNamespace(
                get_location=lambda: (0, 0),
                get_size=lambda: (1920, 1080),
            )
        )

        with (
            patch("bin.screen.sys.platform", "darwin"),
            self.assertRaisesRegex(RuntimeError, "not main output HDMI-2"),
        ):
            verify_psychopy_window_screen(win, screen)

    def test_realized_psychopy_window_confirms_requested_screen(self):
        screen = ScreenGeometry(index=0, x=1920, y=0, width=800, height=480, name="HDMI-2")
        win = types.SimpleNamespace(
            winHandle=types.SimpleNamespace(
                get_location=lambda: (1920, 0),
                get_size=lambda: (800, 480),
            )
        )

        with patch("bin.screen.sys.platform", "darwin"):
            self.assertEqual(
                verify_psychopy_window_screen(win, screen),
                "HDMI-2 at (1920, 0, 800, 480)",
            )

    def test_tk_window_is_positioned_before_true_fullscreen(self):
        screen = ScreenGeometry(
            index=1, x=800, y=0, width=2560, height=1600, name="HDMI-A-2"
        )
        window = Mock()

        set_tk_window_fullscreen(window, screen)

        window.geometry.assert_called_once_with("2560x1600+800+0")
        window.update_idletasks.assert_called_once_with()
        window.attributes.assert_called_once_with("-fullscreen", True)

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

    def test_aspect_cover_uniformly_scales_and_crops_landscape_video(self):
        draw_width, draw_height = compute_aspect_cover_size(
            (1080, 1920),
            (1920, 1080),
        )

        self.assertAlmostEqual(draw_height, 1920.0)
        self.assertGreater(draw_width, 1080.0)
        self.assertAlmostEqual(draw_width / draw_height, 1920.0 / 1080.0)

    def test_aspect_cover_uniformly_scales_and_crops_portrait_video(self):
        draw_width, draw_height = compute_aspect_cover_size(
            (1920, 1080),
            (1080, 1920),
        )

        self.assertAlmostEqual(draw_width, 1920.0)
        self.assertGreater(draw_height, 1080.0)
        self.assertAlmostEqual(draw_width / draw_height, 1080.0 / 1920.0)

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
