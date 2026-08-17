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
    SCREEN_ENV_OVERRIDE_ENV,
    SECONDARY_SCREEN_ENV,
    ScreenGeometry,
    _bind_linux_pyglet_display,
    _parse_xrandr_listactivemonitors,
    _parse_xrandr_query,
    build_reward_hit_boxes,
    compute_aspect_cover_size,
    compute_centered_aspect_fit,
    configure_window_vsync,
    enforce_window_vsync,
    format_experimenter_label,
    load_screen_config,
    measure_window_flip_rate,
    open_psychopy_window,
    oriented_size,
    reward_level_color,
    rotate_centered_point,
    resolve_window_frame_rate,
    resolve_task_screens,
    resolve_scene_size,
    scale_scene_point,
    select_screen,
    set_tk_window_fullscreen,
    software_stimulus_rotation,
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
        pyglet.options = {"xlib_fullscreen_override_redirect": False}

        with patch.dict("sys.modules", {"pyglet": pyglet}):
            with _bind_linux_pyglet_display(target_info) as selection:
                self.assertFalse(pyglet.options["xlib_fullscreen_override_redirect"])
                display = pyglet.canvas.Display(x_screen=0)
                self.assertIs(display.get_screens()[0], target)

        self.assertIsNone(display._screens)
        self.assertEqual(selection["monitor_index"], 1)
        self.assertEqual(selection["selected_rect"], (0, 0, 1600, 2560))
        self.assertEqual(
            selection["available_rects"],
            [(1600, 0, 1920, 1080), (0, 0, 1600, 2560)],
        )
        self.assertFalse(pyglet.options["xlib_fullscreen_override_redirect"])
        self.assertIs(pyglet.canvas.Display, FakeDisplay)

        with patch.dict("sys.modules", {"pyglet": pyglet}):
            with _bind_linux_pyglet_display(target_info, wm_managed=False):
                self.assertTrue(pyglet.options["xlib_fullscreen_override_redirect"])
                pyglet.canvas.Display(x_screen=0)

        self.assertFalse(pyglet.options["xlib_fullscreen_override_redirect"])

    def test_psychopy_display_preselection_is_best_effort_for_every_caller(self):
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
        pyglet.options = {"xlib_fullscreen_override_redirect": False}

        with patch.dict("sys.modules", {"pyglet": pyglet}):
            with _bind_linux_pyglet_display(target_info) as selection:
                display = pyglet.canvas.Display(x_screen=0)

        self.assertEqual(display.get_screens(), [other])
        self.assertIn("could not uniquely match", selection["preselection_error"])

    def test_psychopy_display_preselection_handles_cached_get_display(self):
        target_info = ScreenGeometry(
            index=1, x=0, y=0, width=2560, height=1600, name="HDMI-2"
        )
        other = types.SimpleNamespace(x=2560, y=0, width=1920, height=1080)
        target = types.SimpleNamespace(x=0, y=0, width=2560, height=1600)

        class FakeDisplay:
            def __init__(self):
                self._screens = None

            def get_screens(self):
                return self._screens or [other, target]

        cached_display = FakeDisplay()

        def get_display():
            return cached_display

        pyglet = types.ModuleType("pyglet")
        pyglet.canvas = types.SimpleNamespace(
            Display=FakeDisplay,
            get_display=get_display,
        )
        pyglet.options = {}

        with patch.dict("sys.modules", {"pyglet": pyglet}):
            with _bind_linux_pyglet_display(target_info):
                self.assertIs(pyglet.canvas.get_display().get_screens()[0], target)

        self.assertIsNone(cached_display._screens)
        self.assertIs(pyglet.canvas.get_display, get_display)
        self.assertNotIn("xlib_fullscreen_override_redirect", pyglet.options)

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

    def test_configure_window_vsync_deduplicates_shared_backend_handle(self):
        target = Mock()
        target.set_vsync = Mock()
        win = Mock()
        win.winHandle = target
        win.backend = target

        self.assertTrue(configure_window_vsync(win, True))
        target.set_vsync.assert_called_once_with(True)

    def test_refresh_override_is_measured_and_compared(self):
        win = Mock()
        logger = Mock()

        with patch("bin.screen.measure_window_flip_rate", return_value=59.94):
            fps, frame_duration = resolve_window_frame_rate(
                win,
                configured_fps=60.0,
                msg_logger=logger,
                context="match2cue",
            )

        self.assertEqual(fps, 59.94)
        self.assertAlmostEqual(frame_duration, 1.0 / 59.94)
        messages = [call.args[1] for call in logger.log.call_args_list]
        self.assertTrue(any("status=match" in message for message in messages))

    def test_low_overhead_flip_rate_uses_post_flip_timestamps(self):
        win = Mock()
        with patch(
            "bin.screen.time.perf_counter",
            side_effect=[0.0, 1.0 / 60.0, 2.0 / 60.0, 3.0 / 60.0],
        ):
            measured = measure_window_flip_rate(
                win,
                warmup_frames=2,
                sample_frames=3,
            )

        self.assertAlmostEqual(measured, 60.0)
        self.assertEqual(win.flip.call_count, 6)

    def test_measured_refresh_remains_authoritative_when_config_mismatches(self):
        win = Mock()
        logger = Mock()

        with patch("bin.screen.measure_window_flip_rate", return_value=28.3):
            fps, _ = resolve_window_frame_rate(
                win,
                configured_fps=60.0,
                msg_logger=logger,
                context="active_foraging",
            )

        self.assertEqual(fps, 28.3)
        logger.log.assert_any_call(
            "WARN",
            (
                "refresh_rate_comparison context=active_foraging "
                "configured_fps=60.000000 measured_fps=28.300000 "
                "difference_hz=31.700000 tolerance_hz=0.600000 status=mismatch"
            ),
        )

    def test_refresh_override_is_used_when_measurement_is_unavailable(self):
        win = Mock()

        with patch("bin.screen.measure_window_flip_rate", return_value=None):
            fps, frame_duration = resolve_window_frame_rate(
                win,
                configured_fps=75.0,
            )

        self.assertEqual(fps, 75.0)
        self.assertAlmostEqual(frame_duration, 1.0 / 75.0)

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

    def test_software_stimulus_rotation_tracks_native_output_rotation(self):
        self.assertEqual(software_stimulus_rotation(None), 90)
        self.assertEqual(software_stimulus_rotation("normal"), 90)
        self.assertEqual(software_stimulus_rotation("unknown"), 90)
        self.assertEqual(software_stimulus_rotation("right"), 0)
        self.assertEqual(software_stimulus_rotation("inverted"), 270)
        self.assertEqual(software_stimulus_rotation("left"), 180)

    def test_oriented_size_swaps_axes_for_odd_quarter_turns(self):
        self.assertEqual(oriented_size((2560, 1600), 0), (2560.0, 1600.0))
        self.assertEqual(oriented_size((2560, 1600), 90), (1600.0, 2560.0))
        self.assertEqual(oriented_size((2560, 1600), -90), (1600.0, 2560.0))
        self.assertEqual(oriented_size((2560, 1600), 180), (2560.0, 1600.0))

    def test_rotate_centered_point_uses_positive_clockwise_turns(self):
        self.assertEqual(rotate_centered_point((10, 20), 90), (20.0, -10.0))
        self.assertEqual(rotate_centered_point((10, 20), 180), (-10.0, -20.0))
        self.assertEqual(rotate_centered_point((10, 20), 270), (-20.0, 10.0))
        native = rotate_centered_point((-300, 700), 90)
        self.assertEqual(
            rotate_centered_point(native, -90),
            (-300.0, 700.0),
        )

    def test_subject_preview_transposes_entire_native_frame(self):
        native_size = (2560, 1600)
        subject_size = oriented_size(native_size, 90)
        native_corners = {
            (-1280, 800),
            (1280, 800),
            (-1280, -800),
            (1280, -800),
        }

        subject_corners = {
            rotate_centered_point(point, -90) for point in native_corners
        }

        self.assertEqual(subject_size, (1600.0, 2560.0))
        self.assertEqual(
            subject_corners,
            {
                (-800.0, -1280.0),
                (-800.0, 1280.0),
                (800.0, -1280.0),
                (800.0, 1280.0),
            },
        )

    def test_quarter_turn_geometry_rejects_arbitrary_angles(self):
        with self.assertRaisesRegex(ValueError, "multiple of 90"):
            oriented_size((2560, 1600), 45)

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

    def test_preview_static_and_shared_video_payloads_include_rotation(self):
        preview = object.__new__(ExperimenterPreview)
        preview.poll = lambda: False
        preview._process = Mock()
        preview._process.is_alive.return_value = True
        preview._queue = queue.Queue()

        preview.show_static_scene(
            bg_rgb_255=(0, 0, 0),
            main_size=(2560, 1600),
        )
        static_payload = preview._queue.get_nowait()
        self.assertEqual(static_payload["main_rotation_deg"], 0)

        preview.play_shared_video(
            shared_frame_buffer={"name": "frames", "maximum_frame_bytes": 64},
            minimum_sequence=4,
            video_size=(1920, 1080),
            bg_rgb_255=(0, 0, 0),
            main_size=(2560, 1600),
            main_rotation_deg=90,
        )
        shared_payload = preview._queue.get_nowait()
        self.assertEqual(shared_payload["main_rotation_deg"], 90)
        self.assertEqual(shared_payload["main_size"], [2560, 1600])

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
                SCREEN_ENV_OVERRIDE_ENV: "0",
            },
        ):
            self.assertEqual(load_screen_config(cfg), {"main": 1, "experimenter": "DSI-2"})

    def test_flagged_screen_environment_overrides_config_and_preserves_empty_secondary(self):
        cfg = {"screens": {"main": "HDMI-2", "experimenter": "DSI-2"}}

        with patch.dict(
            os.environ,
            {
                MAIN_SCREEN_ENV: "HDMI-1",
                SECONDARY_SCREEN_ENV: "",
                SCREEN_ENV_OVERRIDE_ENV: "yes",
            },
            clear=True,
        ):
            self.assertEqual(
                load_screen_config(cfg),
                {"main": "HDMI-1", "experimenter": None},
            )

    def test_cli_screen_selectors_remain_above_flagged_environment(self):
        cfg = {"screens": {"main": "HDMI-2", "experimenter": "DSI-2"}}

        with patch.dict(
            os.environ,
            {
                MAIN_SCREEN_ENV: "HDMI-1",
                SECONDARY_SCREEN_ENV: "DSI-1",
                SCREEN_ENV_OVERRIDE_ENV: "true",
            },
            clear=True,
        ):
            self.assertEqual(
                load_screen_config(
                    cfg,
                    cli_main="DP-1",
                    cli_experimenter=3,
                ),
                {"main": "DP-1", "experimenter": 3},
            )

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

        self.assertEqual((main.width, main.height), (2560, 1600))
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
            ) as bind_display,
            patch(
                "bin.screen._wait_for_psychopy_window_screen",
                return_value="HDMI-2 at (0, 0, 1600, 2560)",
            ) as wait_for_screen,
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
        self.assertFalse(captured["checkTiming"])
        self.assertEqual(
            opened._neuro_tasks_fullscreen_path,
            "native managed PsychoPy fullscreen",
        )
        self.assertIn("Xinerama monitor 1", opened._neuro_tasks_pyglet_selection)
        self.assertIsNone(win._neuro_tasks_screen_placement_error)
        bind_display.assert_called_once_with(
            screen,
            wm_managed=True,
        )
        wait_for_screen.assert_called_once_with(win, screen)

    def test_window_callback_precedes_activation_and_final_verification(self):
        events = []
        screen = ScreenGeometry(
            index=0,
            x=0,
            y=0,
            width=2560,
            height=1600,
            name="HDMI-2",
        )
        handle = types.SimpleNamespace(
            activate=lambda: events.append("activate"),
            get_location=lambda: (0, 0),
            get_size=lambda: (2560, 1600),
        )
        win = types.SimpleNamespace(winHandle=handle, close=Mock())

        def create_window(**kwargs):
            events.append("construct")
            return win

        def verify_window(created_win, expected_screen):
            events.append("verify")
            return "HDMI-2 at (0, 0, 2560, 1600)"

        with (
            patch("bin.screen.sys.platform", "linux"),
            patch(
                "bin.screen._bind_linux_pyglet_display",
                return_value=nullcontext({}),
            ),
            patch(
                "bin.screen._wait_for_psychopy_window_screen",
                side_effect=verify_window,
            ),
        ):
            open_psychopy_window(
                types.SimpleNamespace(Window=create_window),
                screen,
                fullscreen=True,
                on_window_created=lambda created_win: events.append("callback"),
            )

        self.assertEqual(events, ["construct", "callback", "activate", "verify"])

    def test_window_callback_failure_closes_unverified_window(self):
        screen = ScreenGeometry(
            index=0, x=0, y=0, width=2560, height=1600, name="HDMI-2"
        )
        win = types.SimpleNamespace(
            winHandle=types.SimpleNamespace(),
            close=Mock(),
        )
        verify = Mock()

        def fail_callback(created_win):
            raise RuntimeError("release failed")

        with (
            patch("bin.screen.sys.platform", "linux"),
            patch(
                "bin.screen._bind_linux_pyglet_display",
                return_value=nullcontext({}),
            ),
            patch("bin.screen._wait_for_psychopy_window_screen", verify),
            self.assertRaisesRegex(RuntimeError, "release failed"),
        ):
            open_psychopy_window(
                types.SimpleNamespace(Window=lambda **kwargs: win),
                screen,
                fullscreen=True,
                on_window_created=fail_callback,
            )

        win.close.assert_called_once_with()
        verify.assert_not_called()

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

    def test_psychopy_preselection_failure_defers_to_realized_window_check(self):
        screen = ScreenGeometry(index=0, x=1920, y=0, width=800, height=480, name="HDMI-2")
        display = types.SimpleNamespace(
            get_screens=lambda: [
                types.SimpleNamespace(x=0, y=0, width=1920, height=1080)
            ]
        )
        win = types.SimpleNamespace(
            winHandle=types.SimpleNamespace(
                get_location=lambda: (0, 0),
                get_size=lambda: (1920, 1080),
            ),
            close=Mock(),
        )
        visual = types.SimpleNamespace(Window=Mock(return_value=win))

        with (
            patch("bin.screen.sys.platform", "darwin"),
            patch("bin.screen._get_pyglet_display", return_value=display),
            self.assertRaisesRegex(RuntimeError, "not main output HDMI-2"),
        ):
            open_psychopy_window(visual, screen, fullscreen=True)

        visual.Window.assert_called_once()
        win.close.assert_called_once_with()

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
