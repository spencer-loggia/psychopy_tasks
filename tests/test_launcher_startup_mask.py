from contextlib import ExitStack
import os
from pathlib import Path
import subprocess
import unittest
from unittest.mock import Mock, patch

from bin.screen import ScreenGeometry
from interface.touch_interface import TouchInterfaceApp
from interface.x11_idle_guard import XInputControlError, XInputTouchscreen


class LauncherStartupMaskTests(unittest.TestCase):
    def setUp(self):
        self.operator = ScreenGeometry(0, 0, 0, 800, 480, "DSI-1")
        self.subject = ScreenGeometry(1, 800, 0, 2560, 1600, "HDMI-1")
        self.cfg = {
            "screens": {"main": None, "experimenter": None},
            "mask_main_inputs": True,
            "main_touchscreen_xinput": "WaveShare WaveShare",
            "screen_mask_rgb": [174, 165, 163],
            "sync_bar": {"enabled": True, "height": 200, "sync_pin": 16},
            "environment": {"python": "python", "working_dir": str(Path.cwd())},
            "subjects": {"Test": "TST"},
            "tasks": {},
        }
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.dict(os.environ, {
            "DISPLAY": ":0", "XDG_SESSION_TYPE": "x11",
            "MAIN_SCREEN": "HDMI-1", "SECONDARY_SCREEN": "DSI-1",
        }, clear=True))
        self.stack.enter_context(patch("interface.x11_idle_guard.sys.platform", "linux"))
        self.screens = self.stack.enter_context(patch(
            "bin.screen.get_monitor_screens", return_value=[self.operator, self.subject],
        ))
        self.touchscreen_cls = self.stack.enter_context(patch("interface.x11_idle_guard.XInputTouchscreen"))
        self.touchscreen = self.touchscreen_cls.return_value
        self.tk = self.stack.enter_context(patch("interface.touch_interface.tk"))
        self.build_ui = self.stack.enter_context(patch.object(TouchInterfaceApp, "_build_ui"))
        self.startup = self.stack.enter_context(patch.object(TouchInterfaceApp, "startup"))
        self.mode_setup = self.stack.enter_context(patch.object(TouchInterfaceApp, "_initialize_is_rig_mode"))
        self.warning = self.stack.enter_context(patch("interface.touch_interface.messagebox.showwarning"))
        self.error = self.stack.enter_context(patch("interface.touch_interface.messagebox.showerror"))
        self.stack.enter_context(patch("builtins.print"))

    def app(self, **kwargs):
        return TouchInterfaceApp(
            Mock(), Path("/rig/launch.json"), self.cfg,
            screen_info=self.operator, **kwargs,
        )

    def assert_debug_opens(self, app):
        with patch("interface.touch_interface.HardwareDebugPanel") as panel:
            app._open_debug_mode()
        panel.assert_called_once()
        self.warning.assert_not_called()
        self.error.assert_not_called()
        self.startup.assert_not_called()
        self.mode_setup.assert_not_called()

    def test_rig_and_portable_startup_show_mask_and_static_bar_before_subject_selection(self):
        configurations = (
            ("1", self.operator, self.subject),
            ("0", self.subject, self.operator),
        )
        for mode, operator, subject in configurations:
            with self.subTest(mode=mode):
                os.environ["IS_RIG"] = mode
                os.environ["MAIN_SCREEN"] = subject.name
                os.environ["SECONDARY_SCREEN"] = operator.name
                self.operator = operator
                self.tk.reset_mock()
                self.touchscreen.reset_mock()
                events = []
                self.build_ui.side_effect = lambda: events.append("menu")
                self.touchscreen.disable.side_effect = lambda: events.append("disable")
                self.tk.Toplevel.return_value.deiconify.side_effect = lambda: events.append("mask")

                app = self.app()

                self.assertEqual(events, ["menu", "disable", "mask"])
                self.assertIsNone(app.experiment)
                self.assertTrue(app.idle_guard.idle)
                self.assertEqual(app.is_rig, mode)
                self.touchscreen_cls.assert_called_with(
                    device_name="WaveShare WaveShare", output_name=subject.name,
                )
                self.tk.Toplevel.return_value.configure.assert_called_with(
                    bg="#aea5a3", cursor="none", takefocus=True,
                )
                self.tk.Toplevel.return_value.geometry.assert_called_with(
                    f"{subject.width}x{subject.height}{subject.x:+d}{subject.y:+d}",
                )
                self.assertEqual(self.tk.Canvas.return_value.create_rectangle.call_count, 5)
                self.tk.Canvas.return_value.after.assert_not_called()
                self.assert_debug_opens(app)

    def test_disabled_mask_does_not_attempt_display_or_touchscreen_setup(self):
        self.cfg["mask_main_inputs"] = False
        app = self.app()
        self.assertIsNone(app.idle_guard)
        self.screens.assert_not_called()
        self.touchscreen_cls.assert_not_called()
        self.tk.Toplevel.assert_not_called()
        self.assert_debug_opens(app)

    def test_missing_main_display_leaves_recovery_display_uncovered(self):
        self.screens.return_value = [self.operator]
        app = self.app()
        self.assertIsNone(app.idle_guard)
        self.touchscreen_cls.assert_not_called()
        self.tk.Toplevel.assert_not_called()
        self.assert_debug_opens(app)

    def test_same_screen_configuration_does_not_mask_operator(self):
        self.cfg["screens"] = {"main": "DSI-1", "experimenter": "DSI-1"}
        app = self.app()
        self.assertIsNone(app.idle_guard)
        self.touchscreen_cls.assert_not_called()
        self.assert_debug_opens(app)

    def test_stale_second_screen_index_cannot_cover_fallback_operator_screen(self):
        fake_secondary = ScreenGeometry(9, 0, 0, 0, 0, "screen9")
        with patch("interface.touch_interface.resolve_task_screens", return_value=(self.operator, fake_secondary)):
            app = self.app()
        self.touchscreen_cls.assert_not_called()
        self.tk.Toplevel.assert_not_called()
        self.assert_debug_opens(app)

    def test_unreadable_config_skips_optional_mask_setup(self):
        app = self.app(config_error="Invalid JSON")
        self.screens.assert_not_called()
        self.touchscreen_cls.assert_not_called()
        self.assert_debug_opens(app)

    def test_invalid_mask_settings_do_not_prevent_debug(self):
        self.cfg["screen_mask_rgb"] = "broken"
        app = self.app()
        self.assertIsNone(app.idle_guard)
        self.touchscreen.disable.assert_not_called()
        self.tk.Toplevel.assert_not_called()
        self.assert_debug_opens(app)

    def test_unavailable_touchscreen_does_not_prevent_debug(self):
        self.touchscreen.validate.side_effect = XInputControlError("touchscreen disconnected")
        app = self.app()
        self.assertIsNone(app.idle_guard)
        self.tk.Toplevel.assert_not_called()
        self.assert_debug_opens(app)

    def test_failed_initial_mask_is_retried_by_explicit_experiment_setup(self):
        self.touchscreen.disable.side_effect = XInputControlError("temporary XInput failure")
        app = self.app()
        self.assertFalse(app.idle_guard.idle)
        self.touchscreen.enable_for_desktop.assert_called_once()
        self.assert_debug_opens(app)
        app.debug_panel = None
        app._startup_done = True
        app.quiet_mode = Mock()
        self.touchscreen.disable.side_effect = None

        self.assertTrue(app._prepare_experiment_environment())

        self.assertTrue(app.idle_guard.idle)
        self.assertEqual(self.touchscreen.disable.call_count, 2)
        self.touchscreen_cls.assert_called_once()

    def test_experiment_setup_reuses_successful_startup_mask(self):
        app = self.app()
        guard = app.idle_guard
        app._startup_done = True
        app.quiet_mode = Mock()

        self.assertTrue(app._prepare_experiment_environment())

        self.assertIs(app.idle_guard, guard)
        self.tk.Toplevel.assert_called_once()
        self.touchscreen.disable.assert_called_once()


class XInputTimeoutTests(unittest.TestCase):
    def test_stalled_xinput_is_bounded_and_reported_as_a_setup_failure(self):
        command = ["xinput", "list", "--id-only", "Main Touchscreen"]
        runner = Mock(side_effect=subprocess.TimeoutExpired(command, 2))
        touchscreen = XInputTouchscreen("Main Touchscreen", "HDMI-1", runner=runner)
        with self.assertRaisesRegex(XInputControlError, "timed out after 2s"):
            touchscreen.validate()
        self.assertEqual(runner.call_args.kwargs["timeout"], 2.0)


if __name__ == "__main__":
    unittest.main()
