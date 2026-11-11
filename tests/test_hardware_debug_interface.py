import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from bin.screen import ScreenGeometry
from interface.hardware_debug import HardwareDebugPanel, debug_settings
from interface.touch_interface import (
    CleanupResult, TouchInterfaceApp, load_launcher_config, main, parse_args,
    resolve_launcher_screen,
)


class DebugLauncherTests(unittest.TestCase):
    def create_app(self, cfg=None, **kwargs):
        root = Mock()
        with patch("interface.touch_interface.tk.StringVar"), patch.object(TouchInterfaceApp, "_build_ui"):
            return TouchInterfaceApp(root, Path("missing.json"), cfg or {}, screen_info=ScreenGeometry(0, 0, 0, 800, 480, "DSI-1"), **kwargs)

    def test_menu_construction_does_not_start_experiment_checks_or_modal_warnings(self):
        cfg = {"subjects": [], "environment": None, "sync_bar": {"height": -1}, "screens": "invalid", "experiment_quiet_mode": "bad", "eye_tracker_experiment": {"enabled": True}}
        with patch("interface.touch_interface.resolve_task_screens") as screens, patch("interface.touch_interface.create_experiment_idle_guard") as guard, patch("interface.touch_interface.create_experiment_quiet_mode") as quiet, patch.object(TouchInterfaceApp, "_initialize_is_rig_mode") as mode, patch.object(TouchInterfaceApp, "startup") as startup, patch("interface.touch_interface.messagebox.showwarning") as warning:
            app = self.create_app(cfg)
        self.assertIsNone(app.experiment)
        for method in (screens, guard, quiet, mode, startup, warning):
            method.assert_not_called()

    def test_debug_opens_despite_invalid_config_and_failed_previous_screen_guard(self):
        app = self.create_app(config_error="Broken JSON")
        app.idle_guard = Mock()
        app.idle_guard.release_for_desktop.side_effect = RuntimeError("unplugged display")
        with patch("interface.touch_interface.HardwareDebugPanel") as panel, patch.object(app, "_prepare_experiment_environment") as prepare:
            app._open_debug_mode()
        panel.assert_called_once()
        prepare.assert_not_called()
        app.idle_guard.release_for_desktop.assert_not_called()
        self.assertIs(app.debug_panel, panel.return_value)

    def test_missing_main_screen_does_not_override_valid_experimenter_selector(self):
        screen = ScreenGeometry(0, 0, 0, 800, 480, "DSI-1")
        with patch.dict(os.environ, {}, clear=True), patch("interface.touch_interface.get_tk_screens", return_value=[screen]), patch("interface.touch_interface.resolve_task_screens", side_effect=AssertionError("must not resolve task screens")):
            result = resolve_launcher_screen(Mock(), {"screens": {"main": None, "experimenter": "DSI-1"}})
        self.assertEqual(result, screen)

    def test_stale_experimenter_index_falls_back_to_only_attached_screen(self):
        screen = ScreenGeometry(0, 0, 0, 800, 480, "DSI-1")
        with patch.dict(os.environ, {}, clear=True), patch("interface.touch_interface.get_tk_screens", return_value=[screen]), patch("builtins.print"):
            result = resolve_launcher_screen(Mock(), {"screens": {"main": "HDMI-absent", "experimenter": 1}})
        self.assertEqual(result, screen)

    def test_broken_display_discovery_uses_tk_screen(self):
        root = Mock()
        root.winfo_screenwidth.return_value = 800
        root.winfo_screenheight.return_value = 480
        with patch("interface.touch_interface.get_tk_screens", side_effect=OSError("xrandr failed")), patch("builtins.print"):
            screen = resolve_launcher_screen(root, {"screens": []})
        self.assertEqual((screen.width, screen.height), (800, 480))

    def test_invalid_config_selector_does_not_hide_valid_secondary_environment(self):
        secondary = ScreenGeometry(0, 0, 0, 800, 480, "DSI-1")
        main_screen = ScreenGeometry(1, 800, 0, 1920, 1080, "HDMI-1")
        for selector in ("unplugged", [], {}, True, -1):
            with self.subTest(selector=selector), patch.dict(os.environ, {"SECONDARY_SCREEN": "DSI-1"}, clear=True), patch("interface.touch_interface.get_tk_screens", return_value=[secondary, main_screen]), patch("builtins.print"):
                result = resolve_launcher_screen(Mock(), {"screens": {"experimenter": selector}})
                self.assertEqual(result, secondary)

    def test_invalid_cli_selector_falls_back_to_configured_secondary(self):
        secondary = ScreenGeometry(0, 0, 0, 800, 480, "DSI-1")
        with patch.dict(os.environ, {}, clear=True), patch("interface.touch_interface.get_tk_screens", return_value=[secondary]), patch("builtins.print"):
            result = resolve_launcher_screen(Mock(), {"screens": {"experimenter": "DSI-1"}}, experimenter_selector="99")
        self.assertEqual(result, secondary)

    def test_menu_does_not_probe_broken_configured_working_directory(self):
        with patch("interface.touch_interface._get_working_directory", side_effect=AssertionError("must be deferred")) as working_dir:
            self.create_app({"environment": {"working_dir": "/unavailable/mount"}})
        working_dir.assert_not_called()

    def test_recovery_cleanup_does_not_reset_fallback_checkout(self):
        app = self.create_app()
        with patch.object(app, "pull_latest_code") as update, patch.object(app, "sync_data") as sync:
            self.assertIs(app.cleanup(), CleanupResult.UNAVAILABLE)
        update.assert_not_called()
        sync.assert_not_called()

    def test_command_line_accepts_no_config_for_recovery(self):
        for argv in (["touch_interface.py"], ["touch_interface.py", "--debug-mode"]):
            with self.subTest(argv=argv), patch.object(sys, "argv", argv):
                self.assertIsNone(parse_args().config)

    def test_fifo_config_is_rejected_without_attempting_to_read_it(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            os.mkfifo(path)
            with self.assertRaisesRegex(ValueError, "regular JSON file"):
                load_launcher_config(path, validate_experiment=False)

    def test_recovery_startup_handles_broken_files_paths_and_config_shapes(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            malformed = folder / "malformed.json"
            malformed.write_text('{"screens":')
            wrong_type = folder / "array.json"
            wrong_type.write_text('[]')
            invalid_fields = folder / "fields.json"
            invalid_fields.write_text('{"environment": [], "screens": false, "tasks": null, "subjects": 0, "sync_bar": "bad", "eye_tracker_experiment": true}')
            invalid_encoding = folder / "bytes.json"
            invalid_encoding.write_bytes(b'\xff\xfe')
            loop = folder / "loop.json"
            loop.symlink_to(loop)
            paths = (None, str(folder / "missing.json"), str(malformed), str(wrong_type), str(invalid_fields), str(invalid_encoding), str(folder), str(loop), "~__neuro_tasks_missing_user__/config.json")
            for config_path in paths:
                with self.subTest(config_path=config_path):
                    created = []

                    def create(*args, **kwargs):
                        app = TouchInterfaceApp(*args, **kwargs)
                        created.append(app)
                        return app

                    screen = ScreenGeometry(0, 0, 0, 800, 480, "DSI-1")
                    args = SimpleNamespace(config=config_path, main_screen="HDMI-missing", experimenter_screen="DSI-1", debug_mode=True)
                    with patch("interface.touch_interface.parse_args", return_value=args), patch("interface.touch_interface.tk.Tk") as root, patch("interface.touch_interface.tk.StringVar"), patch.object(TouchInterfaceApp, "_build_ui"), patch("interface.touch_interface.TouchInterfaceApp", side_effect=create), patch("interface.touch_interface.get_tk_screens", return_value=[screen]), patch("interface.touch_interface.HardwareDebugPanel") as panel, patch("builtins.print"):
                        panel.return_value.controller.busy = False
                        panel.return_value.controller.uncleared = False
                        main()
                    self.assertEqual(created[0].screen_info, screen)
                    root.return_value.mainloop.assert_called_once()
                    panel.assert_called_once()
                    self.assertIsNone(created[0].experiment)

    def test_experiment_action_still_checks_missing_main_screen_and_debug_remains_available(self):
        app = self.create_app({"environment": {"working_dir": str(Path.cwd()), "python": "python"}, "subjects": {"Test": "T"}, "tasks": {}})
        with patch("interface.touch_interface.resolve_task_screens", side_effect=ValueError("Missing main display")), patch("interface.touch_interface.HardwareDebugPanel") as panel, patch("builtins.print"):
            self.assertFalse(app._prepare_experiment_environment())
            app._open_debug_mode()
        self.assertIn("Missing main display", app.status_var.set.call_args.args[0])
        panel.assert_called_once()

    def test_invalid_sync_blocks_experiment_setup_but_not_menu(self):
        for sync in (False, {"height": 0}, {"sync_pin": -1}):
            with self.subTest(sync=sync):
                app = self.create_app({"environment": {"working_dir": str(Path.cwd()), "python": "python"}, "subjects": {"Test": "T"}, "tasks": {}, "sync_bar": sync})
                with patch("interface.touch_interface.resolve_task_screens") as screens, patch("builtins.print"):
                    self.assertFalse(app._prepare_experiment_environment())
                screens.assert_not_called()

    def test_main_with_bad_config_still_opens_debug_mode(self):
        args = SimpleNamespace(config="missing.json", main_screen=None, experimenter_screen=None, debug_mode=True)
        with patch("interface.touch_interface.parse_args", return_value=args), patch("interface.touch_interface.load_launcher_config", side_effect=ValueError("bad JSON")), patch("interface.touch_interface.tk.Tk") as root, patch("interface.touch_interface.resolve_launcher_screen"), patch("interface.touch_interface.TouchInterfaceApp") as app_cls, patch("builtins.print"):
            app = app_cls.return_value
            app.experiment = app.quiet_mode = app.idle_guard = app.debug_panel = None
            main()
        app._open_debug_mode.assert_called_once()
        root.return_value.mainloop.assert_called_once()
        self.assertEqual(app_cls.call_args.kwargs["config_error"], "bad JSON")

    def test_window_close_requests_debug_stop_before_desktop_exit(self):
        app = self.create_app()
        app.debug_panel = Mock()
        app._exit_to_desktop()
        app.debug_panel.close.assert_called_once()
        app.root.destroy.assert_not_called()

    def test_invalid_debug_settings_do_not_block_other_backend(self):
        settings = debug_settings({"debug_mode": {"gpio_chip": "bad", "daq_address": 99}, "system_diagnostic": None})
        self.assertEqual(settings["gpio_chip"], 0)
        self.assertEqual(settings["daq_address"], 0)
        self.assertEqual(settings["daq_module"], "piplates.DAQC2plate")


class DebugPanelTests(unittest.TestCase):
    def test_header_controls_consume_enter_before_root_can_invoke_a_pin(self):
        with patch("interface.hardware_debug.tk.Button") as button_class:
            button = HardwareDebugPanel._button(Mock(), "Stop All", Mock())
        bindings = dict(call.args for call in button.bind.call_args_list)
        for sequence in ("<Return>", "<KP_Enter>"):
            self.assertEqual(bindings[sequence](Mock()), "break")
        self.assertEqual(button_class.return_value.invoke.call_count, 2)

    def panel(self):
        panel = object.__new__(HardwareDebugPanel)
        panel.controller = Mock()
        import queue
        panel.controller.events = queue.SimpleQueue()
        panel.buttons = {}
        panel.states = {}
        panel.status = Mock()
        panel.after = Mock()
        panel.destroy = Mock()
        panel.on_close = Mock()
        panel.closing = True
        panel.releasing = False
        panel.close_started = 0
        return panel

    def test_back_waits_for_off_before_returning_to_experiment_menu(self):
        panel = self.panel()
        panel.controller.busy = False
        panel.controller.uncleared = False
        panel._poll()
        panel.destroy.assert_called_once()
        panel.on_close.assert_called_once()
        panel.controller.close.assert_called_once()

    def test_back_waits_for_helper_processes_to_release_hardware(self):
        panel = self.panel()
        panel.controller.busy = False
        panel.controller.uncleared = False
        panel.controller.close.side_effect = lambda: setattr(panel.controller, "busy", True)
        panel._poll()
        panel.destroy.assert_not_called()
        panel._poll()
        self.assertTrue(panel.closing)
        panel.controller.busy = False
        panel._poll()
        panel.on_close.assert_called_once()

    def test_off_failure_keeps_debug_open_and_other_pins_usable(self):
        panel = self.panel()
        panel.controller.busy = False
        panel.controller.uncleared = True
        panel._poll()
        self.assertFalse(panel.closing)
        panel.destroy.assert_not_called()
        panel.on_close.assert_not_called()

    def test_hung_driver_during_close_does_not_disable_other_controls_forever(self):
        panel = self.panel()
        panel.controller.busy = True
        panel._poll()
        self.assertFalse(panel.closing)
        panel.on_close.assert_not_called()
        self.assertIn("Other pins remain available", panel.status.set.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
