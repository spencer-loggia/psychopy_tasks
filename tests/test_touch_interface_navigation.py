import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from bin.screen import ScreenGeometry
from interface.touch_interface import CleanupReport, CleanupResult, ScrollableButtonFrame, TouchInterfaceApp


class TouchInterfaceNavigationTests(unittest.TestCase):
    def _app(self):
        app = object.__new__(TouchInterfaceApp)
        app.task_active = False
        app.experiment = None
        app.quiet_mode = Mock()
        app.page_stack = []
        app.page_title_var = Mock()
        app.status_var = Mock()
        app._clear_buttons = Mock()
        app.cleanup = Mock()
        return app

    def test_root_menu_has_six_system_actions_in_order(self):
        app = self._app()
        app.page_stack = [("Tasks", {})]
        app._create_start_experiment_button = Mock()
        app._create_diagnostic_button = Mock()
        app._create_shell_button = Mock()
        app._create_rig_mode_button = Mock()
        app._create_desktop_button = Mock()
        app._create_shutdown_button = Mock()

        app._render_root_menu()

        self.assertEqual(app.page_stack, [])
        app.page_title_var.set.assert_called_once_with("Experiment Manager")
        app._create_start_experiment_button.assert_called_once_with(0)
        app._create_diagnostic_button.assert_called_once_with(1)
        app._create_shell_button.assert_called_once_with(2)
        app._create_rig_mode_button.assert_called_once_with(3)
        app._create_desktop_button.assert_called_once_with(4)
        app._create_shutdown_button.assert_called_once_with(5)

    def test_top_level_task_menu_has_shell_and_end_experiment(self):
        app = self._app()
        app.page_stack = [("Tasks", {"Demo": {"launch": "demo.py"}})]
        app._create_task_button = Mock()
        app._create_shell_button = Mock()
        app._create_end_experiment_button = Mock()
        app._create_diagnostic_button = Mock()
        app._create_rig_mode_button = Mock()
        app._create_desktop_button = Mock()
        app._create_shutdown_button = Mock()

        app._render_current_page()

        app._create_task_button.assert_called_once_with(
            0,
            "Demo",
            {"launch": "demo.py"},
        )
        app._create_shell_button.assert_called_once_with(1)
        app._create_end_experiment_button.assert_called_once_with(2)
        app._create_diagnostic_button.assert_not_called()
        app._create_rig_mode_button.assert_not_called()
        app._create_desktop_button.assert_not_called()
        app._create_shutdown_button.assert_not_called()

    def test_end_experiment_returns_to_root_and_allows_another_experiment(self):
        app = self._app()
        experiment = app.experiment = Mock()
        app.page_stack = [("Tasks", {})]
        app._render_root_menu = Mock()

        app._end_experiment()

        self.assertIsNone(app.experiment)
        experiment.finish.assert_called_once_with()
        app.quiet_mode.exit.assert_called_once_with()
        app.cleanup.assert_called_once_with()
        app._render_root_menu.assert_called_once_with()

    def test_active_task_prevents_ending_experiment(self):
        app = self._app()
        experiment = Mock()
        app.task_active = True
        app.experiment = experiment
        app._render_root_menu = Mock()

        app._end_experiment()

        self.assertIs(app.experiment, experiment)
        experiment.finish.assert_not_called()
        app._render_root_menu.assert_not_called()
        app.status_var.set.assert_called_once_with(
            "Cannot end experiment while a task is running"
        )

    def test_end_experiment_keeps_cleanup_status_and_displays_failure_details(self):
        app = self._app()
        app.root = Mock()
        app.cleanup_report = CleanupReport(False, CleanupResult.SUCCESS, "Git permission denied", "Data copied")
        app._render_root_menu = Mock(side_effect=lambda: app.status_var.set("Choose an action"))
        with patch("interface.touch_interface.messagebox.showwarning") as warning:
            app._end_experiment()
        app.status_var.set.assert_called_with("Data sync completed. Code update failed.")
        warning.assert_called_once_with("Experiment Cleanup", app.cleanup_report.details, parent=app.root)

    def test_closing_launcher_also_ends_active_tracker_experiment(self):
        app = self._app()
        app.experiment = Mock()
        app.root = Mock()
        app.idle_guard = Mock()
        app._exit_to_desktop()
        app.experiment.finish.assert_called_once_with()
        app.root.destroy.assert_called_once_with()

    def test_process_launch_hides_the_interface(self):
        app = self._app()
        app.root = Mock()
        app.idle_guard = Mock()

        app._hide_interface_for_process()

        app.root.withdraw.assert_called_once_with()
        app.root.update_idletasks.assert_called_once_with()
        app.idle_guard.prepare_task_launch.assert_called_once_with()

    def test_failed_launch_focus_barrier_restores_interface(self):
        app = self._app()
        app.root = Mock()
        app.idle_guard = Mock()
        app.idle_guard.prepare_task_launch.side_effect = RuntimeError("focus failed")

        with self.assertRaisesRegex(RuntimeError, "focus failed"):
            app._hide_interface_for_process()

        app.idle_guard.restore_interface_focus.assert_called_once_with()

    def test_process_completion_restores_the_guarded_interface(self):
        app = self._app()
        app.root = Mock()
        app.idle_guard = Mock()

        app._restore_interface_after_process()

        app.idle_guard.enter_idle.assert_called_once_with()
        app.root.deiconify.assert_not_called()

    def test_shell_launch_uses_working_directory_secondary_screen_and_venv(self):
        app = self._app()
        app.working_dir = Path("/work/neuro_tasks")
        app.python_cmd = "/opt/psychopy/.venv/bin/python"
        app.screen_info = ScreenGeometry(
            index=1,
            x=1920,
            y=0,
            width=800,
            height=480,
            name="HDMI-2",
        )

        with patch.dict(
            "interface.touch_interface.os.environ",
            {"PATH": "/usr/bin", "PYTHONHOME": "/wrong/python"},
            clear=True,
        ), patch("interface.touch_interface.subprocess.Popen") as popen:
            app._launch_shell()

        command = popen.call_args.args[0]
        kwargs = popen.call_args.kwargs
        self.assertEqual(
            command[0:3],
            ["lxterminal", "--no-remote", "--title=neuro_tasks shell"],
        )
        self.assertIn("--geometry=80x24+1944+24", command)
        self.assertIn("--working-directory=/work/neuro_tasks", command)
        self.assertEqual(kwargs["cwd"], app.working_dir)
        self.assertEqual(kwargs["env"]["VIRTUAL_ENV"], "/opt/psychopy/.venv")
        self.assertEqual(kwargs["env"]["PATH"], "/opt/psychopy/.venv/bin:/usr/bin")
        self.assertNotIn("PYTHONHOME", kwargs["env"])
        app.status_var.set.assert_called_once_with("Shell opened on secondary monitor")

    def test_shell_launch_failure_is_written_to_console(self):
        app = self._app()
        app.working_dir = Path("/work/neuro_tasks")
        app.python_cmd = "/opt/psychopy/.venv/bin/python"
        app.screen_info = ScreenGeometry(1, 1920, 0, 800, 480, "HDMI-2")

        with patch(
            "interface.touch_interface.subprocess.Popen",
            side_effect=FileNotFoundError("lxterminal"),
        ), patch("builtins.print") as print_mock:
            app._launch_shell()

        self.assertEqual(print_mock.call_args.args[0], "Could not launch shell: lxterminal")
        app.status_var.set.assert_called_once_with("Shell launch failed")


class MenuKeyboardTests(unittest.TestCase):
    def setUp(self):
        self.app = object.__new__(TouchInterfaceApp)
        self.app.root = Mock()
        self.app.root.grab_current.return_value = None
        self.app.task_active = False
        self.app.cleanup_active = False
        self.frame = object.__new__(ScrollableButtonFrame)
        self.app.button_frame = self.frame
        self.frame._buttons = []
        self.frame._selected_button = None
        self.frame.inner = Mock()
        self.frame.inner.cget.return_value = "#e9ecef"
        self.frame.canvas = Mock()
        self.frame.canvas.bbox.return_value = (0, 0, 300, 1000)
        self.frame.canvas.yview.return_value = (0.0, 0.3)
        self.frame.canvas.cget.return_value = 20
        self.frame._reset_touch_gesture = Mock()
        self.frame.after_idle = Mock()

    def _button(self, *, state="normal", y=10):
        button = Mock()
        button.cget.return_value = state
        button.winfo_y.return_value = y
        button.winfo_height.return_value = 80
        self.frame.register_button(button)
        return button

    def _key(self, keysym):
        self.assertEqual(self.app._on_menu_key(SimpleNamespace(keysym=keysym)), "break")

    def test_arrows_wrap_skip_disabled_buttons_and_enter_activates_once(self):
        first = self._button()
        disabled = self._button(state="disabled")
        last = self._button(y=820)

        self._key("Down")
        self._key("Return")
        last.invoke.assert_called_once_with()
        self._key("Down")
        self._key("KP_Enter")
        first.invoke.assert_called_once_with()
        self._key("Up")
        self.assertIs(self.frame._selected_button, last)
        disabled.invoke.assert_not_called()
        disabled.focus_set.assert_not_called()

    def test_keyboard_selection_scrolls_into_view_in_both_directions(self):
        self._button()
        self._button(y=820)

        self._key("Down")
        offset = self.frame.canvas.yview_moveto.call_args.args[0]
        self.assertGreaterEqual((offset + 0.3) * 1000, 900)
        self.assertLessEqual(offset * 1000, 820)

        self.frame.canvas.yview.return_value = (offset, offset + 0.3)
        self._key("Up")
        self.frame.canvas.yview_moveto.assert_called_with(0.0)

    def test_enter_follows_mouse_selection_and_resets_on_page_change(self):
        first = self._button()
        clicked = self._button(y=110)
        self.frame._on_touch_press(SimpleNamespace(y_root=120), clicked)
        self._key("Return")
        clicked.invoke.assert_called_once_with()
        first.invoke.assert_not_called()

        self.frame.reset_scroll()
        first.destroy()
        clicked.destroy()
        next_page = self._button()
        self._key("Return")
        next_page.invoke.assert_called_once_with()
        clicked.invoke.assert_called_once_with()

    def test_keys_do_not_activate_menu_during_tasks_cleanup_or_modal_dialogs(self):
        button = self._button()
        for flag in ("task_active", "cleanup_active"):
            with self.subTest(flag=flag):
                setattr(self.app, flag, True)
                self._key("Return")
                self._key("Down")
                setattr(self.app, flag, False)
        self.app.root.grab_current.return_value = Mock()
        self._key("Return")
        self._key("Down")
        button.invoke.assert_not_called()
        button.focus_set.assert_not_called()

    def test_empty_and_disabled_menus_ignore_keys(self):
        self._key("Down")
        self._key("Return")
        button = self._button()
        button.cget.return_value = "disabled"
        self._key("Up")
        self._key("Return")
        button.invoke.assert_not_called()


if __name__ == "__main__":
    unittest.main()
