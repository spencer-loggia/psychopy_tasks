import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from bin.screen import ScreenGeometry
from interface.touch_interface import TouchInterfaceApp


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
        app.experiment = Mock()
        app.page_stack = [("Tasks", {})]
        app._render_root_menu = Mock()

        app._end_experiment()

        self.assertIsNone(app.experiment)
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
        app._render_root_menu.assert_not_called()
        app.status_var.set.assert_called_once_with(
            "Cannot end experiment while a task is running"
        )

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


if __name__ == "__main__":
    unittest.main()
