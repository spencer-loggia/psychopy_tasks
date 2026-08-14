import unittest
from unittest.mock import Mock

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
        return app

    def test_root_menu_has_five_system_actions_in_order(self):
        app = self._app()
        app.page_stack = [("Tasks", {})]
        app._create_start_experiment_button = Mock()
        app._create_diagnostic_button = Mock()
        app._create_rig_mode_button = Mock()
        app._create_desktop_button = Mock()
        app._create_shutdown_button = Mock()

        app._render_root_menu()

        self.assertEqual(app.page_stack, [])
        app.page_title_var.set.assert_called_once_with("Experiment Manager")
        app._create_start_experiment_button.assert_called_once_with(0)
        app._create_diagnostic_button.assert_called_once_with(1)
        app._create_rig_mode_button.assert_called_once_with(2)
        app._create_desktop_button.assert_called_once_with(3)
        app._create_shutdown_button.assert_called_once_with(4)

    def test_top_level_task_menu_ends_experiment_without_system_actions(self):
        app = self._app()
        app.page_stack = [("Tasks", {"Demo": {"launch": "demo.py"}})]
        app._create_task_button = Mock()
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
        app._create_end_experiment_button.assert_called_once_with(1)
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


if __name__ == "__main__":
    unittest.main()
