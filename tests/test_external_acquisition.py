"""External recording prompts gate real experiment state and block creation."""

import datetime as dt
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

from interface.experiment_manager import ExperimentManager
from interface.touch_interface import TouchInterfaceApp


class ExternalAcquisitionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "demo.py").write_text("# task\n")
        (self.root / "demo.json").write_text("{}")
        (self.root / "event_name_library.json").write_text("{}")
        for target, value in (
            ("_git_commit", None), ("_runtime_metadata", {}),
        ):
            patcher = patch(f"interface.experiment_manager.{target}", return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)
        tracker = patch("interface.experiment_manager.ExperimentEyeTracker")
        self.tracker = tracker.start()
        self.addCleanup(tracker.stop)
        self.config = {"external_aquisition": True, "initial_state": {}}
        self.task = {"launch": "demo.py", "config": "demo.json"}
        self.app = object.__new__(TouchInterfaceApp)
        self.app.root = Mock()
        self.app.experiment = None
        self.app.cfg = self.config
        self.app.working_dir = self.root
        self.app.config_path = self.root / "launch.json"
        self.app.tasks_cfg = {"Demo": self.task}
        self.app.page_stack = []
        self.app.task_active = False
        self.app.cleanup_active = False
        self.app.status_var = Mock()
        self.app.quiet_mode = Mock()
        self.app.cleanup = Mock()
        self.app._render_root_menu = Mock()
        self.app._render_current_page = Mock()
        self.app._run_block = Mock(side_effect=self._complete_block)
        constructor = patch("interface.touch_interface.ExperimentManager", side_effect=self._create_manager)
        self.constructor = constructor.start()
        self.addCleanup(constructor.stop)
        dialog = patch("interface.touch_interface.choose_dialog")
        self.dialog = dialog.start()
        self.addCleanup(dialog.stop)

    def _create_manager(self, **kwargs):
        self.manager = ExperimentManager(
            **kwargs,
            now=dt.datetime(2026, 11, 12, 23, 59, tzinfo=dt.timezone(dt.timedelta(hours=-5))),
        )
        return self.manager

    def _start_experiment(self):
        self.app._select_subject("Subject One", "S1")

    def _saved_state(self):
        return json.loads(self.manager.state_path.read_text())

    def _complete_block(self, block):
        self.manager.finish_block(block)
        return subprocess.CompletedProcess(["demo.py"], 0)

    def test_disabled_or_omitted_option_launches_and_ends_without_prompts(self):
        self.config["initial_state"]["associated_external_data"] = True
        for enabled in (None, False):
            with self.subTest(enabled=enabled):
                self.config.pop("external_aquisition", None)
                if enabled is not None:
                    self.config["external_aquisition"] = enabled
                self._start_experiment()
                self.assertIs(self._saved_state()["associated_external_data"], False)
                self.app._run_task("Demo", self.task)
                self.app._end_experiment()
                self.assertEqual(len(self.manager.blocks), 1)
                self.assertIs(self._saved_state()["associated_external_data"], False)
        self.dialog.assert_not_called()

    def test_yes_is_saved_before_recording_confirmation_and_only_first_block_prompts(self):
        def confirm(*args, **kwargs):
            if self.dialog.call_count == 1:
                self.assertIsNone(self.app.experiment)
                self.constructor.assert_not_called()
                self.tracker.assert_not_called()
                self.app.quiet_mode.enter.assert_not_called()
                self.app._render_current_page.assert_not_called()
            elif self.dialog.call_count == 2:
                self.assertTrue(self.app.task_active)
                self.assertEqual(self.manager.blocks, [])
                self.app._run_block.assert_not_called()
                self.assertIs(self._saved_state()["associated_external_data"], True)
            else:
                # Ending must not stop/sync anything before confirmation.
                self.app.quiet_mode.exit.assert_not_called()
                self.app.cleanup.assert_not_called()
                self.manager.eye_tracker.finish.assert_not_called()
            return True

        self.dialog.side_effect = confirm
        self._start_experiment()
        self.dialog.assert_called_once()
        self.app._render_current_page.assert_called_once_with()
        self.assertIs(self._saved_state()["associated_external_data"], True)
        loop = {"launch": ["demo.py"], "config": ["demo.json"],
                "order_mode": "sequential", "n_iters": 3}
        self.app._run_task("Loop", loop)
        self.app._run_task("Demo", self.task)
        self.assertEqual(self.dialog.call_count, 2)
        self.assertEqual(len(self.manager.blocks), 4)
        self.assertEqual(self.app._run_block.call_count, 4)
        self.assertFalse(self.app.task_active)
        self.assertEqual(self.manager.external_data_name, "Subject One_20261112_001")
        self.assertEqual(self.dialog.call_args_list[0].args[2],
                         "Does this experiment need external data acquisition?")
        self.assertEqual(self.dialog.call_args_list[1].args[2],
                         "Please confirm external acquisition system is recording to directory named:\n\n"
                         "Subject One_20261112_001 !")

        self.app._end_experiment()

        self.assertEqual(self.dialog.call_count, 3)
        self.assertEqual(self.dialog.call_args.args[2],
                         "Please ensure external acquisition system is stopped, the data is saved, "
                         "and uploaded with the following name:\n\nSubject One_20261112_001")
        self.manager.eye_tracker.finish.assert_called_once_with()
        self.app.cleanup.assert_called_once_with()
        self.assertIsNone(self.app.experiment)
        self.assertIs(self._saved_state()["associated_external_data"], True)

    def test_no_is_remembered_and_skips_recording_and_end_confirmations(self):
        self.dialog.return_value = False
        self._start_experiment()
        self.assertIs(self._saved_state()["associated_external_data"], False)
        self.app._run_task("Demo", self.task)
        self.app._run_task("Demo", self.task)
        self.app._end_experiment()
        self.dialog.assert_called_once()
        self.assertIs(self._saved_state()["associated_external_data"], False)
        self.assertEqual(len(self.manager.blocks), 2)
        self.assertIsNone(self.app.experiment)

    def test_dismissing_question_cancels_start_without_creating_experiment_and_asks_again_on_retry(self):
        self.dialog.return_value = None
        self._start_experiment()
        self.assertIsNone(self.app.experiment)
        self.assertFalse((self.root / "logs").exists())
        self.constructor.assert_not_called()
        self.tracker.assert_not_called()
        self.app.quiet_mode.enter.assert_not_called()
        self.app._render_current_page.assert_not_called()
        self.app._run_block.assert_not_called()
        self.assertFalse(self.app.task_active)

        self.dialog.return_value = False
        self._start_experiment()
        self.assertIs(self._saved_state()["associated_external_data"], False)
        self.assertEqual(self.manager.experiment_number, 1)
        self.app._run_task("Demo", self.task)
        self.assertEqual(self.dialog.call_count, 2)
        self.assertEqual(len(self.manager.blocks), 1)

    def test_cancelled_recording_confirmation_preserves_yes_and_blocks_until_retry(self):
        self.dialog.side_effect = [True, False]
        self._start_experiment()
        self.app._run_task("Demo", self.task)
        self.assertEqual(self.manager.blocks, [])
        self.assertEqual(list(self.manager.blocks_dir.iterdir()), [])
        self.app._run_block.assert_not_called()
        self.assertIs(self._saved_state()["associated_external_data"], True)
        self.assertFalse(self.app.task_active)

        self.dialog.side_effect = [True]
        self.app._run_task("Demo", self.task)
        self.assertEqual(self.dialog.call_count, 3)
        self.assertEqual(self.dialog.call_args.args[1], "Confirm External Recording")
        self.assertEqual(len(self.manager.blocks), 1)

    def test_failed_initial_state_save_prevents_start_and_restores_quiet_mode(self):
        self.dialog.return_value = True
        with patch.object(ExperimentManager, "_write_state", side_effect=OSError("disk full")), patch(
            "interface.touch_interface.messagebox.showerror"
        ) as error:
            self._start_experiment()
        error.assert_called_once_with("Experiment Error", "disk full")
        self.assertIsNone(self.app.experiment)
        self.tracker.assert_not_called()
        self.app._run_block.assert_not_called()
        self.app._render_current_page.assert_not_called()
        self.app.quiet_mode.enter.assert_called_once_with()
        self.app.quiet_mode.exit.assert_called_once_with()
        self.assertFalse(self.app.task_active)
        self.dialog.assert_called_once()

    def test_cancelled_end_leaves_experiment_and_quiet_mode_active(self):
        self.dialog.return_value = True
        self._start_experiment()
        for answer in (False, None):
            with self.subTest(answer=answer):
                self.dialog.return_value = answer
                self.app._end_experiment()
                self.assertIs(self.app.experiment, self.manager)
                self.manager.eye_tracker.finish.assert_not_called()
                self.app.quiet_mode.exit.assert_not_called()
                self.app.cleanup.assert_not_called()
                self.app._render_root_menu.assert_not_called()

    def test_new_experiment_resets_choice_and_uses_its_own_number(self):
        self.dialog.return_value = True
        self._start_experiment()
        previous_state = self.manager.state_path
        self.app._end_experiment()
        self.config["initial_state"]["associated_external_data"] = True
        self.dialog.return_value = False
        self._start_experiment()
        self.assertEqual(self.manager.external_data_name, "Subject One_20261112_002")
        self.assertIs(self._saved_state()["associated_external_data"], False)
        self.assertIs(json.loads(previous_state.read_text())["associated_external_data"], True)

    def test_hardware_choice_is_saved_before_tracker_start(self):
        def start_tracker(*args, **kwargs):
            state = json.loads((kwargs["directory"] / "state.json").read_text())
            self.assertIs(state["associated_external_data"], True)
            self.app._render_current_page.assert_not_called()
            return Mock()

        self.tracker.side_effect = start_tracker
        self.dialog.return_value = True
        self._start_experiment()
        self.tracker.assert_called_once()
        self.assertIs(self.app.experiment, self.manager)


if __name__ == "__main__":
    unittest.main()
