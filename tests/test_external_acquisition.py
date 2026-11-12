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
            ("ExperimentEyeTracker", Mock()),
        ):
            patcher = patch(f"interface.experiment_manager.{target}", return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.config = {"external_aquisition": True, "initial_state": {}}
        self.task = {"launch": "demo.py", "config": "demo.json"}
        self.manager = self._manager()
        self.app = object.__new__(TouchInterfaceApp)
        self.app.root = Mock()
        self.app.experiment = self.manager
        self.app.task_active = False
        self.app.cleanup_active = False
        self.app.status_var = Mock()
        self.app.quiet_mode = Mock()
        self.app.cleanup = Mock()
        self.app._render_root_menu = Mock()
        self.app._run_block = Mock(side_effect=self._complete_block)
        dialog = patch("interface.touch_interface.choose_dialog")
        self.dialog = dialog.start()
        self.addCleanup(dialog.stop)

    def _manager(self):
        return ExperimentManager(
            working_dir=self.root, launch_config_path=self.root / "launch.json",
            launch_config=self.config, subject_name="Subject One", subject_code="S1",
            now=dt.datetime(2026, 11, 12, 23, 59, tzinfo=dt.timezone(dt.timedelta(hours=-5))),
        )

    def _saved_state(self):
        return json.loads(self.manager.state_path.read_text())

    def _complete_block(self, block):
        self.manager.finish_block(block)
        return subprocess.CompletedProcess(["demo.py"], 0)

    def test_disabled_or_omitted_option_launches_and_ends_without_prompts(self):
        for enabled in (None, False):
            with self.subTest(enabled=enabled):
                self.config.pop("external_aquisition", None)
                if enabled is not None:
                    self.config["external_aquisition"] = enabled
                self.manager = self.app.experiment = self._manager()
                self.app._run_task("Demo", self.task)
                self.app._end_experiment()
                self.assertEqual(len(self.manager.blocks), 1)
                self.assertIs(self._saved_state()["associated_external_data"], False)
        self.dialog.assert_not_called()

    def test_yes_is_saved_before_recording_confirmation_and_only_first_block_prompts(self):
        def confirm(*args, **kwargs):
            if self.dialog.call_count <= 2:
                self.assertTrue(self.app.task_active)
                self.assertEqual(self.manager.blocks, [])
                self.app._run_block.assert_not_called()
                self.assertIs(
                    self._saved_state()["associated_external_data"],
                    self.dialog.call_count == 2,
                )
            else:
                # Ending must not stop/sync anything before confirmation.
                self.app.quiet_mode.exit.assert_not_called()
                self.app.cleanup.assert_not_called()
                self.manager.eye_tracker.finish.assert_not_called()
            return True

        self.dialog.side_effect = confirm
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
                         "Does this experiment use external acquisition hardware?")
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
        self.app._run_task("Demo", self.task)
        self.app._run_task("Demo", self.task)
        self.app._end_experiment()
        self.dialog.assert_called_once()
        self.assertTrue(self.manager.external_acquisition_answered)
        self.assertIs(self._saved_state()["associated_external_data"], False)
        self.assertEqual(len(self.manager.blocks), 2)
        self.assertIsNone(self.app.experiment)

    def test_dismissing_question_cancels_launch_and_asks_again_on_retry(self):
        self.dialog.return_value = None
        self.app._run_task("Demo", self.task)
        self.assertEqual(self.manager.blocks, [])
        self.assertEqual(list(self.manager.blocks_dir.iterdir()), [])
        self.app._run_block.assert_not_called()
        self.assertFalse(self.manager.external_acquisition_answered)
        self.assertFalse(self.app.task_active)
        self.assertIs(self._saved_state()["associated_external_data"], False)

        self.dialog.return_value = False
        self.app._run_task("Demo", self.task)
        self.assertEqual(self.dialog.call_count, 2)
        self.assertEqual(len(self.manager.blocks), 1)

    def test_cancelled_recording_confirmation_preserves_yes_and_blocks_until_retry(self):
        self.dialog.side_effect = [True, False]
        self.app._run_task("Demo", self.task)
        self.assertEqual(self.manager.blocks, [])
        self.assertEqual(list(self.manager.blocks_dir.iterdir()), [])
        self.app._run_block.assert_not_called()
        self.assertTrue(self.manager.external_acquisition_answered)
        self.assertIs(self._saved_state()["associated_external_data"], True)
        self.assertFalse(self.app.task_active)

        self.dialog.side_effect = [True]
        self.app._run_task("Demo", self.task)
        self.assertEqual(self.dialog.call_count, 3)
        self.assertEqual(self.dialog.call_args.args[1], "Confirm External Recording")
        self.assertEqual(len(self.manager.blocks), 1)

    def test_failed_state_save_prevents_launch_and_allows_choice_to_be_retried(self):
        self.dialog.return_value = True
        with patch("interface.experiment_manager._write_json_atomic", side_effect=OSError("disk full")), patch(
            "interface.touch_interface.messagebox.showerror"
        ) as error:
            self.app._run_task("Demo", self.task)
        error.assert_called_once_with("Launch Error", "disk full")
        self.assertEqual(self.manager.blocks, [])
        self.app._run_block.assert_not_called()
        self.assertFalse(self.manager.external_acquisition_answered)
        self.assertIs(self.manager.state["associated_external_data"], False)
        self.assertIs(self._saved_state()["associated_external_data"], False)
        self.assertFalse(self.app.task_active)
        self.dialog.assert_called_once()

    def test_cancelled_end_leaves_experiment_and_quiet_mode_active(self):
        self.manager.set_associated_external_data(True)
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
        self.manager.set_associated_external_data(True)
        self.config["initial_state"]["associated_external_data"] = True
        next_experiment = self._manager()
        self.assertEqual(next_experiment.external_data_name, "Subject One_20261112_002")
        self.assertFalse(next_experiment.external_acquisition_answered)
        self.assertIs(json.loads(next_experiment.state_path.read_text())["associated_external_data"], False)
        self.assertIs(self._saved_state()["associated_external_data"], True)


if __name__ == "__main__":
    unittest.main()
