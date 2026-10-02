import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, call, patch

from interface.data_sync import (
    DirectoryProbeResult,
    ExperimentDataSync,
    RSYNC_IO_TIMEOUT_SECONDS,
    SyncPlan,
    SyncPreparation,
    probe_remote_directory,
    run_command_bounded,
    terminate_process,
)
from interface.touch_interface import (
    DATA_SYNC_STALL_TIMEOUT_SECONDS,
    REMOTE_GIT_TIMEOUT_SECONDS,
    CleanupResult,
    SyncDialogResult,
    TouchInterfaceApp,
)


class ExperimentDataSyncTests(unittest.TestCase):
    def test_prepare_requires_nonempty_nfs_destination(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            logs = root / "logs"
            remote = root / "remote"
            logs.mkdir()
            remote.mkdir()
            (logs / "exp_T_20260921_001").mkdir()

            non_nfs = ExperimentDataSync(
                logs,
                remote,
                filesystem_type=lambda _path: "ext4",
            ).prepare()
            empty_nfs = ExperimentDataSync(
                logs,
                remote,
                filesystem_type=lambda _path: "nfs4",
            ).prepare()

        self.assertIsNone(non_nfs.plan)
        self.assertIn("not on a mounted NFS", non_nfs.warning)
        self.assertIsNone(empty_nfs.plan)
        self.assertIn("empty", empty_nfs.warning)

    def test_prepare_syncs_all_experiments_and_retains_most_recent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            logs = root / "logs"
            remote = root / "remote"
            logs.mkdir()
            remote.mkdir()
            (remote / ".mounted-storage-marker").touch()
            older = logs / "exp_T_20260920_001"
            newest = logs / "exp_T_20260921_001"
            older.mkdir()
            newest.mkdir()
            os.utime(older, ns=(10, 10))
            os.utime(newest, ns=(20, 20))

            prepared = ExperimentDataSync(
                logs,
                remote,
                filesystem_type=lambda _path: "nfs",
            ).prepare()

            self.assertIsNotNone(prepared.plan)
            plan = prepared.plan
            self.assertEqual(plan.experiments, (older, newest))
            self.assertEqual(plan.retained_experiment, newest)
            self.assertEqual(plan.destination, remote / "experiments")
            self.assertEqual(
                plan.command,
                [
                    "rsync",
                    "--archive",
                    f"--timeout={RSYNC_IO_TIMEOUT_SECONDS}",
                    "--progress",
                    "--no-owner",
                    "--no-group",
                    "--",
                    f"{logs}/",
                    f"{remote / 'experiments'}/",
                ],
            )

    def test_prune_removes_only_successfully_planned_older_experiments(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            logs = root / "logs"
            remote = root / "remote"
            logs.mkdir()
            remote.mkdir()
            (remote / "storage").mkdir()
            older = logs / "exp_T_20260920_001"
            newest = logs / "exp_T_20260921_001"
            older.mkdir()
            newest.mkdir()
            os.utime(older, ns=(10, 10))
            os.utime(newest, ns=(20, 20))
            data_sync = ExperimentDataSync(
                logs,
                remote,
                filesystem_type=lambda _path: "nfs4",
            )
            plan = data_sync.prepare().plan

            removed = data_sync.prune(plan)

            self.assertEqual(removed, (older,))
            self.assertFalse(older.exists())
            self.assertTrue(newest.exists())

    def test_terminate_process_escalates_when_terminate_does_not_finish(self):
        process = Mock()
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired("rsync", 0.01), 9]

        terminate_process(process, timeout_s=0.01)

        process.terminate.assert_called_once_with()
        process.kill.assert_called_once_with()

    def test_terminate_process_never_waits_forever_after_kill(self):
        process = Mock()
        process.poll.return_value = None
        process.wait.side_effect = subprocess.TimeoutExpired("rsync", 0.01)

        with patch("interface.data_sync._reap_process_in_background") as reap:
            reaped = terminate_process(process, timeout_s=0.01)

        self.assertFalse(reaped)
        self.assertEqual(
            process.wait.call_args_list,
            [call(timeout=0.01), call(timeout=0.01)],
        )
        reap.assert_called_once_with(process)

    def test_bounded_command_terminates_without_an_unbounded_wait(self):
        process = Mock()
        process.communicate.side_effect = subprocess.TimeoutExpired(
            ["git", "pull"],
            0.01,
        )

        with patch(
            "interface.data_sync.subprocess.Popen",
            return_value=process,
        ), patch("interface.data_sync.terminate_process") as terminate, patch(
            "interface.data_sync._close_process_pipes"
        ) as close_pipes:
            with self.assertRaises(subprocess.TimeoutExpired):
                run_command_bounded(["git", "pull"], timeout_s=0.01)

        terminate.assert_called_once_with(
            process,
            process_group=os.name == "posix",
        )
        close_pipes.assert_called_once_with(process)
        process.wait.assert_not_called()

    def test_remote_directory_probe_timeout_is_reported(self):
        with patch(
            "interface.data_sync.run_command_bounded",
            side_effect=subprocess.TimeoutExpired(["probe"], 0.01),
        ):
            result = probe_remote_directory(Path("/mnt/data"), timeout_s=0.01)

        self.assertFalse(result.is_directory)
        self.assertFalse(result.has_entries)
        self.assertIn("timed out", result.error)

    def test_terminate_process_can_signal_the_whole_process_group(self):
        process = Mock()
        process.pid = 1234
        process.poll.return_value = None
        process.wait.return_value = 0

        with patch("interface.data_sync.os.killpg") as killpg:
            reaped = terminate_process(process, process_group=True)

        self.assertTrue(reaped)
        self.assertEqual(
            killpg.call_args_list,
            [
                call(1234, __import__("signal").SIGTERM),
                call(1234, __import__("signal").SIGKILL),
            ],
        )
        process.terminate.assert_not_called()

    def test_prepare_reports_remote_directory_probe_error(self):
        data_sync = ExperimentDataSync(
            Path("/work/logs"),
            Path("/mnt/data"),
            filesystem_type=lambda _path: "nfs4",
            directory_probe=lambda _path: DirectoryProbeResult(
                False,
                False,
                "probe timed out",
            ),
        )

        preparation = data_sync.prepare()

        self.assertIsNone(preparation.plan)
        self.assertIn("probe timed out", preparation.warning)


class TouchInterfaceCleanupTests(unittest.TestCase):
    def _app(self):
        app = object.__new__(TouchInterfaceApp)
        app.cfg = {
            "remote_git_url": "/mnt/data/task_mirror",
            "remote_data_url": "/mnt/data",
        }
        app.working_dir = Path("/work/tree")
        app.cleanup_active = False
        app.status_var = Mock()
        app.root = Mock()
        return app

    def test_pull_resets_hard_then_pulls_configured_url(self):
        app = self._app()

        with patch("interface.touch_interface.run_command_bounded") as run:
            app.pull_latest_code()

        self.assertEqual(
            run.call_args_list,
            [
                call(
                    ["git", "reset", "--hard"],
                    cwd=app.working_dir,
                    check=True,
                    timeout_s=REMOTE_GIT_TIMEOUT_SECONDS,
                ),
                call(
                    ["git", "pull", "/mnt/data/task_mirror"],
                    cwd=app.working_dir,
                    check=True,
                    timeout_s=REMOTE_GIT_TIMEOUT_SECONDS,
                ),
            ],
        )

    def test_pull_timeout_is_logged_and_reported_as_unavailable(self):
        app = self._app()
        timeout = subprocess.TimeoutExpired(
            ["git", "pull", "/mnt/data/task_mirror"],
            REMOTE_GIT_TIMEOUT_SECONDS,
        )

        with patch(
            "interface.touch_interface.run_command_bounded",
            side_effect=[None, timeout],
        ), patch("builtins.print") as log:
            available = app.pull_latest_code()

        self.assertFalse(available)
        self.assertIn("timed out", log.call_args.args[0])
        self.assertIs(log.call_args.kwargs["file"], __import__("sys").stderr)

    def test_startup_continues_when_code_remote_is_unavailable(self):
        app = self._app()
        app.pull_latest_code = Mock(return_value=False)

        with patch("interface.touch_interface.os.chdir") as chdir:
            app.startup()

        chdir.assert_called_once_with(app.working_dir)
        app.pull_latest_code.assert_called_once_with()

    def test_cleanup_blocks_reentry_and_pulls_before_sync(self):
        app = self._app()
        events = []
        app.sync_data = Mock(
            side_effect=lambda: events.append("sync") or CleanupResult.SUCCESS
        )
        app.pull_latest_code = Mock(side_effect=lambda: events.append("pull") or True)

        result = app.cleanup()

        self.assertIs(result, CleanupResult.SUCCESS)
        self.assertEqual(events, ["pull", "sync"])
        self.assertFalse(app.cleanup_active)

    def test_unavailable_sync_resets_cleanup_state(self):
        app = self._app()
        app.sync_data = Mock(return_value=CleanupResult.UNAVAILABLE)
        app.pull_latest_code = Mock(return_value=True)

        result = app.cleanup()

        self.assertIs(result, CleanupResult.UNAVAILABLE)
        app.pull_latest_code.assert_called_once_with()
        self.assertFalse(app.cleanup_active)

    def test_cleanup_attempts_both_remotes_and_combines_their_status(self):
        for code_available, data_result, expected in (
            (True, CleanupResult.SUCCESS, CleanupResult.SUCCESS),
            (False, CleanupResult.SUCCESS, CleanupResult.UNAVAILABLE),
            (True, CleanupResult.UNAVAILABLE, CleanupResult.UNAVAILABLE),
            (False, CleanupResult.UNAVAILABLE, CleanupResult.UNAVAILABLE),
            (True, CleanupResult.CANCELLED, CleanupResult.CANCELLED),
        ):
            with self.subTest(
                code_available=code_available,
                data_result=data_result,
            ):
                app = self._app()
                app.pull_latest_code = Mock(return_value=code_available)
                app.sync_data = Mock(return_value=data_result)

                self.assertIs(app.cleanup(), expected)
                app.pull_latest_code.assert_called_once_with()
                app.sync_data.assert_called_once_with()
                self.assertFalse(app.cleanup_active)

    def test_prepare_warning_is_logged_and_reported_as_unavailable(self):
        app = self._app()
        data_sync = Mock()
        data_sync.prepare.return_value = SyncPreparation(
            None,
            "data mount unavailable",
        )

        with patch(
            "interface.touch_interface.ExperimentDataSync",
            return_value=data_sync,
        ), patch("builtins.print") as log:
            available = app.sync_data()

        self.assertIs(available, CleanupResult.UNAVAILABLE)
        self.assertIn("data mount unavailable", log.call_args.args[0])

    def test_available_storage_without_experiments_is_successful(self):
        app = self._app()
        data_sync = Mock()
        data_sync.prepare.return_value = SyncPreparation(None)

        with patch(
            "interface.touch_interface.ExperimentDataSync",
            return_value=data_sync,
        ):
            available = app.sync_data()

        self.assertIs(available, CleanupResult.SUCCESS)

    def test_cancelled_or_failed_rsync_never_prunes(self):
        for dialog_result, returncode, expected_result in (
            (SyncDialogResult.CANCELLED, -15, CleanupResult.CANCELLED),
            (SyncDialogResult.TIMED_OUT, None, CleanupResult.UNAVAILABLE),
            (SyncDialogResult.COMPLETED, 23, CleanupResult.UNAVAILABLE),
        ):
            with self.subTest(dialog_result=dialog_result, returncode=returncode):
                app = self._app()
                plan = SyncPlan(
                    source=Path("/work/tree/logs"),
                    experiments=(Path("/work/tree/logs/exp_001"),),
                    retained_experiment=Path("/work/tree/logs/exp_001"),
                    destination=Path("/mnt/data/experiments"),
                )
                data_sync = Mock()
                data_sync.prepare.return_value = SyncPreparation(plan)
                process = Mock()
                process.poll.return_value = returncode
                app._show_sync_dialog = Mock(return_value=dialog_result)

                with patch(
                    "interface.touch_interface.ExperimentDataSync",
                    return_value=data_sync,
                ):
                    with patch(
                        "interface.touch_interface.subprocess.Popen",
                        return_value=process,
                    ):
                        result = app.sync_data()

                self.assertIs(result, expected_result)
                data_sync.prune.assert_not_called()
                if dialog_result is not SyncDialogResult.COMPLETED:
                    process.wait.assert_not_called()

    def test_sync_dialog_cancels_poll_before_destroying(self):
        app = self._app()
        process = Mock()
        plan = SyncPlan(
            source=Path("/work/tree/logs"),
            experiments=(Path("/work/tree/logs/exp_001"),),
            retained_experiment=Path("/work/tree/logs/exp_001"),
            destination=Path("/mnt/data/experiments"),
        )
        dialog = Mock()
        dialog.after.return_value = "poll-1"
        cancel_command = None

        def create_button(_parent, **kwargs):
            nonlocal cancel_command
            cancel_command = kwargs["command"]
            return Mock()

        def cancel_while_waiting(_dialog):
            cancel_command()

        app.root.wait_window.side_effect = cancel_while_waiting

        with patch(
            "interface.touch_interface.tk.Toplevel",
            return_value=dialog,
        ), patch("interface.touch_interface.tk.Label"), patch(
            "interface.touch_interface.tk.Button",
            side_effect=create_button,
        ), patch(
            "interface.touch_interface.terminate_process"
        ) as terminate:
            result = app._show_sync_dialog(process, plan)

        self.assertIs(result, SyncDialogResult.CANCELLED)
        terminate.assert_called_once_with(
            process,
            process_group=os.name == "posix",
        )
        dialog.after_cancel.assert_called_once_with("poll-1")
        dialog.destroy.assert_called_once_with()

    def test_sync_dialog_times_out_after_no_progress(self):
        app = self._app()
        process = Mock()
        process.poll.return_value = None
        plan = SyncPlan(
            source=Path("/work/tree/logs"),
            experiments=(Path("/work/tree/logs/exp_001"),),
            retained_experiment=Path("/work/tree/logs/exp_001"),
            destination=Path("/mnt/data/experiments"),
        )
        dialog = Mock()
        dialog.winfo_exists.return_value = True
        callbacks = []

        def schedule(_delay, callback):
            callbacks.append(callback)
            return f"poll-{len(callbacks)}"

        dialog.after.side_effect = schedule
        app.root.wait_window.side_effect = lambda _dialog: callbacks.pop(0)()

        with patch(
            "interface.touch_interface.tk.Toplevel",
            return_value=dialog,
        ), patch("interface.touch_interface.tk.Label"), patch(
            "interface.touch_interface.tk.Button",
            return_value=Mock(),
        ), patch(
            "interface.touch_interface.time.monotonic",
            side_effect=[0.0, float(DATA_SYNC_STALL_TIMEOUT_SECONDS)],
        ), patch(
            "interface.touch_interface.terminate_process"
        ) as terminate:
            result = app._show_sync_dialog(process, plan)

        self.assertIs(result, SyncDialogResult.TIMED_OUT)
        terminate.assert_called_once_with(
            process,
            process_group=os.name == "posix",
        )
        dialog.destroy.assert_called_once_with()

    def test_sync_progress_resets_stall_deadline(self):
        app = self._app()
        process = Mock()
        process.poll.side_effect = [None, None, 0]
        plan = SyncPlan(
            source=Path("/work/tree/logs"),
            experiments=(Path("/work/tree/logs/exp_001"),),
            retained_experiment=Path("/work/tree/logs/exp_001"),
            destination=Path("/mnt/data/experiments"),
        )
        dialog = Mock()
        dialog.winfo_exists.return_value = True
        callbacks = []

        def schedule(_delay, callback):
            callbacks.append(callback)
            return f"poll-{len(callbacks)}"

        dialog.after.side_effect = schedule

        with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as progress:

            def run_until_complete(_dialog):
                progress.write("progress")
                progress.flush()
                callbacks.pop(0)()
                callbacks.pop(0)()
                callbacks.pop(0)()

            app.root.wait_window.side_effect = run_until_complete
            with patch(
                "interface.touch_interface.tk.Toplevel",
                return_value=dialog,
            ), patch("interface.touch_interface.tk.Label"), patch(
                "interface.touch_interface.tk.Button",
                return_value=Mock(),
            ), patch(
                "interface.touch_interface.time.monotonic",
                side_effect=[
                    0.0,
                    float(DATA_SYNC_STALL_TIMEOUT_SECONDS - 1),
                    float(2 * DATA_SYNC_STALL_TIMEOUT_SECONDS - 2),
                ],
            ), patch(
                "interface.touch_interface.terminate_process"
            ) as terminate:
                result = app._show_sync_dialog(process, plan, progress)

        self.assertIs(result, SyncDialogResult.COMPLETED)
        terminate.assert_not_called()


class TouchInterfaceShutdownTests(unittest.TestCase):
    def _app(self):
        app = object.__new__(TouchInterfaceApp)
        app.task_active = False
        app.cleanup_active = False
        app.status_var = Mock()
        app.root = Mock()
        app.cleanup = Mock()
        app._ask_retry_data_network = Mock()
        app._shutdown_command = Mock(return_value=["shutdown", "-h", "now"])
        return app

    def test_shutdown_retries_cleanup_after_yes(self):
        app = self._app()
        app.cleanup.side_effect = [
            CleanupResult.UNAVAILABLE,
            CleanupResult.SUCCESS,
        ]
        app._ask_retry_data_network.return_value = True

        with patch("interface.touch_interface.subprocess.run") as run:
            app._shutdown_system()

        self.assertEqual(app.cleanup.call_count, 2)
        app._ask_retry_data_network.assert_called_once_with()
        run.assert_called_once_with(
            ["shutdown", "-h", "now"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        app.root.after.assert_called_once_with(1000, app.root.destroy)

    def test_shutdown_no_proceeds_when_network_is_unavailable(self):
        app = self._app()
        app.cleanup.return_value = CleanupResult.UNAVAILABLE
        app._ask_retry_data_network.return_value = False

        with patch("interface.touch_interface.subprocess.run") as run:
            app._shutdown_system()

        app.cleanup.assert_called_once_with()
        app._ask_retry_data_network.assert_called_once_with()
        run.assert_called_once()

    def test_shutdown_with_available_network_does_not_prompt(self):
        app = self._app()
        app.cleanup.return_value = CleanupResult.SUCCESS

        with patch("interface.touch_interface.subprocess.run"):
            app._shutdown_system()

        app._ask_retry_data_network.assert_not_called()

    def test_cancelled_sync_cancels_shutdown_without_network_prompt(self):
        app = self._app()
        app.cleanup.return_value = CleanupResult.CANCELLED

        with patch("interface.touch_interface.subprocess.run") as run:
            app._shutdown_system()

        run.assert_not_called()
        app._ask_retry_data_network.assert_not_called()
        app.status_var.set.assert_called_with("Shutdown cancelled")

    def test_repeated_failure_prompts_until_no_then_proceeds(self):
        app = self._app()
        app.cleanup.return_value = CleanupResult.UNAVAILABLE
        app._ask_retry_data_network.side_effect = [True, False]

        with patch("interface.touch_interface.subprocess.run") as run:
            app._shutdown_system()

        self.assertEqual(app.cleanup.call_count, 2)
        self.assertEqual(app._ask_retry_data_network.call_count, 2)
        run.assert_called_once()

    def test_network_retry_dialog_uses_exact_message_and_touch_buttons(self):
        app = self._app()
        dialog = Mock()
        frame = Mock()
        buttons = {}
        close_kept_dialog_open = []

        def create_button(_parent, **kwargs):
            button = Mock()
            buttons[kwargs["text"]] = (button, kwargs)
            return button

        def close_then_choose_yes(_dialog):
            close_callback = dialog.protocol.call_args.args[1]
            close_callback()
            close_kept_dialog_open.append(not dialog.destroy.called)
            buttons["Yes"][1]["command"]()

        app.root.wait_window.side_effect = close_then_choose_yes

        with patch(
            "interface.touch_interface.tk.Toplevel",
            return_value=dialog,
        ), patch("interface.touch_interface.tk.Label") as label, patch(
            "interface.touch_interface.tk.Frame",
            return_value=frame,
        ), patch(
            "interface.touch_interface.tk.Button",
            side_effect=create_button,
        ):
            retry = TouchInterfaceApp._ask_retry_data_network(app)

        self.assertTrue(retry)
        self.assertEqual(
            label.call_args.kwargs["text"],
            "Data network not available. Try again?",
        )
        self.assertEqual(set(buttons), {"Yes", "No"})
        self.assertEqual(close_kept_dialog_open, [True])
        for _button, kwargs in buttons.values():
            self.assertGreaterEqual(kwargs["height"], 2)
            self.assertGreaterEqual(kwargs["pady"], 16)
        dialog.grab_set.assert_called_once_with()
        app.root.wait_window.assert_called_once_with(dialog)


if __name__ == "__main__":
    unittest.main()
