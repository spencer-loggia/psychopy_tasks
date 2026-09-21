import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, call, patch

from interface.data_sync import (
    ExperimentDataSync,
    SyncPlan,
    SyncPreparation,
    terminate_process,
)
from interface.touch_interface import TouchInterfaceApp


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

        with patch("interface.touch_interface.subprocess.run") as run:
            app.pull_latest_code()

        self.assertEqual(
            run.call_args_list,
            [
                call(
                    ["git", "reset", "--hard"],
                    cwd=app.working_dir,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                ),
                call(
                    ["git", "pull", "/mnt/data/task_mirror"],
                    cwd=app.working_dir,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                ),
            ],
        )

    def test_cleanup_blocks_reentry_and_runs_sync_before_maintenance(self):
        app = self._app()
        events = []
        app.sync_data = Mock(side_effect=lambda: events.append("sync") or True)
        app.attempt_rectify_timezone = Mock(
            side_effect=lambda: events.append("time")
        )
        app.pull_latest_code = Mock(side_effect=lambda: events.append("pull"))

        app.cleanup()

        self.assertEqual(events, ["sync", "time", "pull"])
        self.assertFalse(app.cleanup_active)

    def test_cancelled_sync_ends_cleanup_without_other_work(self):
        app = self._app()
        app.sync_data = Mock(return_value=False)
        app.attempt_rectify_timezone = Mock()
        app.pull_latest_code = Mock()

        app.cleanup()

        app.attempt_rectify_timezone.assert_not_called()
        app.pull_latest_code.assert_not_called()
        self.assertFalse(app.cleanup_active)

    def test_cancelled_or_failed_rsync_never_prunes(self):
        for cancelled, returncode, expected_result in (
            (True, -15, False),
            (False, 23, True),
        ):
            with self.subTest(cancelled=cancelled, returncode=returncode):
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
                process.wait.return_value = returncode
                app._show_sync_dialog = Mock(return_value=cancelled)

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


if __name__ == "__main__":
    unittest.main()
