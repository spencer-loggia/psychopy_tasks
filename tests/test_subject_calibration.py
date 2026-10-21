import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from interface.data_sync import ExperimentDataSync, SyncPlan
from interface.subject_calibration import CALIBRATION_CACHE_DIRECTORY, SubjectCalibrationStore


def calibration(offset=0.1):
    return dict(x_scale=0.002, y_scale=-0.003, x_offset=offset, y_offset=-0.2,
                x_cross=0.001, y_cross=-0.0005, gain_anchor_px=[100, 50],
                gain_directions={"x": [1, 0], "y": [1, 0]},
                input_units="crop_local_pixels", coordinate_units="screen_fraction_centered",
                uart={"simulate": True}, filter={"ema_gamma": 0.2})


class SubjectCalibrationTests(unittest.TestCase):
    def _artifact(self, logs, subject, name, block, value):
        directory = logs / name
        path = directory / "blocks" / f"{block}_calibrate" / "calibration.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        (directory / "state.json").write_text(json.dumps({"subject": subject}))
        path.write_text(json.dumps({"eye_tracker_calibration": value}))
        return directory

    def test_newest_experiment_then_numeric_block_wins_and_subjects_stay_separate(self):
        with tempfile.TemporaryDirectory() as temporary:
            logs = Path(temporary)
            first = self._artifact(logs, "A/B", "exp_A_20261020_099", 20, calibration(1))
            latest = self._artifact(logs, "A/B", "exp_A_20261021_002", 10, calibration(2))
            self._artifact(logs, "A/B", latest.name, 2, calibration(3))
            other = self._artifact(logs, "A B", "exp_B_20261021_100", 1, calibration(4))
            store = SubjectCalibrationStore(logs)
            store.remember_experiments([latest, other, first])
            self.assertEqual(store.load("A/B")["eye_tracker_calibration"], calibration(2))
            self.assertEqual(store.load("A B")["eye_tracker_calibration"], calibration(4))
            self.assertIsNone(store.load("Unknown"))
            # Re-importing an older experiment cannot overwrite the cache.
            shutil.rmtree(latest)
            store.remember_experiments([first])
            self.assertEqual(SubjectCalibrationStore(logs).load("A/B")["block_number"], 10)

    def test_sync_backfills_before_deletion_and_excludes_cache_from_sync_and_pruning(self):
        with tempfile.TemporaryDirectory() as temporary:
            logs = Path(temporary) / "logs"
            a = self._artifact(logs, "A", "exp_A_20261020_001", 1, calibration(1))
            b = self._artifact(logs, "B", "exp_B_20261021_001", 1, calibration(2))
            plan = SyncPlan(logs, (a, b), b, Path(temporary) / "remote")
            sync = ExperimentDataSync(logs, plan.destination)
            self.assertEqual(sync.prune(plan), (a,))
            self.assertEqual(SubjectCalibrationStore(logs).load("A")["eye_tracker_calibration"], calibration(1))
            self.assertEqual(sync._experiment_directories(), [b])
            self.assertIn(f"--exclude=/{CALIBRATION_CACHE_DIRECTORY}/", plan.command)
            if shutil.which("rsync"):
                import subprocess
                subprocess.run(plan.command, check=True, capture_output=True)
                self.assertFalse((plan.destination / CALIBRATION_CACHE_DIRECTORY).exists())

    def test_failed_cache_write_prevents_any_pruning_and_preserves_previous_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            logs = Path(temporary)
            a = self._artifact(logs, "A", "exp_A_20261020_001", 1, calibration(1))
            b = self._artifact(logs, "A", "exp_A_20261021_001", 1, calibration(2))
            store = SubjectCalibrationStore(logs)
            store.remember_experiments([a])
            plan = SyncPlan(logs, (a, b), b, logs / "remote")
            # History import must finish before injecting a cache write failure.
            from interface.experiment_history import ExperimentHistory
            ExperimentHistory(logs).import_existing([a, b])
            with patch("interface.subject_calibration.os.replace", side_effect=OSError("disk full")):
                with self.assertRaisesRegex(OSError, "disk full"):
                    ExperimentDataSync(logs, None).prune(plan)
            self.assertTrue(a.exists())
            self.assertTrue(b.exists())
            self.assertEqual(store.load("A")["eye_tracker_calibration"], calibration(1))

    def test_incomplete_invalid_and_recovery_calibrations_do_not_replace_good_value(self):
        with tempfile.TemporaryDirectory() as temporary:
            logs = Path(temporary)
            good = self._artifact(logs, "A", "exp_A_20261020_001", 1, calibration())
            bad = self._artifact(logs, "A", "exp_A_20261021_001", 1, calibration(float("nan")))
            (bad / "blocks" / "2_calibrate").mkdir()
            (bad / "blocks" / "2_calibrate" / "calibration_recovery.json").write_text(
                json.dumps({"eye_tracker_calibration": calibration(10)}))
            store = SubjectCalibrationStore(logs)
            with self.assertWarnsRegex(RuntimeWarning, "Skipping invalid"):
                store.remember_experiments([bad, good])
            self.assertEqual(store.load("A")["eye_tracker_calibration"], calibration())
            with self.assertRaises(ValueError):
                store.save(subject="A", experiment_name=bad.name, block_number=3,
                           calibration={"x_scale": 1})
            with self.assertRaisesRegex(ValueError, "completed"):
                store.save(subject="A", experiment_name=bad.name, block_number=3,
                           calibration={**calibration(), "completion_status": "interrupted"})

    def test_cache_read_rejects_wrong_subject(self):
        with tempfile.TemporaryDirectory() as temporary:
            logs = Path(temporary)
            a = self._artifact(logs, "A", "exp_A_20261020_001", 1, calibration())
            store = SubjectCalibrationStore(logs)
            store.remember_experiments([a])
            path = next(store.directory.glob("*.json"))
            record = json.loads(path.read_text())
            record["subject"] = "B"
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "incorrect version or subject"):
                store.load("A")
            b = self._artifact(logs, "A", "exp_A_20261021_001", 1, calibration(2))
            plan = SyncPlan(logs, (a, b), b, logs / "remote")
            with self.assertRaisesRegex(OSError, "Could not preserve subject eye calibrations"):
                ExperimentDataSync(logs, None).prune(plan)
            self.assertTrue(a.exists())
            self.assertTrue(b.exists())
