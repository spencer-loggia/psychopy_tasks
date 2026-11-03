import csv
import datetime as dt
import json
import shutil
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from interface.experiment_history import ExperimentHistory, HISTORY_FILENAME


class ExperimentHistoryTests(unittest.TestCase):
    def _rows(self, logs):
        with (logs / HISTORY_FILENAME).open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle, delimiter="\t"))

    def _reserve(self, history, **overrides):
        kwargs = dict(subject="Subject One", subject_code="S1", started_at=dt.datetime(2026, 10, 21, 9))
        kwargs.update(overrides)
        return history.reserve(**kwargs)

    def test_reservations_survive_missing_directories_and_are_per_subject_and_day(self):
        with tempfile.TemporaryDirectory() as temporary:
            logs = Path(temporary) / "logs"
            history = ExperimentHistory(logs)
            first, number = self._reserve(history)
            self.assertEqual(number, 1)
            self.assertFalse(first.exists())  # Crash can occur before mkdir.
            second, number = self._reserve(ExperimentHistory(logs))
            self.assertEqual(number, 2)
            second.mkdir()
            shutil.rmtree(second)
            self.assertEqual(self._reserve(ExperimentHistory(logs))[1], 3)
            self.assertEqual(self._reserve(history, subject="Other", subject_code="S2")[1], 1)
            self.assertEqual(self._reserve(history, started_at=dt.datetime(2026, 10, 22))[1], 1)
            self.assertEqual(len(self._rows(logs)), 5)

    def test_imports_all_legacy_subjects_and_skips_gaps_in_numbers(self):
        with tempfile.TemporaryDirectory() as temporary:
            logs = Path(temporary) / "logs"
            older = logs / "exp_S1_20261021_007"
            other = logs / "exp_S2_20261021_003"
            older.mkdir(parents=True)
            other.mkdir()
            (older / "clock.json").write_text(json.dumps({"origin_datetime": "2026-10-21T09:00:00"}))
            (older / "state.json").write_text(json.dumps({"subject": "Subject One"}))
            (older / "blocks.tsv").write_text(
                "subject\tstart_time\tend_time\n"
                "Subject One\t2026-10-21 09:00:10\t2026-10-21 09:00:15\n"
                "Subject One\t2026-10-21 09:00:20\t\n"
            )
            history = ExperimentHistory(logs)
            self.assertEqual(self._reserve(history)[1], 8)
            row = self._rows(logs)[0]
            self.assertEqual(row["experiment_name"], older.name)
            self.assertEqual(row["subject"], "Subject One")
            self.assertEqual(row["block_count"], "2")
            self.assertEqual(row["completed_block_count"], "1")
            self.assertEqual(row["runtime_seconds"], "20.000000")
            self.assertEqual(row["end_time"], "")
            self.assertEqual(row["status"], "imported")
            shutil.rmtree(other)
            self.assertEqual(self._reserve(ExperimentHistory(logs), subject_code="S2")[1], 4)

    def test_corrupt_legacy_metadata_still_reserves_its_number(self):
        with tempfile.TemporaryDirectory() as temporary:
            logs = Path(temporary) / "logs"
            legacy = logs / "exp_S1_20261021_009"
            legacy.mkdir(parents=True)
            for name in ("state.json", "clock.json", "blocks.tsv"):
                (legacy / name).write_bytes(b"\xffinvalid")
            self.assertEqual(self._reserve(ExperimentHistory(logs))[1], 10)
            row = self._rows(logs)[0]
            self.assertEqual(row["status"], "imported")
            self.assertEqual(row["start_time"], "")
            self.assertEqual(row["block_count"], "")

    def test_corrupt_history_blocks_allocation_and_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            logs = Path(temporary)
            path = logs / HISTORY_FILENAME
            for content in ("", "bad header\n", "\t".join(("experiment_name",))):
                with self.subTest(content=content):
                    path.write_text(content)
                    with self.assertRaisesRegex(OSError, "Cannot read local experiment history"):
                        self._reserve(ExperimentHistory(logs))
                    self.assertEqual(path.read_text(), content)

    def test_failed_atomic_write_leaves_previous_history_intact(self):
        with tempfile.TemporaryDirectory() as temporary:
            logs = Path(temporary)
            history = ExperimentHistory(logs)
            self._reserve(history)
            original = history.path.read_bytes()
            with patch("interface.experiment_history.os.replace", side_effect=OSError("disk error")):
                with self.assertRaisesRegex(OSError, "disk error"):
                    self._reserve(history)
            self.assertEqual(history.path.read_bytes(), original)
            self.assertEqual(self._reserve(ExperimentHistory(logs))[1], 2)

    def test_sync_failure_prevents_reservation_from_returning(self):
        with tempfile.TemporaryDirectory() as temporary:
            logs = Path(temporary)
            with patch("interface.experiment_history.os.fsync", side_effect=OSError("disk error")):
                with self.assertRaisesRegex(OSError, "disk error"):
                    self._reserve(ExperimentHistory(logs))
            self.assertFalse((logs / HISTORY_FILENAME).exists())

    def test_concurrent_processes_reserve_distinct_numbers(self):
        script = """
import datetime as dt
import sys
from pathlib import Path
from interface.experiment_history import ExperimentHistory
_, number = ExperimentHistory(Path(sys.argv[1])).reserve(
    subject='Subject One', subject_code='S1', started_at=dt.datetime(2026, 10, 21))
print(number)
"""
        with tempfile.TemporaryDirectory() as temporary:
            logs = Path(temporary) / "logs"
            processes = [subprocess.Popen(
                [sys.executable, "-c", script, str(logs)],
                cwd=Path(__file__).resolve().parents[1], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            ) for _ in range(4)]
            try:
                numbers = []
                for process in processes:
                    stdout, stderr = process.communicate(timeout=15)
                    self.assertEqual(process.returncode, 0, stderr)
                    numbers.append(int(stdout))
                self.assertEqual(sorted(numbers), [1, 2, 3, 4])
                self.assertEqual(len(self._rows(logs)), 4)
            finally:
                for process in processes:
                    if process.poll() is None:
                        process.kill()
                    process.wait(timeout=5)

    def test_sigkill_keeps_reservation_without_shutdown(self):
        script = """
import datetime as dt
import os
import signal
import sys
from pathlib import Path
from interface.experiment_history import ExperimentHistory
ExperimentHistory(Path(sys.argv[1])).reserve(
    subject='Subject One', subject_code='S1', started_at=dt.datetime(2026, 10, 21))
os.kill(os.getpid(), signal.SIGKILL)
"""
        with tempfile.TemporaryDirectory() as temporary:
            logs = Path(temporary)
            result = subprocess.run(
                [sys.executable, "-c", script, str(logs)],
                cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=15,
            )
            self.assertEqual(result.returncode, -signal.SIGKILL, result.stderr)
            self.assertEqual(self._reserve(ExperimentHistory(logs))[1], 2)
            row = self._rows(logs)[0]
            self.assertEqual(row["status"], "open")
            self.assertEqual(row["end_time"], "")
            self.assertEqual(row["block_count"], "0")

    def test_kill_during_update_releases_lock_and_preserves_previous_history(self):
        script = """
import os
import signal
import sys
from pathlib import Path
from unittest.mock import patch
from interface.experiment_history import ExperimentHistory
with patch('interface.experiment_history.os.replace',
           side_effect=lambda *_args: os.kill(os.getpid(), signal.SIGKILL)):
    ExperimentHistory(Path(sys.argv[1])).update(
        sys.argv[2], block_count=1, completed_block_count=0, elapsed_ns=5_000_000_000)
"""
        with tempfile.TemporaryDirectory() as temporary:
            logs = Path(temporary)
            history = ExperimentHistory(logs)
            first, _ = self._reserve(history)
            original = history.path.read_bytes()
            result = subprocess.run(
                [sys.executable, "-c", script, str(logs), first.name],
                cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=15,
            )
            self.assertEqual(result.returncode, -signal.SIGKILL, result.stderr)
            self.assertEqual(history.path.read_bytes(), original)
            self.assertTrue((logs / ".experiment_history.tsv.tmp").exists())
            self.assertEqual(self._reserve(ExperimentHistory(logs))[1], 2)
            self.assertEqual(self._rows(logs)[0]["block_count"], "0")


if __name__ == "__main__":
    unittest.main()
