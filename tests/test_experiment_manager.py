import csv
import datetime as dt
import json
import os
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bin.logger import SessionLogBundle
from interface.experiment_manager import (
    EVENT_LIBRARY_ENV,
    EXACT_SESSION_DIR_ENV,
    ExperimentManager,
    task_run_sequence,
    task_variants,
)


class FakeClock:
    def __init__(self, value=100.0):
        self.value = float(value)

    def __call__(self):
        return self.value


class ExperimentManagerTests(unittest.TestCase):
    def _project(self, root: Path):
        (root / "task").mkdir()
        (root / "configs").mkdir()
        (root / "task" / "demo.py").write_text("# demo\n", encoding="utf-8")
        (root / "event_name_library.json").write_text(
            json.dumps({"events": {}, "task_event_sets": {}, "event_templates": {}}),
            encoding="utf-8",
        )
        task_config = root / "configs" / "demo.json"
        task_config.write_text(
            json.dumps(
                {
                    "config_name": "demo_block",
                    "output_dir": "old_logs",
                    "subject": "old_subject",
                    "x_scale": 99,
                    "task_value": 7,
                }
            ),
            encoding="utf-8",
        )
        launch_config = {
            "subjects": {"Subject One": "S1"},
            "tasks": {
                "demo": {"launch": "task/demo.py", "config": "configs/demo.json"}
            },
            "initial_state": {
                "subject": None,
                "session_mode": "training",
                "eye_tracker_calibration": [
                    {
                        "x_scale": None,
                        "y_scale": None,
                        "x_offset": None,
                        "y_offset": None,
                        "set_time": 0.0,
                    }
                ],
            },
        }
        launch_path = root / "launch.json"
        launch_path.write_text(json.dumps(launch_config), encoding="utf-8")
        return launch_config, launch_path

    def test_experiment_and_block_state_lifecycle_uses_current_source_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            launch_config, launch_path = self._project(root)
            clock = FakeClock()
            manager = ExperimentManager(
                working_dir=root,
                launch_config_path=launch_path,
                launch_config=launch_config,
                subject_name="Subject One",
                subject_code="S1",
                now=dt.datetime(2026, 8, 13, 9, 30),
                perf_counter=clock,
            )

            self.assertEqual(manager.experiment_dir.name, "exp_S1_20260813_001")
            self.assertTrue((manager.experiment_dir / "launch_config.json").is_file())
            self.assertTrue((manager.experiment_dir / "event_name_library.json").is_file())
            self.assertFalse((manager.experiment_dir / "task_configs").exists())
            initial_state = json.loads(manager.state_path.read_text(encoding="utf-8"))
            self.assertEqual(initial_state["subject"], "Subject One")

            # Each block config is based on the source file as it exists when the
            # block is prepared, even if it changed after experiment creation.
            (root / "configs" / "demo.json").write_text(
                json.dumps({"config_name": "changed_later", "task_value": 999}),
                encoding="utf-8",
            )

            clock.value = 100.125
            block = manager.prepare_block(
                task_name="demo",
                launch_value="task/demo.py",
                config_value="configs/demo.json",
            )
            self.assertEqual(block.output_dir.name, "1_changed_later")
            generated = json.loads(block.config_path.read_text(encoding="utf-8"))
            self.assertEqual(generated["task_value"], 999)
            self.assertEqual(generated["subject"], "Subject One")
            self.assertEqual(generated["session_mode"], "training")
            self.assertEqual(generated["output_dir"], str(block.output_dir))
            self.assertTrue(generated["fullscreen"])
            self.assertIsNone(generated["x_scale"])
            self.assertNotIn("eye_tracker_calibration", generated)
            self.assertNotIn("set_time", generated)

            with manager.blocks_path.open(encoding="utf-8") as handle:
                block_rows = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(block_rows[0]["start_time"], "125.000")
            self.assertEqual(block_rows[0]["end_time"], "")
            self.assertEqual(block_rows[0]["out_dir"], "blocks/1_changed_later")

            (block.output_dir / "calibration.json").write_text(
                json.dumps(
                    {
                        "eye_tracker_calibration": {
                            "x_scale": 0.1,
                            "y_scale": -0.2,
                            "x_offset": 0.3,
                            "y_offset": -0.4,
                            "ignored_metadata": True,
                        }
                    }
                ),
                encoding="utf-8",
            )
            clock.value = 100.5
            manager.finish_block(block)
            state = json.loads(manager.state_path.read_text(encoding="utf-8"))
            latest = state["eye_tracker_calibration"][-1]
            self.assertEqual(
                {key: latest[key] for key in ("x_scale", "y_scale", "x_offset", "y_offset")},
                {"x_scale": 0.1, "y_scale": -0.2, "x_offset": 0.3, "y_offset": -0.4},
            )
            self.assertEqual(latest["set_time"], 500.0)
            self.assertNotIn("ignored_metadata", latest)

            with manager.blocks_path.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(rows[0]["end_time"], "500.000")

            (root / "configs" / "demo.json").write_text(
                json.dumps({"config_name": "changed_again", "task_value": 1000}),
                encoding="utf-8",
            )
            clock.value = 101.0
            second = manager.prepare_block(
                task_name="demo",
                launch_value="task/demo.py",
                config_value="configs/demo.json",
            )
            second_config = json.loads(second.config_path.read_text(encoding="utf-8"))
            self.assertEqual(second.output_dir.name, "2_changed_again")
            self.assertEqual(second_config["task_value"], 1000)
            self.assertEqual(second_config["x_scale"], 0.1)
            self.assertEqual(second_config["y_scale"], -0.2)

    def test_experiment_ids_increment_for_same_subject_and_date(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            launch_config, launch_path = self._project(root)
            kwargs = dict(
                working_dir=root,
                launch_config_path=launch_path,
                launch_config=launch_config,
                subject_name="Subject One",
                subject_code="S1",
                now=dt.datetime(2026, 8, 13),
            )
            first = ExperimentManager(**kwargs)
            second = ExperimentManager(**kwargs)
            self.assertEqual(first.experiment_dir.name, "exp_S1_20260813_001")
            self.assertEqual(second.experiment_dir.name, "exp_S1_20260813_002")

    def test_loop_sequences_validate_and_obey_iteration_policy(self):
        loop = {
            "launch": ["a.py", "b.py"],
            "config": ["a.json", "b.json"],
            "order_mode": "sequential",
            "n_iters": 5,
        }
        self.assertEqual(
            list(task_run_sequence("loop", loop)),
            [
                ("a.py", "a.json"),
                ("b.py", "b.json"),
                ("a.py", "a.json"),
                ("b.py", "b.json"),
                ("a.py", "a.json"),
            ],
        )
        loop["order_mode"] = "random"
        selected = list(task_run_sequence("loop", loop, rng=random.Random(4)))
        self.assertEqual(len(selected), 5)
        self.assertTrue(all(item in task_variants("loop", loop) for item in selected))

        loop["config"] = ["a.json"]
        with self.assertRaisesRegex(ValueError, "equal-length"):
            list(task_run_sequence("loop", loop))

    def test_session_bundle_writes_directly_to_manager_block_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            block_dir = Path(tmpdir) / "block"
            block_dir.mkdir()
            environment = {EXACT_SESSION_DIR_ENV: str(block_dir)}
            with patch.dict(os.environ, environment, clear=False):
                bundle = SessionLogBundle(
                    output_root=block_dir,
                    task_name="active_foraging",
                    config_name="test",
                )
                try:
                    self.assertEqual(bundle.session_dir, block_dir.resolve())
                finally:
                    bundle.close()
            self.assertTrue((block_dir / "event_log.tsv").is_file())
            self.assertFalse(any(path.name.startswith("L_") for path in block_dir.iterdir()))

    def test_manager_subprocess_environment_uses_experiment_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            launch_config, launch_path = self._project(root)
            manager = ExperimentManager(
                working_dir=root,
                launch_config_path=launch_path,
                launch_config=launch_config,
                subject_name="Subject One",
                subject_code="S1",
            )
            block = manager.prepare_block(
                task_name="demo",
                launch_value="task/demo.py",
                config_value="configs/demo.json",
            )
            environment = manager.subprocess_environment(block)
            self.assertEqual(environment[EXACT_SESSION_DIR_ENV], str(block.output_dir))
            self.assertEqual(environment[EVENT_LIBRARY_ENV], str(manager.event_library_path))


if __name__ == "__main__":
    unittest.main()
