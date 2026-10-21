import csv
import datetime as dt
import json
import os
import platform
import random
import socket
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from bin.logger import SessionLogBundle
from bin.task_clock import EXPERIMENT_ORIGIN_ENV
from bin.eye_tracking import (
    EXPERIMENT_BLOCK_NUMBER_ENV,
    EXPERIMENT_UART_PENDING_ENV,
    EXPERIMENT_UART_READY_ENV,
    UARTEyeTracker,
)
from bin.eye_tracker_wire.controller import ResponseTimeout
from bin.eye_tracker_wire.protocol import CommandCode, StatusFlag, crc32c, decode_command
from bin.eye_tracking_config import load_eye_tracking_settings
from interface.experiment_manager import (
    EVENT_LIBRARY_ENV,
    EXACT_SESSION_DIR_ENV,
    ExperimentManager,
    _git_commit,
    _runtime_metadata,
    task_run_sequence,
    task_variants,
)


class FakeClock:
    def __init__(self, value=100.0):
        self.value = float(value)

    def __call__(self):
        return self.value


class ExperimentManagerTests(unittest.TestCase):
    def test_launch_commit_is_from_working_checkout_and_survives_state_updates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config, path = self._project(root)
            config["initial_state"]["git_commit"] = "stale-config-value"
            config["initial_state"].update(
                hostname="old-host", os_details={"system": "old-os"}, python_version="old-python",
            )
            config["environment"] = {"python": sys.executable}

            def git(*args):
                return subprocess.run(
                    ["git", "-C", str(root), *args],
                    check=True, capture_output=True, text=True,
                ).stdout.strip()

            def commit(message):
                git("-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                    "-c", "commit.gpgsign=false", "commit", "--allow-empty", "-m", message)
                return git("rev-parse", "HEAD")

            git("init")
            first_commit = commit("launch version")
            kwargs = dict(
                working_dir=root, launch_config_path=path, launch_config=config,
                subject_name="Subject One", subject_code="S1",
            )
            manager = ExperimentManager(**kwargs)
            state = json.loads(manager.state_path.read_text())
            self.assertEqual(state["git_commit"], first_commit)
            self.assertEqual(state["hostname"], socket.gethostname())
            self.assertEqual(state["python_version"], sys.version)
            self.assertEqual(state["os_details"]["system"], platform.system())
            self.assertEqual(state["os_details"]["release"], platform.release())
            self.assertEqual(state["os_details"]["version"], platform.version())
            self.assertEqual(state["os_details"]["machine"], platform.machine())
            self.assertEqual(state["os_details"]["platform"], platform.platform())
            launch_metadata = {key: state[key] for key in (
                "git_commit", "hostname", "os_details", "python_version",
            )}
            self.assertEqual(config["initial_state"]["git_commit"], "stale-config-value")

            # A later HEAD change must not relabel an experiment's launch version.
            second_commit = commit("next version")
            self.assertNotEqual(first_commit, second_commit)
            block = manager.prepare_block(
                task_name="demo", launch_value="task/demo.py", config_value=None,
            )
            generated_config = json.loads(block.config_path.read_text())
            for key in launch_metadata:
                self.assertNotIn(key, generated_config)
            (block.output_dir / "calibration.json").write_text(json.dumps({
                "eye_tracker_calibration": {
                    "x_scale": 1, "y_scale": 1, "x_offset": 0, "y_offset": 0,
                },
                "git_commit": "incorrect-calibration-value",
            }))
            manager.finish_block(block)
            state = json.loads(manager.state_path.read_text())
            self.assertEqual(len(state["eye_tracker_calibration"]), 2)
            self.assertEqual(state["git_commit"], first_commit)
            self.assertEqual({key: state[key] for key in launch_metadata}, launch_metadata)
            next_manager = ExperimentManager(**kwargs)
            self.assertEqual(json.loads(next_manager.state_path.read_text())["git_commit"], second_commit)

    def test_unavailable_git_identity_is_explicitly_null(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config, path = self._project(root)
            config["initial_state"]["git_commit"] = "stale-config-value"
            manager = ExperimentManager(
                working_dir=root, launch_config_path=path, launch_config=config,
                subject_name="Subject One", subject_code="S1",
            )
            self.assertIsNone(json.loads(manager.state_path.read_text())["git_commit"])
            for error in (FileNotFoundError("git"), subprocess.TimeoutExpired("git", 5)):
                with self.subTest(error=type(error).__name__), patch(
                    "interface.experiment_manager.subprocess.run", side_effect=error,
                ):
                    self.assertIsNone(_git_commit(root))
                    self.assertIsNone(_runtime_metadata(root, "unavailable-python")["python_version"])

    def test_subprocess_blocks_share_one_origin_and_exception_still_records_end(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config, path = self._project(root)
            (root / "event_name_library.json").write_text(
                (Path(__file__).resolve().parents[1] / "event_name_library.json").read_text()
            )
            manager = ExperimentManager(
                working_dir=root, launch_config_path=path, launch_config=config,
                subject_name="Test", subject_code="T",
            )
            boundaries = []
            for index in range(2):
                block = manager.prepare_block(task_name="demo", launch_value="task/demo.py", config_value=None)
                environment = manager.subprocess_environment(block)
                self.assertEqual(int(environment[EXPERIMENT_ORIGIN_ENV]), manager.task_clock.origin_perf_ns)
                script = (
                    "import os; from bin.logger import SessionLogBundle; "
                    "logs = SessionLogBundle(output_root=os.environ['NEURO_TASK_SESSION_DIR'], "
                    "task_name='active_foraging', config_name='test', auto_flush=False); "
                    + ("logs.close()" if index == 0 else "raise RuntimeError('test task failure')")
                )
                result = subprocess.run([sys.executable, "-c", script], env=environment, capture_output=True)
                self.assertEqual(result.returncode, index, result.stderr.decode())
                manager.finish_block(block)
                with (block.output_dir / "event_log.tsv").open() as handle:
                    rows = list(csv.DictReader(handle, delimiter="\t"))
                self.assertEqual([row["event"] for row in rows], ["block_start", "block_end"])
                boundaries.extend(int(row["task_time_ns"]) for row in rows)
            self.assertGreater(boundaries[0], 0)
            self.assertEqual(boundaries, sorted(set(boundaries)))
            metadata = json.loads((manager.experiment_dir / "clock.json").read_text())
            self.assertEqual(metadata["origin_perf_counter_ns"], manager.task_clock.origin_perf_ns)
            with manager.blocks_path.open() as handle:
                for row in csv.DictReader(handle, delimiter="\t"):
                    for field in ("start_time", "end_time"):
                        parsed = dt.datetime.strptime(row[field], "%Y-%m-%d %H:%M:%S")
                        self.assertEqual(parsed.microsecond, 0)

    def test_numbered_experiment_and_skipped_block_reach_uart_with_identical_retries(self):
        fixture = json.loads((Path(__file__).parent / "fixtures/eye_uart_v1.json").read_text())
        packets = []
        channel = Mock()
        channel.__enter__ = Mock(return_value=channel)
        channel.__exit__ = Mock(return_value=False)
        task_tracker = UARTEyeTracker()

        def exchange(packet, timeout_s):
            packets.append(packet)
            command = decode_command(packet)
            # Lose the first ACK of both starts, then acknowledge identical retries.
            if command.code in (CommandCode.START_EXPERIMENT, CommandCode.START_TRACKING):
                if packets.count(packet) == 1:
                    raise ResponseTimeout("lost ACK")
            flags = StatusFlag.EXPERIMENT_ACTIVE
            if command.code == CommandCode.START_TRACKING:
                flags |= StatusFlag.TRACKING | StatusFlag.CONFIGURED | StatusFlag.RECORDING
                task_tracker._stop.set()
            elif command.code == CommandCode.STOP_EXPERIMENT:
                flags = StatusFlag(0)
            response = bytearray.fromhex(fixture["status"])
            struct.pack_into("<H", response, 4, flags)
            struct.pack_into("<I", response, 12, command.sequence)
            struct.pack_into("<I", response, 60, crc32c(response[:60]))
            return bytes(response)

        channel.exchange.side_effect = exchange
        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "interface.experiment_eye_tracker.SerialPacketChannel", return_value=channel
        ), patch("bin.eye_tracking.SerialPacketChannel", return_value=channel), patch(
            "bin.eye_tracker_warnings.show_eye_tracker_warning"
        ) as warning:
            root = Path(tmpdir)
            config, path = self._project(root)
            config["eye_tracker_experiment"] = {"enabled": True}
            (root / "logs" / "exp_Y_20261019_001").mkdir(parents=True)
            manager = ExperimentManager(
                working_dir=root, launch_config_path=path, launch_config=config,
                subject_name="Yuri", subject_code="Y",
                now=dt.datetime(2026, 10, 19, 14, 30, tzinfo=dt.timezone.utc),
            )
            try:
                manager.eye_tracker.start_future.result(timeout=2)
                self.assertEqual(manager.experiment_dir.name, "exp_Y_20261019_002")
                manager.prepare_block(task_name="no_tracking", launch_value="task/demo.py", config_value=None)
                block = manager.prepare_block(task_name="tracking", launch_value="task/demo.py", config_value=None)
                with patch.dict(os.environ, manager.subprocess_environment(block)):
                    task_tracker._run()
            finally:
                manager.finish()
                manager.eye_tracker.stop_future.result(timeout=2)
            warning.assert_not_called()
        commands = [decode_command(packet) for packet in packets]
        self.assertEqual([command.code for command in commands], [
            CommandCode.START_EXPERIMENT, CommandCode.START_EXPERIMENT,
            CommandCode.START_TRACKING, CommandCode.START_TRACKING,
            CommandCode.STOP_TRACKING, CommandCode.STOP_EXPERIMENT,
        ])
        self.assertEqual(commands[0].experiment_number, 2)
        self.assertEqual(commands[0].subject, "Yuri")
        self.assertEqual(commands[2].block_number, 2)
        self.assertEqual(packets[0], packets[1])
        self.assertEqual(packets[2], packets[3])
        for command in commands[4:]:
            self.assertIsNone(command.experiment_number)
            self.assertIsNone(command.block_number)

    def test_lifecycle_uses_full_subject_start_time_and_exports_uart_handoff(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config, config_path = self._project(root)
            started = dt.datetime(2026, 10, 14, 10, 30)
            with patch("interface.experiment_manager.ExperimentEyeTracker") as tracker:
                manager = ExperimentManager(
                    working_dir=root, launch_config_path=config_path, launch_config=config,
                    subject_name="Yuri", subject_code="Y", now=started,
                )
                tracker.assert_called_once_with(
                    config, subject="Yuri", started_at=started, directory=manager.experiment_dir,
                    experiment_number=1,
                )
                tracker.return_value.pending_path = manager.experiment_dir / ".pending"
                tracker.return_value.ready_path = manager.experiment_dir / ".ready"
                block = manager.prepare_block(task_name="demo", launch_value="task/demo.py", config_value="configs/demo.json")
                env = manager.subprocess_environment(block)
                self.assertEqual(env["NEURO_EYE_TRACKER_EXPERIMENT_PENDING"], str(tracker.return_value.pending_path))
                self.assertEqual(env[EXPERIMENT_UART_READY_ENV], str(tracker.return_value.ready_path))
                self.assertEqual(env[EXPERIMENT_BLOCK_NUMBER_ENV], "1")
                # Blocks with no tracking still consume their local number.
                block = manager.prepare_block(task_name="demo", launch_value="task/demo.py", config_value="configs/demo.json")
                self.assertEqual(manager.subprocess_environment(block)[EXPERIMENT_BLOCK_NUMBER_ENV], "2")
                manager.finish()
                tracker.return_value.finish.assert_called_once()

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
                perf_counter_ns=lambda: round(clock() * 1_000_000_000),
            )

            self.assertEqual(manager.experiment_dir.name, "exp_S1_20260813_001")
            self.assertTrue((manager.experiment_dir / "launch_config.json").is_file())
            self.assertTrue((manager.experiment_dir / "event_name_library.json").is_file())
            self.assertFalse((manager.experiment_dir / "task_configs").exists())
            initial_state = json.loads(manager.state_path.read_text(encoding="utf-8"))
            self.assertEqual(initial_state["subject"], "Subject One")
            self.assertEqual(initial_state["eye_tracker_calibration"][0]["task_time_ns"], 0)
            self.assertNotIn("set_time", initial_state["eye_tracker_calibration"][0])

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
            self.assertEqual(block_rows[0]["start_time"], "2026-08-13 09:30:00")
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
                            "input_units": "crop_local_pixels",
                            "uart": {"device": "/dev/test-uart", "simulate": True},
                            "filter": {"ema_gamma": 0.2},
                            "gain_anchor_px": [100, 50],
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
            self.assertEqual(latest["task_time_ns"], 500_000_000)
            self.assertEqual((latest["x_cross"], latest["y_cross"]), (0.0, 0.0))
            self.assertNotIn("ignored_metadata", latest)

            with manager.blocks_path.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(rows[0]["end_time"], "2026-08-13 09:30:00")

            (root / "configs" / "demo.json").write_text(
                json.dumps({"config_name": "changed_again", "task_value": 1000,
                            "eye_tracking": {"enabled": True, "calibration_file": "older.json"}}),
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
            settings = load_eye_tracking_settings(second_config)
            self.assertEqual(settings.uart.device, "/dev/test-uart")
            self.assertTrue(settings.uart.simulate)
            self.assertEqual(settings.filter.ema_gamma, 0.2)
            self.assertEqual(settings.gain_anchor_px, (100, 50))
            self.assertEqual(settings.calibration.x_scale, 0.1)

    def test_affine_calibration_survives_legacy_state_schema_and_block_injection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            launch_config, launch_path = self._project(root)
            manager = ExperimentManager(
                working_dir=root, launch_config_path=launch_path,
                launch_config=launch_config, subject_name="Subject One",
                subject_code="S1", now=dt.datetime(2026, 8, 13),
            )
            # The launch history intentionally declares only the old four
            # coefficients; the completed calibration adds both cross terms.
            coefficients = {
                "x_scale": 0.002, "y_scale": -0.004,
                "x_offset": -0.2, "y_offset": 0.5,
                "x_cross": 0.003, "y_cross": -0.001,
            }
            first = manager.prepare_block(
                task_name="demo", launch_value="task/demo.py", config_value="configs/demo.json",
            )
            calibration_path = first.output_dir / "calibration.json"
            calibration_path.write_text(json.dumps({"eye_tracker_calibration": {
                **coefficients,
                "input_units": "crop_local_pixels",
                "uart": {"device": "/dev/affine-uart", "simulate": True},
                "gain_anchor_px": [100, 50],
            }}), encoding="utf-8")
            manager.finish_block(first)
            state = json.loads(manager.state_path.read_text(encoding="utf-8"))
            latest = state["eye_tracker_calibration"][-1]
            self.assertEqual({key: latest[key] for key in coefficients}, coefficients)

            (root / "configs" / "demo.json").write_text(json.dumps({
                "eye_tracking": {"enabled": True, "calibration_file": "obsolete.json"},
                "x_cross": 99, "y_cross": 99,
            }), encoding="utf-8")
            second = manager.prepare_block(
                task_name="demo", launch_value="task/demo.py", config_value="configs/demo.json",
            )
            generated = json.loads(second.config_path.read_text(encoding="utf-8"))
            self.assertEqual({key: generated[key] for key in coefficients}, coefficients)
            self.assertEqual(generated["eye_tracking"]["calibration_file"], str(calibration_path.resolve()))
            settings = load_eye_tracking_settings(generated)
            self.assertEqual(settings.uart.device, "/dev/affine-uart")
            self.assertEqual(settings.gain_anchor_px, (100, 50))
            self.assertAlmostEqual(settings.calibration.map_position(120, 60)[0], 0.22)
            self.assertAlmostEqual(settings.calibration.map_position(120, 60)[1], 0.14)

            # A later four-coefficient artifact must still import after the
            # history gained cross fields, and clear the old affine cross terms.
            legacy = {key: value for key, value in coefficients.items() if "cross" not in key}
            legacy_path = second.output_dir / "calibration.json"
            legacy_path.write_text(json.dumps({"eye_tracker_calibration": legacy}), encoding="utf-8")
            manager.finish_block(second)
            state = json.loads(manager.state_path.read_text(encoding="utf-8"))
            latest = state["eye_tracker_calibration"][-1]
            self.assertEqual(len(state["eye_tracker_calibration"]), 3)
            self.assertEqual((latest["x_cross"], latest["y_cross"]), (0.0, 0.0))
            third = manager.prepare_block(
                task_name="demo", launch_value="task/demo.py", config_value="configs/demo.json",
            )
            generated = json.loads(third.config_path.read_text(encoding="utf-8"))
            self.assertEqual((generated["x_cross"], generated["y_cross"]), (0.0, 0.0))
            self.assertEqual(generated["eye_tracking"]["calibration_file"], str(legacy_path.resolve()))
            settings = load_eye_tracking_settings(generated)
            self.assertEqual((settings.calibration.x_cross, settings.calibration.y_cross), (0.0, 0.0))

    def test_generated_block_persists_launcher_screen_selection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            launch_config, launch_path = self._project(root)
            (root / "configs" / "demo.json").write_text(
                json.dumps(
                    {
                        "config_name": "screen_parity",
                        "screens": {"main": 0, "experimenter": 1},
                        "main_screen": 0,
                        "secondary_screen": 1,
                    }
                ),
                encoding="utf-8",
            )

            with patch.dict(
                os.environ,
                {
                    "NEURO_TASK_SCREEN_ENV_OVERRIDE": "1",
                    "MAIN_SCREEN": "HDMI-1",
                    "SECONDARY_SCREEN": "DSI-1",
                },
                clear=False,
            ):
                manager = ExperimentManager(
                    working_dir=root,
                    launch_config_path=launch_path,
                    launch_config=launch_config,
                    subject_name="Subject One",
                    subject_code="S1",
                    now=dt.datetime(2026, 8, 13, 9, 30),
                )
                block = manager.prepare_block(
                    task_name="demo",
                    launch_value="task/demo.py",
                    config_value="configs/demo.json",
                )

            generated = json.loads(block.config_path.read_text(encoding="utf-8"))
            self.assertEqual(
                generated["screens"],
                {"main": "HDMI-1", "experimenter": "DSI-1"},
            )
            self.assertNotIn("main_screen", generated)
            self.assertNotIn("secondary_screen", generated)

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
            self.assertEqual(first.experiment_number, 1)
            self.assertEqual(second.experiment_number, 2)

    def test_disabled_and_simulated_sync_clear_inherited_uart_context(self):
        for options in ({}, {"enabled": False}, {"enabled": True, "uart": {"simulate": True}}):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                config, path = self._project(root)
                config["eye_tracker_experiment"] = options
                manager = ExperimentManager(
                    working_dir=root, launch_config_path=path, launch_config=config,
                    subject_name="Yuri", subject_code="Y",
                )
                block = manager.prepare_block(task_name="demo", launch_value="task/demo.py", config_value=None)
                inherited = {
                    EXPERIMENT_UART_PENDING_ENV: "/old/pending",
                    EXPERIMENT_UART_READY_ENV: "/old/ready",
                    EXPERIMENT_BLOCK_NUMBER_ENV: "99",
                }
                with patch.dict(os.environ, inherited):
                    env = manager.subprocess_environment(block)
                for key in inherited:
                    self.assertNotIn(key, env)

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
