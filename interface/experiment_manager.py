"""Experiment-level state, block records, and loop selection."""
from __future__ import annotations

import copy
import csv
import datetime as dt
import json
import os
import platform
import random
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Optional

from bin.task_clock import EXPERIMENT_ORIGIN_ENV, TaskClock
from bin.config import load_config
from bin.eye_tracking import (
    EXPERIMENT_BLOCK_NUMBER_ENV,
    EXPERIMENT_UART_PENDING_ENV,
    EXPERIMENT_UART_READY_ENV,
)
from bin.eye_tracking_config import resolve_eye_tracking_paths
from bin.logger import (
    CALIBRATION_FILENAME,
    EVENT_LIBRARY_ENV,
    EXACT_SESSION_DIR_ENV,
    sanitize_filename_component,
)
from bin.screen import (
    MAIN_SCREEN_ENV,
    SCREEN_ENV_OVERRIDE_ENV,
    SECONDARY_SCREEN_ENV,
)
from interface.experiment_eye_tracker import ExperimentEyeTracker
from interface.experiment_history import ExperimentHistory


BLOCKS_FILENAME = "blocks.tsv"
STATE_FILENAME = "state.json"
BLOCK_CONFIG_FILENAME = "config.json"
_EYE_CALIBRATION_COEFFICIENTS = (
    "x_scale", "y_scale", "x_offset", "y_offset", "x_cross", "y_cross",
)
BLOCK_FIELDS = (
    "block_name",
    "block_num",
    "subject",
    "start_time",
    "end_time",
    "out_dir",
)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temp_path = path.with_name(f".{path.name}.tmp")
    temp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temp_path, path)


def _git_commit(working_dir: Path) -> Optional[str]:
    """Read the launch checkout's full HEAD identity once, without modifying it."""
    try:
        result = subprocess.run(
            ["git", "-C", str(working_dir), "rev-parse", "--verify", "HEAD^{commit}"],
            check=True, capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        # Source archives and machines without Git can still run experiments.
        return None
    return result.stdout.strip() or None


def _runtime_metadata(working_dir: Path, python_command: str = "") -> dict[str, Any]:
    """Snapshot the host and configured task interpreter at experiment launch."""
    os_details: dict[str, Any] = {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "platform": platform.platform(),
    }
    if os_details["system"] == "Linux":
        try:
            os_details["distribution"] = platform.freedesktop_os_release()
        except (AttributeError, OSError):
            pass

    python_version: Optional[str] = sys.version
    if python_command:
        try:
            result = subprocess.run(
                [python_command, "-c", "import sys; print(sys.version)"],
                cwd=working_dir, check=True, capture_output=True, text=True, timeout=5,
            )
            python_version = result.stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            python_version = None
    return {
        "hostname": socket.gethostname(),
        "os_details": os_details,
        "python_version": python_version,
    }


def _resolve_path(path_value: str | Path, search_roots: Iterable[Path]) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path.resolve()
    roots = tuple(search_roots)
    for root in roots:
        candidate = (root / path).resolve()
        if candidate.exists():
            return candidate
    return ((roots[0] if roots else Path.cwd()) / path).resolve()


def task_variants(task_name: str, task_cfg: Mapping[str, Any]) -> list[tuple[str, Optional[str]]]:
    """Return validated ``(launch, config)`` variants for one task button."""
    launch = task_cfg.get("launch")
    config = task_cfg.get("config")
    if isinstance(launch, list):
        if not isinstance(config, list):
            raise ValueError(
                f"Loop task '{task_name}' must define 'config' as a list when 'launch' is a list"
            )
        if not launch or len(launch) != len(config):
            raise ValueError(
                f"Loop task '{task_name}' must have non-empty, equal-length 'launch' and 'config' lists"
            )
        variants = []
        for index, (launch_value, config_value) in enumerate(zip(launch, config), start=1):
            if not isinstance(launch_value, str) or not launch_value.strip():
                raise ValueError(f"Loop task '{task_name}' launch item {index} must be a path string")
            if not isinstance(config_value, str) or not config_value.strip():
                raise ValueError(f"Loop task '{task_name}' config item {index} must be a path string")
            variants.append((launch_value, config_value))
        return variants

    if not isinstance(launch, str) or not launch.strip():
        raise ValueError(f"Task '{task_name}' is missing a valid 'launch' path")
    if isinstance(config, list):
        raise ValueError(f"Task '{task_name}' cannot use a config list with a scalar launch path")
    if config is not None and (not isinstance(config, str) or not config.strip()):
        raise ValueError(f"Task '{task_name}' config must be a path string when provided")
    return [(launch, config)]


def task_run_sequence(
    task_name: str,
    task_cfg: Mapping[str, Any],
    *,
    rng: Optional[random.Random] = None,
) -> Iterator[tuple[str, Optional[str]]]:
    """Yield task variants according to scalar or loop launch configuration."""
    variants = task_variants(task_name, task_cfg)
    if not isinstance(task_cfg.get("launch"), list):
        yield variants[0]
        return

    order_mode = str(task_cfg.get("order_mode", "")).strip().lower()
    if order_mode not in {"sequential", "random"}:
        raise ValueError(
            f"Loop task '{task_name}' order_mode must be 'sequential' or 'random'"
        )
    n_iters = task_cfg.get("n_iters")
    if n_iters is not None and (
        not isinstance(n_iters, int) or isinstance(n_iters, bool) or n_iters < 0
    ):
        raise ValueError(f"Loop task '{task_name}' n_iters must be a non-negative integer or null")

    chooser = rng or random.Random()
    iteration = 0
    while n_iters is None or iteration < n_iters:
        if order_mode == "sequential":
            yield variants[iteration % len(variants)]
        else:
            yield chooser.choice(variants)
        iteration += 1


@dataclass(frozen=True)
class PreparedBlock:
    block_num: int
    block_name: str
    launch_path: Path
    config_path: Path
    output_dir: Path


class ExperimentManager:
    """Own one experiment directory and its mutable experiment state."""

    def __init__(
        self,
        *,
        working_dir: Path,
        launch_config_path: Path,
        launch_config: Mapping[str, Any],
        subject_name: str,
        subject_code: str,
        now: Optional[dt.datetime] = None,
        perf_counter_ns: Callable[[], int] = time.perf_counter_ns,
    ):
        self.working_dir = Path(working_dir).resolve()
        self.launch_config_path = Path(launch_config_path).resolve()
        self.config_dir = self.launch_config_path.parent
        self.launch_config = copy.deepcopy(dict(launch_config))
        self.subject_name = str(subject_name)
        self.subject_code = str(subject_code)
        self._perf_counter_ns = perf_counter_ns
        self.task_clock = TaskClock(origin_perf_ns=perf_counter_ns())
        self.started_at = now or dt.datetime.now()
        self.blocks: list[dict[str, Any]] = []
        self._eye_calibration_path: Optional[Path] = None

        initial_state = self.launch_config.get("initial_state")
        if not isinstance(initial_state, dict):
            raise ValueError("Launcher config field 'initial_state' must be a JSON object")
        self.state = copy.deepcopy(initial_state)
        # Normalize legacy calibration-history timestamps at the config boundary.
        for history in self.state.values():
            if isinstance(history, list):
                for entry in history:
                    if isinstance(entry, dict) and "set_time" in entry:
                        milliseconds = entry.pop("set_time")
                        entry.setdefault("task_time_ns", round(float(milliseconds) * 1_000_000))
        self.state["subject"] = self.subject_name
        self.state["git_commit"] = _git_commit(self.working_dir)
        python_command = str(
            (self.launch_config.get("environment") or {}).get("python") or ""
        ).strip()
        self.state.update(_runtime_metadata(self.working_dir, python_command))

        self.logs_dir = self.working_dir / "logs"
        self.history = ExperimentHistory(self.logs_dir)
        self.experiment_dir = self._create_experiment_dir()
        self.blocks_dir = self.experiment_dir / "blocks"
        self.blocks_dir.mkdir()
        self.state_path = self.experiment_dir / STATE_FILENAME
        self.blocks_path = self.experiment_dir / BLOCKS_FILENAME
        self.event_library_path = self.experiment_dir / "event_name_library.json"

        shutil.copy2(self.launch_config_path, self.experiment_dir / "launch_config.json")
        source_event_library = self.working_dir / "event_name_library.json"
        if not source_event_library.is_file():
            raise FileNotFoundError(f"Event library not found: {source_event_library}")
        shutil.copy2(source_event_library, self.event_library_path)
        _write_json_atomic(self.experiment_dir / "clock.json", {
            "origin_perf_counter_ns": self.task_clock.origin_perf_ns,
            "origin_datetime": self.started_at.isoformat(timespec="microseconds"),
        })
        self._write_state()
        self._write_blocks()
        self.eye_tracker = ExperimentEyeTracker(
            self.launch_config, subject=self.subject_name,
            started_at=self.started_at, directory=self.experiment_dir,
            experiment_number=self.experiment_number,
        )

    def finish(self) -> None:
        """Save the final local summary and request remote recording closure."""
        try:
            self._update_history(ended=True)
        finally:
            self.eye_tracker.finish()

    def _create_experiment_dir(self) -> Path:
        experiment_dir, self.experiment_number = self.history.reserve(
            subject=self.subject_name, subject_code=self.subject_code, started_at=self.started_at,
        )
        experiment_dir.mkdir(exist_ok=False)
        return experiment_dir

    def _update_history(self, *, ended: bool = False) -> None:
        self.history.update(
            self.experiment_dir.name, block_count=len(self.blocks),
            completed_block_count=sum(bool(block["end_time"]) for block in self.blocks),
            elapsed_ns=self.task_time_ns(), ended=ended,
        )

    def task_time_ns(self) -> int:
        return self.task_clock.task_time_ns(timestamp_perf_ns=self._perf_counter_ns())

    def _block_datetime(self) -> str:
        # Derive human-readable times from the same experiment origin, so wall
        # clock adjustments during a run cannot change elapsed block timing.
        when = self.started_at + dt.timedelta(microseconds=self.task_time_ns() // 1000)
        return when.isoformat(sep=" ", timespec="seconds")

    def _write_state(self) -> None:
        _write_json_atomic(self.state_path, self.state)

    def _write_blocks(self) -> None:
        temp_path = self.blocks_path.with_name(f".{self.blocks_path.name}.tmp")
        with temp_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=BLOCK_FIELDS, delimiter="\t")
            writer.writeheader()
            writer.writerows(self.blocks)
        os.replace(temp_path, self.blocks_path)
        self._update_history()

    def prepare_block(
        self,
        *,
        task_name: str,
        launch_value: str,
        config_value: Optional[str],
    ) -> PreparedBlock:
        launch_path = _resolve_path(launch_value, (self.working_dir, self.config_dir))
        if not launch_path.is_file():
            raise FileNotFoundError(f"Task launch script not found: {launch_path}")

        if config_value is None:
            block_config: dict[str, Any] = {}
        else:
            source_config = _resolve_path(config_value, (self.working_dir, self.config_dir))
            if not source_config.is_file():
                raise FileNotFoundError(f"Task config not found: {source_config}")
            block_config = load_config(str(source_config))

        raw_block_name = block_config.get("config_name") or task_name
        block_name = str(raw_block_name).strip() or str(task_name)
        block_num = len(self.blocks) + 1
        output_dir = self.blocks_dir / f"{block_num}_{sanitize_filename_component(block_name)}"
        output_dir.mkdir(exist_ok=False)

        injected: dict[str, Any] = {}

        def _inject(key: str, value: Any, state_name: str) -> None:
            if key in {"config_name", "output_dir", "subject"}:
                raise ValueError(
                    f"Experiment state field '{state_name}' cannot inject reserved config key '{key}'"
                )
            if key in injected:
                raise ValueError(
                    f"Experiment state fields collide on generated config key '{key}' "
                    f"while injecting '{state_name}'"
                )
            injected[key] = copy.deepcopy(value)

        for state_name, state_value in self.state.items():
            if state_name in {"subject", "git_commit", "hostname", "os_details", "python_version"}:
                continue
            if isinstance(state_value, list) and state_value and isinstance(state_value[-1], dict):
                for key, value in state_value[-1].items():
                    if key not in {"set_time", "task_time_ns"}:
                        _inject(key, value, state_name)
            elif not isinstance(state_value, list):
                _inject(state_name, state_value, state_name)

        for key in ("config_name", "output_dir", "subject", *injected):
            block_config.pop(key, None)
        block_config["config_name"] = block_name
        block_config["output_dir"] = str(output_dir)
        block_config["subject"] = self.subject_name
        block_config.update(injected)
        resolve_eye_tracking_paths(block_config, self.launch_config_path.parent)
        # History retains declared scalar fields and all affine eye coefficients.
        # Also reference the accepted source artifact so enabled eye tracking
        # inherits UART/filter settings, units and the gain anchor losslessly.
        if self._eye_calibration_path is not None:
            eye_options = block_config.setdefault("eye_tracking", {})
            if isinstance(eye_options, dict):
                # Match the metadata to the same latest history entry whose
                # coefficients were injected above, even if the source task
                # config originally referenced an older calibration file.
                eye_options["calibration_file"] = str(self._eye_calibration_path)
        block_config["fullscreen"] = True
        runtime_screen_override = str(
            os.environ.get(SCREEN_ENV_OVERRIDE_ENV, "")
        ).strip().lower() in {"1", "true", "yes", "on"}
        if runtime_screen_override and MAIN_SCREEN_ENV in os.environ:
            # Persist the launcher's canonical physical-output names into every
            # generated task config. This makes diagnostic/task parity explicit
            # and avoids depending on task-local legacy numeric selectors or on
            # environment precedence surviving another launch wrapper.
            block_config.pop("main_screen", None)
            block_config.pop("experimenter_screen", None)
            block_config.pop("secondary_screen", None)
            block_config["screens"] = {
                "main": os.environ[MAIN_SCREEN_ENV],
                "experimenter": os.environ.get(SECONDARY_SCREEN_ENV, ""),
            }

        config_path = output_dir / BLOCK_CONFIG_FILENAME
        _write_json_atomic(config_path, block_config)
        relative_output = output_dir.relative_to(self.experiment_dir)
        self.blocks.append(
            {
                "block_name": block_name,
                "block_num": block_num,
                "subject": self.subject_name,
                "start_time": self._block_datetime(),
                "end_time": "",
                "out_dir": str(relative_output),
            }
        )
        self._write_blocks()
        return PreparedBlock(
            block_num=block_num,
            block_name=block_name,
            launch_path=launch_path,
            config_path=config_path,
            output_dir=output_dir,
        )

    def subprocess_environment(self, block: PreparedBlock) -> dict[str, str]:
        environment = os.environ.copy()
        environment[EXACT_SESSION_DIR_ENV] = str(block.output_dir)
        environment[EVENT_LIBRARY_ENV] = str(self.event_library_path)
        environment[EXPERIMENT_ORIGIN_ENV] = str(self.task_clock.origin_perf_ns)
        for key in (EXPERIMENT_UART_PENDING_ENV, EXPERIMENT_UART_READY_ENV, EXPERIMENT_BLOCK_NUMBER_ENV):
            environment.pop(key, None)
        if self.eye_tracker.start_future is not None:
            environment[EXPERIMENT_UART_PENDING_ENV] = str(self.eye_tracker.pending_path)
            environment[EXPERIMENT_UART_READY_ENV] = str(self.eye_tracker.ready_path)
            environment[EXPERIMENT_BLOCK_NUMBER_ENV] = str(block.block_num)
        return environment

    def finish_block(self, block: PreparedBlock) -> None:
        if not self.blocks or int(self.blocks[-1]["block_num"]) != block.block_num:
            raise ValueError(f"Block {block.block_num} is not the active block")
        self.blocks[-1]["end_time"] = self._block_datetime()
        self._write_blocks()
        self._import_calibration(block.output_dir / CALIBRATION_FILENAME)

    def _import_calibration(self, calibration_path: Path) -> bool:
        if not calibration_path.is_file():
            return False
        payload = json.loads(calibration_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Calibration file must contain a JSON object: {calibration_path}")

        changed = False
        for state_name, history in self.state.items():
            if not isinstance(history, list) or not history or not isinstance(history[-1], dict):
                continue
            candidate = payload.get(state_name)
            if not isinstance(candidate, dict):
                continue
            required_fields = [key for key in history[-1] if key not in {"set_time", "task_time_ns"}]
            if state_name == "eye_tracker_calibration":
                # Older launch configs declare only diagonal gains and offsets.
                # Keep fitted cross terms in state so block-level overrides do
                # not silently reduce a full affine calibration to that schema.
                # A later legacy calibration must explicitly clear cross terms
                # from the previous fit, rather than inheriting stale values.
                candidate = {"x_cross": 0.0, "y_cross": 0.0, **candidate}
                required_fields = list(dict.fromkeys(
                    [*required_fields, *_EYE_CALIBRATION_COEFFICIENTS]
                ))
            if not all(field in candidate for field in required_fields):
                continue
            entry = {field: copy.deepcopy(candidate[field]) for field in required_fields}
            entry["task_time_ns"] = self.task_time_ns()
            history.append(entry)
            if state_name == "eye_tracker_calibration":
                self._eye_calibration_path = calibration_path.resolve()
            changed = True
        if changed:
            self._write_state()
        return changed
