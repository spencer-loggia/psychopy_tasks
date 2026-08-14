"""Experiment-level state, block records, and loop selection."""
from __future__ import annotations

import copy
import csv
import datetime as dt
import json
import os
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Optional

from bin.config import load_config
from bin.logger import (
    CALIBRATION_FILENAME,
    EVENT_LIBRARY_ENV,
    EXACT_SESSION_DIR_ENV,
    sanitize_filename_component,
)


BLOCKS_FILENAME = "blocks.tsv"
STATE_FILENAME = "state.json"
BLOCK_CONFIG_FILENAME = "config.json"
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
        perf_counter: Callable[[], float] = time.perf_counter,
    ):
        self.working_dir = Path(working_dir).resolve()
        self.launch_config_path = Path(launch_config_path).resolve()
        self.config_dir = self.launch_config_path.parent
        self.launch_config = copy.deepcopy(dict(launch_config))
        self.subject_name = str(subject_name)
        self.subject_code = str(subject_code)
        self._perf_counter = perf_counter
        self.started_perf_s = float(perf_counter())
        self.started_at = now or dt.datetime.now()
        self.blocks: list[dict[str, Any]] = []

        initial_state = self.launch_config.get("initial_state")
        if not isinstance(initial_state, dict):
            raise ValueError("Launcher config field 'initial_state' must be a JSON object")
        self.state = copy.deepcopy(initial_state)
        self.state["subject"] = self.subject_name

        self.logs_dir = self.working_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
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
        self._write_state()
        self._write_blocks()

    def _create_experiment_dir(self) -> Path:
        safe_code = sanitize_filename_component(self.subject_code)
        prefix = f"exp_{safe_code}_{self.started_at:%Y%m%d}_"
        used_ids: set[int] = set()
        for candidate in self.logs_dir.glob(f"{prefix}*"):
            suffix = candidate.name[len(prefix):]
            if suffix.isdigit():
                used_ids.add(int(suffix))
        experiment_id = 1
        while experiment_id in used_ids:
            experiment_id += 1
        experiment_dir = self.logs_dir / f"{prefix}{experiment_id:03d}"
        experiment_dir.mkdir(exist_ok=False)
        return experiment_dir

    def milliseconds_since_start(self) -> float:
        return (float(self._perf_counter()) - self.started_perf_s) * 1000.0

    def _write_state(self) -> None:
        _write_json_atomic(self.state_path, self.state)

    def _write_blocks(self) -> None:
        temp_path = self.blocks_path.with_name(f".{self.blocks_path.name}.tmp")
        with temp_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=BLOCK_FIELDS, delimiter="\t")
            writer.writeheader()
            writer.writerows(self.blocks)
        os.replace(temp_path, self.blocks_path)

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
            if state_name == "subject":
                continue
            if isinstance(state_value, list) and state_value and isinstance(state_value[-1], dict):
                for key, value in state_value[-1].items():
                    if key != "set_time":
                        _inject(key, value, state_name)
            elif not isinstance(state_value, list):
                _inject(state_name, state_value, state_name)

        for key in ("config_name", "output_dir", "subject", *injected):
            block_config.pop(key, None)
        block_config["config_name"] = block_name
        block_config["output_dir"] = str(output_dir)
        block_config["subject"] = self.subject_name
        block_config.update(injected)
        block_config["fullscreen"] = True

        config_path = output_dir / BLOCK_CONFIG_FILENAME
        _write_json_atomic(config_path, block_config)
        relative_output = output_dir.relative_to(self.experiment_dir)
        self.blocks.append(
            {
                "block_name": block_name,
                "block_num": block_num,
                "subject": self.subject_name,
                "start_time": f"{self.milliseconds_since_start():.3f}",
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
        return environment

    def finish_block(self, block: PreparedBlock) -> None:
        if not self.blocks or int(self.blocks[-1]["block_num"]) != block.block_num:
            raise ValueError(f"Block {block.block_num} is not the active block")
        self.blocks[-1]["end_time"] = f"{self.milliseconds_since_start():.3f}"
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
            required_fields = [key for key in history[-1] if key != "set_time"]
            if not all(field in candidate for field in required_fields):
                continue
            entry = {field: copy.deepcopy(candidate[field]) for field in required_fields}
            entry["set_time"] = self.milliseconds_since_start()
            history.append(entry)
            changed = True
        if changed:
            self._write_state()
        return changed
