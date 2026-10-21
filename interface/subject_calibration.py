"""Durable subject-specific gaze calibration, independent of experiment retention."""

from __future__ import annotations

import copy
import datetime as dt
import fcntl
import hashlib
import json
import os
import re
import tempfile
import warnings
from pathlib import Path
from typing import Iterable

from bin.eye_tracking_config import load_eye_tracking_settings
from interface.experiment_history import _sync_directory


CALIBRATION_CACHE_DIRECTORY = ".subject_calibrations"
CALIBRATION_SYNC_EXCLUDE = f"/{CALIBRATION_CACHE_DIRECTORY}/"
_EXPERIMENT_NAME = re.compile(r"exp_(.+)_(\d{8})_(\d+)")


def _key(experiment_name: str, block_number: int) -> tuple[str, int, int]:
    match = _EXPERIMENT_NAME.fullmatch(experiment_name)
    if match is None or int(match[3]) < 1:
        raise ValueError(f"Invalid calibration experiment: {experiment_name}")
    dt.datetime.strptime(match[2], "%Y%m%d")
    if isinstance(block_number, bool) or not isinstance(block_number, int) or block_number < 1:
        raise ValueError("Calibration block number must be positive")
    return match[2], int(match[3]), block_number


def validate_eye_calibration(candidate: dict) -> dict:
    """Validate complete saved calibrations without filling missing gains from defaults."""
    if (not isinstance(candidate, dict) or "interruption_error" in candidate
            or candidate.get("completion_status", "complete") != "complete"):
        raise ValueError("Expected a completed eye calibration")
    candidate = {"x_cross": 0.0, "y_cross": 0.0, **copy.deepcopy(candidate)}
    load_eye_tracking_settings({
        "eye_tracking": {"enabled": True}, "eye_tracker_calibration": candidate,
    })
    return candidate


class SubjectCalibrationStore:
    def __init__(self, logs_dir: Path):
        self.directory = Path(logs_dir) / CALIBRATION_CACHE_DIRECTORY

    def _path(self, subject: str) -> Path:
        # Full names are identities; sanitizing names or using subject codes can
        # conflate different subjects. A digest is also safe as a filename.
        return self.directory / (hashlib.sha256(subject.encode("utf-8")).hexdigest() + ".json")

    def load(self, subject: str) -> dict | None:
        path = self._path(subject)
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        try:
            if record["version"] != 1 or record["subject"] != subject:
                raise ValueError("incorrect version or subject")
            _key(record["experiment_name"], record["block_number"])
            validate_eye_calibration(record["eye_tracker_calibration"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid subject calibration cache {path}: {exc}") from exc
        return record

    def save(self, *, subject: str, experiment_name: str, block_number: int,
             calibration: dict) -> None:
        candidate = validate_eye_calibration(calibration)
        key = _key(experiment_name, block_number)
        record = dict(version=1, subject=subject, experiment_name=experiment_name,
                      block_number=block_number, eye_tracker_calibration=candidate)
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self._path(subject)
        with path.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            previous = self.load(subject)
            if previous is not None and key <= _key(
                previous["experiment_name"], previous["block_number"],
            ):
                # Retry a failed directory flush from an earlier replacement
                # before cleanup is allowed to remove the original artifact.
                _sync_directory(self.directory)
                _sync_directory(self.directory.parent)
                return
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.directory,
                                                 prefix=".calibration-", delete=False) as handle:
                    temporary = Path(handle.name)
                    json.dump(record, handle, indent=2, allow_nan=False)
                    handle.write("\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, path)
                _sync_directory(self.directory)
                _sync_directory(self.directory.parent)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)

    def remember_experiments(self, directories: Iterable[Path], *, subject: str | None = None) -> None:
        """Backfill completed block artifacts before launch or sync cleanup.

        Experiment and block numbers determine recency, never modification
        times. Initial settings and interrupted recovery files are not promoted.
        """
        for directory in directories:
            if directory.is_symlink() or not _EXPERIMENT_NAME.fullmatch(directory.name):
                continue
            artifacts = list(directory.glob("blocks/*/calibration.json"))
            if not artifacts:
                continue
            state = json.loads((directory / "state.json").read_text(encoding="utf-8"))
            saved_subject = state.get("subject") if isinstance(state, dict) else None
            if not isinstance(saved_subject, str) or not saved_subject:
                raise ValueError(f"Missing calibration subject in {directory / 'state.json'}")
            if subject is not None and saved_subject != subject:
                continue
            for artifact in artifacts:
                match = re.fullmatch(r"([1-9][0-9]*)_.+", artifact.parent.name)
                if match is None or artifact.is_symlink() or artifact.parent.is_symlink():
                    continue
                payload = json.loads(artifact.read_text(encoding="utf-8"))
                if not isinstance(payload, dict) or "eye_tracker_calibration" not in payload:
                    continue
                try:
                    candidate = validate_eye_calibration(payload["eye_tracker_calibration"])
                except (TypeError, ValueError) as exc:
                    warnings.warn(f"Skipping invalid eye calibration {artifact}: {exc}", RuntimeWarning)
                    continue
                self.save(subject=saved_subject, experiment_name=directory.name,
                          block_number=int(match[1]), calibration=candidate)
