"""Safe synchronization and retention for locally cached experiment data."""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional


NFS_FILESYSTEM_TYPES = frozenset({"nfs", "nfs4"})


def mounted_filesystem_type(path: Path) -> Optional[str]:
    """Return the filesystem type containing *path*, or ``None`` on failure."""
    try:
        result = subprocess.run(
            ["findmnt", "--target", str(path), "--noheadings", "--output", "FSTYPE"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    filesystem_type = result.stdout.strip().splitlines()
    return filesystem_type[-1].strip().lower() if filesystem_type else None


@dataclass(frozen=True)
class SyncPlan:
    """A validated set of local experiment directories to copy."""

    source: Path
    experiments: tuple[Path, ...]
    retained_experiment: Path
    destination: Path

    @property
    def command(self) -> list[str]:
        return [
            "rsync",
            "--archive",
            "--",
            f"{self.source}/",
            f"{self.destination}/",
        ]


@dataclass(frozen=True)
class SyncPreparation:
    plan: Optional[SyncPlan]
    warning: Optional[str] = None


class ExperimentDataSync:
    """Plan an rsync and prune old local experiments only after it succeeds."""

    def __init__(
        self,
        logs_dir: Path,
        remote_data_dir: Optional[Path],
        *,
        filesystem_type: Callable[[Path], Optional[str]] = mounted_filesystem_type,
    ) -> None:
        self.logs_dir = Path(logs_dir)
        self.remote_data_dir = (
            None if remote_data_dir is None else Path(remote_data_dir).expanduser()
        )
        self._filesystem_type = filesystem_type

    def prepare(self) -> SyncPreparation:
        """Validate storage and return an immutable sync plan when work exists."""
        destination = self.remote_data_dir
        if destination is None:
            return SyncPreparation(
                None,
                "remote_data_url is not configured; data was not synced",
            )
        filesystem_type = self._filesystem_type(destination)
        if filesystem_type not in NFS_FILESYSTEM_TYPES:
            detail = filesystem_type or "unknown"
            return SyncPreparation(
                None,
                f"Data destination is not on a mounted NFS filesystem "
                f"({destination}; type={detail}); data was not synced",
            )

        if not destination.is_dir():
            return SyncPreparation(
                None,
                f"Data destination is unavailable or is not a directory: {destination}",
            )

        try:
            next(destination.iterdir())
        except StopIteration:
            return SyncPreparation(
                None,
                f"Data destination is empty and may not be mounted: {destination}",
            )
        except OSError as exc:
            return SyncPreparation(
                None,
                f"Could not read data destination {destination}: {exc}",
            )

        try:
            experiments = self._experiment_directories()
        except OSError as exc:
            return SyncPreparation(
                None,
                f"Could not read local experiment cache {self.logs_dir}: {exc}",
            )
        if not experiments:
            return SyncPreparation(None)
        retained = max(experiments, key=self._recency_key)
        return SyncPreparation(
            SyncPlan(
                source=self.logs_dir,
                experiments=tuple(experiments),
                retained_experiment=retained,
                destination=destination,
            )
        )

    def prune(self, plan: SyncPlan) -> tuple[Path, ...]:
        """Delete synced experiments except the most recently modified one."""
        removed = []
        for experiment_dir in plan.experiments:
            if experiment_dir == plan.retained_experiment:
                continue
            shutil.rmtree(experiment_dir)
            removed.append(experiment_dir)
        return tuple(removed)

    def _experiment_directories(self) -> list[Path]:
        if not self.logs_dir.is_dir():
            return []
        return sorted(
            (
                child
                for child in self.logs_dir.iterdir()
                if child.is_dir() and not child.is_symlink()
            ),
            key=lambda path: path.name,
        )

    @staticmethod
    def _recency_key(path: Path) -> tuple[int, str]:
        return path.stat().st_mtime_ns, path.name


def terminate_process(process: subprocess.Popen, *, timeout_s: float = 0.25) -> None:
    """Stop an rsync promptly, escalating to kill if it does not terminate."""
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
