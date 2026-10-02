"""Safe synchronization and retention for locally cached experiment data."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence


NFS_FILESYSTEM_TYPES = frozenset({"nfs", "nfs4"})
FILESYSTEM_TYPE_PROBE_TIMEOUT_S = 5.0
REMOTE_DIRECTORY_PROBE_TIMEOUT_S = 5.0
RSYNC_IO_TIMEOUT_SECONDS = 30
PROCESS_KILL_SIGNAL = getattr(signal, "SIGKILL", signal.SIGTERM)


_REMOTE_DIRECTORY_PROBE_SCRIPT = r"""
import json
import os
import sys

path = sys.argv[1]
if not os.path.isdir(path):
    result = {"is_directory": False, "has_entries": False, "error": None}
else:
    try:
        with os.scandir(path) as entries:
            try:
                next(entries)
            except StopIteration:
                has_entries = False
            else:
                has_entries = True
        result = {
            "is_directory": True,
            "has_entries": has_entries,
            "error": None,
        }
    except OSError as exc:
        result = {
            "is_directory": True,
            "has_entries": False,
            "error": str(exc),
        }

sys.stdout.write(json.dumps(result))
"""


def run_command_bounded(
    command: Sequence[str],
    *,
    timeout_s: float,
    cwd: Optional[Path] = None,
    check: bool = False,
) -> subprocess.CompletedProcess:
    """Run a captured text command without an unbounded post-timeout wait.

    Unlike ``subprocess.run(timeout=...)`` on POSIX, this function does not
    wait forever after sending SIGKILL if the child is stuck in an
    uninterruptible filesystem syscall.  Such a child is handed to a daemon
    reaper and ``TimeoutExpired`` is raised promptly.
    """
    args = list(command)
    timeout_s = max(0.0, float(timeout_s))
    process = subprocess.Popen(
        args,
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=os.name == "posix",
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        terminate_process(process, process_group=os.name == "posix")
        _close_process_pipes(process)
        raise subprocess.TimeoutExpired(
            args,
            timeout_s,
            output=exc.output,
            stderr=exc.stderr,
        ) from None

    completed = subprocess.CompletedProcess(args, process.returncode, stdout, stderr)
    if check and completed.returncode != 0:
        raise subprocess.CalledProcessError(
            completed.returncode,
            completed.args,
            output=completed.stdout,
            stderr=completed.stderr,
        )
    return completed


def mounted_filesystem_type(path: Path) -> Optional[str]:
    """Return the filesystem type containing *path*, or ``None`` on failure."""
    try:
        result = run_command_bounded(
            ["findmnt", "--target", str(path), "--noheadings", "--output", "FSTYPE"],
            check=True,
            timeout_s=FILESYSTEM_TYPE_PROBE_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    filesystem_type = result.stdout.strip().splitlines()
    return filesystem_type[-1].strip().lower() if filesystem_type else None


@dataclass(frozen=True)
class DirectoryProbeResult:
    """Bounded result of checking a possibly stale remote directory."""

    is_directory: bool
    has_entries: bool
    error: Optional[str] = None


def _reap_process_in_background(process: subprocess.Popen) -> None:
    """Eventually reap an uninterruptible child without blocking its caller."""

    def reap() -> None:
        try:
            process.wait()
        except (OSError, subprocess.SubprocessError):
            pass

    threading.Thread(
        target=reap,
        name="remote-filesystem-process-reaper",
        daemon=True,
    ).start()


def _close_process_pipes(process: subprocess.Popen) -> None:
    for stream_name in ("stdin", "stdout", "stderr"):
        stream = getattr(process, stream_name, None)
        if stream is None:
            continue
        try:
            stream.close()
        except OSError:
            pass


def probe_remote_directory(
    path: Path,
    *,
    timeout_s: float = REMOTE_DIRECTORY_PROBE_TIMEOUT_S,
) -> DirectoryProbeResult:
    """Check directory existence/content without touching the path in this process.

    Remote filesystem calls may remain blocked in the kernel even after their
    process receives SIGKILL.  The probe therefore runs in a child, applies a
    deadline, and leaves any still-uninterruptible child with a daemon reaper
    rather than waiting indefinitely in the launcher/UI process.
    """
    timeout_s = max(0.0, float(timeout_s))
    try:
        completed = run_command_bounded(
            [sys.executable, "-c", _REMOTE_DIRECTORY_PROBE_SCRIPT, str(path)],
            timeout_s=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return DirectoryProbeResult(
            False,
            False,
            f"remote directory probe timed out after {timeout_s:g} seconds",
        )
    except (OSError, ValueError) as exc:
        return DirectoryProbeResult(False, False, f"could not start probe: {exc}")

    if completed.returncode != 0:
        detail = completed.stderr.strip() or (
            f"probe exited with status {completed.returncode}"
        )
        return DirectoryProbeResult(False, False, detail)

    try:
        payload = json.loads(completed.stdout)
        is_directory = payload["is_directory"]
        has_entries = payload["has_entries"]
        error = payload["error"]
        if not isinstance(is_directory, bool) or not isinstance(has_entries, bool):
            raise ValueError("probe booleans are invalid")
        if error is not None and not isinstance(error, str):
            raise ValueError("probe error is invalid")
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return DirectoryProbeResult(False, False, f"invalid probe response: {exc}")

    return DirectoryProbeResult(is_directory, has_entries, error)


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
            f"--timeout={RSYNC_IO_TIMEOUT_SECONDS}",
            "--progress",
            "--no-owner",
            "--no-group",
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
        directory_probe: Callable[
            [Path], DirectoryProbeResult
        ] = probe_remote_directory,
    ) -> None:
        self.logs_dir = Path(logs_dir)
        self.remote_data_dir = (
            None if remote_data_dir is None else Path(remote_data_dir).expanduser()
        )
        self._filesystem_type = filesystem_type
        self._directory_probe = directory_probe

    def prepare(self) -> SyncPreparation:
        """Validate storage and return an immutable sync plan when work exists."""
        remote_root = self.remote_data_dir
        if remote_root is None:
            return SyncPreparation(
                None,
                "remote_data_url is not configured; data was not synced",
            )
        filesystem_type = self._filesystem_type(remote_root)
        if filesystem_type not in NFS_FILESYSTEM_TYPES:
            detail = filesystem_type or "unknown"
            return SyncPreparation(
                None,
                f"Data destination is not on a mounted NFS filesystem "
                f"({remote_root}; type={detail}); data was not synced",
            )

        try:
            directory = self._directory_probe(remote_root)
        except OSError as exc:
            return SyncPreparation(
                None,
                f"Could not read data destination {remote_root}: {exc}",
            )
        if directory.error:
            return SyncPreparation(
                None,
                f"Could not read data destination {remote_root}: {directory.error}",
            )
        if not directory.is_directory:
            return SyncPreparation(
                None,
                f"Data destination is unavailable or is not a directory: {remote_root}",
            )
        if not directory.has_entries:
            return SyncPreparation(
                None,
                f"Data destination is empty and may not be mounted: {remote_root}",
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
                destination=remote_root / "experiments",
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


def _signal_process(
    process: subprocess.Popen,
    signal_number: int,
    *,
    process_group: bool,
) -> None:
    if process_group and hasattr(os, "killpg"):
        try:
            os.killpg(process.pid, signal_number)
            return
        except OSError:
            pass
    if signal_number == signal.SIGTERM:
        process.terminate()
    else:
        process.kill()


def _kill_remaining_process_group(process: subprocess.Popen) -> None:
    """Best-effort cleanup for descendants left after the group leader exits."""
    if not hasattr(os, "killpg"):
        return
    try:
        os.killpg(process.pid, PROCESS_KILL_SIGNAL)
    except OSError:
        pass


def terminate_process(
    process: subprocess.Popen,
    *,
    timeout_s: float = 0.25,
    process_group: bool = False,
) -> bool:
    """Stop a process with bounded waits; return whether it was reaped."""
    if process.poll() is not None:
        if process_group:
            _signal_process(
                process,
                signal.SIGTERM,
                process_group=True,
            )
            _kill_remaining_process_group(process)
        return True
    timeout_s = max(0.0, float(timeout_s))
    try:
        _signal_process(
            process,
            signal.SIGTERM,
            process_group=process_group,
        )
        process.wait(timeout=timeout_s)
        if process_group:
            _kill_remaining_process_group(process)
        return True
    except OSError:
        if process.poll() is not None:
            if process_group:
                _kill_remaining_process_group(process)
            return True
    except subprocess.TimeoutExpired:
        pass

    try:
        _signal_process(
            process,
            PROCESS_KILL_SIGNAL,
            process_group=process_group,
        )
        process.wait(timeout=timeout_s)
        return True
    except OSError:
        if process.poll() is not None:
            if process_group:
                _kill_remaining_process_group(process)
            return True
    except subprocess.TimeoutExpired:
        pass

    _reap_process_in_background(process)
    return False
