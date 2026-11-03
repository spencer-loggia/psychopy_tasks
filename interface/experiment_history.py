"""Permanent, local-only experiment numbers and summaries (Linux/macOS)."""

from __future__ import annotations

import csv
import datetime as dt
import fcntl
import json
import math
import os
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

from bin.logger import sanitize_filename_component


HISTORY_FILENAME = "experiment_history.tsv"
HISTORY_SYNC_EXCLUDES = (f"/{HISTORY_FILENAME}", "/.experiment_history.*")
HISTORY_FIELDS = (
    "experiment_name", "subject", "subject_code", "date", "experiment_number",
    "start_time", "last_update_time", "end_time", "block_count",
    "completed_block_count", "runtime_seconds", "status",
)
_EXPERIMENT_NAME = re.compile(r"exp_(.+)_(\d{8})_(\d+)")


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _identity(name: str) -> tuple[str, str, int] | None:
    match = _EXPERIMENT_NAME.fullmatch(name)
    if match is None:
        return None
    code, date, number = match.groups()
    dt.datetime.strptime(date, "%Y%m%d")
    if int(number) < 1:
        raise ValueError(f"Invalid experiment number in {name}")
    return code, date, int(number)


def _legacy_summary(directory: Path) -> dict[str, str]:
    """Recover available metadata without requiring intact legacy run files."""
    code, date, number = _identity(directory.name)
    row = dict.fromkeys(HISTORY_FIELDS, "")
    row.update(
        experiment_name=directory.name, subject_code=code, date=date,
        experiment_number=str(number), status="imported",
    )
    try:
        state = json.loads((directory / "state.json").read_text(encoding="utf-8"))
        if isinstance(state, dict):
            row["subject"] = str(state.get("subject") or "")
    except (OSError, ValueError):
        pass
    origin = None
    try:
        clock = json.loads((directory / "clock.json").read_text(encoding="utf-8"))
        origin = dt.datetime.fromisoformat(clock["origin_datetime"])
        row["start_time"] = origin.isoformat(timespec="microseconds")
        row["last_update_time"] = row["start_time"]
        row["runtime_seconds"] = "0.000000"
    except (OSError, ValueError, KeyError, TypeError):
        pass
    try:
        with (directory / "blocks.tsv").open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t", strict=True)
            if not {"subject", "start_time", "end_time"}.issubset(reader.fieldnames or ()):
                return row
            blocks = list(reader)
        row["block_count"] = str(len(blocks))
        row["completed_block_count"] = str(sum(bool(block.get("end_time")) for block in blocks))
        for block in blocks:
            row["subject"] = row["subject"] or block.get("subject") or ""
            if origin is None:
                continue
            for field in ("start_time", "end_time"):
                try:
                    when = dt.datetime.fromisoformat(block[field])
                    elapsed = (when - origin).total_seconds()
                except (ValueError, TypeError):
                    continue
                if elapsed >= float(row["runtime_seconds"]):
                    row["runtime_seconds"] = f"{elapsed:.6f}"
                    row["last_update_time"] = when.isoformat(timespec="microseconds")
    except (OSError, ValueError, csv.Error):
        pass
    # A block end does not establish that the experiment itself ended cleanly.
    return row


class ExperimentHistory:
    """A readable TSV, serialized by a persistent lock and replaced durably.

    Reservations are committed before experiment directories are created. An
    interruption may leave a consumed number, but cannot release it for reuse.
    """

    def __init__(self, logs_dir: Path):
        self.logs_dir = Path(logs_dir)
        self.path = self.logs_dir / HISTORY_FILENAME

    def _read(self) -> dict[str, dict[str, str]]:
        try:
            handle = self.path.open(encoding="utf-8", newline="")
        except FileNotFoundError:
            return {}
        try:
            with handle:
                reader = csv.DictReader(handle, delimiter="\t", strict=True)
                if reader.fieldnames != list(HISTORY_FIELDS):
                    raise ValueError("unexpected header")
                rows = {}
                for row in reader:
                    if None in row or any(value is None for value in row.values()):
                        raise ValueError("incomplete row")
                    name = row["experiment_name"]
                    identity = _identity(name)
                    if identity is None or identity != (
                        row["subject_code"], row["date"], int(row["experiment_number"]),
                    ) or name in rows:
                        raise ValueError(f"invalid or duplicate experiment: {name}")
                    if row["status"] not in {"open", "ended", "imported"}:
                        raise ValueError(f"invalid status for {name}")
                    for field in ("block_count", "completed_block_count"):
                        if row[field] and int(row[field]) < 0:
                            raise ValueError(f"invalid {field} for {name}")
                    if row["runtime_seconds"]:
                        runtime = float(row["runtime_seconds"])
                        if not math.isfinite(runtime) or runtime < 0:
                            raise ValueError(f"invalid runtime for {name}")
                    for field in ("start_time", "last_update_time", "end_time"):
                        if row[field]:
                            dt.datetime.fromisoformat(row[field])
                    rows[name] = row
            return rows
        except (ValueError, csv.Error) as exc:
            raise OSError(f"Cannot read local experiment history {self.path}: {exc}") from exc

    @contextmanager
    def _transaction(self):
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        # Lock a separate, permanent inode: replacing the TSV must not release
        # serialization between two launcher processes.
        with (self.logs_dir / ".experiment_history.lock").open("a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if not self.path.exists():
                # Another process may have just made logs/. Persist its parent
                # entry too, before committing the first reservation inside it.
                _sync_directory(self.logs_dir.parent)
            rows = self._read()
            original = {name: row.copy() for name, row in rows.items()}
            yield rows
            if rows != original:
                temporary = self.logs_dir / ".experiment_history.tsv.tmp"
                with temporary.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS, delimiter="\t", lineterminator="\n")
                    writer.writeheader()
                    writer.writerows(rows.values())
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, self.path)
                _sync_directory(self.logs_dir)

    @staticmethod
    def _import(rows: dict[str, dict[str, str]], directories: Iterable[Path]) -> None:
        for directory in directories:
            if directory.name in rows or not directory.is_dir() or directory.is_symlink():
                continue
            try:
                identity = _identity(directory.name)
            except ValueError:
                continue
            if identity is not None:
                rows[directory.name] = _legacy_summary(directory)

    def import_existing(self, directories: Iterable[Path]) -> None:
        """Persist legacy identities before cache cleanup can remove them."""
        with self._transaction() as rows:
            self._import(rows, directories)

    def reserve(self, *, subject: str, subject_code: str, started_at: dt.datetime) -> tuple[Path, int]:
        code = sanitize_filename_component(subject_code)
        date = started_at.strftime("%Y%m%d")
        with self._transaction() as rows:
            self._import(rows, sorted(self.logs_dir.iterdir()))
            number = 1 + max((
                int(row["experiment_number"]) for row in rows.values()
                if row["subject_code"] == code and row["date"] == date
            ), default=0)
            name = f"exp_{code}_{date}_{number:03d}"
            stamp = started_at.isoformat(timespec="microseconds")
            rows[name] = dict(
                experiment_name=name, subject=subject, subject_code=code, date=date,
                experiment_number=str(number), start_time=stamp, last_update_time=stamp,
                end_time="", block_count="0", completed_block_count="0",
                runtime_seconds="0.000000", status="open",
            )
        return self.logs_dir / name, number

    def update(self, name: str, *, block_count: int, completed_block_count: int,
               elapsed_ns: int, ended: bool = False) -> None:
        with self._transaction() as rows:
            if name not in rows:
                raise OSError(f"Experiment {name} is missing from {self.path}")
            row = rows[name]
            if row["status"] == "ended":
                return
            origin = dt.datetime.fromisoformat(row["start_time"])
            when = origin + dt.timedelta(microseconds=elapsed_ns // 1000)
            row.update(
                last_update_time=when.isoformat(timespec="microseconds"),
                block_count=str(block_count), completed_block_count=str(completed_block_count),
                runtime_seconds=f"{elapsed_ns / 1_000_000_000:.6f}",
            )
            if ended:
                row.update(status="ended", end_time=row["last_update_time"])
