"""
Simple TSV event logger for experiments.
"""
import csv
import datetime as dt
from pathlib import Path
import re
from typing import Any, Optional
import time


def sanitize_filename_component(value: str) -> str:
    text = str(value).strip()
    if not text:
        return "unnamed"
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("._-") or "unnamed"


def build_run_log_filename(
    config_name: str,
    log_kind: str,
    when: Optional[dt.datetime] = None,
    in_progress: bool = False,
) -> str:
    timestamp = when or dt.datetime.now()
    safe_config = sanitize_filename_component(config_name)
    safe_kind = sanitize_filename_component(log_kind)
    base = f"{safe_config}_{safe_kind}_{timestamp:%Y%m%d}_{timestamp:%H_%M_%S}"
    if in_progress:
        base = f"{base}_in_progress"
    return f"{base}.tsv"


def _ensure_tsv_path(path: Path) -> Path:
    if path.suffix.lower() == ".tsv":
        return path
    return path.with_suffix(".tsv")


def _uniquify_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    idx = 1
    while True:
        candidate = parent / f"{stem}_{idx}{suffix}"
        if not candidate.exists():
            return candidate
        idx += 1


def finalize_output_file(path: Path, final_filename: str) -> Path:
    current_path = Path(path)
    target_path = _ensure_tsv_path(current_path.parent / final_filename)
    target_path = _uniquify_path(target_path)
    if current_path.resolve() == target_path.resolve():
        return current_path
    current_path.replace(target_path)
    return target_path


class EventLogger:
    """
    TSV logger that writes rows with high-resolution timestamps and event info.
    """

    def __init__(self, out_dir: str, filename: str = "image_sequence_log.tsv"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        # ensure .tsv extension
        if not filename.endswith(".tsv"):
            filename = Path(filename).stem + ".tsv"
        self.path = self.out_dir / filename
        self._file = open(self.path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file, delimiter="\t")
        self._writer.writerow(
            [
                "row_idx",
                "event",
                "image_name",
                "requested_duration_s",
                "flip_time_psychopy_s",
                "flip_time_perf_s",
                "end_time_perf_s",
                "notes",
            ]
        )
        self._idx = 0
        # write a metadata file
        meta = self.out_dir / "meta.txt"
        with open(meta, "w", encoding="utf-8") as mf:
            mf.write(f"created: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    def log(
        self,
        event: str,
        image_name: str = "",
        requested_duration_s: float = None,
        flip_time_psychopy_s: float = None,
        flip_time_perf_s: float = None,
        end_time_perf_s: float = None,
        notes: str = "",
    ):
        self._idx += 1
        self._writer.writerow(
            [
                self._idx,
                event,
                image_name,
                f"{requested_duration_s:.6f}" if requested_duration_s is not None else "",
                f"{flip_time_psychopy_s:.6f}" if flip_time_psychopy_s is not None else "",
                f"{flip_time_perf_s:.9f}" if flip_time_perf_s is not None else "",
                f"{end_time_perf_s:.9f}" if end_time_perf_s is not None else "",
                notes,
            ]
        )
        self._file.flush()

    def close(self):
        self._file.close()

    def finalize(self, final_filename: str) -> Path:
        self.close()
        self.path = finalize_output_file(self.path, final_filename)
        return self.path


class MessageLogger:
    """Simple TSV logger for textual messages (warnings, debug, info).

    Columns: row_idx, time_iso, level, message
    """

    def __init__(self, out_dir: str, filename: str = "message_log.tsv"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        if not filename.endswith(".tsv"):
            filename = Path(filename).stem + ".tsv"
        self.path = self.out_dir / filename
        self._file = open(self.path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file, delimiter="\t")
        self._writer.writerow(["row_idx", "time_iso", "level", "message"])
        self._idx = 0

    def log(self, level: str, message: str):
        self._idx += 1
        timestr = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
        self._writer.writerow([self._idx, timestr, level.upper(), message])
        self._file.flush()

    def close(self):
        self._file.close()

    def finalize(self, final_filename: str) -> Path:
        self.close()
        self.path = finalize_output_file(self.path, final_filename)
        return self.path
