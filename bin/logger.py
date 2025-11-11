"""
Simple TSV event logger for experiments.
"""
import csv
from pathlib import Path
from typing import Any
import time


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
