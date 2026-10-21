"""One performance-counter origin shared by all blocks of an experiment."""

from __future__ import annotations

import datetime as dt
import os
import time


EXPERIMENT_ORIGIN_ENV = "NEURO_EXPERIMENT_ORIGIN_PERF_NS"


class TaskClock:
    def __init__(self, *, origin_perf_ns=None, started_at=None):
        if origin_perf_ns is None:
            inherited = os.environ.get(EXPERIMENT_ORIGIN_ENV)
            origin_perf_ns = int(inherited) if inherited is not None else time.perf_counter_ns()
        self.origin_perf_ns = int(origin_perf_ns)
        self.started_at = started_at or dt.datetime.now()

    @property
    def start_perf_s(self):
        """Same origin in seconds for presentation APIs that use float seconds."""
        return self.origin_perf_ns / 1_000_000_000

    def task_time_ns(self, timestamp_perf_s=None, *, timestamp_perf_ns=None):
        """Convert an existing capture, or capture now, to experiment nanoseconds.

        Float inputs adapt presentation/hardware APIs using perf_counter seconds;
        native nanosecond captures never pass through a float.
        """
        if timestamp_perf_s is not None and timestamp_perf_ns is not None:
            raise ValueError("Supply only one performance-counter timestamp")
        if timestamp_perf_ns is None:
            timestamp_perf_ns = (
                time.perf_counter_ns() if timestamp_perf_s is None
                else round(timestamp_perf_s * 1_000_000_000)
            )
        return int(timestamp_perf_ns) - self.origin_perf_ns
