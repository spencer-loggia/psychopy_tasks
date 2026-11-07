"""Portable runtime numeric settings with preview IPC and timestamped audit logs.

The preview is the sole request writer; the task is the sole application owner.
Task-side application attempts the shared lock once, without waiting. A busy
preview defers edits to the next safe boundary. Ordered confirmations travel on
an independent queue, drained only during ordinary log flushes and shutdown;
scene mailbox coalescing cannot discard edits or their original timestamps.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import json
import math
import multiprocessing as mp
import queue
import time


@dataclass(frozen=True)
class SettingSpec:
    name: str
    label: str
    title: str
    units: str
    minimum: float = 0.0
    allow_minimum: bool = True
    maximum: float | None = None  # exclusive

    def validate(self, value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"Enter a number ({self.units})") from None
        if not math.isfinite(value) or value < self.minimum or (
            value == self.minimum and not self.allow_minimum
        ):
            comparison = ">=" if self.allow_minimum else ">"
            raise ValueError(f"Must be finite and {comparison} {self.minimum:g}")
        if self.maximum is not None and value >= self.maximum:
            raise ValueError(f"Must be less than {self.maximum:g} {self.units}")
        return value


class LiveSettings:
    def __init__(self, specs, initial, *, boundary="trial", less_than=()):
        self.specs = tuple(specs)
        self.boundary = boundary
        self.less_than = tuple(less_than)
        self._indices = {spec.name: i for i, spec in enumerate(self.specs)}
        if len(self._indices) != len(self.specs):
            raise ValueError("Live setting names must be unique")
        values = {spec.name: spec.validate(initial[spec.name]) for spec in self.specs}
        self._validate_relations(values)
        ctx = mp.get_context("spawn")
        self._lock = ctx.Lock()
        self._requested = ctx.Array("d", list(values.values()), lock=False)
        self._applied = ctx.Array("d", list(values.values()), lock=False)
        self._request_times = ctx.Array("q", [0] * len(values), lock=False)
        self._request_ids = ctx.Array("q", [0] * len(values), lock=False)
        self._confirmation_count = ctx.Value("q", 0, lock=False)
        self._confirmations = ctx.Queue()
        self._logged_confirmations = 0
        self._received_confirmations = deque()
        self._current = values  # Task-owned snapshot; reading it never uses IPC.
        self._closed = False

    def _validate_relations(self, values):
        for smaller, larger in self.less_than:
            if values[smaller] >= values[larger]:
                raise ValueError(f"{smaller} must be shorter than {larger}")

    @property
    def values(self):
        """Task-owned applied snapshot, suitable for a behavior row."""
        return dict(self._current)

    def snapshot(self):
        """Preview-side coherent display values; never called in trial timing."""
        with self._lock:
            return {spec.name: (self._applied[i], self._requested[i])
                    for i, spec in enumerate(self.specs)}

    def request(self, name, value):
        """Validate and timestamp a confirmed edit in the preview process."""
        index = self._indices[name]
        value = self.specs[index].validate(value)
        with self._lock:
            desired = {spec.name: self._requested[i] for i, spec in enumerate(self.specs)}
            old = desired[name]
            desired[name] = value
            self._validate_relations(desired)
            if value == old:
                return
            requested_ns = time.perf_counter_ns()
            request_id = self._confirmation_count.value + 1
            self._requested[index] = value
            self._request_times[index] = requested_ns
            self._request_ids[index] = request_id
            # Unbounded Queue.put_nowait only hands this tiny record to the
            # preview's feeder thread; it never waits for the task or disk.
            self._confirmations.put_nowait((request_id, name, old, value, requested_ns))
            self._confirmation_count.value = request_id

    def apply_pending(self, logger=None, *, on_apply=None):
        """Apply a coherent group at a task-chosen safe boundary, or defer.

        on_apply(values, timestamp_perf_ns) updates task policies before the
        applied UI snapshot is published. It must not perform I/O or waits;
        returning False defers the group without publishing/logging a change.
        Logging is buffered by task MessageLogger(auto_flush=False).
        """
        if not self._lock.acquire(False):
            return self.values
        changes = []
        try:
            desired = {spec.name: self._requested[i] for i, spec in enumerate(self.specs)}
            if desired == self._current:
                return self.values
            applied_ns = time.perf_counter_ns()
            if on_apply is not None:
                if on_apply(desired, applied_ns) is False:
                    return self.values
            for i, spec in enumerate(self.specs):
                old, new = self._current[spec.name], desired[spec.name]
                if old != new:
                    changes.append(dict(name=spec.name, old_value=old, new_value=new,
                                        units=spec.units, request_id=self._request_ids[i],
                                        requested_perf_ns=self._request_times[i]))
                self._applied[i] = new
            self._current = desired
        finally:
            self._lock.release()
        if logger is not None:
            for change in changes:
                self._log(logger, "applied", applied_ns, **change)
        return self.values

    def log_initial(self, logger):
        now_ns = time.perf_counter_ns()
        for spec in self.specs:
            self._log(logger, "initial", now_ns, name=spec.name,
                      new_value=self._current[spec.name], units=spec.units)

    @staticmethod
    def _log(logger, phase, timestamp_ns, **fields):
        logger.log("INFO", "live_setting " + json.dumps(
            dict(phase=phase, **fields), sort_keys=True, separators=(",", ":"),
        ), timestamp_perf_ns=timestamp_ns)

    def log_requests(self, logger, *, finish=False):
        """Drain during log flushes. Only shutdown may wait for the feeder."""
        if self._closed:
            return
        target = self._confirmation_count.value
        while self._logged_confirmations < target:
            try:
                if self._received_confirmations:
                    record = self._received_confirmations.popleft()
                else:
                    record = self._confirmations.get(timeout=1.0) if finish else self._confirmations.get_nowait()
            except queue.Empty:
                if finish:
                    raise RuntimeError("Live setting confirmation log did not drain at shutdown")
                break
            request_id, name, old, value, timestamp_ns = record
            self._log(logger, "confirmed", timestamp_ns, name=name, old_value=old,
                      new_value=value, units=self.specs[self._indices[name]].units, request_id=request_id)
            self._logged_confirmations = request_id

    def collect_requests(self):
        """Drain into parent memory while joining the preview at shutdown.

        A Queue producer joins its feeder on exit. Draining here prevents a
        full pipe from making a healthy preview look hung and being terminated.
        This path does not format messages or write files.
        """
        if self._closed:
            return
        while True:
            try:
                self._received_confirmations.append(self._confirmations.get_nowait())
            except queue.Empty:
                return

    def close(self, logger):
        if self._closed:
            return
        try:
            self.log_requests(logger, finish=True)
        finally:
            self._confirmations.close()
            self._confirmations.join_thread()  # Shutdown only, after the preview exits.
            self._closed = True


def reward_settings(pulse_s, *, allow_zero=True, maximum_s=None, reward_interval=None,
                    fixation_radius=None, fixation_window_radius=None, boundary="trial",
                    interval_label="Reward interval"):
    """Common reward specs; other tasks can construct LiveSettings directly."""
    specs = [SettingSpec("pump_pulse_time", "Pump", "Pump pulse duration (seconds)", "s",
                         allow_minimum=allow_zero,
                         maximum=maximum_s if reward_interval is None else None)]
    initial = {"pump_pulse_time": pulse_s}
    relations = []
    if reward_interval is not None:
        specs.append(SettingSpec("reward_interval", interval_label, f"{interval_label} (seconds)", "s",
                                 allow_minimum=False))
        initial["reward_interval"] = reward_interval
        relations.append(("pump_pulse_time", "reward_interval"))
    if fixation_radius is not None:
        specs.append(SettingSpec("fixation_radius", "Fix radius", "Fixation radius (screen fraction)",
                                 "frac", allow_minimum=False))
        initial["fixation_radius"] = fixation_radius
    if fixation_window_radius is not None:
        specs.append(SettingSpec("fixation_window_radius", "Fix radius", "Fixation window radius (pixels)",
                                 "px", allow_minimum=False))
        initial["fixation_window_radius"] = fixation_window_radius
    return LiveSettings(specs, initial, boundary=boundary, less_than=relations)
