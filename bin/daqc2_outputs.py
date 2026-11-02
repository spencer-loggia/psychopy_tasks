"""Pi-Plates DAQC2 digital-output helpers."""
from __future__ import annotations

import importlib
import math
import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass(frozen=True)
class DOUTPulseEdge:
    """One requested and observed edge from a periodic DOUT pulse."""

    active: bool
    pulse_index: int
    requested_perf_ns: int
    actual_perf_ns: int
    pulse_duration_s: Optional[float] = None

    @property
    def requested_perf_s(self):
        return self.requested_perf_ns / 1_000_000_000

    @property
    def actual_perf_s(self):
        return self.actual_perf_ns / 1_000_000_000


@dataclass(frozen=True)
class _PulseTiming:
    interval_s: float
    pulse_duration_s: float
    interval_revision: int = 0
    interval_changed_perf_s: Optional[float] = None


class DAQC2DigitalOutputs:
    """Small adapter around DAQC2plate open-drain DOUT writes."""

    def __init__(
        self,
        *,
        address: int = 0,
        module_name: str = "piplates.DAQC2plate",
        enabled: bool = True,
    ):
        self.address = self.validate_address(address)
        self.module_name = str(module_name)
        self.enabled = bool(enabled)
        self._module: Optional[object] = None
        self._set_dout_bit: Optional[Callable[[int, int], object]] = None
        self._clear_dout_bit: Optional[Callable[[int, int], object]] = None

    def open(self) -> None:
        if not self.enabled:
            return
        module = importlib.import_module(self.module_name)
        missing = [
            name
            for name in ("setDOUTbit", "clrDOUTbit")
            if not hasattr(module, name)
        ]
        if missing:
            raise RuntimeError(
                f"{self.module_name} does not expose required DAQC2 DOUT functions: {', '.join(missing)}"
            )
        self._module = module
        self._set_dout_bit = module.setDOUTbit
        self._clear_dout_bit = module.clrDOUTbit

    def write(self, bit: int, active: bool) -> None:
        """Set a DOUT bit logically on/off.

        ``active=True`` calls ``setDOUTbit`` and turns on the open-drain
        transistor. With a pull-up or high-side load, that pulls the DOUT
        terminal near 0 V. ``active=False`` calls ``clrDOUTbit`` and turns the
        transistor off, allowing the terminal to return high.
        """
        bit = self.validate_bit(bit)
        if not self.enabled:
            return
        if self._module is None:
            self.open()
        if self._set_dout_bit is None or self._clear_dout_bit is None:
            return
        if active:
            self._set_dout_bit(self.address, bit)
        else:
            self._clear_dout_bit(self.address, bit)

    def bind_bit(self, bit: int) -> tuple[Callable[[], None], Callable[[], None]]:
        """Return prevalidated on/off callables for a DOUT bit."""
        bit = self.validate_bit(bit)

        def set_bit() -> None:
            if not self.enabled:
                return
            if self._set_dout_bit is None:
                self.open()
            if self._set_dout_bit is not None:
                self._set_dout_bit(self.address, bit)

        def clear_bit() -> None:
            if not self.enabled:
                return
            if self._clear_dout_bit is None:
                self.open()
            if self._clear_dout_bit is not None:
                self._clear_dout_bit(self.address, bit)

        return set_bit, clear_bit

    @staticmethod
    def validate_address(value: int) -> int:
        address = int(value)
        if address < 0 or address > 7:
            raise ValueError("DAQC2 address must be in the range 0 through 7")
        return address

    @staticmethod
    def validate_bit(value: int) -> int:
        bit = int(value)
        if bit < 0 or bit > 7:
            raise ValueError("DAQC2 DOUT bit must be in the range 0 through 7")
        return bit


class PeriodicDOUTPulseController:
    """Drive fixed-width DOUT pulses on absolute, non-accumulating deadlines.

    Hardware writes run on a dedicated thread. If that thread is delayed, it
    skips expired intervals instead of emitting catch-up pulses. Edge records
    are queued for the task's main thread so log writers remain single-threaded.
    An optional fixation reward policy continuously checks rolling gaze windows
    once a full interval can exist, resetting their history at each reward onset.
    Timing updates publish one immutable snapshot without waiting for the worker
    or performing hardware I/O on the presentation thread.
    """

    _TIMING_POLL_SECONDS = 0.005

    def __init__(
        self,
        outputs: DAQC2DigitalOutputs,
        *,
        bit: int,
        interval_s: float,
        pulse_duration_s: float,
        clock: Callable[[], float] = time.perf_counter,
        reward_policy=None,
    ):
        self.outputs = outputs
        self.bit = outputs.validate_bit(bit)
        interval_s, pulse_duration_s = self._validate_timing(
            interval_s, pulse_duration_s,
        )
        self._timing = _PulseTiming(interval_s, pulse_duration_s)
        self._clock = clock
        self._reward_policy = reward_policy
        self._stop_event = threading.Event()
        self._failed_event = threading.Event()
        self._edges: queue.SimpleQueue[DOUTPulseEdge] = queue.SimpleQueue()
        self._thread: Optional[threading.Thread] = None
        self._failure: Optional[BaseException] = None
        self._anchor_perf_s: Optional[float] = None

    @staticmethod
    def _validate_timing(interval_s: float, pulse_duration_s: float) -> tuple[float, float]:
        interval_s = float(interval_s)
        pulse_duration_s = float(pulse_duration_s)
        if not math.isfinite(interval_s) or interval_s <= 0.0:
            raise ValueError("reward_interval must be a positive finite value")
        if not math.isfinite(pulse_duration_s) or pulse_duration_s <= 0.0:
            raise ValueError("pump_pulse_time_seconds must be a positive finite value")
        if pulse_duration_s >= interval_s:
            raise ValueError(
                "pump_pulse_time_seconds must be shorter than reward_interval"
            )
        return interval_s, pulse_duration_s

    @property
    def interval_s(self) -> float:
        return self._timing.interval_s

    @property
    def pulse_duration_s(self) -> float:
        return self._timing.pulse_duration_s

    @property
    def failed(self) -> bool:
        return self._failed_event.is_set()

    @property
    def failure(self) -> Optional[BaseException]:
        return self._failure

    def start(self, *, anchor_perf_s: Optional[float] = None) -> None:
        if self._thread is not None:
            raise RuntimeError("Periodic DOUT pulse controller is already started")
        self.outputs.open()
        self.outputs.write(self.bit, False)
        self._anchor_perf_s = (
            self._clock() if anchor_perf_s is None else float(anchor_perf_s)
        )
        self._thread = threading.Thread(
            target=self._run,
            name=f"daqc2-dout{self.bit}-periodic-pulse",
            daemon=True,
        )
        self._thread.start()

    def set_pulse_duration(self, duration_s: float) -> None:
        """Update subsequent pulses; an active pulse keeps its captured width."""
        self.set_timing(self.interval_s, duration_s)

    def set_timing(self, interval_s: float, pulse_duration_s: float) -> None:
        """Publish timing for subsequent pulses without blocking presentation.

        Call from the task's settings thread. An interval change requires a full
        new interval from this call; a width-only change keeps the schedule.
        The worker captures each pulse's width before raising the output, so an
        in-flight pulse always completes with its original width. When changing
        a separate fixation policy, publish this timing before updating that
        policy so the new interval fences out its previous eligibility.
        """
        interval_s, pulse_duration_s = self._validate_timing(
            interval_s, pulse_duration_s,
        )
        previous = self._timing
        interval_changed = interval_s != previous.interval_s
        self._timing = _PulseTiming(
            interval_s,
            pulse_duration_s,
            previous.interval_revision + int(interval_changed),
            self._clock() if interval_changed else previous.interval_changed_perf_s,
        )

    def drain_edges(self) -> list[DOUTPulseEdge]:
        edges: list[DOUTPulseEdge] = []
        while True:
            try:
                edges.append(self._edges.get_nowait())
            except queue.Empty:
                return edges

    def stop(self, *, join_timeout_s: float = 2.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(max(0.0, float(join_timeout_s)))
            if self._thread.is_alive():
                raise RuntimeError("Periodic DOUT pulse thread did not stop")
        # A final clear is intentional even if the worker already cleared it.
        self.outputs.write(self.bit, False)

    def _record_edge(
        self,
        *,
        active: bool,
        pulse_index: int,
        pulse_duration_s: Optional[float] = None,
    ) -> None:
        # The schedule target is not the observed hardware request time.
        requested_ns = time.perf_counter_ns()
        self.outputs.write(self.bit, active)
        actual_ns = time.perf_counter_ns()
        self._edges.put(
            DOUTPulseEdge(
                active=bool(active),
                pulse_index=int(pulse_index),
                requested_perf_ns=requested_ns,
                actual_perf_ns=actual_ns,
                pulse_duration_s=pulse_duration_s,
            )
        )

    def _run(self) -> None:
        if self._anchor_perf_s is None:
            return
        timing = self._timing
        next_on_perf_s = self._anchor_perf_s + timing.interval_s
        pulse_index = 0
        output_active = False
        try:
            while True:
                latest_timing = self._timing
                if latest_timing.interval_revision != timing.interval_revision:
                    next_on_perf_s = (
                        latest_timing.interval_changed_perf_s + latest_timing.interval_s
                    )
                timing = latest_timing
                remaining_s = next_on_perf_s - self._clock()
                if self._stop_event.wait(min(
                    max(0.0, remaining_s), self._TIMING_POLL_SECONDS,
                )):
                    break

                now_perf_s = self._clock()
                if self._timing is not timing or now_perf_s < next_on_perf_s:
                    continue
                if self._reward_policy is not None:
                    if not self._reward_policy.status(now_s=now_perf_s).due:
                        # Keep checking the rolling windows without waiting for
                        # another scheduled interval or blocking presentation.
                        next_on_perf_s = now_perf_s + self._TIMING_POLL_SECONDS
                        continue
                elif now_perf_s >= next_on_perf_s + timing.interval_s:
                    skipped_intervals = int(
                        (now_perf_s - next_on_perf_s) // timing.interval_s
                    )
                    next_on_perf_s += skipped_intervals * timing.interval_s
                    pulse_index += skipped_intervals

                # A policy check may run concurrently with a settings update.
                if self._timing is not timing:
                    continue
                pulse_index += 1
                requested_on_perf_s = next_on_perf_s
                pulse_duration_s = timing.pulse_duration_s
                self._record_edge(
                    active=True,
                    pulse_index=pulse_index,
                    pulse_duration_s=pulse_duration_s,
                )
                output_active = True
                if self._reward_policy is not None:
                    requested_on_perf_s = self._clock()
                    self._reward_policy.reward_delivered(now_s=requested_on_perf_s)

                if self._stop_event.wait(pulse_duration_s):
                    break
                self._record_edge(
                    active=False,
                    pulse_index=pulse_index,
                )
                output_active = False
                next_on_perf_s = requested_on_perf_s + timing.interval_s
                now_perf_s = self._clock()
                if self._reward_policy is None and next_on_perf_s <= now_perf_s:
                    expired = int(
                        math.floor((now_perf_s - next_on_perf_s) / timing.interval_s)
                    ) + 1
                    next_on_perf_s += expired * timing.interval_s
                    pulse_index += expired
        except BaseException as exc:
            self._failure = exc
            self._failed_event.set()
        finally:
            if output_active:
                try:
                    self._record_edge(
                        active=False,
                        pulse_index=pulse_index,
                    )
                except BaseException as exc:
                    if self._failure is None:
                        self._failure = exc
                        self._failed_event.set()
