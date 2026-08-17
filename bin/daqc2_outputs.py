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
    requested_perf_s: float
    actual_perf_s: float


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
    """

    def __init__(
        self,
        outputs: DAQC2DigitalOutputs,
        *,
        bit: int,
        interval_s: float,
        pulse_duration_s: float,
        clock: Callable[[], float] = time.perf_counter,
    ):
        self.outputs = outputs
        self.bit = outputs.validate_bit(bit)
        self.interval_s = float(interval_s)
        self.pulse_duration_s = float(pulse_duration_s)
        if not math.isfinite(self.interval_s) or self.interval_s <= 0.0:
            raise ValueError("pump_interval must be a positive finite value")
        if not math.isfinite(self.pulse_duration_s) or self.pulse_duration_s <= 0.0:
            raise ValueError("pump_pulse_time_seconds must be a positive finite value")
        if self.pulse_duration_s >= self.interval_s:
            raise ValueError(
                "pump_pulse_time_seconds must be shorter than pump_interval"
            )
        self._clock = clock
        self._stop_event = threading.Event()
        self._failed_event = threading.Event()
        self._edges: queue.SimpleQueue[DOUTPulseEdge] = queue.SimpleQueue()
        self._thread: Optional[threading.Thread] = None
        self._failure: Optional[BaseException] = None
        self._anchor_perf_s: Optional[float] = None

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
        requested_perf_s: float,
    ) -> None:
        self.outputs.write(self.bit, active)
        self._edges.put(
            DOUTPulseEdge(
                active=bool(active),
                pulse_index=int(pulse_index),
                requested_perf_s=float(requested_perf_s),
                actual_perf_s=self._clock(),
            )
        )

    def _run(self) -> None:
        if self._anchor_perf_s is None:
            return
        next_on_perf_s = self._anchor_perf_s + self.interval_s
        pulse_index = 0
        output_active = False
        try:
            while True:
                remaining_s = next_on_perf_s - self._clock()
                if self._stop_event.wait(max(0.0, remaining_s)):
                    break

                now_perf_s = self._clock()
                if now_perf_s >= next_on_perf_s + self.interval_s:
                    skipped_intervals = int(
                        (now_perf_s - next_on_perf_s) // self.interval_s
                    )
                    next_on_perf_s += skipped_intervals * self.interval_s
                    pulse_index += skipped_intervals

                pulse_index += 1
                requested_on_perf_s = next_on_perf_s
                self._record_edge(
                    active=True,
                    pulse_index=pulse_index,
                    requested_perf_s=requested_on_perf_s,
                )
                output_active = True

                requested_off_perf_s = requested_on_perf_s + self.pulse_duration_s
                if self._stop_event.wait(self.pulse_duration_s):
                    break
                self._record_edge(
                    active=False,
                    pulse_index=pulse_index,
                    requested_perf_s=requested_off_perf_s,
                )
                output_active = False
                next_on_perf_s = requested_on_perf_s + self.interval_s
                now_perf_s = self._clock()
                if next_on_perf_s <= now_perf_s:
                    expired = int(
                        math.floor((now_perf_s - next_on_perf_s) / self.interval_s)
                    ) + 1
                    next_on_perf_s += expired * self.interval_s
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
                        requested_perf_s=self._clock(),
                    )
                except BaseException as exc:
                    if self._failure is None:
                        self._failure = exc
                        self._failed_event.set()
