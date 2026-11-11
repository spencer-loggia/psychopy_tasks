"""Independent, cancellable five-second hardware pulses for the launcher.

Drivers are imported only after a pin is requested. No experiment, display, or
tracker setup belongs here. Hardware calls never run on Tk's event thread.
"""
from __future__ import annotations

import importlib
import math
import queue
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass


PULSE_SECONDS = 5.0
HARDWARE_ERRORS = (Exception, SystemExit)  # Some Pi drivers exit on import failure.


def _checked(result):
    if isinstance(result, (int, float)) and result < 0:
        raise RuntimeError(f"Driver returned error {result}")
    return result


@dataclass(frozen=True)
class OutputPin:
    backend: str
    pin: int
    device: int = 0  # GPIO chip number or DAQC2 board address

    def __post_init__(self):
        if self.backend not in {"gpio", "daqc2"}:
            raise ValueError("Unknown output backend")
        if not 0 <= self.pin <= (27 if self.backend == "gpio" else 7):
            raise ValueError("GPIO must be BCM 0–27; DAQC2 DOUT must be 0–7")
        if self.device < 0 or (self.backend == "daqc2" and self.device > 7):
            raise ValueError("Invalid chip number or board address")

    @property
    def label(self):
        if self.backend == "gpio":
            return f"GPIO {self.pin} (BCM), chip {self.device}"
        return f"DAQC2 {self.device} / DOUT{self.pin}"


class GPIOOutput:
    def __init__(self, target):
        self.target = target
        self.module = None
        self.handle = None
        self.claimed = False

    def open(self):
        self.module = importlib.import_module("lgpio")
        self.handle = _checked(self.module.gpiochip_open(self.target.device))
        _checked(self.module.gpio_claim_output(self.handle, self.target.pin, 0))
        self.claimed = True

    def write(self, active):
        if self.claimed:
            _checked(self.module.gpio_write(self.handle, self.target.pin, int(active)))

    def close(self):
        if self.handle is not None:
            _checked(self.module.gpiochip_close(self.handle))
            self.handle = None


class DAQC2Bus:
    """Serialize SPI transactions, never the five-second pulse wait."""

    def __init__(self, module_name="piplates.DAQC2plate"):
        self.module_name = module_name
        self.module = None
        self.lock = threading.Lock()

    @contextmanager
    def _access(self):
        if not self.lock.acquire(timeout=1.0):
            raise TimeoutError("DAQC2 driver is busy; other GPIO outputs remain available")
        try:
            if self.module is None:
                self.module = importlib.import_module(self.module_name)
            yield self.module
        finally:
            self.lock.release()

    @staticmethod
    def _check_reply(module):
        # Pi-Plates writes return None even when the board fails to ACK. New
        # drivers put this flag in CMD; older ones keep it on the plate module.
        if getattr(getattr(module, "CMD", module), "DataGood", None) is False:
            raise OSError("DAQC2 did not acknowledge the command")

    def call(self, name, *args):
        with self._access() as module:
            result = _checked(getattr(module, name)(*args))
            self._check_reply(module)
            return result

    def prepare(self, address):
        with self._access() as module:
            get_address = getattr(module, "getADDR", None)
            if callable(get_address):
                reply = get_address(address)
                self._check_reply(module)
                if reply != address:
                    raise RuntimeError(f"No DAQC2 response at address {address}")
                # VerifyADDR uses an import-time presence cache. Refresh only
                # this confirmed board so reconnecting works without a global
                # quietPoll()/reload, calibration reads, or resetting outputs.
                present = getattr(module, "daqc2sPresent", None)
                if isinstance(present, list) and len(present) > address:
                    present[address] = 1
            else:
                # Compatibility with alternate drivers exposing only the ADC
                # presence check used by the existing system diagnostic.
                supply = float(module.getADC(address, 8))
                self._check_reply(module)
                if not math.isfinite(supply) or supply <= 0:
                    raise RuntimeError(f"No DAQC2 response at address {address}")


class DAQC2Output:
    def __init__(self, target, bus):
        self.target = target
        self.bus = bus

    def open(self):
        # An imported Pi-Plates module alone does not establish board presence.
        self.bus.prepare(self.target.device)

    def write(self, active):
        self.bus.call("setDOUTbit" if active else "clrDOUTbit", self.target.device, self.target.pin)

    def close(self):
        pass  # Never reset a whole board or disturb a different active bit.


class DebugPulseController:
    """One worker per selected pin; failures never disable unrelated outputs."""

    def __init__(self, *, daq_module="piplates.DAQC2plate", output_factory=None):
        self.bus = DAQC2Bus(daq_module)
        self.output_factory = output_factory or self._output
        self.events = queue.SimpleQueue()
        self._lock = threading.Lock()
        self._jobs = {}
        self._uncleared = {}

    def _output(self, target):
        return GPIOOutput(target) if target.backend == "gpio" else DAQC2Output(target, self.bus)

    @property
    def busy(self):
        with self._lock:
            return bool(self._jobs)

    @property
    def uncleared(self):
        with self._lock:
            return bool(self._uncleared)

    def pulse(self, target):
        with self._lock:
            if target in self._jobs or target in self._uncleared:
                return False
            stop = threading.Event()
            self._jobs[target] = stop
        self.events.put((target, "starting", "Opening output…"))
        self._start_worker(target, stop)
        return True

    def _start_worker(self, target, stop, output=None):
        try:
            threading.Thread(
                target=self._run, args=(target, stop, output),
                name=f"debug-{target.backend}-{target.device}-{target.pin}", daemon=True,
            ).start()
        except HARDWARE_ERRORS as exc:
            with self._lock:
                self._jobs.pop(target, None)
            self.events.put((target, "error", str(exc)))

    def _run(self, target, stop, output=None):
        errors = []
        must_clear = output is not None
        try:
            if output is None:
                output = self.output_factory(target)
                output.open()
                if not stop.is_set():
                    # Even a failed ON call may have reached the physical output.
                    must_clear = True
                    output.write(True)
                    self.events.put((target, "active", "ON — 5 seconds"))
                    stop.wait(PULSE_SECONDS)
        except HARDWARE_ERRORS as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
        finally:
            cleared = not must_clear
            if must_clear:
                for attempt in range(3):
                    try:
                        output.write(False)
                        cleared = True
                        break
                    except HARDWARE_ERRORS as exc:
                        if attempt == 2:
                            errors.append(f"OFF failed: {exc}. Tap Stop All to retry.")
                        else:
                            time.sleep(0.05)
            release_failed = False
            if output is not None and cleared:
                try:
                    output.close()
                except HARDWARE_ERRORS as exc:
                    release_failed = True
                    errors.append(f"Release failed: {exc}. Tap Stop All to retry.")
            with self._lock:
                if not cleared or release_failed:
                    self._uncleared[target] = output
                else:
                    self._uncleared.pop(target, None)
                self._jobs.pop(target, None)
                if not cleared:
                    state = "uncleared"
                elif release_failed:
                    state = "cleanup"
                else:
                    state = "error" if errors else "done"
                detail = "; ".join(errors) if errors else (
                    "OFF — stopped" if stop.is_set() else "OFF — pulse complete"
                )
                self.events.put((target, state, detail))

    def stop_all(self):
        """Cancel active pulses and retry failed OFF writes without blocking Tk."""
        retries = []
        with self._lock:
            for stop in self._jobs.values():
                stop.set()
            for target, output in self._uncleared.items():
                if target not in self._jobs:
                    stop = threading.Event()
                    stop.set()
                    self._jobs[target] = stop
                    retries.append((target, stop, output))
        for target, stop, output in retries:
            self._start_worker(target, stop, output)
