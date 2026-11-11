#!/usr/bin/env python3
"""Private hardware worker; runs in the task interpreter without Tk or PsychoPy."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib
import json
import os
from pathlib import Path
import queue
import signal
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bin.daqc2_outputs import DAQC2DigitalOutputs
from bin.debug_outputs import DebugPulseController, HARDWARE_ERRORS, OutputPin


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("gpio", "daqc2"), required=True)
    parser.add_argument("--daq-module", default="piplates.DAQC2plate")
    args = parser.parse_args()

    # Also cover environment.python supplied as a PATH command (e.g. python3).
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
    if sys.prefix != sys.base_prefix:
        os.environ["VIRTUAL_ENV"] = sys.prefix

    # Driver prints/warnings cannot corrupt the JSON protocol.
    protocol = sys.stdout
    sys.stdout = sys.stderr
    write_lock = threading.Lock()

    def emit(message):
        with write_lock:
            protocol.write(json.dumps(message) + "\n")
            protocol.flush()

    try:
        if args.backend == "gpio":
            importlib.import_module("lgpio")
        else:
            # Same import and API validation as the experiment tasks; no output
            # is written during environment selection or worker startup.
            DAQC2DigitalOutputs(module_name=args.daq_module).open()
    except HARDWARE_ERRORS as exc:
        emit({"kind": "unavailable", "detail": f"{type(exc).__name__}: {exc}"})
        return 1

    controller = DebugPulseController(daq_module=args.daq_module)
    finished = threading.Event()

    def forward_events():
        while not finished.is_set() or not controller.events.empty():
            try:
                target, state, detail = controller.events.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                emit({"kind": "event", "target": asdict(target), "state": state, "detail": detail})
            except (BrokenPipeError, OSError):
                return

    def interrupt(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    sender = threading.Thread(target=forward_events, daemon=True)
    sender.start()
    try:
        emit({"kind": "ready", "python": sys.executable})
        for line in sys.stdin:
            command = json.loads(line)
            action = command["action"]
            if action == "close":
                break
            if action == "stop_all":
                controller.stop_all()
                continue
            target = OutputPin(**command["target"])
            if target.backend != args.backend:
                raise ValueError("Output does not belong to this worker")
            if action == "pulse":
                controller.pulse(target)
            elif action == "clear":
                controller.clear(target)
            else:
                raise ValueError("Unknown hardware command")
    except (KeyboardInterrupt, BrokenPipeError):
        pass
    finally:
        # EOF also handles a launcher crash/exit: don't leave DOUT energized.
        controller.stop_all()
        deadline = time.monotonic() + 6.0
        while controller.busy and time.monotonic() < deadline:
            time.sleep(0.01)
        finished.set()
        sender.join(timeout=1.0)
    return int(controller.busy or controller.uncleared)


if __name__ == "__main__":
    raise SystemExit(main())
