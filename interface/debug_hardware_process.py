"""Run debug hardware in the configured task environment, keeping Tk independent."""
from __future__ import annotations

from collections import deque
from dataclasses import asdict
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

from bin.debug_outputs import OutputPin


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKER_SCRIPT = PROJECT_ROOT / "bin" / "debug_output_worker.py"
STARTUP_TIMEOUT = 8.0
FINAL_STATES = {"done", "error", "uncleared", "cleanup"}


def debug_python_candidates(cfg, *, config_dir=None):
    """Keep the venv executable path intact: resolving its symlink loses the venv."""
    environment = cfg.get("environment")
    environment = environment if isinstance(environment, dict) else {}
    configured = environment.get("python")
    candidates = []
    if isinstance(configured, str) and configured.strip():
        try:
            value = configured.strip()
            if "/" in value or value.startswith("~"):
                path = Path(value).expanduser()
                if not path.is_absolute():
                    base = Path(config_dir or PROJECT_ROOT)
                    working_dir = environment.get("working_directory", environment.get("working_dir"))
                    if isinstance(working_dir, str) and working_dir.strip():
                        working_path = Path(working_dir).expanduser()
                        base = working_path if working_path.is_absolute() else base / working_path
                    path = base / path
                value = str(path)
            candidates.append(value)
        except (ValueError, RuntimeError, OSError):
            pass
    if sys.executable not in candidates:
        candidates.append(sys.executable)
    return candidates


def hardware_environment(python):
    env = os.environ.copy()
    # Invoking the venv's Python activates its site-packages. Match activation
    # for helpers the driver itself may run (notably `pip show Pi-Plates`).
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    env.pop("VIRTUAL_ENV", None)
    executable = Path(python)
    if executable.is_absolute():
        env["PATH"] = str(executable.parent) + os.pathsep + env.get("PATH", "")
        if executable.parent.name == "bin":
            env["VIRTUAL_ENV"] = str(executable.parent.parent)
    env["PYTHONUNBUFFERED"] = "1"
    return env


class _BackendWorker:
    def __init__(self, backend, candidates, daq_module, events):
        self.backend = backend
        self.candidates = candidates
        self.daq_module = daq_module
        self.events = events
        self.lock = threading.RLock()
        self.jobs = {}
        self.failed = set()
        self.commands = queue.SimpleQueue()
        self.thread = None
        self.closing = False

    @property
    def busy(self):
        with self.lock:
            return bool(self.jobs) or bool(self.closing and self.thread and self.thread.is_alive())

    @property
    def uncleared(self):
        with self.lock:
            return bool(self.failed)

    def _start(self):
        if self.thread is None or not self.thread.is_alive():
            self.thread = threading.Thread(target=self._run, name=f"debug-{self.backend}-process", daemon=True)
            self.thread.start()

    def pulse(self, target):
        with self.lock:
            if self.closing or target in self.jobs or target in self.failed:
                return False
            cancelled = threading.Event()
            self.jobs[target] = cancelled
            self.events.put((target, "starting", "Opening output in the task Python environment…"))
            self.commands.put(("pulse", target, cancelled))
            self._start()
            return True

    def stop_all(self):
        with self.lock:
            for cancelled in self.jobs.values():
                cancelled.set()
            if self.thread is not None and self.thread.is_alive():
                self.commands.put(("stop_all", None, None))
            for target in self.failed:
                if target not in self.jobs:
                    cancelled = threading.Event()
                    self.jobs[target] = cancelled
                    self.commands.put(("clear", target, cancelled))
            if self.jobs:
                self._start()

    def close(self):
        with self.lock:
            self.closing = True
            for cancelled in self.jobs.values():
                cancelled.set()
            self.commands.put(("close", None, None))

    def _event(self, target, state, detail):
        with self.lock:
            if state in FINAL_STATES:
                self.jobs.pop(target, None)
                if state in {"uncleared", "cleanup"}:
                    self.failed.add(target)
                else:
                    self.failed.discard(target)
            self.events.put((target, state, detail))

    @staticmethod
    def _read_output(process, incoming):
        try:
            for line in process.stdout:
                incoming.put(line)
        finally:
            incoming.put(None)

    @staticmethod
    def _dispose(process):
        if process is None:
            return
        if process.stdin:
            try:
                process.stdin.close()  # EOF requests OFF before the helper exits.
            except OSError:
                pass
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=6.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1.0)
        finally:
            if process.stdout:
                process.stdout.close()

    def _launch(self):
        errors = []
        for python in self.candidates:
            process = None
            recent = deque(maxlen=8)
            try:
                process = subprocess.Popen(
                    [python, "-u", str(WORKER_SCRIPT), "--backend", self.backend, "--daq-module", self.daq_module],
                    cwd=PROJECT_ROOT, env=hardware_environment(python), stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                )
                incoming = queue.SimpleQueue()
                threading.Thread(target=self._read_output, args=(process, incoming), daemon=True).start()
                deadline = time.monotonic() + STARTUP_TIMEOUT
                while time.monotonic() < deadline:
                    if self.closing:
                        raise RuntimeError("Debug Mode closed during environment startup")
                    try:
                        line = incoming.get(timeout=0.05)
                    except queue.Empty:
                        continue
                    if line is None:
                        raise RuntimeError("; ".join(recent) or "Hardware worker exited before startup")
                    recent.append(line.strip()[:500])
                    try:
                        message = json.loads(line)
                    except ValueError:
                        continue
                    if not isinstance(message, dict):
                        continue
                    if message.get("kind") == "ready":
                        return process, incoming, str(message.get("python") or python)
                    if message.get("kind") == "unavailable":
                        raise RuntimeError(message["detail"])
                raise TimeoutError("Hardware environment did not become ready")
            except Exception as exc:
                errors.append(f"{python}: {exc}")
                self._dispose(process)
                if self.closing:
                    break
        raise RuntimeError("; ".join(errors))

    def _run(self):
        process = None
        sent = set()
        try:
            process, incoming, python = self._launch()
            while True:
                # Read hardware results before accepting another request.
                while True:
                    try:
                        line = incoming.get_nowait()
                    except queue.Empty:
                        break
                    if line is None:
                        if self.closing and not self.jobs:
                            return
                        raise RuntimeError(f"Hardware worker exited ({python})")
                    try:
                        message = json.loads(line)
                    except ValueError:
                        continue  # Driver output is diagnostic text, not protocol.
                    if isinstance(message, dict) and message.get("kind") == "event":
                        target = OutputPin(**message["target"])
                        state = message["state"]
                        if state in FINAL_STATES:
                            sent.discard(target)
                        self._event(target, state, f"{message['detail']} [{python}]")
                try:
                    action, target, cancelled = self.commands.get(timeout=0.02)
                except queue.Empty:
                    continue
                if action == "pulse" and cancelled.is_set():
                    self._event(target, "done", "Stopped before output was activated")
                    continue
                command = {"action": action}
                if target is not None:
                    command["target"] = asdict(target)
                    sent.add(target)
                process.stdin.write(json.dumps(command) + "\n")
                process.stdin.flush()
        except Exception as exc:
            with self.lock:
                for target, cancelled in list(self.jobs.items()):
                    state = "uncleared" if target in sent or target in self.failed else "error"
                    detail = f"{exc}. OFF not confirmed; tap Stop All to retry." if state == "uncleared" else str(exc)
                    if cancelled.is_set() and state == "error":
                        state, detail = "done", "Stopped before output was activated"
                    self._event(target, state, detail)
                # Do not replay queued pulses when an environment is retried.
                self.commands = queue.SimpleQueue()
                if self.failed:
                    self.closing = False  # Keep OFF retry possible after a crash.
        finally:
            try:
                self._dispose(process)
            finally:
                with self.lock:
                    self.thread = None
                    # A retry may arrive while the old process is being reaped.
                    if self.jobs and not self.closing:
                        self._start()


class DebugHardwareController:
    """Facade matching the touch panel's controller API; no I/O on Tk's thread."""

    def __init__(self, cfg, *, config_dir=None, daq_module="piplates.DAQC2plate"):
        self.events = queue.SimpleQueue()
        candidates = debug_python_candidates(cfg, config_dir=config_dir)
        self.workers = {
            backend: _BackendWorker(backend, candidates, daq_module, self.events)
            for backend in ("gpio", "daqc2")
        }

    @property
    def busy(self):
        return any(worker.busy for worker in self.workers.values())

    @property
    def uncleared(self):
        return any(worker.uncleared for worker in self.workers.values())

    def pulse(self, target):
        return self.workers[target.backend].pulse(target)

    def stop_all(self):
        for worker in self.workers.values():
            worker.stop_all()

    def close(self):
        for worker in self.workers.values():
            worker.close()
