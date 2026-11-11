import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import venv

from bin.debug_outputs import OutputPin
from interface.debug_hardware_process import (
    DebugHardwareController, debug_python_candidates, hardware_environment,
)


class DebugEnvironmentTests(unittest.TestCase):
    def test_configured_interpreter_is_first_without_resolving_venv_symlink(self):
        path = "/opt/psychopy/PsychoPy-2025.1.1-Python3.10/.venv/bin/python"
        candidates = debug_python_candidates({"environment": {"python": path}})
        self.assertEqual(candidates[0], path)
        self.assertEqual(candidates[-1], sys.executable)

    def test_broken_configuration_still_has_current_interpreter_fallback(self):
        for cfg in ({}, {"environment": []}, {"environment": {"python": []}}, {"environment": {"python": "~__missing_debug_user__/python"}}):
            with self.subTest(cfg=cfg):
                self.assertEqual(debug_python_candidates(cfg), [sys.executable])

    def test_relative_interpreter_uses_configured_working_directory(self):
        cfg = {"environment": {"python": ".venv/bin/python", "working_dir": "../tasks"}}
        self.assertEqual(debug_python_candidates(cfg, config_dir=Path("/rig/config"))[0], "/rig/config/../tasks/.venv/bin/python")

    def test_activation_uses_selected_venv_and_removes_foreign_python_settings(self):
        with patch.dict(os.environ, {"PYTHONHOME": "/wrong", "PYTHONPATH": "/wrong", "VIRTUAL_ENV": "/other", "PATH": "/usr/bin"}, clear=True):
            env = hardware_environment("/opt/task/.venv/bin/python")
        self.assertEqual(env["VIRTUAL_ENV"], "/opt/task/.venv")
        self.assertEqual(env["PATH"], "/opt/task/.venv/bin:/usr/bin")
        self.assertNotIn("PYTHONHOME", env)
        self.assertNotIn("PYTHONPATH", env)


class DebugHardwareProcessTests(unittest.TestCase):
    """Exercise real subprocesses with fake drivers available only in a test venv."""

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="neuro_debug_process_test_")
        cls.folder = Path(cls.temp.name)
        cls.venv_dir = cls.folder / "task environment"
        venv.EnvBuilder(with_pip=False, symlinks=True).create(cls.venv_dir)
        cls.empty_venv = cls.folder / "empty environment"
        venv.EnvBuilder(with_pip=False, symlinks=True).create(cls.empty_venv)
        cls.python = str(cls.venv_dir / "bin" / "python")
        site = Path(subprocess.check_output(
            [cls.python, "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"], text=True,
        ).strip())
        cls.log = cls.folder / "outputs.jsonl"
        cls.control = cls.folder / "control.json"
        common = f'''
import json, os, sys, threading, time, traceback
from pathlib import Path
log = Path({str(cls.log)!r})
control = Path({str(cls.control)!r})
def options():
    return json.loads(control.read_text())
def record(action, device, bit):
    with log.open("a") as stream:
        stream.write(json.dumps(dict(action=action, device=device, bit=bit,
            time=time.monotonic(), python=sys.executable,
            venv=os.environ.get("VIRTUAL_ENV"),
            caller=[frame.filename for frame in traceback.extract_stack()])) + "\\n")
'''
        package = site / "piplates"
        package.mkdir()
        (package / "__init__.py").write_text("")
        (package / "DAQC2plate.py").write_text(common + '''
assert threading.current_thread() is threading.main_thread(), "Driver import must match task main-thread initialization"
if options().get("missing_daq"):
    raise ImportError("test DAQC driver unavailable")
time.sleep(options().get("delay_import", 0))
print("Driver startup message which is not JSON")
def setDOUTbit(address, bit):
    record("on", address, bit)
    if options().get("crash_on"):
        os._exit(7)
def clrDOUTbit(address, bit):
    if options().get("fail_off"):
        raise OSError("test OFF failure")
    record("off", address, bit)
def getADC(*args):
    raise AssertionError("Tasks do not gate DOUT on ADC")
def getADDR(*args):
    raise AssertionError("No extra board probe before a task-style DOUT write")
''')
        (site / "lgpio.py").write_text(common + '''
def gpiochip_open(chip):
    return chip
def gpio_claim_output(chip, bit, level):
    return 0
def gpio_write(chip, bit, level):
    record("gpio_on" if level else "gpio_off", chip, bit)
    return 0
def gpiochip_close(chip):
    return 0
''')

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.log.write_text("")
        self.control.write_text("{}")
        self.controllers = []

    def tearDown(self):
        self.control.write_text("{}")
        for controller in self.controllers:
            controller.stop_all()
            self.wait_until(lambda: not controller.busy)
            controller.close()
            self.wait_until(lambda: not controller.busy)

    def controller(self, candidates=None):
        # Never fall back to real hardware libraries on the test host.
        with patch("interface.debug_hardware_process.debug_python_candidates", return_value=candidates or [self.python]):
            result = DebugHardwareController({"environment": {"python": self.python}})
        self.controllers.append(result)
        return result

    def wait_until(self, condition, timeout=4):
        deadline = time.monotonic() + timeout
        while not condition() and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(condition(), "Hardware process did not reach expected state")

    def records(self):
        return [json.loads(line) for line in self.log.read_text().splitlines() if line]

    def event(self, controller, state, timeout=4):
        deadline = time.monotonic() + timeout
        seen = []
        while time.monotonic() < deadline:
            try:
                item = controller.events.get(timeout=0.05)
            except queue.Empty:
                continue
            seen.append(item)
            if item[1] == state:
                return item
        self.fail(f"No {state} event: {seen}")

    def test_five_second_dout_in_configured_venv_uses_the_task_adapter(self):
        controller = self.controller()
        controller.pulse(OutputPin("daqc2", 7, 3))
        self.event(controller, "done", timeout=8)
        records = self.records()
        self.assertEqual([(r["action"], r["device"], r["bit"]) for r in records], [("on", 3, 7), ("off", 3, 7)])
        self.assertGreaterEqual(records[1]["time"] - records[0]["time"], 4.9)
        self.assertLess(records[1]["time"] - records[0]["time"], 6.0)
        self.assertEqual(records[0]["python"], self.python)
        self.assertEqual(records[0]["venv"], str(self.venv_dir))
        self.assertTrue(any(path.endswith("bin/daqc2_outputs.py") for path in records[0]["caller"]))

    def test_missing_configured_python_falls_back_before_any_output_is_sent(self):
        controller = self.controller(["/missing/task/python", self.python])
        controller.pulse(OutputPin("daqc2", 0))
        self.assertIn(self.python, self.event(controller, "active")[2])
        controller.stop_all()
        self.event(controller, "done")
        self.assertEqual([r["action"] for r in self.records()], ["on", "off"])

    def test_valid_python_missing_piplates_also_falls_back_before_activation(self):
        empty_python = str(self.empty_venv / "bin" / "python")
        controller = self.controller([empty_python, self.python])
        controller.pulse(OutputPin("daqc2", 0))
        self.assertIn(self.python, self.event(controller, "active")[2])
        controller.stop_all()
        self.event(controller, "done")
        self.assertEqual([r["action"] for r in self.records()], ["on", "off"])

    def test_path_python_reports_actual_interpreter_and_activates_its_helpers(self):
        controller = self.controller(["python"])
        with patch.dict(os.environ, {"PATH": str(self.venv_dir / "bin") + os.pathsep + os.environ.get("PATH", "")}):
            controller.pulse(OutputPin("daqc2", 0))
            self.assertIn(self.python, self.event(controller, "active")[2])
            controller.stop_all()
            self.event(controller, "done")
        self.assertEqual(self.records()[0]["venv"], str(self.venv_dir))

    def test_stop_during_environment_startup_prevents_late_activation(self):
        self.control.write_text('{"delay_import": 0.2}')
        controller = self.controller()
        controller.pulse(OutputPin("daqc2", 0))
        controller.stop_all()
        self.event(controller, "done")
        self.assertEqual(self.records(), [])

    def test_missing_daq_driver_does_not_block_gpio_in_same_environment(self):
        self.control.write_text('{"missing_daq": true}')
        controller = self.controller()
        controller.pulse(OutputPin("daqc2", 0))
        self.event(controller, "error")
        controller.pulse(OutputPin("gpio", 18, 4))
        self.event(controller, "active")
        controller.stop_all()
        self.event(controller, "done")
        self.assertEqual([(r["action"], r["device"], r["bit"]) for r in self.records()], [("gpio_on", 4, 18), ("gpio_off", 4, 18)])

    def test_stop_all_clears_overlapping_dout_pulses(self):
        controller = self.controller()
        controller.pulse(OutputPin("daqc2", 0))
        controller.pulse(OutputPin("daqc2", 1))
        self.event(controller, "active")
        self.event(controller, "active")
        controller.stop_all()
        self.wait_until(lambda: not controller.busy)
        self.assertCountEqual([(r["action"], r["bit"]) for r in self.records()], [("on", 0), ("on", 1), ("off", 0), ("off", 1)])

    def test_off_failure_is_retryable_without_reenergizing(self):
        controller = self.controller()
        controller.pulse(OutputPin("daqc2", 1))
        self.event(controller, "active")
        self.control.write_text('{"fail_off": true}')
        controller.stop_all()
        self.event(controller, "uncleared")
        self.assertTrue(controller.uncleared)
        self.control.write_text("{}")
        controller.stop_all()
        self.event(controller, "done")
        self.assertFalse(controller.uncleared)
        self.assertEqual([r["action"] for r in self.records()], ["on", "off"])

    def test_worker_crash_marks_unknown_output_and_new_worker_can_clear_it(self):
        self.control.write_text('{"crash_on": true}')
        controller = self.controller()
        controller.pulse(OutputPin("daqc2", 2))
        self.event(controller, "uncleared")
        self.assertTrue(controller.uncleared)
        self.control.write_text("{}")
        controller.stop_all()
        self.event(controller, "done")
        self.assertFalse(controller.uncleared)
        self.assertEqual([r["action"] for r in self.records()], ["on", "off"])

    def test_closing_worker_during_pulse_clears_pin_before_process_exit(self):
        controller = self.controller()
        controller.pulse(OutputPin("daqc2", 0))
        self.event(controller, "active")
        controller.close()
        self.wait_until(lambda: not controller.busy)
        self.assertEqual([r["action"] for r in self.records()], ["on", "off"])

    def test_losing_launcher_stdin_clears_active_output(self):
        from interface.debug_hardware_process import WORKER_SCRIPT
        process = subprocess.Popen(
            [self.python, "-u", str(WORKER_SCRIPT), "--backend", "daqc2"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, env=hardware_environment(self.python),
        )
        try:
            def receive(kind, state=None):
                for line in process.stdout:
                    try:
                        message = json.loads(line)
                    except ValueError:
                        continue
                    if message.get("kind") == kind and (state is None or message.get("state") == state):
                        return
                self.fail("Hardware worker exited before expected event")

            receive("ready")
            process.stdin.write(json.dumps({"action": "pulse", "target": {"backend": "daqc2", "device": 0, "pin": 0}}) + "\n")
            process.stdin.flush()
            receive("event", "active")
            process.stdin.close()
            self.assertEqual(process.wait(timeout=3), 0)
            self.assertEqual([r["action"] for r in self.records()], ["on", "off"])
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=3)
            process.stdout.close()


if __name__ == "__main__":
    unittest.main()
