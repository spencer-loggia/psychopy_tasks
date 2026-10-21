"""Exercise flip GPIO callbacks without importing a display runtime."""
import ast
from pathlib import Path
import sys
import time
import types
import unittest
from unittest.mock import patch


def load_gpio_callbacks():
    path = Path(__file__).resolve().parents[1] / "bin" / "utils.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names = {"_set_gpio_level_on_flip", "_send_led_pulse_on_flip"}
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {"time": time}
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace


class GPIOTimestampTests(unittest.TestCase):
    def test_onset_is_captured_before_async_pulse_scheduling(self):
        callbacks = load_gpio_callbacks()
        now = 9_007_199_254_740_993
        request_ns = now
        capture = {}
        calls = []

        def gpio_write(chip, pin, level):
            nonlocal now
            calls.append(("write", chip, pin, level))
            now += 200
            return 0

        def tx_pulse(chip, pin, on_us, off_us, offset, cycles):
            nonlocal now
            # Even slow scheduling must not shift the already-captured onset.
            self.assertEqual(capture["actual_perf_ns"], request_ns + 200)
            calls.append(("schedule", on_us, off_us, cycles))
            now += 5_000_000
            return 0

        gpio = types.SimpleNamespace(gpio_write=gpio_write, tx_pulse=tx_pulse)
        with (
            patch.dict(sys.modules, {"lgpio": gpio}),
            patch.object(time, "perf_counter_ns", side_effect=lambda: now),
            patch.object(time, "sleep", side_effect=AssertionError("Pulse callback must not wait")),
        ):
            callbacks["_send_led_pulse_on_flip"]("chip", 18, 250_000, capture)

        self.assertEqual(capture, {
            "requested_perf_ns": request_ns,
            "actual_perf_ns": request_ns + 200,
        })
        self.assertEqual(calls[0], ("write", "chip", 18, 1))
        self.assertEqual(len(calls), 2)
        self.assertLess(now - request_ns, 250_000_000)

    def test_both_levels_are_timestamped_immediately_around_the_write(self):
        write_level = load_gpio_callbacks()["_set_gpio_level_on_flip"]
        now = 1_000_000_003

        def gpio_write(chip, pin, level):
            nonlocal now
            now += 37
            return 0

        gpio = types.SimpleNamespace(gpio_write=gpio_write)
        for level in (1, 0):
            request_ns = now
            capture = {}
            with patch.object(time, "perf_counter_ns", side_effect=lambda: now):
                write_level(gpio, "chip", 18, level, capture)
            self.assertEqual(capture["requested_perf_ns"], request_ns)
            self.assertEqual(capture["actual_perf_ns"], request_ns + 37)

    def test_failed_write_cannot_be_committed_as_success(self):
        write_level = load_gpio_callbacks()["_set_gpio_level_on_flip"]
        gpio = types.SimpleNamespace(gpio_write=lambda *args: -5)
        with self.assertRaisesRegex(RuntimeError, "GPIO write failed"):
            write_level(gpio, "chip", 18, 1, {})


if __name__ == "__main__":
    unittest.main()
