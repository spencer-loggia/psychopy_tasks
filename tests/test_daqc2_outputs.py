import types
import time
import unittest
from unittest.mock import patch

from bin.daqc2_outputs import DAQC2DigitalOutputs, PeriodicDOUTPulseController


class DAQC2DigitalOutputsTests(unittest.TestCase):
    def test_write_maps_logical_on_to_set_and_off_to_clear(self):
        calls = []

        fake_module = types.SimpleNamespace(
            setDOUTbit=lambda addr, bit: calls.append(("set", addr, bit)),
            clrDOUTbit=lambda addr, bit: calls.append(("clear", addr, bit)),
        )

        with patch("importlib.import_module", return_value=fake_module):
            outputs = DAQC2DigitalOutputs(address=2)
            outputs.write(3, True)
            outputs.write(3, False)

        self.assertEqual(calls, [("set", 2, 3), ("clear", 2, 3)])

    def test_bound_bit_callables_reuse_opened_functions(self):
        calls = []

        fake_module = types.SimpleNamespace(
            setDOUTbit=lambda addr, bit: calls.append(("set", addr, bit)),
            clrDOUTbit=lambda addr, bit: calls.append(("clear", addr, bit)),
        )

        with patch("importlib.import_module", return_value=fake_module) as import_module:
            outputs = DAQC2DigitalOutputs(address=1)
            outputs.open()
            turn_on, turn_off = outputs.bind_bit(0)
            turn_on()
            turn_off()

        import_module.assert_called_once()
        self.assertEqual(calls, [("set", 1, 0), ("clear", 1, 0)])

    def test_rejects_invalid_address_and_bit(self):
        with self.assertRaises(ValueError):
            DAQC2DigitalOutputs(address=8)

        outputs = DAQC2DigitalOutputs(enabled=False)
        with self.assertRaises(ValueError):
            outputs.write(24, True)

    def test_periodic_controller_pulses_on_worker_thread_and_clears_on_stop(self):
        writes = []
        fake_module = types.SimpleNamespace(
            setDOUTbit=lambda addr, bit: writes.append((True, addr, bit, time.perf_counter())),
            clrDOUTbit=lambda addr, bit: writes.append((False, addr, bit, time.perf_counter())),
        )
        with patch("importlib.import_module", return_value=fake_module):
            controller = PeriodicDOUTPulseController(
                DAQC2DigitalOutputs(address=2),
                bit=3,
                interval_s=0.03,
                pulse_duration_s=0.005,
            )
            controller.start()
            deadline = time.perf_counter() + 0.5
            while (
                sum(active for active, *_ in writes) < 2
                and time.perf_counter() < deadline
            ):
                time.sleep(0.005)
            controller.stop()

        edges = controller.drain_edges()
        self.assertGreaterEqual(sum(edge.active for edge in edges), 2)
        self.assertEqual([edge.active for edge in edges[:4]], [True, False, True, False])
        self.assertFalse(writes[-1][0])
        self.assertTrue(all(write[1:3] == (2, 3) for write in writes))

    def test_periodic_controller_rejects_overlapping_pulses(self):
        outputs = DAQC2DigitalOutputs(enabled=False)
        with self.assertRaisesRegex(ValueError, "shorter than pump_interval"):
            PeriodicDOUTPulseController(
                outputs,
                bit=0,
                interval_s=0.25,
                pulse_duration_s=0.25,
            )


if __name__ == "__main__":
    unittest.main()
