import types
import unittest
from unittest.mock import patch

from bin.daqc2_outputs import DAQC2DigitalOutputs


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


if __name__ == "__main__":
    unittest.main()
