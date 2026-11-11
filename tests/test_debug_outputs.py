import queue
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

from bin.debug_outputs import DAQC2Bus, DAQC2Output, DebugPulseController, GPIOOutput, OutputPin


class DebugOutputTests(unittest.TestCase):
    def wait_idle(self, controller):
        deadline = time.monotonic() + 2
        while controller.busy and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertFalse(controller.busy, "Pulse worker did not finish")

    def events(self, controller):
        result = []
        while True:
            try:
                result.append(controller.events.get_nowait())
            except queue.Empty:
                return result

    def test_five_second_pulse_clears_and_releases_only_selected_pin(self):
        output = Mock()
        controller = DebugPulseController(output_factory=Mock(return_value=output))
        target = OutputPin("gpio", 18)
        stop = Mock()
        stop.is_set.return_value = False
        controller._run(target, stop)
        stop.wait.assert_called_once_with(5.0)
        self.assertEqual(output.mock_calls, [call.open(), call.write(True), call.write(False), call.close()])
        self.assertEqual(self.events(controller)[-1][1], "done")

    def test_gpio_uses_bcm_number_selected_chip_and_negative_driver_status(self):
        module = Mock()
        module.gpiochip_open.return_value = 12
        module.gpio_claim_output.return_value = 0
        module.gpio_write.return_value = -5
        module.gpiochip_close.return_value = 0
        with patch("bin.debug_outputs.importlib.import_module", return_value=module):
            output = GPIOOutput(OutputPin("gpio", 27, 4))
            output.open()
            with self.assertRaisesRegex(RuntimeError, "-5"):
                output.write(True)
            output.close()
        module.gpiochip_open.assert_called_once_with(4)
        module.gpio_claim_output.assert_called_once_with(12, 27, 0)
        module.gpio_write.assert_called_once_with(12, 27, 1)
        module.gpiochip_close.assert_called_once_with(12)

    def test_claim_failure_releases_handle_without_writing_busy_gpio(self):
        module = Mock()
        module.gpiochip_open.return_value = 12
        module.gpio_claim_output.side_effect = RuntimeError("GPIO busy")
        module.gpiochip_close.return_value = 0
        controller = DebugPulseController()
        with patch("bin.debug_outputs.importlib.import_module", return_value=module):
            controller._run(OutputPin("gpio", 23), threading.Event())
        module.gpio_write.assert_not_called()
        module.gpiochip_close.assert_called_once_with(12)
        self.assertIn("GPIO busy", self.events(controller)[-1][2])

    def test_daq_uses_task_adapter_without_extra_board_or_adc_checks(self):
        module = SimpleNamespace(
            getADC=Mock(side_effect=AssertionError("No ADC check in tasks")),
            getADDR=Mock(side_effect=AssertionError("No address probe in tasks")),
            setDOUTbit=Mock(return_value=None), clrDOUTbit=Mock(return_value=None),
        )
        with patch("bin.debug_outputs.importlib.import_module", return_value=module):
            output = DAQC2Output(OutputPin("daqc2", 7, 3), DAQC2Bus())
            output.open()
            output.write(True)
            output.write(False)
        module.getADC.assert_not_called()
        module.getADDR.assert_not_called()
        module.setDOUTbit.assert_called_once_with(3, 7)
        module.clrDOUTbit.assert_called_once_with(3, 7)

    def test_missing_daq_driver_and_system_exit_do_not_block_gpio(self):
        for error in (ImportError("No SPI"), SystemExit(1)):
            with self.subTest(error=error):
                good = Mock()
                bad = Mock()
                bad.open.side_effect = error
                controller = DebugPulseController(output_factory=lambda pin: bad if pin.backend == "daqc2" else good)
                with patch("bin.debug_outputs.PULSE_SECONDS", 0.001):
                    controller.pulse(OutputPin("daqc2", 0))
                    controller.pulse(OutputPin("gpio", 18))
                    self.wait_idle(controller)
                good.write.assert_has_calls([call(True), call(False)])
                self.assertEqual({event[1] for event in self.events(controller)}, {"starting", "active", "error", "done"})

    def test_daq_reconnection_refreshes_only_selected_board_without_analog_or_global_poll(self):
        module = SimpleNamespace(
            getADDR=Mock(return_value=3), daqc2sPresent=[0] * 8,
            CMD=SimpleNamespace(DataGood=True), getADC=Mock(), quietPoll=Mock(),
            setDOUTbit=Mock(return_value=None), clrDOUTbit=Mock(return_value=None),
        )
        with patch("bin.debug_outputs.importlib.import_module", return_value=module):
            output = DAQC2Output(OutputPin("daqc2", 7, 3), DAQC2Bus())
            output.open()
            output.write(True)
        self.assertEqual(module.daqc2sPresent, [0, 0, 0, 1, 0, 0, 0, 0])
        module.getADC.assert_not_called()
        module.quietPoll.assert_not_called()
        module.setDOUTbit.assert_called_once_with(3, 7)

    def test_silent_daq_ack_failure_is_reported_even_when_write_returns_none(self):
        module = SimpleNamespace(CMD=SimpleNamespace(DataGood=True))

        def failed_write(*args):
            module.CMD.DataGood = False

        module.clrDOUTbit = failed_write
        with patch("bin.debug_outputs.importlib.import_module", return_value=module):
            with self.assertRaisesRegex(OSError, "did not acknowledge"):
                DAQC2Bus().call("clrDOUTbit", 0, 1)

    def test_hung_initialization_does_not_block_other_pins_and_stop_prevents_late_on(self):
        entered, release = threading.Event(), threading.Event()
        blocked, good = Mock(), Mock()

        def block():
            entered.set()
            release.wait(2)

        blocked.open.side_effect = block
        controller = DebugPulseController(output_factory=lambda pin: blocked if pin.pin == 1 else good)
        try:
            with patch("bin.debug_outputs.PULSE_SECONDS", 0.001):
                controller.pulse(OutputPin("gpio", 1))
                self.assertTrue(entered.wait(1))
                self.assertTrue(controller.pulse(OutputPin("gpio", 2)))
                deadline = time.monotonic() + 1
                while not good.close.called and time.monotonic() < deadline:
                    time.sleep(0.005)
                good.close.assert_called_once()
                controller.stop_all()
        finally:
            release.set()
            self.wait_idle(controller)
        blocked.write.assert_not_called()

    def test_stop_interrupts_pulse_and_repeat_taps_do_not_extend_it(self):
        active = threading.Event()
        output = Mock()
        output.write.side_effect = lambda state: active.set() if state else None
        controller = DebugPulseController(output_factory=lambda pin: output)
        target = OutputPin("gpio", 18)
        self.assertTrue(controller.pulse(target))
        self.assertTrue(active.wait(1))
        self.assertFalse(controller.pulse(target))
        controller.stop_all()
        self.wait_idle(controller)
        output.write.assert_has_calls([call(True), call(False)])
        self.assertIn("stopped", self.events(controller)[-1][2])

    def test_failed_on_still_attempts_off(self):
        output = Mock()
        output.write.side_effect = [RuntimeError("partial write"), None]
        controller = DebugPulseController(output_factory=lambda pin: output)
        controller._run(OutputPin("daqc2", 0), threading.Event())
        output.write.assert_has_calls([call(True), call(False)])
        self.assertFalse(controller.uncleared)
        self.assertEqual(self.events(controller)[-1][1], "error")

    def test_off_failure_is_visible_and_retryable_without_blocking_other_outputs(self):
        failed = Mock()
        failed.write.side_effect = [None, OSError("disconnected"), OSError("disconnected"), OSError("disconnected")]
        good = Mock()
        controller = DebugPulseController(output_factory=lambda pin: failed if pin.pin == 1 else good)
        target = OutputPin("daqc2", 1)
        stop = Mock()
        stop.is_set.return_value = False
        controller._run(target, stop)
        self.assertTrue(controller.uncleared)
        self.assertFalse(controller.pulse(target))
        failed.close.assert_not_called()
        self.assertEqual(self.events(controller)[-1][1], "uncleared")
        with patch("bin.debug_outputs.PULSE_SECONDS", 0.001):
            controller.pulse(OutputPin("daqc2", 2))
            self.wait_idle(controller)
        good.write.assert_has_calls([call(True), call(False)])
        failed.write.side_effect = None
        controller.stop_all()
        self.wait_idle(controller)
        self.assertFalse(controller.uncleared)
        failed.write.assert_called_with(False)
        failed.close.assert_called_once()

    def test_daq_pulses_can_overlap_without_holding_spi_lock_for_five_seconds(self):
        module = SimpleNamespace(getADC=lambda *_: 5.0, setDOUTbit=Mock(return_value=None), clrDOUTbit=Mock(return_value=None))
        controller = DebugPulseController()
        with patch("bin.debug_outputs.importlib.import_module", return_value=module):
            try:
                controller.pulse(OutputPin("daqc2", 1))
                controller.pulse(OutputPin("daqc2", 2))
                deadline = time.monotonic() + 1
                while module.setDOUTbit.call_count < 2 and time.monotonic() < deadline:
                    time.sleep(0.005)
                self.assertEqual(module.setDOUTbit.call_count, 2)
                module.clrDOUTbit.assert_not_called()
            finally:
                controller.stop_all()
                self.wait_idle(controller)
        module.clrDOUTbit.assert_has_calls([call(0, 1), call(0, 2)], any_order=True)

    def test_failed_gpio_release_retains_handle_for_retry(self):
        output = Mock()
        output.close.side_effect = [OSError("release failed"), None]
        controller = DebugPulseController(output_factory=lambda pin: output)
        stop = Mock()
        stop.is_set.return_value = False
        controller._run(OutputPin("gpio", 18), stop)
        self.assertEqual(self.events(controller)[-1][1], "cleanup")
        self.assertTrue(controller.uncleared)
        controller.stop_all()
        self.wait_idle(controller)
        self.assertFalse(controller.uncleared)
        self.assertEqual(output.close.call_count, 2)

    def test_clear_only_retry_never_sends_on_even_if_open_failed(self):
        output = Mock()
        output.open.side_effect = OSError("temporary open failure")
        output.write.side_effect = OSError("still disconnected")
        controller = DebugPulseController(output_factory=lambda pin: output)
        target = OutputPin("daqc2", 1)
        controller.clear(target)
        self.wait_idle(controller)
        self.assertTrue(controller.uncleared)
        output.open.side_effect = None
        output.write.side_effect = None
        controller.stop_all()
        self.wait_idle(controller)
        self.assertFalse(controller.uncleared)
        self.assertTrue(all(args == call(False) for args in output.write.call_args_list))


if __name__ == "__main__":
    unittest.main()
