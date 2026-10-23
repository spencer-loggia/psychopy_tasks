import types
import time
import unittest
from unittest.mock import patch

from bin.daqc2_outputs import DAQC2DigitalOutputs, PeriodicDOUTPulseController
from bin.eye_tracking import EyeCalibration, EyeTrackerState
from bin.fixation_reward import ScreenAutoReward


class DAQC2DigitalOutputsTests(unittest.TestCase):
    def test_queued_edges_keep_write_times_without_waiting_for_pulse_duration(self):
        now = 9_007_199_254_740_993

        def write(bit, active):
            nonlocal now
            now += 500

        outputs = types.SimpleNamespace(validate_bit=int, write=write)
        controller = PeriodicDOUTPulseController(
            outputs, bit=0, interval_s=2, pulse_duration_s=0.25,
        )
        on_request = now
        with patch("bin.daqc2_outputs.time.perf_counter_ns", side_effect=lambda: now):
            controller._record_edge(active=True, pulse_index=1)
            now += 250_000_000
            off_request = now
            controller._record_edge(active=False, pulse_index=1)
            now += 1_000_000_000  # Log draining can happen much later.
            on, off = controller.drain_edges()

        self.assertEqual((on.requested_perf_ns, on.actual_perf_ns),
                         (on_request, on_request + 500))
        self.assertEqual((off.requested_perf_ns, off.actual_perf_ns),
                         (off_request, off_request + 500))

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
        with self.assertRaisesRegex(ValueError, "shorter than reward_interval"):
            PeriodicDOUTPulseController(
                outputs,
                bit=0,
                interval_s=0.25,
                pulse_duration_s=0.25,
            )

    def run_fixation_pulses(self, *, misses=(), samples_end_ms=2300, stop_s=2.3):
        policy = ScreenAutoReward(
            reward_interval=1, calibration=EyeCalibration(x_scale=1, y_scale=1),
            max_sample_age_s=0.15,
        )
        policy.toggle(now_s=0)
        now = 0.0
        last_sample_ms = -1
        writes = []

        def wait(seconds):
            nonlocal now, last_sample_ms
            now = min(stop_s, round(now + seconds, 9))
            end_ms = min(samples_end_ms, int(round(now * 1000)))
            for ms in range(last_sample_ms + 1, end_ms + 1):
                policy.observe(EyeTrackerState(
                    timestamp_perf_s=ms / 1000, frame_sequence=ms + 1,
                    x_smooth_px=1 if ms in misses else 0, y_smooth_px=0, valid=True,
                ))
            last_sample_ms = end_ms
            return now >= stop_s

        outputs = types.SimpleNamespace(
            validate_bit=int, write=lambda bit, active: writes.append((active, now)),
        )
        controller = PeriodicDOUTPulseController(
            outputs, bit=0, interval_s=1, pulse_duration_s=0.1,
            clock=lambda: now, reward_policy=policy,
        )
        controller._anchor_perf_s = 0
        controller._stop_event = types.SimpleNamespace(wait=wait)
        controller._run()
        self.assertIsNone(controller.failure)
        return writes

    def test_fixation_pump_waits_until_gaze_qualifies_and_spaces_delayed_rewards(self):
        writes = self.run_fixation_pulses(misses=range(980, 1000))
        onsets = [when for active, when in writes if active]
        self.assertEqual(len(onsets), 2)
        self.assertGreater(onsets[0], 1.095)
        self.assertLessEqual(onsets[0], 1.105)
        # History starts with the first new camera sample after reset; the
        # worker delivers at the first check with a complete fresh interval.
        self.assertGreaterEqual(onsets[1] - onsets[0], 1.001)
        self.assertLessEqual(onsets[1] - onsets[0], 1.006)
        self.assertEqual([active for active, _ in writes], [True, False, True, False])
        self.assertAlmostEqual(writes[1][1] - writes[0][1], 0.1)

    def test_fixation_pump_blocks_without_samples_and_when_acquisition_stops(self):
        self.assertEqual(self.run_fixation_pulses(samples_end_ms=-1), [])
        self.assertEqual(self.run_fixation_pulses(misses=range(2301)), [])
        writes = self.run_fixation_pulses(samples_end_ms=1000)
        self.assertEqual([active for active, _ in writes], [True, False])

    def test_fixation_pump_stopping_during_pulse_clears_output(self):
        writes = self.run_fixation_pulses(stop_s=1.05)
        self.assertEqual(writes, [(True, 1), (False, 1.05)])


if __name__ == "__main__":
    unittest.main()
