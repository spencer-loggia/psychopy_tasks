import unittest
from unittest.mock import Mock, patch

from bin.frame_timing import wait_until

from task.active_foraging_timing import (
    duration_requires_visible_phase,
    reward_interval_duration,
    run_outcome_interval,
    validate_duration_for_presentation_mode,
)


class FakeClock:
    def __init__(self):
        self.ns = 10_000_000_000

    def seconds(self):
        return self.ns / 1_000_000_000

    def sleep(self, seconds):
        self.ns += max(1, round(seconds * 1_000_000_000))


class OutcomeTimingTests(unittest.TestCase):
    def _run(self, *, pulses, timeout=False, poll=None, log_delay=0.0, hardware_delay=0.0, **overrides):
        clock = FakeClock()
        start = clock.ns
        edges = []
        logger = Mock()
        logger.log_signal.side_effect = lambda **kw: clock.sleep(log_delay) if log_delay else None
        logger.log_frame_flip.side_effect = lambda **kw: clock.sleep(log_delay) if log_delay else None
        def pump(value):
            if hardware_delay:
                clock.sleep(hardware_delay)
            edges.append(("pump", value, clock.ns - start))

        kwargs = dict(
            trial_num=7, onset_perf_ns=start, num_pulses=pulses, apply_timeout=timeout,
            onset_requested_perf_ns=start - 100,
            reward_interval_s=0.8, timeout_duration_s=1.5, pump_delay_s=0.2,
            pump_pulse_s=0.25, inter_pump_interval_s=0.1, logger=logger,
            set_pump=pump,
            set_buzzer=lambda value: edges.append(("buzzer", value, clock.ns - start)),
            poll_callback=None if poll is None else lambda: poll(clock.ns - start),
        )
        kwargs.update(overrides)
        with patch("time.perf_counter", side_effect=clock.seconds), patch(
            "time.perf_counter_ns", side_effect=lambda: clock.ns,
        ), patch("time.sleep", side_effect=clock.sleep):
            end = run_outcome_interval(**kwargs)
            if end is not None:
                # The caller must still hold a complete, separate ITI.
                wait_until((end + 300_000_000) / 1e9)
        return end, clock.ns, edges, logger

    def test_zero_single_and_double_rewards_have_identical_interval_plus_full_iti(self):
        expected_edges = {
            0: [],
            1: [("pump", 1, 200_000_000), ("pump", 0, 450_000_000)],
            2: [("pump", 1, 200_000_000), ("pump", 0, 450_000_000),
                ("pump", 1, 550_000_000), ("pump", 0, 800_000_000)],
        }
        for count in (0, 1, 2):
            with self.subTest(pulses=count):
                end, next_cue, edges, logger = self._run(pulses=count)
                self.assertEqual(end, 10_800_000_000)
                self.assertEqual(next_cue, 11_100_000_000)
                self.assertEqual(edges, expected_edges[count])
                logger.log_frame_flip.assert_called_once_with(
                    trial_num=7, event="reward_interval", timestamp_perf_ns=10_000_000_000,
                    requested_timestamp_perf_ns=9_999_999_900, requested_duration=0.8)
                logger.log_event.assert_not_called()
                self.assertEqual(logger.log_signal.call_count, count * 2)

    def test_timeout_replaces_reward_even_when_mapping_also_requests_pulses(self):
        end, next_cue, edges, logger = self._run(pulses=2, timeout=True)
        self.assertEqual(end, 11_500_000_000)
        self.assertEqual(next_cue, 11_800_000_000)
        self.assertEqual(edges, [("buzzer", 1, 0), ("buzzer", 0, 1_500_000_000)])
        logger.log_frame_flip.assert_called_once_with(
            trial_num=7, event="timeout_interval", timestamp_perf_ns=10_000_000_000,
            requested_timestamp_perf_ns=9_999_999_900, requested_duration=1.5)

    def test_slow_logging_does_not_extend_pulses_reward_interval_or_restart_iti(self):
        for count in (0, 1, 2):
            with self.subTest(pulses=count):
                end, next_cue, edges, logger = self._run(pulses=count, log_delay=0.01)
                self.assertEqual(end, 10_800_000_000)
                self.assertEqual(next_cue, 11_100_000_000)
                for index in range(0, len(edges), 2):
                    self.assertEqual(edges[index + 1][2] - edges[index][2], 250_000_000)
                for event, edge in zip(logger.log_signal.call_args_list, edges):
                    self.assertEqual(event.kwargs["timestamp_perf_ns"], 10_000_000_000 + edge[2])

    def test_abort_at_every_stage_clears_outputs_without_proceeding_to_iti(self):
        for pulses, timeout, abort_ns in ((2, False, 100_000_000), (2, False, 300_000_000),
                                         (2, False, 500_000_000), (1, False, 600_000_000),
                                         (0, False, 100_000_000), (2, True, 100_000_000)):
            with self.subTest(pulses=pulses, timeout=timeout, abort_ns=abort_ns):
                end, _, edges, logger = self._run(pulses=pulses, timeout=timeout,
                                                 poll=lambda elapsed: elapsed >= abort_ns)
                self.assertIsNone(end)
                self.assertEqual(logger.log_frame_flip.call_count, 1)
                self.assertEqual(logger.log_frame_flip.call_args.kwargs["timestamp_perf_ns"], 10_000_000_000)
                self.assertEqual(logger.log_frame_flip.call_args.kwargs["requested_timestamp_perf_ns"], 9_999_999_900)
                if edges:
                    self.assertEqual(edges[-1][1], 0)

    def test_hardware_overrun_never_shortens_pulse_or_consumes_the_iti(self):
        end, next_cue, edges, _ = self._run(pulses=2, hardware_delay=0.02)
        self.assertEqual(end, 10_880_000_000)
        self.assertEqual(next_cue - end, 300_000_000)
        for index in range(0, len(edges), 2):
            self.assertGreaterEqual(edges[index + 1][2] - edges[index][2], 250_000_000)

    def test_poller_exception_still_turns_pump_off(self):
        def fail_after_on(elapsed):
            if elapsed >= 200_000_000:
                raise RuntimeError("poll failed")
            return False
        outputs = []
        with self.assertRaisesRegex(RuntimeError, "poll failed"):
            self._run(pulses=2, poll=fail_after_on, set_pump=outputs.append)
        # The failure occurs at the delay boundary, before the first output.
        self.assertEqual(outputs, [])
        calls = 0
        def fail_when_pumping(elapsed):
            nonlocal calls
            if elapsed >= 200_000_000:
                calls += 1
                if calls == 2:
                    raise RuntimeError("poll failed")
            return False
        with self.assertRaisesRegex(RuntimeError, "poll failed"):
            self._run(pulses=2, poll=fail_when_pumping, set_pump=outputs.append)
        self.assertEqual(outputs, [1, 0])

    def test_zero_duration_preserves_actual_flip_without_noop_hardware_events(self):
        for timeout in (False, True):
            with self.subTest(timeout=timeout):
                end, _, edges, logger = self._run(pulses=0, timeout=timeout,
                                                 reward_interval_s=0, timeout_duration_s=0)
                self.assertEqual(end, 10_000_000_000)
                self.assertEqual(edges, [])
                logger.log_frame_flip.assert_called_once_with(
                    trial_num=7, event="timeout_interval" if timeout else "reward_interval",
                    timestamp_perf_ns=10_000_000_000, requested_timestamp_perf_ns=9_999_999_900,
                    requested_duration=0,
                )
                logger.log_event.assert_not_called()
                logger.log_signal.assert_not_called()

    def test_duration_uses_largest_configured_reward_and_validates_timings(self):
        kwargs = dict(pump_delay_time=0.2, pump_pulse_time_seconds=0.25,
                      inter_pump_interval=0.1, reward_to_pulse_map={"0": 0, "1": 0, "2": 1, "3": 2})
        self.assertAlmostEqual(reward_interval_duration(**kwargs), 0.8)
        self.assertAlmostEqual(reward_interval_duration(**{**kwargs, "reward_to_pulse_map": {"4": 3}}), 1.15)
        for name in ("pump_delay_time", "pump_pulse_time_seconds", "inter_pump_interval"):
            for value in (-1, float("nan"), float("inf")):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    reward_interval_duration(**{**kwargs, name: value})
        for value in (-1, 1.5, True, "2"):
            with self.subTest(count=value), self.assertRaises(ValueError):
                reward_interval_duration(**{**kwargs, "reward_to_pulse_map": {"3": value}})


class ActiveForagingTimingTests(unittest.TestCase):
    def test_duration_must_be_positive_for_memory_even_when_simultaneous(self):
        self.assertTrue(duration_requires_visible_phase(sequential=False, is_memory=True))
        validate_duration_for_presentation_mode(0.05, sequential=False, is_memory=True)

        with self.assertRaises(ValueError):
            validate_duration_for_presentation_mode(0.0, sequential=False, is_memory=True)

    def test_duration_stays_zero_for_simultaneous_non_memory(self):
        self.assertFalse(duration_requires_visible_phase(sequential=False, is_memory=False))
        validate_duration_for_presentation_mode(0.0, sequential=False, is_memory=False)

        with self.assertRaises(ValueError):
            validate_duration_for_presentation_mode(0.05, sequential=False, is_memory=False)

    def test_duration_must_be_positive_for_sequential(self):
        self.assertTrue(duration_requires_visible_phase(sequential=True, is_memory=False))
        self.assertTrue(duration_requires_visible_phase(sequential=True, is_memory=True))

    def test_duration_must_be_finite(self):
        with self.assertRaises(ValueError):
            validate_duration_for_presentation_mode(
                float("nan"), sequential=True, is_memory=True
            )

        with self.assertRaises(ValueError):
            validate_duration_for_presentation_mode(0.0, sequential=True, is_memory=False)


if __name__ == "__main__":
    unittest.main()
