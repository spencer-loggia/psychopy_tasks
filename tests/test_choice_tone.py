import threading
import time
import unittest
from unittest.mock import Mock, patch

from bin.choice_tone import ChoicePeriodTone, validate_choice_tone
from bin.config import validate_config
from bin.config_snapshot import build_task_config_snapshot
from tests.test_active_foraging_timing import FakeClock


class ManualTimer:
    def __init__(self, interval, function, args=()):
        self.interval, self.function, self.args = interval, function, args
        self.started = self.cancelled = False

    def start(self):
        self.started = True

    def cancel(self):
        self.cancelled = True

    def fire(self):
        # Also exercise a callback already dispatched when cancel() was called.
        self.function(*self.args)


class ChoiceToneTests(unittest.TestCase):
    def setUp(self):
        self.clock, self.edges, self.timers = FakeClock(), [], []
        self.tone = ChoicePeriodTone(lambda active: self.edges.append((active, self.clock.ns)))
        self.logger = Mock()
        self.clock_patch = patch("bin.choice_tone.time.perf_counter_ns", side_effect=lambda: self.clock.ns)
        self.timer_patch = patch("bin.choice_tone.threading.Timer", side_effect=self.make_timer)
        self.clock_patch.start()
        self.timer_patch.start()
        self.addCleanup(self.clock_patch.stop)
        self.addCleanup(self.timer_patch.stop)
        self.addCleanup(self.tone.stop)

    def make_timer(self, *args, **kwargs):
        timer = ManualTimer(*args, **kwargs)
        self.timers.append(timer)
        return timer

    def test_pulse_returns_immediately_and_logs_50_ms_edges_on_main_thread(self):
        self.tone.start(7)
        self.assertEqual(self.clock.ns, 10_000_000_000)
        self.assertEqual(self.edges, [(True, self.clock.ns)])
        self.assertEqual(self.timers[0].interval, 0.050)
        self.assertTrue(self.timers[0].started)
        self.assertTrue(self.timers[0].daemon)
        self.logger.log_signal.assert_not_called()
        self.clock.sleep(0.050)
        self.timers[0].fire()
        self.tone.flush_events(self.logger)
        self.assertEqual(self.edges, [(True, 10_000_000_000), (False, 10_050_000_000)])
        rows = [call.kwargs for call in self.logger.log_signal.call_args_list]
        self.assertEqual([row["event"] for row in rows], ["buzzer_on", "buzzer_off"])
        self.assertEqual([row["trial_num"] for row in rows], [7, 7])
        self.assertEqual(rows[0]["requested_duration"], 0.050)
        self.assertEqual(rows[1]["timestamp_perf_ns"] - rows[0]["timestamp_perf_ns"], 50_000_000)
        self.tone.stop()
        self.assertEqual(len(self.edges), 2)

    def test_cancelled_timer_cannot_clear_timeout_or_a_subsequent_trial(self):
        self.tone.start(1)
        old = self.timers[0]
        self.clock.sleep(0.010)
        self.tone.stop()
        self.edges.append((True, self.clock.ns))  # Timeout now owns the buzzer.
        old.fire()
        self.assertTrue(self.edges[-1][0])
        self.assertEqual(len(self.edges), 3)
        self.tone.start(2)
        old.fire()
        self.assertTrue(self.edges[-1][0])
        self.assertEqual(len(self.edges), 4)
        self.clock.sleep(0.050)
        self.timers[-1].fire()
        self.assertFalse(self.edges[-1][0])

    def test_timer_failure_clears_buzzer_and_background_error_reaches_main_thread(self):
        with patch("bin.choice_tone.threading.Timer", side_effect=RuntimeError("timer failed")):
            with self.assertRaisesRegex(RuntimeError, "timer failed"):
                self.tone.start(1)
        self.assertEqual([active for active, _ in self.edges], [True, False])
        self.tone.start(2)
        with patch.object(self.tone, "_set_buzzer", side_effect=RuntimeError("write failed")):
            self.timers[-1].fire()
        with self.assertRaisesRegex(RuntimeError, "Choice-period buzzer failed: write failed"):
            self.tone.flush_events(self.logger)
        self.tone.stop()
        self.assertFalse(self.edges[-1][0])


class ChoiceToneConfigurationTests(unittest.TestCase):
    def test_boolean_validation_and_snapshot_preservation(self):
        for value in (False, True):
            self.assertIs(validate_choice_tone(value), value)
            cfg = {"choice_period_idicator_tone": value}
            validate_config(cfg)
            for task in ("active_foraging", "match2cue"):
                self.assertEqual(build_task_config_snapshot(cfg, launch_path=f"task/{task}.py"), cfg)
        for value in (0, 1, "true", "false", None, {}, []):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "must be true or false"):
                validate_config({"choice_period_idicator_tone": value})

class ChoiceToneThreadTests(unittest.TestCase):
    def test_real_timer_turns_buzzer_off_without_task_polling(self):
        finished, edges = threading.Event(), []

        def output(active):
            edges.append((active, time.perf_counter()))
            if not active:
                finished.set()

        tone = ChoicePeriodTone(output)
        try:
            tone.start(1)
            self.assertTrue(finished.wait(1.0))
            self.assertEqual([active for active, _ in edges], [True, False])
            self.assertGreaterEqual(edges[1][1] - edges[0][1], 0.050)
        finally:
            tone.stop()
