import math
import unittest
from unittest.mock import Mock, patch

from bin.frame_timing import (
    FlipTimestamps,
    FrameTransitionScheduler,
    arm_flip_timestamp,
    flip_submission_time,
    flip_with_timestamps,
    plan_frame_duration,
    validate_requested_durations,
    wait_until,
)


class FrameDurationPlanTests(unittest.TestCase):
    def test_arbitrary_duration_uses_nearest_refresh_and_keeps_request(self):
        plan = plan_frame_duration(0.052, 120.0, minimum_frames=1)

        self.assertEqual(plan.requested_s, 0.052)
        self.assertEqual(plan.frame_count, 6)
        self.assertAlmostEqual(plan.scheduled_s, 0.05)
        self.assertAlmostEqual(plan.error_s, -0.002)

    def test_half_frame_tie_rounds_up(self):
        self.assertEqual(plan_frame_duration(0.025, 60.0).frame_count, 2)

    def test_positive_visible_phase_can_be_clamped_to_one_frame(self):
        plan = plan_frame_duration(0.001, 59.94, minimum_frames=1)

        self.assertEqual(plan.frame_count, 1)
        self.assertAlmostEqual(plan.scheduled_s, 1.0 / 59.94)

    def test_rejects_invalid_values(self):
        for requested in (-0.1, math.nan, math.inf):
            with self.subTest(requested=requested):
                with self.assertRaises(ValueError):
                    plan_frame_duration(requested, 60.0)
        for fps in (0.0, -60.0, math.nan, math.inf):
            with self.subTest(fps=fps):
                with self.assertRaises(ValueError):
                    plan_frame_duration(0.1, fps)

    def test_semantic_validation_does_not_require_frame_multiples(self):
        validate_requested_durations(
            {"duration": 0.052, "isi": 0.003},
            positive={"duration"},
        )

        with self.assertRaises(ValueError):
            validate_requested_durations(
                {"choice_time": 0.0}, positive={"choice_time"}
            )


class FlipTimestampTests(unittest.TestCase):
    def test_flip_capture_precedes_slow_gpio_callback_and_keeps_native_ns(self):
        now = 9_007_199_254_740_993

        class Window:
            def __init__(self):
                self.callbacks = []

            def callOnFlip(self, callback, *args):
                self.callbacks.append((callback, args))

            def flip(self):
                nonlocal now
                now += 10_000_000
                for callback, args in self.callbacks:
                    callback(*args)
                now += 5_000_000  # Work after callbacks, before flip returns.

        def slow_gpio_write():
            nonlocal now
            now += 2_000_000

        win = Window()
        request_ns = now
        with patch("bin.frame_timing.time.perf_counter_ns", side_effect=lambda: now):
            arm_flip_timestamp(win)
            win.callOnFlip(slow_gpio_write)
            result = flip_with_timestamps(win)

        self.assertEqual(result.requested_perf_ns, request_ns)
        self.assertEqual(result.actual_perf_ns, request_ns + 10_000_000)
        self.assertEqual(now, result.actual_perf_ns + 7_000_000)
        self.assertEqual(len(win.callbacks), 2)

    def test_captures_request_before_realized_flip(self):
        class Window:
            def flip(self):
                return 42.0

        with patch("bin.frame_timing.time.perf_counter_ns", side_effect=[10_000_000_000, 10_010_000_000]):
            result = flip_with_timestamps(Window())

        self.assertEqual(result.psychopy_s, 42.0)
        self.assertEqual(result.requested_perf_s, 10.0)
        self.assertEqual(result.actual_perf_s, 10.01)

    def test_uses_call_on_flip_for_the_realized_perf_timestamp(self):
        class Window:
            def callOnFlip(self, callback, *args):
                self.callback = callback
                self.callback_args = args

            def flip(self):
                self.callback(*self.callback_args)
                return 42.0

        with patch(
            "bin.frame_timing.time.perf_counter_ns",
            side_effect=[10_000_000_000, 10_005_000_000, 10_010_000_000],
        ):
            result = flip_with_timestamps(Window())

        self.assertEqual(result.requested_perf_s, 10.0)
        self.assertEqual(result.actual_perf_s, 10.005)


class SparseTransitionTimingTests(unittest.TestCase):
    def test_submission_precedes_target_by_just_under_half_a_frame(self):
        self.assertAlmostEqual(
            flip_submission_time(10.0, 1.0 / 60.0),
            10.0 - 0.49 / 60.0,
        )

    def test_wait_until_sleeps_once_without_a_poll_callback(self):
        waits = []

        with patch(
            "bin.frame_timing.time.perf_counter",
            side_effect=[9.0, 9.5],
        ):
            completed = wait_until(9.5, wait_fn=waits.append)

        self.assertTrue(completed)
        self.assertEqual(waits, [0.5])

    def test_wait_until_can_be_cancelled_while_polling(self):
        completed = wait_until(10.0, poll_callback=lambda: True)

        self.assertFalse(completed)

    def test_scheduler_counts_only_transitions_outside_tolerance(self):
        scheduler = FrameTransitionScheduler(1.0 / 60.0)

        scheduler.record(1.0, 1.004)
        scheduler.record(2.0, 2.02)

        self.assertEqual(scheduler.transition_count, 2)
        self.assertEqual(scheduler.missed_transition_count, 1)
        self.assertAlmostEqual(scheduler.maximum_absolute_error_s, 0.02)
        self.assertAlmostEqual(scheduler.maximum_lateness_s, 0.02)

    def test_scheduler_flips_once_at_a_static_phase_boundary(self):
        scheduler = FrameTransitionScheduler(1.0 / 60.0)
        timing = FlipTimestamps(1.0, 9.99, 10.0)
        before_flip = Mock()

        with patch(
            "bin.frame_timing.wait_until_flip_submission",
            return_value=True,
        ), patch(
            "bin.frame_timing.flip_with_timestamps",
            return_value=timing,
        ) as flip:
            result = scheduler.flip_at(
                object(),
                10.0,
                before_flip_callback=before_flip,
            )

        self.assertIs(result, timing)
        before_flip.assert_called_once_with()
        flip.assert_called_once()
        self.assertEqual(scheduler.transition_count, 1)
        self.assertEqual(scheduler.missed_transition_count, 0)


if __name__ == "__main__":
    unittest.main()
