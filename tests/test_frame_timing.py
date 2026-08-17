import math
import unittest
from unittest.mock import patch

from bin.frame_timing import (
    flip_with_timestamps,
    plan_frame_duration,
    validate_requested_durations,
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
    def test_captures_request_before_realized_flip(self):
        class Window:
            def flip(self):
                return 42.0

        with patch("bin.frame_timing.time.perf_counter", side_effect=[10.0, 10.01]):
            result = flip_with_timestamps(Window())

        self.assertEqual(result.psychopy_s, 42.0)
        self.assertEqual(result.requested_perf_s, 10.0)
        self.assertEqual(result.actual_perf_s, 10.01)


if __name__ == "__main__":
    unittest.main()
