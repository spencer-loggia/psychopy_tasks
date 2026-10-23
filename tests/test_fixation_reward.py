"""Full-screen fixation acceptance uses the calibration reward time windows."""

import unittest

from bin.eye_tracking import EyeCalibration, EyeTrackerState
from bin.eye_calibration_reward import CalibrationAutoReward
from bin.fixation_reward import ScreenAutoReward


class ScreenAutoRewardTests(unittest.TestCase):
    def make_policy(self, calibration=None):
        policy = ScreenAutoReward(
            reward_interval=1.0,
            calibration=calibration or EyeCalibration(x_scale=1, y_scale=1),
            max_sample_age_s=0.15,
        )
        policy.toggle(now_s=0)
        return policy

    def feed(self, policy, start_ms=0, end_ms=1000, *, position=(0, 0), misses=()):
        for ms in range(start_ms, end_ms + 1):
            x, y = (1, 0) if ms in misses else position
            policy.observe(EyeTrackerState(
                timestamp_perf_s=ms / 1000, frame_sequence=ms + 1,
                x_smooth_px=x, y_smooth_px=y, valid=True,
            ))

    def test_entire_rectangle_is_accepted_but_offscreen_and_nonfinite_are_not(self):
        for x, y in ((0, 0), (-0.5, -0.5), (-0.5, 0.5), (0.5, -0.5), (0.5, 0.5),
                     (0.5001, 0), (-0.5001, 0), (0, 0.5001), (0, -0.5001),
                     (float("nan"), 0), (0, float("inf"))):
            with self.subTest(position=(x, y)):
                policy = self.make_policy()
                self.feed(policy, position=(x, y))
                self.assertEqual(
                    policy.status(now_s=1).due,
                    -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5,
                )

    def test_acceptance_uses_unclamped_affine_calibration_in_subject_coordinates(self):
        calibration = EyeCalibration(
            x_scale=0, x_cross=0.01, x_offset=-1,
            y_scale=0, y_cross=-0.02, y_offset=1,
        )
        for raw, expected in (((50, 100), True), ((25, 150), True), ((50, 151), False)):
            with self.subTest(raw=raw):
                policy = self.make_policy(calibration)
                self.feed(policy, position=raw)
                self.assertEqual(policy.status(now_s=1).due, expected)

    def test_both_strict_time_thresholds_match_calibration(self):
        for misses, expected in ((range(950, 954), True), (range(950, 955), False),
                                 (range(200, 399), True), (range(200, 400), False)):
            with self.subTest(misses=misses):
                policy = self.make_policy()
                self.feed(policy, misses=misses)
                self.assertEqual(policy.status(now_s=1).due, expected)

    def test_missing_current_gaze_blocks_even_with_qualifying_history(self):
        for reason in ("blink", "lost", "stale", "uart_error", "service_restarted"):
            with self.subTest(reason=reason):
                policy = self.make_policy()
                self.feed(policy)
                policy.observe(EyeTrackerState(
                    timestamp_perf_s=1, frame_sequence=1001,
                    valid=False, rejection_reason=reason,
                ))
                self.assertFalse(policy.status(now_s=1).due)
        policy = self.make_policy()
        self.assertFalse(policy.status(now_s=10).due)  # No tracker samples.
        self.feed(policy)
        self.assertFalse(policy.status(now_s=1.101).due)  # Acquisition stopped.
        self.feed(policy, 2000, 2000)
        self.assertFalse(policy.status(now_s=2).due)  # Rewarm after a gap.
        self.feed(policy, 2001, 3000)
        self.assertTrue(policy.status(now_s=3).due)

    def test_duplicate_samples_and_current_offscreen_gaze_cannot_earn_rewards(self):
        policy = self.make_policy()
        for _ in range(200):
            self.feed(policy, 0, 0)
        self.assertFalse(policy.status(now_s=1).due)
        self.feed(policy)
        self.feed(policy, 1001, 1001, position=(0.6, 0))
        self.assertFalse(policy.status(now_s=1.001).due)


class RollingRewardTests(unittest.TestCase):
    def make_policies(self):
        settings = dict(
            reward_interval=1.0, calibration=EyeCalibration(x_scale=1, y_scale=1),
            max_sample_age_s=0.15,
        )
        policies = [
            ScreenAutoReward(**settings),
            CalibrationAutoReward(
                **settings, fixation=(0, 0), screen_size=(1000, 500), fix_diameter=0.1,
            ),
        ]
        for policy in policies:
            policy.toggle(now_s=0)
        return policies

    def feed(self, policy, start_ms, end_ms, *, blinks=()):
        for ms in range(start_ms, end_ms + 1):
            policy.observe(EyeTrackerState(
                timestamp_perf_s=ms / 1000, frame_sequence=ms + 1,
                valid=ms not in blinks,
                rejection_reason="blink" if ms in blinks else None,
            ))

    def test_blink_delays_reward_only_until_the_rolling_windows_qualify(self):
        for policy in self.make_policies():
            with self.subTest(mode=type(policy).__name__):
                # A 300 ms blink prevents reward at 1 s. Its contribution then
                # shrinks continuously as it leaves the one-second window.
                self.feed(policy, 0, 1000, blinks=range(200, 500))
                self.assertFalse(policy.status(now_s=1).due)
                self.feed(policy, 1001, 1300)
                status = policy.status(now_s=1.3)
                self.assertAlmostEqual(status.interval_fraction, 0.80)
                self.assertFalse(status.due)  # Exactly 80% is insufficient.
                self.feed(policy, 1301, 1301)
                self.assertTrue(policy.status(now_s=1.301).due)
                policy.reward_delivered(now_s=1.301)
                self.feed(policy, 1302, 2301)
                self.assertFalse(policy.status(now_s=2.301).due)
                self.feed(policy, 2302, 2302)
                self.assertTrue(policy.status(now_s=2.302).due)

    def test_reward_reset_excludes_replayed_pre_reward_samples(self):
        for policy in self.make_policies():
            with self.subTest(mode=type(policy).__name__):
                self.feed(policy, 0, 1000)
                self.assertTrue(policy.status(now_s=1.05).due)
                policy.reward_delivered(now_s=1.05)
                # The UART can repeat the last camera sample after a reward.
                self.feed(policy, 1000, 1000)
                status = policy.status(now_s=1.05)
                self.assertFalse(status.due)
                self.assertEqual((status.recent_fraction, status.interval_fraction), (0, 0))
                self.feed(policy, 1051, 2050)
                self.assertFalse(policy.status(now_s=2.05).due)
                self.feed(policy, 2051, 2051)
                self.assertTrue(policy.status(now_s=2.051).due)


if __name__ == "__main__":
    unittest.main()
