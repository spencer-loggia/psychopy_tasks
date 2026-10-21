"""Deterministic checks of both time windows and automatic reward scheduling."""

import unittest

from bin.eye_calibration_reward import CalibrationAutoReward
from bin.eye_tracking import EyeCalibration, EyeTrackerState


class CalibrationAutoRewardTests(unittest.TestCase):
    def make_policy(self):
        return CalibrationAutoReward(
            reward_interval=1.0, calibration=EyeCalibration(x_scale=1, y_scale=1),
            fixation=(0, 0), screen_size=(1000, 500), fix_diameter=0.1,
            max_sample_age_s=0.15,
        )

    def feed(self, policy, start_ms, end_ms, *, misses=(), invalid_reason=None):
        for ms in range(start_ms, end_ms + 1):
            miss = ms in misses
            policy.observe(EyeTrackerState(
                timestamp_perf_s=ms / 1000, frame_sequence=ms + 1,
                x_smooth_px=1 if miss else 0, y_smooth_px=0,
                valid=not (miss and invalid_reason),
                rejection_reason=invalid_reason if miss else None,
            ))

    def test_mode_starts_off_and_toggle_requires_a_full_new_interval(self):
        policy = self.make_policy()
        self.feed(policy, 0, 1000)
        self.assertFalse(policy.status(now_s=1).due)
        self.assertTrue(policy.toggle(now_s=1))
        self.feed(policy, 1001, 1500)
        self.assertFalse(policy.status(now_s=1.5).due)
        self.feed(policy, 1501, 2001)
        self.assertTrue(policy.status(now_s=2.001).due)
        self.assertFalse(policy.toggle(now_s=2.001))
        self.feed(policy, 2002, 3002)
        self.assertFalse(policy.status(now_s=3.002).due)

    def test_recent_window_must_be_strictly_above_95_percent(self):
        for missing_ms, expected in ((4, True), (5, False), (6, False)):
            with self.subTest(missing_ms=missing_ms):
                policy = self.make_policy()
                policy.toggle(now_s=0)
                self.feed(policy, 0, 1000, misses=range(950, 950 + missing_ms))
                status = policy.status(now_s=1)
                self.assertAlmostEqual(status.recent_fraction, 1 - missing_ms / 100)
                self.assertGreater(status.interval_fraction, 0.85)
                self.assertEqual(status.due, expected)

    def test_interval_window_must_be_strictly_above_85_percent_including_blinks(self):
        for missing_ms, expected in ((149, True), (150, False), (151, False)):
            for reason in (None, "blink", "lost"):
                with self.subTest(missing_ms=missing_ms, reason=reason):
                    policy = self.make_policy()
                    policy.toggle(now_s=0)
                    self.feed(
                        policy, 0, 1000, misses=range(200, 200 + missing_ms),
                        invalid_reason=reason,
                    )
                    status = policy.status(now_s=1)
                    self.assertAlmostEqual(status.interval_fraction, 1 - missing_ms / 1000)
                    self.assertAlmostEqual(status.recent_fraction, 1)
                    self.assertEqual(status.due, expected)

    def test_continuous_fixation_repeats_without_clearing_history_after_pulses(self):
        policy = self.make_policy()
        policy.toggle(now_s=0)
        self.feed(policy, 0, 1000)
        self.assertTrue(policy.status(now_s=1).due)
        policy.reward_delivered(now_s=1)
        # Acquisition continues while the main thread is occupied by the pump.
        self.feed(policy, 1001, 1999)
        self.assertFalse(policy.status(now_s=1.999).due)
        self.feed(policy, 2000, 2000)
        self.assertTrue(policy.status(now_s=2).due)
        policy.reward_delivered(now_s=2)
        self.feed(policy, 2001, 2400)
        policy.reward_delivered(now_s=2.4)  # Manual R also spaces later rewards.
        self.feed(policy, 2401, 3000)
        self.assertFalse(policy.status(now_s=3).due)
        self.feed(policy, 3001, 3400)
        self.assertTrue(policy.status(now_s=3.4).due)

    def test_failed_check_waits_for_fixation_without_catchup_rewards(self):
        policy = self.make_policy()
        policy.toggle(now_s=0)
        self.feed(policy, 0, 1000, misses=range(980, 1001))
        self.assertFalse(policy.status(now_s=1).due)
        self.feed(policy, 1001, 1101)
        self.assertTrue(policy.status(now_s=1.101).due)
        policy.reward_delivered(now_s=1.101)
        self.feed(policy, 1102, 2000)
        self.assertFalse(policy.status(now_s=2).due)

    def test_current_loss_expired_data_gaps_and_mapping_changes_block_reward(self):
        for reason in ("blink", "lost", "stale", "uart_error", "service_restarted"):
            with self.subTest(reason=reason):
                policy = self.make_policy()
                policy.toggle(now_s=0)
                self.feed(policy, 0, 1000)
                policy.observe(EyeTrackerState(
                    timestamp_perf_s=1, frame_sequence=1001,
                    valid=False, rejection_reason=reason,
                ))
                self.assertFalse(policy.status(now_s=1).due)
        policy = self.make_policy()
        policy.toggle(now_s=0)
        self.feed(policy, 0, 1000)
        self.assertFalse(policy.status(now_s=1.101).due)
        self.feed(policy, 2000, 2000)  # A long observation gap requires rewarming.
        self.assertFalse(policy.status(now_s=2).due)
        self.feed(policy, 2001, 3000)
        self.assertTrue(policy.status(now_s=3).due)
        policy.set_mapping(EyeCalibration(), (0, 0), now_s=3)
        self.assertFalse(policy.status(now_s=3).due)

    def test_duplicate_frames_do_not_manufacture_fixation_history(self):
        policy = self.make_policy()
        policy.toggle(now_s=0)
        state = EyeTrackerState(timestamp_perf_s=0, frame_sequence=1, valid=True)
        for _ in range(200):
            policy.observe(state)
        self.assertFalse(policy.status(now_s=1).due)
        self.assertEqual(len(policy._interval.history), 1)


if __name__ == "__main__":
    unittest.main()
