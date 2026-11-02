"""Full-screen fixation acceptance uses the calibration reward time windows."""

import unittest
import threading

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

    def test_interval_changes_require_a_full_new_interval_and_ignore_old_samples(self):
        for new_interval in (0.3, 2.0):
            for policy in self.make_policies():
                with self.subTest(mode=type(policy).__name__, interval=new_interval):
                    self.feed(policy, 0, 1000)
                    self.assertTrue(policy.status(now_s=1).due)
                    self.assertTrue(policy.set_reward_interval(new_interval, now_s=1.05))
                    self.assertEqual(policy.reward_interval, new_interval)
                    self.feed(policy, 1000, 1000)  # An in-flight old UART sample.
                    status = policy.status(now_s=1.05)
                    self.assertTrue(status.enabled)
                    self.assertFalse(status.due)
                    self.assertEqual((status.recent_fraction, status.interval_fraction), (0, 0))
                    interval_ms = round(new_interval * 1000)
                    self.feed(policy, 1051, 1050 + interval_ms)
                    self.assertFalse(policy.status(now_s=(1050 + interval_ms) / 1000).due)
                    self.feed(policy, 1051 + interval_ms, 1051 + interval_ms)
                    self.assertTrue(policy.status(now_s=(1051 + interval_ms) / 1000).due)

    def test_unchanged_and_invalid_intervals_preserve_history(self):
        for policy in self.make_policies():
            with self.subTest(mode=type(policy).__name__):
                self.feed(policy, 0, 1000)
                self.assertFalse(policy.set_reward_interval(1.0, now_s=1))
                for invalid in (0, -1, float("nan"), float("inf"), "invalid", None):
                    with self.subTest(interval=invalid):
                        with self.assertRaisesRegex(ValueError, "reward_interval"):
                            policy.set_reward_interval(invalid, now_s=1)
                        self.assertEqual(policy.reward_interval, 1.0)
                        self.assertTrue(policy.status(now_s=1).due)
                with self.assertRaisesRegex(ValueError, "now_s"):
                    policy.set_reward_interval(0.3, now_s=float("nan"))
                self.assertTrue(policy.status(now_s=1).due)

    def test_interval_change_clears_an_observation_already_in_flight(self):
        entered = threading.Event()
        release = threading.Event()
        changed = threading.Event()

        class BlockingPolicy(ScreenAutoReward):
            def _in_window(self, position):
                entered.set()
                release.wait(timeout=2)
                return super()._in_window(position)

        policy = BlockingPolicy(
            reward_interval=1, calibration=EyeCalibration(), max_sample_age_s=0.15,
        )
        sample_thread = threading.Thread(target=policy.observe, args=(EyeTrackerState(
            timestamp_perf_s=1, frame_sequence=1, valid=True,
        ),))

        def update():
            policy.set_reward_interval(0.3, now_s=1.1)
            changed.set()

        update_thread = threading.Thread(target=update)
        sample_thread.start()
        try:
            self.assertTrue(entered.wait(timeout=2))
            update_thread.start()
            self.assertFalse(changed.is_set())
        finally:
            release.set()
            sample_thread.join(timeout=2)
            if update_thread.ident is not None:
                update_thread.join(timeout=2)
        self.assertTrue(changed.is_set())
        self.assertFalse(sample_thread.is_alive())
        self.assertFalse(update_thread.is_alive())
        status = policy.status(now_s=1.1)
        self.assertEqual((status.recent_fraction, status.interval_fraction), (0, 0))

    def test_try_configure_returns_without_waiting_or_publishing_when_lock_is_busy(self):
        for policy in self.make_policies():
            with self.subTest(mode=type(policy).__name__):
                self.feed(policy, 0, 1000)
                completed = threading.Event()
                results, callbacks = [], []
                settings = {"reward_interval": 0.3}
                if isinstance(policy, CalibrationAutoReward):
                    settings["fixation_radius"] = 0.1

                def update():
                    results.append(policy.try_configure(
                        **settings, now_s=1.05, before_update=lambda: callbacks.append(True),
                    ))
                    completed.set()

                worker = threading.Thread(target=update)
                policy._lock.acquire()
                try:
                    worker.start()
                    self.assertTrue(completed.wait(timeout=2), "Setting edit waited for acquisition")
                    self.assertEqual(results, [None])
                    self.assertEqual(callbacks, [])
                    self.assertEqual(policy.reward_interval, 1)
                    if isinstance(policy, CalibrationAutoReward):
                        self.assertEqual(policy.fixation_radius, 0.05)
                    with self.assertRaisesRegex(ValueError, "reward_interval"):
                        policy.try_configure(reward_interval=0, now_s=1)
                finally:
                    policy._lock.release()
                    worker.join(timeout=2)
                self.assertFalse(worker.is_alive())
                self.assertTrue(policy.status(now_s=1).due)

    def test_try_configure_publishes_before_mutation_and_preserves_unchanged_history(self):
        for policy in self.make_policies():
            with self.subTest(mode=type(policy).__name__):
                self.feed(policy, 0, 1000)
                history = list(policy._interval.history)
                published = []

                def publish():
                    published.append(policy.reward_interval)
                    self.assertEqual(list(policy._interval.history), history)
                    if isinstance(policy, CalibrationAutoReward):
                        self.assertEqual(policy.fixation_radius, 0.05)
                    acquired = policy._lock.acquire(False)
                    if acquired:
                        policy._lock.release()
                    self.assertFalse(acquired, "Publication must hold the policy lock")

                self.assertFalse(policy.try_configure(
                    reward_interval=1, now_s=1, before_update=publish,
                ))
                self.assertTrue(policy.status(now_s=1).due)
                settings = {"reward_interval": 0.3}
                if isinstance(policy, CalibrationAutoReward):
                    settings["fixation_radius"] = 0.1
                self.assertTrue(policy.try_configure(
                    **settings, now_s=1.05, before_update=publish,
                ))
                self.assertEqual(published, [1, 1])
                self.assertEqual(policy.reward_interval, 0.3)
                if isinstance(policy, CalibrationAutoReward):
                    self.assertEqual(policy.fixation_radius, 0.1)
                self.assertFalse(policy.status(now_s=1.05).due)
                self.assertEqual(len(policy._interval.history), 0)

    def test_try_configure_callback_errors_leave_policy_unchanged_and_unlock(self):
        for policy in self.make_policies():
            with self.subTest(mode=type(policy).__name__):
                self.feed(policy, 0, 1000)

                def fail_publish():
                    raise RuntimeError("publication failed")

                settings = {"reward_interval": 0.3}
                if isinstance(policy, CalibrationAutoReward):
                    settings["fixation_radius"] = 0.1
                with self.assertRaisesRegex(RuntimeError, "publication failed"):
                    policy.try_configure(**settings, now_s=1.05, before_update=fail_publish)
                acquired = policy._lock.acquire(False)
                self.assertTrue(acquired)
                policy._lock.release()
                self.assertEqual(policy.reward_interval, 1)
                if isinstance(policy, CalibrationAutoReward):
                    self.assertEqual(policy.fixation_radius, 0.05)
                self.assertTrue(policy.status(now_s=1).due)


if __name__ == "__main__":
    unittest.main()
