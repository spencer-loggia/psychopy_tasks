import unittest
from unittest.mock import patch

from bin.touch_input import (
    AFC_HOLD_CONFIRMATION_WINDOW_S,
    AFC_HOLD_MAX_BREAK_S,
    MousePressSample,
    MousePressTracker,
    TouchHoldTracker,
)


class FakeMouse:
    def __init__(self, samples):
        self.samples = list(samples)
        self.current_position = (0.0, 0.0)
        self.click_reset_count = 0

    def getPressed(self, getTime=False):
        buttons, times, position = self.samples.pop(0)
        self.current_position = position
        if getTime:
            return list(buttons), list(times)
        return list(buttons)

    def getPos(self):
        return self.current_position

    def clickReset(self):
        self.click_reset_count += 1


class MousePressTrackerTests(unittest.TestCase):
    def test_buffered_press_preserves_event_time_despite_delayed_poll_and_position(self):
        now = 9_007_199_254_740_993
        origin = now

        class DelayedMouse(FakeMouse):
            def clickReset(self):
                nonlocal now
                now += 200  # Reset's midpoint anchors PsychoPy's elapsed times.
                super().clickReset()

            def getPos(self):
                nonlocal now
                now += 100_000_000
                return super().getPos()

        mouse = DelayedMouse([
            ((0, 0, 0), (0, 0, 0), (0, 0)),
            ((0, 0, 0), (0.012, 0, 0), (5, 6)),
        ])
        tracker = MousePressTracker(mouse)
        with patch("bin.touch_input.time.perf_counter_ns", side_effect=lambda: now):
            tracker.reset()
            now = origin + 50_000_000
            sample = tracker.poll()

        self.assertEqual(sample.timestamp_perf_ns, origin + 100 + 12_000_000)
        self.assertEqual(sample.observed_perf_ns, origin + 50_000_000)
        self.assertTrue(sample.press_started)
        self.assertGreater(now, sample.observed_perf_ns + 100_000_000)

    def test_missing_or_invalid_event_time_uses_detection_before_position_lookup(self):
        for press_time in (0, float("nan"), float("inf"), -1, 1000):
            with self.subTest(press_time=press_time):
                mouse = FakeMouse([((1, 0, 0), (press_time, 0, 0), (5, 6))])
                tracker = MousePressTracker(mouse)
                tracker._reset_perf_ns = 1_000_000_000
                with patch("bin.touch_input.time.perf_counter_ns", return_value=2_000_000_003):
                    sample = tracker.poll()
                self.assertEqual(sample.timestamp_perf_ns, 2_000_000_003)
                self.assertFalse(sample.press_timestamp_valid)

    def test_contact_evidence_expires_and_never_uses_cursor_or_fallback_time(self):
        def sample(**kwargs):
            return MousePressSample(**dict(dict(
                position=(0, 0), buttons=(False, False, False),
                press_started=True, buffered_press=True, press_timestamp_valid=True,
                timestamp_perf_ns=1_000_000_000, observed_perf_ns=1_050_000_000,
            ), **kwargs))

        self.assertEqual(sample().contact_time_ns(max_age_s=.05), 1_000_000_000)
        for changes in (
            dict(observed_perf_ns=1_050_000_001), dict(press_started=False),
            dict(press_timestamp_valid=False), dict(timestamp_perf_ns=1_060_000_000),
        ):
            with self.subTest(changes=changes):
                self.assertIsNone(sample(**changes).contact_time_ns(max_age_s=.05))
        self.assertEqual(sample(buttons=(True, False, False)).contact_time_ns(max_age_s=.05),
                         1_050_000_000)

    def test_completed_tap_is_reported_from_buffered_press_time(self):
        mouse = FakeMouse(
            [
                ((0, 0, 0), (0.0, 0.0, 0.0), (0, 0)),
                ((0, 0, 0), (0.012, 0.0, 0.0), (125, -75)),
                ((0, 0, 0), (0.0, 0.0, 0.0), (125, -75)),
            ]
        )
        tracker = MousePressTracker(mouse)

        tracker.reset()
        sample = tracker.poll()

        self.assertFalse(sample.down)
        self.assertTrue(sample.press_started)
        self.assertTrue(sample.buffered_press)
        self.assertTrue(sample.active)
        self.assertEqual(sample.position, (125.0, -75.0))
        self.assertFalse(tracker.poll().press_started)
        self.assertEqual(mouse.click_reset_count, 2)

    def test_held_press_has_one_edge_but_remains_active(self):
        mouse = FakeMouse(
            [
                ((0, 0, 0), (0.0, 0.0, 0.0), (0, 0)),
                ((1, 0, 0), (0.004, 0.0, 0.0), (20, 30)),
                ((1, 0, 0), (0.0, 0.0, 0.0), (22, 31)),
            ]
        )
        tracker = MousePressTracker(mouse)
        tracker.reset()

        first = tracker.poll()
        second = tracker.poll()

        self.assertTrue(first.press_started)
        self.assertTrue(first.down)
        self.assertFalse(second.press_started)
        self.assertTrue(second.active)

    def test_backend_without_press_times_falls_back_to_button_edge(self):
        class StateOnlyMouse:
            def __init__(self):
                self.states = iter(((0, 0, 0), (1, 0, 0)))

            def getPressed(self, getTime=False):
                return next(self.states)

            def getPos(self):
                return (5, 6)

        tracker = MousePressTracker(StateOnlyMouse())
        tracker.reset()

        sample = tracker.poll()

        self.assertTrue(sample.down)
        self.assertTrue(sample.press_started)
        self.assertFalse(sample.buffered_press)

    def test_reset_while_held_requires_release_and_a_new_press_edge(self):
        mouse = FakeMouse(
            [
                ((1, 0, 0), (0.020, 0.0, 0.0), (10, 10)),
                ((1, 0, 0), (0.0, 0.0, 0.0), (10, 10)),
                ((0, 0, 0), (0.0, 0.0, 0.0), (10, 10)),
                ((1, 0, 0), (0.006, 0.0, 0.0), (30, 40)),
            ]
        )
        tracker = MousePressTracker(mouse)

        tracker.reset()
        self.assertFalse(tracker.poll().press_started)
        self.assertFalse(tracker.poll().press_started)
        self.assertTrue(tracker.poll().press_started)

    def test_repeated_clicks_do_not_poison_later_response_windows(self):
        up = ((0, 0, 0), (0.0, 0.0, 0.0), (0, 0))
        press = ((1, 0, 0), (0.001, 0.0, 0.0), (10, 10))
        mouse = FakeMouse([up, *([press, up] * 100), up, press, up, up, press])
        tracker = MousePressTracker(mouse)

        tracker.reset()
        for _ in range(100):
            self.assertTrue(tracker.poll().press_started)
            self.assertFalse(tracker.poll().press_started)

        tracker.reset()
        self.assertTrue(tracker.poll().press_started)
        self.assertFalse(tracker.poll().press_started)

        tracker.reset()
        self.assertTrue(tracker.poll().press_started)


class TouchHoldTrackerTests(unittest.TestCase):
    def test_afc_tolerance_includes_250_ms_but_default_requires_current_contact(self):
        self.assertEqual(AFC_HOLD_MAX_BREAK_S, .250)
        for recovered in (False, True):
            tracker = TouchHoldTracker(.1, max_break_s=AFC_HOLD_MAX_BREAK_S)
            tracker.update(True, 10)
            missing = tracker.update(False, 10.1)
            self.assertFalse(missing.qualified)
            self.assertFalse(missing.released)
            final = tracker.update(recovered, 10.35)
            self.assertFalse(final.released)
            self.assertEqual(final.qualified, recovered)

    def test_return_after_250_ms_cannot_rescue_hold_even_without_intermediate_polls(self):
        for recovered in (False, True):
            tracker = TouchHoldTracker(.1, max_break_s=AFC_HOLD_MAX_BREAK_S)
            tracker.update(True, 10)
            tracker.update(False, 10.1, timestamp_perf_ns=10_100_000_000)
            final = tracker.update(recovered, 10.350001)
            self.assertTrue(final.released)
            self.assertFalse(final.qualified)
            self.assertEqual(final.missing_since_perf_ns, 10_100_000_000)

    def test_recent_evidence_allows_transition_but_expires_before_hold_is_lost(self):
        tracker = TouchHoldTracker(.1, max_break_s=AFC_HOLD_MAX_BREAK_S,
                                   confirmation_window_s=AFC_HOLD_CONFIRMATION_WINDOW_S)
        tracker.update(True, 10)
        self.assertTrue(tracker.update(True, 10.1).qualified)
        self.assertTrue(tracker.update(False, 10.15).qualified)
        missing = tracker.update(False, 10.150001)
        self.assertFalse(missing.qualified)
        self.assertFalse(missing.released)
        self.assertTrue(tracker.update(True, 10.3).qualified)

    def test_single_tap_or_last_contact_before_required_duration_cannot_qualify(self):
        for last_contact in (10, 10.099):
            tracker = TouchHoldTracker(.1, max_break_s=.25, confirmation_window_s=.05)
            tracker.update(True, 10)
            tracker.update(True, last_contact)
            self.assertFalse(tracker.update(False, 10.1).qualified)
            self.assertFalse(tracker.update(False, 10.14).qualified)
            self.assertTrue(tracker.update(False, 10.36).released)

    def test_buffered_recontacts_sustain_hold_using_event_not_poll_time(self):
        tracker = TouchHoldTracker(.1, max_break_s=.25, confirmation_window_s=.05)
        tracker.start(10)
        self.assertFalse(tracker.update(True, 10.04, contact_perf_ns=10_030_000_000).qualified)
        tracker.update(False, 10.05)
        self.assertTrue(tracker.update(True, 10.16, contact_perf_ns=10_130_000_000).qualified)
        self.assertTrue(tracker.update(False, 10.18).qualified)
        self.assertFalse(tracker.update(False, 10.181).qualified)
        self.assertTrue(tracker.update(False, 10.381).released)
        self.assertFalse(tracker.update(True, 10.39).qualified)

    def test_buffered_touch_before_hold_deadline_does_not_gain_delayed_poll_time(self):
        tracker = TouchHoldTracker(.1, max_break_s=.25, confirmation_window_s=.05)
        tracker.start(10)
        self.assertFalse(tracker.update(True, 10.12, contact_perf_ns=10_090_000_000).qualified)
        self.assertFalse(tracker.update(False, 10.14).qualified)

    def test_slow_feedback_does_not_extend_an_isolated_initial_tap(self):
        tracker = TouchHoldTracker(.1, max_break_s=.25, confirmation_window_s=.05)
        tracker.start(10)
        tracker.update(False, 10.2, timestamp_perf_ns=10_200_000_000)
        late_return = tracker.update(True, 10.26)
        self.assertTrue(late_return.released)
        self.assertFalse(late_return.qualified)
        self.assertEqual(late_return.missing_since_perf_ns, 10_200_000_000)

    def test_stale_buffered_contact_cannot_refresh_a_hold(self):
        tracker = TouchHoldTracker(.1, max_break_s=.25, confirmation_window_s=.05)
        tracker.update(True, 10)
        tracker.update(True, 10.1)
        tracker.update(False, 10.11)
        self.assertFalse(tracker.update(True, 10.2, contact_perf_ns=10_050_000_000).qualified)
        self.assertTrue(tracker.update(False, 10.361).released)

    def test_buffered_recovery_must_precede_expiry_even_if_poll_is_late(self):
        for contact_ns, expired in ((10_240_000_000, False), (10_260_000_000, True)):
            with self.subTest(contact_ns=contact_ns):
                tracker = TouchHoldTracker(.1, max_break_s=.25, confirmation_window_s=.05)
                tracker.update(True, 9.9)
                tracker.update(False, 10, timestamp_perf_ns=10_000_000_000)
                recovered = tracker.update(True, 10.27, contact_perf_ns=contact_ns,
                                           timestamp_perf_ns=10_270_000_000)
                self.assertEqual(recovered.released, expired)
                self.assertEqual(recovered.qualified, not expired)
                self.assertEqual(recovered.missing_since_perf_ns,
                                 10_000_000_000 if expired else 10_270_000_000)

    def test_confirmation_window_must_fit_inside_release_tolerance(self):
        for window in (-.1, .251, float("nan"), float("inf")):
            with self.subTest(window=window), self.assertRaisesRegex(ValueError, "confirmation_window_s"):
                TouchHoldTracker(.1, max_break_s=.25, confirmation_window_s=window)

    def test_release_retains_first_detection_ns_across_grace_period(self):
        tracker = TouchHoldTracker(0.5, max_break_s=0.1)
        tracker.start(20.0)
        tracker.update(False, 20.2, timestamp_perf_ns=20_200_000_003)
        final = tracker.update(False, 20.31, timestamp_perf_ns=20_310_000_007)

        self.assertTrue(final.released)
        self.assertEqual(final.missing_since_perf_ns, 20_200_000_003)

    def test_short_registration_break_is_tolerated(self):
        tracker = TouchHoldTracker(0.2, max_break_s=0.1)

        tracker.start(10.0)
        self.assertFalse(tracker.update(False, 10.05).released)
        self.assertFalse(tracker.update(True, 10.09).released)
        final = tracker.update(True, 10.21)

        self.assertTrue(final.qualified)
        self.assertFalse(final.released)

    def test_break_longer_than_tolerance_records_release_start(self):
        tracker = TouchHoldTracker(0.5, max_break_s=0.1)

        tracker.start(20.0)
        tracker.update(False, 20.2)
        final = tracker.update(False, 20.300001)

        self.assertTrue(final.released)
        self.assertFalse(final.qualified)
        self.assertEqual(final.missing_since_s, 20.2)

    def test_reset_allows_a_new_hold(self):
        tracker = TouchHoldTracker(0.1)
        tracker.start(1.0)
        tracker.update(False, 1.0)
        self.assertTrue(tracker.update(False, 1.100001).released)

        tracker.reset()
        tracker.start(2.0)

        self.assertTrue(tracker.update(True, 2.1).qualified)

if __name__ == "__main__":
    unittest.main()
