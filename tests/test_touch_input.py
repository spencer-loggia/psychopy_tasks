import unittest

from bin.touch_input import (
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
