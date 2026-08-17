import unittest

from bin.touch_input import (
    MousePressSample,
    MousePressTracker,
    advance_release_armed_touch_gate,
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

        self.assertFalse(tracker.reset())
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

        self.assertTrue(tracker.reset())
        self.assertFalse(tracker.poll().press_started)
        self.assertFalse(tracker.poll().press_started)
        self.assertTrue(tracker.poll().press_started)


class ReleaseArmedTouchGateTests(unittest.TestCase):
    def test_held_initiation_press_stays_blocked_until_release(self):
        held = MousePressSample((0.0, 0.0), (True, False, False), False, False)
        released = MousePressSample((0.0, 0.0), (False, False, False), False, False)
        new_press = MousePressSample((5.0, 6.0), (True, False, False), True, False)

        armed, eligible = advance_release_armed_touch_gate(False, held)
        self.assertFalse(armed)
        self.assertFalse(eligible)

        armed, eligible = advance_release_armed_touch_gate(armed, released)
        self.assertTrue(armed)
        self.assertFalse(eligible)

        armed, eligible = advance_release_armed_touch_gate(armed, new_press)
        self.assertTrue(armed)
        self.assertTrue(eligible)

    def test_buffered_repress_during_flip_arms_and_activates_gate(self):
        buffered_repress = MousePressSample(
            (5.0, 6.0),
            (False, False, False),
            True,
            True,
        )

        armed, eligible = advance_release_armed_touch_gate(
            False,
            buffered_repress,
        )

        self.assertTrue(armed)
        self.assertTrue(eligible)


if __name__ == "__main__":
    unittest.main()
