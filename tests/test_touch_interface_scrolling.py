import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from interface.touch_interface import (
    ScrollableButtonFrame,
    TOUCH_SCROLL_THRESHOLD_PX,
    touch_drag_exceeds_threshold,
    touch_drag_scroll_fraction,
    wheel_scroll_units,
)


class TouchInterfaceScrollingTests(unittest.TestCase):
    def test_x11_wheel_buttons_scroll_in_expected_direction(self):
        self.assertLess(wheel_scroll_units(button_num=4), 0)
        self.assertGreater(wheel_scroll_units(button_num=5), 0)

    def test_mousewheel_deltas_are_normalized_across_platform_styles(self):
        self.assertLess(wheel_scroll_units(delta=120), 0)
        self.assertGreater(wheel_scroll_units(delta=-120), 0)
        self.assertLess(wheel_scroll_units(delta=1), 0)
        self.assertEqual(wheel_scroll_units(), 0)

    def test_touch_motion_becomes_a_scroll_only_after_threshold(self):
        self.assertFalse(
            touch_drag_exceeds_threshold(100, 100 + TOUCH_SCROLL_THRESHOLD_PX - 1)
        )
        self.assertTrue(
            touch_drag_exceeds_threshold(100, 100 + TOUCH_SCROLL_THRESHOLD_PX)
        )

    def test_upward_touch_drag_scrolls_down_and_clamps_to_content(self):
        self.assertAlmostEqual(
            touch_drag_scroll_fraction(
                initial_first=0.1,
                start_y=300,
                current_y=200,
                content_height=1000,
                visible_fraction=0.4,
            ),
            0.2,
        )
        self.assertEqual(
            touch_drag_scroll_fraction(
                initial_first=0.5,
                start_y=300,
                current_y=-1000,
                content_height=1000,
                visible_fraction=0.4,
            ),
            0.6,
        )

    def test_non_overflowing_content_stays_at_top(self):
        self.assertEqual(
            touch_drag_scroll_fraction(
                initial_first=0.0,
                start_y=300,
                current_y=100,
                content_height=500,
                visible_fraction=1.0,
            ),
            0.0,
        )

    def test_swipe_release_does_not_invoke_its_starting_button(self):
        button = Mock()
        frame = SimpleNamespace(
            _pressed_button=button,
            _touch_moved=True,
            _reset_touch_gesture=Mock(),
            _event_is_inside_button=Mock(return_value=True),
        )

        result = ScrollableButtonFrame._on_touch_release(frame, Mock())

        self.assertEqual(result, "break")
        button.invoke.assert_not_called()

    def test_stationary_touch_release_invokes_its_button(self):
        button = Mock()
        button.cget.return_value = "normal"
        frame = SimpleNamespace(
            _pressed_button=button,
            _touch_moved=False,
            _reset_touch_gesture=Mock(),
            _event_is_inside_button=Mock(return_value=True),
        )

        ScrollableButtonFrame._on_touch_release(frame, Mock())

        button.invoke.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
