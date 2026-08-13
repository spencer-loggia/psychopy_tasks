import math
import random
import unittest

from bin.afc_geometry import compute_afc_positions, resolve_stimulus_circle


class AFCGeometryTests(unittest.TestCase):
    def test_fixed_positions_are_evenly_spaced_on_configured_circle(self):
        screen_positions, psychopy_positions = compute_afc_positions(
            fixed_positions=True,
            num_afc=4,
            center_point=(400, 300),
            stim_range_radius=100,
            stim_size=(20, 20),
            effective_win_size=(800, 600),
        )

        self.assertEqual(len(screen_positions), 4)
        for (screen_x, screen_y), (psychopy_x, psychopy_y) in zip(
            screen_positions,
            psychopy_positions,
        ):
            self.assertAlmostEqual(math.hypot(screen_x - 400, screen_y - 300), 100)
            self.assertAlmostEqual(psychopy_x, screen_x - 400)
            self.assertAlmostEqual(psychopy_y, 300 - screen_y)

    def test_random_positions_are_repeatable_with_explicit_rng(self):
        kwargs = dict(
            fixed_positions=False,
            num_afc=3,
            center_point=None,
            stim_range_radius=120,
            stim_size=(30, 30),
            effective_win_size=(800, 600),
        )

        first = compute_afc_positions(**kwargs, rng=random.Random(7))
        second = compute_afc_positions(**kwargs, rng=random.Random(7))

        self.assertEqual(first, second)

    def test_circle_must_fit_inside_main_screen(self):
        with self.assertRaisesRegex(ValueError, "outside the main screen bounds"):
            resolve_stimulus_circle(
                center_point=(20, 20),
                stim_range_radius=30,
                effective_win_size=(800, 600),
            )


if __name__ == "__main__":
    unittest.main()
