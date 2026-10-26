import math
import random
import unittest
from unittest.mock import Mock

from bin.afc_geometry import (
    compute_afc_positions,
    prepare_afc_placement,
    resolve_stimulus_circle,
    screen_px_to_psychopy,
    subject_centered_to_screen_px,
)
from bin.screen import oriented_size, rotate_centered_point, software_stimulus_rotation


class AFCGeometryTests(unittest.TestCase):
    def test_subject_centered_coordinates_follow_participant_view_for_all_rotations(self):
        for output_rotation in ("normal", "right", "inverted", "left"):
            rotation = software_stimulus_rotation(output_rotation)
            size = oriented_size((1600, 2560), rotation)
            with self.subTest(output_rotation=output_rotation):
                self.assertEqual(subject_centered_to_screen_px((0, 0), size, rotation), (size[0] / 2, size[1] / 2))
                for subject_pos in ((-500, -500), (250, 340), (-300, 700)):
                    px = subject_centered_to_screen_px(subject_pos, size, rotation)
                    native = screen_px_to_psychopy(px, size)
                    self.assertEqual(rotate_centered_point(native, -rotation), subject_pos)

    def test_centered_headfix_config_samples_inward_arc(self):
        for output_rotation in ("normal", "right", "inverted", "left"):
            rotation = software_stimulus_rotation(output_rotation)
            size = oriented_size((1600, 2560), rotation)
            center = subject_centered_to_screen_px((-500, -500), size, rotation)
            placement = prepare_afc_placement(
                False, 2, center, 500, (150, 150), size, display_margin=200,
            )
            rng = random.Random(2)
            with self.subTest(output_rotation=output_rotation):
                self.assertLess(sum(stop - start for start, stop in placement.valid_arcs), math.tau)
                for _ in range(100):
                    _, positions = placement.sample(rng=rng)
                    for native in positions:
                        x, y = rotate_centered_point(native, -rotation)
                        self.assertAlmostEqual(math.hypot(x + 500, y + 500), 500)
                        self.assertLessEqual(abs(x) + 75, 800 - 200 + 1e-9)
                        self.assertLessEqual(abs(y) + 75, 1280 - 200 + 1e-9)

    def test_screen_position_is_converted_from_upper_left_origin(self):
        self.assertEqual(screen_px_to_psychopy((0, 0), (2560, 1600)), (-1280.0, 800.0))
        self.assertEqual(screen_px_to_psychopy((1280, 800), (2560, 1600)), (0.0, 0.0))

    def test_screen_position_must_be_inside_main_screen(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            screen_px_to_psychopy((2561, 800), (2560, 1600))

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

    def test_circle_may_extend_outside_main_screen(self):
        self.assertEqual(
            resolve_stimulus_circle((20, 20), 30, (800, 600)),
            ((20.0, 20.0), 30.0),
        )

    def test_null_center_and_radius_keep_existing_defaults(self):
        self.assertEqual(resolve_stimulus_circle(None, None, (800, 600)), ((400, 300), 150))
        self.assertEqual(resolve_stimulus_circle((100, 200), None, (800, 600)), ((100, 200), 50))

    def test_offscreen_center_requires_explicit_radius(self):
        with self.assertRaisesRegex(ValueError, "Set stim_range_radius explicitly"):
            resolve_stimulus_circle((-500, -500), None, (2560, 1600))

    def test_offscreen_circle_samples_visible_arc_through_screen_center(self):
        center = (-500, -500)
        radius = math.hypot(1280 - center[0], 800 - center[1])
        placement = prepare_afc_placement(
            False, 2, center, radius, (150, 150), (2560, 1600), display_margin=200,
        )
        self.assertEqual(len(placement.valid_arcs), 1)
        center_angle = math.atan2(1300, 1780)
        start, stop = placement.valid_arcs[0]
        self.assertLess(start, center_angle)
        self.assertGreater(stop, center_angle)
        rng = random.Random(17)
        for _ in range(100):
            points, _ = placement.sample(rng=rng)
            for x, y in points:
                self.assertAlmostEqual(math.hypot(x + 500, y + 500), radius)
                self.assertGreaterEqual(x - 75, 200 - 1e-9)
                self.assertGreaterEqual(y - 75, 200 - 1e-9)
                self.assertLessEqual(x + 75, 2560 - 200 + 1e-9)
                self.assertLessEqual(y + 75, 1600 - 200 + 1e-9)
            self.assertTrue(abs(points[0][0] - points[1][0]) >= 150
                            or abs(points[0][1] - points[1][1]) >= 150)

    def test_narrow_visible_arc_needs_only_one_random_draw(self):
        placement = prepare_afc_placement(
            False, 1, (-1000, 300), 1100.0001, (100, 40), (800, 600), display_margin=50,
        )
        rng = Mock(wraps=random.Random(42))
        for _ in range(30):
            points, _ = placement.sample(rng=rng)
            x, y = points[0]
            self.assertGreaterEqual(x - 50, 50 - 1e-9)
            self.assertGreaterEqual(y - 20, 50)
            self.assertLessEqual(y + 20, 550)
        self.assertEqual(rng.uniform.call_count, 30)

    def test_disconnected_arcs_are_sampled_in_proportion_to_length(self):
        placement = prepare_afc_placement(
            False, 1, (350, 280), 330, (80, 40), (800, 600), display_margin=30,
        )
        self.assertGreater(len(placement.valid_arcs), 1)
        lengths = [stop - start for start, stop in placement.valid_arcs]
        self.assertGreater(max(lengths) - min(lengths), 0.1)
        total = sum(lengths)
        counts = [0] * len(lengths)
        rng = random.Random(3)
        for _ in range(6000):
            points, _ = placement.sample(rng=rng)
            x, y = points[0]
            angle = math.atan2(y - 280, x - 350) % math.tau
            matches = [i for i, (start, stop) in enumerate(placement.valid_arcs) if start <= angle <= stop]
            self.assertEqual(len(matches), 1)
            counts[matches[0]] += 1
            self.assertTrue(70 - 1e-9 <= x <= 730 + 1e-9)
            self.assertTrue(50 - 1e-9 <= y <= 550 + 1e-9)
        for count, length in zip(counts, lengths):
            self.assertAlmostEqual(count / 6000, length / total, delta=0.025)

    def test_rectangular_stimulus_corners_respect_margin_for_all_output_rotations(self):
        image_size, margin = (240, 80), 200
        for output_rotation in ("normal", "right", "inverted", "left"):
            with self.subTest(output_rotation=output_rotation):
                rotation = software_stimulus_rotation(output_rotation)
                native_size = oriented_size((1600, 2560), rotation)
                # Place the circle center at subject-centered (-500, -500).
                cx, cy = rotate_centered_point((-500, -500), rotation)
                center = (native_size[0] / 2 + cx, native_size[1] / 2 - cy)
                placement = prepare_afc_placement(
                    False, 1, center, 900, oriented_size(image_size, rotation), native_size,
                    display_margin=margin,
                )
                rng = random.Random(8)
                for _ in range(100):
                    _, positions = placement.sample(rng=rng)
                    px, py = positions[0]
                    for dx in (-120, 120):
                        for dy in (-40, 40):
                            rx, ry = rotate_centered_point((dx, dy), rotation)
                            sx, sy = rotate_centered_point((px + rx, py + ry), -rotation)
                            self.assertLessEqual(abs(sx), 800 - margin + 1e-9)
                            self.assertLessEqual(abs(sy), 1280 - margin + 1e-9)

    def test_no_valid_arc_and_tangency_fail_before_sampling(self):
        for center, radius, size, margin in (
            ((-1000, 300), 1100, (100, 40), 50),  # Only a tangent point.
            ((400, 300), 500, (20, 20), 50),  # Circle encloses the safe rectangle.
            ((400, 300), 100, (20, 20), 300),  # Margin consumes the screen.
            ((400, 300), 100, (900, 40), 0),  # Image larger than the screen.
            ((-500, -500), 50, (20, 20), 0),  # Circle entirely outside.
        ):
            with self.subTest(center=center, radius=radius, size=size, margin=margin):
                with self.assertRaisesRegex(ValueError, "No valid stimulus circle arcs"):
                    prepare_afc_placement(False, 1, center, radius, size, (800, 600), display_margin=margin)

    def test_fixed_positions_reject_image_edge_crossing_margin(self):
        with self.assertRaisesRegex(ValueError, "Fixed.*outside display_margin"):
            prepare_afc_placement(True, 2, (400, 300), 320, (100, 40), (800, 600), display_margin=50)

    def test_zero_margin_still_keeps_full_image_on_screen(self):
        placement = prepare_afc_placement(False, 1, (400, 300), 390, (100, 40), (800, 600))
        rng = random.Random(4)
        for _ in range(100):
            points, _ = placement.sample(rng=rng)
            x, y = points[0]
            self.assertTrue(50 - 1e-9 <= x <= 750 + 1e-9)
            self.assertTrue(20 - 1e-9 <= y <= 580 + 1e-9)

    def test_invalid_geometry_rejected(self):
        defaults = dict(fixed_positions=False, num_afc=2, center_point=None, stim_range_radius=100,
                        stim_size=(20, 20), effective_win_size=(800, 600))
        for override in (
            {"display_margin": -1}, {"display_margin": float("nan")}, {"display_margin": float("inf")},
            {"stim_range_radius": 0}, {"stim_range_radius": float("nan")},
            {"center_point": (float("inf"), 100)}, {"stim_size": (0, 20)},
            {"effective_win_size": (800, float("nan"))}, {"num_afc": 0},
        ):
            with self.subTest(override=override), self.assertRaises(ValueError):
                prepare_afc_placement(**(defaults | override))


if __name__ == "__main__":
    unittest.main()
