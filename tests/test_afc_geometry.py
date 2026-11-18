import math
import random
import unittest
from unittest.mock import Mock

from bin.afc_geometry import (
    normalize_placement_distribution,
    prepare_afc_placement,
    resolve_stimulus_circle,
)
from bin.screen import oriented_size, rotate_centered_point, software_stimulus_rotation


class AFCGeometryTests(unittest.TestCase):
    def test_scalar_margin_matches_equal_edge_margins_in_all_placement_modes(self):
        for distribution, fixed in (("uniform", False), ("circle", False), ("circle", True)):
            with self.subTest(distribution=distribution, fixed=fixed):
                kwargs = dict(fixed_positions=fixed, num_afc=3, center_point=None,
                              stim_range_radius=120, stim_size=(30, 20),
                              effective_win_size=(800, 600), placement_distribution=distribution)
                scalar = prepare_afc_placement(**kwargs, display_margin=25)
                edges = prepare_afc_placement(**kwargs, display_margin=(25, 25, 25, 25))
                self.assertEqual(scalar, edges)
                self.assertEqual(scalar.sample(rng=random.Random(19)), edges.sample(rng=random.Random(19)))

    def test_asymmetric_margins_bound_each_image_edge_in_circle_and_uniform_modes(self):
        # Participant width/height 800x600, image width/height 80x40.
        # Left=250, right=40, top=150, bottom=30 gives these center limits.
        expected_bounds = (-110, -250, 320, 130)
        for distribution in ("circle", "uniform"):
            with self.subTest(distribution=distribution):
                placement = prepare_afc_placement(
                    False, 3, None, 220, (80, 40), (800, 600),
                    display_margin=[250, 40, 150, 30], placement_distribution=distribution,
                )
                self.assertEqual(placement.center_bounds, expected_bounds)
                rng = random.Random(19)
                for _ in range(100):
                    for x, y in placement.sample(rng=rng):
                        self.assertGreaterEqual(x - 40, -400 + 250 - 1e-9)
                        self.assertLessEqual(x + 40, 400 - 40 + 1e-9)
                        self.assertLessEqual(y + 20, 300 - 150 + 1e-9)
                        self.assertGreaterEqual(y - 20, -300 + 30 - 1e-9)
                        if distribution == "circle":
                            self.assertAlmostEqual(math.hypot(x, y), 220)

    def test_asymmetric_margins_reject_invalid_fixed_positions_and_empty_regions(self):
        prepare_afc_placement(True, 4, None, 200, (40, 40), (800, 600),
                              display_margin=(20, 20, 100, 20))
        with self.assertRaisesRegex(ValueError, "Fixed.*outside display_margin"):
            prepare_afc_placement(True, 4, None, 200, (40, 40), (800, 600),
                                  display_margin=(20, 20, 150, 20))
        for distribution in ("circle", "uniform"):
            for margins in ((800, 0, 0, 0), (0, 800, 0, 0), (0, 0, 600, 0), (0, 0, 0, 600)):
                with self.subTest(distribution=distribution, margins=margins), self.assertRaisesRegex(ValueError, "No valid"):
                    prepare_afc_placement(False, 1, None, 100, (40, 20), (800, 600),
                                          display_margin=margins, placement_distribution=distribution)

    def test_edge_margins_require_exactly_four_finite_nonnegative_values(self):
        for margins in ([], [10], [10, 20], [10, 20, 30], [0] * 5,
                        [0, -1, 0, 0], [0, 0, float("nan"), 0],
                        [0, 0, 0, float("inf")], [None, 0, 0, 0]):
            with self.subTest(margins=margins), self.assertRaisesRegex(ValueError, "display_margin"):
                prepare_afc_placement(False, 1, None, 100, (40, 20), (800, 600), display_margin=margins)

    def test_centered_headfix_config_samples_inward_arc(self):
        for output_rotation in ("normal", "right", "inverted", "left"):
            rotation = software_stimulus_rotation(output_rotation)
            size = (1600, 2560)
            center = (-500, -500)
            placement = prepare_afc_placement(
                False, 2, center, 500, (150, 150), size, display_margin=200,
            )
            rng = random.Random(2)
            with self.subTest(output_rotation=output_rotation):
                self.assertLess(sum(stop - start for start, stop in placement.valid_arcs), math.tau)
                for _ in range(100):
                    positions = placement.sample(rng=rng)
                    for x, y in positions:
                        self.assertAlmostEqual(math.hypot(x + 500, y + 500), 500)
                        self.assertLessEqual(abs(x) + 75, 800 - 200 + 1e-9)
                        self.assertLessEqual(abs(y) + 75, 1280 - 200 + 1e-9)

    def test_fixed_positions_are_evenly_spaced_on_configured_circle(self):
        positions = prepare_afc_placement(
            fixed_positions=True,
            num_afc=4,
            center_point=(120, -80),
            stim_range_radius=100,
            stim_size=(20, 20),
            effective_win_size=(800, 600),
        ).sample()

        self.assertEqual(len(positions), 4)
        for x, y in positions:
            self.assertAlmostEqual(math.hypot(x - 120, y + 80), 100)

    def test_random_positions_are_repeatable_with_explicit_rng(self):
        kwargs = dict(
            fixed_positions=False,
            num_afc=3,
            center_point=None,
            stim_range_radius=120,
            stim_size=(30, 30),
            effective_win_size=(800, 600),
        )

        first = prepare_afc_placement(**kwargs).sample(rng=random.Random(7))
        second = prepare_afc_placement(**kwargs).sample(rng=random.Random(7))

        self.assertEqual(first, second)

    def test_explicit_circle_keeps_default_placement_and_rng_sequence(self):
        for fixed in (False, True):
            kwargs = dict(fixed_positions=fixed, num_afc=3, center_point=None,
                          stim_range_radius=120, stim_size=(30, 30), effective_win_size=(800, 600))
            implicit_rng, explicit_rng = random.Random(7), random.Random(7)
            implicit = prepare_afc_placement(**kwargs)
            explicit = prepare_afc_placement(**kwargs, placement_distribution="circle")
            with self.subTest(fixed=fixed):
                for _ in range(20):
                    self.assertEqual(implicit.sample(rng=implicit_rng), explicit.sample(rng=explicit_rng))
                self.assertEqual(implicit_rng.getstate(), explicit_rng.getstate())

    def test_uniform_ignores_circle_settings_and_is_seed_repeatable(self):
        kwargs = dict(num_afc=4, stim_size=(60, 40), effective_win_size=(800, 600),
                      display_margin=30, placement_distribution="uniform")
        regular = prepare_afc_placement(False, center_point=None, stim_range_radius=100, **kwargs)
        inactive_invalid = prepare_afc_placement(
            True, center_point="unused", stim_range_radius=100, **kwargs,
        )
        self.assertIsNone(inactive_invalid.center_px)
        self.assertIsNone(inactive_invalid.radius_px)
        self.assertIsNone(inactive_invalid.fixed_positions_px)
        self.assertEqual(inactive_invalid.valid_arcs, ())
        self.assertEqual(inactive_invalid.center_bounds, (-340, -250, 340, 250))
        rng1, rng2 = random.Random(17), random.Random(17)
        samples = []
        for _ in range(10):
            sample = inactive_invalid.sample(rng=rng1)
            self.assertEqual(sample, regular.sample(rng=rng2))
            samples.append(sample)
        self.assertNotEqual(samples[0], samples[1], "Uniform remains random even with fixed_positions=true")

    def test_uniform_full_rectangle_sampling_is_uniform_in_area(self):
        placement = prepare_afc_placement(
            False, 1, None, None, (80, 40), (800, 600), display_margin=30,
            placement_distribution="uniform",
        )
        left, bottom, right, top = placement.center_bounds
        counts = [[0] * 4 for _ in range(4)]
        normalized = []
        rng = random.Random(3)
        for _ in range(6000):
            points = placement.sample(rng=rng)
            x, y = points[0]
            nx, ny = (x-left)/(right-left), (y-bottom)/(top-bottom)
            normalized.append((nx, ny))
            counts[min(3, int(nx*4))][min(3, int(ny*4))] += 1
        # Fixed seed and generous bounds check spatial coverage and equal-area
        # density without requiring a statistical test to pass by chance.
        for row in counts:
            for count in row:
                self.assertAlmostEqual(count/6000, 1/16, delta=.015)
        for axis in (0, 1):
            values = [p[axis] for p in normalized]
            self.assertLess(min(values), .005)
            self.assertGreater(max(values), .995)
            self.assertAlmostEqual(sum(values)/len(values), .5, delta=.015)

    def test_uniform_rectangular_footprints_and_nonoverlap_for_all_rotations(self):
        image_size, margin = (240, 80), 200
        for rotation in (0, 90, 180, 270):
            native_size = oriented_size((1600, 2560), rotation)
            placement = prepare_afc_placement(
                False, 8, None, None, image_size, (1600, 2560),
                display_margin=margin, placement_distribution="uniform",
            )
            rng = random.Random(8)
            for _ in range(100):
                positions = placement.sample(rng=rng)
                for i, (px, py) in enumerate(positions):
                    for dx in (-120, 120):
                        for dy in (-40, 40):
                            nx, ny = rotate_centered_point((px + dx, py + dy), rotation)
                            self.assertLessEqual(abs(nx), native_size[0] / 2 - margin + 1e-9)
                            self.assertLessEqual(abs(ny), native_size[1] / 2 - margin + 1e-9)
                    for qx, qy in positions[:i]:
                        self.assertTrue(abs(px-qx) > image_size[0] or abs(py-qy) > image_size[1])

    def test_uniform_rejects_overlapping_proposals(self):
        placement = prepare_afc_placement(
            False, 2, None, None, (50, 50), (800, 600), placement_distribution="uniform",
        )
        rng = Mock()
        rng.uniform.side_effect = [100, 100, 110, 110, 200, 100]
        points = placement.sample(rng=rng)
        self.assertEqual(points, [(100, 100), (200, 100)])
        self.assertEqual(rng.uniform.call_count, 6)

    def test_uniform_impossible_packing_fails_instead_of_overlapping(self):
        placement = prepare_afc_placement(
            False, 2, None, None, (600, 400), (800, 600), placement_distribution="uniform",
        )
        with self.assertRaisesRegex(RuntimeError, "Could not place 2.*uniform placement area"):
            placement.sample(rng=random.Random(2))

    def test_uniform_requires_positive_area_and_valid_active_geometry(self):
        kwargs = dict(fixed_positions=False, num_afc=2, center_point=None, stim_range_radius=None,
                      stim_size=(40, 20), effective_win_size=(800, 600), placement_distribution="uniform")
        for override in ({"display_margin": 300}, {"stim_size": (800, 20)}, {"stim_size": (900, 20)}):
            with self.subTest(override=override), self.assertRaisesRegex(ValueError, "No valid uniform placement area"):
                prepare_afc_placement(**(kwargs | override))
        for override in ({"effective_win_size": (800, float("nan"))}, {"display_margin": -1},
                         {"stim_size": (-20, 20)}, {"num_afc": 0}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                prepare_afc_placement(**(kwargs | override))

    def test_unknown_placement_distribution_is_rejected(self):
        for distribution in (None, "", "rectangle", "random", 1):
            with self.subTest(distribution=distribution), self.assertRaisesRegex(ValueError, "placement_distribution"):
                prepare_afc_placement(False, 1, None, None, (20, 20), (800, 600),
                                      placement_distribution=distribution)

    def test_placement_distribution_normalizes_before_geometry_is_parsed(self):
        self.assertEqual(normalize_placement_distribution(), "circle")
        self.assertEqual(normalize_placement_distribution(" CIRCLE "), "circle")
        self.assertEqual(normalize_placement_distribution(" Uniform "), "uniform")
        placement = prepare_afc_placement(
            True, 1, "unused", None, (20, 20), (800, 600), placement_distribution=" Uniform ",
        )
        self.assertEqual(placement.placement_distribution, "uniform")

    def test_touch_hitboxes_and_uniform_cue_exclusion_for_all_rotations(self):
        subject_size = (600, 800)
        cue = (120, -160)
        for rotation in (0, 90, 180, 270):
            native_size = oriented_size(subject_size, rotation)
            for distribution in ("circle", "uniform"):
                with self.subTest(rotation=rotation, distribution=distribution):
                    placement = prepare_afc_placement(
                        False, 3, None, 180, (100, 60), subject_size,
                        placement_distribution=distribution, display_margin=30,
                        choice_hitbox_scale=1.25,
                        initiation_cue_position_px=cue,
                    )
                    rng = random.Random(71)
                    for _ in range(60):
                        positions = placement.sample(rng=rng)
                        for index, (x, y) in enumerate(positions):
                            self.assertLessEqual(abs(x) + 50, 300 - 30)
                            self.assertLessEqual(abs(y) + 30, 400 - 30)
                            if distribution == "uniform":
                                self.assertGreaterEqual(math.hypot(x - cue[0], y - cue[1]), 180)
                            for px, py in positions[:index]:
                                self.assertTrue(abs(x - px) > 125 or abs(y - py) > 75)

    def test_fixed_layout_rejects_hitbox_overlap_even_when_images_do_not_overlap(self):
        prepare_afc_placement(True, 2, None, 55, (100, 20), (800, 600))
        with self.assertRaisesRegex(ValueError, "overlap"):
            prepare_afc_placement(True, 2, None, 55, (100, 20), (800, 600), choice_hitbox_scale=1.25)

    def test_uniform_exclusion_rejects_empty_region_and_invalid_radius(self):
        for radius in (-1, float("inf"), float("nan")):
            with self.subTest(radius=radius), self.assertRaisesRegex(ValueError, "stim_range_radius"):
                prepare_afc_placement(False, 1, None, radius, (20, 20), (800, 600),
                                      placement_distribution="uniform")
        with self.assertRaisesRegex(ValueError, "No valid uniform placement area outside"):
            prepare_afc_placement(False, 1, None, 1000, (20, 20), (800, 600),
                                  placement_distribution="uniform")

    def test_circle_may_extend_outside_main_screen(self):
        self.assertEqual(
            resolve_stimulus_circle((20, 20), 30, (800, 600)),
            ((20.0, 20.0), 30.0),
        )

    def test_null_center_and_radius_keep_existing_defaults(self):
        self.assertEqual(resolve_stimulus_circle(None, None, (800, 600)), ((0, 0), 150))
        self.assertEqual(resolve_stimulus_circle((100, 200), None, (800, 600)), ((100, 200), 50))

    def test_offscreen_center_requires_explicit_radius(self):
        with self.assertRaisesRegex(ValueError, "Set stim_range_radius explicitly"):
            resolve_stimulus_circle((-1780, 1300), None, (2560, 1600))

    def test_offscreen_circle_samples_visible_arc_through_screen_center(self):
        center = (-1780, 1300)
        radius = math.hypot(*center)
        placement = prepare_afc_placement(
            False, 2, center, radius, (150, 150), (2560, 1600), display_margin=200,
        )
        self.assertEqual(len(placement.valid_arcs), 1)
        center_angle = math.atan2(-1300, 1780) % math.tau
        start, stop = placement.valid_arcs[0]
        self.assertLess(start, center_angle)
        self.assertGreater(stop, center_angle)
        rng = random.Random(17)
        for _ in range(100):
            points = placement.sample(rng=rng)
            for x, y in points:
                self.assertAlmostEqual(math.hypot(x - center[0], y - center[1]), radius)
                self.assertGreaterEqual(x - 75, -1280 + 200 - 1e-9)
                self.assertGreaterEqual(y - 75, -800 + 200 - 1e-9)
                self.assertLessEqual(x + 75, 1280 - 200 + 1e-9)
                self.assertLessEqual(y + 75, 800 - 200 + 1e-9)
            self.assertTrue(abs(points[0][0] - points[1][0]) >= 150
                            or abs(points[0][1] - points[1][1]) >= 150)

    def test_narrow_visible_arc_needs_only_one_random_draw(self):
        placement = prepare_afc_placement(
            False, 1, (-1400, 0), 1100.0001, (100, 40), (800, 600), display_margin=50,
        )
        rng = Mock(wraps=random.Random(42))
        for _ in range(30):
            points = placement.sample(rng=rng)
            x, y = points[0]
            self.assertGreaterEqual(x - 50, -350 - 1e-9)
            self.assertGreaterEqual(y - 20, -250)
            self.assertLessEqual(y + 20, 250)
        self.assertEqual(rng.uniform.call_count, 30)

    def test_disconnected_arcs_are_sampled_in_proportion_to_length(self):
        placement = prepare_afc_placement(
            False, 1, (-50, 20), 330, (80, 40), (800, 600), display_margin=30,
        )
        self.assertGreater(len(placement.valid_arcs), 1)
        lengths = [stop - start for start, stop in placement.valid_arcs]
        self.assertGreater(max(lengths) - min(lengths), 0.1)
        total = sum(lengths)
        counts = [0] * len(lengths)
        rng = random.Random(3)
        for _ in range(6000):
            points = placement.sample(rng=rng)
            x, y = points[0]
            angle = math.atan2(y - 20, x + 50) % math.tau
            matches = [i for i, (start, stop) in enumerate(placement.valid_arcs) if start <= angle <= stop]
            self.assertEqual(len(matches), 1)
            counts[matches[0]] += 1
            self.assertTrue(-330 - 1e-9 <= x <= 330 + 1e-9)
            self.assertTrue(-250 - 1e-9 <= y <= 250 + 1e-9)
        for count, length in zip(counts, lengths):
            self.assertAlmostEqual(count / 6000, length / total, delta=0.025)

    def test_rectangular_stimulus_corners_respect_margin_for_all_output_rotations(self):
        image_size, margin = (240, 80), 200
        for output_rotation in ("normal", "right", "inverted", "left"):
            with self.subTest(output_rotation=output_rotation):
                rotation = software_stimulus_rotation(output_rotation)
                native_size = oriented_size((1600, 2560), rotation)
                placement = prepare_afc_placement(
                    False, 1, (-500, -500), 900, image_size, (1600, 2560), display_margin=margin,
                )
                rng = random.Random(8)
                for _ in range(100):
                    positions = placement.sample(rng=rng)
                    px, py = positions[0]
                    for dx in (-120, 120):
                        for dy in (-40, 40):
                            sx, sy = px + dx, py + dy
                            self.assertLessEqual(abs(sx), 800 - margin + 1e-9)
                            self.assertLessEqual(abs(sy), 1280 - margin + 1e-9)

    def test_no_valid_arc_and_tangency_fail_before_sampling(self):
        for center, radius, size, margin in (
            ((-1400, 0), 1100, (100, 40), 50),  # Only a tangent point.
            ((0, 0), 500, (20, 20), 50),  # Circle encloses the safe rectangle.
            ((0, 0), 100, (20, 20), 300),  # Margin consumes the screen.
            ((0, 0), 100, (900, 40), 0),  # Image larger than the screen.
            ((-500, -500), 50, (20, 20), 0),  # Circle entirely outside.
        ):
            with self.subTest(center=center, radius=radius, size=size, margin=margin):
                with self.assertRaisesRegex(ValueError, "No valid stimulus circle arcs"):
                    prepare_afc_placement(False, 1, center, radius, size, (800, 600), display_margin=margin)

    def test_fixed_positions_reject_image_edge_crossing_margin(self):
        with self.assertRaisesRegex(ValueError, "Fixed.*outside display_margin"):
            prepare_afc_placement(True, 2, (0, 0), 320, (100, 40), (800, 600), display_margin=50)

    def test_zero_margin_still_keeps_full_image_on_screen(self):
        placement = prepare_afc_placement(False, 1, (0, 0), 390, (100, 40), (800, 600))
        rng = random.Random(4)
        for _ in range(100):
            points = placement.sample(rng=rng)
            x, y = points[0]
            self.assertTrue(-350 - 1e-9 <= x <= 350 + 1e-9)
            self.assertTrue(-280 - 1e-9 <= y <= 280 + 1e-9)

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
