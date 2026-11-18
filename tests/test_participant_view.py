import math
import random
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from bin.afc_geometry import prepare_afc_placement
from bin.initiation import initiation_cue_position
from bin.participant_view import ParticipantView
from bin.screen import oriented_size


class ParticipantViewTests(unittest.TestCase):
    def test_configured_cue_keeps_its_physical_position_and_canonical_identity(self):
        # Existing [300,-480] configurations keep their physical positions.
        for rotation, native_position in ((0, (300, -480)), (90, (-480, -300)),
                                          (180, (-300, 480)), (270, (480, 300))):
            with self.subTest(rotation=rotation):
                view = ParticipantView(rotation)
                native = SimpleNamespace(size=(100, 80), draw=Mock())
                position = initiation_cue_position((300, -480), oriented_size((1600, 2560), rotation), rotation)
                stimulus = view.stimulus(native, pos=position)
                self.assertEqual(stimulus.pos, (300, -480))
                self.assertEqual(native.pos, native_position)
                self.assertEqual(stimulus.ori, 0)
                self.assertEqual(native.ori, rotation)
                stimulus.draw()
                native.draw.assert_called_once()
                self.assertEqual(stimulus.pos, (300, -480))
                self.assertTrue(stimulus.contains((349, -441)))
                self.assertFalse(stimulus.contains((351, -480)))
                mouse = Mock()
                mouse.getPos.return_value = native_position
                self.assertEqual(view.mouse(mouse).getPos(), stimulus.pos)

    def test_fixed_circle_layout_preserves_previous_physical_locations(self):
        for rotation in (0, 90, 180, 270):
            view = ParticipantView(rotation)
            center = (120, -80)
            native_center = view.to_native(center)
            positions = prepare_afc_placement(
                True, 4, center, 200, (40, 20), (1600, 2560),
                fixed_layout_rotation_degrees=rotation,
            ).sample()
            for i, position in enumerate(positions):
                old_angle = math.pi / 2 + math.pi / 4 + i * math.pi / 2
                expected = (native_center[0] + 200 * math.cos(old_angle),
                            native_center[1] - 200 * math.sin(old_angle))
                for actual, value in zip(view.to_native(position), expected):
                    self.assertAlmostEqual(actual, value)

    def test_random_placement_is_identical_in_participant_space_across_display_rotations(self):
        reference = None
        for rotation in (0, 90, 180, 270):
            placement = prepare_afc_placement(
                False, 4, None, 150, (80, 50), (600, 800),
                placement_distribution="uniform", initiation_cue_position_px=(120, -160),
                fixed_layout_rotation_degrees=rotation, choice_hitbox_scale=1.25,
            )
            positions = placement.sample(rng=random.Random(51))
            if reference is None:
                reference = positions
            self.assertEqual(positions, reference)


if __name__ == "__main__":
    unittest.main()
