import unittest

from bin.initiation import build_initiation_cue_image


class InitiationCueTests(unittest.TestCase):
    def test_blob_uses_init_dot_color_and_matches_checker_size(self):
        checker = build_initiation_cue_image(
            75,
            bg_rgb_255=(168, 169, 166),
            style="checker",
        )
        blob = build_initiation_cue_image(
            75,
            bg_rgb_255=(168, 169, 166),
            style="blob",
            cue_color=(50, 60, 70),
        )

        self.assertEqual(blob.size, checker.size)
        center = blob.getpixel((blob.width // 2, blob.height // 2))
        self.assertEqual(center[:3], (50, 60, 70))
        self.assertGreater(center[3], 250)

    def test_pressed_cue_is_darker_with_the_same_gaussian_mask(self):
        idle = build_initiation_cue_image(
            75,
            bg_rgb_255=(168, 169, 166),
            style="blob",
            cue_color=(80, 90, 100),
        )
        pressed = build_initiation_cue_image(
            75,
            bg_rgb_255=(168, 169, 166),
            style="blob",
            cue_color=(80, 90, 100),
            pressed=True,
        )
        center = (idle.width // 2, idle.height // 2)

        self.assertEqual(idle.getchannel("A").tobytes(), pressed.getchannel("A").tobytes())
        self.assertEqual(idle.getpixel(center)[:3], (80, 90, 100))
        self.assertEqual(pressed.getpixel(center)[:3], (48, 58, 68))


if __name__ == "__main__":
    unittest.main()
