import unittest

from bin.initiation import resolve_initiation_config


class InitiationConfigTests(unittest.TestCase):
    def test_settings_are_normalized(self):
        settings = resolve_initiation_config(
            {
                "initiation_cue_center_position": [120, 340.5],
                "initiation_cue_style": " BLOB ",
                "hold_before_choice": True,
                "hold_cue_to_init_time_s": 0.2,
            }
        )

        self.assertEqual(settings.cue_center_position, (120.0, 340.5))
        self.assertEqual(settings.cue_style, "blob")
        self.assertTrue(settings.hold_before_choice)
        self.assertEqual(settings.hold_cue_to_init_time_s, 0.2)

    def test_all_four_settings_are_required(self):
        with self.assertRaisesRegex(KeyError, "hold_before_choice"):
            resolve_initiation_config(
                {
                    "initiation_cue_center_position": None,
                    "initiation_cue_style": "checker",
                    "hold_cue_to_init_time_s": 0.0,
                }
            )

    def test_invalid_values_are_rejected(self):
        base = {
            "initiation_cue_center_position": None,
            "initiation_cue_style": "checker",
            "hold_before_choice": False,
            "hold_cue_to_init_time_s": 0.0,
        }
        invalid_values = (
            ("initiation_cue_center_position", [1]),
            ("initiation_cue_style", "circle"),
            ("hold_before_choice", 1),
            ("hold_cue_to_init_time_s", -0.1),
        )
        for key, value in invalid_values:
            with self.subTest(key=key, value=value):
                cfg = dict(base)
                cfg[key] = value
                with self.assertRaises(ValueError):
                    resolve_initiation_config(cfg)


if __name__ == "__main__":
    unittest.main()
