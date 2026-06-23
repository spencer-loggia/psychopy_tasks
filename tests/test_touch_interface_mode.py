import unittest

from interface.rig_mode import (
    PORTABLE_MODE_VALUE,
    RIG_MODE_VALUE,
    experimenter_cursor_visible_for_touchscreen,
    is_rig_mode,
    mode_button_label,
    mode_command_for_target_mode,
    mode_script_for_target_mode,
    normalize_is_rig,
    target_mode_for_current_mode,
)


class TouchInterfaceModeTests(unittest.TestCase):
    def test_normalize_is_rig_accepts_only_zero_or_one(self):
        self.assertEqual(normalize_is_rig("0"), PORTABLE_MODE_VALUE)
        self.assertEqual(normalize_is_rig(" 1 "), RIG_MODE_VALUE)
        self.assertIsNone(normalize_is_rig(None))
        self.assertIsNone(normalize_is_rig(""))
        self.assertIsNone(normalize_is_rig("true"))

    def test_is_rig_mode_accepts_only_rig_value(self):
        self.assertFalse(is_rig_mode(PORTABLE_MODE_VALUE))
        self.assertTrue(is_rig_mode(RIG_MODE_VALUE))
        self.assertFalse(is_rig_mode(None))
        self.assertFalse(is_rig_mode("true"))

    def test_touchscreen_experimenter_cursor_policy_depends_on_rig_mode(self):
        self.assertTrue(experimenter_cursor_visible_for_touchscreen(False, PORTABLE_MODE_VALUE))
        self.assertFalse(experimenter_cursor_visible_for_touchscreen(True, PORTABLE_MODE_VALUE))
        self.assertTrue(experimenter_cursor_visible_for_touchscreen(True, RIG_MODE_VALUE))
        self.assertFalse(experimenter_cursor_visible_for_touchscreen(True, None))

    def test_button_label_advertises_target_mode(self):
        self.assertEqual(mode_button_label(PORTABLE_MODE_VALUE), "rig mode")
        self.assertEqual(mode_button_label(RIG_MODE_VALUE), "portable mode")

    def test_target_mode_toggles_current_mode(self):
        self.assertEqual(target_mode_for_current_mode(PORTABLE_MODE_VALUE), RIG_MODE_VALUE)
        self.assertEqual(target_mode_for_current_mode(RIG_MODE_VALUE), PORTABLE_MODE_VALUE)

    def test_target_mode_selects_expected_desktop_script(self):
        self.assertEqual(mode_script_for_target_mode(RIG_MODE_VALUE).name, "switch_to_rig_mode.sh")
        self.assertEqual(mode_script_for_target_mode(PORTABLE_MODE_VALUE).name, "switch_to_portable_mode.sh")

    def test_target_mode_runs_script_with_sudo(self):
        command = mode_command_for_target_mode(RIG_MODE_VALUE)

        self.assertEqual(command[:3], ["sudo", "-n", "bash"])
        self.assertEqual(command[3].split("/")[-1], "switch_to_rig_mode.sh")


if __name__ == "__main__":
    unittest.main()
