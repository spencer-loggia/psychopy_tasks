import os
import unittest
from unittest.mock import patch

from bin.screen import (
    MAIN_SCREEN_ENV,
    SECONDARY_SCREEN_ENV,
    load_screen_config,
)


class ScreenConfigTests(unittest.TestCase):
    def test_none_screen_values_inherit_environment(self):
        cfg = {"screens": {"main": None, "experimenter": None}}
        with patch.dict(
            os.environ,
            {
                MAIN_SCREEN_ENV: "HDMI-1",
                SECONDARY_SCREEN_ENV: "DSI-1",
            },
        ):
            self.assertEqual(
                load_screen_config(cfg),
                {"main": "HDMI-1", "experimenter": "DSI-1"},
            )

    def test_configured_screen_values_override_environment(self):
        cfg = {"screens": {"main": 1, "experimenter": "DSI-2"}}
        with patch.dict(
            os.environ,
            {
                MAIN_SCREEN_ENV: "HDMI-1",
                SECONDARY_SCREEN_ENV: "DSI-1",
            },
        ):
            self.assertEqual(load_screen_config(cfg), {"main": 1, "experimenter": "DSI-2"})


if __name__ == "__main__":
    unittest.main()
