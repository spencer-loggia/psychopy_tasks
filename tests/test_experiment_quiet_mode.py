import subprocess
import unittest
from unittest.mock import patch

from interface.experiment_quiet_mode import (
    DEFAULT_SYSTEMD_UNITS,
    ExperimentQuietMode,
    create_experiment_quiet_mode,
)


class ExperimentQuietModeTests(unittest.TestCase):
    def test_true_uses_the_default_maintenance_units(self):
        guard = create_experiment_quiet_mode({"experiment_quiet_mode": True})

        self.assertEqual(guard.units, DEFAULT_SYSTEMD_UNITS)

    def test_active_units_are_stopped_and_only_those_units_are_restored(self):
        calls = []
        active = {"cron.service", "apt-daily.timer"}

        def runner(command, **_kwargs):
            calls.append(command)
            returncode = 0
            if command[:3] == ["systemctl", "is-active", "--quiet"]:
                returncode = 0 if command[3] in active else 3
            return subprocess.CompletedProcess(command, returncode, "", "")

        guard = ExperimentQuietMode(
            ("apt-daily.timer", "man-db.timer", "cron.service"),
            runner=runner,
        )
        with (
            patch("interface.experiment_quiet_mode.sys.platform", "linux"),
            patch("interface.experiment_quiet_mode.os.geteuid", return_value=0),
        ):
            guard.enter()
            guard.exit()

        self.assertIn(["systemctl", "stop", "apt-daily.timer"], calls)
        self.assertIn(["systemctl", "stop", "cron.service"], calls)
        self.assertNotIn(["systemctl", "stop", "man-db.timer"], calls)
        self.assertEqual(
            calls[-2:],
            [
                ["systemctl", "start", "cron.service"],
                ["systemctl", "start", "apt-daily.timer"],
            ],
        )

    def test_custom_unit_list_is_validated(self):
        guard = create_experiment_quiet_mode(
            {"experiment_quiet_mode": {"systemd_units": ["cron.service"]}}
        )
        self.assertEqual(guard.units, ("cron.service",))

        with self.assertRaisesRegex(ValueError, "systemd_units"):
            create_experiment_quiet_mode(
                {"experiment_quiet_mode": {"systemd_units": "cron.service"}}
            )

    def test_running_package_update_blocks_and_restores_quiet_mode(self):
        calls = []

        def runner(command, **_kwargs):
            calls.append(command)
            active = command[-1] in {"cron.service", "apt-daily.service"}
            returncode = int(not active) if "is-active" in command else 0
            return subprocess.CompletedProcess(command, returncode, "", "")

        guard = ExperimentQuietMode(("cron.service",), runner=runner)
        with (
            patch("interface.experiment_quiet_mode.sys.platform", "linux"),
            patch("interface.experiment_quiet_mode.os.geteuid", return_value=0),
            self.assertRaisesRegex(RuntimeError, "apt-daily.service"),
        ):
            guard.enter()

        self.assertIn(["systemctl", "start", "cron.service"], calls)
        self.assertFalse(guard.active)


if __name__ == "__main__":
    unittest.main()
