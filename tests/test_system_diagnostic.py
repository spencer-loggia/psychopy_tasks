import types
import unittest
from unittest.mock import Mock, patch

from interface.touch_interface import format_diagnostic_report
from task.system_diagnostic import (
    evaluate_flip_lock,
    parse_xrandr_active_refresh_rate,
    pin_diagnostic_to_cpu_zero,
    probe_gpio,
    probe_piplate,
    run_display_diagnostic,
)


class SystemDiagnosticTests(unittest.TestCase):
    def test_xrandr_active_mode_is_the_monitor_refresh_reference(self):
        xrandr_output = """\
HDMI-1 connected primary 1920x1080+0+0
   1920x1080     60.33*+  59.94
DSI-1 connected 1280x800+1920+0
   1280x800      59.99*+
"""

        self.assertEqual(
            parse_xrandr_active_refresh_rate(xrandr_output, "HDMI-1"),
            60.33,
        )

    def test_stable_one_refresh_intervals_pass_flip_lock(self):
        passed, metrics = evaluate_flip_lock(120.0, [1.0 / 120.0] * 120)

        self.assertTrue(passed)
        self.assertEqual(metrics["sample_count"], 120)
        self.assertEqual(metrics["locked_fraction"], 1.0)
        self.assertEqual(metrics["dropped_interval_count"], 0)

    def test_half_rate_flips_fail_requested_refresh_lock(self):
        passed, metrics = evaluate_flip_lock(120.0, [1.0 / 60.0] * 120)

        self.assertFalse(passed)
        self.assertEqual(metrics["locked_fraction"], 0.0)
        self.assertEqual(metrics["dropped_interval_count"], 120)

    def test_gpio_probe_opens_and_closes_chip_zero(self):
        calls = []
        fake_lgpio = types.SimpleNamespace(
            gpiochip_open=lambda chip: calls.append(("open", chip)) or 7,
            gpiochip_close=lambda handle: calls.append(("close", handle)),
        )

        with patch("task.system_diagnostic.importlib.import_module", return_value=fake_lgpio):
            result = probe_gpio()

        self.assertEqual(result["status"], "pass")
        self.assertEqual(calls, [("open", 0), ("close", 7)])

    def test_piplate_probe_reads_non_mutating_supply_channel(self):
        calls = []
        fake_daq = types.SimpleNamespace(
            getADC=lambda address, channel: calls.append((address, channel)) or 5.03
        )

        with patch("task.system_diagnostic.importlib.import_module", return_value=fake_daq):
            result = probe_piplate(address=2)

        self.assertEqual(result["status"], "pass")
        self.assertEqual(calls, [(2, 8)])
        self.assertIn("5.030 V", result["detail"])

    def test_cpu_affinity_check_requires_cpu_zero_to_be_applied(self):
        with (
            patch(
                "bin.affinity.build_main_and_worker_affinity_plan",
                return_value={"supported": True, "main_cpu_affinity": [0]},
            ),
            patch(
                "bin.affinity.set_process_cpu_affinity",
                return_value=(True, "current process cpu_affinity=[0]"),
            ) as set_affinity,
        ):
            result = pin_diagnostic_to_cpu_zero()

        self.assertEqual(result["status"], "pass")
        set_affinity.assert_called_once_with([0])

    def test_display_diagnostic_compares_independent_flips_to_measured_rate(self):
        win = Mock()
        win.waitBlanking = True
        win.getActualFrameRate.return_value = 120.0
        visual = types.SimpleNamespace(Window=Mock(return_value=win))

        with (
            patch("bin.screen.load_screen_config", return_value={}),
            patch("bin.screen.resolve_task_screens", return_value=(Mock(), None)),
            patch("bin.screen.get_psychopy_window_kwargs", return_value={}),
            patch("bin.screen.enforce_window_vsync", return_value=True),
            patch("bin.task_lifecycle.signal_task_window_ready"),
            patch(
                "task.system_diagnostic.query_main_monitor_refresh_rate",
                return_value=(120.0, "xrandr active mode for output HDMI-1"),
            ),
            patch(
                "task.system_diagnostic._measure_flip_intervals",
                return_value=[1.0 / 120.0] * 120,
            ),
        ):
            checks, refresh_rate, metrics = run_display_diagnostic(
                cfg={},
                visual_module=visual,
            )

        self.assertEqual([check["status"] for check in checks], ["pass"] * 3)
        self.assertEqual(refresh_rate, 120.0)
        self.assertEqual(metrics["locked_fraction"], 1.0)
        win.close.assert_called_once_with()

    def test_display_diagnostic_does_not_self_validate_an_estimated_rate(self):
        win = Mock()
        win.waitBlanking = True
        win.getActualFrameRate.return_value = None
        visual = types.SimpleNamespace(Window=Mock(return_value=win))

        with (
            patch("bin.screen.load_screen_config", return_value={}),
            patch("bin.screen.resolve_task_screens", return_value=(Mock(), None)),
            patch("bin.screen.get_psychopy_window_kwargs", return_value={}),
            patch("bin.screen.enforce_window_vsync", return_value=True),
            patch("bin.task_lifecycle.signal_task_window_ready"),
            patch(
                "task.system_diagnostic.query_main_monitor_refresh_rate",
                return_value=(None, "xrandr unavailable"),
            ),
            patch(
                "task.system_diagnostic._measure_flip_intervals",
                return_value=[1.0 / 60.0] * 120,
            ),
        ):
            checks, refresh_rate, metrics = run_display_diagnostic(
                cfg={},
                visual_module=visual,
            )

        self.assertEqual(
            [check["status"] for check in checks],
            ["pass", "fail", "skip"],
        )
        self.assertAlmostEqual(refresh_rate, 60.0)
        self.assertIsNone(metrics)

    def test_xrandr_rate_exposes_psychopy_every_other_refresh_failure(self):
        win = Mock()
        win.waitBlanking = True
        win.getActualFrameRate.return_value = 28.827
        visual = types.SimpleNamespace(Window=Mock(return_value=win))

        with (
            patch("bin.screen.load_screen_config", return_value={}),
            patch("bin.screen.resolve_task_screens", return_value=(Mock(), None)),
            patch("bin.screen.get_psychopy_window_kwargs", return_value={}),
            patch("bin.screen.enforce_window_vsync", return_value=True),
            patch("bin.task_lifecycle.signal_task_window_ready"),
            patch(
                "task.system_diagnostic.query_main_monitor_refresh_rate",
                return_value=(60.33, "xrandr active mode for output HDMI-1"),
            ),
            patch(
                "task.system_diagnostic._measure_flip_intervals",
                return_value=[0.031253] * 120,
            ),
        ):
            checks, refresh_rate, metrics = run_display_diagnostic(
                cfg={},
                visual_module=visual,
            )

        self.assertEqual(refresh_rate, 60.33)
        self.assertEqual(
            [check["status"] for check in checks],
            ["pass", "pass", "fail"],
        )
        self.assertEqual(metrics["locked_fraction"], 0.0)
        self.assertIn("PsychoPy measured 28.827 Hz", checks[2]["detail"])
        self.assertIn("median interval error", checks[2]["error"])

    def test_report_places_refresh_rate_and_errors_in_completion_summary(self):
        report = format_diagnostic_report(
            {
                "success": False,
                "refresh_rate_hz": 119.98,
                "checks": [
                    {
                        "name": "PsychoPy",
                        "status": "pass",
                        "detail": "available",
                    },
                    {
                        "name": "Flip synchronization",
                        "status": "fail",
                        "detail": "80.0% locked",
                        "error": "Flips were not refresh locked",
                    },
                ],
            }
        )

        self.assertIn("Main monitor refresh rate: 119.980 Hz", report)
        self.assertIn("[FAIL] Flip synchronization", report)
        self.assertIn("Errors:\n- Flip synchronization: Flips were not refresh locked", report)


if __name__ == "__main__":
    unittest.main()
