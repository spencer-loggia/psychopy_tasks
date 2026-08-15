import types
import unittest
from unittest.mock import Mock, patch

from interface.touch_interface import format_diagnostic_report
from task.system_diagnostic import (
    _measure_flip_phase,
    evaluate_flip_lock,
    parse_xrandr_active_refresh_rate,
    pin_diagnostic_to_cpu_zero,
    prepare_diagnostic_affinity,
    probe_gpio,
    probe_piplate,
    query_main_monitor_refresh_rate,
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

    def test_refresh_query_uses_the_resolved_main_output(self):
        screen = types.SimpleNamespace(
            name="HDMI-1", width=1920, height=1080, x=0, y=0
        )
        completed = types.SimpleNamespace(stdout="""\
HDMI-1 connected primary 1920x1080+0+0
   1920x1080     60.33*+
DSI-1 connected 1280x800+1920+0
   1280x800      59.99*+
""")

        with patch("task.system_diagnostic.subprocess.run", return_value=completed):
            rate, detail = query_main_monitor_refresh_rate(screen)

        self.assertEqual(rate, 60.33)
        self.assertIn("resolved main output HDMI-1", detail)

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

    def test_flip_phase_reports_retraces_per_completed_swap(self):
        with (
            patch(
                "task.system_diagnostic.query_glx_sync_values",
                side_effect=[
                    ({"ust": 1_000_000, "msc": 60, "sbc": 10}, "OML"),
                    ({"ust": 3_000_000, "msc": 180, "sbc": 70}, "OML"),
                ],
            ),
            patch(
                "task.system_diagnostic._measure_flip_intervals",
                return_value=[1.0 / 30.0] * 60,
            ),
        ):
            _intervals, progress, detail = _measure_flip_phase(Mock())

        self.assertEqual(detail, "OML")
        self.assertEqual(progress["delta_msc"], 120)
        self.assertEqual(progress["delta_sbc"], 60)
        self.assertEqual(progress["msc_per_completed_swap"], 2.0)
        self.assertEqual(progress["msc_rate_hz"], 60.0)

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

    def test_diagnostic_stages_window_creation_off_cpu_zero(self):
        plan = {
            "supported": True,
            "main_cpu_affinity": [0],
            "worker_cpu_affinity": [1, 2, 3],
        }
        with (
            patch(
                "bin.affinity.build_main_and_worker_affinity_plan",
                return_value=plan,
            ),
            patch(
                "bin.affinity.set_process_cpu_affinity",
                return_value=(True, "current process cpu_affinity=[1,2,3]"),
            ) as set_affinity,
        ):
            actual_plan, preparation = prepare_diagnostic_affinity()

        self.assertIs(actual_plan, plan)
        self.assertTrue(preparation[0])
        set_affinity.assert_called_once_with([1, 2, 3])

    def test_display_diagnostic_compares_independent_flips_to_measured_rate(self):
        win = Mock()
        win._neuro_tasks_screen_placement = "HDMI-1 at (0, 0, 1920, 1080)"
        win.waitBlanking = True
        visual = types.SimpleNamespace(Window=Mock(return_value=win))

        with (
            patch("bin.screen.load_screen_config", return_value={}),
            patch("bin.screen.resolve_task_screens", return_value=(Mock(), None)),
            patch(
                "bin.screen.open_psychopy_window",
                return_value=win,
            ),
            patch("bin.screen.enforce_window_vsync", return_value=True),
            patch(
                "task.system_diagnostic.query_glx_swap_interval",
                return_value=(1, "GLX_EXT_swap_control"),
            ),
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

        self.assertEqual([check["status"] for check in checks], ["pass"] * 4)
        self.assertEqual(refresh_rate, 120.0)
        self.assertEqual(metrics["locked_fraction"], 1.0)
        win.close.assert_called_once_with()

    def test_display_diagnostic_does_not_self_validate_an_estimated_rate(self):
        win = Mock()
        win._neuro_tasks_screen_placement = "HDMI-1 at (0, 0, 1920, 1080)"
        win.waitBlanking = True
        visual = types.SimpleNamespace(Window=Mock(return_value=win))

        with (
            patch("bin.screen.load_screen_config", return_value={}),
            patch("bin.screen.resolve_task_screens", return_value=(Mock(), None)),
            patch(
                "bin.screen.open_psychopy_window",
                return_value=win,
            ),
            patch("bin.screen.enforce_window_vsync", return_value=True),
            patch(
                "task.system_diagnostic.query_glx_swap_interval",
                return_value=(1, "GLX_EXT_swap_control"),
            ),
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
            ["pass", "pass", "fail", "fail"],
        )
        self.assertIsNone(refresh_rate)
        self.assertIsNone(metrics)
        self.assertIn("Recorded 120 flip intervals", checks[3]["detail"])

    def test_xrandr_rate_exposes_psychopy_every_other_refresh_failure(self):
        win = Mock()
        win._neuro_tasks_screen_placement = "HDMI-1 at (0, 0, 1920, 1080)"
        win.waitBlanking = True
        visual = types.SimpleNamespace(Window=Mock(return_value=win))

        with (
            patch("bin.screen.load_screen_config", return_value={}),
            patch("bin.screen.resolve_task_screens", return_value=(Mock(), None)),
            patch(
                "bin.screen.open_psychopy_window",
                return_value=win,
            ),
            patch("bin.screen.enforce_window_vsync", return_value=True),
            patch(
                "task.system_diagnostic.query_glx_swap_interval",
                return_value=(2, "GLX_EXT_swap_control"),
            ),
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
            ["pass", "fail", "pass", "fail"],
        )
        self.assertIn("swap interval 2", checks[1]["detail"])
        self.assertEqual(metrics["locked_fraction"], 0.0)
        self.assertIn("simple-flip median 31.253 ms (31.997 Hz)", checks[3]["detail"])
        self.assertIn("median interval error", checks[3]["error"])

    def test_window_creation_failure_stops_timing_but_reports_main_rate(self):
        win = Mock()
        visual = types.SimpleNamespace(Window=Mock(return_value=win))

        with (
            patch("bin.screen.load_screen_config", return_value={}),
            patch("bin.screen.resolve_task_screens", return_value=(Mock(), None)),
            patch(
                "bin.screen.open_psychopy_window",
                side_effect=RuntimeError("window realized on HDMI-1"),
            ),
            patch(
                "task.system_diagnostic.query_main_monitor_refresh_rate",
                return_value=(60.33, "xrandr active mode for output HDMI-2"),
            ),
            patch("task.system_diagnostic._measure_flip_intervals") as measure,
        ):
            checks, refresh_rate, metrics = run_display_diagnostic(
                cfg={},
                visual_module=visual,
            )

        self.assertEqual(
            [check["status"] for check in checks],
            ["fail", "skip", "pass", "skip"],
        )
        self.assertIn("window realized on HDMI-1", checks[0]["error"])
        self.assertEqual(refresh_rate, 60.33)
        self.assertIsNone(metrics)
        measure.assert_not_called()

    def test_wrong_monitor_still_runs_labeled_timing_checks(self):
        main = types.SimpleNamespace(name="HDMI-2")
        realized = types.SimpleNamespace(name="HDMI-1")
        win = Mock()
        win._neuro_tasks_screen_placement = "HDMI-1 at (1600, 0, 1920, 1080)"
        win._neuro_tasks_screen_placement_error = (
            "RuntimeError: native window did not cover HDMI-2"
        )
        win._neuro_tasks_realized_screen = realized
        win._neuro_tasks_fullscreen_path = "native PsychoPy fullscreen"
        win.waitBlanking = True
        visual = types.SimpleNamespace(Window=Mock(return_value=win))

        with (
            patch("bin.screen.load_screen_config", return_value={}),
            patch("bin.screen.resolve_task_screens", return_value=(main, None)),
            patch("bin.screen.open_psychopy_window", return_value=win) as open_window,
            patch("bin.screen.enforce_window_vsync", return_value=True),
            patch(
                "task.system_diagnostic.query_glx_swap_interval",
                return_value=(1, "GLX_EXT_swap_control"),
            ),
            patch("bin.task_lifecycle.signal_task_window_ready"),
            patch(
                "task.system_diagnostic.query_main_monitor_refresh_rate",
                side_effect=[
                    (60.33, "xrandr main HDMI-2"),
                    (60.0, "xrandr realized HDMI-1"),
                ],
            ),
            patch(
                "task.system_diagnostic._measure_flip_intervals",
                return_value=[1.0 / 60.0] * 120,
            ) as measure,
        ):
            checks, refresh_rate, metrics = run_display_diagnostic(
                cfg={},
                visual_module=visual,
            )

        self.assertEqual(
            [check["status"] for check in checks],
            ["fail", "pass", "pass", "pass"],
        )
        self.assertEqual(refresh_rate, 60.33)
        self.assertEqual(metrics["timing_output_name"], "HDMI-1")
        self.assertFalse(metrics["main_display_placement_verified"])
        self.assertIn("WRONG-OUTPUT TIMING ONLY", checks[3]["detail"])
        measure.assert_called_once_with(win)
        self.assertFalse(open_window.call_args.kwargs["require_correct_placement"])

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
