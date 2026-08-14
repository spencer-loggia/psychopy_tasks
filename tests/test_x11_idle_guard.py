import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from bin.screen import ScreenGeometry
from interface.experiment_manager import PreparedBlock
from interface.touch_interface import TouchInterfaceApp
from interface.x11_idle_guard import (
    ExperimentIdleGuard,
    XInputControlError,
    XInputTouchscreen,
    configured_main_touchscreen,
    create_experiment_idle_guard,
    mask_main_inputs_enabled,
    wait_for_task_process,
)


class X11IdleGuardTests(unittest.TestCase):
    def setUp(self):
        self.main_screen = ScreenGeometry(
            index=0,
            x=0,
            y=0,
            width=2560,
            height=1600,
            name="HDMI-1",
        )
        self.experimenter_screen = ScreenGeometry(
            index=1,
            x=2560,
            y=0,
            width=800,
            height=480,
            name="DSI-1",
        )

    def test_mask_flag_defaults_false_and_requires_boolean(self):
        self.assertFalse(mask_main_inputs_enabled({}))
        self.assertTrue(mask_main_inputs_enabled({"mask_main_inputs": True}))
        with self.assertRaisesRegex(ValueError, "must be true or false"):
            mask_main_inputs_enabled({"mask_main_inputs": "true"})

    def test_touchscreen_name_is_required_only_when_requested(self):
        self.assertIsNone(configured_main_touchscreen({}, required=False))
        with self.assertRaisesRegex(ValueError, "main_touchscreen_xinput"):
            configured_main_touchscreen({}, required=True)
        with self.assertRaisesRegex(ValueError, "stable device name"):
            configured_main_touchscreen(
                {"main_touchscreen_xinput": "12"},
                required=True,
            )

    def test_same_screen_debug_mode_ignores_missing_touchscreen_name(self):
        with patch("interface.x11_idle_guard.sys.platform", "darwin"):
            guard = create_experiment_idle_guard(
                Mock(),
                {"mask_main_inputs": True},
                self.main_screen,
                None,
                tk_module=Mock(),
            )
        self.assertIsNone(guard)

    def test_distinct_screens_require_touchscreen_name(self):
        with patch("interface.x11_idle_guard.sys.platform", "linux"):
            with patch.dict("interface.x11_idle_guard.os.environ", {"DISPLAY": ":0"}):
                with self.assertRaisesRegex(ValueError, "main_touchscreen_xinput"):
                    create_experiment_idle_guard(
                        Mock(),
                        {"mask_main_inputs": True},
                        self.main_screen,
                        self.experimenter_screen,
                        tk_module=Mock(),
                    )

    def test_xinput_resolves_name_each_time_and_maps_before_enabling(self):
        calls = []

        def runner(args, **_kwargs):
            calls.append(args)
            stdout = "27\n" if args[:3] == ["xinput", "list", "--id-only"] else ""
            return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")

        touchscreen = XInputTouchscreen(
            "Main Touchscreen",
            "HDMI-1",
            runner=runner,
        )
        touchscreen.disable()
        touchscreen.enable_for_task()
        touchscreen.enable_for_desktop()

        self.assertEqual(
            calls,
            [
                ["xinput", "list", "--id-only", "Main Touchscreen"],
                ["xinput", "disable", "27"],
                ["xinput", "list", "--id-only", "Main Touchscreen"],
                ["xinput", "map-to-output", "27", "HDMI-1"],
                ["xinput", "enable", "27"],
                ["xinput", "list", "--id-only", "Main Touchscreen"],
                ["xinput", "enable", "27"],
            ],
        )

    def test_xinput_rejects_ambiguous_device_name(self):
        def runner(args, **_kwargs):
            return subprocess.CompletedProcess(args, 0, stdout="12\n15\n", stderr="")

        touchscreen = XInputTouchscreen("Duplicate Touch", "HDMI-1", runner=runner)
        with self.assertRaisesRegex(XInputControlError, "exactly one"):
            touchscreen.validate()

    def test_guard_transitions_disable_cover_enable_and_restore(self):
        events = []
        root = Mock()
        touchscreen = Mock()
        touchscreen.disable.side_effect = lambda: events.append("disable")
        touchscreen.enable_for_task.side_effect = lambda: events.append("enable_task")
        touchscreen.enable_for_desktop.side_effect = lambda: events.append("enable_desktop")
        curtain = Mock()
        curtain.show.side_effect = lambda: events.append("show")
        curtain.hide.side_effect = lambda: events.append("hide")
        curtain.close.side_effect = lambda: events.append("close")
        guard = ExperimentIdleGuard(root, touchscreen, curtain)

        guard.enter_idle()
        guard.task_window_ready()
        guard.enter_idle()
        guard.release_for_desktop()

        self.assertEqual(
            events,
            [
                "disable",
                "show",
                "hide",
                "enable_task",
                "disable",
                "show",
                "enable_desktop",
                "close",
            ],
        )
        self.assertGreaterEqual(root.focus_force.call_count, 2)

    def test_process_wait_releases_input_when_ready_marker_appears(self):
        class FakeProcess:
            def __init__(self, ready_path):
                self.wait_calls = 0
                self.ready_path = ready_path

            def wait(self, timeout=None):
                self.wait_calls += 1
                if self.wait_calls == 1:
                    self.ready_path.write_text("ready\n")
                    raise subprocess.TimeoutExpired("task", timeout)
                return 0

        with tempfile.TemporaryDirectory() as tmpdir:
            ready_path = Path(tmpdir) / "ready"
            callback = Mock()
            returncode = wait_for_task_process(
                FakeProcess(ready_path),
                ready_path=ready_path,
                on_window_ready=callback,
                poll_interval_s=0.01,
            )

        self.assertEqual(returncode, 0)
        callback.assert_called_once_with()

    def test_interface_block_passes_ready_marker_and_restores_idle_guard(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            block_dir = Path(tmpdir) / "1_demo"
            block_dir.mkdir()
            block = PreparedBlock(
                block_num=1,
                block_name="demo",
                launch_path=Path(tmpdir) / "demo.py",
                config_path=block_dir / "config.json",
                output_dir=block_dir,
            )
            app = object.__new__(TouchInterfaceApp)
            app.python_cmd = "python"
            app.working_dir = Path(tmpdir)
            app.root = Mock()
            app.status_var = Mock()
            app.idle_guard = Mock()
            app.experiment = Mock()
            app.experiment.subprocess_environment.return_value = {"BASE": "1"}
            process = Mock()

            def complete_task(_process, **kwargs):
                kwargs["ready_path"].write_text("ready\n")
                return 0

            with patch(
                "interface.touch_interface.subprocess.Popen",
                return_value=process,
            ) as popen:
                with patch(
                    "interface.touch_interface.wait_for_task_process",
                    side_effect=complete_task,
                ) as wait_for_process:
                    result = app._run_block(block)

        self.assertEqual(result.returncode, 0)
        child_env = popen.call_args.kwargs["env"]
        self.assertEqual(child_env["BASE"], "1")
        self.assertEqual(
            child_env["NEURO_TASK_WINDOW_READY_PATH"],
            str(block.output_dir / ".task_window_ready"),
        )
        self.assertEqual(
            wait_for_process.call_args.kwargs["ready_path"],
            block.output_dir / ".task_window_ready",
        )
        self.assertIs(
            wait_for_process.call_args.kwargs["on_window_ready"],
            app.idle_guard.task_window_ready,
        )
        app.idle_guard.enter_idle.assert_called_once_with()
        app.experiment.finish_block.assert_called_once_with(block)
        self.assertFalse((block.output_dir / ".task_window_ready").exists())

    def test_desktop_exit_restores_touchscreen_before_destroying_interface(self):
        app = object.__new__(TouchInterfaceApp)
        app.task_active = False
        app.idle_guard = Mock()
        app.root = Mock()
        app.status_var = Mock()

        app._exit_to_desktop()

        app.idle_guard.release_for_desktop.assert_called_once_with()
        app.root.destroy.assert_called_once_with()

    def test_desktop_exit_stays_open_if_touchscreen_restore_fails(self):
        app = object.__new__(TouchInterfaceApp)
        app.task_active = False
        app.idle_guard = Mock()
        app.idle_guard.release_for_desktop.side_effect = XInputControlError("failed")
        app.root = Mock()
        app.status_var = Mock()

        with patch("interface.touch_interface.messagebox.showerror") as showerror:
            app._exit_to_desktop()

        app.root.destroy.assert_not_called()
        showerror.assert_called_once()


if __name__ == "__main__":
    unittest.main()
