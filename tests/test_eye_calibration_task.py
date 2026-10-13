"""Exercise calibration wiring and pump cleanup without PsychoPy or hardware."""

import importlib.util
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import bin
from bin.eye_tracking import EyeCalibration, EyeTrackerState, UARTEyeTrackerConfig
from bin.eye_tracking_config import load_eye_tracking_settings
from bin.screen import ScreenGeometry


def load_task():
    """Stub only display imports; use real geometry, config and session logging."""
    psychopy = types.ModuleType("psychopy")
    psychopy.core = types.SimpleNamespace(wait=lambda _: None)
    psychopy.event = types.SimpleNamespace(getKeys=lambda *args, **kwargs: [])
    psychopy.logging = types.SimpleNamespace(console=Mock(), CRITICAL=50)
    utils = types.ModuleType("bin.utils")
    for name in (
        "setup_task_window",
        "close_task_window",
        "make_bg_rect",
        "verify_task_window_vblank",
    ):
        setattr(utils, name, Mock())
    renderer = types.ModuleType("bin.eye_calibration_preview")
    renderer._make_fixation_cross = Mock(return_value=[Mock(), Mock()])
    renderer._draw_fixation_cross = Mock()
    path = Path(__file__).resolve().parents[1] / "task/calibrate_eye_tracker.py"
    spec = importlib.util.spec_from_file_location("_eye_calibration_under_test", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(
        sys.modules,
        {
            "psychopy": psychopy,
            "bin.utils": utils,
            "bin.eye_calibration_preview": renderer,
        },
    ), patch.object(bin, "utils", utils, create=True):
        spec.loader.exec_module(module)
    return module


class EyeCalibrationTaskTests(unittest.TestCase):
    def setUp(self):
        self.task = load_task()

    def test_reward_clears_dout_even_if_logger_raises(self):
        outputs = Mock()
        logger = Mock()
        logger.log_signal.side_effect = RuntimeError("log failed")
        with self.assertRaises(RuntimeError):
            self.task._deliver_reward_pulse(
                outputs=outputs, pump_pin=3, pulse_duration_s=0.2, logger=logger
            )
        self.assertEqual(
            [call.args for call in outputs.write.call_args_list],
            [(3, True), (3, False)],
        )

    def test_reward_exit_shortens_pulse_and_zero_duration_never_activates(self):
        outputs, logger = Mock(), Mock()
        self.task._deliver_reward_pulse(
            outputs=outputs,
            pump_pin=0,
            pulse_duration_s=5,
            logger=logger,
            should_exit=lambda: True,
        )
        self.assertEqual(
            [call.args for call in outputs.write.call_args_list],
            [(0, True), (0, False)],
        )
        self.assertEqual(
            [call.kwargs["event"] for call in logger.log_signal.call_args_list],
            ["pump_on", "pump_off"],
        )
        outputs.reset_mock()
        self.task._deliver_reward_pulse(
            outputs=outputs, pump_pin=0, pulse_duration_s=0, logger=logger
        )
        outputs.write.assert_not_called()

    def test_task_uses_shared_preview_windowed_rotation_and_uart_metadata(self):
        task = self.task
        win = Mock(size=(800, 600))
        main = ScreenGeometry(
            index=0, name="main", width=1920, height=1080, x=0, y=0, rotation="normal"
        )
        secondary = ScreenGeometry(
            index=1, name="exp", width=1280, height=1024, x=1920, y=0
        )
        task.utils.setup_task_window.return_value = win, main, secondary
        preview, tracker, outputs = Mock(), Mock(), Mock()
        preview.poll.side_effect = [False, False, False, False, True]
        preview.consume_manual_reward_request.return_value = False
        first_controls = {
            "x_scale": 0.005,
            "y_scale": -0.005,
            "fix_x": 0.25,
            "fix_y": -0.25,
            "revision": 1,
            "zero_count": 1,
        }
        preview.get_calibration_controls.side_effect = [
            first_controls,
            {
                **first_controls,
                "revision": 2,
                "attention_count": 1,
                "fix_x": -0.2,
                "fix_y": 0.2,
                "x_scale": 0.01,
                "y_scale": -0.015,
            },
            {
                **first_controls,
                "revision": 3,
                "x_scale": 0,
                "y_scale": 0,
                "attention_count": 1,
            },
            {
                **first_controls,
                "revision": 4,
                "x_scale": -0.01,
                "y_scale": 0.01,
                "attention_count": 1,
            },
        ]
        outputs.validate_bit.return_value = 0
        first_state = EyeTrackerState(
            timestamp_perf_s=1,
            sample_index=1,
            valid=True,
            x_smooth_px=100,
            y_smooth_px=50,
        )
        tracker.get_state.side_effect = [first_state] + [
            replace(
                first_state,
                x_smooth_px=110,
                y_smooth_px=60,
                sample_index=i,
                timestamp_perf_s=1 + i * 0.01,
            )
            for i in (2, 3, 4)
        ]
        clock = [0.0]
        f_keys = iter([["f"], [], [], []])

        def get_keys(keys):
            if "f" in keys:
                clock[0] += 2
                return next(f_keys)
            return []

        with tempfile.TemporaryDirectory() as tmp, patch.object(
            task, "ExperimenterPreview", return_value=preview
        ) as preview_class, patch.object(
            task, "UARTEyeTracker", return_value=tracker
        ), patch.object(
            task, "DAQC2DigitalOutputs", return_value=outputs
        ), patch.object(
            task.event, "getKeys", side_effect=get_keys
        ), patch.object(
            task.time, "perf_counter", side_effect=lambda: clock[0]
        ):
            result = task.run_task(
                output_dir=tmp,
                fullscreen=False,
                win_size=(800, 600),
                uart_config=UARTEyeTrackerConfig(simulate=True),
            )
            saved = json.loads(result.read_text())["eye_tracker_calibration"]
            reloaded = load_eye_tracking_settings(
                {"eye_tracking": {"enabled": True, "calibration_file": str(result)}}
            )
        self.assertEqual(saved["main_screen_size_px"], [600, 800])
        self.assertEqual(saved["main_native_size_px"], [800, 600])
        self.assertEqual(saved["main_rotation_deg"], 90)
        self.assertEqual(saved["input_units"], "crop_local_pixels")
        self.assertAlmostEqual(saved["x_offset"], 1.25)
        self.assertAlmostEqual(saved["y_offset"], -0.75)
        self.assertEqual(reloaded.gain_anchor_px, (100, 50))
        # Moving the target, doubling gain, crossing zero and reversing sign
        # must all preserve the previously zeroed position's mapping.
        for call, expected in zip(
            preview.show_calibration_state.call_args_list,
            [(0.25, -0.25), (0.35, -0.4), (0.25, -0.25), (0.15, -0.15)],
        ):
            for actual, value in zip(call.kwargs["eye_fraction"], expected):
                self.assertAlmostEqual(actual, value)
        for actual, expected in zip(
            reloaded.calibration.map_position(100, 50), (0.25, -0.25)
        ):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(
            preview_class.call_args.kwargs["calibration_config"]["subject_size"],
            (600, 800),
        )
        self.assertEqual(
            task._draw_fixation_cross.call_args_list[0].args[1], (-200, -150)
        )
        # Main-window F starts the cue; preview-window F restarts it. An
        # unchanged request counter must not restart it on later frames.
        for draw_calls in (
            task._draw_fixation_cross.call_args_list,
            preview.show_calibration_state.call_args_list,
        ):
            self.assertEqual(
                [c.kwargs["attention_started_s"] for c in draw_calls],
                [2.0, 4.0, 4.0, 4.0],
            )
        task.utils.verify_task_window_vblank.assert_called_once_with(win)
        task.utils.close_task_window.assert_called_once_with(win)
        tracker.stop.assert_called_once()
        preview.close.assert_called_once()
        self.assertEqual(outputs.write.call_args.args, (0, False))

    def test_legacy_gpio_pin_rejected_before_opening_windows(self):
        with self.assertRaisesRegex(ValueError, "DOUT bit"):
            self.task.run_task(pump_pin=17)
        self.task.utils.setup_task_window.assert_not_called()

    def test_hotkeys_work_from_either_window_and_merge_simultaneous_requests(self):
        for source in ("main", "preview", "both"):
            for valid in (True, False):
                with self.subTest(source=source, valid=valid):
                    self._exercise_hotkeys(source, valid)

    def _exercise_hotkeys(self, source, valid):
        task = self.task
        win = Mock(size=(800, 600))
        main = ScreenGeometry(index=0, name="main", width=800, height=600, x=0, y=0)
        secondary = ScreenGeometry(
            index=1, name="exp", width=1280, height=1024, x=800, y=0
        )
        task.utils.setup_task_window.return_value = win, main, secondary
        preview, tracker = Mock(), Mock()
        # Reward polls for Escape while active; stop only after three rendered
        # frames rather than consuming a fixed number of poll calls.
        preview.poll.side_effect = lambda: win.flip.call_count >= 3
        plate = Mock()
        clock = [0.0]
        on_times, off_times = [], []
        plate.setDOUTbit.side_effect = lambda *args: on_times.append(clock[0])
        plate.clrDOUTbit.side_effect = lambda *args: off_times.append(clock[0])
        main_request = source in ("main", "both")
        preview_request = source in ("preview", "both")
        initial = EyeCalibration(
            x_scale=0.003, y_scale=-0.004, x_offset=-0.2, y_offset=0.3
        )
        controls = {
            "x_scale": initial.x_scale,
            "y_scale": initial.y_scale,
            "fix_x": 0.25,
            "fix_y": -0.1,
            "revision": 1,
            "zero_count": 0,
            "attention_count": 0,
        }
        requested = {
            **controls,
            "revision": 2 if preview_request else 1,
            "zero_count": int(preview_request),
            "attention_count": int(preview_request),
        }
        preview.get_calibration_controls.side_effect = [controls, requested, requested]
        preview.consume_manual_reward_request.side_effect = [
            False,
            preview_request,
            False,
        ]
        first = EyeTrackerState(
            timestamp_perf_s=1,
            sample_index=1,
            valid=True,
            x_smooth_px=100,
            y_smooth_px=50,
        )
        tracker.get_state.side_effect = [
            first,
            replace(
                first,
                timestamp_perf_s=1.02,
                sample_index=2,
                valid=valid,
                x_smooth_px=120,
                y_smooth_px=60,
            ),
            replace(
                first,
                timestamp_perf_s=1.04,
                sample_index=3,
                x_smooth_px=130,
                y_smooth_px=70,
            ),
        ]
        key_frames = iter([
            [], ["f", "z", "r", "right", "right", "up"] if main_request else [], []
        ])

        def get_keys(keys):
            return next(key_frames) if "f" in keys else []

        def wait(duration):
            clock[0] += duration

        with tempfile.TemporaryDirectory() as tmp, patch.object(
            task, "ExperimenterPreview", return_value=preview
        ), patch.object(task, "UARTEyeTracker", return_value=tracker), patch.dict(
            sys.modules, {"_calibration_test_plate": plate}
        ), patch.object(
            task.event, "getKeys", side_effect=get_keys
        ), patch.object(task.time, "perf_counter", side_effect=lambda: clock[0]), patch.object(
            task.core, "wait", side_effect=wait
        ):
            result = task.run_task(
                output_dir=tmp, fullscreen=False, initial_calibration=initial,
                raspi=True, daq_address=2, daq_module_name="_calibration_test_plate",
                pump_pin=3, pump_pulse_time_seconds=0.25,
            )
            saved = load_eye_tracking_settings(
                {"eye_tracking": {"enabled": True, "calibration_file": str(result)}}
            )

        # Z must work even when the preview's calibration revision never
        # changes, and stale/invalid samples must still leave offsets alone.
        self.assertEqual(saved.calibration.x_scale, 0.003)
        self.assertEqual(saved.calibration.y_scale, -0.004)
        if valid:
            for actual, target in zip(
                saved.calibration.map_position(120, 60), (0.25, -0.1)
            ):
                self.assertAlmostEqual(actual, target)
            self.assertEqual(saved.gain_anchor_px, (120, 60))
        else:
            self.assertEqual(
                (saved.calibration.x_offset, saved.calibration.y_offset), (-0.2, 0.3)
            )
            self.assertEqual(saved.gain_anchor_px, (100, 50))
        # R fires once even if both sources report it in the same frame; F
        # likewise starts once, without restarting on the following frame.
        plate.setDOUTbit.assert_called_once_with(2, 3)
        self.assertTrue(all(call.args == (2, 3) for call in plate.clrDOUTbit.call_args_list))
        self.assertEqual(len(off_times), 3)  # Startup, pulse end, task cleanup.
        self.assertAlmostEqual(off_times[1] - on_times[0], 0.25)
        if main_request:
            preview.request_calibration_scale_steps.assert_called_once_with(x_steps=2, y_steps=1)
        else:
            preview.request_calibration_scale_steps.assert_not_called()
        starts = [
            call.kwargs["attention_started_s"]
            for call in preview.show_calibration_state.call_args_list
        ]
        self.assertIsNone(starts[0])
        self.assertIsNotNone(starts[1])
        self.assertEqual(starts[1], starts[2])

    def test_preview_abort_recovers_settings_and_still_cleans_up(self):
        task = self.task
        win = Mock(size=(800, 600))
        main = ScreenGeometry(index=0, name="main", width=800, height=600, x=0, y=0)
        secondary = ScreenGeometry(
            index=1, name="exp", width=1280, height=1024, x=800, y=0
        )
        task.utils.setup_task_window.return_value = win, main, secondary
        preview, tracker, outputs = Mock(), Mock(), Mock()
        crash = RuntimeError(
            "Experimenter preview process exited unexpectedly with status -6"
        )
        preview.poll.side_effect = [False, crash]
        preview.consume_manual_reward_request.return_value = False
        preview.get_calibration_controls.return_value = {
            "x_scale": 0.005,
            "y_scale": -0.005,
            "fix_x": 0.25,
            "fix_y": -0.25,
            "revision": 1,
            "zero_count": 1,
        }
        outputs.validate_bit.return_value = 0
        tracker.get_state.return_value = EyeTrackerState(
            timestamp_perf_s=1,
            sample_index=1,
            valid=True,
            x_smooth_px=100,
            y_smooth_px=50,
        )
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            task, "ExperimenterPreview", return_value=preview
        ), patch.object(task, "UARTEyeTracker", return_value=tracker), patch.object(
            task, "DAQC2DigitalOutputs", return_value=outputs
        ):
            with self.assertRaises(RuntimeError) as raised:
                task.run_task(output_dir=tmp, fullscreen=False)
            self.assertIs(raised.exception, crash)
            # The manager must not silently import an interrupted calibration.
            self.assertEqual(list(Path(tmp).rglob("calibration.json")), [])
            (recovery,) = Path(tmp).rglob("calibration_recovery.json")
            saved = json.loads(recovery.read_text())["eye_tracker_calibration"]
            self.assertEqual(saved["completion_status"], "interrupted")
            self.assertIn("status -6", saved["interruption_error"])
            self.assertEqual(saved["gain_anchor_px"], [100, 50])
            settings = load_eye_tracking_settings(
                {"eye_tracking": {"calibration_file": str(recovery)}},
                for_calibration=True,
            )
            self.assertEqual(settings.calibration.map_position(100, 50), (0.25, -0.25))
            self.assertEqual(list(Path(tmp).rglob("*.tmp")), [])
        self.assertEqual(outputs.write.call_args.args, (0, False))
        tracker.stop.assert_called_once()
        preview.close.assert_called_once()
        task.utils.close_task_window.assert_called_once_with(win)

    def test_main_accepts_null_calibration_values_from_experiment_manager(self):
        task = self.task
        args = types.SimpleNamespace(
            config="generated.json",
            output_dir=None,
            fullscreen=None,
            win_size=None,
            main_screen=None,
            experimenter_screen=None,
            uart_device=None,
            simulate_eye=None,
            raspi=None,
        )
        config = {
            "x_scale": None,
            "y_scale": None,
            "x_offset": None,
            "y_offset": None,
            "initial_x_scale": 0.003,
            "initial_y_scale": -0.004,
        }
        with patch.object(task, "parse_args", return_value=args), patch.object(
            task, "load_config", return_value=config
        ), patch.object(task, "run_task") as run, patch("builtins.print"):
            with self.assertRaises(SystemExit):
                task.main()
        initial = run.call_args.kwargs["initial_calibration"]
        self.assertEqual(
            (initial.x_scale, initial.y_scale, initial.x_offset, initial.y_offset),
            (0.003, -0.004, 0.0, 0.0),
        )

    def test_launch_configs_enable_reward_on_pi_and_leave_desktop_disabled(self):
        root = Path(__file__).resolve().parents[1]
        for launcher, enabled in (
            ("config_files/interface/rpi_launch_config.json", True),
            ("test_configs/launch_config_rpi.json", True),
            ("test_configs/launch_config_mac.json", False),
        ):
            with self.subTest(launcher=launcher):
                config = json.loads((root / launcher).read_text())
                entry = config["tasks"]["setup"]["calibrate_eye_tracker"]
                argv = [entry["launch"], "--config", str(root / entry["config"])]
                with patch.object(sys, "argv", argv), patch.object(
                    self.task, "run_task"
                ) as run, patch("builtins.print"):
                    with self.assertRaises(SystemExit):
                        self.task.main()
                self.assertIs(run.call_args.kwargs["raspi"], enabled)
                self.assertEqual(run.call_args.kwargs["pump_pin"], 0)
                self.assertEqual(run.call_args.kwargs["pump_pulse_time_seconds"], 0.25)


if __name__ == "__main__":
    unittest.main()
