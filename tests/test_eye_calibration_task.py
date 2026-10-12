"""Exercise calibration wiring and pump cleanup without PsychoPy or hardware."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import bin
from bin.eye_tracking import EyeTrackerState, UARTEyeTrackerConfig
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
        preview.poll.side_effect = [False, True]
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
        ) as preview_class, patch.object(
            task, "UARTEyeTracker", return_value=tracker
        ), patch.object(
            task, "DAQC2DigitalOutputs", return_value=outputs
        ):
            result = task.run_task(
                output_dir=tmp,
                fullscreen=False,
                win_size=(800, 600),
                uart_config=UARTEyeTrackerConfig(simulate=True),
            )
            saved = json.loads(result.read_text())["eye_tracker_calibration"]
        self.assertEqual(saved["main_screen_size_px"], [600, 800])
        self.assertEqual(saved["main_native_size_px"], [800, 600])
        self.assertEqual(saved["main_rotation_deg"], 90)
        self.assertEqual(saved["input_units"], "crop_local_pixels")
        self.assertAlmostEqual(saved["x_offset"], -0.25)
        self.assertAlmostEqual(saved["y_offset"], 0)
        self.assertEqual(
            preview_class.call_args.kwargs["calibration_config"]["subject_size"],
            (600, 800),
        )
        self.assertEqual(task._draw_fixation_cross.call_args.args[1], (-200, -150))
        task.utils.verify_task_window_vblank.assert_called_once_with(win)
        task.utils.close_task_window.assert_called_once_with(win)
        tracker.stop.assert_called_once()
        preview.close.assert_called_once()
        self.assertEqual(outputs.write.call_args.args, (0, False))

    def test_legacy_gpio_pin_rejected_before_opening_windows(self):
        with self.assertRaisesRegex(ValueError, "DOUT bit"):
            self.task.run_task(pump_pin=17)
        self.task.utils.setup_task_window.assert_not_called()

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


if __name__ == "__main__":
    unittest.main()
