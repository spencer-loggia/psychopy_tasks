"""Calibration reuse must survive both standalone and manager-copied configs."""

from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

from bin.config import load_config
from bin.eye_tracking import EyeCalibration
from bin.eye_tracking_config import load_eye_tracking_settings


class EyeTrackingConfigTests(unittest.TestCase):
    def test_disabled_does_not_require_calibration_or_read_file(self):
        self.assertIsNone(load_eye_tracking_settings({}))
        self.assertIsNone(
            load_eye_tracking_settings(
                {"eye_tracking": {"enabled": False, "calibration_file": "missing.json"}}
            )
        )
        # State injection alone must not open a UART port for every task.
        self.assertIsNone(load_eye_tracking_settings(asdict(EyeCalibration())))

    def test_tasks_require_calibration_but_calibrator_accepts_null_placeholders(self):
        with self.assertRaisesRegex(ValueError, "x_scale is missing"):
            load_eye_tracking_settings({"eye_tracking": {"enabled": True}})
        config = dict.fromkeys(asdict(EyeCalibration()))
        config.update(initial_x_scale=0.002, initial_y_scale=None)
        settings = load_eye_tracking_settings(config, for_calibration=True)
        self.assertEqual(settings.calibration.x_scale, 0.002)
        self.assertEqual(settings.calibration.y_scale, -0.005)

    def test_relative_saved_file_settings_and_explicit_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            saved = EyeCalibration(x_offset=-0.4).to_json_dict()
            saved.update(
                uart={"device": "/dev/test", "simulate": True},
                filter={"ema_gamma": 0.25},
                gain_anchor_px=[80, 40],
            )
            (root / "calibration.json").write_text(
                json.dumps({"eye_tracker_calibration": saved})
            )
            source = root / "task.json"
            source.write_text(
                json.dumps(
                    {
                        "eye_tracking": {
                            "enabled": True,
                            "calibration_file": "calibration.json",
                            "uart": {"simulate": False},
                            "filter": {"ema_gamma": 0.1},
                        },
                        "x_scale": 0.002,
                        "x_offset": None,
                    }
                )
            )
            config = load_config(source)
            self.assertEqual(
                config["eye_tracking"]["calibration_file"],
                str((root / "calibration.json").resolve()),
            )
            copied = root / "block" / "config.json"
            copied.parent.mkdir()
            copied.write_text(json.dumps(config))
            settings = load_eye_tracking_settings(load_config(copied))
        self.assertEqual(settings.calibration.x_offset, -0.4)
        self.assertEqual(settings.calibration.x_scale, 0.002)
        self.assertEqual(settings.uart.device, "/dev/test")
        self.assertFalse(settings.uart.simulate)
        self.assertEqual(settings.filter.ema_gamma, 0.1)
        self.assertEqual(settings.gain_anchor_px, (80, 40))

    def test_latest_embedded_history_or_flat_calibration(self):
        calibration = EyeCalibration(x_offset=-0.5, y_offset=0.3)
        for fields in (
            asdict(calibration),
            {"eye_tracker_calibration": calibration.to_json_dict()},
            {
                "eye_tracker_calibration": [
                    asdict(EyeCalibration()),
                    calibration.to_json_dict(),
                ]
            },
            {
                "eye_tracking": {
                    "enabled": True,
                    "calibration": calibration.to_json_dict(),
                }
            },
        ):
            with self.subTest(fields=fields):
                config = {"eye_tracking": {"enabled": True}, **fields}
                self.assertEqual(
                    load_eye_tracking_settings(config).calibration, calibration
                )

    def test_recording_rate_defaults_to_60_and_can_be_set_independently_of_uart(self):
        cfg = {**asdict(EyeCalibration()), "eye_tracking": {"enabled": True}}
        self.assertEqual(load_eye_tracking_settings(cfg).record_rate_hz, 60)
        cfg["eye_tracking"]["record_rate_hz"] = 50
        settings = load_eye_tracking_settings(cfg)
        self.assertEqual(settings.record_rate_hz, 50)
        self.assertEqual(settings.uart.poll_rate_hz, 100)
        for value in (0, -1, float("nan"), float("inf"), True, "bad"):
            cfg["eye_tracking"]["record_rate_hz"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                load_eye_tracking_settings(cfg)

    def test_invalid_units_eye_coefficients_and_boolean_flags_rejected(self):
        for override in (
            {"input_units": "volts"},
            {"eye_id": 1},
            {"coordinate_units": "pixels"},
            {"x_scale": float("nan")},
            {"gain_anchor_px": [1]},
            {"gain_anchor_px": [1, float("inf")]},
            {"uart": {"simulate": "false"}},
            {"filter": {"ema_gamma": 1}},
        ):
            with self.subTest(override=override), self.assertRaises(ValueError):
                load_eye_tracking_settings(
                    {
                        "eye_tracking": {"enabled": True},
                        "eye_tracker_calibration": {
                            **EyeCalibration().to_json_dict(),
                            **override,
                        },
                    }
                )
        for key in ("enabled", "show_preview"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                load_eye_tracking_settings(
                    {
                        **asdict(EyeCalibration()),
                        "eye_tracking": {"enabled": True, key: "false"},
                    }
                )


if __name__ == "__main__":
    unittest.main()
