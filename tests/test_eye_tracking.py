import unittest

from bin.eye_tracking import (
    DAQC2AnalogConfig,
    EyeCalibration,
    EyeFilterConfig,
    EyePositionFilter,
    fraction_distance_px,
    fraction_position_within_diameter,
)
from bin.logger import load_task_event_definitions


class EyeTrackingTests(unittest.TestCase):
    def test_filter_uses_exponential_moving_average(self):
        eye_filter = EyePositionFilter(
            calibration=EyeCalibration(x_scale=0.1, y_scale=0.1),
            daq_config=DAQC2AnalogConfig(voltage_min=-10.0, voltage_max=10.0),
            filter_config=EyeFilterConfig(ema_gamma=0.5),
        )

        first = eye_filter.update(2.0, 4.0, timestamp_perf_s=1.0)
        second = eye_filter.update(4.0, 2.0, timestamp_perf_s=2.0)

        self.assertEqual(first.smoothed_voltage, (2.0, 4.0))
        self.assertEqual(second.smoothed_voltage, (3.0, 3.0))
        self.assertEqual(second.position, (0.30000000000000004, 0.30000000000000004))

    def test_filter_rejects_blink_artifacts_without_updating_smooth_position(self):
        eye_filter = EyePositionFilter(
            calibration=EyeCalibration(x_scale=0.1, y_scale=0.1),
            daq_config=DAQC2AnalogConfig(voltage_min=-10.0, voltage_max=10.0),
            filter_config=EyeFilterConfig(ema_gamma=0.5),
        )

        accepted = eye_filter.update(1.0, -1.0, timestamp_perf_s=1.0)
        rejected = eye_filter.update(11.0, -1.0, timestamp_perf_s=2.0)

        self.assertFalse(accepted.last_rejected)
        self.assertTrue(rejected.last_rejected)
        self.assertEqual(rejected.rejection_reason, "out_of_range")
        self.assertEqual(rejected.smoothed_voltage, (1.0, -1.0))
        self.assertEqual(rejected.position, (0.1, -0.1))
        self.assertEqual(rejected.accepted_count, 1)
        self.assertEqual(rejected.rejected_count, 1)

    def test_calibration_offsets_are_set_from_smoothed_voltage(self):
        calibration = EyeCalibration(x_scale=0.05, y_scale=0.05)
        calibration.set_offsets_for_fixation(
            fixation_fraction=(0.1, -0.2),
            x_voltage=2.0,
            y_voltage=-4.0,
        )

        self.assertEqual(calibration.x_offset, 0.0)
        self.assertEqual(calibration.y_offset, 0.0)

    def test_calibration_task_can_log_pump_events(self):
        definitions, _ = load_task_event_definitions("calibrate_eye_tracker")

        self.assertIn("pump_on", definitions)
        self.assertIn("pump_off", definitions)

    def test_fraction_position_within_diameter_uses_screen_pixels(self):
        screen_size = (1000, 500)

        self.assertEqual(fraction_distance_px((0.0, 0.0), (0.1, 0.0), screen_size), 100.0)
        self.assertTrue(
            fraction_position_within_diameter(
                (0.0, 0.0),
                (0.04, 0.0),
                diameter_fraction=0.20,
                screen_size=screen_size,
            )
        )
        self.assertFalse(
            fraction_position_within_diameter(
                (0.0, 0.0),
                (0.06, 0.0),
                diameter_fraction=0.20,
                screen_size=screen_size,
            )
        )


if __name__ == "__main__":
    unittest.main()
