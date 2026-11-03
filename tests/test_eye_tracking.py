import json
from dataclasses import replace
from pathlib import Path
import time
import unittest
from unittest.mock import Mock, patch

from bin.eye_tracking import (
    EXPERIMENT_BLOCK_NUMBER_ENV,
    UARTEyeTracker,
    UARTEyeTrackerConfig,
    EyeCalibration,
    EyeFilterConfig,
    EyePositionFilter,
    EyeTrackerState,
    FixationWindow,
    calibration_payload,
    fraction_distance_px,
    fraction_position_within_diameter,
)
from bin.eye_tracker_wire.protocol import (
    decode_status,
    StatusFlag,
    StatusCode,
    CommandCode,
)
from bin.logger import load_task_event_definitions
from bin.task_clock import TaskClock
from bin.eye_tracker_wire.controller import ResponseTimeout

FIXTURE = json.loads((Path(__file__).parent / "fixtures/eye_uart_v1.json").read_text())
STATUS = decode_status(bytes.fromhex(FIXTURE["status"]))


def sample(frame=1, **kwargs):
    return replace(
        STATUS, frame_sequence=frame, sensor_timestamp_ns=frame * 10000000, **kwargs
    )


class EyeTrackingTests(unittest.TestCase):
    def test_poll_warnings_require_sustained_failure_and_reset_after_recovery(self):
        for failures, warn in (([1, 1], False), ([2, 2], False), ([3], True)):
            with self.subTest(failures=failures):
                self.warning.reset_mock()
                states = []
                tracker = UARTEyeTracker(
                    on_state=states.append, uart_config=UARTEyeTrackerConfig(expect_recording=False),
                )
                controller = Mock()
                controller.start_tracking.return_value = sample()
                controller.send_state_command.return_value = sample(flags=StatusFlag(0))
                replies = []
                for count in failures:
                    replies.extend([ResponseTimeout("lost telemetry")] * count)
                    replies.append(sample(len(replies) + 2))
                def poll():
                    result = replies.pop(0)
                    if not replies:
                        tracker._stop.set()
                    if isinstance(result, Exception):
                        raise result
                    return result
                controller.poll.side_effect = poll
                with patch("bin.eye_tracking.SerialPacketChannel"), patch(
                    "bin.eye_tracking.EyeTrackerController", return_value=controller,
                ):
                    tracker._run()
                self.assertTrue(any(state.rejection_reason == "uart_error" for state in states))
                self.assertTrue(states[-2].valid)
                if warn:
                    self.warning.assert_called_once()
                    self.assertIn("3 consecutive poll failures", self.warning.call_args.args[0])
                else:
                    self.warning.assert_not_called()

    def test_confirmed_stop_with_health_error_is_not_a_communication_failure(self):
        tracker = UARTEyeTracker(uart_config=UARTEyeTrackerConfig(expect_recording=False))
        controller = Mock()
        def start(**kwargs):
            tracker._stop.set()
            return sample()
        controller.start_tracking.side_effect = start
        controller.send_state_command.return_value = sample(flags=StatusFlag.ERROR)
        with patch("bin.eye_tracking.SerialPacketChannel"), patch(
            "bin.eye_tracking.EyeTrackerController", return_value=controller,
        ):
            tracker._run()
        self.warning.assert_called_once()
        self.assertIn("confirmed tracking/recording stopped", self.warning.call_args.args[0])
        self.assertIn("health error", self.warning.call_args.args[0])
        self.assertIsNone(tracker.get_state().error)

    def test_acquisition_publishes_each_camera_measurement_once_with_exact_clocks(self):
        origin = 2**60 + 123
        received = []
        tracker = UARTEyeTracker(
            task_clock=TaskClock(origin_perf_ns=origin), on_measurement=received.append,
        )
        status = replace(sample(), sensor_timestamp_ns=2**55 + 7, result_age_ms=10)
        tracker._consume_measurement(status, received_perf_ns=origin + 101, round_trip_s=0.002)
        tracker._consume_measurement(status, received_perf_ns=origin + 10_000_101)
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].task_time_ns, 101)
        self.assertEqual(received[0].camera_time_ns, 2**55 + 7)
        self.assertTrue(received[0].valid)
        blink = replace(status.eyes[0], valid=False, blink=True)
        second = replace(status, frame_sequence=2, sensor_timestamp_ns=status.sensor_timestamp_ns + 20_000_000,
                         eyes=(blink, status.eyes[1]))
        tracker._consume_measurement(second, received_perf_ns=origin + 20_000_103)
        tracker._consume_measurement(second, received_perf_ns=origin + 30_000_103)
        self.assertEqual(len(received), 2)
        self.assertEqual(received[1].task_time_ns, 20_000_103)
        self.assertEqual(received[1].camera_time_ns, second.sensor_timestamp_ns)
        self.assertEqual(received[1].rejection_reason, "blink")
        self.assertFalse(received[1].valid)
        # A new invalid/stale measurement must carry its own camera identity,
        # rather than being mislabeled with the previous valid sample's time.
        stale = replace(second, frame_sequence=3, sensor_timestamp_ns=second.sensor_timestamp_ns + 20_000_000,
                        flags=second.flags | StatusFlag.RESULT_STALE)
        tracker._consume_measurement(stale, received_perf_ns=origin + 40_000_103)
        self.assertEqual(received[-1].camera_time_ns, stale.sensor_timestamp_ns)
        self.assertEqual(received[-1].rejection_reason, "stale")
        tracker._consume_measurement(status, received_perf_ns=origin + 50_000_103)
        tracker._consume_measurement(stale, received_perf_ns=origin + 60_000_103)
        self.assertEqual(len(received), 3)

    def setUp(self):
        patcher = patch("bin.eye_tracker_warnings.show_eye_tracker_warning")
        self.warning = patcher.start()
        self.addCleanup(patcher.stop)

    def test_missing_recording_flag_warns_once_and_preserves_valid_gaze(self):
        tracker = UARTEyeTracker()
        status = sample(flags=StatusFlag.TRACKING | StatusFlag.CONFIGURED | StatusFlag.EXPERIMENT_ACTIVE)
        tracker.filter.consume(status, now_s=time.perf_counter())
        for _ in range(5):
            tracker._check_status(status)
        self.warning.assert_called_once()
        self.assertIn("recording did not start", self.warning.call_args.args[0])
        self.assertTrue(tracker.get_state().valid)

    def test_intentionally_disabled_recording_does_not_warn(self):
        tracker = UARTEyeTracker(uart_config=UARTEyeTrackerConfig(expect_recording=False))
        tracker._check_status(sample())
        self.warning.assert_not_called()

    def test_shutting_down_flag_warns_and_invalidates_gaze(self):
        tracker = UARTEyeTracker()
        status = sample(flags=StatusFlag.TRACKING | StatusFlag.CONFIGURED | StatusFlag.RECORDING | StatusFlag.SHUTTING_DOWN)
        tracker.filter.consume(status, now_s=time.perf_counter())
        tracker._check_status(status)
        self.assertFalse(tracker.get_state().valid)
        self.warning.assert_called_once()
        self.assertIn("shutting down", self.warning.call_args.args[0])

    def test_error_and_calibration_flags_warn_without_accepting_gaze(self):
        tracker = UARTEyeTracker()
        status = sample(flags=StatusFlag.CALIBRATING | StatusFlag.EXPERIMENT_ACTIVE | StatusFlag.ERROR)
        tracker.filter.consume(status, now_s=time.perf_counter())
        tracker._check_status(status)
        self.assertFalse(tracker.get_state().valid)
        self.warning.assert_called_once()
        self.assertIn("calibration mode", self.warning.call_args.args[0])

    def test_stop_failure_warns_without_raising(self):
        tracker = UARTEyeTracker()
        tracker._cleanup_error = "lost STOP acknowledgement"
        tracker.stop()
        tracker.stop()
        self.warning.assert_called_once()
        self.assertIn("lost STOP acknowledgement", self.warning.call_args.args[0])

    def test_calibration_payload_uses_pixel_units_and_eye_zero(self):
        payload = calibration_payload(
            EyeCalibration(x_scale=0.1, y_scale=-0.2, x_offset=0.3, y_offset=-0.4),
            uart_config=UARTEyeTrackerConfig(),
            filter_config=EyeFilterConfig(),
            main_screen_size=(1920, 1080),
            fixation_fraction=(0.25, -0.25),
        )
        self.assertEqual(list(payload), ["eye_tracker_calibration"])
        calibration = payload["eye_tracker_calibration"]
        self.assertEqual(calibration["input_units"], "crop_local_pixels")
        self.assertEqual(calibration["eye_id"], 0)
        self.assertEqual(calibration["x_scale"], 0.1)
        self.assertEqual(calibration["uart"]["device"], "/dev/ttyAMA10")
        self.assertNotIn("daq", calibration)

    def test_ema_only_updates_on_new_frames_and_uses_eye_zero(self):
        f = EyePositionFilter(filter_config=EyeFilterConfig(ema_gamma=0.5))
        first = f.consume(sample(), now_s=1)
        self.assertEqual(first.smoothed_position, (123.25, -45.5))
        changed = replace(STATUS.eyes[0], x=125.25, y=-43.5)
        second = f.consume(sample(2, eyes=(changed, STATUS.eyes[1])), now_s=1.03)
        self.assertEqual(second.smoothed_position, (124.25, -44.5))
        duplicate = f.consume(sample(2, eyes=(changed, STATUS.eyes[1])), now_s=1.04)
        self.assertEqual(duplicate, second)
        self.assertEqual(duplicate.frame_sequence, 2)

    def test_bad_eye_one_does_not_invalidate_eye_zero(self):
        f = EyePositionFilter()
        lost = replace(STATUS.eyes[1], valid=False, pupil_diameter=0)
        self.assertTrue(f.consume(sample(eyes=(STATUS.eyes[0], lost)), now_s=1).valid)

    def test_blink_and_loss_never_hold_valid_gaze_and_filter_resets(self):
        for blink in (True, False):
            f = EyePositionFilter(filter_config=EyeFilterConfig(ema_gamma=0.9))
            f.consume(sample(), now_s=1)
            invalid = replace(
                STATUS.eyes[0], valid=False, blink=blink, pupil_diameter=0
            )
            state = f.consume(sample(2, eyes=(invalid, STATUS.eyes[1])), now_s=1.03)
            self.assertFalse(state.valid)
            self.assertEqual(state.rejection_reason, "blink" if blink else "lost")
            changed = replace(STATUS.eyes[0], x=200)
            good = f.consume(sample(3, eyes=(changed, STATUS.eyes[1])), now_s=1.06)
            self.assertTrue(good.valid)
            self.assertEqual(good.x_smooth_px, 200)

    def test_unusable_status_is_checked_even_for_duplicate_frame(self):
        statuses = (
            sample(flags=StatusFlag.CONFIGURED),
            sample(configured_eye_count=0),
            sample(flags=STATUS.flags | StatusFlag.RESULT_STALE),
            sample(flags=STATUS.flags | StatusFlag.ERROR),
            sample(status=StatusCode.INTERNAL_ERROR),
            sample(result_age_ms=151),
        )
        for status in statuses:
            with self.subTest(status=status):
                f = EyePositionFilter()
                f.consume(sample(), now_s=1)
                self.assertFalse(f.consume(status, now_s=1.01).valid)

    def test_local_age_and_round_trip_protect_against_frozen_link(self):
        f = EyePositionFilter()
        self.assertFalse(f.get_state(now_s=1).valid)
        f.consume(sample(), now_s=1, round_trip_s=0.01)
        self.assertTrue(f.get_state(now_s=1.1).valid)
        self.assertFalse(f.get_state(now_s=1.14).valid)
        self.assertFalse(f.consume(sample(2), now_s=2, round_trip_s=0.2).valid)

    def test_restart_is_latched_and_requires_new_calibration_session(self):
        f = EyePositionFilter()
        f.consume(sample(), now_s=1)
        for frame in (2, 3):
            state = f.consume(sample(frame, session_id=999), now_s=1 + frame / 100)
            self.assertFalse(state.valid)
            self.assertEqual(state.rejection_reason, "service_restarted")

    def test_link_error_clears_on_next_new_valid_frame(self):
        f = EyePositionFilter()
        f.consume(sample(), now_s=1)
        f.invalidate("uart_error", error="timeout")
        self.assertFalse(f.consume(sample(), now_s=1.01).valid)
        state = f.consume(sample(2), now_s=1.03)
        self.assertTrue(state.valid)
        self.assertIsNone(state.error)

    def test_offsets_map_crop_pixels_to_fixation(self):
        c = EyeCalibration()
        c.set_offsets_for_fixation(fixation_fraction=(0.1, -0.2), x_px=150, y_px=75)
        x, y = c.map_position(150, 75)
        self.assertAlmostEqual(x, 0.1)
        self.assertAlmostEqual(y, -0.2)

    def test_fixation_requires_distinct_samples_and_valid_current_eye(self):
        window = FixationWindow(0.1, 0.95)
        state = EyeTrackerState(timestamp_perf_s=1, frame_sequence=1, valid=True)
        for _ in range(20):
            self.assertFalse(window.update(state, True))
        self.assertEqual(len(window.history), 1)
        state = replace(state, timestamp_perf_s=1.11, frame_sequence=2)
        self.assertTrue(window.update(state, True))
        self.assertFalse(
            window.update(replace(state, valid=False, rejection_reason="stale"), True)
        )
        self.assertEqual(len(window.history), 0)

    def test_fixation_does_not_reward_when_current_eye_is_outside(self):
        window = FixationWindow(0.1, 0.5)
        window.update(
            EyeTrackerState(timestamp_perf_s=1, frame_sequence=1, valid=True), True
        )
        state = EyeTrackerState(timestamp_perf_s=1.11, frame_sequence=2, valid=True)
        self.assertFalse(window.update(state, False))

    def test_missing_observations_cannot_complete_fixation_window(self):
        window = FixationWindow(2, 0.95)
        window.update(
            EyeTrackerState(timestamp_perf_s=1, frame_sequence=1, valid=True), True
        )
        self.assertFalse(
            window.update(
                EyeTrackerState(timestamp_perf_s=3.1, frame_sequence=2, valid=True), True
            )
        )
        self.assertEqual(len(window.history), 1)

    def test_fixation_fraction_weights_elapsed_time_instead_of_sample_count(self):
        window = FixationWindow(1, 0.85, strict=True)
        times_ms = list(range(0, 800, 100)) + list(range(800, 920)) + [920, 950, 1000]
        for index, ms in enumerate(times_ms):
            window.update(
                EyeTrackerState(timestamp_perf_s=ms / 1000, frame_sequence=index, valid=True),
                not 800 <= ms < 920,
            )
        # 120 ms outside / 1 s = 88% fixation, despite the many out-of-window
        # samples in that brief burst and relatively sparse fixating samples.
        self.assertAlmostEqual(window.acceptance, 0.88)
        self.assertTrue(window.accepts(now_s=1))

    def test_old_sensor_frame_is_rejected_even_with_new_frame_sequence(self):
        f = EyePositionFilter()
        f.consume(sample(3), now_s=1)
        self.assertFalse(f.consume(sample(2), now_s=1.01).valid)
        self.assertEqual(f.state.rejection_reason, "out_of_order")

    def test_worker_starts_polls_and_stops_with_matching_state_commands(self):
        controller = Mock()
        controller.start_tracking.return_value = sample()
        controller.send_state_command.return_value = sample()
        controller.poll.return_value = sample(2)
        with patch("bin.eye_tracking.SerialPacketChannel") as channel_class, patch(
            "bin.eye_tracking.EyeTrackerController", return_value=controller
        ):
            tracker = UARTEyeTracker()
            tracker.start()
            deadline = time.monotonic() + 1
            while not tracker.get_state().valid and time.monotonic() < deadline:
                time.sleep(0.001)
            tracker.stop()
            controller.start_tracking.assert_called_once_with(block_number=None)
            self.assertEqual(
                [c.args[0] for c in controller.send_state_command.call_args_list],
                [CommandCode.STOP_TRACKING],
            )
            channel_class.return_value.close.assert_called_once()

    def test_poll_only_mode_never_changes_remote_tracking_state(self):
        with patch("bin.eye_tracking.SerialPacketChannel"), patch(
            "bin.eye_tracking.EyeTrackerController"
        ) as cls:
            cls.return_value.poll.return_value = sample()
            tracker = UARTEyeTracker(
                uart_config=UARTEyeTrackerConfig(manage_tracking=False)
            )
            tracker.start()
            tracker.stop()
            cls.return_value.send_state_command.assert_not_called()
            cls.return_value.start_tracking.assert_not_called()

    def test_worker_starts_the_launcher_block_number(self):
        controller = Mock()
        controller.send_state_command.return_value = sample(flags=StatusFlag(0))
        tracker = UARTEyeTracker()

        def started(**kwargs):
            tracker._stop.set()
            return sample()

        controller.start_tracking.side_effect = started
        with patch.dict("os.environ", {EXPERIMENT_BLOCK_NUMBER_ENV: "7"}), patch(
            "bin.eye_tracking.SerialPacketChannel"
        ), patch("bin.eye_tracking.EyeTrackerController", return_value=controller):
            tracker._run()
        controller.start_tracking.assert_called_once_with(block_number=7)
        controller.send_state_command.assert_called_once_with(CommandCode.STOP_TRACKING)

    def test_invalid_block_context_does_not_fall_back_to_auto_numbering(self):
        for number in ("0", "-1", "1.5", "bad", str(1 << 32)):
            with self.subTest(number=number), patch.dict("os.environ", {
                EXPERIMENT_BLOCK_NUMBER_ENV: number,
            }), patch("bin.eye_tracking.SerialPacketChannel"), patch(
                "bin.eye_tracking.EyeTrackerController"
            ) as controller:
                tracker = UARTEyeTracker()
                tracker._run()
                controller.return_value.start_tracking.assert_not_called()
                controller.return_value.send_state_command.assert_not_called()
                self.assertIsNotNone(tracker.get_state().error)

    def test_invalid_config_rejected(self):
        for values in (
            {"baud": 115200},
            {"poll_rate_hz": 0},
            {"max_sample_age_s": float("nan")},
        ):
            with self.assertRaises(ValueError):
                UARTEyeTrackerConfig(**values)
        with self.assertRaises(ValueError):
            EyeFilterConfig(ema_gamma=1)

    def test_calibration_task_can_log_pump_events(self):
        definitions, _ = load_task_event_definitions("calibrate_eye_tracker")
        self.assertIn("pump_on", definitions)
        self.assertIn("pump_off", definitions)

    def test_fraction_position_within_diameter_uses_screen_pixels(self):
        self.assertEqual(fraction_distance_px((0, 0), (0.1, 0), (1000, 500)), 100)
        self.assertTrue(
            fraction_position_within_diameter(
                (0, 0), (0.04, 0), diameter_fraction=0.2, screen_size=(1000, 500)
            )
        )
        self.assertFalse(
            fraction_position_within_diameter(
                (0, 0), (0.06, 0), diameter_fraction=0.2, screen_size=(1000, 500)
            )
        )


if __name__ == "__main__":
    unittest.main()
