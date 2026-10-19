import datetime as dt
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

from bin.eye_tracking import (
    EXPERIMENT_BLOCK_NUMBER_ENV,
    EXPERIMENT_UART_PENDING_ENV,
    EXPERIMENT_UART_READY_ENV,
    UARTEyeTracker,
    UARTEyeTrackerConfig,
)
from bin.eye_tracker_wire.protocol import StatusFlag
from interface.experiment_eye_tracker import ExperimentEyeTracker, request_eye_tracker_shutdown


class ExperimentEyeTrackerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.config = {"eye_tracker_experiment": {"enabled": True, "uart": {"device": "/dev/test", "baud": 460800}}}
        self.started_at = dt.datetime(2026, 10, 14, 10, 30, 0, 123456, tzinfo=dt.timezone(dt.timedelta(hours=-4)))
        warning = patch("bin.eye_tracker_warnings.show_eye_tracker_warning")
        self.warning = warning.start()
        self.addCleanup(warning.stop)
        channel = patch("interface.experiment_eye_tracker.SerialPacketChannel")
        self.channel = channel.start()
        self.addCleanup(channel.stop)
        controller = patch("interface.experiment_eye_tracker.EyeTrackerController")
        self.controller = controller.start().return_value
        self.addCleanup(controller.stop)
        self.controller.start_experiment.return_value = Mock(flags=StatusFlag.EXPERIMENT_ACTIVE)
        self.controller.stop_experiment.return_value = Mock(flags=StatusFlag(0))
        self.controller.shutdown.return_value = Mock(flags=StatusFlag.SHUTTING_DOWN)

    def test_shutdown_uses_configured_uart_without_an_active_experiment(self):
        request_eye_tracker_shutdown(self.config).result(timeout=2)
        self.channel.assert_called_once_with("/dev/test", baud=460800)
        self.controller.shutdown.assert_called_once_with()
        self.controller.start_experiment.assert_not_called()
        self.channel.return_value.__exit__.assert_called_once()
        self.warning.assert_not_called()

    def test_shutdown_waits_behind_pending_experiment_stop(self):
        entered, release = threading.Event(), threading.Event()

        def slow_stop():
            entered.set()
            release.wait(2)
            return Mock(flags=StatusFlag(0))

        tracker = self.create()
        tracker.start_future.result(timeout=2)
        self.controller.stop_experiment.side_effect = slow_stop
        tracker.finish()
        future = request_eye_tracker_shutdown(self.config)
        try:
            self.assertTrue(entered.wait(1))
            self.assertFalse(future.done())
            self.controller.shutdown.assert_not_called()
        finally:
            release.set()
            future.result(timeout=2)
        self.controller.shutdown.assert_called_once()

    def test_shutdown_failures_warn_without_failing_future(self):
        self.controller.shutdown.side_effect = RuntimeError("no acknowledgement")
        request_eye_tracker_shutdown(self.config).result(timeout=2)
        self.warning.assert_called_once()
        self.assertIn("Local shutdown will continue", self.warning.call_args.args[0])

    def test_shutdown_requires_shutting_down_flag_without_error(self):
        for flags in (StatusFlag(0), StatusFlag.SHUTTING_DOWN | StatusFlag.ERROR):
            with self.subTest(flags=flags):
                self.warning.reset_mock()
                self.controller.shutdown.return_value = Mock(flags=flags)
                request_eye_tracker_shutdown(self.config).result(timeout=2)
                self.warning.assert_called_once()

    def test_shutdown_disabled_and_simulation_never_send_uart(self):
        for config in ({}, {"eye_tracker_experiment": {"enabled": False}},
                       {"eye_tracker_experiment": {"enabled": True, "uart": {"simulate": True}}}):
            self.assertIsNone(request_eye_tracker_shutdown(config))
        self.channel.assert_not_called()

    def create(self, **kwargs):
        return ExperimentEyeTracker(
            kwargs.get("config", self.config), subject=kwargs.get("subject", "Yuri"),
            started_at=self.started_at, directory=self.directory,
            experiment_number=kwargs.get("experiment_number", 7),
        )

    def test_subject_utc_date_and_stop_share_configured_uart_and_release_it(self):
        tracker = self.create()
        tracker.start_future.result(timeout=2)
        self.controller.start_experiment.assert_called_once_with(
            "Yuri", timestamp_ns=1791988200123456000, experiment_number=7,
        )
        self.assertFalse(tracker.pending_path.exists())
        self.assertTrue(tracker.ready_path.is_file())
        tracker.finish()
        tracker.finish()
        tracker.stop_future.result(timeout=2)
        self.controller.stop_experiment.assert_called_once_with()
        self.assertEqual(self.channel.call_count, 2)
        self.channel.assert_called_with("/dev/test", baud=460800)
        self.assertEqual(self.channel.return_value.__exit__.call_count, 2)
        self.warning.assert_not_called()

    def test_commands_do_not_wait_on_ui_and_end_is_ordered_after_start(self):
        entered, release = threading.Event(), threading.Event()

        def slow_start(*args, **kwargs):
            entered.set()
            release.wait(2)
            return Mock(flags=StatusFlag.EXPERIMENT_ACTIVE)

        self.controller.start_experiment.side_effect = slow_start
        tracker = self.create()
        try:
            self.assertTrue(entered.wait(1))
            self.assertTrue(tracker.pending_path.exists())
            tracker.finish()
            self.assertFalse(tracker.stop_future.done())
            self.controller.stop_experiment.assert_not_called()
        finally:
            release.set()
            tracker.start_future.result(timeout=2)
            tracker.stop_future.result(timeout=2)

    def test_unavailable_uart_warns_but_start_and_end_complete(self):
        self.channel.side_effect = OSError("no UART")
        tracker = self.create()
        tracker.start_future.result(timeout=2)
        self.assertFalse(tracker.pending_path.exists())
        self.assertFalse(tracker.ready_path.exists())
        tracker.finish()
        tracker.stop_future.result(timeout=2)
        self.assertEqual(self.warning.call_count, 2)
        self.assertIn("no UART", self.warning.call_args.args[0])

    def test_rejected_start_still_sends_end_to_clean_up_uncertain_remote_state(self):
        self.controller.start_experiment.side_effect = RuntimeError("lost acknowledgement")
        tracker = self.create()
        tracker.start_future.result(timeout=2)
        self.assertFalse(tracker.ready_path.exists())
        tracker.finish()
        tracker.stop_future.result(timeout=2)
        self.controller.stop_experiment.assert_called_once()
        self.warning.assert_called_once()

    def test_unexpected_status_flags_warn_without_failing_futures(self):
        self.controller.start_experiment.return_value = Mock(flags=StatusFlag.ERROR)
        self.controller.stop_experiment.return_value = Mock(flags=StatusFlag.RECORDING)
        tracker = self.create()
        tracker.start_future.result(timeout=2)
        tracker.finish()
        tracker.stop_future.result(timeout=2)
        self.assertEqual(self.warning.call_count, 2)
        self.assertIn("recording may not have been flushed", self.warning.call_args.args[0])
        self.assertFalse(tracker.ready_path.exists())

    def test_unconfirmed_experiment_prevents_numbered_tracking_in_previous_experiment(self):
        for flags in (StatusFlag(0), StatusFlag.ERROR | StatusFlag.EXPERIMENT_ACTIVE):
            with self.subTest(flags=flags):
                self.controller.start_experiment.return_value = Mock(flags=flags)
                tracker = self.create()
                tracker.start_future.result(timeout=2)
                with patch.dict("os.environ", {
                    EXPERIMENT_UART_PENDING_ENV: str(tracker.pending_path),
                    EXPERIMENT_UART_READY_ENV: str(tracker.ready_path),
                    EXPERIMENT_BLOCK_NUMBER_ENV: "3",
                }), patch("bin.eye_tracking.SerialPacketChannel") as channel:
                    task_tracker = UARTEyeTracker()
                    task_tracker._run()
                channel.assert_not_called()
                self.assertIn("experiment start was not confirmed", task_tracker.get_state().error)

    def test_disabled_simulated_and_invalid_settings_do_not_open_uart(self):
        for config in ({}, {"eye_tracker_experiment": {"enabled": True, "uart": {"simulate": True}}}, {"eye_tracker_experiment": {"enabled": "yes"}}):
            with self.subTest(config=config):
                tracker = self.create(config=config)
                tracker.finish()
                self.assertIsNone(tracker.start_future)
                self.assertIsNone(tracker.stop_future)
        self.channel.assert_not_called()
        self.warning.assert_called_once()

    def test_task_worker_waits_for_start_handoff_without_waiting_on_task_thread(self):
        pending = self.directory / "pending"
        pending.touch()
        opened = threading.Event()

        def open_channel(*args, **kwargs):
            opened.set()
            raise OSError("no UART")

        with patch.dict("os.environ", {EXPERIMENT_UART_PENDING_ENV: str(pending)}), patch(
            "bin.eye_tracking.SerialPacketChannel", side_effect=open_channel
        ):
            task_tracker = UARTEyeTracker(uart_config=UARTEyeTrackerConfig())
            task_tracker.start()
            try:
                self.assertFalse(opened.wait(0.1))
                self.assertFalse(task_tracker.get_state().valid)
                pending.unlink()
                self.assertTrue(opened.wait(1))
            finally:
                task_tracker.stop()
        self.assertEqual(task_tracker.get_state().error, "no UART")


if __name__ == "__main__":
    unittest.main()
