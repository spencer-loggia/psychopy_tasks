"""Boundary timing, reliable preview edits, and exact runtime-setting audit logs."""
import csv
import json
import multiprocessing as mp
import os
import queue
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

from bin.live_settings import LiveSettings, SettingSpec, reward_settings
from bin.logger import SessionLogBundle
from bin.screen import ExperimenterPreview
from bin.task_clock import EXPERIMENT_ORIGIN_ENV


def _preview_requests(settings, requests):
    for name, value, timestamp in requests:
        with patch("bin.live_settings.time.perf_counter_ns", return_value=timestamp):
            settings.request(name, value)


def _preview_burst(settings, count, ready, stop):
    for index in range(count):
        settings.request("fixation_radius", 0.2 + index / 100000)
    ready.set()
    stop.wait(timeout=10)


class LiveSettingsTests(unittest.TestCase):
    def make_settings(self, **kwargs):
        settings = reward_settings(0.25, reward_interval=2, fixation_radius=0.1, **kwargs)
        self.addCleanup(settings.close, Mock())
        return settings

    def make_bundle(self, settings, *, initial_ns=None, origin=2**60 + 333, **bundle_kwargs):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        with patch.dict(os.environ, {EXPERIMENT_ORIGIN_ENV: str(origin)}), patch(
            "bin.live_settings.time.perf_counter_ns", return_value=initial_ns or origin + 1,
        ):
            bundle = SessionLogBundle(
                output_root=directory.name, task_name="active_foraging", config_name="live",
                live_settings=settings, auto_flush=False, **bundle_kwargs,
            )
        self.addCleanup(bundle.close)
        return bundle, origin

    def messages(self, bundle):
        with bundle.message_logger.path.open(encoding="utf-8", newline="") as handle:
            return [
                (int(row["task_time_ns"]), json.loads(row["message"].removeprefix("live_setting ")))
                for row in csv.DictReader(handle, delimiter="\t")
                if row["message"].startswith("live_setting ")
            ]

    def test_busy_preview_lock_never_blocks_task_or_calls_task_callback(self):
        settings = self.make_settings()
        settings.request("reward_interval", 3)
        logger, callback = Mock(), Mock()
        result = []
        settings._lock.acquire()
        worker = threading.Thread(target=lambda: result.append(settings.apply_pending(logger, on_apply=callback)), daemon=True)
        try:
            worker.start()
            worker.join(timeout=1)
            returned_while_locked = not worker.is_alive()
        finally:
            settings._lock.release()
            worker.join(timeout=1)
        self.assertTrue(returned_while_locked, "Trial thread waited for preview's lock")
        self.assertEqual(result, [{"pump_pulse_time": 0.25, "reward_interval": 2, "fixation_radius": 0.1}])
        logger.log.assert_not_called()
        callback.assert_not_called()
        self.assertEqual(settings.apply_pending()["reward_interval"], 3)

    def test_boundary_updates_policies_as_one_group_and_only_buffers_logs(self):
        settings = self.make_settings()
        bundle, origin = self.make_bundle(settings)
        settings.request("reward_interval", 4)
        settings.request("pump_pulse_time", 3)
        applied_at = origin + 12_345_679
        callback_observations = []

        def apply_policy(values, timestamp):
            callback_observations.append((dict(values), timestamp, settings.values))

        with patch.object(bundle.message_logger, "_writer") as disk_writer, patch.object(
            bundle.message_logger, "_file",
        ) as disk_file, patch.object(settings, "log_requests") as drain, patch(
            "bin.live_settings.time.perf_counter_ns", return_value=applied_at,
        ):
            applied = settings.apply_pending(bundle.message_logger, on_apply=apply_policy)
            disk_writer.assert_not_called()
            self.assertEqual(disk_writer.mock_calls, [])
            self.assertEqual(disk_file.mock_calls, [])
            drain.assert_not_called()
        self.assertEqual(applied, {"pump_pulse_time": 3, "reward_interval": 4, "fixation_radius": 0.1})
        self.assertEqual(callback_observations, [(applied, applied_at, {
            "pump_pulse_time": 0.25, "reward_interval": 2, "fixation_radius": 0.1,
        })])
        self.assertEqual(settings.snapshot(), {name: (value, value) for name, value in applied.items()})
        # The behavior snapshot is owned by the task; editing a returned dict
        # cannot alter the applied policy or a later trial's recorded settings.
        applied["pump_pulse_time"] = 99
        self.assertEqual(settings.values["pump_pulse_time"], 3)
        bundle.close()
        application_rows = [(stamp, fields) for stamp, fields in self.messages(bundle) if fields["phase"] == "applied"]
        self.assertEqual(len(application_rows), 2)
        self.assertEqual([stamp for stamp, _ in application_rows], [12_345_679] * 2)

    def test_related_settings_are_validated_against_complete_pending_group(self):
        settings = self.make_settings()
        before = settings.snapshot()
        for name, value in (("pump_pulse_time", 2), ("reward_interval", 0.25),
                            ("reward_interval", 0), ("fixation_radius", 0),
                            ("fixation_radius", float("inf")), ("pump_pulse_time", float("nan"))):
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                settings.request(name, value)
            self.assertEqual(settings.snapshot(), before)
        settings.request("reward_interval", 4)
        settings.request("pump_pulse_time", 3)
        with self.assertRaises(ValueError):
            settings.request("reward_interval", 2)
        self.assertEqual(settings.values["pump_pulse_time"], 0.25)
        self.assertEqual(settings.apply_pending()["pump_pulse_time"], 3)
        self.assertEqual(settings.values["reward_interval"], 4)

    def test_every_confirmation_survives_overwrite_and_revert_with_exact_times(self):
        settings = self.make_settings()
        bundle, origin = self.make_bundle(settings)
        requests = (
            ("pump_pulse_time", 0.3, origin + 101),
            ("pump_pulse_time", 0.4, origin + 103),
            ("pump_pulse_time", 0.25, origin + 107),
            ("fixation_radius", 0.2, origin + 109),
            ("fixation_radius", 0.3, origin + 113),
        )
        _preview_requests(settings, requests)
        with patch("bin.live_settings.time.perf_counter_ns", return_value=origin + 127):
            settings.apply_pending(bundle.message_logger)
        bundle.close()
        rows = self.messages(bundle)
        initial = [(stamp, fields) for stamp, fields in rows if fields["phase"] == "initial"]
        self.assertEqual([stamp for stamp, _ in initial], [1, 1, 1])
        self.assertEqual({fields["name"]: fields["new_value"] for _, fields in initial}, {
            "pump_pulse_time": 0.25, "reward_interval": 2, "fixation_radius": 0.1,
        })
        confirmed = [(stamp, fields) for stamp, fields in rows if fields["phase"] == "confirmed"]
        self.assertEqual([(fields["name"], fields["new_value"], stamp) for stamp, fields in confirmed],
                         [(name, value, stamp - origin) for name, value, stamp in requests])
        self.assertEqual([fields["request_id"] for _, fields in confirmed], [1, 2, 3, 4, 5])
        self.assertEqual([fields["old_value"] for _, fields in confirmed], [0.25, 0.3, 0.4, 0.1, 0.2])
        applied = [(stamp, fields) for stamp, fields in rows if fields["phase"] == "applied"]
        self.assertEqual(applied, [(127, {
            "phase": "applied", "name": "fixation_radius", "old_value": 0.1,
            "new_value": 0.3, "units": "frac", "request_id": 5,
            "requested_perf_ns": origin + 113,
        })])

    def test_shutdown_drains_all_queued_edits_even_without_a_boundary_or_flush(self):
        settings = self.make_settings()
        bundle, origin = self.make_bundle(settings)
        requests = [("fixation_radius", (index + 1) / 1000, origin + 1001 + index) for index in range(80)]
        _preview_requests(settings, requests)
        bundle.close()
        bundle.close()
        rows = self.messages(bundle)
        confirmed = [(stamp, fields) for stamp, fields in rows if fields["phase"] == "confirmed"]
        self.assertEqual([(stamp, fields["new_value"]) for stamp, fields in confirmed],
                         [(timestamp - origin, value) for _, value, timestamp in requests])
        self.assertFalse(any(fields["phase"] == "applied" for _, fields in rows))
        self.assertEqual(settings.values["fixation_radius"], 0.1)

    def test_spawned_preview_requests_reach_task_without_scene_updates(self):
        settings = self.make_settings()
        bundle, origin = self.make_bundle(settings)
        requests = [("reward_interval", 5, origin + 211), ("pump_pulse_time", 1, origin + 223)]
        child = mp.get_context("spawn").Process(target=_preview_requests, args=(settings, requests))
        child.start()
        child.join(timeout=10)
        if child.is_alive():
            child.terminate()
            child.join(timeout=2)
            self.fail("Preview setting requests did not complete")
        self.assertEqual(child.exitcode, 0)
        self.assertEqual(settings.snapshot()["reward_interval"], (2, 5))
        with patch("bin.live_settings.time.perf_counter_ns", return_value=origin + 227):
            self.assertEqual(settings.apply_pending(bundle.message_logger)["pump_pulse_time"], 1)
        bundle.close()
        confirmed = [(stamp, fields) for stamp, fields in self.messages(bundle) if fields["phase"] == "confirmed"]
        self.assertEqual([(stamp, fields["name"], fields["new_value"]) for stamp, fields in confirmed],
                         [(timestamp - origin, name, value) for name, value, timestamp in requests])

    def test_regular_flush_does_not_wait_for_queue_feeder(self):
        settings = self.make_settings()
        bundle, _ = self.make_bundle(settings)
        settings.request("fixation_radius", 0.2)
        with patch.object(settings._confirmations, "get_nowait", side_effect=queue.Empty) as poll, patch.object(
            settings._confirmations, "get", side_effect=AssertionError("Normal flush must never wait"),
        ):
            bundle.flush()
        poll.assert_called_once_with()
        self.assertEqual(settings._logged_confirmations, 0)
        bundle.close()
        self.assertEqual(len([fields for _, fields in self.messages(bundle) if fields["phase"] == "confirmed"]), 1)

    def test_preview_shutdown_drains_large_producer_queue_without_termination(self):
        settings = self.make_settings()
        bundle, _ = self.make_bundle(settings)
        ctx = mp.get_context("spawn")
        ready, stop = ctx.Event(), ctx.Event()
        count = 3000  # Exceeds the OS pipe capacity even for these tiny records.
        child = ctx.Process(target=_preview_burst, args=(settings, count, ready, stop))
        child.start()

        def stop_child():
            stop.set()
            if child.is_alive():
                child.terminate()
            child.join(timeout=2)

        self.addCleanup(stop_child)
        self.assertTrue(ready.wait(timeout=10), "Preview burst did not finish")
        preview = object.__new__(ExperimenterPreview)
        preview._closed = False
        preview._live_settings = settings
        preview._process = child
        preview._stop_event = stop
        preview._sender_stop = threading.Event()
        preview._sender_thread = None
        preview._queue = Mock()
        preview._calibration_actions = None
        with patch.object(child, "terminate", wraps=child.terminate) as terminate:
            preview.close()
        terminate.assert_not_called()
        self.assertEqual(child.exitcode, 0)
        self.assertEqual(settings._logged_confirmations, 0, "Joining must not format or write logs")
        bundle.close()
        confirmations = [fields for _, fields in self.messages(bundle) if fields["phase"] == "confirmed"]
        self.assertEqual(len(confirmations), count)
        self.assertEqual([fields["request_id"] for fields in confirmations], list(range(1, count + 1)))
        self.assertEqual(confirmations[-1]["new_value"], 0.2 + (count - 1) / 100000)

    def test_busy_policy_defers_whole_group_without_application_log_then_retries(self):
        settings = self.make_settings()
        bundle, origin = self.make_bundle(settings)
        with patch("bin.live_settings.time.perf_counter_ns", return_value=origin + 11):
            settings.request("reward_interval", 4)
            settings.request("pump_pulse_time", 3)
        before = settings.values
        callback = Mock(return_value=False)
        with patch("bin.live_settings.time.perf_counter_ns", return_value=origin + 13):
            self.assertEqual(settings.apply_pending(bundle.message_logger, on_apply=callback), before)
        self.assertEqual(settings.snapshot()["pump_pulse_time"], (0.25, 3))
        self.assertEqual(settings.snapshot()["reward_interval"], (2, 4))
        callback.return_value = True
        with patch("bin.live_settings.time.perf_counter_ns", return_value=origin + 17):
            self.assertEqual(settings.apply_pending(bundle.message_logger, on_apply=callback)["pump_pulse_time"], 3)
        self.assertEqual(callback.call_count, 2)
        bundle.close()
        applications = [(stamp, fields) for stamp, fields in self.messages(bundle) if fields["phase"] == "applied"]
        self.assertEqual(len(applications), 2)
        self.assertEqual([stamp for stamp, _ in applications], [17, 17])

    def test_audit_failure_still_finalizes_remaining_session_logs(self):
        settings = self.make_settings()
        bundle, _ = self.make_bundle(
            settings, behavior_fieldnames=["trial_num", "pump_pulse_time"],
            additional_table_fieldnames={"extra": ["value"]},
        )
        bundle.behavior_logger.writerow({"trial_num": 1, "pump_pulse_time": 0.25})
        bundle.table_loggers["extra"].writerow({"value": "kept"})
        # Model a terminated preview whose final confirmation never arrived.
        settings._confirmation_count.value = 1
        with patch.object(settings._confirmations, "get", side_effect=queue.Empty), self.assertRaisesRegex(
            RuntimeError, "did not drain",
        ):
            bundle.close()
        for logger in (bundle.message_logger, bundle.event_logger, bundle.behavior_logger, bundle.table_loggers["extra"]):
            self.assertTrue(logger._file.closed)
        with bundle.message_logger.path.open() as handle:
            messages = list(csv.DictReader(handle, delimiter="\t"))
        self.assertTrue(any(row["level"] == "ERROR" and "live_setting_audit_incomplete" in row["message"] for row in messages))
        with bundle.event_logger.path.open() as handle:
            self.assertEqual(list(csv.DictReader(handle, delimiter="\t"))[-1]["event"], "block_end")
        with bundle.behavior_logger.path.open() as handle:
            self.assertEqual(list(csv.DictReader(handle, delimiter="\t")), [{"trial_num": "1", "pump_pulse_time": "0.25"}])
        with bundle.table_loggers["extra"].path.open() as handle:
            self.assertEqual(list(csv.DictReader(handle, delimiter="\t")), [{"value": "kept"}])

    def test_unchanged_confirmations_and_empty_boundaries_do_not_log_false_changes(self):
        settings = self.make_settings()
        bundle, _ = self.make_bundle(settings)
        callback = Mock()
        for name, value in settings.values.items():
            settings.request(name, value)
        settings.apply_pending(bundle.message_logger, on_apply=callback)
        callback.assert_not_called()
        bundle.close()
        self.assertEqual([fields["phase"] for _, fields in self.messages(bundle)], ["initial"] * 3)

    def test_policy_error_leaves_settings_pending_and_releases_shared_lock(self):
        settings = self.make_settings()
        settings.request("fixation_radius", 0.2)
        with self.assertRaisesRegex(RuntimeError, "policy"):
            settings.apply_pending(on_apply=Mock(side_effect=RuntimeError("policy")))
        self.assertEqual(settings.snapshot()["fixation_radius"], (0.1, 0.2))
        self.assertEqual(settings.apply_pending()["fixation_radius"], 0.2)

    def test_setting_schema_is_reusable_and_rejects_invalid_initial_values(self):
        spec = SettingSpec("custom", "Custom", "Custom distance", "px", minimum=1, maximum=10)
        settings = LiveSettings([spec], {"custom": 2}, boundary="stimulus")
        self.addCleanup(settings.close, Mock())
        settings.request("custom", "3.125")
        self.assertEqual(settings.snapshot(), {"custom": (2, 3.125)})
        self.assertEqual(settings.apply_pending(), {"custom": 3.125})
        for value in ("not a number", 0, 10, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                LiveSettings([spec], {"custom": value})
        with self.assertRaisesRegex(ValueError, "unique"):
            LiveSettings([spec, spec], {"custom": 2})
