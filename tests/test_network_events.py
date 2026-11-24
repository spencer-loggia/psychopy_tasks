import csv
import datetime as dt
from decimal import Decimal
import importlib.util
import json
import os
import socket
import tempfile
import threading
import types
import unittest
from unittest.mock import patch

from bin.block_sync import BLOCK_NUMBER_ENV, EXPERIMENT_NUMBER_ENV
from bin.logger import SessionLogBundle
from bin.network_events import NetworkEventsConfig, load_network_event_settings
from bin.task_clock import EXPERIMENT_ORIGIN_ENV


class FakeZMQ(types.ModuleType):
    PUB, LINGER, SNDHWM, DONTWAIT = 1, 2, 3, 4

    class Again(Exception):
        pass

    def __init__(self, *, full=False, gate=None, fail=False):
        super().__init__("zmq")
        self.full, self.gate, self.fail = full, gate, fail
        self.messages = []
        self.options = {}
        self.thread_ids = set()
        self.attempted = threading.Event()
        self.closed = threading.Event()

    def _worker_call(self):
        self.thread_ids.add(threading.get_ident())

    def Context(self):
        self._worker_call()
        if self.fail:
            raise RuntimeError("test connection failure")
        return self

    def socket(self, kind):
        self._worker_call()
        assert kind == self.PUB
        return self

    def setsockopt(self, key, value):
        self._worker_call()
        self.options[key] = value

    def bind(self, endpoint):
        self._worker_call()
        self.endpoint = endpoint

    def send_json(self, message, *, flags, **kwargs):
        self._worker_call()
        assert flags == self.DONTWAIT
        self.attempted.set()
        if self.gate is not None:
            self.gate.wait(timeout=3)
        if self.full:
            raise self.Again()
        self.messages.append(json.loads(json.dumps(message, **kwargs)))

    def close(self, *, linger):
        self._worker_call()
        assert linger == 0

    def term(self):
        self._worker_call()
        self.closed.set()


def read_tsv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


class NetworkEventsTests(unittest.TestCase):
    def test_opt_in_validation_and_source_filename(self):
        self.assertIsNone(load_network_event_settings({}))
        self.assertIsNone(load_network_event_settings({"send_network_events": False}))
        self.assertIsNone(load_network_event_settings({"send_network_events": {"enabled": False}}))
        with self.assertRaisesRegex(ValueError, "boolean"):
            load_network_event_settings({"send_network_events": "false"})
        for options in ({"port": 420420}, {"port": True}, {"port": 0},
                        {"bind_address": ""}, {"queue_size": 0},
                        {"enabled": "true"}, {"queue_size": True}, {"send_high_water_mark": -1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                load_network_event_settings({"send_network_events": {"enabled": True, **options}})
        settings = load_network_event_settings({
            "subject": "Yuri", "experiment_number": 7,
            "source_config_file": "/configs/original.json",
            "send_network_events": {"enabled": True, "port": 5556, "queue_size": 32},
        }, config_path="/logs/block/config.json")
        self.assertEqual(settings.config_file_name, "original.json")
        self.assertEqual(settings.subject, "Yuri")
        self.assertEqual(settings.experiment_number, 7)
        self.assertEqual(settings.endpoint, "tcp://*:5556")
        self.assertEqual(settings.queue_size, 32)
        self.assertEqual(load_network_event_settings(
            {"send_network_events": True}, config_path="/configs/standalone.json",
        ).config_file_name, "standalone.json")
        self.assertIsNone(load_network_event_settings({
            "send_network_events": True, "source_config_file": None,
        }, config_path="/logs/block/config.json").config_file_name)

    def test_disabled_sessions_do_not_create_a_sender_or_import_zmq(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True), \
                patch.dict("sys.modules", {"zmq": None}), patch("bin.logger.NetworkEventSender") as sender:
            logs = SessionLogBundle(output_root=directory, task_name="active_foraging", config_name="test")
            logs.close()
            sender.assert_not_called()
            self.assertEqual(len(read_tsv(logs.event_logger.path)), 2)

    def test_every_row_matches_tsv_with_metadata_first_and_original_timestamps(self):
        for auto_flush in (False, True):
            fake = FakeZMQ()
            origin = 2**60
            when = dt.datetime(2026, 11, 23, 10, 11, 12, 345678)
            with self.subTest(auto_flush=auto_flush), tempfile.TemporaryDirectory() as directory, \
                    patch.dict("sys.modules", {"zmq": fake}), patch.dict(os.environ, {
                        EXPERIMENT_ORIGIN_ENV: str(origin), EXPERIMENT_NUMBER_ENV: "17",
                        BLOCK_NUMBER_ENV: "3",
                    }, clear=True):
                logs = SessionLogBundle(
                    output_root=directory, task_name="active_foraging", config_name="block config",
                    when=when, auto_flush=auto_flush,
                    behavior_fieldnames=["trial_num", "value", "blank", "missing", "text"],
                    additional_table_fieldnames={"other": ["value"]},
                    network_events_config=NetworkEventsConfig(
                        subject="Yuri", config_file_name="original.json", experiment_number=99,
                    ),
                )
                self.addCleanup(logs.close)
                logs.event_logger.log_frame_flip(
                    trial_num=1, event="options_on", timestamp_perf_ns=origin + 12345678901234567,
                    requested_timestamp_perf_ns=origin + 12345678901234566, requested_duration=0.125,
                )
                row = {"trial_num": 1, "value": Decimal("1.250"), "blank": None,
                       "text": "tab\tand\nnewline é", "ignored": "extra"}
                logs.behavior_logger.writerow(row)
                row["text"] = "mutated"
                logs.message_logger.log("INFO", "local only")
                logs.table_loggers["other"].writerow({"value": "local only"})
                logs.flush()
                logs.flush()
                logs.close()
                logs.close()
                self.assertTrue(fake.closed.is_set())
                self.assertEqual([m["sequence"] for m in fake.messages], list(range(5)))
                self.assertEqual([m["type"] for m in fake.messages], [
                    "block_start", "event_log", "event_log", "behavior_log", "event_log",
                ])
                metadata = fake.messages[0]["metadata"]
                self.assertEqual(metadata["date"], "2026-11-23")
                self.assertEqual(metadata["time"], when.astimezone().timetz().isoformat(timespec="microseconds"))
                self.assertEqual(metadata["experiment_number"], 17)
                self.assertEqual(metadata["block_number"], 3)
                self.assertEqual(metadata["subject"], "Yuri")
                self.assertEqual(metadata["block_config_name"], "block config")
                self.assertEqual(metadata["config_file_name"], "original.json")
                for message in fake.messages:
                    self.assertEqual(message["block_id"], logs.identity["block_id"])
                    self.assertEqual(message["version"], 1)
                self.assertEqual([m["row"] for m in fake.messages if m["type"] == "event_log"],
                                 read_tsv(logs.event_logger.path))
                self.assertEqual([m["row"] for m in fake.messages if m["type"] == "behavior_log"],
                                 read_tsv(logs.behavior_logger.path))
                self.assertEqual(fake.messages[2]["row"]["task_time_ns"], "12345678901234567")
                self.assertEqual(fake.options[fake.LINGER], 0)
                self.assertEqual(fake.options[fake.SNDHWM], 8192)
                self.assertEqual(len(fake.thread_ids), 1)
                self.assertNotIn(threading.get_ident(), fake.thread_ids)

    def test_full_transport_drops_without_retrying_or_losing_local_rows(self):
        fake = FakeZMQ(full=True)
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True), \
                patch.dict("sys.modules", {"zmq": fake}):
            logs = SessionLogBundle(
                output_root=directory, task_name="active_foraging", config_name="test",
                auto_flush=False, behavior_fieldnames=["trial_num"],
                network_events_config=NetworkEventsConfig(),
            )
            for trial in range(20):
                logs.event_logger.log_signal(trial_num=trial, event="pump_on")
                logs.behavior_logger.writerow({"trial_num": trial})
            logs.close()
            self.assertEqual(len(read_tsv(logs.event_logger.path)), 22)
            self.assertEqual(len(read_tsv(logs.behavior_logger.path)), 20)
            self.assertEqual(logs.network_events.send_dropped, 43)
            self.assertEqual(logs.network_events.sent, 0)
            self.assertIsNone(logs.network_events.error)

    def test_stalled_worker_does_not_block_logging_or_shutdown(self):
        gate = threading.Event()
        fake = FakeZMQ(gate=gate)
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True), \
                patch.dict("sys.modules", {"zmq": fake}):
            logs = SessionLogBundle(
                output_root=directory, task_name="active_foraging", config_name="test",
                auto_flush=False, network_events_config=NetworkEventsConfig(queue_size=2),
            )
            try:
                self.assertTrue(fake.attempted.wait(timeout=1))
                for trial in range(100):
                    logs.event_logger.log_signal(trial_num=trial, event="pump_on")
                self.assertGreater(logs.network_events.queue_dropped, 0)
                self.assertFalse(gate.is_set())
                closer = threading.Thread(target=logs.close)
                closer.start()
                closer.join(timeout=1)
                self.assertFalse(closer.is_alive(), "Shutdown waited for the network worker")
                self.assertEqual(len(read_tsv(logs.event_logger.path)), 102)
            finally:
                gate.set()
                logs.network_events._thread.join(timeout=1)
                logs.close()

    def test_missing_dependency_and_worker_errors_leave_local_logging_usable(self):
        for module in (None, FakeZMQ(fail=True)):
            with self.subTest(module=module), tempfile.TemporaryDirectory() as directory, \
                    patch.dict(os.environ, {}, clear=True), patch.dict("sys.modules", {"zmq": module}):
                logs = SessionLogBundle(
                    output_root=directory, task_name="active_foraging", config_name="test",
                    network_events_config=NetworkEventsConfig(),
                )
                logs.network_events._thread.join(timeout=1)
                logs.event_logger.log_signal(trial_num=1, event="pump_on")
                logs.close()
                self.assertEqual(len(read_tsv(logs.event_logger.path)), 3)
                warnings = [r for r in read_tsv(logs.message_logger.path) if r["level"] == "WARN"]
                self.assertEqual(len(warnings), 1)
                self.assertIn("network_events_failed", warnings[0]["message"])

    @unittest.skipUnless(importlib.util.find_spec("zmq"), "PyZMQ is not installed")
    def test_one_collector_combines_two_publisher_ports_and_keeps_block_identities(self):
        import zmq

        with socket.socket() as first, socket.socket() as second:
            first.bind(("127.0.0.1", 0))
            second.bind(("127.0.0.1", 0))
            ports = [first.getsockname()[1], second.getsockname()[1]]
        with zmq.Context() as context, context.socket(zmq.SUB) as receiver:
            receiver.setsockopt(zmq.LINGER, 0)
            receiver.setsockopt(zmq.SUBSCRIBE, b"")
            receiver.setsockopt(zmq.RECONNECT_IVL, 10)
            for port in ports:
                receiver.connect(f"tcp://127.0.0.1:{port}")
            with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
                bundles = []
                for index, port in enumerate(ports):
                    logs = SessionLogBundle(
                        output_root=directory, task_name="active_foraging", config_name=f"source{index}",
                        network_events_config=NetworkEventsConfig(endpoint=f"tcp://127.0.0.1:{port}"),
                    )
                    self.addCleanup(logs.close)
                    bundles.append(logs)
                messages = []
                for _ in range(4):
                    self.assertTrue(receiver.poll(2000))
                    messages.append(receiver.recv_json())
                for logs in bundles:
                    logs.event_logger.log_signal(trial_num=1, event="pump_on")
                    logs.close()
                for _ in range(4):
                    self.assertTrue(receiver.poll(2000))
                    messages.append(receiver.recv_json())
                for index, logs in enumerate(bundles):
                    stream = [m for m in messages if m["block_id"] == logs.identity["block_id"]]
                    self.assertEqual([m["sequence"] for m in stream], list(range(4)))
                    self.assertEqual(stream[0]["metadata"]["block_config_name"], f"source{index}")
                    self.assertEqual([m["row"] for m in stream[1:]], read_tsv(logs.event_logger.path))

    @unittest.skipUnless(importlib.util.find_spec("zmq"), "PyZMQ is not installed")
    def test_real_zmq_broadcasts_metadata_and_rows_to_multiple_subscribers(self):
        import zmq

        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        with zmq.Context() as context, context.socket(zmq.SUB) as receiver, context.socket(zmq.SUB) as second:
            for subscriber in (receiver, second):
                subscriber.setsockopt(zmq.LINGER, 0)
                subscriber.setsockopt(zmq.SUBSCRIBE, b"")
                subscriber.setsockopt(zmq.RECONNECT_IVL, 10)
                subscriber.connect(f"tcp://127.0.0.1:{port}")
            with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
                logs = SessionLogBundle(
                    output_root=directory, task_name="active_foraging", config_name="loopback",
                    behavior_fieldnames=["trial_num"], auto_flush=False,
                    network_events_config=NetworkEventsConfig(endpoint=f"tcp://127.0.0.1:{port}"),
                )
                self.addCleanup(logs.close)
                messages = []
                for _ in range(2):
                    self.assertTrue(receiver.poll(2000))
                    messages.append(receiver.recv_json())
                logs.behavior_logger.writerow({"trial_num": 1})
                self.assertTrue(receiver.poll(2000))
                messages.append(receiver.recv_json())
                logs.close()
                self.assertTrue(receiver.poll(2000))
                messages.append(receiver.recv_json())
                self.assertEqual([m["type"] for m in messages], [
                    "block_start", "event_log", "behavior_log", "event_log",
                ])
                self.assertEqual(messages[-1]["row"]["event"], "block_end")
                self.assertEqual([m["sequence"] for m in messages], list(range(4)))
                self.assertFalse(receiver.poll(20))
                for message in messages:
                    self.assertTrue(second.poll(2000))
                    self.assertEqual(second.recv_json(), message)


if __name__ == "__main__":
    unittest.main()
