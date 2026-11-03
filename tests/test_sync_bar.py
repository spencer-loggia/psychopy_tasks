import csv
import math
import os
import random
import tempfile
import unittest
from unittest.mock import Mock, patch

from bin.logger import SessionLogBundle
from bin.screen import ExperimenterPreview, oriented_size, rotate_centered_point
from bin.sync_bar import SyncBar, SyncBarConfig, sync_bar_rectangles
from bin.task_clock import EXPERIMENT_ORIGIN_ENV


class FakeWindow:
    size = (1003, 602)

    def __init__(self):
        self.callbacks = []

    def callOnFlip(self, callback, *args):
        self.callbacks.append((callback, args))

    def flip(self):
        callbacks, self.callbacks = self.callbacks, []
        for callback, args in callbacks:
            callback(*args)


class SyncConfigTests(unittest.TestCase):
    def test_defaults_are_seconds_and_disabled(self):
        self.assertEqual(SyncBarConfig.from_mapping(), SyncBarConfig(False, 20, 1.0, 0.2))

    def test_invalid_and_obsolete_fields_fail_before_display_startup(self):
        invalid = [
            {"enabled": "false"}, {"enabled": 1}, {"height": 0},
            {"height": True}, {"height": 1.5}, {"interval_mean_seconds": 0},
            {"interval_mean_seconds": True}, {"interval_mean_seconds": math.inf},
            {"interval_variance_seconds": -0.1}, {"interval_variance_seconds": math.nan},
            {"interval_variance_seconds": 1}, {"minimum_duration_seconds": 0.06},
        ]
        for config in invalid:
            with self.subTest(config=config), self.assertRaises(ValueError):
                SyncBarConfig.from_mapping(config)
        for config in (True, [], "enabled"):
            with self.subTest(config=config), self.assertRaises(ValueError):
                SyncBarConfig.from_mapping(config)


class SyncGeometryTests(unittest.TestCase):
    def test_top_edge_and_integer_width_survive_every_rotation(self):
        native_size = (1003, 602)
        for rotation in (0, 90, 180, 270):
            with self.subTest(rotation=rotation):
                width, height = oriented_size(native_size, rotation)
                rectangles = sync_bar_rectangles(native_size, 20, rotation)
                self.assertEqual(len(rectangles), 5)
                for index, rectangle in enumerate(rectangles):
                    subject_pos = rotate_centered_point(rectangle["pos"], -rotation)
                    self.assertEqual(rectangle["width"], int(width) // 5)
                    self.assertEqual(subject_pos[1] + rectangle["height"] / 2, height / 2)
                    self.assertEqual(subject_pos[0] - rectangle["width"] / 2,
                                     -width / 2 + index * (int(width) // 5))

    def test_bar_must_fit_oriented_screen(self):
        with self.assertRaises(ValueError):
            sync_bar_rectangles((100, 50), 51)

    def test_preview_publishes_oriented_geometry_and_each_phase_without_queueing(self):
        preview = ExperimenterPreview.__new__(ExperimenterPreview)
        preview._sync_bar_shared = [0.0] * 4
        preview.configure_sync_bar(20, (1003, 602), 90)
        self.assertEqual(preview._sync_bar_shared, [20, 602, 1003, 0])
        preview.set_sync_bar_phase(1)
        self.assertEqual(preview._sync_bar_shared, [20, 602, 1003, 1])
        preview.set_sync_bar_phase(0)
        self.assertEqual(preview._sync_bar_shared, [20, 602, 1003, 0])


class SyncTimingTests(unittest.TestCase):
    def test_transition_numbers_and_phases_survive_flushes_and_restart_per_block(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for _ in range(2):
                bundle = SessionLogBundle(output_root=tmpdir, task_name="active_foraging",
                                          config_name="sync", auto_flush=False)
                bar = SyncBar(
                    self.window, SyncBarConfig(True, 20, 1, 0),
                    event_logger=bundle.event_logger, gpio_module=self.gpio, gpio_chip=0,
                    sync_pin=18, clock=lambda: self.now_ns,
                    rect_factory=lambda *a, **k: Mock(),
                )
                bar.start()
                for index in range(1, 4):
                    target = bar.next_flip_perf_s
                    bar.prepare_flip(target)
                    self.assertEqual(bar.transition_num, index - 1)
                    self.now_ns = round(target * 1e9)
                    self.window.flip()
                    bar.flush_events()
                    bundle.flush()
                bar.close()
                bundle.close()
                with bundle.event_logger.path.open() as handle:
                    rows = list(csv.DictReader(handle, delimiter="\t"))
                sync_rows = [row for row in rows if row["event"].startswith("sync_")]
                self.assertEqual([row["sync_transition_num"] for row in sync_rows],
                                 ["1", "1", "2", "2", "3", "3"])
                self.assertEqual([row["sync_phase"] for row in sync_rows],
                                 ["1", "1", "0", "0", "1", "1"])
                self.assertTrue(all(row["sync_transition_num"] == row["sync_phase"] == ""
                                    for row in rows if not row["event"].startswith("sync_")))

    def setUp(self):
        self.now_ns = 10_000_000_000
        self.window = FakeWindow()
        self.logger = Mock()
        self.gpio = Mock()
        self.gpio.gpio_write.return_value = 0
        self.preview = Mock()
        self.bar = SyncBar(
            self.window, SyncBarConfig(True, 20, 1.0, 0.0),
            event_logger=self.logger, gpio_module=self.gpio, gpio_chip=0,
            sync_pin=18, clock=lambda: self.now_ns,
            preview_callback=self.preview, rect_factory=lambda *args, **kwargs: Mock(),
        )

    def test_edge_occurs_on_actual_swap_and_logs_keep_requested_deadline(self):
        self.bar.start()
        self.assertFalse(self.bar.prepare_flip(10.5))
        self.assertEqual([r.fillColor[0] for r in self.bar.rectangles], [0, 255, 0, 255, 0])
        self.assertTrue(self.bar.prepare_flip(11.0))
        self.assertEqual([r.fillColor[0] for r in self.bar.rectangles], [255, 0, 255, 0, 255])
        self.gpio.gpio_write.assert_not_called()
        self.logger.log_event.assert_not_called()
        self.now_ns = 11_002_345_678
        self.window.flip()
        self.gpio.gpio_write.assert_called_once_with(0, 18, 1)
        self.logger.log_event.assert_not_called()
        records = self.bar.flush_events()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].requested_perf_ns, 11_000_000_000)
        self.assertEqual(records[0].actual_perf_ns, self.now_ns)
        self.assertEqual(self.bar.next_flip_perf_s, 12.002345678)
        flip_event, gpio_event = [call.kwargs for call in self.logger.log_event.call_args_list]
        self.assertEqual(flip_event["event"], "sync_bar_flip")
        self.assertEqual(flip_event["requested_timestamp_perf_ns"], 11_000_000_000)
        self.assertEqual(flip_event["timestamp_perf_ns"], self.now_ns)
        self.assertEqual(gpio_event["event"], "sync_signal_on")
        self.assertEqual(gpio_event["requested_timestamp_perf_ns"], self.now_ns)
        self.preview.assert_called_once_with(1)

    def test_one_edge_per_refresh_and_no_overdue_catchup_burst(self):
        self.bar.start()
        self.assertTrue(self.bar.prepare_flip(20))
        self.assertTrue(self.bar.prepare_flip(20))
        self.assertEqual(len(self.window.callbacks), 1)
        self.now_ns = 20_000_000_000
        self.window.flip()
        self.assertFalse(self.bar.prepare_flip(20.1))
        self.assertTrue(self.bar.prepare_flip(21))
        self.now_ns = 21_000_000_000
        self.window.flip()
        self.assertEqual([call.args[2] for call in self.gpio.gpio_write.call_args_list], [1, 0])
        self.assertEqual(self.bar.next_flip_perf_s, 22)

    def test_uniform_schedule_is_bounded_and_does_not_touch_task_rng(self):
        global_state = random.getstate()
        bar = SyncBar(self.window, SyncBarConfig(True, 20, 1, 0.2),
                      rect_factory=lambda *args, **kwargs: Mock(), rng=random.Random(17), clock=lambda: self.now_ns)
        bar.start()
        intervals = [bar.next_flip_perf_s - 10]
        for _ in range(100):
            self.now_ns = round(bar.next_flip_perf_s * 1_000_000_000)
            bar.prepare_flip(self.now_ns / 1_000_000_000)
            self.window.flip()
            intervals.append(bar.next_flip_perf_s - self.now_ns / 1_000_000_000)
        self.assertTrue(all(0.8 <= interval <= 1.2 for interval in intervals))
        self.assertGreater(len(set(intervals)), 90)
        self.assertEqual(random.getstate(), global_state)

    def test_disabled_needs_neither_psychopy_nor_clock_callbacks(self):
        factory, clock = Mock(), Mock()
        bar = SyncBar(object(), rect_factory=factory, clock=clock)
        self.assertFalse(bar.prepare_flip())
        self.assertEqual(bar.next_flip_perf_s, math.inf)
        self.assertEqual(bar.flush_events(), [])
        factory.assert_not_called()
        clock.assert_not_called()

    def test_gpio_failure_preserves_visual_event_and_cleanup_clears_pin(self):
        self.bar.start()
        self.bar.prepare_flip(11)
        self.gpio.gpio_write.return_value = -1
        self.window.flip()
        self.gpio.gpio_write.return_value = 0
        with self.assertRaisesRegex(RuntimeError, "Synchronization GPIO failed"):
            self.bar.close()
        self.assertEqual(self.logger.log_event.call_count, 1)
        self.assertEqual(self.logger.log_event.call_args.kwargs["event"], "sync_bar_flip")
        self.gpio.gpio_write.assert_called_with(0, 18, 0)

    def test_close_prevents_queued_callback_from_reasserting_gpio(self):
        self.bar.start()
        self.bar.prepare_flip(11)
        self.bar.close()
        self.window.flip()
        self.gpio.gpio_write.assert_called_once_with(0, 18, 0)

    def test_real_event_log_keeps_exact_requested_and_display_nanoseconds(self):
        origin = 2**55 + 123
        self.now_ns = origin
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(
            os.environ, {EXPERIMENT_ORIGIN_ENV: str(origin)}
        ):
            bundle = SessionLogBundle(output_root=tmpdir, task_name="active_foraging",
                                      config_name="sync", auto_flush=False)
            self.bar.event_logger = bundle.event_logger
            def gpio_write(*args):
                self.now_ns += 500
                return 0

            self.gpio.gpio_write.side_effect = gpio_write
            self.bar.start()
            self.bar.prepare_flip((origin + 1_100_000_000) / 1_000_000_000)
            self.now_ns = origin + 1_000_000_123
            self.window.flip()
            self.bar.close()
            bundle.close()
            with bundle.event_logger.path.open() as handle:
                rows = [row for row in csv.DictReader(handle, delimiter="\t")
                        if row["event"].startswith("sync_")]
        self.assertEqual([row["event"] for row in rows], ["sync_bar_flip", "sync_signal_on"])
        self.assertEqual([row["event_code"] for row in rows], ["120", "304"])
        self.assertEqual([row["event_type"] for row in rows], ["frame_flip", "signal"])
        self.assertEqual(rows[0]["requested_task_time_ns"], "1000000000")
        self.assertEqual(rows[0]["task_time_ns"], "1000000123")
        self.assertEqual(rows[1]["requested_task_time_ns"], "1000000123")
        self.assertEqual(rows[1]["task_time_ns"], "1000000623")


if __name__ == "__main__":
    unittest.main()
