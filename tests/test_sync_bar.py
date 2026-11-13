import csv
import datetime as dt
from itertools import groupby
import math
import os
import random
import tempfile
import unittest
from unittest.mock import Mock, patch

from bin.logger import SessionLogBundle
from bin.screen import ExperimenterPreview, oriented_size, rotate_centered_point
from bin.sync_bar import (BIT_PERIOD_NS, START_CODE, END_CODE, SyncBar, SyncBarConfig,
                          start_codeword, end_codeword, sync_bar_rectangles)
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
        self.assertEqual(SyncBarConfig.from_mapping().sync_pin, 18)

    def test_nested_pin_overrides_legacy_pin_including_visual_only(self):
        for pin in (0, 23, None):
            with self.subTest(pin=pin):
                self.assertEqual(SyncBarConfig.from_mapping({"sync_pin": pin}, sync_pin=24).sync_pin, pin)
        self.assertEqual(SyncBarConfig.from_mapping({"enabled": True}, sync_pin=24).sync_pin, 24)
        self.assertIsNone(SyncBarConfig.from_mapping({"enabled": True}, sync_pin=None).sync_pin)

    def test_invalid_and_obsolete_fields_fail_before_display_startup(self):
        invalid = [
            {"enabled": "false"}, {"enabled": 1}, {"height": 0},
            {"height": True}, {"height": 1.5}, {"interval_mean_seconds": 0},
            {"interval_mean_seconds": True}, {"interval_mean_seconds": math.inf},
            {"interval_variance_seconds": -0.1}, {"interval_variance_seconds": math.nan},
            {"interval_variance_seconds": 1}, {"minimum_duration_seconds": 0.06},
            {"sync_pin": -1}, {"sync_pin": True}, {"sync_pin": "18"}, {"sync_pin": 18.5},
        ]
        for config in invalid:
            with self.subTest(config=config), self.assertRaises(ValueError):
                SyncBarConfig.from_mapping(config)
        for config in (True, [], "enabled"):
            with self.subTest(config=config), self.assertRaises(ValueError):
                SyncBarConfig.from_mapping(config)


class SyncCodewordTests(unittest.TestCase):
    def test_payload_order_width_and_ranges(self):
        self.assertEqual(''.join(map(str, start_codeword(128, 1))), START_CODE + '10000000' + '00000001')
        self.assertEqual(''.join(map(str, end_codeword(dt.date(2026, 11, 13)))),
                         END_CODE + '0011010' + '1011' + '01101')
        for value in (-1, 256, True, 1.5):
            with self.assertRaises(ValueError):
                start_codeword(value, 1)
            with self.assertRaises(ValueError):
                start_codeword(1, value)
        for date in (dt.date(1999, 12, 31), dt.date(2128, 1, 1)):
            with self.assertRaises(ValueError):
                end_codeword(date)
        with self.assertRaises(ValueError):
            SyncBarConfig(enabled=True, interval_mean_seconds=0.3, interval_variance_seconds=0.2)

    def test_markers_cannot_occur_at_wrong_offset_for_any_payload(self):
        # '?' ranges independently over both levels, covering all 65536 payloads.
        words = (START_CODE, END_CODE)
        for target in words:
            for left in words:
                for right in words:
                    stream = left + '?' * 16 + right
                    for offset in range(1, 34):
                        candidate = stream[offset:offset + 18]
                        self.assertFalse(all(b == '?' or a == b for a, b in zip(target, candidate)),
                                         (target, left, right, offset))
                # Footer resets to idle low immediately after its final bit.
                stream = left + '?' * 16 + '0' * 18
                for offset in range(1, 34):
                    candidate = stream[offset:offset + 18]
                    self.assertFalse(all(b == '?' or a == b for a, b in zip(target, candidate)),
                                     (target, left, "idle", offset))
                # Slow train / marker / arbitrary payload / slow train.
                # Rounding a minimum three-cell random interval to a refresh
                # can leave only TWO whole sampled cells. After the payload,
                # its final full cell plus that interval guarantees THREE.
                # Use those weaker bounds so refresh phase cannot fake a marker.
                for offset in range(-17, 34):
                    if offset == 0:
                        continue
                    before, after, conflict = [], [], False
                    for index, bit in enumerate(target):
                        position = offset + index
                        if 0 <= position < 18 and bit != left[position]:
                            conflict = True
                        if position < 0:
                            before.append(bit)
                        if position >= 33:
                            after.append(bit)
                    runs_before = [len(list(g)) for _, g in groupby(before)]
                    runs_after = [len(list(g)) for _, g in groupby(after)]
                    conflict |= any(n < 2 for n in runs_before[1:-1])
                    conflict |= len(runs_after) > 1 and runs_after[0] < 3
                    conflict |= any(n < 2 for n in runs_after[1:-1])
                    self.assertTrue(conflict, (target, left, offset))


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

    def test_codewords_hold_repeated_bits_and_resume_only_after_final_hold(self):
        bits = start_codeword(255, 1)
        self.assertTrue(self.bar.start(block_start=True, experiment_number=255, block_number=1))
        self.assertFalse(self.bar.start(block_start=True))
        changes = []
        previous = 0
        for index, bit in enumerate(bits):
            target_ns = 10_000_000_000 + index * BIT_PERIOD_NS
            self.assertAlmostEqual(self.bar.next_flip_perf_s, target_ns / 1e9)
            self.assertIsNone(self.bar.word_end_perf_s)
            self.assertTrue(self.bar.prepare_flip(target_ns / 1e9))
            self.now_ns = target_ns
            self.window.flip()
            records = self.bar.flush_events()
            self.assertEqual(self.bar.phase, bit)
            self.assertEqual(len(records), int(bit != previous))
            if bit != previous:
                changes.append(bit)
            previous = bit
        complete = (10_000_000_000 + len(bits) * BIT_PERIOD_NS) / 1e9
        self.assertEqual(self.bar.word_end_perf_s, complete)
        self.assertAlmostEqual(self.bar.next_flip_perf_s, complete + 1)
        self.assertEqual([call.args[2] for call in self.gpio.gpio_write.call_args_list], changes)
        self.assertFalse(self.bar.prepare_flip(complete))
        self.now_ns += BIT_PERIOD_NS
        footer = end_codeword(dt.date(2026, 11, 13))
        self.bar.finish(dt.date(2026, 11, 13))
        for bit in footer:
            self.now_ns = round(self.bar.next_flip_perf_s * 1e9)
            self.bar.prepare_flip(self.now_ns / 1e9)
            self.window.flip()
            self.bar.flush_events()
            self.assertEqual(self.bar.phase, bit)
        self.assertEqual(self.bar.next_flip_perf_s, math.inf)
        self.assertEqual(self.bar.word_end_perf_s, (self.now_ns + BIT_PERIOD_NS) / 1e9)

    def test_delayed_codeword_bit_preserves_full_next_bit_without_catchup(self):
        self.bar.start(block_start=True)
        for actual_ns in (10_020_000_000, 11_200_000_000):
            self.assertTrue(self.bar.prepare_flip(actual_ns / 1e9))
            self.assertTrue(self.bar.prepare_flip(actual_ns / 1e9))
            self.assertEqual(len(self.window.callbacks), 1)
            self.now_ns = actual_ns
            self.window.flip()
            self.assertEqual(self.bar.next_flip_perf_s, (actual_ns + BIT_PERIOD_NS) / 1e9)
            self.assertFalse(self.bar.prepare_flip((actual_ns + 10_000_000) / 1e9))
        # The first two bits are both high: no invented second electrical edge.
        self.assertEqual([call.args[2] for call in self.gpio.gpio_write.call_args_list], [1])

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
        self.assertFalse(bar.start(block_start=True))
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
