"""Training timing, real event logs, and display/hardware wiring without a rig."""
import csv
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import bin
from PIL import Image
from bin.eye_tracking_config import EyeTrackingSettings
from bin.screen import ScreenGeometry, rotate_centered_point, software_stimulus_rotation
from task.trainings import touch_and_hold as task
from task.trainings.touch_and_hold_state import TouchAndHoldState
from tests.test_active_foraging_timing import FakeClock


ROOT = Path(__file__).resolve().parents[1]


class TouchAndHoldStateTests(unittest.TestCase):
    def state(self, **kwargs):
        return TouchAndHoldState(**dict(
            dict(duration=5, iti=0.2, reward_interval=1, end_on_release=True), **kwargs,
        ))

    def test_rewards_require_a_full_uninterrupted_interval(self):
        state = self.state(end_on_release=False)
        self.assertFalse(state.update(0, touching=True, pressed=True))
        self.assertFalse(state.update(0.99, touching=True))
        self.assertTrue(state.update(1, touching=True))
        state.reward_delivered(1)
        self.assertFalse(state.update(1.9, touching=False))
        self.assertFalse(state.update(2, touching=True))
        self.assertFalse(state.update(2.99, touching=True))
        self.assertTrue(state.update(3, touching=True))
        state.reward_delivered(3)
        self.assertFalse(state.update(5, touching=True))
        self.assertEqual(state.phase, "iti")
        self.assertEqual(state.end_reason, "duration")

    def test_null_duration_release_is_the_only_trial_boundary(self):
        state = self.state(duration=None)
        state.update(0, touching=True, pressed=True)
        self.assertTrue(state.update(1000, touching=True))
        self.assertEqual(state.phase, "holding")
        self.assertFalse(state.update(1000.001, touching=False))
        self.assertEqual(state.phase, "iti")
        self.assertEqual(state.end_reason, "release")
        state.start_iti(1000.01)
        state.update(1000.2, touching=True, pressed=True)
        self.assertEqual(state.phase, "iti")
        state.update(1000.22, touching=True, pressed=True)
        self.assertEqual(state.phase, "waiting")
        self.assertEqual(state.trial_num, 1)
        state.update(1000.3, touching=True)
        self.assertEqual(state.phase, "waiting")
        state.update(1000.4, touching=True, pressed=True)
        self.assertEqual(state.trial_num, 2)

    def test_unbounded_continuous_mode_has_no_trial_boundaries(self):
        state = self.state(duration=None, end_on_release=False, n=1)
        state.update(0, touching=True)
        self.assertTrue(state.update(1, touching=True))
        state.reward_delivered(1)
        state.update(2, touching=False)
        self.assertTrue(state.cue_visible)
        self.assertEqual(state.phase, "continuous")
        state.update(50, touching=True)
        self.assertFalse(state.update(50.9, touching=True))
        self.assertTrue(state.update(51, touching=True))
        self.assertEqual(state.trial_num, 0)

    def test_release_at_reward_deadline_does_not_reward(self):
        state = self.state()
        state.update(0, touching=True, pressed=True)
        self.assertFalse(state.update(1, touching=False))
        self.assertFalse(state.cue_visible)

    def test_short_buffered_tap_starts_and_ends_without_reward(self):
        state = self.state(duration=None, n=1)
        self.assertFalse(state.update(0.1, touching=False, pressed=True, press_s=0.09))
        self.assertEqual(state.trial_started_s, 0.09)
        self.assertEqual(state.completed_trials, 1)
        state.start_iti(0.11)
        state.update(0.32, touching=False)
        self.assertEqual(state.phase, "done")

    def test_delayed_poll_does_not_queue_catchup_rewards(self):
        state = self.state(duration=None)
        state.update(0, touching=True, pressed=True)
        self.assertTrue(state.update(10, touching=True))
        state.reward_delivered(10)
        self.assertFalse(state.update(10.001, touching=True))
        self.assertTrue(state.update(11, touching=True))

    def test_interval_edit_requires_a_fresh_full_hold_without_catchup_rewards(self):
        for interval in (0.25, 2):
            with self.subTest(interval=interval):
                state = self.state(duration=None, end_on_release=False)
                state.update(0, touching=True)
                self.assertTrue(state.set_reward_interval(interval, now_s=0.8))
                deadline = 0.8 + interval
                self.assertEqual(state.next_reward_s, deadline)
                self.assertFalse(state.update(deadline - 0.001, touching=True))
                self.assertTrue(state.update(deadline, touching=True))
                state.reward_delivered(10)  # A delayed delivery starts one new interval.
                self.assertFalse(state.update(10.001, touching=True))
                self.assertTrue(state.update(10 + interval, touching=True))

    def test_interval_edit_without_contact_waits_for_a_new_hold(self):
        state = self.state(duration=None, end_on_release=False)
        self.assertTrue(state.set_reward_interval(0.5, now_s=5))
        self.assertIsNone(state.next_reward_s)
        self.assertFalse(state.update(10, touching=False))
        self.assertFalse(state.update(11, touching=True))
        self.assertFalse(state.update(11.499, touching=True))
        self.assertTrue(state.update(11.5, touching=True))

    def test_between_trial_interval_edit_preserves_iti_and_uses_new_trial_start(self):
        state = self.state(duration=None)
        state.update(0, touching=True, pressed=True)
        state.update(0.1, touching=False)
        state.start_iti(0.11)
        iti_until = state.iti_until_s
        self.assertTrue(state.set_reward_interval(0.5, now_s=0.15))
        self.assertEqual(state.iti_until_s, iti_until)
        self.assertIsNone(state.next_reward_s)
        state.update(0.32, touching=False)
        self.assertFalse(state.update(1, touching=True, pressed=True))
        self.assertFalse(state.update(1.499, touching=True))
        self.assertTrue(state.update(1.5, touching=True))

    def test_unchanged_or_invalid_interval_edits_preserve_reward_deadline(self):
        state = self.state(duration=None, end_on_release=False)
        state.update(0, touching=True)
        self.assertFalse(state.set_reward_interval(1, now_s=0.8))
        for invalid in (0, -1, float("nan"), float("inf"), True, "1", None):
            with self.subTest(interval=invalid):
                with self.assertRaisesRegex(ValueError, "reward_interval"):
                    state.set_reward_interval(invalid, now_s=0.8)
                self.assertEqual(state.reward_interval, 1)
                self.assertEqual(state.next_reward_s, 1)
        self.assertTrue(state.update(1, touching=True))

    def test_buffered_repress_detects_a_break_without_an_up_sample(self):
        for duration, end_on_release in ((None, True), (5, True), (None, False), (5, False)):
            with self.subTest(duration=duration, end_on_release=end_on_release):
                state = self.state(duration=duration, end_on_release=end_on_release)
                state.update(0, touching=True, pressed=True)
                self.assertFalse(state.update(1, touching=True, pressed=True, press_s=0.99))
                if end_on_release:
                    self.assertEqual(state.phase, "iti")
                    self.assertEqual(state.end_reason, "release")
                else:
                    self.assertFalse(state.update(1.9, touching=True))
                    self.assertTrue(state.update(2, touching=True))

    def test_invalid_timings_fail_before_hardware_or_display(self):
        for name, value in (
            ("duration", 0), ("duration", True), ("duration", float("nan")),
            ("iti", -1), ("iti", float("inf")), ("reward_interval", 0),
            ("reward_interval", "1"), ("end_on_release", "false"), ("n", 0), ("n", True),
        ):
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                self.state(**{name: value})
        with self.assertRaisesRegex(ValueError, "shorter"):
            task.run_task(colors_tsv="unused", reward_interval=0.2, pump_pulse_time_seconds=0.2)


class TouchAndHoldTaskTests(unittest.TestCase):
    def exercise(self, *, contacts, until=3, rotation="normal", preview_enabled=False,
                 preview_failure=False, manual_at=None, eye_enabled=False, pulse_change=None,
                 setting_changes=(), buffered_presses=(), sync_pause=None, **options):
        clock = FakeClock()
        origin = clock.seconds()
        origin_ns = clock.ns
        pending_changes = list(setting_changes)
        def elapsed():
            return clock.seconds() - origin
        def contact():
            return next((position for start, end, position in contacts if start <= elapsed() < end), None)
        mouse = Mock()
        reset_s = [0.0]
        mouse.clickReset.side_effect = lambda: reset_s.__setitem__(0, elapsed())

        def read_buttons(**kwargs):
            buffered = next((press - reset_s[0] for press in buffered_presses
                             if reset_s[0] < press <= elapsed()), 0)
            return [contact() is not None, False, False], [buffered, 0, 0]

        mouse.getPressed.side_effect = read_buttons
        mouse.getPos.side_effect = lambda: contact() or (0, 0)
        manual_sent = False
        def keys(**kwargs):
            nonlocal manual_sent
            if elapsed() >= until:
                return ["escape"]
            if manual_at is not None and elapsed() >= manual_at and not manual_sent:
                manual_sent = True
                return ["r"]
            return []
        psychopy = types.ModuleType("psychopy")
        psychopy.core = types.SimpleNamespace(wait=clock.sleep)
        psychopy.event = types.SimpleNamespace(getKeys=keys, Mouse=Mock(return_value=mouse))
        utils = Mock()
        win = Mock(size=(800, 600))
        win.flip.side_effect = lambda: clock.sleep(0.01)
        main = ScreenGeometry(0, 0, 0, 800, 600, rotation=rotation)
        secondary = ScreenGeometry(1, 800, 0, 800, 600) if preview_enabled else None
        utils.setup_task_window.return_value = win, main, secondary
        utils.resolve_frame_rate.return_value = (60.0, 1.0 / 60.0)

        def service_sync(*args, **kwargs):
            nonlocal sync_pause
            if sync_pause is not None and elapsed() >= sync_pause[0]:
                clock.sleep(sync_pause[1])
                sync_pause = None

        utils.service_sync_bar.side_effect = service_sync
        utils.split_background_from_palette.return_value = (128, 128, 128), {}
        cue = Mock(pos=(0, 0), size=(100, 100), image=Image.new("RGBA", (100, 100)))
        cue.contains.side_effect = lambda pos: abs(pos[0]) <= 50 and abs(pos[1]) <= 50
        pressed_cue = Mock(pos=(0, 0), size=(100, 100), image=cue.image)
        utils.make_initiation_cue_stims.return_value = cue, pressed_cue
        preview = Mock()
        preview.setting_requests = []
        def poll_preview():
            nonlocal pulse_change
            if pulse_change is not None and elapsed() >= pulse_change[0]:
                preview_factory.call_args.kwargs["pump_control"].request(pulse_change[1])
                pulse_change = None
            while pending_changes and elapsed() >= pending_changes[0][0]:
                _, name, value = pending_changes.pop(0)
                preview.setting_requests.append({
                    "name": name, "new_value": value, "task_time_ns": clock.ns - origin_ns,
                    "timestamp_perf_ns": clock.ns,
                })
                preview_factory.call_args.kwargs["pump_control"].settings.request(name, value)
            if preview_failure and elapsed() >= 1.15:
                raise RuntimeError("preview failed")
            return False
        preview.poll.side_effect = poll_preview
        preview.consume_manual_reward_request.return_value = False
        outputs = Mock()
        outputs.validate_bit.side_effect = int
        edges = []
        outputs.write.side_effect = lambda bit, active: edges.append((active, elapsed()))
        eye = Mock()
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {
            "psychopy": psychopy, "bin.utils": utils,
        }), patch.object(bin, "utils", utils, create=True), patch(
            "time.perf_counter", side_effect=clock.seconds,
        ), patch("time.perf_counter_ns", side_effect=lambda: clock.ns), patch.object(
            task, "DAQC2DigitalOutputs", return_value=outputs,
        ), patch.object(task, "ExperimenterPreview", return_value=preview) as preview_factory, patch(
            "bin.task_eye_tracking.TaskEyeTrackingSession", return_value=eye,
        ) as eye_factory:
            kwargs = dict(colors_tsv="palette.tsv", output_dir=tmp, duration=2.5, iti=0.2,
                          reward_interval=1, pump_pulse_time_seconds=0.275, n=1,
                          initiation_cue_center_position=(120, -75), fullscreen=False,
                          eye_tracking_config=EyeTrackingSettings() if eye_enabled else None)
            kwargs.update(options)
            if preview_failure:
                with self.assertRaisesRegex(RuntimeError, "preview failed"):
                    task.run_task(**kwargs)
                status = "failed"
            else:
                status = task.run_task(**kwargs)
            event_path = next(Path(tmp).rglob("event_log.tsv"))
            with event_path.open() as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            metadata = json.loads((event_path.parent / "training.json").read_text())
            behavior_path = event_path.parent / "behavior_log.tsv"
            with behavior_path.open() if behavior_path.exists() else nullcontext() as handle:
                preview.behavior_rows = list(csv.DictReader(handle, delimiter="\t")) if handle else []
            with (event_path.parent / "message_log.tsv").open() as handle:
                preview.message_rows = list(csv.DictReader(handle, delimiter="\t"))
            preview.setting_rows = [
                dict(json.loads(row["message"].split(" ", 1)[1]),
                     task_time_ns=int(row["task_time_ns"]))
                for row in preview.message_rows if row["message"].startswith("live_setting ")
            ]
        self.assertFalse(edges[-1][0])
        self.assertEqual(rows[-1]["event"], "block_end")
        self.assertFalse(metadata["experimental_data"])
        utils.close_task_window.assert_called_once_with(win)
        if preview_enabled:
            preview.close.assert_called_once()
        if eye_enabled:
            eye_factory.assert_called_once()
            eye.set_screen_geometry.assert_called_once_with((800, 600), software_stimulus_rotation(rotation))
            self.assertIs(preview_factory.call_args.kwargs["eye_tracking_shared"], eye.preview_state)
            eye.close.assert_called_once()
        return status, rows, edges, utils, preview

    def test_sync_bar_is_configured_with_rotation_and_serviced_during_static_hold(self):
        config = {"enabled": True, "height": 24}
        _, _, _, utils, preview = self.exercise(
            contacts=[], until=0.1, preview_enabled=True, rotation="inverted",
            sync_bar=config, sync_pin=23,
        )
        call = utils.configure_sync_bar.call_args
        self.assertIs(call.args[1], config)
        self.assertEqual(call.kwargs["sync_pin"], 23)
        self.assertEqual(call.kwargs["rotation_degrees"], 270)
        self.assertEqual(call.kwargs["frame_duration_s"], 1.0 / 60.0)
        self.assertIs(call.kwargs["experimenter_preview"], preview)
        self.assertGreater(utils.service_sync_bar.call_count, 2)

    def test_sync_swaps_respect_an_active_reward_deadline(self):
        _, _, _, utils, _ = self.exercise(
            contacts=[(0.1, 1.5, (0, 0))], duration=None,
            sync_bar={"enabled": True},
        )
        deadlines = [call.kwargs["deadline_perf_s"]
                     for call in utils.service_sync_bar.call_args_list]
        self.assertTrue(any(deadline is not None for deadline in deadlines))
        self.assertTrue(any(deadline is None for deadline in deadlines))

    def test_short_tap_during_sync_swap_keeps_original_press_timestamp(self):
        _, rows, _, _, _ = self.exercise(
            contacts=[(0.055, 0.056, (0, 0))], buffered_presses=[0.055],
            sync_pause=(0.050, 0.020), sync_bar={"enabled": True}, duration=None,
        )
        presses = [row for row in rows if row["event"] == "cue_touch"]
        releases = [row for row in rows if row["event"] == "training_cue_release"]
        self.assertEqual(len(presses), 1)
        self.assertEqual(int(presses[0]["task_time_ns"]), 55_000_000)
        self.assertEqual(len(releases), 1)
        self.assertGreaterEqual(int(releases[0]["task_time_ns"]), 70_000_000)

    def test_release_cancels_active_reward_and_starts_full_iti(self):
        status, rows, edges, _, _ = self.exercise(
            contacts=[(0.1, 1.15, (0, 0))], duration=None,
        )
        self.assertEqual(status, "done")
        on = next(t for active, t in edges if active)
        off = next(t for active, t in edges if not active and t > on)
        self.assertAlmostEqual(on, 1.1, delta=0.002)
        self.assertAlmostEqual(off, 1.15, delta=0.002)
        self.assertLess(off - on, 0.275)
        gray = next(row for row in rows if row["event"] == "gray_inter_trial_interval")
        self.assertGreaterEqual(int(rows[-1]["task_time_ns"]) - int(gray["task_time_ns"]), 200_000_000)

    def test_pulse_edit_applies_to_next_trial_and_is_logged_for_each_trial(self):
        status, rows, _, _, preview = self.exercise(
            contacts=[(0.1, 1.5, (0, 0)), (1.8, 3.5, (0, 0))],
            duration=1.3, n=2, until=3.5, preview_enabled=True, pulse_change=(0.5, 0.15),
        )
        self.assertEqual(status, "done")
        self.assertEqual([float(row["pump_pulse_time"]) for row in preview.behavior_rows], [0.275, 0.15])
        self.assertEqual([float(row["requested_duration"]) for row in rows if row["event"] == "pump_on"],
                         [0.275, 0.15])

    def test_continuous_pulse_edit_applies_on_next_reward(self):
        _, rows, _, _, preview = self.exercise(
            contacts=[(0.1, 3, (0, 0))], duration=None, end_on_release=False,
            preview_enabled=True, pulse_change=(1.15, 0.15),
        )
        self.assertEqual([float(row["requested_duration"]) for row in rows if row["event"] == "pump_on"],
                         [0.275, 0.15])
        self.assertEqual(preview.behavior_rows, [])

    def test_hold_interval_edit_is_frozen_per_trial_and_logs_confirmation_and_application(self):
        status, rows, edges, _, preview = self.exercise(
            contacts=[(0.1, 1.5, (0, 0)), (1.8, 3.5, (0, 0))],
            duration=1.3, n=2, until=3.5, preview_enabled=True,
            setting_changes=[(0.5, "reward_interval", 0.5)],
        )
        self.assertEqual(status, "done")
        self.assertEqual([float(row["reward_interval"]) for row in preview.behavior_rows], [1, 0.5])
        self.assertEqual([float(row["pump_pulse_time"]) for row in preview.behavior_rows], [0.275, 0.275])
        onsets = [timestamp for active, timestamp in edges if active]
        self.assertEqual(len(onsets), 3)
        for actual, expected in zip(onsets, (1.1, 2.3, 2.8)):
            self.assertAlmostEqual(actual, expected, delta=0.002)
        changes = {row["phase"]: row for row in preview.setting_rows if row["name"] == "reward_interval"}
        request, confirmed, applied = preview.setting_requests[0], changes["confirmed"], changes["applied"]
        self.assertEqual(confirmed["task_time_ns"], request["task_time_ns"])
        self.assertEqual((confirmed["old_value"], confirmed["new_value"]), (1, 0.5))
        self.assertEqual((applied["old_value"], applied["new_value"]), (1, 0.5))
        self.assertEqual(applied["request_id"], confirmed["request_id"])
        self.assertEqual(applied["requested_perf_ns"], request["timestamp_perf_ns"])
        first_end = next(int(row["task_time_ns"]) for row in rows if row["event"] == "training_trial_end")
        second_start = next(int(row["task_time_ns"]) for row in rows
                            if row["event"] == "training_trial_start" and row["trial_num"] == "2")
        self.assertGreater(applied["task_time_ns"], first_end)
        self.assertLess(applied["task_time_ns"], second_start)

    def test_continuous_hold_interval_edit_requires_full_new_hold(self):
        _, _, edges, _, preview = self.exercise(
            contacts=[(0.1, 3, (0, 0))], duration=None, end_on_release=False,
            until=2.2, preview_enabled=True, setting_changes=[(0.8, "reward_interval", 0.5)],
        )
        onsets = [timestamp for active, timestamp in edges if active]
        self.assertEqual(len(onsets), 2)
        applied = next(row for row in preview.setting_rows
                       if row["name"] == "reward_interval" and row["phase"] == "applied")
        self.assertEqual(applied["task_time_ns"], preview.setting_requests[0]["task_time_ns"])
        self.assertAlmostEqual(onsets[0], applied["task_time_ns"] / 1e9 + 0.5, delta=0.002)
        self.assertAlmostEqual(onsets[1] - onsets[0], 0.5, delta=0.002)
        self.assertEqual(preview.behavior_rows, [])

    def test_pump_only_and_unchanged_interval_edits_preserve_reward_schedule(self):
        _, rows, edges, _, preview = self.exercise(
            contacts=[(0.1, 3, (0, 0))], duration=None, end_on_release=False,
            until=2.5, preview_enabled=True,
            setting_changes=[(0.8, "reward_interval", 1), (0.8, "pump_pulse_time", 0.15)],
        )
        onsets = [timestamp for active, timestamp in edges if active]
        self.assertEqual(len(onsets), 2)
        for actual, expected in zip(onsets, (1.1, 2.1)):
            self.assertAlmostEqual(actual, expected, delta=0.002)
        self.assertEqual([float(row["requested_duration"]) for row in rows if row["event"] == "pump_on"],
                         [0.15, 0.15])
        self.assertFalse(any(row["name"] == "reward_interval" and row["phase"] != "initial"
                             for row in preview.setting_rows))

    def test_interval_edit_during_pulse_waits_for_pump_off_before_restarting_hold(self):
        _, _, edges, _, preview = self.exercise(
            contacts=[(0.1, 3, (0, 0))], duration=None, end_on_release=False,
            until=2.3, preview_enabled=True, setting_changes=[(1.15, "reward_interval", 0.5)],
        )
        onsets = [timestamp for active, timestamp in edges if active]
        self.assertEqual(len(onsets), 2)
        first_off = next(timestamp for active, timestamp in edges if not active and timestamp > onsets[0])
        changes = {row["phase"]: row for row in preview.setting_rows if row["name"] == "reward_interval"}
        confirmed, applied = changes["confirmed"], changes["applied"]
        self.assertEqual(confirmed["task_time_ns"], preview.setting_requests[0]["task_time_ns"])
        self.assertGreater(confirmed["task_time_ns"] / 1e9, onsets[0])
        self.assertLess(confirmed["task_time_ns"] / 1e9, first_off)
        self.assertGreater(applied["task_time_ns"] / 1e9, first_off)
        self.assertAlmostEqual(first_off - onsets[0], 0.275, delta=0.002)
        self.assertAlmostEqual(onsets[1], applied["task_time_ns"] / 1e9 + 0.5, delta=0.002)

    def test_duration_expires_even_when_release_is_disabled(self):
        status, rows, edges, _, _ = self.exercise(
            contacts=[(0.1, 4, (0, 0))], end_on_release=False, until=4,
        )
        self.assertEqual(status, "done")
        self.assertEqual(sum(active for active, _ in edges), 2)
        start = next(row for row in rows if row["event"] == "training_trial_start")
        end = next(row for row in rows if row["event"] == "training_trial_end")
        self.assertAlmostEqual((int(end["task_time_ns"]) - int(start["task_time_ns"])) / 1e9, 2.5, delta=0.002)

    def test_moving_outside_cue_ends_trial_and_held_contact_does_not_restart(self):
        status, rows, _, _, _ = self.exercise(
            contacts=[(0.1, 0.5, (0, 0)), (0.5, 0.6, (300, 0)), (0.6, 2, (0, 0))],
            duration=None, n=2, until=1.5,
        )
        self.assertEqual(status, "user_exit")
        self.assertEqual(sum(row["event"] == "training_trial_start" for row in rows), 1)
        self.assertEqual(sum(row["event"] == "onset_cue_on" for row in rows), 2)

    def test_continuous_mode_resets_reward_clock_without_hiding_cue(self):
        status, rows, edges, _, _ = self.exercise(
            contacts=[(0.1, 1.3, (0, 0)), (1.5, 3, (0, 0))],
            duration=None, end_on_release=False,
        )
        self.assertEqual(status, "user_exit")
        self.assertEqual(sum(active for active, _ in edges), 2)
        self.assertEqual(sum(row["event"] == "onset_cue_on" for row in rows), 1)
        self.assertFalse(any(row["event"] in {"training_trial_start", "training_trial_end", "gray_inter_trial_interval"} for row in rows))
        self.assertTrue(all(row["trial_num"] == "" for row in rows))

    def test_preview_failure_clears_active_pump_and_closes_everything(self):
        _, _, edges, _, _ = self.exercise(
            contacts=[(0.1, 3, (0, 0))], preview_enabled=True, preview_failure=True,
        )
        self.assertTrue(any(active for active, _ in edges))
        self.assertLess(edges[-1][1], 1.375)

    def test_manual_reward_exit_and_shared_eye_preview(self):
        status, rows, edges, _, preview = self.exercise(
            contacts=[], preview_enabled=True, manual_at=0.1, until=0.2, eye_enabled=True,
        )
        self.assertEqual(status, "user_exit")
        self.assertEqual(sum(active for active, _ in edges), 1)
        self.assertEqual(preview.show_static_scene.call_args.kwargs["status_counts"]["Manual rewards"], 1)
        self.assertTrue(all(row["trial_num"] == "" for row in rows if row["event"].startswith("pump_")))

    def test_rotated_cue_uses_same_subject_coordinates(self):
        for rotation in ("normal", "left", "inverted", "right"):
            with self.subTest(rotation=rotation):
                _, _, _, utils, _ = self.exercise(contacts=[], until=0.05, rotation=rotation)
                pos = utils.make_initiation_cue_stims.call_args.kwargs["position"]
                self.assertEqual(rotate_centered_point(pos, -software_stimulus_rotation(rotation)), (120, -75))

    def test_launcher_config_and_cue_settings_match_reference(self):
        launcher = json.loads((ROOT / "config_files/interface/rpi_launch_config.json").read_text())
        entry = launcher["tasks"]["trainings"]["touch_and_hold"]
        self.assertTrue((ROOT / entry["launch"]).is_file())
        cfg = json.loads((ROOT / entry["config"]).read_text())
        reference = json.loads((ROOT / "config_files/csc2/yuri_headfix_train_noseq_nomem_20260930.json").read_text())
        for key in ("colors_tsv", "initiation_cue_center_position", "initiation_cue_style", "init_dot_color", "eye_tracking"):
            self.assertEqual(cfg[key], reference[key])
        cfg["duration"] = None
        cfg["eye_tracking"] = {"enabled": False}
        cfg["screens"] = {"main": 0, "experimenter": 1}
        cfg["daq"] = {"address": 2, "module_name": "fake.DAQC2plate"}
        with patch.object(sys, "argv", ["touch_and_hold", "--config", entry["config"]]), patch.object(
            task, "load_config", return_value=cfg,
        ), patch.object(task, "run_task", return_value="done") as run:
            self.assertEqual(task.main(), 0)
        self.assertIsNone(run.call_args.kwargs["duration"])
        self.assertTrue(run.call_args.kwargs["end_on_release"])
        self.assertEqual(run.call_args.kwargs["daq_address"], 2)
        self.assertEqual(run.call_args.kwargs["daq_module_name"], "fake.DAQC2plate")


if __name__ == "__main__":
    unittest.main()
