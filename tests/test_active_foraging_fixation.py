"""Headless regressions for gaze-gated AFC presentation and flip timing."""

import csv
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

from bin.logger import EventCodeLibrary, EventLogger, load_task_event_definitions
from bin.dot_fixation import FIXATION_UNAVAILABLE
from bin.task_clock import TaskClock
from tests.test_active_foraging_timing import FakeClock


class Stimulus:
    def __init__(self, win=None, **kwargs):
        self.win = win
        self.pos = (0, 0)
        self.size = (20, 20)
        for key, value in kwargs.items():
            setattr(self, key, value)

    def draw(self):
        if self.win is not None and getattr(self, "is_image", False):
            self.win.pending_images.append(tuple(self.pos))

    def contains(self, position):
        return (
            hasattr(self, "width") and hasattr(self, "height")
            and abs(position[0] - self.pos[0]) <= self.width / 2
            and abs(position[1] - self.pos[1]) <= self.height / 2
        )


class Display:
    def __init__(self, clock):
        self.clock = clock
        self.size = (800, 600)
        self.anchor_ns = clock.ns
        self.refresh = 0
        self.flips = []
        self.callbacks = []
        self.pending_images = []
        self.scenes = []

    def callOnFlip(self, callback, *args, **kwargs):
        self.callbacks.append((callback, args, kwargs))

    def flip(self):
        self.refresh = max(
            self.refresh + 1,
            math.ceil((self.clock.ns - self.anchor_ns) * 60 / 1e9 - 1e-7),
        )
        self.clock.ns = self.anchor_ns + round(self.refresh / 60 * 1e9)
        self.flips.append(self.clock.ns)
        self.scenes.append(tuple(self.pending_images))
        self.pending_images.clear()
        callbacks, self.callbacks = self.callbacks, []
        for callback, args, kwargs in callbacks:
            callback(*args, **kwargs)
        return self.clock.seconds()


class Gate:
    """Controllable acquisition result; the real sample policy has separate tests."""

    fixation_window_radius = 40

    def __init__(self, accept_after_s, schedule=None, arrival_after_s=...):
        self.accept_after_s = accept_after_s
        self.schedule = schedule
        self.arrival_after_s = arrival_after_s
        self.starts = []
        self.stopped = False

    def start(self, positions, **kwargs):
        self.starts.append((positions, kwargs))
        self.onset_s = kwargs["onset_perf_s"]
        self.stopped = False

    def status(self, *, now_s):
        if self.schedule is not None:
            accepted = None
            for onset, value in self.schedule:
                if now_s - self.onset_s >= onset - 1e-9:
                    accepted = value
            return accepted
        delay = self._per_dot(self.accept_after_s)
        if delay is None:
            return None
        return 1 if now_s - self.onset_s >= delay - 1e-9 else None

    def _per_dot(self, value):
        return value[len(self.starts) - 1] if isinstance(value, tuple) else value

    def stop(self):
        self.stopped = True

    def reached_dot_before(self, deadline_perf_s):
        arrival = self._per_dot(self.arrival_after_s)
        if arrival is ...:
            arrival = (
                next((onset for onset, target in self.schedule if target is not None), None)
                if self.schedule is not None else self._per_dot(self.accept_after_s)
            )
        return arrival is not None and self.onset_s + arrival <= min(
            time.perf_counter(), deadline_perf_s,
        ) + 1e-9


def load_presenter(clock):
    psychopy = types.ModuleType("psychopy")
    psychopy.visual = types.SimpleNamespace(
        Window=Display, ImageStim=Stimulus, Circle=Stimulus, Rect=Stimulus,
        TextStim=Stimulus,
    )
    mouse = Mock()
    mouse.getPressed.return_value = ((False, False, False), (0, 0, 0))
    mouse.getPos.return_value = (0, 0)
    psychopy.event = types.SimpleNamespace(
        Mouse=Mock(return_value=mouse), clearEvents=Mock(),
        getKeys=Mock(return_value=[]),
    )
    psychopy.core = types.SimpleNamespace(wait=clock.sleep)
    path = Path(__file__).resolve().parents[1] / "bin/utils.py"
    spec = importlib.util.spec_from_file_location("bin._foraging_fixation_under_test", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"psychopy": psychopy}):
        spec.loader.exec_module(module)
    return module, psychopy


class ActiveForagingFixationTests(unittest.TestCase):
    def present(self, *, accepted_after=0.2, sequential=True, memory=True,
                isi=0.1, timeout=0.5, gated=True, abort_after=None,
                schedule=None, duration=0.1, choice_time=0.1, event_logger=None,
                sync_service=None, arrival_after=..., click_after=None,
                click_intervals=None, click_index=1, buffered_touch=None,
                on_choice_start=None, event_profile="active_foraging", hold_cue=False,
                require_saccade_to_all_options=True, num_options=2, msg_logger=None,
                hold_cue_to_init_time_s=0.0, setup_mouse=None):
        clock = FakeClock()
        module, psychopy = load_presenter(clock)
        window = Display(clock)
        gate = Gate(accepted_after, schedule=schedule, arrival_after_s=arrival_after) if gated else None
        logger = Mock() if event_logger is None else event_logger
        preview = Mock()
        metadata = {}
        options = [(index + 1, 1) for index in range(num_options)]
        positions = [(index * 200 - 100, 0) for index in range(num_options)]
        tone_times = []

        def choice_callback():
            tone_times.append(clock.ns)
            if on_choice_start is not None:
                on_choice_start()

        if hold_cue:
            mouse = psychopy.event.Mouse.return_value
            mouse.getPressed.side_effect = lambda **kwargs: (
                (bool(window.flips and not tone_times), False, False), (0, 0, 0),
            )
        if click_after is not None or click_intervals is not None:
            mouse = psychopy.event.Mouse.return_value
            mouse.getPos.return_value = positions[click_index - 1]
            intervals = click_intervals if click_intervals is not None else [(click_after, math.inf)]
            mouse.getPressed.side_effect = lambda **kwargs: (
                (bool(window.flips and any(start <= clock.seconds() - window.flips[0] / 1e9 < stop
                                          for start, stop in intervals)), False, False),
                (0, 0, 0),
            )
        if buffered_touch is not None:
            mouse = psychopy.event.Mouse.return_value
            mouse.getPos.return_value = positions[click_index - 1]
            reset_s = [clock.seconds()]
            sent = []
            mouse.clickReset.side_effect = lambda: reset_s.__setitem__(0, clock.seconds())

            def buffered_buttons(**kwargs):
                observed_after, pressed_after = buffered_touch
                elapsed = 0.0
                if window.flips and not sent and clock.seconds() - window.flips[0] / 1e9 >= observed_after:
                    elapsed = window.flips[0] / 1e9 + pressed_after - reset_s[0]
                    sent.append(True)
                return ((False, False, False), (elapsed, 0, 0))

            mouse.getPressed.side_effect = buffered_buttons

        if setup_mouse is not None:
            setup_mouse(clock, window, psychopy.event.Mouse.return_value)

        def abort():
            return bool(abort_after is not None and gate.starts
                        and clock.seconds() - gate.onset_s >= abort_after)

        with patch.dict(sys.modules, {"psychopy": psychopy}), patch(
            "time.perf_counter", side_effect=clock.seconds,
        ), patch("time.perf_counter_ns", side_effect=lambda: clock.ns), patch(
            "time.sleep", side_effect=clock.sleep,
        ), patch.object(module, "make_image_stim_from_array", side_effect=lambda *a, **kw: Stimulus(window, is_image=True)), patch.object(
            module, "service_sync_bar", side_effect=sync_service,
        ), patch.object(module, "serialize_preview_image", return_value=None):
            result = module.present_trial_with_persistent_dots(
                win=window,
                preloaded={option: np.zeros((20, 20, 3), dtype=np.uint8) for option in options},
                trial_options=options, positions=positions,
                duration=duration if sequential or memory else 0,
                choice_time=choice_time, dot_size=10, dot_color=(50, 50, 50),
                bg_rect=Stimulus(), fix=None, logger=logger, trial_num=3,
                isi=isi, bg_rgb_255=(128, 128, 128), fps=60,
                sequential=sequential, is_memory=memory, trial_meta=metadata,
                experimenter_preview=preview, external_abort_checker=abort,
                scene_main_size=window.size, event_profile=event_profile,
                reward_levels=[3] + [0] * (num_options - 1),
                dot_fixation_gate=gate, saccade_wait_timeout_s=timeout,
                require_saccade_to_all_options=require_saccade_to_all_options,
                msg_logger=msg_logger,
                on_choice_start=choice_callback if on_choice_start is not None or hold_cue else None,
                onset_cue=Stimulus() if hold_cue else None,
                pressed_onset_cue=Stimulus() if hold_cue else None,
                hold_before_choice=hold_cue,
                hold_cue_to_init_time_s=hold_cue_to_init_time_s,
            )
        events = (
            [call.kwargs for call in logger.method_calls if "event" in call.kwargs]
            if event_logger is None else []
        )
        for event in events:
            if "timestamp_perf_ns" not in event and "timestamp_perf_s" in event:
                event["timestamp_perf_ns"] = round(event["timestamp_perf_s"] * 1e9)
        return result, events, metadata, gate, window, preview

    def test_choice_callback_matches_opening_flip_in_all_modes(self):
        for profile in ("active_foraging", "match2cue"):
            for sequential, memory in ((False, False), (False, True), (True, False), (True, True)):
                for gated in (False, True):
                    with self.subTest(profile=profile, sequential=sequential, memory=memory, gated=gated):
                        times = []

                        def tone():
                            times.append(time.perf_counter_ns())
                            time.sleep(0.004)  # Hardware latency cannot move the captured flip time.

                        result, events, _, _, window, _ = self.present(
                            sequential=sequential, memory=memory, gated=gated,
                            event_profile=profile, on_choice_start=tone,
                        )
                        choice = next(row for row in events if row["event"] == "choice_start")
                        self.assertEqual(times, [choice["timestamp_perf_ns"]])
                        self.assertIn(times[0], window.flips)

    def test_tone_allows_immediate_held_cue_release_and_skips_unreached_choice(self):
        callback = Mock()
        result, events, *_ = self.present(hold_cue=True, on_choice_start=callback)
        self.assertEqual(result, (False, None))
        callback.assert_called_once_with()
        self.assertNotIn("initiation_cue_release", [event["event"] for event in events])
        callback.reset_mock()
        self.present(accepted_after=None, timeout=0.05, on_choice_start=callback)
        callback.assert_not_called()

    def test_each_sequential_dot_waits_for_fixation_and_logs_once(self):
        result, events, metadata, gate, window, preview = self.present()
        self.assertEqual(result, (False, None))
        self.assertNotIn("saccade_timeout", metadata)
        accepted = [event for event in events if event["event"] == "fixation_accepted"]
        self.assertEqual(len(accepted), 2)
        self.assertEqual(len(gate.starts), 2)
        for index, acceptance in enumerate(accepted, 1):
            dot = next(event for event in events if event["event"] == f"option_{index}_dot")
            stimulus = next(event for event in events if event["event"] == f"option_{index}_on")
            self.assertGreaterEqual(acceptance["timestamp_perf_ns"] - dot["timestamp_perf_ns"], 200_000_000 - 1)
            self.assertGreaterEqual(stimulus["timestamp_perf_ns"], acceptance["timestamp_perf_ns"])
            self.assertLess(stimulus["timestamp_perf_ns"] - acceptance["timestamp_perf_ns"], 20_000_000)
        self.assertTrue(gate.stopped)
        # Static waiting does not create repeated subject-screen swaps.
        self.assertEqual(len(window.flips), 7)
        circles = [circle for call in preview.show_static_scene.call_args_list
                   for circle in call.kwargs.get("circles", [])]
        self.assertTrue(circles)
        self.assertTrue(all(circle["radius"] == 40 for circle in circles))

    def test_fast_fixation_does_not_shorten_minimum_isi(self):
        _, events, _, _, _, _ = self.present(accepted_after=0.02, isi=0.2)
        for index in (1, 2):
            dot = next(event for event in events if event["event"] == f"option_{index}_dot")
            stimulus = next(event for event in events if event["event"] == f"option_{index}_on")
            self.assertAlmostEqual((stimulus["timestamp_perf_ns"] - dot["timestamp_perf_ns"]) / 1e9, 0.2, places=8)

    def test_timeout_clears_without_presenting_stimulus_or_opening_choice(self):
        result, events, metadata, gate, window, _ = self.present(accepted_after=None, timeout=0.25)
        self.assertEqual(result, (False, None))
        self.assertTrue(metadata["saccade_timeout"])
        names = [event["event"] for event in events]
        self.assertNotIn("option_1_on", names)
        self.assertNotIn("choice_start", names)
        self.assertNotIn("fixation_accepted", names)
        dot = next(event for event in events if event["event"] == "option_1_dot")
        self.assertGreaterEqual(metadata["gray_flip_perf_ns"] - dot["timestamp_perf_ns"], 250_000_000)
        self.assertLess(metadata["gray_flip_perf_ns"] - dot["timestamp_perf_ns"], 270_000_000)
        self.assertEqual(len(window.flips), 2)
        self.assertTrue(gate.stopped)

    def test_manual_abort_still_stops_task(self):
        result, events, metadata, gate, _, _ = self.present(accepted_after=None, abort_after=0.05)
        self.assertEqual(result, (True, None))
        self.assertFalse(metadata.get("saccade_timeout", False))
        self.assertNotIn("fixation_accepted", [event["event"] for event in events])
        self.assertTrue(gate.stopped)

    def test_simultaneous_reveals_only_accepted_option_and_hides_on_gaze_loss(self):
        result, events, metadata, gate, window, _ = self.present(
            sequential=False, memory=False, choice_time=0.35,
            schedule=[(0.02, 1), (0.14, None), (0.20, 2), (0.29, None), (0.33, 1)],
        )
        self.assertEqual(result, (False, None))
        self.assertFalse(metadata.get("saccade_timeout"))
        self.assertEqual(len(gate.starts), 1)
        self.assertEqual(list(gate.starts[0][0]), [(-100, 0), (100, 0)])
        self.assertEqual(window.scenes, [(), ((-100, 0),), (), ((100, 0),), (), ((-100, 0),), ()])
        names = [event["event"] for event in events]
        self.assertNotIn("options_on", names)
        self.assertEqual(names.count("fixation_accepted"), 3)
        dots = next(event for event in events if event["event"] == "options_dot")
        first = next(event for event in events if event["event"] == "option_1_on")
        choice = next(event for event in events if event["event"] == "choice_start")
        self.assertAlmostEqual((first["timestamp_perf_ns"] - dots["timestamp_perf_ns"]) / 1e9, 0.1, places=8)
        self.assertEqual(choice["timestamp_perf_ns"], dots["timestamp_perf_ns"])
        self.assertAlmostEqual((metadata["gray_flip_perf_ns"] - choice["timestamp_perf_ns"]) / 1e9, 0.35, places=8)
        self.assertTrue(gate.stopped)

    def test_simultaneous_memory_returns_to_dots_for_the_entire_choice_window(self):
        _, events, metadata, _, window, _ = self.present(
            sequential=False, memory=True, choice_time=0.2, duration=0.15,
            schedule=[(0.02, 1), (0.15, 2)],
        )
        names = [event["event"] for event in events]
        self.assertEqual(names.count("choice_start"), 1)
        first = next(event for event in events if event["event"] == "option_1_on")
        choice = next(event for event in events if event["event"] == "choice_start")
        self.assertAlmostEqual((choice["timestamp_perf_ns"] - first["timestamp_perf_ns"]) / 1e9, 0.15, places=8)
        self.assertAlmostEqual((metadata["gray_flip_perf_ns"] - choice["timestamp_perf_ns"]) / 1e9, 0.2, places=8)
        self.assertEqual(window.scenes, [(), ((-100, 0),), ((100, 0),), (), ()])

    def test_unavailable_fixation_does_not_hide_or_log_another_acceptance(self):
        _, events, metadata, _, window, _ = self.present(
            sequential=False, memory=False, choice_time=0.4,
            schedule=[(0.02, 1), (0.14, FIXATION_UNAVAILABLE), (0.20, 1),
                      (0.25, None), (0.30, 1)],
        )
        self.assertEqual(window.scenes, [(), ((-100, 0),), (), ((-100, 0),), ()])
        names = [event["event"] for event in events]
        self.assertEqual(names.count("fixation_accepted"), 2)
        self.assertEqual(names.count("option_1_on"), 2)
        self.assertEqual(names.count("option_1_off"), 2)
        choice = next(event for event in events if event["event"] == "choice_start")
        self.assertEqual(metadata["gray_flip_perf_ns"] - choice["timestamp_perf_ns"], 400_000_000)

    def test_unavailable_fixation_cannot_accept_or_suspend_deadlines(self):
        for sequential in (False, True):
            with self.subTest(sequential=sequential):
                result, events, metadata, _, window, _ = self.present(
                    sequential=sequential, memory=False, choice_time=0.3,
                    timeout=0.15, arrival_after=None, schedule=[(0, FIXATION_UNAVAILABLE)],
                )
                self.assertEqual(result, (False, None))
                self.assertTrue(metadata["saccade_timeout"])
                self.assertNotIn("fixation_accepted", [event["event"] for event in events])
                self.assertEqual(window.scenes, [(), ()])
        _, events, metadata, _, window, _ = self.present(
            sequential=False, memory=False, choice_time=0.3,
            schedule=[(0.02, 1), (0.14, FIXATION_UNAVAILABLE)],
        )
        choice = next(event for event in events if event["event"] == "choice_start")
        self.assertEqual(metadata["gray_flip_perf_ns"] - choice["timestamp_perf_ns"], 300_000_000)
        self.assertEqual(window.scenes, [(), ((-100, 0),), ()])

    def test_gray_clear_logs_offsets_for_exactly_the_visible_options(self):
        cases = (
            (dict(sequential=False, memory=False, schedule=[(0.02, 1)]), [1]),
            (dict(sequential=False, memory=False, schedule=[(0.02, 1), (0.14, None)]), []),
            (dict(sequential=False, memory=False, schedule=[(0.02, 1)], click_after=0.2), [1]),
            (dict(sequential=False, memory=False, gated=False), [1, 2]),
            (dict(sequential=True, memory=False, gated=False), [1, 2]),
            (dict(sequential=False, memory=True, schedule=[(0.02, 1)]), []),
            (dict(sequential=True, memory=True, gated=False), []),
        )
        for options, indices in cases:
            with self.subTest(options=options):
                _, events, metadata, _, _, _ = self.present(choice_time=0.3, **options)
                final_off = [event for event in events if event["event"].endswith("_off")
                             and event["timestamp_perf_ns"] == metadata["gray_flip_perf_ns"]]
                self.assertEqual([event["event"] for event in final_off],
                                 [f"option_{index}_off" for index in indices])
                for event in final_off:
                    self.assertEqual(event["requested_timestamp_perf_ns"], metadata["gray_flip_requested_perf_ns"])

    def test_memory_choice_logs_offset_at_dot_flip_without_duplicate_at_gray(self):
        _, events, _, _, _, _ = self.present(
            sequential=False, memory=True, schedule=[(0.02, 1)], choice_time=0.2,
        )
        choice = next(event for event in events if event["event"] == "choice_start")
        offsets = [event for event in events if event["event"] == "option_1_off"]
        self.assertEqual(len(offsets), 1)
        self.assertEqual(offsets[0]["timestamp_perf_ns"], choice["timestamp_perf_ns"])

    def test_touch_and_failed_hold_log_original_detection_times_with_real_logger(self):
        for during_initiation in (False, True):
            with self.subTest(during_initiation=during_initiation), tempfile.TemporaryDirectory() as tmp:
                definitions, patterns = load_task_event_definitions("active_foraging")
                logger = EventLogger(tmp, TaskClock(origin_perf_ns=10_000_000_000),
                                     EventCodeLibrary(definitions, event_patterns=patterns), auto_flush=False)
                captures = {}

                def setup(clock, window, mouse):
                    def buttons(**kwargs):
                        down = bool(window.flips and clock.ns < window.flips[0] + 170_000_003)
                        if down:
                            captures.setdefault("press", clock.ns)
                        elif "press" in captures:
                            captures.setdefault("release", clock.ns)
                        return ((down, False, False), (0, 0, 0))

                    def position():
                        clock.ns += 3_000_007  # Position lookup must not shift either capture.
                        return (0, 0)

                    mouse.getPressed.side_effect = buttons
                    mouse.getPos.side_effect = position

                try:
                    result, _, metadata, _, _, _ = self.present(
                        hold_cue=True, setup_mouse=setup, event_logger=logger,
                        hold_cue_to_init_time_s=1.0 if during_initiation else 0.0,
                        accepted_after=0.02, duration=0.5,
                    )
                finally:
                    logger.close()
                self.assertEqual(result, (False, None))
                with logger.path.open() as handle:
                    rows = list(csv.DictReader(handle, delimiter="\t"))
                press = next(row for row in rows if row["event"] == "cue_touch")
                release = next(row for row in rows if row["event"] == "initiation_cue_release")
                self.assertEqual(int(press["task_time_ns"]), captures["press"] - 10_000_000_000)
                self.assertEqual(int(release["task_time_ns"]), captures["release"] - 10_000_000_000)
                self.assertEqual(metadata["initiation_cue_release_perf_ns"], captures["release"])
                self.assertEqual(metadata["initiation_cue_release_task_time_ns"], int(release["task_time_ns"]))
                self.assertGreater(metadata["gray_flip_perf_ns"] - captures["release"], 70_000_000)
                if not during_initiation:
                    offsets = [row for row in rows if row["event"] == "option_1_off"]
                    self.assertEqual(len(offsets), 1)
                    self.assertEqual(int(offsets[0]["task_time_ns"]), metadata["gray_flip_perf_ns"] - 10_000_000_000)

    def test_buffered_choice_touch_keeps_press_timestamp_in_event_and_attempt_log(self):
        messages = Mock()
        _, events, _, _, window, _ = self.present(
            sequential=False, memory=False, choice_time=0.4, accepted_after=0.02,
            buffered_touch=(0.25, 0.20), msg_logger=messages,
        )
        press_ns = window.flips[0] + 200_000_000
        touch = next(event for event in events if event["event"] == "option_touch")
        self.assertEqual(touch["timestamp_perf_ns"], press_ns)
        attempts = [call for call in messages.log.call_args_list if call.args[1].startswith("choice_touch_attempt ")]
        self.assertEqual(len(attempts), 1)
        self.assertEqual(attempts[0].kwargs["timestamp_perf_ns"], press_ns)

    def test_late_fixation_does_not_rescue_initial_timeout(self):
        for sequential in (False, True):
            with self.subTest(sequential=sequential):
                result, events, metadata, gate, window, _ = self.present(
                    sequential=sequential, memory=False,
                    accepted_after=0.3, timeout=0.25, choice_time=0.5,
                )
                self.assertEqual(result, (False, None))
                self.assertTrue(metadata["saccade_timeout"])
                self.assertNotIn("fixation_accepted", [event["event"] for event in events])
                self.assertEqual(window.scenes, [(), ()])
                self.assertTrue(gate.stopped)

    def test_simultaneous_timeout_is_satisfied_once_and_not_rearmed_by_gaze_loss(self):
        for memory in (False, True):
            with self.subTest(memory=memory):
                result, events, metadata, gate, _, _ = self.present(
                    sequential=False, memory=memory, timeout=0.1,
                    duration=0.1, choice_time=0.4,
                    schedule=[(0.02, 1), (0.16, None), (0.29, 2)],
                )
                self.assertEqual(result, (False, None))
                self.assertFalse(metadata.get("saccade_timeout"))
                names = [event["event"] for event in events]
                self.assertEqual(names.count("fixation_accepted"), 2)
                self.assertIn("choice_start", names)
                self.assertNotIn("saccade_timeout", names)
                if not memory:
                    self.assertIn("option_2_on", names)
                choice = next(event for event in events if event["event"] == "choice_start")
                self.assertAlmostEqual((metadata["gray_flip_perf_ns"] - choice["timestamp_perf_ns"]) / 1e9, 0.4, places=8)
                self.assertTrue(gate.stopped)

    def test_null_timeout_allows_fixation_after_an_indefinite_wait(self):
        for sequential, memory in ((True, False), (False, True)):
            with self.subTest(sequential=sequential, memory=memory):
                result, events, metadata, gate, _, _ = self.present(
                    sequential=sequential, memory=memory,
                    timeout=None, accepted_after=1.5,
                )
                self.assertEqual(result, (False, None))
                self.assertFalse(metadata.get("saccade_timeout"))
                self.assertEqual([event["event"] for event in events].count("fixation_accepted"), 2 if sequential else 1)
                self.assertTrue(gate.stopped)

    def test_nonmemory_choice_ends_on_time_even_without_any_fixation(self):
        for timeout in (None, 0.3, 1.0):
            with self.subTest(timeout=timeout):
                result, events, metadata, gate, window, _ = self.present(
                    sequential=False, memory=False, accepted_after=None,
                    choice_time=0.3, timeout=timeout,
                )
                self.assertEqual(result, (False, None))
                self.assertFalse(metadata.get("saccade_timeout"))
                names = [event["event"] for event in events]
                self.assertNotIn("fixation_accepted", names)
                self.assertNotIn("saccade_timeout", names)
                choice = next(event for event in events if event["event"] == "choice_start")
                dots = next(event for event in events if event["event"] == "options_dot")
                self.assertEqual(choice["timestamp_perf_ns"], dots["timestamp_perf_ns"])
                self.assertAlmostEqual((metadata["gray_flip_perf_ns"] - dots["timestamp_perf_ns"]) / 1e9, 0.3, places=8)
                self.assertEqual(window.scenes, [(), ()])
                self.assertTrue(gate.stopped)

    def test_late_first_fixation_does_not_extend_nonmemory_choice(self):
        _, events, metadata, _, window, _ = self.present(
            sequential=False, memory=False, accepted_after=0.25,
            choice_time=0.3, timeout=None,
        )
        choice = next(event for event in events if event["event"] == "choice_start")
        reveal = next(event for event in events if event["event"] == "option_1_on")
        self.assertGreaterEqual(reveal["timestamp_perf_ns"] - choice["timestamp_perf_ns"], 250_000_000)
        self.assertAlmostEqual((metadata["gray_flip_perf_ns"] - choice["timestamp_perf_ns"]) / 1e9, 0.3, places=8)
        self.assertEqual(window.scenes, [(), ((-100, 0),), ()])

    def test_final_refresh_gaze_changes_do_not_delay_choice_clear(self):
        cases = (
            ([(0.29, 1)], [(), ()]),
            ([(0.02, 1), (0.29, None)], [(), ((-100, 0),), ()]),
        )
        for schedule, expected_scenes in cases:
            with self.subTest(schedule=schedule):
                result, events, metadata, _, window, _ = self.present(
                    sequential=False, memory=False, schedule=schedule,
                    choice_time=0.3, timeout=None,
                )
                self.assertEqual(result, (False, None))
                choice = next(event for event in events if event["event"] == "choice_start")
                self.assertAlmostEqual((metadata["gray_flip_perf_ns"] - choice["timestamp_perf_ns"]) / 1e9, 0.3, places=8)
                self.assertEqual(window.scenes, expected_scenes)

    def test_timeout_inside_final_refresh_is_not_mistaken_for_normal_choice_end(self):
        result, events, metadata, _, window, _ = self.present(
            sequential=False, memory=False, accepted_after=None,
            choice_time=0.1, timeout=0.099,
        )
        self.assertEqual(result, (False, None))
        self.assertTrue(metadata["saccade_timeout"])
        dots = next(event for event in events if event["event"] == "options_dot")
        timeout = next(event for event in events if event["event"] == "saccade_timeout")
        self.assertAlmostEqual((timeout["timestamp_perf_ns"] - dots["timestamp_perf_ns"]) / 1e9, 0.099, delta=0.002)
        self.assertEqual(window.scenes, [(), ()])

    def test_expired_choice_wins_over_equal_or_later_timeout_after_a_delayed_poll(self):
        for timeout in (0.3, 0.5):
            with self.subTest(timeout=timeout):
                delayed = []

                def delay_once(window, *, deadline_perf_s=None):
                    if not delayed and window.clock.ns - window.flips[0] >= 50_000_000:
                        window.clock.ns += 1_000_000_000
                        delayed.append(True)

                result, events, metadata, _, _, _ = self.present(
                    sequential=False, memory=False, accepted_after=None,
                    choice_time=0.3, timeout=timeout, sync_service=delay_once,
                )
                self.assertTrue(delayed)
                self.assertEqual(result, (False, None))
                self.assertFalse(metadata.get("saccade_timeout"))
                self.assertNotIn("saccade_timeout", [event["event"] for event in events])

    def test_nonmemory_dot_choice_is_rejected_until_an_option_has_flipped(self):
        for click_after in (0, 0.05):
            with self.subTest(click_after=click_after):
                result, events, metadata, _, window, _ = self.present(
                    sequential=False, memory=False, accepted_after=None,
                    choice_time=0.3, timeout=None, click_after=click_after,
                )
                self.assertEqual(result, (False, None))
                self.assertNotIn("fixation_accepted", [event["event"] for event in events])
                self.assertNotIn("option_touch", [event["event"] for event in events])
                self.assertEqual(window.scenes, [(), ()])
                self.assertAlmostEqual((metadata["gray_flip_perf_ns"] - window.flips[0]) / 1e9, 0.3, places=8)

    def test_held_early_touch_does_not_become_a_choice_after_reveal(self):
        result, events, _, _, _, _ = self.present(
            sequential=False, memory=False, accepted_after=0.12,
            choice_time=0.3, timeout=None, click_after=0.05,
        )
        self.assertEqual(result, (False, None))
        self.assertIn("option_1_on", [e["event"] for e in events])
        self.assertNotIn("option_touch", [e["event"] for e in events])

    def test_fresh_touch_after_reveal_succeeds_after_an_early_rejected_touch(self):
        result, events, _, _, _, _ = self.present(
            sequential=False, memory=False, accepted_after=0.12,
            choice_time=0.3, timeout=None, click_intervals=[(0.05, 0.07), (0.20, 0.22)],
        )
        self.assertEqual(result[1]["chosen_index"], 1)
        self.assertAlmostEqual(result[1]["reaction_time_s"], 0.2, delta=0.002)
        self.assertEqual([e["event"] for e in events].count("option_touch"), 1)

    def test_buffered_pre_reveal_touch_is_rejected_when_polled_after_reveal(self):
        result, events, _, _, _, _ = self.present(
            sequential=False, memory=False, accepted_after=0.02,
            choice_time=0.3, timeout=None, buffered_touch=(0.12, 0.09),
        )
        self.assertEqual(result, (False, None))
        self.assertIn("option_1_on", [e["event"] for e in events])
        self.assertNotIn("option_touch", [e["event"] for e in events])

    def test_fixation_acceptance_alone_does_not_unlock_choices_before_actual_reveal(self):
        result, events, _, _, _, _ = self.present(
            sequential=False, memory=False, accepted_after=0.02,
            isi=0.2, choice_time=0.3, timeout=None, click_intervals=[(0.05, 0.07)],
        )
        self.assertEqual(result, (False, None))
        self.assertIn("fixation_accepted", [e["event"] for e in events])
        self.assertIn("option_1_on", [e["event"] for e in events])
        self.assertNotIn("option_touch", [e["event"] for e in events])

    def test_any_option_is_selectable_after_one_reveal_and_status_persists_after_hiding(self):
        result, events, _, _, window, preview = self.present(
            sequential=False, memory=False, schedule=[(0.02, 1), (0.14, None)],
            choice_time=0.3, timeout=None, click_after=0.2, click_index=2,
        )
        self.assertEqual(result[1]["chosen_index"], 2)
        self.assertNotIn("option_2_on", [e["event"] for e in events])
        scenes = [c.kwargs for c in preview.show_static_scene.call_args_list]
        option_scenes = [s for s in scenes if "overlay_dots" in s]
        self.assertTrue(option_scenes)
        for scene in option_scenes:
            self.assertEqual(len(scene["images"]), 2)
            self.assertEqual([b["color"] for b in scene["hit_boxes"]], [[60, 180, 75], [220, 60, 60]])
        self.assertEqual(len(option_scenes[0]["overlay_dots"]), 2)
        self.assertEqual(option_scenes[0]["status_boxes"], [])
        self.assertEqual([d["pos"] for d in option_scenes[1]["overlay_dots"]], [(100.0, 0.0)])
        self.assertEqual(option_scenes[1]["status_boxes"][0]["color"], (50, 130, 255))
        chosen = option_scenes[-1]
        self.assertEqual(len(chosen["overlay_dots"]), 2)
        self.assertEqual([b["color"] for b in chosen["status_boxes"]], [(50, 130, 255), (180, 80, 255)])
        for status in chosen["status_boxes"]:
            reward = next(b for b in chosen["hit_boxes"] if tuple(b["pos"]) == tuple(status["pos"]))
            self.assertTrue(all(a > b for a, b in zip(status["size"], reward["size"])))
        self.assertEqual(scenes[-1]["images"], [])
        self.assertEqual(window.scenes, [(), ((-100, 0),), (), ()])

    def test_preview_visibility_matches_each_main_flip_in_all_presentation_modes(self):
        for sequential in (False, True):
            for memory in (False, True):
                for gated in (False, True):
                    with self.subTest(sequential=sequential, memory=memory, gated=gated):
                        _, _, _, _, window, preview = self.present(
                            sequential=sequential, memory=memory, gated=gated,
                            accepted_after=0.02, choice_time=0.3,
                        )
                        scenes = [c.kwargs for c in preview.show_static_scene.call_args_list]
                        self.assertEqual(len(scenes), len(window.scenes))
                        revealed = set()
                        for main_images, scene in zip(window.scenes, scenes):
                            if "overlay_dots" not in scene:
                                self.assertEqual(main_images, ())
                                continue
                            identities = {tuple(i["pos"]) for i in scene["images"]}
                            hidden = {tuple(d["pos"]) for d in scene["overlay_dots"]}
                            self.assertEqual(identities - hidden, set(main_images))
                            self.assertEqual(len(scene["hit_boxes"]), 2)
                            revealed.update(main_images)
                            self.assertEqual({tuple(b["pos"]) for b in scene["status_boxes"]}, revealed)

    def test_choices_remain_available_after_reveal_in_all_presentation_modes(self):
        for sequential in (False, True):
            for memory in (False, True):
                for gated in (False, True):
                    with self.subTest(sequential=sequential, memory=memory, gated=gated):
                        result, events, _, _, _, preview = self.present(
                            sequential=sequential, memory=memory, gated=gated,
                            accepted_after=0.02, choice_time=1.0, click_after=0.65,
                        )
                        self.assertEqual(result[1]["chosen_index"], 1)
                        touch = next(e for e in events if e["event"] == "option_touch")
                        first = next(e for e in events if e["event"] in ["option_1_on", "options_on"])
                        self.assertGreaterEqual(touch["timestamp_perf_ns"], first["timestamp_perf_ns"])
                        self.assertTrue(any(
                            b["color"] == (180, 80, 255)
                            for c in preview.show_static_scene.call_args_list
                            for b in c.kwargs.get("status_boxes", [])
                        ))

    def test_on_time_saccade_allows_fixation_criterion_to_finish_after_deadline(self):
        for sequential, memory in ((True, True), (False, False), (False, True)):
            with self.subTest(sequential=sequential, memory=memory):
                result, events, metadata, gate, _, _ = self.present(
                    sequential=sequential, memory=memory, accepted_after=0.2,
                    arrival_after=0.08, timeout=0.1, choice_time=0.4,
                )
                self.assertEqual(result, (False, None))
                self.assertFalse(metadata.get("saccade_timeout"))
                self.assertEqual([event["event"] for event in events].count("fixation_accepted"), 2 if sequential else 1)
                self.assertTrue(gate.stopped)

    def test_each_new_sequential_dot_restarts_the_saccade_deadline(self):
        def reached(gate, deadline_perf_s):
            return len(gate.starts) == 1

        def accepted(gate, *, now_s):
            return 1 if len(gate.starts) == 1 and now_s - gate.onset_s >= 0.02 else None

        with patch.object(Gate, "reached_dot_before", reached), patch.object(Gate, "status", accepted):
            result, events, metadata, gate, _, _ = self.present(timeout=0.15)
        self.assertEqual(result, (False, None))
        self.assertTrue(metadata["saccade_timeout"])
        self.assertEqual(len(gate.starts), 2)
        second_dot = next(event for event in events if event["event"] == "option_2_dot")
        timeout = next(event for event in events if event["event"] == "saccade_timeout")
        self.assertAlmostEqual((timeout["timestamp_perf_ns"] - second_dot["timestamp_perf_ns"]) / 1e9, 0.15, delta=0.002)
        self.assertNotIn("option_2_on", [event["event"] for event in events])

    def test_optional_later_dot_skips_to_choice_without_revealing_or_aborting(self):
        for memory in (False, True):
            for isi, timeout in ((0.1, 0.15), (0.2, 0.05)):
                with self.subTest(memory=memory, isi=isi, timeout=timeout):
                    messages, choice_callback = Mock(), Mock()
                    result, events, metadata, gate, window, preview = self.present(
                        memory=memory, accepted_after=(0.02, None), timeout=timeout, isi=isi,
                        require_saccade_to_all_options=False, msg_logger=messages,
                        on_choice_start=choice_callback, hold_cue=True,
                    )
                    self.assertEqual(result, (False, None))
                    self.assertFalse(metadata.get("saccade_timeout"))
                    names = [event["event"] for event in events]
                    self.assertIn("option_1_on", names)
                    self.assertNotIn("option_2_on", names)
                    self.assertNotIn("saccade_timeout", names)
                    self.assertEqual(names.count("fixation_accepted"), 1)
                    choice_callback.assert_called_once_with()
                    dot = next(event for event in events if event["event"] == "option_2_dot")
                    choice = next(event for event in events if event["event"] == "choice_start")
                    elapsed = (choice["timestamp_perf_ns"] - dot["timestamp_perf_ns"]) / 1e9
                    self.assertGreaterEqual(elapsed, max(isi, timeout) - 1e-9)
                    self.assertLess(elapsed, max(isi, timeout) + 1 / 60 + 1e-9)
                    self.assertFalse(any((100, 0) in scene for scene in window.scenes))
                    self.assertEqual(metadata["main_display_transition_misses"], 0)
                    self.assertTrue(gate.stopped)
                    self.assertEqual(len(gate.starts), 2)
                    self.assertTrue(any("option_skipped trial_num=3 option_index=2" in call.args[1]
                                        for call in messages.log.call_args_list))
                    self.assertTrue(any(
                        (100.0, 0.0) in [tuple(dot["pos"]) for dot in call.kwargs.get("overlay_dots", [])]
                        for call in preview.show_static_scene.call_args_list
                    ))

    def test_optional_skip_continues_to_later_options_and_still_requires_the_first(self):
        for memory in (False, True):
            with self.subTest(memory=memory):
                result, events, metadata, gate, window, _ = self.present(
                    memory=memory, accepted_after=(0.02, None, 0.02), num_options=3,
                    timeout=0.15, require_saccade_to_all_options=False,
                )
                self.assertEqual(result, (False, None))
                self.assertFalse(metadata.get("saccade_timeout"))
                names = [event["event"] for event in events]
                self.assertIn("option_1_on", names)
                self.assertIn("option_3_on", names)
                self.assertNotIn("option_2_on", names)
                self.assertFalse(any((100, 0) in scene for scene in window.scenes))
                self.assertEqual(len(gate.starts), 3)
        _, events, metadata, gate, _, _ = self.present(
            accepted_after=None, timeout=0.05, require_saccade_to_all_options=False,
        )
        self.assertTrue(metadata["saccade_timeout"])
        self.assertNotIn("choice_start", [event["event"] for event in events])
        self.assertEqual(len(gate.starts), 1)

    def test_optional_gaze_wait_is_bounded_even_after_a_brief_arrival(self):
        for second_acceptance, revealed in ((0.04, True), (0.2, False), (None, False)):
            with self.subTest(second_acceptance=second_acceptance):
                _, events, metadata, _, _, _ = self.present(
                    accepted_after=(0.02, second_acceptance), arrival_after=(0.01, 0.01),
                    timeout=0.1, require_saccade_to_all_options=False,
                )
                names = [event["event"] for event in events]
                self.assertEqual("option_2_on" in names, revealed)
                self.assertIn("choice_start", names)
                self.assertFalse(metadata.get("saccade_timeout"))

    def test_skipped_sequential_option_remains_selectable_without_being_revealed(self):
        for memory in (False, True):
            with self.subTest(memory=memory):
                result, events, _, _, window, _ = self.present(
                    memory=memory, accepted_after=(0.02, None), timeout=0.15,
                    choice_time=0.3, require_saccade_to_all_options=False,
                    click_after=0.55, click_index=2,
                )
                self.assertEqual(result[1]["chosen_index"], 2)
                self.assertNotIn("option_2_on", [event["event"] for event in events])
                self.assertFalse(any((100, 0) in scene for scene in window.scenes))

    def test_null_timeout_allows_long_gaze_loss_during_simultaneous_choice(self):
        result, events, metadata, _, window, _ = self.present(
            sequential=False, memory=False, timeout=None, choice_time=0.4,
            schedule=[(0.02, 1), (0.14, None)],
        )
        self.assertEqual(result, (False, None))
        self.assertFalse(metadata.get("saccade_timeout"))
        self.assertEqual(window.scenes, [(), ((-100, 0),), (), ()])
        choice = next(event for event in events if event["event"] == "choice_start")
        self.assertAlmostEqual((metadata["gray_flip_perf_ns"] - choice["timestamp_perf_ns"]) / 1e9, 0.4, places=8)

    def test_steady_simultaneous_choice_leaves_headroom_for_sync_swaps_and_clears_preview(self):
        sync_swaps = []

        def service_sync(window, *, deadline_perf_s=None):
            # A pending sync inversion becomes due after choice has opened.
            # Like SyncDisplay, require one refresh of headroom and preserve
            # both the currently visible scene and any prepared next scene.
            now = window.clock.seconds()
            due = window.flips[0] / 1e9 + 0.2
            if sync_swaps or now < due or not window.scenes[-1]:
                return
            if deadline_perf_s is None or deadline_perf_s - now <= 1 / 60:
                return
            headroom = deadline_perf_s - now
            pending = list(window.pending_images)
            window.pending_images[:] = window.scenes[-1]
            window.flip()
            window.pending_images[:] = pending
            sync_swaps.append((window.clock.ns, headroom))

        result, events, metadata, _, window, preview = self.present(
            sequential=False, memory=False, accepted_after=0.02,
            choice_time=0.3, sync_service=service_sync,
        )
        self.assertEqual(result, (False, None))
        self.assertEqual(len(sync_swaps), 1)
        self.assertGreater(sync_swaps[0][1], 1 / 60)
        choice = next(event for event in events if event["event"] == "choice_start")
        self.assertGreater(sync_swaps[0][0], choice["timestamp_perf_ns"])
        self.assertLess(sync_swaps[0][0], metadata["gray_flip_perf_ns"])
        self.assertAlmostEqual((metadata["gray_flip_perf_ns"] - choice["timestamp_perf_ns"]) / 1e9, 0.3, places=8)
        self.assertEqual(window.scenes, [(), ((-100, 0),), ((-100, 0),), ()])
        final_preview = preview.show_static_scene.call_args.kwargs
        self.assertEqual(final_preview["images"], [])
        self.assertEqual(final_preview["dots"], [])
        self.assertEqual(final_preview["circles"], [])

    def test_presenter_events_are_validated_and_written_by_the_real_event_logger(self):
        cases = (
            ({}, {"fixation_accepted": ("205", "interaction"),
                  "option_1_on": ("1101", "frame_flip"), "option_2_on": ("1102", "frame_flip")}),
            ({"accepted_after": None, "timeout": 0.25},
             {"saccade_timeout": ("206", "interaction")}),
            ({"sequential": False, "memory": False, "choice_time": 0.3,
              "schedule": [(0.02, 1), (0.14, None), (0.20, 2)]},
             {"fixation_accepted": ("205", "interaction"),
              "option_1_off": ("1201", "frame_flip"), "option_2_on": ("1102", "frame_flip"),
              "option_2_off": ("1202", "frame_flip")}),
            ({"sequential": False, "memory": False, "choice_time": 0.3,
              "schedule": [(0.02, 1), (0.14, None), (0.20, 1)]},
             {"option_1_on": ("1101", "frame_flip"), "option_1_off": ("1201", "frame_flip")}),
        )
        for options, expected in cases:
            with self.subTest(options=options), tempfile.TemporaryDirectory() as tmp:
                definitions, patterns = load_task_event_definitions("active_foraging")
                library = EventCodeLibrary(definitions, event_patterns=patterns)
                logger = EventLogger(tmp, TaskClock(origin_perf_ns=10_000_000_000), library, auto_flush=False)
                try:
                    result, _, _, _, _, _ = self.present(event_logger=logger, **options)
                    self.assertEqual(result, (False, None))
                finally:
                    logger.close()
                with logger.path.open() as handle:
                    rows = list(csv.DictReader(handle, delimiter="\t"))
                by_name = {row["event"]: row for row in rows}
                for name, (code, kind) in expected.items():
                    self.assertEqual(by_name[name]["event_code"], code)
                    self.assertEqual(by_name[name]["event_type"], kind)
                    self.assertEqual(by_name[name]["trial_num"], "3")
                    self.assertGreater(int(by_name[name]["task_time_ns"]), 0)
                used_path = library.export_used(Path(tmp) / "event_code_library.json")
                used = json.loads(used_path.read_text())
                for row in rows:
                    self.assertEqual(used[row["event_code"]]["event"], row["event"])
                if options.get("schedule", [])[-1:] == [(0.20, 1)]:
                    for name in ("option_1_on", "option_1_off"):
                        repeated = [row for row in rows if row["event"] == name]
                        self.assertEqual(len(repeated), 2)
                        self.assertLess(int(repeated[0]["task_time_ns"]), int(repeated[1]["task_time_ns"]))

    def test_disabled_gate_preserves_scheduled_dot_duration(self):
        _, events, _, _, _, preview = self.present(gated=False)
        self.assertNotIn("fixation_accepted", [event["event"] for event in events])
        for index in (1, 2):
            dot = next(event for event in events if event["event"] == f"option_{index}_dot")
            stimulus = next(event for event in events if event["event"] == f"option_{index}_on")
            self.assertAlmostEqual((stimulus["timestamp_perf_ns"] - dot["timestamp_perf_ns"]) / 1e9, 0.1, places=8)
        self.assertFalse(any(call.kwargs.get("circles") for call in preview.show_static_scene.call_args_list))


if __name__ == "__main__":
    unittest.main()
