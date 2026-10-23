"""Run the real trial loop and logs with a simulated clock and display."""

import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import bin
import numpy as np
from bin.screen import ScreenGeometry
from tests.test_active_foraging_timing import FakeClock


def load_task():
    psychopy = types.ModuleType("psychopy")
    psychopy.core = types.SimpleNamespace(wait=Mock())
    psychopy.event = types.SimpleNamespace(getKeys=lambda **kwargs: [])
    psychopy.logging = types.SimpleNamespace(console=Mock(), CRITICAL=50)
    utils = types.ModuleType("bin.utils")
    for name in ("set_debug", "load_color_palette", "load_shape_definitions",
                 "split_background_from_palette", "setup_task_window", "close_task_window",
                 "resolve_frame_rate", "make_fixation_cross", "make_bg_rect",
                 "make_initiation_cue_stims", "TrialBufferManager",
                 "present_trial_with_persistent_dots"):
        setattr(utils, name, Mock())
    path = Path(__file__).resolve().parents[1] / "task/active_foraging.py"
    spec = importlib.util.spec_from_file_location("_active_foraging_under_test", path)
    task = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"psychopy": psychopy, "bin.utils": utils}), patch.object(
        bin, "utils", utils, create=True,
    ):
        spec.loader.exec_module(task)
    return task


class ActiveForagingTaskTests(unittest.TestCase):
    def test_all_modes_keep_full_iti_after_fixed_reward_or_replacement_timeout(self):
        for sequential, memory in ((False, False), (False, True), (True, False), (True, True)):
            with self.subTest(sequential=sequential, memory=memory), tempfile.TemporaryDirectory() as tmp:
                self._exercise_task(Path(tmp), sequential, memory)

    def test_omitted_inter_pump_interval_defaults_to_pulse_width(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._exercise_task(Path(tmp), False, False, gap=None, reward_s=0.95)

    def _exercise_task(self, root, sequential, memory, *, gap=0.1, reward_s=0.8):
        task = load_task()
        clock = FakeClock()
        utils = task.utils
        utils.split_background_from_palette.return_value = (128, 128, 128), {1: (255, 0, 0)}
        utils.load_shape_definitions.return_value = {i: Path("shape.svg") for i in range(4)}
        utils.setup_task_window.return_value = (
            Mock(size=(800, 600)), ScreenGeometry(0, 0, 0, 800, 600), None,
        )
        utils.resolve_frame_rate.return_value = 60.0, 1 / 60.0
        utils.make_initiation_cue_stims.return_value = Mock(), Mock()
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        choices = [2, 3, 4, 1, None, 2]  # levels 1, 2, 3, timeout, no response, next trial
        utils.TrialBufferManager.return_value.get_next_trial.side_effect = [dict(
            type="trial", trial_idx=i, trial_options=[(sid, 1) for sid in range(4)],
            trial_option_meta=[(sid, 0, 0) for sid in range(4)],
            rendered={f"{sid}_1": image for sid in range(4)},
        ) for i in range(1, len(choices) + 1)]
        cue_times, gray_times = [], []

        def present(**kwargs):
            self.assertEqual(kwargs["sequential"], sequential)
            self.assertEqual(kwargs["is_memory"], memory)
            cue_times.append(clock.ns)
            kwargs["logger"].log_frame_flip(trial_num=kwargs["trial_num"], event="onset_cue_on",
                                            timestamp_perf_ns=clock.ns)
            clock.sleep(0.05)
            gray_times.append(clock.ns)
            kwargs["trial_meta"].update(gray_flip_perf_ns=clock.ns,
                                        gray_flip_requested_perf_ns=clock.ns - 100)
            index = choices[kwargs["trial_num"] - 1]
            return False, None if index is None else {"chosen_index": index}

        utils.present_trial_with_persistent_dots.side_effect = present
        frequency, rewards = root / "frequency.csv", root / "rewards.csv"
        frequency.write_text("frequency\n1\n1\n1\n1\n")
        rewards.write_text("reward\n0\n1\n2\n3\n")
        with patch.object(task, "build_main_and_worker_affinity_plan", return_value={"supported": False}), patch(
            "time.perf_counter", side_effect=clock.seconds,
        ), patch("time.perf_counter_ns", side_effect=lambda: clock.ns), patch(
            "time.sleep", side_effect=clock.sleep,
        ):
            result = task.run_task(
                colors_tsv="colors.tsv", shapes_tsv="shapes.tsv", n_trials=len(choices), num_afc=4,
                duration=0.1 if sequential or memory else 0, choice_time=1, isi=0.1, iti=0.2,
                init_dot_color=None, dot_size=10, dot_color=(128, 128, 128),
                output_dir=str(root / "logs"), subject="Test", image_size=(10, 10),
                freq_space_tsv={"Test": str(frequency)}, reward_space_tsv={"Test": str(rewards)},
                sequential=sequential, is_memory=memory, self_initiation=True, fixation_size=0,
                pump_delay_time=0.2, pump_pulse_time_seconds=0.25, inter_pump_interval=gap,
                timeout_duration_seconds=1.5,
                reward_to_pulse_map={"0": 0, "1": 0, "2": 1, "3": 2},
                reward_to_timeout_map={"0": 1, "1": 0, "2": 0, "3": 0},
            )
        self.assertEqual(result, "done")
        reward_and_iti_ns = round((reward_s + 0.2) * 1_000_000_000)
        for trial, duration_ns in enumerate((reward_and_iti_ns, reward_and_iti_ns, reward_and_iti_ns,
                                            1_700_000_000, reward_and_iti_ns)):
            self.assertEqual(cue_times[trial + 1] - gray_times[trial], duration_ns)

        event_path = next((root / "logs").rglob("event_log.tsv"))
        with event_path.open() as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        self.assertNotIn("grey_inter_trial_interval", [row["event"] for row in rows])
        self.assertNotIn("109", [row["event_code"] for row in rows])
        for trial in range(1, len(choices) + 1):
            events = [row for row in rows if row["trial_num"] == str(trial)]
            timeout = trial == 4
            phases = [row for row in events if row["event"] in (
                "reward_interval", "timeout_interval", "inter_trial_interval")]
            self.assertEqual([row["event"] for row in phases],
                             ["timeout_interval" if timeout else "reward_interval", "inter_trial_interval"])
            self.assertEqual([row["event_code"] for row in phases], ["4" if timeout else "3", "5"])
            self.assertEqual([row["event_type"] for row in phases], ["frame_flip", "lifecycle"])
            self.assertEqual(int(phases[0]["task_time_ns"]) - int(phases[0]["requested_task_time_ns"]), 100)
            cue_event = next(row for row in events if row["event"] == "onset_cue_on")
            self.assertEqual(int(phases[0]["task_time_ns"]) - int(cue_event["task_time_ns"]), 50_000_000)
            self.assertEqual(phases[1]["requested_task_time_ns"], "")
            self.assertEqual(float(phases[0]["requested_duration"]), 1.5 if timeout else reward_s)
            self.assertEqual(float(phases[1]["requested_duration"]), 0.2)
            self.assertEqual(int(phases[1]["task_time_ns"]) - int(phases[0]["task_time_ns"]),
                             1_500_000_000 if timeout else round(reward_s * 1_000_000_000))
            self.assertEqual(len([row for row in events if row["event"] == "pump_on"]),
                             {2: 1, 3: 2, 6: 0}.get(trial, 0))
        library = json.loads((event_path.parent / "event_code_library.json").read_text())
        self.assertNotIn("109", library)
        for code, name in (("3", "reward_interval"), ("4", "timeout_interval"), ("5", "inter_trial_interval")):
            self.assertEqual(library[code]["event"], name)
            self.assertEqual(library[code]["event_type"], "lifecycle" if code == "5" else "frame_flip")
