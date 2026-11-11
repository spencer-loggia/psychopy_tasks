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
from bin.config_snapshot import build_task_config_snapshot
from bin.screen import ScreenGeometry, oriented_size, rotate_centered_point, software_stimulus_rotation
from bin.pump_control import PumpPulseControl
from bin.eye_tracking import EyeCalibration, EyeTrackerState
from bin.eye_tracking_config import EyeTrackingSettings
from tests.test_active_foraging_timing import FakeClock


def load_task():
    psychopy = types.ModuleType("psychopy")
    psychopy.core = types.SimpleNamespace(wait=Mock())
    psychopy.event = types.SimpleNamespace(getKeys=lambda **kwargs: [])
    psychopy.logging = types.SimpleNamespace(console=Mock(), CRITICAL=50)
    utils = types.ModuleType("bin.utils")
    for name in ("set_debug", "load_color_palette", "load_shape_definitions",
                 "split_background_from_palette", "setup_task_window", "close_task_window",
                 "verify_task_window_vblank",
                 "resolve_frame_rate", "make_fixation_cross", "make_bg_rect",
                 "configure_sync_bar", "service_sync_bar",
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
    def test_choice_tone_config_defaults_to_false_and_reaches_task(self):
        path = Path(__file__).resolve().parents[1] / "config_files/csc2/af_classic_eyetrack_20261106.json"
        for configured in (None, False, True):
            with self.subTest(configured=configured):
                task = load_task()
                cfg = json.loads(path.read_text())
                cfg["screens"] = {"main": 0, "experimenter": 1}
                if configured is not None:
                    cfg["choice_period_idicator_tone"] = configured
                with patch.object(sys, "argv", ["active_foraging", "--config", str(path)]), patch.object(
                    task, "load_config", return_value=cfg,
                ), patch.object(task, "load_eye_tracking_settings", return_value=None), patch.object(
                    task, "run_task", return_value="done",
                ) as run:
                    task.main()
                self.assertIs(run.call_args.kwargs["choice_period_idicator_tone"], configured is True)

    def test_choice_tone_callback_and_timeout_handoff(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._exercise_task(Path(tmp), True, True, tone_enabled=True)

    def test_placement_distribution_defaults_and_cli_override_reach_task(self):
        path = Path(__file__).resolve().parents[1] / "config_files/csc2/af_classic_eyetrack_20261106.json"
        for configured, cli, expected in (
            (None, [], "circle"), ("uniform", [], "uniform"),
            ("uniform", ["--placement_distribution", "circle"], "circle"),
        ):
            with self.subTest(configured=configured, cli=cli):
                task = load_task()
                cfg = json.loads(path.read_text())
                cfg["screens"] = {"main": 0, "experimenter": 1}
                cfg.pop("placement_distribution", None)
                if configured is not None:
                    cfg["placement_distribution"] = configured
                with patch.object(sys, "argv", ["active_foraging", "--config", str(path), *cli]), patch.object(
                    task, "load_config", return_value=cfg,
                ), patch.object(task, "load_eye_tracking_settings", return_value=None), patch.object(
                    task, "run_task", return_value="done",
                ) as run:
                    task.main()
                self.assertEqual(run.call_args.kwargs["placement_distribution"], expected)

    def test_pruned_classic_config_runs_without_inactive_duration_or_initiation(self):
        task = load_task()
        path = Path(__file__).resolve().parents[1] / "config_files/csc2/af_classic_eyetrack_20261106.json"
        cfg = json.loads(path.read_text())
        cfg["screens"] = {"main": 0, "experimenter": 1}
        cfg.update(self_initiation=False, placement_distribution="uniform",
                   center_point="inactive", stim_range_radius="inactive", subject="Yuri",
                   eye_tracking={"enabled": False}, daq_address=2, daq={"address": 0})
        snapshot = build_task_config_snapshot(cfg, launch_path="task/active_foraging.py")
        for key in (*task.INITIATION_CONFIG_KEYS, "duration", "center_point", "stim_range_radius"):
            self.assertNotIn(key, snapshot)
        with tempfile.TemporaryDirectory() as temporary:
            saved_path = Path(temporary) / "config.json"
            saved_path.write_text(json.dumps(snapshot), encoding="utf-8")
            with patch.object(sys, "argv", ["active_foraging", "--config", str(saved_path)]), patch.object(
                task, "run_task", return_value="done",
            ) as run:
                task.main()
        self.assertEqual(run.call_args.kwargs["duration"], 0)
        self.assertFalse(run.call_args.kwargs["self_initiation"])
        self.assertIsNone(run.call_args.kwargs["initiation_cue_center_position"])
        self.assertIsNone(run.call_args.kwargs["center_point"])
        self.assertIsNone(run.call_args.kwargs["stim_range_radius"])
        self.assertEqual(run.call_args.kwargs["daq_address"], 2)
        self.assertEqual(set(run.call_args.kwargs["freq_space_tsv"]), {"Yuri"})
        self.assertEqual(run.call_args.kwargs["saccade_wait_timeout_s"], 4.0)

    def test_active_duration_remains_required_in_memory_and_sequential_modes(self):
        path = Path(__file__).resolve().parents[1] / "config_files/csc2/af_classic_eyetrack_20261106.json"
        for sequential, memory in ((False, True), (True, False), (True, True)):
            with self.subTest(sequential=sequential, memory=memory):
                task = load_task()
                cfg = json.loads(path.read_text())
                cfg["screens"] = {"main": 0, "experimenter": 1}
                cfg.update(sequential=sequential, is_memory=memory)
                cfg.pop("duration")
                with patch.object(sys, "argv", ["active_foraging", "--config", str(path)]), patch.object(
                    task, "load_config", return_value=cfg,
                ), patch.object(sys, "stderr"), self.assertRaises(SystemExit) as error:
                    task.main()
                self.assertEqual(error.exception.code, 2)

    def test_display_margin_config_and_cli_reach_task(self):
        config_path = Path(__file__).resolve().parents[1] / "config_files/csc2/yuri_headfix_train_noseq_nomem_20260930.json"
        cfg = json.loads(config_path.read_text())
        for configured, cli, expected in ((None, [], 0), (200, [], 200), (200, ["--display_margin", "75.5"], 75.5)):
            with self.subTest(configured=configured, cli=cli):
                task = load_task()
                config = dict(cfg)
                config["screens"] = {"main": 0, "experimenter": 1}
                config["eye_tracking"] = {"enabled": False}
                config["wait_for_saccade_on_dot"] = False
                config.pop("display_margin", None)
                if configured is not None:
                    config["display_margin"] = configured
                with patch.object(sys, "argv", ["active_foraging", "--config", str(config_path), *cli]), patch.object(
                    task, "load_config", return_value=config,
                ), patch.object(task, "run_task", return_value="done") as run:
                    task.main()
                self.assertEqual(run.call_args.kwargs["display_margin"], expected)

    def test_sync_config_and_pin_cli_override_reach_task(self):
        config_path = Path(__file__).resolve().parents[1] / "config_files/csc2/yuri_headfix_train_noseq_nomem_20260930.json"
        config = json.loads(config_path.read_text())
        config["screens"] = {"main": 0, "experimenter": 1}
        config["eye_tracking"] = {"enabled": False}
        config["wait_for_saccade_on_dot"] = False
        config["sync_pin"] = 18
        config["sync_bar"] = {"enabled": True, "height": 24}
        for configured_pin, extra_args, expected_pin in ((18, [], 18), (18, ["--sync_pin", "23"], 23), (None, [], None)):
            config["sync_pin"] = configured_pin
            task = load_task()
            with patch.object(sys, "argv", ["active_foraging", "--config", str(config_path), *extra_args]), patch.object(
                task, "load_config", return_value=config,
            ), patch.object(task, "run_task", return_value="done") as run:
                task.main()
            self.assertEqual(run.call_args.kwargs["sync_pin"], expected_pin)
            expected_sync = {**config["sync_bar"], **({"sync_pin": expected_pin} if extra_args else {})}
            self.assertEqual(run.call_args.kwargs["sync_bar"], expected_sync)

        for nested_pin in (24, None):
            config["sync_bar"]["sync_pin"] = nested_pin
            for extra_args in ([], ["--sync_pin", "23"]):
                task = load_task()
                with patch.object(sys, "argv", ["active_foraging", "--config", str(config_path), *extra_args]), patch.object(
                    task, "load_config", return_value=config,
                ), patch.object(task, "run_task", return_value="done") as run:
                    task.main()
                self.assertEqual(run.call_args.kwargs["sync_bar"]["sync_pin"], 23 if extra_args else nested_pin)
                self.assertEqual(config["sync_bar"]["sync_pin"], nested_pin)

    def test_dot_fixation_config_reaches_task(self):
        task = load_task()
        config_path = Path(__file__).resolve().parents[1] / "config_files/csc2/yuri_headfix_train_noseq_nomem_20260930.json"
        config = json.loads(config_path.read_text())
        config["screens"] = {"main": 0, "experimenter": 1}
        config.update(wait_for_saccade_on_dot=True, saccade_wait_timeout_s=2.5,
                      fixation_criteria_s=0.15, fixation_window_radius=75)
        settings = object()
        for timeout, require_all in ((2.5, True), (2.5, False), (None, True)):
            config["saccade_wait_timeout_s"] = timeout
            config["require_saccade_to_all_options"] = require_all
            with self.subTest(timeout=timeout):
                with patch.object(sys, "argv", ["active_foraging", "--config", str(config_path)]), patch.object(
                    task, "load_config", return_value=config,
                ), patch.object(task, "load_eye_tracking_settings", return_value=settings), patch.object(
                    task, "run_task", return_value="done",
                ) as run:
                    task.main()
                for key in ("wait_for_saccade_on_dot", "saccade_wait_timeout_s", "fixation_criteria_s",
                            "fixation_window_radius", "require_saccade_to_all_options"):
                    self.assertEqual(run.call_args.kwargs[key], config[key])
                self.assertIs(run.call_args.kwargs["eye_tracking_config"], settings)

    def test_invalid_dot_fixation_dependencies_fail_before_starting_resources(self):
        from bin.eye_tracking_config import EyeTrackingSettings
        for isi, settings, message in ((0.0, EyeTrackingSettings(), "isi"), (0.1, None, "eye_tracking")):
            with self.subTest(isi=isi, enabled=settings is not None):
                task = load_task()
                with patch.object(task, "SessionLogBundle") as logs, self.assertRaisesRegex(ValueError, message):
                    task.run_task(
                        colors_tsv="unused", shapes_tsv="unused", n_trials=1, num_afc=1,
                        duration=0.1, choice_time=1, iti=0.2, isi=isi,
                        init_dot_color=None, dot_size=10, dot_color=(128, 128, 128),
                        output_dir="unused", subject="Test", freq_space_tsv={"Test": "unused"},
                        reward_space_tsv={"Test": "unused"},
                        wait_for_saccade_on_dot=True, eye_tracking_config=settings,
                    )
                logs.assert_not_called()
                task.utils.setup_task_window.assert_not_called()

    def test_all_modes_keep_full_iti_after_fixed_reward_or_replacement_timeout(self):
        for sequential, memory in ((False, False), (False, True), (True, False), (True, True)):
            with self.subTest(sequential=sequential, memory=memory), tempfile.TemporaryDirectory() as tmp:
                self._exercise_task(Path(tmp), sequential, memory)

    def test_omitted_inter_pump_interval_defaults_to_pulse_width(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._exercise_task(Path(tmp), False, False, gap=None)

    def test_live_edit_keeps_current_trial_width_and_updates_next_trial_timing_and_log(self):
        for gap in (None, 0.1):
            with self.subTest(gap=gap), tempfile.TemporaryDirectory() as tmp:
                self._exercise_task(Path(tmp), False, False, gap=gap, change_pulse=True)

    def test_fixation_timeout_skips_outcome_interval_and_continues_after_full_iti(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._exercise_task(Path(tmp), True, True, saccade_timeout=True)

    def test_optional_saccade_policy_reaches_presenter(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._exercise_task(Path(tmp), True, True, change_fixation=True,
                                placement_options={"require_saccade_to_all_options": False})

    def test_all_preview_modes_preserve_events_without_synthetic_postchoice_images(self):
        for sequential, memory in ((False, False), (False, True), (True, False), (True, True)):
            for gaze_gated in (False, True):
                with self.subTest(sequential=sequential, memory=memory, gaze_gated=gaze_gated), tempfile.TemporaryDirectory() as tmp:
                    self._exercise_task(
                        Path(tmp), sequential, memory, image_size=(80, 20), output_rotation="right",
                        with_preview=True, change_fixation=gaze_gated,
                        placement_options=dict(fixed_positions=True, stim_range_radius=200),
                    )

    def test_live_fixation_radius_applies_next_trial_and_is_logged_in_pixels(self):
        for sequential, memory in ((True, True), (False, False)):
            with self.subTest(sequential=sequential), tempfile.TemporaryDirectory() as tmp:
                self._exercise_task(Path(tmp), sequential, memory, change_fixation=True)

    def test_trials_reuse_safe_arcs_with_rotated_rectangular_images(self):
        for rotation in ("normal", "right", "inverted", "left"):
            with self.subTest(rotation=rotation), tempfile.TemporaryDirectory() as tmp:
                self._exercise_task(
                    Path(tmp), True, True, image_size=(80, 20), output_rotation=rotation,
                    placement_options=dict(center_point=(-235, 0), stim_range_radius=200, display_margin=50,
                                           initiation_cue_center_position=(-120, -75)),
                )

    def test_uniform_trials_ignore_circle_fields_and_keep_rotated_images_inside_margin(self):
        for rotation in ("normal", "right", "inverted", "left"):
            with self.subTest(rotation=rotation), tempfile.TemporaryDirectory() as tmp:
                self._exercise_task(
                    Path(tmp), False, False, image_size=(80, 20), output_rotation=rotation,
                    placement_options=dict(placement_distribution="uniform", fixed_positions=True,
                                           center_point="unused", stim_range_radius="unused", display_margin=50,
                                           initiation_cue_center_position=(-120, -75)),
                )

    def _exercise_task(self, root, sequential, memory, *, gap=0.1,
                       image_size=(10, 10), output_rotation="normal", placement_options=None,
                       change_pulse=False, saccade_timeout=False, change_fixation=False,
                       with_preview=False, tone_enabled=False):
        task = load_task()
        tone_factory = task.ChoicePeriodTone = Mock()
        control = PumpPulseControl(0.25, fixation_window_radius=150 if change_fixation else None)
        task.PumpPulseControl = Mock(return_value=control)
        widths = [0.25] * 6
        if change_pulse:
            widths[2:] = [0.4] * 4
        reward_durations = [0.2 + 2 * width + (width if gap is None else gap) for width in widths]
        placement_options = placement_options or {}
        clock = FakeClock()
        utils = task.utils
        utils.split_background_from_palette.return_value = (128, 128, 128), {1: (255, 0, 0)}
        utils.load_shape_definitions.return_value = {i: Path("shape.svg") for i in range(4)}
        preview = Mock()
        reward_count_updates = []
        preview.set_reward_counts.side_effect = lambda counts: reward_count_updates.append(dict(counts))
        preview.poll.return_value = False
        preview.consume_manual_reward_request.return_value = False
        task.ExperimenterPreview = Mock(return_value=preview)
        utils.setup_task_window.return_value = (
            Mock(size=(800, 600)), ScreenGeometry(0, 0, 0, 800, 600, rotation=output_rotation),
            ScreenGeometry(1, 800, 0, 800, 600) if with_preview else None,
        )
        utils.make_fixation_cross.return_value = None
        utils.resolve_frame_rate.return_value = 60.0, 1 / 60.0
        utils.make_initiation_cue_stims.return_value = Mock(), Mock()
        image = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
        choices = [2, 3, 4, 1, None, 2]  # levels 1, 2, 3, timeout, no response, next trial
        utils.TrialBufferManager.return_value.get_next_trial.side_effect = [dict(
            type="trial", trial_idx=i, trial_options=[(sid, 1) for sid in range(4)],
            trial_option_meta=[(sid, 0, 0) for sid in range(4)],
            rendered={f"{sid}_1": image for sid in range(4)},
        ) for i in range(1, len(choices) + 1)]
        cue_times, gray_times = [], []

        def present(**kwargs):
            self.assertIs(kwargs["require_saccade_to_all_options"],
                          placement_options.get("require_saccade_to_all_options", True))
            if tone_enabled:
                kwargs["on_choice_start"]()
                tone_factory.return_value.start.assert_called_with(kwargs["trial_num"])
            else:
                self.assertIsNone(kwargs["on_choice_start"])
            sync_calls = utils.service_sync_bar.call_count
            self.assertFalse(kwargs["external_abort_checker"]())
            self.assertEqual(utils.service_sync_bar.call_count, sync_calls)
            if change_pulse and kwargs["trial_num"] == 2:
                control.request(0.4)
            if change_fixation:
                gate = kwargs["dot_fixation_gate"]
                if kwargs["trial_num"] == 2:
                    control.settings.request("fixation_window_radius", 225)
                radius = 150 if kwargs["trial_num"] <= 2 else 225
                self.assertEqual(gate.fixation_window_radius, radius)
                # A gaze 200 pixels from the target changes from outside to
                # inside only on the trial after the edit is confirmed.
                onset = clock.seconds()
                gate.start([(0, 0)], native_size=(800, 600), rotation_degrees=0, onset_perf_s=onset)
                for index in range(12):
                    gate.observe(EyeTrackerState(
                        timestamp_perf_s=onset + index * 0.01, frame_sequence=index,
                        valid=True, x_smooth_px=0.25, y_smooth_px=0,
                    ))
                self.assertEqual(gate.status(now_s=onset + 0.11), None if radius == 150 else 1)
                gate.stop()
            self.assertEqual(kwargs["sequential"], sequential)
            self.assertEqual(kwargs["is_memory"], memory)
            margin = placement_options.get("display_margin", 0)
            stim_w, stim_h = oriented_size(image_size, software_stimulus_rotation(output_rotation))
            for x, y in kwargs["positions"]:
                self.assertLessEqual(abs(x) + stim_w / 2, 400 - margin + 1e-9)
                self.assertLessEqual(abs(y) + stim_h / 2, 300 - margin + 1e-9)
                if "center_point" in placement_options and placement_options.get("placement_distribution", "circle") == "circle":
                    sx, sy = rotate_centered_point((x, y), -software_stimulus_rotation(output_rotation))
                    cx, cy = placement_options["center_point"]
                    self.assertAlmostEqual(np.hypot(sx - cx, sy - cy), placement_options["stim_range_radius"])
            cue_times.append(clock.ns)
            kwargs["logger"].log_frame_flip(trial_num=kwargs["trial_num"], event="onset_cue_on",
                                            timestamp_perf_ns=clock.ns)
            clock.sleep(0.05)
            gray_times.append(clock.ns)
            kwargs["trial_meta"].update(gray_flip_perf_ns=clock.ns,
                                        gray_flip_requested_perf_ns=clock.ns - 100)
            if saccade_timeout and kwargs["trial_num"] == 5:
                kwargs["trial_meta"]["saccade_timeout"] = True
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
        ), patch.object(task, "prepare_afc_placement", wraps=task.prepare_afc_placement) as prepare, patch(
            "bin.task_eye_tracking.TaskEyeTrackingSession",
        ):
            result = task.run_task(
                colors_tsv="colors.tsv", shapes_tsv="shapes.tsv", n_trials=len(choices), num_afc=4,
                duration=0.1 if sequential or memory else 0, choice_time=1, isi=0.1, iti=0.2,
                init_dot_color=None, dot_size=10, dot_color=(128, 128, 128),
                output_dir=str(root / "logs"), subject="Test", image_size=image_size, seed=4,
                freq_space_tsv={"Test": str(frequency)}, reward_space_tsv={"Test": str(rewards)},
                sequential=sequential, is_memory=memory, self_initiation=True, fixation_size=0,
                pump_delay_time=0.2, pump_pulse_time_seconds=0.25, inter_pump_interval=gap,
                timeout_duration_seconds=1.5,
                choice_period_idicator_tone=tone_enabled,
                reward_to_pulse_map={"0": 0, "1": 0, "2": 1, "3": 2},
                reward_to_timeout_map={"0": 1, "1": 0, "2": 0, "3": 0},
                wait_for_saccade_on_dot=change_fixation,
                eye_tracking_config=(
                    EyeTrackingSettings(calibration=EyeCalibration(x_scale=1, y_scale=1))
                    if change_fixation else None
                ),
                **placement_options,
            )
        task.PumpPulseControl.assert_called_once_with(
            0.25, fixation_window_radius=150 if change_fixation else None,
        )
        prepare.assert_called_once()
        self.assertEqual(prepare.call_args.kwargs["placement_distribution"],
                         placement_options.get("placement_distribution", "circle"))
        cue_pos = utils.make_initiation_cue_stims.call_args.kwargs["position"]
        self.assertEqual(
            rotate_centered_point(cue_pos, -software_stimulus_rotation(output_rotation)),
            placement_options.get("initiation_cue_center_position", (0, 0)),
        )
        self.assertEqual(result, "done")
        if tone_enabled:
            tone_factory.assert_called_once()
            self.assertEqual(tone_factory.return_value.start.call_count, len(choices))
            self.assertEqual(tone_factory.return_value.stop.call_count, 2)  # Timeout handoff and cleanup.
        else:
            tone_factory.assert_not_called()
        if with_preview:
            task.ExperimenterPreview.assert_called_once()
            self.assertTrue(task.ExperimenterPreview.call_args.kwargs["preserve_scene_transitions"])
            # Presentation owns every image and status event. The task loop
            # must not fabricate a post-choice array that the subject never saw.
            self.assertTrue(all(not call.kwargs["images"]
                                for call in preview.show_static_scene.call_args_list))
            self.assertEqual(reward_count_updates, [
                {0: 0, 1: 1, 2: 0, 3: 0},
                {0: 0, 1: 1, 2: 1, 3: 0},
                {0: 0, 1: 1, 2: 1, 3: 1},
                {0: 1, 1: 1, 2: 1, 3: 1},
                {0: 1, 1: 2, 2: 1, 3: 1},
            ])
            preview.close.assert_called_once()
        for trial, duration_ns in enumerate(
            round(((0 if saccade_timeout and i == 4 else 1.5 if i == 3 else reward_durations[i]) + 0.2) * 1e9) for i in range(5)
        ):
            self.assertEqual(cue_times[trial + 1] - gray_times[trial], duration_ns)

        event_path = next((root / "logs").rglob("event_log.tsv"))
        if change_fixation:
            with (event_path.parent / "message_log.tsv").open() as handle:
                edits = [dict(json.loads(row["message"].removeprefix("live_setting ")),
                              task_time_ns=int(row["task_time_ns"]))
                         for row in csv.DictReader(handle, delimiter="\t")
                         if row["message"].startswith("live_setting ")]
            radius_edits = [item for item in edits if item["name"] == "fixation_window_radius"]
            self.assertCountEqual([item["phase"] for item in radius_edits], ["initial", "confirmed", "applied"])
            by_phase = {item["phase"]: item for item in radius_edits}
            self.assertEqual([by_phase[phase]["new_value"] for phase in ("initial", "confirmed", "applied")],
                             [150, 225, 225])
            self.assertLess(by_phase["confirmed"]["task_time_ns"], by_phase["applied"]["task_time_ns"])
            self.assertTrue(all(item["units"] == "px" for item in radius_edits))
        with event_path.open() as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        with (event_path.parent / "behavior_log.tsv").open() as handle:
            behavior = list(csv.DictReader(handle, delimiter="\t"))
        self.assertEqual([float(row["pump_pulse_time"]) for row in behavior], widths)
        self.assertNotIn("grey_inter_trial_interval", [row["event"] for row in rows])
        self.assertNotIn("109", [row["event_code"] for row in rows])
        for trial in range(1, len(choices) + 1):
            reward_s = reward_durations[trial - 1]
            events = [row for row in rows if row["trial_num"] == str(trial)]
            for row in events:
                if row["event"] == "pump_on":
                    self.assertEqual(float(row["requested_duration"]), widths[trial - 1])
            timeout = trial == 4
            phases = [row for row in events if row["event"] in (
                "reward_interval", "timeout_interval", "inter_trial_interval")]
            if saccade_timeout and trial == 5:
                self.assertEqual([row["event"] for row in phases], ["inter_trial_interval"])
                self.assertEqual(float(phases[0]["requested_duration"]), 0.2)
                cue_event = next(row for row in events if row["event"] == "onset_cue_on")
                self.assertEqual(int(phases[0]["task_time_ns"]) - int(cue_event["task_time_ns"]), 50_000_000)
                self.assertFalse(any(row["event"] in ("pump_on", "buzzer_on") for row in events))
                continue
            self.assertEqual([row["event"] for row in phases],
                             ["timeout_interval" if timeout else "reward_interval", "inter_trial_interval"])
            self.assertEqual([row["event_code"] for row in phases], ["4" if timeout else "3", "5"])
            self.assertEqual([row["event_type"] for row in phases], ["frame_flip", "lifecycle"])
            self.assertEqual(int(phases[0]["task_time_ns"]) - int(phases[0]["requested_task_time_ns"]), 100)
            cue_event = next(row for row in events if row["event"] == "onset_cue_on")
            self.assertEqual(int(phases[0]["task_time_ns"]) - int(cue_event["task_time_ns"]), 50_000_000)
            self.assertEqual(phases[1]["requested_task_time_ns"], "")
            self.assertAlmostEqual(float(phases[0]["requested_duration"]), 1.5 if timeout else reward_s)
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
