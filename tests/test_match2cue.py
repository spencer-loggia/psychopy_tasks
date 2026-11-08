from contextlib import ExitStack
import importlib.util
import io
import json
import random
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image

import bin as bin_package

from bin.afc_stimuli import load_afc_stimulus_space, render_afc_stimulus
from bin.logger import EventCodeLibrary, load_task_event_definitions
from bin.match2cue_logic import (
    Match2CueTrial,
    execute_reward_train,
    generate_match2cue_trial,
    resolve_match2cue_reward_settings,
    reward_train_duration,
    score_match2cue_choice,
    should_deliver_match_cue_tap_reward,
)


def load_task_without_display():
    """Import the entry point without importing PsychoPy or opening resources."""
    psychopy = types.ModuleType("psychopy")
    psychopy.event = types.SimpleNamespace(getKeys=lambda **kwargs: [])
    psychopy.logging = types.SimpleNamespace(console=Mock(), CRITICAL=50)
    utils = types.ModuleType("bin.utils")
    for name in (
        "set_debug", "setup_task_window", "close_task_window", "resolve_frame_rate",
        "make_fixation_cross", "make_bg_rect", "make_initiation_cue_stims",
        "configure_sync_bar", "service_sync_bar", "TrialBufferManager",
    ):
        setattr(utils, name, Mock())
    path = Path(__file__).resolve().parents[1] / "task/match2cue.py"
    spec = importlib.util.spec_from_file_location("_match2cue_config_under_test", path)
    task = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"psychopy": psychopy, "bin.utils": utils}), patch.object(
        bin_package, "utils", utils, create=True,
    ):
        spec.loader.exec_module(task)
    return task


class Match2CueConfigurationTests(unittest.TestCase):
    def config(self, **overrides):
        return {
            "config_name": "pruned_match", "subject": "Yuri", "n": 1,
            "colors_tsv": "colors.tsv", "shapes_tsv": "shapes.tsv",
            "n_colors": 0, "n_shapes": 2, "image_size": [20, 20],
            "match_cue_duration": 0.2, "delay_time": 0.0,
            "screens": {"main": 0, "experimenter": ""},
            **overrides,
        }

    def test_cli_accepts_pruned_snapshot_without_disabled_initiation_settings(self):
        task = load_task_without_display()
        cfg = self.config(self_initiation=False, placement_distribution="uniform", display_margin=25)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(cfg), encoding="utf-8")
            with patch.object(sys, "argv", ["match2cue", "--config", str(path)]), patch.object(
                task, "run_task", return_value="done",
            ) as run:
                task.main()
        self.assertEqual(run.call_args.args[0], cfg)
        self.assertEqual(run.call_args.kwargs["screen_config"], {"main": 0, "experimenter": None})

    def test_cli_default_self_initiation_still_requires_initiation_settings(self):
        task = load_task_without_display()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(self.config()), encoding="utf-8")
            with patch.object(sys, "argv", ["match2cue", "--config", str(path)]), patch.object(
                task, "run_task",
            ) as run, patch.object(sys, "stderr", new_callable=io.StringIO) as stderr:
                with self.assertRaises(SystemExit) as raised:
                    task.main()
        self.assertEqual(raised.exception.code, 1)
        self.assertIn("initiation_cue_center_position", stderr.getvalue())
        run.assert_not_called()

    def test_disabled_initiation_ignores_inactive_invalid_fields_before_resource_startup(self):
        invalid = {
            "initiation_cue_center_position": "not coordinates",
            "initiation_cue_style": "not a style", "hold_before_choice": "not a bool",
            "hold_cue_to_init_time_s": -1,
        }
        for fields in ({}, invalid):
            with self.subTest(fields=fields):
                task = load_task_without_display()
                with patch.object(task, "load_afc_stimulus_space", side_effect=RuntimeError("resource boundary")) as load, patch.object(
                    task, "SessionLogBundle",
                ) as logs:
                    with self.assertRaisesRegex(RuntimeError, "resource boundary"):
                        task.run_task(self.config(self_initiation=False, **fields), screen_config={})
                load.assert_called_once()
                logs.assert_not_called()
                task.utils.setup_task_window.assert_not_called()

    def test_run_task_default_self_initiation_validates_before_resource_startup(self):
        task = load_task_without_display()
        with patch.object(task, "load_afc_stimulus_space") as load:
            with self.assertRaisesRegex(KeyError, "initiation_cue_center_position"):
                task.run_task(self.config(), screen_config={})
        load.assert_not_called()

    def test_uniform_geometry_routes_margin_and_ignores_inactive_circle_values(self):
        task = load_task_without_display()
        cfg = self.config(
            self_initiation=False, placement_distribution=" UNIFORM ", display_margin=37.5,
            center_point="invalid inactive coordinates", stim_range_radius="invalid inactive radius",
            fixed_positions=True,
        )
        space = types.SimpleNamespace(
            bg=(128, 128, 128), stimuli=((1, None), (2, None)), shapes={}, colors={},
        )
        trial = Match2CueTrial(cue=(1, None), options=((1, None), (2, None)), reward_draw=0)
        preloaded = {(1, None): Image.new("RGB", (20, 20))}
        task.utils.setup_task_window.return_value = (
            types.SimpleNamespace(size=(800, 600)), types.SimpleNamespace(rotation="normal"), None,
        )
        task.utils.resolve_frame_rate.return_value = (60, 1 / 60)
        task.utils.make_fixation_cross.return_value = None
        task.utils.TrialBufferManager.return_value.get_next_trial.return_value = {"trial_idx": 1}
        with ExitStack() as stack:
            stack.enter_context(patch.object(task, "load_afc_stimulus_space", return_value=space))
            stack.enter_context(patch.object(task, "SessionLogBundle", return_value=Mock()))
            stack.enter_context(patch.object(task, "describe_screen", return_value="test screen"))
            stack.enter_context(patch.object(task, "resolve_scene_size", return_value=(800, 600)))
            stack.enter_context(patch.object(task, "_decode_trial_payload", return_value=(trial, preloaded)))
            placement = stack.enter_context(patch.object(
                task, "compute_afc_positions", side_effect=RuntimeError("placement boundary"),
            ))
            with self.assertRaisesRegex(RuntimeError, "placement boundary"):
                task.run_task(cfg, screen_config={})
        self.assertEqual(placement.call_args.args[2:4], (None, None))
        self.assertEqual(placement.call_args.kwargs["placement_distribution"], "uniform")
        self.assertEqual(placement.call_args.kwargs["display_margin"], 37.5)
        task.utils.make_initiation_cue_stims.assert_not_called()


class Match2CueLogicTests(unittest.TestCase):
    def test_trial_always_contains_a_cue_match(self):
        stimuli = [(1, None), (2, None), (3, None)]

        for seed in range(20):
            trial = generate_match2cue_trial(stimuli, 4, rng=random.Random(seed))
            self.assertEqual(len(trial.options), 4)
            self.assertIn(trial.cue, trial.options)
            self.assertEqual(
                trial.matching_count,
                sum(option == trial.cue for option in trial.options),
            )

    def test_distractors_are_sampled_with_replacement(self):
        trial = generate_match2cue_trial([(7, None)], 3, rng=random.Random(1))

        self.assertEqual(trial.options, ((7, None), (7, None), (7, None)))
        self.assertEqual(trial.matching_count, 3)
        self.assertAlmostEqual(
            score_match2cue_choice(trial, 1).reward_probability,
            1.0 / 3.0,
        )

    def test_trial_requires_at_least_one_option(self):
        with self.assertRaisesRegex(ValueError, "at least 1"):
            generate_match2cue_trial([(7, None)], 0)

    def test_match_cue_reward_draw_does_not_shift_legacy_seeded_trial_values(self):
        trial = generate_match2cue_trial(
            [(1, None), (2, None), (3, None)],
            4,
            rng=random.Random(17),
        )

        self.assertEqual(trial.cue, (3, None))
        self.assertEqual(
            trial.options,
            ((2, None), (2, None), (3, None), (2, None)),
        )
        self.assertAlmostEqual(trial.reward_draw, 0.11016204891721182)
        self.assertAlmostEqual(trial.match_cue_reward_draw, 0.026936778790526805)

    def test_matching_choice_uses_inverse_duplicate_reward_probability(self):
        rewarded_trial = Match2CueTrial(
            cue=(1, None),
            options=((1, None), (2, None), (1, None)),
            reward_draw=0.49,
        )
        unrewarded_trial = Match2CueTrial(
            cue=(1, None),
            options=rewarded_trial.options,
            reward_draw=0.50,
        )

        rewarded = score_match2cue_choice(rewarded_trial, 1)
        unrewarded = score_match2cue_choice(unrewarded_trial, 3)

        self.assertTrue(rewarded.correct)
        self.assertEqual(rewarded.reward_probability, 0.5)
        self.assertTrue(rewarded.reward_delivered)
        self.assertTrue(unrewarded.correct)
        self.assertFalse(unrewarded.reward_delivered)

    def test_all_tie_mode_rewards_every_matching_option(self):
        trial = Match2CueTrial(
            cue=(1, None),
            options=((1, None), (2, None), (1, None)),
            reward_draw=0.99,
        )

        first_match = score_match2cue_choice(trial, 1, tie_mode="all")
        second_match = score_match2cue_choice(trial, 3, tie_mode="all")

        for outcome in (first_match, second_match):
            self.assertTrue(outcome.correct)
            self.assertEqual(outcome.reward_probability, 1.0)
            self.assertTrue(outcome.reward_delivered)

    def test_unique_match_is_always_rewarded_in_both_tie_modes(self):
        trial = Match2CueTrial(
            cue=(1, None),
            options=((2, None), (1, None), (3, None)),
            reward_draw=0.99,
        )

        for tie_mode in ("random", "all"):
            with self.subTest(tie_mode=tie_mode):
                outcome = score_match2cue_choice(trial, 2, tie_mode=tie_mode)
                self.assertTrue(outcome.correct)
                self.assertEqual(outcome.reward_probability, 1.0)
                self.assertTrue(outcome.reward_delivered)

    def test_invalid_tie_mode_is_rejected(self):
        trial = Match2CueTrial(
            cue=(1, None),
            options=((1, None),),
            reward_draw=0.0,
        )

        with self.assertRaisesRegex(ValueError, "tie_mode"):
            score_match2cue_choice(trial, 1, tie_mode="first")

    def test_nonmatching_choice_is_incorrect_and_never_rewarded(self):
        trial = Match2CueTrial(
            cue=(1, 4),
            options=((1, 4), (1, 5), (2, 4)),
            reward_draw=0.0,
        )

        outcome = score_match2cue_choice(trial, 2)

        self.assertFalse(outcome.correct)
        self.assertEqual(outcome.reward_probability, 0.0)
        self.assertFalse(outcome.reward_delivered)

    def test_omission_is_neither_correct_nor_rewarded(self):
        trial = Match2CueTrial(
            cue=(1, None),
            options=((1, None), (2, None)),
            reward_draw=0.0,
        )

        outcome = score_match2cue_choice(trial, None)

        self.assertIsNone(outcome.correct)
        self.assertFalse(outcome.reward_delivered)

    def test_match_cue_tap_and_choice_rewards_use_independent_draws(self):
        trial = Match2CueTrial(
            cue=(1, None),
            options=((1, None),),
            reward_draw=0.1,
            match_cue_reward_draw=0.9,
        )

        self.assertTrue(score_match2cue_choice(trial, 1).reward_delivered)
        self.assertFalse(should_deliver_match_cue_tap_reward(trial, 0.7))

    def test_match_cue_tap_reward_probability_boundaries(self):
        trial = Match2CueTrial(
            cue=(1, None),
            options=((1, None),),
            reward_draw=0.5,
            match_cue_reward_draw=0.5,
        )

        self.assertFalse(should_deliver_match_cue_tap_reward(trial, 0.0))
        self.assertTrue(should_deliver_match_cue_tap_reward(trial, 1.0))


class Match2CueRewardSettingsTests(unittest.TestCase):
    def test_defaults_preserve_legacy_choice_reward_behavior(self):
        settings = resolve_match2cue_reward_settings(
            pump_pulse_time_seconds=0.6,
        )

        self.assertEqual(settings.reward_match_cue_prob, 0.0)
        self.assertEqual(settings.correct_num_pulse, 1)
        self.assertEqual(settings.inter_pump_interval, 0.6)
        self.assertEqual(settings.tie_mode, "random")

    def test_explicit_reward_settings_are_normalized(self):
        settings = resolve_match2cue_reward_settings(
            reward_match_cue_prob=0.7,
            correct_num_pulse=2.0,
            inter_pump_interval=0.2,
            pump_pulse_time_seconds=0.6,
            tie_mode=" ALL ",
        )

        self.assertEqual(settings.reward_match_cue_prob, 0.7)
        self.assertEqual(settings.correct_num_pulse, 2)
        self.assertEqual(settings.inter_pump_interval, 0.2)
        self.assertEqual(settings.tie_mode, "all")

    def test_invalid_cue_reward_probabilities_are_rejected(self):
        for value in (-0.1, 1.1, float("nan"), float("inf"), True):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "reward_match_cue_prob"):
                    resolve_match2cue_reward_settings(
                        reward_match_cue_prob=value,
                    )

    def test_invalid_correct_pulse_counts_are_rejected(self):
        for value in (0, -1, 1.5, True, "2"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "correct_num_pulse"):
                    resolve_match2cue_reward_settings(
                        correct_num_pulse=value,
                    )

    def test_invalid_inter_pump_intervals_are_rejected(self):
        for value in (-0.1, float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "inter_pump_interval"):
                    resolve_match2cue_reward_settings(
                        inter_pump_interval=value,
                    )

    def test_reward_train_duration_includes_only_between_pulse_gaps(self):
        self.assertAlmostEqual(reward_train_duration(2, 0.6, 0.2), 1.4)
        self.assertAlmostEqual(reward_train_duration(1, 0.6, 0.2), 0.6)

    def test_reward_train_executes_two_pulses_with_one_between_pulse_wait(self):
        events = []

        aborted = execute_reward_train(
            2,
            deliver_pulse=lambda pulse_num: bool(
                events.append(("pulse", pulse_num))
            ),
            wait_between_pulses=lambda pulse_num: bool(
                events.append(("wait", pulse_num))
            ),
        )

        self.assertFalse(aborted)
        self.assertEqual(events, [("pulse", 1), ("wait", 1), ("pulse", 2)])

    def test_reward_train_stops_when_between_pulse_wait_aborts(self):
        events = []

        aborted = execute_reward_train(
            3,
            deliver_pulse=lambda pulse_num: bool(
                events.append(("pulse", pulse_num))
            ),
            wait_between_pulses=lambda pulse_num: (
                events.append(("wait", pulse_num)) or True
            ),
        )

        self.assertTrue(aborted)
        self.assertEqual(events, [("pulse", 1), ("wait", 1)])


class Match2CueStimulusTests(unittest.TestCase):
    def _write_native_space(self, root: Path, *, extra_color: bool = False):
        svg_path = root / "shape.svg"
        svg_path.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10">'
            '<rect width="10" height="10" fill="#123456"/></svg>',
            encoding="utf-8",
        )
        shapes_path = root / "shapes.tsv"
        shapes_path.write_text(f"id\tpath\n1\t{svg_path}\n", encoding="utf-8")
        colors_path = root / "colors.tsv"
        extra = "1\t1\t2\t3\n" if extra_color else ""
        colors_path.write_text(
            "id\tr\tg\tb\n0\t168\t169\t166\n" + extra,
            encoding="utf-8",
        )
        return colors_path, shapes_path, svg_path

    def test_zero_colors_builds_shape_only_stimulus_space(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            colors_path, shapes_path, _ = self._write_native_space(Path(tmpdir))

            space = load_afc_stimulus_space(
                colors_tsv=colors_path,
                shapes_tsv=shapes_path,
                n_colors=0,
                n_shapes=1,
                n_lum_levels=0,
            )

        self.assertTrue(space.native_svg_mode)
        self.assertEqual(space.bg, (168, 169, 166))
        self.assertEqual(space.stimuli, ((1, None),))
        self.assertEqual(space.metadata[(1, None)], (0, None, None))

    def test_zero_colors_rejects_non_background_palette_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            colors_path, shapes_path, _ = self._write_native_space(
                Path(tmpdir),
                extra_color=True,
            )

            with self.assertRaisesRegex(ValueError, "only the background row"):
                load_afc_stimulus_space(
                    colors_tsv=colors_path,
                    shapes_tsv=shapes_path,
                    n_colors=0,
                    n_shapes=1,
                    n_lum_levels=0,
                )

    def test_shape_only_rendering_preserves_authored_svg(self):
        sentinel = Image.new("RGBA", (10, 10))
        native = Mock(return_value=sentinel)
        recolored = Mock()
        fake_utils = types.SimpleNamespace(
            rasterize_svg=native,
            rasterize_svg_with_color=recolored,
        )
        with patch.dict(sys.modules, {"bin.utils": fake_utils}), patch.object(
            bin_package, "utils", fake_utils, create=True,
        ):
            rendered = render_afc_stimulus(
                (1, None),
                shapes={1: Path("shape.svg")},
                colors={},
                image_size=(10, 10),
                bg=(168, 169, 166),
            )

        self.assertIs(rendered, sentinel)
        native.assert_called_once()
        recolored.assert_not_called()


class Match2CueEventTests(unittest.TestCase):
    def test_task_event_set_and_sequential_templates_are_registered(self):
        definitions, patterns = load_task_event_definitions("match2cue")
        library = EventCodeLibrary(definitions, event_patterns=patterns)

        self.assertEqual(definitions["match_cue_on"].code, 116)
        self.assertEqual(definitions["match_cue_touch"].code, 202)
        self.assertEqual(definitions["initiation_cue_release"].code, 203)
        self.assertEqual(definitions["delay_start"].code, 114)
        self.assertEqual(definitions["inter_stimulus_interval"].code, 117)
        self.assertEqual(library.ensure("option_3_dot", "frame_flip").code, 1003)
        self.assertEqual(library.ensure("option_3_on", "frame_flip").code, 1103)


if __name__ == "__main__":
    unittest.main()
