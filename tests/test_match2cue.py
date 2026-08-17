import random
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image

from bin.afc_stimuli import load_afc_stimulus_space, render_afc_stimulus
from bin.logger import EventCodeLibrary, load_task_event_definitions
from bin.match2cue_logic import (
    Match2CueTrial,
    generate_match2cue_trial,
    resolve_match2cue_reward_settings,
    reward_train_duration,
    score_match2cue_choice,
    should_deliver_cue_tap_reward,
)


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

    def test_cue_tap_and_choice_rewards_use_independent_draws(self):
        trial = Match2CueTrial(
            cue=(1, None),
            options=((1, None),),
            reward_draw=0.1,
            cue_reward_draw=0.9,
        )

        self.assertTrue(score_match2cue_choice(trial, 1).reward_delivered)
        self.assertFalse(should_deliver_cue_tap_reward(trial, 0.7))


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
        for value in (-0.1, 1.1, float("nan"), float("inf")):
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
        with patch.dict(sys.modules, {"bin.utils": fake_utils}):
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
        self.assertEqual(definitions["delay_start"].code, 114)
        self.assertEqual(library.ensure("option_3_dot", "frame_flip").code, 1003)
        self.assertEqual(library.ensure("option_3_on", "frame_flip").code, 1103)


if __name__ == "__main__":
    unittest.main()
