"""Pure trial generation and scoring for the match2cue task."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional, Sequence

from bin.afc_stimuli import StimulusKey


@dataclass(frozen=True)
class Match2CueTrial:
    cue: StimulusKey
    options: tuple[StimulusKey, ...]
    reward_draw: float

    @property
    def matching_count(self) -> int:
        return sum(option == self.cue for option in self.options)


@dataclass(frozen=True)
class Match2CueOutcome:
    correct: Optional[bool]
    matching_count: int
    reward_probability: float
    reward_delivered: bool


def generate_match2cue_trial(
    stimuli: Sequence[StimulusKey],
    num_afc: int,
    *,
    rng: Optional[random.Random] = None,
) -> Match2CueTrial:
    """Generate a trial with one guaranteed cue match and sampled distractors.

    Distractors are independent draws with replacement from the complete
    stimulus space. They may therefore duplicate the cue or one another.
    """
    if not stimuli:
        raise ValueError("The match2cue stimulus space must not be empty")
    if int(num_afc) < 1:
        raise ValueError("num_afc must be at least 1")
    chooser = rng or random.Random()
    normalized = tuple(
        (int(item[0]), None if item[1] is None else int(item[1]))
        for item in stimuli
    )
    cue = chooser.choice(normalized)
    options = [cue]
    options.extend(chooser.choice(normalized) for _ in range(int(num_afc) - 1))
    chooser.shuffle(options)
    return Match2CueTrial(
        cue=cue,
        options=tuple(options),
        reward_draw=chooser.random(),
    )


def score_match2cue_choice(
    trial: Match2CueTrial,
    chosen_index_1based: Optional[int],
) -> Match2CueOutcome:
    """Score a selection and apply the duplicate-match reward probability."""
    if chosen_index_1based is None:
        return Match2CueOutcome(
            correct=None,
            matching_count=trial.matching_count,
            reward_probability=0.0,
            reward_delivered=False,
        )
    chosen_index = int(chosen_index_1based)
    if not 1 <= chosen_index <= len(trial.options):
        raise ValueError(
            f"chosen_index_1based must be between 1 and {len(trial.options)}"
        )
    correct = trial.options[chosen_index - 1] == trial.cue
    reward_probability = (1.0 / float(trial.matching_count)) if correct else 0.0
    return Match2CueOutcome(
        correct=correct,
        matching_count=trial.matching_count,
        reward_probability=reward_probability,
        reward_delivered=bool(correct and trial.reward_draw < reward_probability),
    )
