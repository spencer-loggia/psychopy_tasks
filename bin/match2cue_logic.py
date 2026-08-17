"""Pure trial generation and scoring for the match2cue task."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

from bin.afc_stimuli import StimulusKey


VALID_TIE_MODES = ("all", "random")


@dataclass(frozen=True)
class Match2CueTrial:
    cue: StimulusKey
    options: tuple[StimulusKey, ...]
    reward_draw: float
    # Kept separate from reward_draw so match-cue-tap and choice rewards are
    # independent. The default preserves compatibility with callers that
    # construct trials directly and only score the option choice.
    match_cue_reward_draw: float = 0.0

    @property
    def matching_count(self) -> int:
        return sum(option == self.cue for option in self.options)


@dataclass(frozen=True)
class Match2CueOutcome:
    correct: Optional[bool]
    matching_count: int
    reward_probability: float
    reward_delivered: bool


@dataclass(frozen=True)
class Match2CueRewardSettings:
    reward_match_cue_prob: float
    correct_num_pulse: int
    inter_pump_interval: float
    tie_mode: str


def normalize_tie_mode(value: Any) -> str:
    """Return a validated match2cue tie mode."""
    if not isinstance(value, str):
        raise ValueError("tie_mode must be 'all' or 'random'")
    mode = value.strip().lower()
    if mode not in VALID_TIE_MODES:
        raise ValueError("tie_mode must be 'all' or 'random'")
    return mode


def resolve_match2cue_reward_settings(
    *,
    reward_match_cue_prob: Any = 0.0,
    correct_num_pulse: Any = 1,
    inter_pump_interval: Any = None,
    pump_pulse_time_seconds: Any = 0.25,
    tie_mode: Any = "random",
) -> Match2CueRewardSettings:
    """Validate reward options and apply backward-compatible defaults."""
    if isinstance(reward_match_cue_prob, bool):
        raise ValueError("reward_match_cue_prob must be a number from 0 to 1")
    try:
        cue_probability = float(reward_match_cue_prob)
    except (TypeError, ValueError) as exc:
        raise ValueError("reward_match_cue_prob must be a number from 0 to 1") from exc
    if not math.isfinite(cue_probability) or not 0.0 <= cue_probability <= 1.0:
        raise ValueError("reward_match_cue_prob must be a finite number from 0 to 1")

    if isinstance(correct_num_pulse, bool) or not isinstance(
        correct_num_pulse, (int, float)
    ):
        raise ValueError("correct_num_pulse must be a positive integer")
    pulse_count_value = float(correct_num_pulse)
    if (
        not math.isfinite(pulse_count_value)
        or not pulse_count_value.is_integer()
        or pulse_count_value < 1.0
    ):
        raise ValueError("correct_num_pulse must be a positive integer")
    pulse_count = int(pulse_count_value)

    try:
        pump_pulse_time = float(pump_pulse_time_seconds)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "pump_pulse_time_seconds must be a finite non-negative value"
        ) from exc
    if not math.isfinite(pump_pulse_time) or pump_pulse_time < 0.0:
        raise ValueError(
            "pump_pulse_time_seconds must be a finite non-negative value"
        )

    interval_value = (
        pump_pulse_time if inter_pump_interval is None else inter_pump_interval
    )
    try:
        interval = float(interval_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "inter_pump_interval must be a finite non-negative value"
        ) from exc
    if not math.isfinite(interval) or interval < 0.0:
        raise ValueError("inter_pump_interval must be a finite non-negative value")

    return Match2CueRewardSettings(
        reward_match_cue_prob=cue_probability,
        correct_num_pulse=pulse_count,
        inter_pump_interval=interval,
        tie_mode=normalize_tie_mode(tie_mode),
    )


def reward_train_duration(
    num_pulses: int,
    pump_pulse_time_seconds: float,
    inter_pump_interval: float,
) -> float:
    """Return the wall-clock duration of a repeated pump-pulse train."""
    count = int(num_pulses)
    return float(count * pump_pulse_time_seconds) + float(
        max(0, count - 1) * inter_pump_interval
    )


def execute_reward_train(
    num_pulses: int,
    *,
    deliver_pulse: Callable[[int], bool],
    wait_between_pulses: Callable[[int], bool],
) -> bool:
    """Execute N pulses and N-1 gaps; callbacks return true to abort."""
    count = int(num_pulses)
    for pulse_num in range(1, count + 1):
        if deliver_pulse(pulse_num):
            return True
        if pulse_num < count and wait_between_pulses(pulse_num):
            return True
    return False


def should_deliver_match_cue_tap_reward(
    trial: Match2CueTrial,
    reward_probability: float,
) -> bool:
    """Apply the trial's independent draw to the match-cue-tap lottery."""
    return bool(trial.match_cue_reward_draw < float(reward_probability))


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
        # Draw this after the legacy choice-reward draw so seeded cue/options
        # and choice rewards retain their prior random sequence.
        match_cue_reward_draw=chooser.random(),
    )


def score_match2cue_choice(
    trial: Match2CueTrial,
    chosen_index_1based: Optional[int],
    *,
    tie_mode: str = "random",
) -> Match2CueOutcome:
    """Score a selection according to the configured duplicate-match policy."""
    resolved_tie_mode = normalize_tie_mode(tie_mode)
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
    if not correct:
        reward_probability = 0.0
    elif resolved_tie_mode == "all":
        reward_probability = 1.0
    else:
        reward_probability = 1.0 / float(trial.matching_count)
    return Match2CueOutcome(
        correct=correct,
        matching_count=trial.matching_count,
        reward_probability=reward_probability,
        reward_delivered=bool(correct and trial.reward_draw < reward_probability),
    )
