"""Display-independent timing for touch-and-hold training (seconds)."""
from __future__ import annotations

import math


def seconds(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number of seconds")
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return float(value)


class TouchAndHoldState:
    """Reward uninterrupted contact; never deliver missed rewards in a burst.

    A bounded trial starts on a fresh press, including buffered short taps.
    Moving outside the cue counts as release. Re-entering an ongoing trial
    starts a new reward interval. ITI begins at the actual cue-removal flip.
    """

    def __init__(self, *, duration, iti, reward_interval, end_on_release, n=None):
        self.duration = None if duration is None else seconds(duration, "duration", positive=True)
        self.iti = seconds(iti, "iti")
        self.reward_interval = seconds(reward_interval, "reward_interval", positive=True)
        if not isinstance(end_on_release, bool):
            raise ValueError("end_on_release must be a boolean")
        if n is not None and (isinstance(n, bool) or not isinstance(n, int) or n < 1):
            raise ValueError("n must be null or an integer >= 1")
        self.end_on_release = end_on_release
        self.continuous = duration is None and not end_on_release
        self.n = n
        self.phase = "continuous" if self.continuous else "waiting"
        self.trial_num = 0
        self.completed_trials = 0
        self.trial_started_s = None
        self.iti_until_s = None
        self.next_reward_s = None
        self.touching = False
        self.end_reason = None

    @property
    def cue_visible(self):
        return self.phase in {"waiting", "holding", "continuous"}

    def start_iti(self, flip_s):
        self.iti_until_s = flip_s + self.iti

    def update(self, now, *, touching, pressed=False, press_s=None):
        """Return whether a reward is due; caller acknowledges actual delivery."""
        if self.phase == "iti":
            if self.iti_until_s is not None and now >= self.iti_until_s:
                self.phase = "done" if self.n is not None and self.completed_trials >= self.n else "waiting"
            # Do not use a press collected while the cue was hidden.
            return False
        if self.phase == "done":
            return False
        # A second buffered press proves a release happened between polls even
        # if the pointer is already down again (e.g. during a display flip).
        released_between_samples = self.touching and pressed
        if self.phase == "waiting":
            if not pressed:
                return False
            self.trial_num += 1
            self.trial_started_s = now if press_s is None else min(now, press_s)
            self.phase = "holding"
            self.end_reason = None
            # A buffered tap has already released by the time it is sampled.
            self.touching = True
            self.next_reward_s = self.trial_started_s + self.reward_interval

        self.end_reason = None
        if self.phase == "holding":
            if self.end_on_release and (not touching or released_between_samples):
                self.end_reason = "release"
            elif self.duration is not None and now >= self.trial_started_s + self.duration:
                self.end_reason = "duration"
            if self.end_reason is not None:
                self.completed_trials += 1
                self.phase = "iti"
                self.iti_until_s = None
                self.touching = False
                self.next_reward_s = None
                return False

        if not touching:
            self.next_reward_s = None
        elif not self.touching or released_between_samples:
            self.next_reward_s = now + self.reward_interval
        self.touching = bool(touching)
        return self.next_reward_s is not None and now >= self.next_reward_s

    def reward_delivered(self, now):
        self.next_reward_s = now + self.reward_interval
