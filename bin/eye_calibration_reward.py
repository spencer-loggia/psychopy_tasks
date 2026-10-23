"""Fixation-gated calibration rewards, independent of screen and pump waits."""

from dataclasses import replace

from .eye_tracking import fraction_position_within_diameter
from .fixation_reward import AutoRewardStatus, FixationAutoReward


class CalibrationAutoReward(FixationAutoReward):
    """Collect UART observations in its worker; decide/deliver on the main thread.

    Both rolling windows reset at reward onset. Samples collected during pump
    pulses count toward the next full interval.
    """

    def __init__(
        self, *, reward_interval, calibration, fixation, screen_size,
        fix_diameter, max_sample_age_s,
    ):
        super().__init__(
            reward_interval=reward_interval, calibration=calibration,
            max_sample_age_s=max_sample_age_s,
        )
        self._fixation = tuple(fixation)
        self._screen_size = tuple(screen_size)
        self._fix_diameter = fix_diameter

    def set_mapping(self, calibration, fixation, *, now_s):
        with self._lock:
            # Never expose half-applied gain/offset edits to the UART callback.
            self._calibration = replace(calibration)
            self._fixation = tuple(fixation)
            self._reset(now_s)

    def _in_window(self, position):
        return fraction_position_within_diameter(
            position, self._fixation, diameter_fraction=self._fix_diameter,
            screen_size=self._screen_size,
        )
