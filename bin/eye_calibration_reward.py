"""Fixation-gated calibration rewards, independent of screen and pump waits."""

from dataclasses import replace

from .eye_tracking import fraction_position_within_diameter
from .fixation_reward import (
    AutoRewardStatus, FixationAutoReward, _positive_finite_setting,
)


class CalibrationAutoReward(FixationAutoReward):
    """Collect UART observations in its worker; decide/deliver on the main thread.

    Both rolling windows reset at reward onset. Samples collected during pump
    pulses count toward the next full interval.
    """

    def __init__(
        self, *, reward_interval, calibration, fixation, screen_size,
        fix_diameter=None, fixation_radius=None, max_sample_age_s,
    ):
        super().__init__(
            reward_interval=reward_interval, calibration=calibration,
            max_sample_age_s=max_sample_age_s,
        )
        self._fixation = tuple(fixation)
        self._screen_size = tuple(screen_size)
        if fixation_radius is None:
            fixation_radius = _positive_finite_setting(fix_diameter, "fix_diameter") / 2
        self.fixation_radius = _positive_finite_setting(
            fixation_radius, "fixation_radius",
        )

    def _update_settings(self, settings):
        changed = super()._update_settings(settings)
        fixation_radius = settings.get("fixation_radius")
        if fixation_radius is not None and fixation_radius != self.fixation_radius:
            self.fixation_radius = fixation_radius
            changed = True
        return changed

    def configure(self, *, reward_interval=None, fixation_radius=None, now_s):
        """Set interval/radius atomically, waiting for an in-flight observation.

        Radius is a fraction of the shorter subject-screen dimension. Explicit
        radius takes precedence over legacy diameter in the constructor.
        """
        return self._configure(
            {"reward_interval": reward_interval, "fixation_radius": fixation_radius},
            now_s=now_s, blocking=True,
        )

    def try_configure(self, *, reward_interval=None, fixation_radius=None,
                      now_s, before_update=None):
        """Nonblocking :meth:`configure`; see the base callback/return contract."""
        return self._configure(
            {"reward_interval": reward_interval, "fixation_radius": fixation_radius},
            now_s=now_s, blocking=False, before_update=before_update,
        )

    def set_fixation_radius(self, fixation_radius, *, now_s):
        return self.configure(
            fixation_radius=_positive_finite_setting(fixation_radius, "fixation_radius"),
            now_s=now_s,
        )

    def set_mapping(self, calibration, fixation, *, now_s):
        with self._lock:
            # Never expose half-applied gain/offset edits to the UART callback.
            self._calibration = replace(calibration)
            self._fixation = tuple(fixation)
            self._reset(now_s)

    def _in_window(self, position):
        return fraction_position_within_diameter(
            position, self._fixation, diameter_fraction=2 * self.fixation_radius,
            screen_size=self._screen_size,
        )
