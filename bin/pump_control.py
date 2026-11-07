"""Compatibility facade for the shared runtime settings and preview editor."""
from .live_settings import reward_settings
from .live_settings_ui import SettingsPanel


class PumpPulseControl:
    def __init__(self, duration_s, *, allow_zero=True, maximum_s=None, boundary="trial",
                 reward_interval=None, fixation_radius=None, fixation_window_radius=None,
                 interval_label="Reward interval"):
        self.settings = reward_settings(
            duration_s, allow_zero=allow_zero, maximum_s=maximum_s, boundary=boundary,
            reward_interval=reward_interval, fixation_radius=fixation_radius, interval_label=interval_label,
            fixation_window_radius=fixation_window_radius,
        )

    def request(self, value):
        self.settings.request("pump_pulse_time", value)

    def snapshot(self):
        return self.settings.snapshot()["pump_pulse_time"]

    def apply_pending(self, logger=None):
        return self.settings.apply_pending(logger)["pump_pulse_time"]


class PumpPulseEditor(SettingsPanel):
    def __init__(self, win, control, *, visual):
        super().__init__(win, control.settings, visual=visual)
        self.control = control
