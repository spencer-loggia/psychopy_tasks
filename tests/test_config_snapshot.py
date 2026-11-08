"""Saved configuration pruning must retain executable task semantics."""

import copy
from pathlib import Path
import unittest

from bin.config import resolve_subject_mapped_value
from bin.config_snapshot import build_launch_config_snapshot, build_task_config_snapshot


class ConfigSnapshotTests(unittest.TestCase):
    def snapshot(self, cfg, task="active_foraging"):
        original = copy.deepcopy(cfg)
        result = build_task_config_snapshot(cfg, launch_path=Path("task") / f"{task}.py")
        self.assertEqual(cfg, original, "Pruning must not mutate source/state configuration")
        return result

    def test_uniform_discards_only_circle_placement_knobs(self):
        cfg = {"placement_distribution": "uniform", "center_point": [300, -480],
               "stim_range_radius": 700, "fixed_positions": True,
               "display_margin": 80, "image_size": [100, 150], "seed": 23,
               "extension": {"value": [1, 2]}}
        result = self.snapshot(cfg)
        for key in ("center_point", "stim_range_radius", "fixed_positions"):
            self.assertNotIn(key, result)
        for key in ("display_margin", "image_size", "seed", "extension", "placement_distribution"):
            self.assertEqual(result[key], cfg[key])
        result["extension"]["value"].append(3)
        self.assertEqual(cfg["extension"]["value"], [1, 2])

    def test_circle_and_omitted_distribution_keep_placement(self):
        for distribution in (None, "circle", "new_distribution"):
            cfg = {"center_point": [1, 2], "stim_range_radius": 200, "fixed_positions": False}
            if distribution is not None:
                cfg["placement_distribution"] = distribution
            self.assertEqual(self.snapshot(cfg), cfg)

    def test_match2cue_uniform_uses_same_inactive_circle_rules(self):
        cfg = {"placement_distribution": "uniform", "center_point": [1, 2],
               "stim_range_radius": 200, "fixed_positions": True, "display_margin": 40}
        self.assertEqual(self.snapshot(cfg, "match2cue"), {"placement_distribution": "uniform", "display_margin": 40})

    def test_uniform_pruning_matches_case_and_whitespace_normalization(self):
        for task in ("active_foraging", "match2cue"):
            for distribution in (" Uniform ", "UNIFORM", "uniform"):
                cfg = {"placement_distribution": distribution, "center_point": [1, 2],
                       "stim_range_radius": 200, "fixed_positions": True}
                self.assertEqual(self.snapshot(cfg, task), {"placement_distribution": distribution})
            for invalid in (None, False, 1, ["uniform"], {"uniform": True}, "unknown"):
                cfg = {"placement_distribution": invalid, "center_point": [1, 2], "stim_range_radius": 200}
                self.assertEqual(self.snapshot(cfg, task), cfg)

    def test_gaze_parameters_removed_only_when_gate_disabled(self):
        params = {"fixation_criteria_s": .3, "fixation_window_radius": 175,
                  "saccade_wait_timeout_s": 4, "eye_tracking": {"enabled": True}}
        for enabled in (False, True):
            cfg = {**params, "wait_for_saccade_on_dot": enabled}
            result = self.snapshot(cfg)
            for key in params.keys() - {"eye_tracking"}:
                self.assertEqual(key in result, enabled)
            self.assertEqual(result["eye_tracking"], {"enabled": True})
        self.assertNotIn("fixation_criteria_s", self.snapshot(params))
        self.assertIn("fixation_criteria_s", self.snapshot({**params, "wait_for_saccade_on_dot": "invalid"}))

    def test_disabled_initiation_prunes_cue_and_hold_settings(self):
        params = {"initiation_cue_center_position": [10, 20], "initiation_cue_style": "blob",
                  "hold_before_choice": True, "hold_cue_to_init_time_s": .25}
        for task in ("active_foraging", "match2cue"):
            self.assertEqual(self.snapshot({**params, "self_initiation": False}, task), {"self_initiation": False})
            cfg = {**params, "self_initiation": True, "hold_before_choice": False}
            self.assertEqual(self.snapshot(cfg, task), cfg)
        # The two task entry points have different self-initiation defaults.
        self.assertEqual(self.snapshot(params), {})
        self.assertEqual(self.snapshot(params, "match2cue"), params)

    def test_init_dot_color_is_kept_for_either_initiation_or_isi_dots(self):
        for task in ("active_foraging", "match2cue"):
            for initiation, isi in ((False, 0), (False, .1), (True, 0), (True, .1)):
                cfg = {"self_initiation": initiation, "isi": isi, "init_dot_color": [20, 30, 40],
                       "dot_color": [50, 60, 70], "dot_size": 20}
                result = self.snapshot(cfg, task)
                self.assertEqual("init_dot_color" in result, initiation or isi > 0)
                self.assertEqual(result["dot_color"], cfg["dot_color"])
                self.assertEqual(result["dot_size"], cfg["dot_size"])
        # AF's omitted ISI defaults to .5 s; match2cue's defaults to zero.
        self.assertIn("init_dot_color", self.snapshot({"self_initiation": False, "init_dot_color": [1, 2, 3]}))
        self.assertNotIn("init_dot_color", self.snapshot({"self_initiation": False, "init_dot_color": [1, 2, 3]}, "match2cue"))

    def test_zero_duration_omitted_only_in_simultaneous_nonmemory_mode(self):
        for task in ("active_foraging", "match2cue"):
            for sequential, memory in ((False, False), (False, True), (True, False), (True, True)):
                for duration in (0.0, 0.1, -1, False):
                    cfg = {"sequential": sequential, "is_memory": memory, "duration": duration}
                    result = self.snapshot(cfg, task)
                    should_omit = not sequential and not memory and duration == 0 and duration is not False
                    self.assertEqual("duration" not in result, should_omit)
        self.assertIn("duration", self.snapshot({"duration": 0}))
        self.assertNotIn("duration", self.snapshot({"duration": 0}, "match2cue"))

    def test_subject_maps_narrow_without_changing_resolver_contract(self):
        cfg = {"subject": "Yuri", "freq_space_tsv": {"Yuri": "freq.csv", "Tina": "other.csv"},
               "reward_space_tsv": {"Yuri": "reward.csv", "Tina": "other_reward.csv"}}
        result = self.snapshot(cfg)
        for name in ("freq_space_tsv", "reward_space_tsv"):
            self.assertEqual(set(result[name]), {"Yuri"})
            self.assertEqual(resolve_subject_mapped_value(cfg[name], subject="Yuri", field_name=name),
                             resolve_subject_mapped_value(result[name], subject="Yuri", field_name=name))

    def test_missing_subject_or_map_entry_preserves_invalid_input_for_validation(self):
        for subject in (None, "Unknown"):
            cfg = {"subject": subject, "freq_space_tsv": {"Yuri": "freq.csv"}, "reward_space_tsv": "invalid.csv"}
            self.assertEqual(self.snapshot(cfg), cfg)

    def test_disabled_tracking_removes_known_details_and_inherited_coefficients(self):
        cfg = {"eye_tracking": {"enabled": False, "calibration_file": "unused.json",
                                "uart": {"device": "unused"}, "filter": {"ema_gamma": .5},
                                "calibration": {"x_scale": 1}, "show_preview": True, "future": "keep"},
               "x_scale": 1, "y_scale": 2, "x_offset": 3, "y_offset": 4, "x_cross": 5, "y_cross": 6,
               "eye_tracker_calibration": [{"x_scale": 2}], "uart": {"baud": 123},
               "eye_filter": {"ema_gamma": .1}, "gain_directions": {"x": [1, 0]}, "extension": 9}
        self.assertEqual(self.snapshot(cfg), {"eye_tracking": {"enabled": False, "future": "keep"}, "extension": 9})

    def test_disabled_tracking_default_removes_injected_history(self):
        self.assertEqual(self.snapshot({"x_scale": 1, "eye_tracker_calibration": {"y_scale": 2}}), {})

    def test_enabled_tracking_and_calibration_task_preserve_all_tracking_settings(self):
        cfg = {"eye_tracking": {"enabled": True, "calibration_file": "used.json", "uart": {"device": "port"}},
               "x_scale": 1, "y_scale": 2, "x_offset": 3, "y_offset": 4, "eye_filter": {"ema_gamma": .2}}
        self.assertEqual(self.snapshot(cfg), cfg)
        cfg["eye_tracking"]["enabled"] = False
        self.assertEqual(self.snapshot(cfg, "calibrate_eye_tracker"), cfg)

    def test_sync_pin_stays_when_sync_enabled_even_without_daq(self):
        cfg = {"raspi": False, "sync_pin": 18, "sync_bar": {"enabled": True, "height": 40,
               "interval_mean_seconds": 2, "interval_variance_seconds": .4}}
        self.assertEqual(self.snapshot(cfg), cfg)
        cfg["sync_bar"]["enabled"] = False
        self.assertEqual(self.snapshot(cfg), {"raspi": False, "sync_bar": {"enabled": False}})
        self.assertNotIn("sync_pin", self.snapshot({"sync_pin": 18}))

    def test_nested_sync_pin_shadows_legacy_pin_and_is_pruned_when_disabled(self):
        for pin in (23, None):
            cfg = {"sync_pin": 24, "sync_bar": {"enabled": True, "sync_pin": pin}}
            self.assertEqual(self.snapshot(cfg), {"sync_bar": {"enabled": True, "sync_pin": pin}})
            cfg["sync_bar"]["enabled"] = False
            self.assertEqual(self.snapshot(cfg), {"sync_bar": {"enabled": False}})

    def test_fullscreen_drops_window_size_but_windowed_keeps_it(self):
        for fullscreen in (False, True):
            result = self.snapshot({"fullscreen": fullscreen, "win_size": [800, 600]})
            self.assertEqual("win_size" in result, not fullscreen)
        self.assertEqual(self.snapshot({"win_size": [800, 600]}), {})

    def test_gallery_defaults_windowed_and_preserves_requested_size(self):
        cfg = {"win_size": [800, 600]}
        self.assertEqual(self.snapshot(cfg, "stim_set_gallery"), cfg)
        self.assertEqual(self.snapshot({**cfg, "fullscreen": False}, "stim_set_gallery"),
                         {**cfg, "fullscreen": False})
        self.assertEqual(self.snapshot({**cfg, "fullscreen": True}, "stim_set_gallery"),
                         {"fullscreen": True})

    def test_no_daq_removes_wiring_but_keeps_foraging_reward_timing(self):
        timing = {"pump_delay_time": .2, "pump_pulse_time_seconds": .25, "inter_pump_interval": .1,
                  "timeout_duration_seconds": 2.5, "reward_to_pulse_map": {"3": 2},
                  "reward_to_timeout_map": {"0": 1}}
        cfg = {**timing, "raspi": False, "daq": {"address": 2, "module_name": "driver", "extension": 7},
               "daq_address": 1, "daq_module_name": "old_driver", "pump_pin": 3, "buzz_pin": 4}
        self.assertEqual(self.snapshot(cfg), {**timing, "raspi": False, "daq": {"extension": 7}})

    def test_enabled_daq_keeps_each_tasks_selected_address_and_module(self):
        cfg = {"raspi": True, "daq": {"address": 2, "module_name": "driver"},
               "daq_address": 1, "daq_module_name": "old_driver", "pump_pin": 3, "buzz_pin": 4}
        for task in ("active_foraging", "afc_csc1", "play_video"):
            result = self.snapshot(cfg, task)
            self.assertEqual(result["daq_address"], 1)
            self.assertEqual(result["daq"], {"module_name": "driver"})
            self.assertNotIn("daq_module_name", result)
            self.assertEqual(result["pump_pin"], 3)
        for task in ("match2cue", "touch_and_hold", "calibrate_eye_tracker"):
            result = self.snapshot(cfg, task)
            self.assertNotIn("daq_address", result)
            self.assertNotIn("daq_module_name", result)
            self.assertEqual(result["daq"], cfg["daq"])
            self.assertEqual(result["pump_pin"], 3)

    def test_daq_partial_configs_preserve_unshadowed_hardware_fields(self):
        for task in ("active_foraging", "afc_csc1", "play_video", "match2cue", "touch_and_hold", "calibrate_eye_tracker"):
            for cfg in ({"raspi": True, "daq_address": 3, "daq_module_name": "driver"},
                        {"raspi": True, "daq": {"address": 2}, "daq_module_name": "driver"},
                        {"raspi": True, "daq": {"module_name": "driver"}, "daq_address": 3}):
                self.assertEqual(self.snapshot(cfg, task), cfg)

    def test_video_daq_module_overrides_nested_and_legacy_module_names(self):
        cfg = {"raspi": True, "daq_module": "actual_driver", "daq_module_name": "legacy_driver",
               "daq_address": 1, "daq": {"address": 2, "module_name": "nested_driver"}}
        self.assertEqual(self.snapshot(cfg, "play_video"),
                         {"raspi": True, "daq_module": "actual_driver", "daq_address": 1})
        # The physical-host default can enable rewards even without raspi.
        cfg.pop("raspi")
        self.assertEqual(self.snapshot(cfg, "play_video"), {"daq_module": "actual_driver", "daq_address": 1})
        self.assertEqual(self.snapshot({**cfg, "raspi": False}, "play_video"), {"raspi": False})

    def test_invalid_primary_daq_address_is_not_replaced_with_nested_fallback(self):
        cfg = {"raspi": True, "daq_address": None, "daq": {"address": 3}}
        self.assertEqual(self.snapshot(cfg), {"raspi": True, "daq_address": None})

    def test_video_host_default_does_not_imply_disabled_rewards(self):
        cfg = {"pump_pin": 2, "pump_pulse_time_seconds": .1, "reward_interval": 5}
        self.assertEqual(self.snapshot(cfg, "play_video"), cfg)
        self.assertEqual(self.snapshot({**cfg, "raspi": False}, "play_video"), {"raspi": False})
        self.assertEqual(self.snapshot({**cfg, "raspi": True}, "play_video"), {**cfg, "raspi": True})

    def test_active_foraging_ignores_known_legacy_visual_options(self):
        self.assertEqual(self.snapshot({"bg": [1, 2, 3], "svg_size": [200, 200], "margin": 50,
                                        "choice_hitbox_scale": 2, "display_margin": 100}), {"display_margin": 100})

    def test_aliases_stay_when_they_supply_effective_values(self):
        cfg = {"ibi": .2, "refrech_rate": 60}
        self.assertEqual(self.snapshot(cfg), cfg)
        self.assertEqual(self.snapshot({**cfg, "iti": .5, "refresh_rate": 120}), {"iti": .5, "refresh_rate": 120})

    def test_csc1_inactive_branch_parameters(self):
        cfg = {"self_initiation": False, "show_fixation": False, "fixation_size": 20,
               "fixed_positions": False, "position_spacing": 200, "pairing_mode": "all", "num_stim": 5,
               "cue_time": .5, "duration": 1, "pump_pulse_time_seconds": .2, "reward_pulse_time": .1}
        result = self.snapshot(cfg, "afc_csc1")
        for name in ("self_initiation", "fixation_size", "position_spacing", "num_stim", "duration", "reward_pulse_time"):
            self.assertNotIn(name, result)
        for name in ("show_fixation", "fixed_positions", "pairing_mode", "cue_time", "pump_pulse_time_seconds"):
            self.assertEqual(result[name], cfg[name])
        self.assertEqual(self.snapshot({"trial_type": "shape_to_shape", "pairing_mode": "matched", "num_stim": 5,
                                        "likelihood_tsv": "unused.csv"}, "afc_csc1"), {"trial_type": "shape_to_shape"})

    def test_continuous_touch_training_omits_unused_trial_and_hold_controls(self):
        cfg = {"duration": None, "end_on_release": False, "n": 20, "iti": .2,
               "hold_before_choice": True, "hold_cue_to_init_time_s": .25, "self_initiation": False,
               "initiation_cue_style": "checker", "reward_interval": 1, "pump_pulse_time_seconds": .1}
        result = self.snapshot(cfg, "touch_and_hold")
        self.assertEqual(result, {"duration": None, "end_on_release": False, "initiation_cue_style": "checker",
                                  "reward_interval": 1, "pump_pulse_time_seconds": .1})
        bounded = {"duration": 5, "end_on_release": False, "n": 20, "iti": .2}
        self.assertEqual(self.snapshot(bounded, "touch_and_hold"), bounded)

    def test_unknown_task_and_extension_keys_are_not_schema_filtered(self):
        cfg = {"config_name": "active_foraging", "fullscreen": True, "win_size": [800, 600],
               "placement_distribution": "uniform", "center_point": [0, 0], "custom": {"enabled": False, "keep": 2}}
        self.assertEqual(self.snapshot(cfg, "custom_task"), cfg)
        self.assertEqual(self.snapshot({"custom": cfg["custom"]})["custom"], cfg["custom"])

    def test_output_tests_keep_actual_driver_fields_and_ignore_raspi_flag(self):
        for task, pin in (("pulse_pump", "pump_pin"), ("pulse_buzzer", "buzz_pin")):
            cfg = {"duration": 2, "daq_address": 3, "daq_module": "driver", pin: 4,
                   "raspi": False, "screens": {"main": 0}, "fullscreen": True,
                   "eye_tracking": {"enabled": True}, "x_scale": 1,
                   "sync_bar": {"enabled": True}, "sync_pin": 18, "experiment_id": "run-uuid"}
            self.assertEqual(self.snapshot(cfg, task), {"duration": 2, "daq_address": 3,
                             "daq_module": "driver", pin: 4, "experiment_id": "run-uuid"})

    def test_launch_snapshot_preserves_task_menu_and_initial_state(self):
        cfg = {"record_experiment_video": False, "video_config": "unused.json",
               "eye_tracker_experiment": {"enabled": False, "uart": {"baud": 1}, "start_timeout_s": 15, "future": 2},
               "sync_bar": {"enabled": False, "sync_pin": 23, "height": 100, "interval_mean_seconds": 2, "interval_variance_seconds": .4},
               "tasks": {"A": {"launch": "task/a.py"}, "B": {"launch": "task/b.py"}},
               "initial_state": {"eye_tracker_calibration": [{"x_scale": 2}]}, "unknown": 7}
        original = copy.deepcopy(cfg)
        result = build_launch_config_snapshot(cfg)
        self.assertEqual(cfg, original)
        self.assertNotIn("video_config", result)
        self.assertEqual(result["eye_tracker_experiment"], {"enabled": False, "future": 2})
        self.assertEqual(result["sync_bar"], {"enabled": False})
        for key in ("tasks", "initial_state", "unknown"):
            self.assertEqual(result[key], cfg[key])
        result["tasks"]["A"]["launch"] = "changed"
        self.assertEqual(cfg, original)

    def test_enabled_launch_services_keep_every_setting(self):
        cfg = {"record_experiment_video": True, "video_config": "used.json",
               "eye_tracker_experiment": {"enabled": True, "uart": {"baud": 1}, "start_timeout_s": 15},
               "sync_bar": {"enabled": True, "sync_pin": None, "height": 100, "interval_mean_seconds": 2, "interval_variance_seconds": .4}}
        self.assertEqual(build_launch_config_snapshot(cfg), cfg)


if __name__ == "__main__":
    unittest.main()
