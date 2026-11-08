"""Prune known inactive settings from executable experiment config snapshots.

These helpers copy their inputs. They are deliberately not schema allowlists:
unknown task scripts and unknown extension keys survive, so saving a snapshot
cannot silently disable a new option. Rules follow the task entry points and
their defaults; they must be updated alongside changes to those contracts.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping


_DISPLAY_TASKS = {
    "active_foraging", "match2cue", "afc_csc1", "afc_trial_sequence",
    "random_image_sequence", "play_video", "calibrate_eye_tracker",
    "touch_and_hold", "stim_set_gallery",
}
_DAQ_TASKS = {
    "active_foraging", "match2cue", "afc_csc1", "play_video",
    "calibrate_eye_tracker", "touch_and_hold",
}
_TRACKING_TASKS = _DISPLAY_TASKS - {"stim_set_gallery"}
_OUTPUT_TEST_TASKS = {"pulse_pump", "pulse_buzzer"}
_SYNC_DETAILS = {"height", "interval_mean_seconds", "interval_variance_seconds", "sync_pin"}
_EYE_DETAILS = {"calibration_file", "calibration", "uart", "filter", "show_preview"}
_EYE_TOP_LEVEL = {
    "x_scale", "y_scale", "x_offset", "y_offset", "x_cross", "y_cross",
    "eye_tracker_calibration", "uart", "eye_filter", "input_units",
    "coordinate_units", "eye_id", "gain_directions",
}
_INITIATION_KEYS = {
    "initiation_cue_center_position", "initiation_cue_style",
    "hold_before_choice", "hold_cue_to_init_time_s",
}
_GAZE_KEYS = {"fixation_criteria_s", "fixation_window_radius", "saccade_wait_timeout_s"}


def _drop(config: dict[str, Any], keys) -> None:
    for key in keys:
        config.pop(key, None)


def _disabled_section(config: Mapping[str, Any], name: str) -> bool:
    section = config.get(name)
    return section is None or (
        isinstance(section, Mapping) and section.get("enabled", False) is False
    )


def _prune_disabled_details(config: dict[str, Any], name: str, keys) -> None:
    if _disabled_section(config, name) and isinstance(config.get(name), dict):
        _drop(config[name], keys)


def _prune_daq(config: dict[str, Any], task: str) -> None:
    # play_video's CLI defaults raspi from the physical host. An absent flag
    # there is not proof that reward hardware is disabled.
    default_raspi = None if task == "play_video" else False
    if config.get("raspi", default_raspi) is False:
        _drop(config, {"daq_address", "daq_module_name", "pump_pin", "buzz_pin"})
        if isinstance(config.get("daq"), dict):
            _drop(config["daq"], {"address", "module_name"})
            if not config["daq"]:
                config.pop("daq")
        # Other reward tasks reserve/log outcome timing even in simulation.
        # Video's periodic reward controller, however, exists only on the rig.
        if task == "play_video":
            _drop(config, {"daq_module", "pump_pulse_time_seconds", "reward_interval"})
    else:
        daq = config.get("daq")
        if isinstance(daq, dict):
            # These entry points route address through _get(), whose cfg
            # lookup precedes its nested-DAQ fallback. The remaining tasks
            # read daq.address first. Preserve the actual selected hardware.
            if task in {"active_foraging", "afc_csc1", "play_video"} and "daq_address" in config:
                daq.pop("address", None)
            elif "address" in daq:
                config.pop("daq_address", None)

            # Video has an additional top-level daq_module option; the other
            # tasks use nested module_name before legacy daq_module_name.
            if task == "play_video" and "daq_module" in config:
                daq.pop("module_name", None)
            elif "module_name" in daq:
                config.pop("daq_module_name", None)
            if not daq:
                config.pop("daq")
        if task == "play_video" and "daq_module" in config:
            config.pop("daq_module_name", None)


def _narrow_subject_maps(config: dict[str, Any]) -> None:
    subject = config.get("subject")
    if not isinstance(subject, str):
        return
    subject = subject.strip()
    for key in ("freq_space_tsv", "reward_space_tsv"):
        choices = config.get(key)
        if isinstance(choices, Mapping) and subject in choices:
            # Keep the mapping shape required by resolve_subject_mapped_value.
            # Missing/invalid selected entries remain visible to task validation.
            config[key] = {subject: choices[subject]}


def build_task_config_snapshot(
    config: Mapping[str, Any], *, launch_path: str | Path,
) -> dict[str, Any]:
    """Return a pruned copy that remains suitable as the task's --config input.

    Dispatch uses the launch script rather than a user-facing task/config name.
    Unknown scripts are copied unchanged. Known tasks retain unknown fields;
    only settings proven inactive or shadowed by an active setting are removed.
    """
    result = copy.deepcopy(dict(config))
    task = Path(launch_path).stem
    if task in _OUTPUT_TEST_TASKS:
        # These utilities always drive one DAQC2 output and have no display,
        # tracker, or reward scheduler. Their CLI reads top-level daq_address
        # and daq_module, rather than the nested task DAQ object.
        _drop(result, _EYE_TOP_LEVEL | {
            "eye_tracking", "sync_bar", "sync_pin", "fullscreen", "win_size",
            "screens", "main_screen", "experimenter_screen", "secondary_screen",
            "raspi", "daq", "daq_module_name", "pump_pulse_time_seconds", "reward_interval",
        })
        result.pop("buzz_pin" if task == "pulse_pump" else "pump_pin", None)
        return result
    if task not in _DISPLAY_TASKS:
        return result

    # The gallery opens windowed by default; presentation tasks default to
    # fullscreen. Do not discard the gallery's effective requested window.
    default_fullscreen = task != "stim_set_gallery"
    if result.get("fullscreen", default_fullscreen) is True:
        result.pop("win_size", None)
    if "refresh_rate" in result:
        result.pop("refrech_rate", None)
    if "iti" in result and task in {"active_foraging", "afc_csc1", "afc_trial_sequence"}:
        result.pop("ibi", None)

    _prune_disabled_details(result, "sync_bar", _SYNC_DETAILS)
    sync_options = result.get("sync_bar")
    if _disabled_section(result, "sync_bar") or (
        isinstance(sync_options, Mapping) and "sync_pin" in sync_options
    ):
        result.pop("sync_pin", None)
    # The visual sync bar uses its GPIO independently of DAQC2's raspi flag.
    if task in _TRACKING_TASKS and task != "calibrate_eye_tracker":
        if _disabled_section(result, "eye_tracking"):
            _prune_disabled_details(result, "eye_tracking", _EYE_DETAILS)
            _drop(result, _EYE_TOP_LEVEL)
    if task in _DAQ_TASKS:
        _prune_daq(result, task)

    if task in {"active_foraging", "match2cue"}:
        default_initiation = task == "match2cue"
        if result.get("self_initiation", default_initiation) is False:
            _drop(result, _INITIATION_KEYS)
            isi = result.get("isi", 0.0 if task == "match2cue" else 0.5)
            if isinstance(isi, (int, float)) and not isinstance(isi, bool) and isi == 0:
                # Without initiation or pre-stimulus dots this color is never
                # drawn; ordinary/preview hidden dots use dot_color instead.
                result.pop("init_dot_color", None)
        distribution = result.get("placement_distribution", "circle")
        if isinstance(distribution, str) and distribution.strip().lower() == "uniform":
            _drop(result, {"fixed_positions", "center_point", "stim_range_radius"})
        # A positive duration in this mode is invalid, not an inactive value:
        # preserve it so the task still rejects the inconsistent configuration.
        default_sequence = task == "active_foraging"
        if (result.get("sequential", default_sequence) is False
                and result.get("is_memory", default_sequence) is False
                and result.get("duration") == 0
                and not isinstance(result.get("duration"), bool)):
            result.pop("duration", None)

    if task == "active_foraging":
        if result.get("wait_for_saccade_on_dot", False) is False:
            _drop(result, _GAZE_KEYS)
        _narrow_subject_maps(result)
        # The palette determines the background; these legacy knobs are not
        # forwarded by active_foraging.main to the presenter or renderer.
        _drop(result, {"bg", "svg_size", "margin", "choice_hitbox_scale"})

    elif task == "afc_csc1":
        # CSC1 always self-initiates, regardless of this deprecated flag.
        result.pop("self_initiation", None)
        if result.get("show_fixation", False) is False:
            result.pop("fixation_size", None)
        if result.get("fixed_positions", False) is False:
            result.pop("position_spacing", None)
        if result.get("trial_type", "random") == "shape_to_shape":
            _drop(result, {"pairing_mode", "num_stim", "likelihood_tsv"})
        elif result.get("pairing_mode", "matched") == "all":
            result.pop("num_stim", None)
        if result.get("cue_time") is not None:
            result.pop("duration", None)
        if result.get("pump_pulse_time_seconds") is not None:
            result.pop("reward_pulse_time", None)
        if result.get("inter_pump_interval") is not None:
            result.pop("reward_inter_pulse_time", None)

    elif task == "touch_and_hold":
        _drop(result, {"self_initiation", "hold_before_choice", "hold_cue_to_init_time_s"})
        if "init_dot_color" in result:
            result.pop("dot_color", None)
        if result.get("duration", 5.0) is None and result.get("end_on_release", True) is False:
            _drop(result, {"n", "iti"})

    elif task == "calibrate_eye_tracker":
        # This task opens the tracker even if eye_tracking.enabled is false.
        # Preserve calibration/UART fields, including inherited calibration.
        if "fixation_cross_size" in result:
            result.pop("fixation_size", None)
        if "reward_interval" in result:
            result.pop("fix_accept_time", None)
        if result.get("fixation_radius") is not None:
            result.pop("fix_diameter", None)

    return result


def build_launch_config_snapshot(config: Mapping[str, Any]) -> dict[str, Any]:
    """Copy launcher provenance, removing only known disabled-service details.

    Task menus and initial state are provenance for the entire experiment and
    must not be filtered by whichever block happened to run first.
    """
    result = copy.deepcopy(dict(config))
    if result.get("record_experiment_video", False) is False:
        result.pop("video_config", None)
    _prune_disabled_details(result, "eye_tracker_experiment", {"uart", "start_timeout_s"})
    _prune_disabled_details(result, "sync_bar", _SYNC_DETAILS)
    return result
