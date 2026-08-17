#!/usr/bin/env python3
"""Match-to-cue task using the shared AFC presentation engine.

Trial sequence:
    onset cue -> optional cue reward -> match cue -> delay -> options -> choice
    -> optional choice reward -> inter-trial interval

One option always equals the match cue. Other options are sampled independently
from the full configured stimulus space, so duplicate matches are possible.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from psychopy import core, event, logging as pylogging

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from bin import utils
from bin.afc_geometry import compute_afc_positions, stimulus_size
from bin.afc_stimuli import (
    AFCStimulusSpace,
    StimulusKey,
    load_afc_stimulus_space,
    render_afc_stimuli,
    stimulus_from_json,
    stimulus_from_storage_key,
    stimulus_storage_key,
    stimulus_to_json,
)
from bin.config import load_config, validate_config
from bin.daqc2_outputs import DAQC2DigitalOutputs
from bin.frame_timing import plan_frame_duration, validate_requested_durations
from bin.logger import SessionLogBundle
from bin.match2cue_logic import (
    Match2CueTrial,
    generate_match2cue_trial,
    resolve_match2cue_reward_settings,
    reward_train_duration,
    score_match2cue_choice,
    should_deliver_cue_tap_reward,
)
from bin.screen import (
    ExperimenterPreview,
    describe_screen,
    load_screen_config,
    oriented_size,
    resolve_scene_size,
    set_window_mouse_visible,
    software_stimulus_rotation,
)
from bin.task_lifecycle import USER_EXIT_CODE
from interface.rig_mode import IS_RIG_ENV_VAR, experimenter_cursor_visible_for_touchscreen
from task.active_foraging_timing import (
    validate_duration_for_presentation_mode,
)


def _generate_trial_payload(trial_idx: int, config: dict) -> dict:
    n_trials = int(config["n_trials"])
    if n_trials > 0 and trial_idx > n_trials:
        return {"type": "done"}

    seed = config.get("seed")
    rng = random.Random(None if seed is None else int(seed) + int(trial_idx))
    stimuli = [stimulus_from_json(value) for value in config["stimuli"]]
    trial = generate_match2cue_trial(stimuli, int(config["num_afc"]), rng=rng)
    shapes = {int(key): Path(value) for key, value in config["shapes"].items()}
    colors = {int(key): tuple(value) for key, value in config["colors"].items()}
    rendered = render_afc_stimuli(
        [trial.cue, *trial.options],
        shapes=shapes,
        colors=colors,
        image_size=tuple(config["image_size"]),
        bg=tuple(config["bg"]),
        stroke_width=config.get("stroke_width"),
        stroke_color=(
            tuple(config["stroke_color"])
            if config.get("stroke_color") is not None
            else None
        ),
        stroke_linejoin=config.get("stroke_linejoin"),
        stroke_linecap=config.get("stroke_linecap"),
    )
    return {
        "type": "trial",
        "trial_idx": int(trial_idx),
        "cue": stimulus_to_json(trial.cue),
        "options": [stimulus_to_json(value) for value in trial.options],
        "reward_draw": float(trial.reward_draw),
        "cue_reward_draw": float(trial.cue_reward_draw),
        "rendered": {
            stimulus_storage_key(key): value for key, value in rendered.items()
        },
    }


def _decode_trial_payload(
    payload: dict,
) -> Tuple[Optional[Match2CueTrial], Dict[StimulusKey, Any]]:
    if payload.get("type") == "done":
        return None, {}
    if payload.get("type") != "trial":
        raise RuntimeError(f"Unexpected payload from trial buffer: {payload}")
    cue = stimulus_from_json(payload["cue"])
    options = tuple(stimulus_from_json(value) for value in payload["options"])
    trial = Match2CueTrial(
        cue=cue,
        options=options,
        reward_draw=float(payload["reward_draw"]),
        cue_reward_draw=float(payload.get("cue_reward_draw", 1.0)),
    )
    rendered = {
        stimulus_from_storage_key(key): value
        for key, value in payload["rendered"].items()
    }
    return trial, rendered


def _meta_values(
    space: AFCStimulusSpace,
    stimulus: Optional[StimulusKey],
) -> Tuple[Any, Any, Any]:
    if stimulus is None:
        return "", "", ""
    return space.metadata[stimulus]


def _build_behavior_fieldnames(num_afc: int) -> List[str]:
    fields = [
        "trial_num",
        "initiation_time",
        "reaction_time",
        "cue_shape",
        "cue_color",
        "cue_lum",
        "matching_option_count",
        "tie_mode",
    ]
    for option_idx in range(int(num_afc)):
        fields.extend(
            [
                f"shape_{option_idx}",
                f"color_{option_idx}",
                f"lum_{option_idx}",
            ]
        )
    fields.extend(
        [
            "choice_made_index",
            "choice_made_shape",
            "choice_made_color",
            "choice_made_lum",
            "choice_correct",
            "cue_reward_probability",
            "cue_reward_delivered",
            "reward_probability",
            "reward_delivered",
            "choice_reward_pulse_count",
            "choice_touch_x",
            "choice_touch_y",
            "choice_reaction_time",
            "main_display_dropped_frames",
        ]
    )
    return fields


def _optional_float(value: Any) -> str:
    if value is None or value == "":
        return ""
    return f"{float(value):.9f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Match-to-cue task")
    parser.add_argument("--config", required=True, help="Path to task JSON config")
    parser.add_argument("--main_screen", default=None, help="Main task screen index or output name")
    parser.add_argument("--experimenter_screen", default=None, help="Experimenter screen index or output name")
    return parser.parse_args()


def run_task(cfg: Dict[str, Any], *, screen_config: Dict[str, Any]) -> str:
    subject_value = cfg["subject"]
    if not isinstance(subject_value, str) or not subject_value.strip():
        raise ValueError("Config field 'subject' must be a non-empty string")
    subject = subject_value.strip()

    n_trials = int(cfg["n"])
    num_afc = int(cfg.get("num_afc", 3))
    duration = float(cfg.get("duration", 0.0))
    match_cue_duration = float(cfg["match_cue_duration"])
    delay_time = float(cfg["delay_time"])
    choice_time = float(cfg.get("choice_time", 2.0))
    isi = float(cfg.get("isi", 0.0))
    iti = float(cfg.get("iti", 1.0))
    sequential = bool(cfg.get("sequential", False))
    is_memory = bool(cfg.get("is_memory", False))
    self_initiation = bool(cfg.get("self_initiation", True))
    if num_afc < 1:
        raise ValueError("num_afc must be at least 1")
    validate_requested_durations(
        {
            "match_cue_duration": match_cue_duration,
            "delay_time": delay_time,
            "choice_time": choice_time,
            "isi": isi,
            "iti": iti,
        },
        positive={"match_cue_duration", "choice_time"},
        context="match2cue",
    )
    validate_duration_for_presentation_mode(
        duration,
        sequential=sequential,
        is_memory=is_memory,
        context="match2cue",
    )

    image_size_raw = cfg.get("image_size")
    if not isinstance(image_size_raw, list) or len(image_size_raw) != 2:
        raise ValueError("Config field 'image_size' must contain [width, height]")
    image_size = (int(image_size_raw[0]), int(image_size_raw[1]))
    if image_size[0] <= 0 or image_size[1] <= 0:
        raise ValueError("image_size values must be positive")

    utils.set_debug(bool(cfg.get("debug", False)))
    seed = cfg.get("seed")
    position_rng = random.Random(seed)
    space = load_afc_stimulus_space(
        colors_tsv=cfg["colors_tsv"],
        shapes_tsv=cfg["shapes_tsv"],
        n_colors=int(cfg["n_colors"]),
        n_shapes=int(cfg["n_shapes"]),
        n_lum_levels=int(cfg.get("n_lum_levels", 0)),
    )

    raspi = bool(cfg.get("raspi", False))
    trial_start_pin = int(cfg.get("trial_start_pin", 18))
    daq_cfg = cfg.get("daq", {}) if isinstance(cfg.get("daq"), dict) else {}
    daq_address = int(daq_cfg.get("address", cfg.get("daq_address", 0)))
    daq_module_name = str(
        daq_cfg.get("module_name", cfg.get("daq_module_name", "piplates.DAQC2plate"))
    )
    pump_pin = int(cfg.get("pump_pin", 0))
    pump_delay_time = max(0.0, float(cfg.get("pump_delay_time", 0.0)))
    pump_pulse_time = max(0.0, float(cfg.get("pump_pulse_time_seconds", 0.25)))
    reward_settings = resolve_match2cue_reward_settings(
        reward_match_cue_prob=cfg.get("reward_match_cue_prob", 0.0),
        correct_num_pulse=cfg.get("correct_num_pulse", 1),
        inter_pump_interval=cfg.get("inter_pump_interval"),
        pump_pulse_time_seconds=pump_pulse_time,
        tie_mode=cfg.get("tie_mode", "random"),
    )
    if raspi:
        DAQC2DigitalOutputs.validate_address(daq_address)
        DAQC2DigitalOutputs.validate_bit(pump_pin)

    config_name = str(cfg.get("config_name", "match2cue")).strip() or "match2cue"
    session_logs = SessionLogBundle(
        output_root=str(cfg.get("output_dir", "./logs")),
        task_name="match2cue",
        config_name=config_name,
        behavior_fieldnames=_build_behavior_fieldnames(num_afc),
        auto_flush=False,
    )
    logger = session_logs.event_logger
    msg_logger = session_logs.message_logger
    behavior_logger = session_logs.behavior_logger
    if behavior_logger is None:
        raise RuntimeError("match2cue requires a behavior logger")
    msg_logger.log(
        "INFO",
        f"session_start task=match2cue config_name={config_name} subject={subject} session_dir={session_logs.session_dir}",
    )
    msg_logger.log(
        "INFO",
        (
            "reward_settings "
            f"reward_match_cue_prob={reward_settings.reward_match_cue_prob:.9f} "
            f"correct_num_pulse={reward_settings.correct_num_pulse} "
            f"inter_pump_interval={reward_settings.inter_pump_interval:.9f} "
            f"tie_mode={reward_settings.tie_mode}"
        ),
    )

    fullscreen = bool(cfg.get("fullscreen", True))
    win_size_value = cfg.get("win_size")
    win_size = tuple(win_size_value) if win_size_value is not None else None
    win, main_screen, experimenter_screen = utils.setup_task_window(
        screen_config,
        bg_rgb_255=space.bg,
        fullscreen=fullscreen,
        size=win_size,
        allow_same_screen=True,
    )
    msg_logger.log(
        "INFO",
        f"resolved_screens main={describe_screen(main_screen)} experimenter={describe_screen(experimenter_screen)}",
    )
    touchscreen = bool(cfg.get("touchscreen", False))
    experimenter_preview = None
    status_counts = {"Correct": 0, "Incorrect": 0, "Rewards delivered": 0}
    if touchscreen:
        set_window_mouse_visible(win, False)

    refresh_rate = cfg.get("refresh_rate", cfg.get("refrech_rate"))
    fps, frame_duration = utils.resolve_frame_rate(
        win,
        refresh_rate,
        msg_logger=msg_logger,
        context="match2cue",
    )
    if experimenter_screen is not None:
        experimenter_preview = ExperimenterPreview(
            experimenter_screen,
            task_label=config_name,
            subject=subject,
            current_trial_num=1,
            total_trials=n_trials,
            status_counts=status_counts,
            start_perf_s=time.perf_counter(),
            update_interval_s=0.1,
            mouse_visible=experimenter_cursor_visible_for_touchscreen(
                touchscreen,
                os.environ.get(IS_RIG_ENV_VAR),
            ),
        )

    main_scene_size = resolve_scene_size(
        main_screen,
        fullscreen=fullscreen,
        requested_size=win_size,
        realized_size=tuple(win.size),
    )
    stimulus_rotation_degrees = software_stimulus_rotation(main_screen.rotation)
    subject_scene_size = oriented_size(
        main_scene_size,
        stimulus_rotation_degrees,
    )
    msg_logger.log(
        "INFO",
        f"resolved_main_scene_size native_size={main_scene_size[0]}x{main_scene_size[1]} "
        f"subject_size={subject_scene_size[0]}x{subject_scene_size[1]} "
        f"output_rotation={main_screen.rotation} stimulus_rotation_deg={stimulus_rotation_degrees} "
        f"fullscreen={int(fullscreen)} requested_win_size={win_size} "
        f"realized_win_size={tuple(win.size)}",
    )
    fixation_size = int(cfg.get("fixation_size", 0))
    fix = utils.make_fixation_cross(
        win,
        size=fixation_size,
        ori=stimulus_rotation_degrees,
    )
    bg_rect = utils.make_bg_rect(win, space.bg)
    dot_size = int(cfg.get("dot_size", 40))
    dot_color = tuple(cfg.get("dot_color", [50, 50, 50]))
    init_dot_value = cfg.get("init_dot_color")
    init_dot_color = tuple(init_dot_value) if init_dot_value is not None else None
    onset_stim = None
    if self_initiation:
        onset_stim = utils.make_onset_cue_stim(
            win,
            bg_rgb_255=space.bg,
            size_frac=0.125,
            cells=8,
            sigma_frac=0.22,
            zero_threshold=1,
            ori=stimulus_rotation_degrees,
        )

    def update_preview_counts() -> None:
        if experimenter_preview is not None:
            experimenter_preview.set_status_counts(status_counts)

    def show_preview_idle() -> None:
        if experimenter_preview is None:
            return
        experimenter_preview.show_static_scene(
            bg_rgb_255=space.bg,
            main_size=main_scene_size,
            images=[],
            dots=[],
            fixation_size=(int(getattr(fix, "height", 0)) if fix is not None else None),
            fixation_color=(0, 0, 0),
            status_counts=status_counts,
            highlight_box=None,
            main_rotation_deg=stimulus_rotation_degrees,
        )

    pylogging.console.setLevel(pylogging.CRITICAL)
    pigpio_chip = None
    daqc2_outputs: Optional[DAQC2DigitalOutputs] = None
    pump_on = None
    pump_off = None
    if raspi:
        try:
            import lgpio

            pigpio_chip = lgpio.gpiochip_open(0)
            lgpio.gpio_claim_output(pigpio_chip, trial_start_pin)
        except Exception as exc:
            pigpio_chip = None
            msg_logger.log("WARN", f"lgpio unavailable: {exc}; trial-start signal will be logged only")
        try:
            daqc2_outputs = DAQC2DigitalOutputs(
                address=daq_address,
                module_name=daq_module_name,
            )
            daqc2_outputs.open()
            pump_on, pump_off = daqc2_outputs.bind_bit(pump_pin)
            pump_off()
        except Exception as exc:
            daqc2_outputs = None
            pump_on = pump_off = None
            msg_logger.log("WARN", f"DAQC2 unavailable: {exc}; pump events will be logged only")

    def set_pump(active: bool) -> None:
        callback = pump_on if active else pump_off
        if callback is not None:
            callback()

    task_end_status = "done"

    def poll_controls(*, allow_manual_reward: bool = True) -> bool:
        manual_reward = False
        if allow_manual_reward:
            try:
                manual_reward = bool(event.getKeys(keyList=["r"]))
            except Exception:
                pass
            if experimenter_preview is not None:
                manual_reward = (
                    experimenter_preview.consume_manual_reward_request()
                    or manual_reward
                )
            if manual_reward:
                deliver_reward(None, context="manual_reward")
        return bool(experimenter_preview is not None and experimenter_preview.poll())

    def wait_or_abort(duration_s: float, *, allow_manual_reward: bool = True) -> bool:
        deadline = time.perf_counter() + max(0.0, float(duration_s))
        while time.perf_counter() < deadline:
            if poll_controls(allow_manual_reward=allow_manual_reward):
                return True
            remaining = deadline - time.perf_counter()
            if remaining > 0:
                time.sleep(min(0.05, remaining))
        return poll_controls(allow_manual_reward=allow_manual_reward)

    def deliver_reward(trial_num: Optional[int], *, context: str) -> bool:
        nonlocal task_end_status
        start_requested_perf = time.perf_counter()
        aborted = False
        set_pump(True)
        start_perf = time.perf_counter()
        try:
            logger.log_signal(
                trial_num=trial_num,
                event="pump_on",
                timestamp_perf_s=start_perf,
                requested_timestamp_perf_s=start_requested_perf,
                requested_duration=pump_pulse_time,
            )
            if context == "manual_reward":
                core.wait(pump_pulse_time)
            else:
                aborted = wait_or_abort(
                    pump_pulse_time,
                    allow_manual_reward=False,
                )
        finally:
            end_requested_perf = time.perf_counter()
            set_pump(False)
            end_perf = time.perf_counter()
            logger.log_signal(
                trial_num=trial_num,
                event="pump_off",
                timestamp_perf_s=end_perf,
                requested_timestamp_perf_s=end_requested_perf,
            )
        status_counts["Rewards delivered"] += 1
        update_preview_counts()
        msg_logger.log(
            "INFO",
            f"reward_pulse trial_num={trial_num} context={context}",
        )
        if aborted:
            task_end_status = "experimenter_exit"
        return aborted

    def deliver_reward_train(
        trial_num: int,
        *,
        num_pulses: int,
        context: str,
    ) -> bool:
        """Deliver an abort-aware pulse train with gaps only between pulses."""
        nonlocal task_end_status
        for pulse_num in range(1, int(num_pulses) + 1):
            if deliver_reward(trial_num, context=context):
                return True
            if pulse_num >= int(num_pulses):
                continue
            if wait_or_abort(
                reward_settings.inter_pump_interval,
                allow_manual_reward=False,
            ):
                task_end_status = "experimenter_exit"
                msg_logger.log(
                    "WARN",
                    (
                        "experimenter_exit_during_inter_pump_interval "
                        f"trial_num={trial_num} pulse={pulse_num}"
                    ),
                )
                return True
        return False

    worker_config = {
        "n_trials": n_trials,
        "num_afc": num_afc,
        "stimuli": [stimulus_to_json(value) for value in space.stimuli],
        "shapes": {str(key): str(value) for key, value in space.shapes.items()},
        "colors": {str(key): list(value) for key, value in space.colors.items()},
        "image_size": list(image_size),
        "bg": list(space.bg),
        "seed": seed,
        "stroke_width": cfg.get("stroke_width"),
        "stroke_color": cfg.get("stroke_color"),
        "stroke_linejoin": cfg.get("stroke_linejoin"),
        "stroke_linecap": cfg.get("stroke_linecap"),
    }
    buffer_manager = utils.TrialBufferManager(
        trial_generator_func=_generate_trial_payload,
        config=worker_config,
        buffer_size=max(1, int(cfg.get("buffer_len_trials", 5))),
        start_idx=1,
    )
    iti_plan = plan_frame_duration(iti, fps)
    iti_frames = iti_plan.frame_count
    run_indefinitely = n_trials <= 0
    fixed_positions = bool(cfg.get("fixed_positions", False))
    center_value = cfg.get("center_point")
    center_point = tuple(center_value) if center_value is not None else None
    radius_value = cfg.get("stim_range_radius")
    stim_range_radius = float(radius_value) if radius_value is not None else None

    show_preview_idle()
    try:
        trial_num = 1
        while run_indefinitely or trial_num <= n_trials:
            if experimenter_preview is not None:
                experimenter_preview.set_trial_progress(trial_num, n_trials)
            if poll_controls():
                task_end_status = "experimenter_exit"
                break

            payload = buffer_manager.get_next_trial()
            trial, preloaded = _decode_trial_payload(payload)
            if trial is None:
                break
            trial_num = int(payload["trial_idx"])
            cue_image = preloaded[trial.cue]
            effective_win_size = main_scene_size
            sampled_positions, positions = compute_afc_positions(
                fixed_positions,
                num_afc,
                center_point,
                stim_range_radius,
                oriented_size(
                    stimulus_size(preloaded, trial.options[0]),
                    stimulus_rotation_degrees,
                ),
                effective_win_size,
                rng=position_rng,
            )
            msg_logger.log(
                "INFO",
                f"trial_loaded trial_num={trial_num} cue={trial.cue} options={trial.options} matches={trial.matching_count}",
            )
            for option_num, (screen_pos, psycho_pos) in enumerate(
                zip(sampled_positions, positions),
                start=1,
            ):
                msg_logger.log(
                    "INFO",
                    f"position_assigned trial_num={trial_num} option_num={option_num} screen_px={screen_pos} psychopy_pos={psycho_pos}",
                )

            cue_reward_state = {"delivered": False}
            cue_reward_won = bool(
                self_initiation
                and should_deliver_cue_tap_reward(
                    trial,
                    reward_settings.reward_match_cue_prob,
                )
            )

            def deliver_won_cue_reward() -> bool:
                cue_reward_state["delivered"] = True
                return deliver_reward(trial_num, context="cue_tap_reward")

            trial_meta: Dict[str, Any] = {}
            aborted, choice_info = utils.present_trial_with_persistent_dots(
                win=win,
                preloaded=preloaded,
                trial_options=list(trial.options),
                positions=positions,
                duration=duration,
                choice_time=choice_time,
                dot_size=dot_size,
                dot_color=dot_color,
                bg_rect=bg_rect,
                fix=fix,
                logger=logger,
                trial_num=trial_num,
                isi=isi,
                init_dot_color=init_dot_color,
                bg_rgb_255=space.bg,
                onset_cue=onset_stim,
                on_onset_cue_touch=(
                    deliver_won_cue_reward if cue_reward_won else None
                ),
                msg_logger=msg_logger,
                fps=fps,
                raspi=bool(raspi and pigpio_chip is not None),
                pigpio_pi=pigpio_chip,
                raspi_pin=trial_start_pin,
                sequential=sequential,
                is_memory=is_memory,
                choice_hitbox_scale=1.25 if touchscreen else 1.0,
                trial_meta=trial_meta,
                experimenter_preview=experimenter_preview,
                external_abort_checker=poll_controls,
                scene_main_size=effective_win_size,
                event_profile="match2cue",
                pre_options_cue_image=cue_image,
                pre_options_cue_duration=match_cue_duration,
                pre_options_delay=delay_time,
                stimulus_rotation_degrees=stimulus_rotation_degrees,
            )
            timing_monitor = trial_meta.get("_main_display_frame_timing_monitor")
            if aborted:
                if timing_monitor is not None:
                    msg_logger.log(
                        "INFO",
                        (
                            f"main_display_timing trial_num={trial_num} "
                            f"missed_refreshes={timing_monitor.missed_refreshes} "
                            "scope=continuous_frame_sequences"
                        ),
                    )
                if task_end_status != "experimenter_exit":
                    task_end_status = (
                        "experimenter_exit" if poll_controls() else "aborted"
                    )
                break

            chosen_index = choice_info.get("chosen_index") if choice_info is not None else None
            outcome = score_match2cue_choice(
                trial,
                chosen_index,
                tie_mode=reward_settings.tie_mode,
            )
            gray_start_perf = trial_meta.get("gray_flip_perf_s")
            if gray_start_perf is not None:
                gray_duration = float(iti)
                if outcome.reward_delivered:
                    gray_duration += pump_delay_time
                    gray_duration += reward_train_duration(
                        reward_settings.correct_num_pulse,
                        pump_pulse_time,
                        reward_settings.inter_pump_interval,
                    )
                logger.log_frame_flip(
                    trial_num=trial_num,
                    event="gray_inter_trial_interval",
                    timestamp_perf_s=float(gray_start_perf),
                    requested_timestamp_perf_s=trial_meta.get(
                        "gray_flip_requested_perf_s"
                    ),
                    requested_duration=(gray_duration if gray_duration > 0.0 else None),
                )
            if outcome.correct is True:
                status_counts["Correct"] += 1
            elif outcome.correct is False:
                status_counts["Incorrect"] += 1
            update_preview_counts()

            if outcome.reward_delivered:
                if pump_delay_time > 0 and wait_or_abort(pump_delay_time):
                    task_end_status = "experimenter_exit"
                    break
                if deliver_reward_train(
                    trial_num,
                    num_pulses=reward_settings.correct_num_pulse,
                    context="correct_choice_reward",
                ):
                    break

            cue_shape, cue_color, cue_lum = _meta_values(space, trial.cue)
            chosen_stimulus = (
                trial.options[int(chosen_index) - 1]
                if chosen_index is not None
                else None
            )
            chosen_shape, chosen_color, chosen_lum = _meta_values(space, chosen_stimulus)
            behavior_row: Dict[str, Any] = {
                "trial_num": trial_num,
                "initiation_time": _optional_float(trial_meta.get("initiation_time_s")),
                "reaction_time": _optional_float(
                    choice_info.get("reaction_time_s") if choice_info is not None else None
                ),
                "cue_shape": cue_shape,
                "cue_color": cue_color,
                "cue_lum": cue_lum,
                "matching_option_count": outcome.matching_count,
                "tie_mode": reward_settings.tie_mode,
                "choice_made_index": (int(chosen_index) - 1 if chosen_index is not None else ""),
                "choice_made_shape": chosen_shape,
                "choice_made_color": chosen_color,
                "choice_made_lum": chosen_lum,
                "choice_correct": (int(outcome.correct) if outcome.correct is not None else ""),
                "cue_reward_probability": f"{reward_settings.reward_match_cue_prob:.9f}",
                "cue_reward_delivered": int(cue_reward_state["delivered"]),
                "reward_probability": f"{outcome.reward_probability:.9f}",
                "reward_delivered": int(outcome.reward_delivered),
                "choice_reward_pulse_count": (
                    reward_settings.correct_num_pulse
                    if outcome.reward_delivered
                    else 0
                ),
                "choice_touch_x": _optional_float(
                    choice_info.get("touch_x") if choice_info is not None else None
                ),
                "choice_touch_y": _optional_float(
                    choice_info.get("touch_y") if choice_info is not None else None
                ),
                "choice_reaction_time": _optional_float(
                    choice_info.get("reaction_time_s") if choice_info is not None else None
                ),
            }
            for option_idx, option in enumerate(trial.options):
                shape_idx, color_idx, lum_idx = _meta_values(space, option)
                behavior_row[f"shape_{option_idx}"] = shape_idx
                behavior_row[f"color_{option_idx}"] = color_idx
                behavior_row[f"lum_{option_idx}"] = lum_idx

            show_preview_idle()
            hold_frames = iti_frames if outcome.reward_delivered else max(0, iti_frames - 1)
            timing_context = (
                timing_monitor.continuous_sequence()
                if timing_monitor is not None and hold_frames > 0
                else nullcontext()
            )
            with timing_context:
                for _ in range(hold_frames):
                    if poll_controls():
                        task_end_status = "experimenter_exit"
                        break
                    bg_rect.draw()
                    if fix is not None:
                        fix.draw()
                    win.flip()
            missed_refreshes = (
                timing_monitor.missed_refreshes
                if timing_monitor is not None
                else ""
            )
            behavior_row["main_display_dropped_frames"] = missed_refreshes
            behavior_logger.writerow(behavior_row)
            msg_logger.log(
                "INFO",
                (
                    f"main_display_timing trial_num={trial_num} "
                    f"missed_refreshes={missed_refreshes} "
                    "scope=continuous_frame_sequences"
                ),
            )
            if task_end_status != "done":
                break
            session_logs.flush()
            trial_num += 1
    except Exception:
        task_end_status = "error"
        raise
    finally:
        try:
            set_pump(False)
        except Exception:
            pass
        try:
            buffer_manager.close()
        except Exception:
            pass
        try:
            msg_logger.log("INFO", f"session_end status={task_end_status}")
            session_logs.close()
        except Exception:
            pass
        try:
            if experimenter_preview is not None:
                experimenter_preview.close()
        except Exception:
            pass
        try:
            if daqc2_outputs is not None:
                daqc2_outputs.close()
        except Exception:
            pass
        if pigpio_chip is not None:
            try:
                import lgpio

                lgpio.gpiochip_close(pigpio_chip)
            except Exception:
                pass
        try:
            win.close()
        except Exception:
            pass
    return task_end_status


def main() -> None:
    args = parse_args()
    try:
        cfg = load_config(args.config)
        validate_config(
            cfg,
            required=[
                "config_name",
                "subject",
                "colors_tsv",
                "shapes_tsv",
                "n_colors",
                "n_shapes",
                "n",
                "match_cue_duration",
                "delay_time",
            ],
            allow_zero_duration=True,
        )
        screen_config = load_screen_config(
            cfg,
            cli_main=args.main_screen,
            cli_experimenter=args.experimenter_screen,
        )
        status = run_task(cfg, screen_config=screen_config)
        if status != "done":
            raise SystemExit(USER_EXIT_CODE)
    except SystemExit:
        raise
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
