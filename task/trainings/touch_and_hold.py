#!/usr/bin/env python3
"""Teach sustained cue contact with periodic juice rewards.

All times are seconds. Null duration removes the time limit; end_on_release
still ends trials. With neither boundary enabled, the cue is continuous.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys
import time

_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from bin.config import load_config
from bin.daqc2_outputs import DAQC2DigitalOutputs
from bin.eye_tracking_config import load_eye_tracking_settings
from bin.frame_timing import flip_with_timestamps
from bin.initiation import initiation_cue_position, resolve_initiation_config
from bin.logger import SessionLogBundle
from bin.screen import (
    ExperimenterPreview, describe_screen, load_screen_config, resolve_scene_size,
    serialize_preview_image, set_window_mouse_visible, software_stimulus_rotation,
)
from bin.task_lifecycle import USER_EXIT_CODE
from bin.touch_input import MousePressTracker
from interface.rig_mode import IS_RIG_ENV_VAR, experimenter_cursor_visible_for_touchscreen
from task.trainings.touch_and_hold_state import TouchAndHoldState, seconds


def run_task(
    *, colors_tsv, duration=5.0, iti=0.2, reward_interval=1.0,
    end_on_release=True, n=None, output_dir="logs/trainings",
    config_name="touch_and_hold", subject=None, fullscreen=True, win_size=None,
    screen_config=None, touchscreen=False, initiation_cue_center_position=None,
    initiation_cue_style="checker", init_dot_color=(50, 50, 50),
    raspi=False, daq_address=0, daq_module_name="piplates.DAQC2plate",
    pump_pin=0, pump_pulse_time_seconds=0.275, eye_tracking_config=None,
):
    state = TouchAndHoldState(
        duration=duration, iti=iti, reward_interval=reward_interval,
        end_on_release=end_on_release, n=n,
    )
    pulse_s = seconds(pump_pulse_time_seconds, "pump_pulse_time_seconds", positive=True)
    if pulse_s >= state.reward_interval:
        raise ValueError("pump_pulse_time_seconds must be shorter than reward_interval")
    outputs = DAQC2DigitalOutputs(address=daq_address, module_name=daq_module_name, enabled=raspi)
    pump_pin = outputs.validate_bit(pump_pin)

    # Keep timing/config helpers importable without PsychoPy or a connected rig.
    from psychopy import core, event
    from bin import utils

    bg, _ = utils.split_background_from_palette(utils.load_color_palette(Path(colors_tsv)))
    with ExitStack() as cleanup:
        logs = SessionLogBundle(
            output_root=output_dir, task_name="touch_and_hold", config_name=config_name,
            eye_tracking_config=eye_tracking_config, auto_flush=False,
        )
        cleanup.callback(logs.close)
        logger = logs.event_logger
        messages = logs.message_logger
        # Keep training sessions identifiable even when the experiment manager
        # supplies an exact block directory instead of the standalone root.
        (logs.session_dir / "training.json").write_text(json.dumps({
            "session_type": "training", "experimental_data": False,
            "task": "touch_and_hold", "config_name": config_name, "subject": subject,
            "duration": duration, "iti": iti, "reward_interval": reward_interval,
            "end_on_release": end_on_release, "n": n,
        }, indent=2) + "\n", encoding="utf-8")
        win, main_screen, experimenter_screen = utils.setup_task_window(
            screen_config, bg_rgb_255=bg, fullscreen=fullscreen, size=win_size,
            allow_same_screen=True,
        )
        cleanup.callback(utils.close_task_window, win)
        if touchscreen:
            set_window_mouse_visible(win, False)
        messages.log("INFO", f"training_session subject={subject} main={describe_screen(main_screen)} "
                     f"experimenter={describe_screen(experimenter_screen)}")
        utils.resolve_frame_rate(win, None, msg_logger=messages, context="touch_and_hold")
        scene_size = resolve_scene_size(
            main_screen, fullscreen=fullscreen, requested_size=win_size, realized_size=tuple(win.size),
        )
        rotation = software_stimulus_rotation(main_screen.rotation)
        logs.set_eye_tracking_screen(scene_size, rotation)
        cue, pressed_cue = utils.make_initiation_cue_stims(
            win, bg_rgb_255=bg, cue_color=init_dot_color, style=initiation_cue_style,
            position=initiation_cue_position(initiation_cue_center_position, scene_size, rotation),
            ori=rotation,
        )
        background = utils.make_bg_rect(win, bg)
        preview = None
        if experimenter_screen is not None:
            preview = ExperimenterPreview(
                experimenter_screen, task_label=f"Training: {config_name}", subject=subject,
                current_trial_num=None if state.continuous else 1,
                total_trials=None if state.continuous else n,
                eye_tracking_shared=logs.eye_tracking_preview,
                mouse_visible=experimenter_cursor_visible_for_touchscreen(
                    touchscreen, os.environ.get(IS_RIG_ENV_VAR),
                ),
            )
            cleanup.callback(preview.close)
            utils.verify_task_window_vblank(win)

        # Cache the shared cue images once; send only scene/counter changes.
        preview_images = [dict(
            image_payload=serialize_preview_image(stim.image),
            pos=list(stim.pos), size=list(stim.size), ori=rotation,
        ) for stim in (cue, pressed_cue)] if preview is not None else []
        automatic_rewards = 0
        manual_rewards = 0
        pump_until_s = None
        pump_trial = None
        pump_manual = False

        def stop_pump():
            nonlocal pump_until_s
            requested_ns = time.perf_counter_ns()
            outputs.write(pump_pin, False)
            if pump_until_s is not None:
                pump_until_s = None
                logger.log_signal(
                    trial_num=pump_trial, event="pump_off", timestamp_perf_ns=time.perf_counter_ns(),
                    requested_timestamp_perf_ns=requested_ns,
                )

        # Register before opening hardware: initialization and later failures
        # both attempt to clear the output, then close preview/window/logs.
        cleanup.callback(stop_pump)
        outputs.open()
        outputs.write(pump_pin, False)
        if not raspi:
            messages.log("INFO", "raspi=False; rewards are simulated and logged")

        def start_pump(manual=False):
            nonlocal pump_until_s, pump_trial, pump_manual, automatic_rewards, manual_rewards
            if pump_until_s is not None:
                return
            pump_trial = None if manual or state.continuous else state.trial_num
            pump_manual = manual
            requested_ns = time.perf_counter_ns()
            outputs.write(pump_pin, True)
            actual_ns = time.perf_counter_ns()
            pump_until_s = actual_ns / 1e9 + pulse_s
            logger.log_signal(
                trial_num=pump_trial, event="pump_on", timestamp_perf_ns=actual_ns,
                requested_timestamp_perf_ns=requested_ns, requested_duration=pulse_s,
            )
            if manual:
                manual_rewards += 1
            else:
                automatic_rewards += 1
                state.reward_delivered(actual_ns / 1e9)

        mouse = MousePressTracker(event.Mouse(win=win))
        display_key = None
        preview_key = None
        cue_on_s = None
        next_flush_s = time.perf_counter() + 1.0

        def show_scene():
            nonlocal display_key, preview_key, cue_on_s
            key = (state.cue_visible, state.touching)
            if key != display_key:
                opening = state.cue_visible and (display_key is None or not display_key[0])
                background.draw()
                if state.cue_visible:
                    (pressed_cue if state.touching else cue).draw()
                if opening:
                    mouse.reset()
                timing = flip_with_timestamps(win)
                if opening:
                    cue_on_s = timing.actual_perf_s
                    logger.log_frame_flip(
                        trial_num=None if state.continuous else state.trial_num + 1,
                        event="onset_cue_on", timestamp_perf_ns=timing.actual_perf_ns,
                        requested_timestamp_perf_ns=timing.requested_perf_ns,
                    )
                elif state.phase == "iti":
                    state.start_iti(timing.actual_perf_s)
                    logger.log_frame_flip(
                        trial_num=state.trial_num, event="gray_inter_trial_interval",
                        timestamp_perf_ns=timing.actual_perf_ns,
                        requested_timestamp_perf_ns=timing.requested_perf_ns, requested_duration=state.iti,
                    )
                display_key = key
            counts = (state.completed_trials, automatic_rewards, manual_rewards)
            new_preview_key = (key, counts, state.trial_num)
            if preview is not None and new_preview_key != preview_key:
                if not state.continuous:
                    preview.set_trial_progress(
                        state.trial_num + (state.phase == "waiting"), n,
                    )
                preview.show_static_scene(
                    bg_rgb_255=bg, main_size=scene_size, main_rotation_deg=rotation,
                    images=[preview_images[int(state.touching)]] if state.cue_visible else [],
                    status_counts={"Completed trials": counts[0], "Hold rewards": counts[1],
                                   "Manual rewards": counts[2]},
                )
                preview_key = new_preview_key

        show_scene()
        status = "done"
        while state.phase != "done":
            keys = event.getKeys(keyList=["escape", "r"])
            if "escape" in keys or (preview is not None and preview.poll()):
                status = "user_exit"
                break
            manual = "r" in keys
            if preview is not None:
                manual = preview.consume_manual_reward_request() or manual
            sample = mouse.poll()
            now = sample.observed_perf_s
            was_visible, was_touching = state.cue_visible, state.touching
            previous_trial = state.trial_num
            in_cue = bool(cue.contains(sample.position))
            touching = sample.down and in_cue
            pressed = sample.press_started and in_cue
            due = state.update(
                now, touching=touching, pressed=pressed,
                press_s=max(cue_on_s, sample.timestamp_perf_s),
            )
            trial_num = None if state.continuous else state.trial_num
            if state.trial_num != previous_trial:
                logger.log_event(
                    trial_num=trial_num, event="training_trial_start", event_type="lifecycle",
                    timestamp_perf_s=state.trial_started_s, requested_duration=duration,
                )
            if was_visible and ((state.touching and not was_touching) or pressed):
                logger.log_interaction(trial_num=trial_num, event="cue_touch", timestamp_perf_ns=sample.observed_perf_ns)
            if was_visible and (
                (was_touching and (not touching or pressed)) or (pressed and not touching)
            ):
                logger.log_interaction(trial_num=trial_num, event="training_cue_release", timestamp_perf_ns=sample.observed_perf_ns)
            if was_visible and state.phase == "iti":
                logger.log_event(trial_num=trial_num, event="training_trial_end", event_type="lifecycle",
                                 timestamp_perf_ns=sample.observed_perf_ns)
                messages.log("INFO", f"trial_end trial={trial_num} reason={state.end_reason}")
            if pump_until_s is not None and (
                now >= pump_until_s or (not pump_manual and not state.touching)
            ):
                stop_pump()
            if manual:
                start_pump(manual=True)
            elif due:
                start_pump()
            show_scene()
            # Continuous mode has no between-trial gap in which to flush.
            # Never perform disk flushes while a pump output is active.
            if pump_until_s is None and time.perf_counter() >= next_flush_s:
                logs.flush()
                next_flush_s = time.perf_counter() + 1.0
            core.wait(0.001)
        messages.log("INFO", f"training_end status={status} completed_trials={state.completed_trials} "
                     f"hold_rewards={automatic_rewards} manual_rewards={manual_rewards}")
        return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--main_screen", default=None)
    parser.add_argument("--experimenter_screen", default=None)
    args = parser.parse_args()
    try:
        cfg = load_config(args.config)
        initiation = resolve_initiation_config(cfg, require_keys=False)
        daq = cfg.get("daq", {})
        status = run_task(
            colors_tsv=cfg["colors_tsv"], duration=cfg.get("duration", 5.0), iti=cfg.get("iti", 0.2),
            reward_interval=cfg.get("reward_interval", 1.0), end_on_release=cfg.get("end_on_release", True),
            n=cfg.get("n"), output_dir=cfg.get("output_dir", "logs/trainings"),
            config_name=cfg.get("config_name", "touch_and_hold"), subject=cfg.get("subject"),
            fullscreen=cfg.get("fullscreen", True), win_size=cfg.get("win_size"),
            screen_config=load_screen_config(cfg, cli_main=args.main_screen, cli_experimenter=args.experimenter_screen),
            touchscreen=cfg.get("touchscreen", False),
            initiation_cue_center_position=initiation.cue_center_position, initiation_cue_style=initiation.cue_style,
            init_dot_color=tuple(cfg.get("init_dot_color", cfg.get("dot_color", (50, 50, 50)))),
            raspi=cfg.get("raspi", False), daq_address=daq.get("address", cfg.get("daq_address", 0)),
            daq_module_name=daq.get("module_name", cfg.get("daq_module_name", "piplates.DAQC2plate")),
            pump_pin=cfg.get("pump_pin", 0),
            pump_pulse_time_seconds=cfg.get("pump_pulse_time_seconds", 0.275),
            eye_tracking_config=load_eye_tracking_settings(cfg),
        )
        return 0 if status == "done" else USER_EXIT_CODE
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
