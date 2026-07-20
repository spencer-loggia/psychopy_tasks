"""
Delayed CSC1 AFC task.

This version follows the same logging/timing style as the working tasks:
``SessionLogBundle`` creates event, message, and behavior logs; visual timing is
validated against the detected/overridden frame rate; per-trial presentation is
frame-locked and uses the active_foraging event vocabulary, plus ``delay_start``.

Trial sequence:
    onset cue click/touch -> optional pre-cue ISI -> feature cue -> delay_start -> AFC choices -> gray/IBI

For ``shape_to_color`` trials, the cue is the target shape and the choices are
color patches.  For ``color_to_shape`` trials, the cue is the target color patch
and the choices are shape outlines.  ``num_afc`` controls the number of choices;
``delay_time`` controls the blank delay between cue offset and choice onset.
Every trial is self-initiated by clicking/touching the checkerboard onset-cue stimulus
created by ``utils.make_onset_cue_stim``. The fixation cross is not drawn unless
``show_fixation`` is true.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from psychopy import core, logging as pylogging

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from bin import utils
from bin.config import load_config, validate_config
from bin.logger import SessionLogBundle
from bin.screen import (
    ExperimenterPreview,
    describe_screen,
    load_screen_config,
    resolve_scene_size,
    resolve_task_screens,
    serialize_preview_image,
    set_window_mouse_visible,
)
from interface.rig_mode import IS_RIG_ENV_VAR, experimenter_cursor_visible_for_touchscreen


def parse_args():
    p = argparse.ArgumentParser(description="Delayed CSC1 AFC task")
    p.add_argument("--config", help="Path to JSON config file. CLI overrides config keys.")
    p.add_argument("--colors_tsv", help="Path to color TSV (overrides config)")
    p.add_argument("--shapes_tsv", help="Path to shapes TSV (overrides config)")
    p.add_argument("--n", type=int, default=None, help="Number of trials/blocks")
    p.add_argument("--num_afc", type=int, default=None, help="Number of choices per trial")
    p.add_argument("--num_stim", type=int, default=None, help="Pool size for matched shape-color pairs")
    p.add_argument("--pairing_mode", choices=["matched", "all"], default=None, help="matched: shape_i-color_i pool; all: full shape x color pool")
    p.add_argument("--duration", type=float, default=None, help="Alias for cue_time when cue_time is omitted")
    p.add_argument("--cue_time", type=float, default=None, help="Cue duration in seconds")
    p.add_argument("--delay_time", type=float, default=None, help="Blank delay after cue and before choices")
    p.add_argument("--choice_time", type=float, default=None, help="Choice window duration in seconds")
    p.add_argument("--ibi", type=float, default=None, help="Inter-block interval in seconds")
    p.add_argument("--isi", type=float, default=None, help="Optional pre-cue fixation interval in seconds")
    p.add_argument("--timeout_time", type=float, default=None, help="Optional gray timeout after incorrect/no-choice trials")
    p.add_argument("--reward_pulse_time", type=float, default=None, help="Optional logged reward pulse duration after correct trials")
    p.add_argument("--reward_inter_pulse_time", type=float, default=None, help="Reserved for compatibility; currently only one logged reward pulse is emitted")
    p.add_argument("--trial_type", choices=["random", "shape_to_color", "color_to_shape"], default=None, help="Feature mapping for trials")
    p.add_argument("--shape_cue_color", type=int, nargs=3, default=None, help="Neutral RGB color for shape-only cues/choices")
    p.add_argument("--bg", type=int, nargs=3, default=None, help="Background RGB; if omitted and colors_tsv_has_bg_row=true, use the first color row")
    p.add_argument("--colors_tsv_has_bg_row", action="store_true", default=None, help="Treat first colors_tsv row as background and skip it for stimuli")
    p.add_argument("--no_colors_tsv_has_bg_row", action="store_false", dest="colors_tsv_has_bg_row", help="Use every colors_tsv row as a stimulus color")
    p.add_argument("--dot_size", type=int, default=None, help="Kept for config compatibility; not used by delayed AFC choices")
    p.add_argument("--dot_color", type=int, nargs=3, default=None, help="Kept for config compatibility; not used by delayed AFC choices")
    p.add_argument("--init_dot_color", type=int, nargs=3, default=None, help="Kept for config compatibility")
    p.add_argument("--output_dir", default=None, help="Output dir for logs")
    p.add_argument("--seed", type=int, default=None, help="Random seed")
    p.add_argument("--fullscreen", action="store_true", default=None, help="Fullscreen")
    p.add_argument("--win_size", type=int, nargs=2, default=None, help="Window size when not fullscreen")
    p.add_argument("--image_size", type=int, nargs=2, default=None, help="Raster draw size (W H)")
    p.add_argument("--likelihood_tsv", help="Optional TSV with color,shape,prob columns")
    p.add_argument("--debug", action="store_true", default=None, help="Enable debug outputs")
    p.add_argument("--self_initiation", action="store_true", default=None, help="Deprecated compatibility flag; AFC trials always require checkerboard onset-cue initiation")
    p.add_argument("--no_self_initiation", action="store_false", dest="self_initiation", help="Deprecated compatibility flag; ignored because AFC trials always require checkerboard onset-cue initiation")
    p.add_argument("--fixation_size", type=int, default=None, help="Fixation cross size in pixels when --show_fixation is enabled")
    p.add_argument("--show_fixation", action="store_true", default=None, help="Draw the fixation cross during AFC blank/cue/choice screens; default is off")
    p.add_argument("--fixed_positions", action="store_true", default=None, help="Use evenly spaced fixed positions on a circle")
    p.add_argument("--position_spacing", type=int, default=None, help="Radius in pixels for fixed positions")
    p.add_argument("--margin", type=int, default=None, help="Margin for random placement")
    p.add_argument("--choice_hitbox_scale", type=float, default=None, help="Scale choice hitboxes relative to stimulus size")
    p.add_argument("--refresh_rate", type=float, default=None, help="Override detected display refresh rate")
    p.add_argument("--touchscreen", action="store_true", default=None, help="Enable touchscreen mode and hide the main-screen cursor")
    p.add_argument("--raspi", action="store_true", default=None, help="Enable Raspberry Pi GPIO trial-start pulses")
    p.add_argument("--trial_start_pin", type=int, default=None, help="GPIO pin to use for trial start pulses (BCM numbering)")
    p.add_argument("--pump_pin", type=int, default=None, help="GPIO pin for pump reward delivery")
    p.add_argument("--buzz_pin", type=int, default=None, help="GPIO pin for timeout buzzer")    
    p.add_argument(
        "--pump_delay_time",
        type=float,
        default=None,
        help="Delay in seconds between a rewarded choice and the first pump pulse",
    )
    p.add_argument("--pump_pulse_time_seconds", type=float, default=None, help="Duration of pump pulse in seconds")
    p.add_argument(
        "--inter_pump_interval",
        type=float,
        default=None,
        help="Delay in seconds between pump pulses; defaults to pump_pulse_time_seconds",
    )    
    p.add_argument("--main_screen", default=None, help="Main task screen index or output name")
    p.add_argument("--experimenter_screen", default=None, help="Experimenter screen index or output name")
    
    return p.parse_args()


def _fmt_optional(value: Any) -> str:
    if value == "" or value is None:
        return ""
    return f"{float(value):.9f}"


def _build_behavior_fieldnames(num_afc: int) -> List[str]:
    fields = [
        "trial_num",
        "trial_type",
        "cue_feature",
        "choice_feature",
        "target_index",
        "target_shape",
        "target_color",
        "initiation_time",
    ]
    for idx in range(int(num_afc)):
        fields.extend([f"option_{idx}_shape", f"option_{idx}_color"])
    fields.extend(
        [
            "choice_made_index",
            "choice_made_shape",
            "choice_made_color",
            "is_correct",
            "choice_touch_x",
            "choice_touch_y",
            "choice_reaction_time",
        ]
    )
    return fields


def _load_likelihood(
    path: Optional[str],
    color_ids: List[int],
    shape_ids: List[int],
    msg_logger,
) -> np.ndarray:
    arr = np.ones((len(color_ids), len(shape_ids)), dtype=float)
    if path is None:
        arr /= float(arr.sum())
        msg_logger.log("WARN", "No likelihood TSV provided; using uniform distribution")
        return arr

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Likelihood TSV not found: {path}")

    arr = np.zeros((len(color_ids), len(shape_ids)), dtype=float)
    with p.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError("Likelihood TSV must have header columns: color,shape,prob")
        for row in reader:
            try:
                cid = int(row.get("color") or row.get("Color") or row.get(reader.fieldnames[0]))
                sid = int(row.get("shape") or row.get("Shape") or row.get(reader.fieldnames[1]))
                prob = float(row.get("prob") or row.get("Prob") or row.get(reader.fieldnames[2]))
            except Exception as e:
                raise ValueError(f"Invalid row in likelihood TSV: {row}") from e
            if cid not in color_ids:
                raise ValueError(f"Likelihood TSV references unknown color id: {cid}")
            if sid not in shape_ids:
                raise ValueError(f"Likelihood TSV references unknown shape id: {sid}")
            arr[color_ids.index(cid), shape_ids.index(sid)] = prob

    total = float(arr.sum())
    if total <= 0.0:
        msg_logger.log("WARN", "Likelihood TSV sums to zero; using uniform distribution")
        arr = np.ones_like(arr, dtype=float)
        arr /= float(arr.sum())
    elif not np.isclose(total, 1.0):
        msg_logger.log("WARN", f"Likelihood TSV sum is {total:.6f}; normalizing to sum to 1")
        arr /= total
    return arr


def _build_pair_pool(
    shape_ids: List[int],
    color_ids: List[int],
    *,
    pairing_mode: str,
    num_stim: Optional[int],
) -> List[Tuple[int, int]]:
    color_ids = sorted(color_ids)
    if pairing_mode == "matched":
        max_pairs = min(len(shape_ids), len(color_ids))
        pool_n = max_pairs if num_stim is None else int(num_stim)
        if pool_n < 1:
            raise ValueError("num_stim must be >= 1 when pairing_mode='matched'")
        if pool_n > max_pairs:
            raise ValueError(
                f"num_stim={pool_n} but only {max_pairs} matched shape-color pairs are available"
            )
        return [(int(shape_ids[i]), int(color_ids[i])) for i in range(pool_n)]

    if pairing_mode == "all":
        return [(int(sid), int(cid)) for sid in shape_ids for cid in color_ids]

    raise ValueError("pairing_mode must be 'matched' or 'all'")


def _pair_probabilities(
    all_pairs: List[Tuple[int, int]],
    likelihood: np.ndarray,
    color_ids: List[int],
    shape_ids: List[int],
    msg_logger,
) -> np.ndarray:
    color_to_idx = {int(cid): idx for idx, cid in enumerate(color_ids)}
    shape_to_idx = {int(sid): idx for idx, sid in enumerate(shape_ids)}
    probs = np.array(
        [likelihood[color_to_idx[int(cid)], shape_to_idx[int(sid)]] for sid, cid in all_pairs],
        dtype=float,
    )
    probs = np.clip(probs, 0.0, None)
    if probs.sum() <= 0.0:
        msg_logger.log("WARN", "All pair probabilities are zero; using uniform pair sampling")
        probs = np.ones(len(all_pairs), dtype=float)
    probs /= float(probs.sum())
    return probs


def _sample_pair_block(
    *,
    all_pairs: List[Tuple[int, int]],
    probs: np.ndarray,
    num_afc: int,
    trial_type: str,
    rng: np.random.Generator,
    msg_logger,
) -> List[Tuple[int, int]]:
    if num_afc > len(all_pairs):
        raise ValueError("num_afc cannot be larger than the number of available pairs")

    unique_feature_idx = 1 if trial_type == "shape_to_color" else 0
    replace = int(np.count_nonzero(probs)) < int(num_afc)

    for _attempt in range(1000):
        picks = rng.choice(len(all_pairs), size=num_afc, replace=replace, p=probs)
        block = [all_pairs[int(i)] for i in picks]
        features = [pair[unique_feature_idx] for pair in block]
        if len(set(features)) == len(features):
            return block

    msg_logger.log(
        "WARN",
        (
            f"Could not sample {num_afc} choices with unique "
            f"{'colors' if trial_type == 'shape_to_color' else 'shapes'}; using last sampled block"
        ),
    )
    return block


def _fixed_circle_positions(num_afc: int, radius_px: float) -> List[Tuple[float, float]]:
    if num_afc < 1:
        raise ValueError("num_afc must be >= 1")
    spacing = (2.0 * math.pi) / float(num_afc)
    start = (math.pi / 2.0) + (spacing / 2.0)
    return [
        (float(radius_px) * math.cos(start + idx * spacing), float(radius_px) * math.sin(start + idx * spacing))
        for idx in range(num_afc)
    ]


def _compute_positions(
    *,
    num_afc: int,
    stim_size: Tuple[int, int],
    effective_win_size: Tuple[int, int],
    fixed_positions: bool,
    position_spacing: Optional[int],
    margin: int,
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    if fixed_positions:
        radius = float(position_spacing) if position_spacing is not None else min(effective_win_size) * 0.25
        sampled_positions = _fixed_circle_positions(num_afc, radius)
    else:
        sampled_positions = utils.sample_non_overlapping_positions(
            num_afc,
            stim_size,
            effective_win_size,
            margin=margin,
        )
    positions = utils.clamp_positions(sampled_positions, stim_size, effective_win_size, margin=margin)
    return sampled_positions, positions


def _log_global_timing(msg_logger, fps: float, frame_dur: float, timings: Dict[str, float]) -> Dict[str, Tuple[int, float]]:
    quantized: Dict[str, Tuple[int, float]] = {}

    def q(label: str, seconds: float, at_least_one: bool = False):
        frames = int(round(max(0.0, float(seconds)) * float(fps)))
        if at_least_one:
            frames = max(1, frames)
        quantized[label] = (frames, frames / float(fps))

    q("cue_time", timings["cue_time"], at_least_one=True)
    q("delay_time", timings["delay_time"], at_least_one=False)
    q("isi", timings["isi"], at_least_one=False)
    q("choice_time", timings["choice_time"], at_least_one=True)
    q("ibi", timings["ibi"], at_least_one=False)
    q("timeout_time", timings["timeout_time"], at_least_one=False)

    msg_logger.log(
        "INFO",
        (
            f"timing_quantization_global fps={fps:.6f} frame_dur_s={frame_dur:.9f} "
            + " ".join(
                f"{label}={float(timings[label]):.6f}s-> {frames}fr({seconds:.6f}s)"
                for label, (frames, seconds) in quantized.items()
            )
        ),
    )
    return quantized


def run_task(
    *,
    colors_tsv: str,
    shapes_tsv: str,
    n_blocks: int,
    num_afc: int,
    cue_time: float,
    delay_time: float,
    choice_time: float,
    ibi: float,
    output_dir: str,
    isi: float = 0.0,
    bg: Optional[Tuple[int, int, int]] = None,
    colors_tsv_has_bg_row: bool = True,
    pairing_mode: str = "matched",
    num_stim: Optional[int] = None,
    image_size: Optional[Tuple[int, int]] = None,
    shape_cue_color: Tuple[int, int, int] = (0, 0, 0),
    seed: Optional[int] = None,
    fullscreen: bool = False,
    win_size: Optional[Tuple[int, int]] = None,
    margin: int = 50,
    debug: bool = False,
    likelihood_tsv: Optional[str] = None,
    self_initiation: bool = True,  # kept for config compatibility; AFC always uses checkerboard initiation
    fixation_size: Optional[int] = None,
    show_fixation: bool = False,
    refresh_rate: Optional[float] = None,
    touchscreen: bool = False,
    raspi: bool = False,
    trial_start_pin: int = 18,
    pump_pin: int = 17, 
    buzz_pin: int = 16, 
    buzz: bool = False,
    fixed_positions: bool = False,
    position_spacing: Optional[int] = None,
    trial_type: str = "random",
    timeout_time: float = 3.0,
    reward_pulse_time: float = 0.0,
    reward_inter_pulse_time: float = 0.0,
    pump_delay_time: float = 0.0,
    pump_pulse_time_seconds: Optional[float] = None,
    inter_pump_interval: Optional[float] = None,
    choice_hitbox_scale: float = 1.0,
    config_name: Optional[str] = None,
    screen_config: Optional[Dict[str, Any]] = None,
) -> None:
    utils.set_debug(debug)
    if seed is not None:
        random.seed(int(seed))
    rng = np.random.default_rng(seed)

    if image_size is None:
        raise ValueError("image_size is required for SVG rasterization")
    image_size = (int(image_size[0]), int(image_size[1]))

    colors_all = utils.load_color_palette(Path(colors_tsv))
    if colors_tsv_has_bg_row:
        bg_from_palette, colors = utils.split_background_from_palette(colors_all)
        bg_rgb = tuple(bg_from_palette if bg is None else bg)
    else:
        colors = colors_all
        bg_rgb = tuple((128, 128, 128) if bg is None else bg)

    shapes = utils.load_shape_definitions(Path(shapes_tsv))
    color_ids = list(colors.keys())
    shape_ids = list(shapes.keys())

    if num_afc < 1:
        raise ValueError("num_afc must be >= 1")
    if n_blocks < 1:
        raise ValueError("n must be >= 1")

    resolved_config_name = str(config_name).strip() if config_name else "afc_csc1"
    behavior_fieldnames = _build_behavior_fieldnames(num_afc)
    session_logs = SessionLogBundle(
        output_root=output_dir,
        task_name="afc_csc1",
        config_name=resolved_config_name,
        behavior_fieldnames=behavior_fieldnames,
        auto_flush=False,
    )
    logger = session_logs.event_logger
    msg_logger = session_logs.message_logger
    behavior_logger = session_logs.behavior_logger
    if behavior_logger is None:
        raise RuntimeError("afc_csc1 requires a behavior logger")

    msg_logger.log(
        "INFO",
        (
            f"session_start task=afc_csc1 config_name={resolved_config_name} "
            f"session_dir={session_logs.session_dir}"
        ),
    )

    all_pairs = _build_pair_pool(
        shape_ids,
        color_ids,
        pairing_mode=pairing_mode,
        num_stim=num_stim,
    )
    if num_afc > len(all_pairs):
        raise ValueError("num_afc cannot be larger than the number of available color-shape pairs")
    msg_logger.log(
        "INFO",
        (
            f"stimulus_pool pairing_mode={pairing_mode} pool_size={len(all_pairs)} "
            f"n_shapes={len(shape_ids)} n_colors={len(color_ids)} num_afc={num_afc}"
        ),
    )

    likelihood = _load_likelihood(likelihood_tsv, color_ids, shape_ids, msg_logger)
    pair_probs = _pair_probabilities(all_pairs, likelihood, color_ids, shape_ids, msg_logger)

    # Pre-render only the feature-level images used by this delayed AFC task.
    preloaded: Dict[Any, Any] = {}
    for sid in shape_ids:
        preloaded[("shape_only", int(sid))] = utils.rasterize_svg_with_color(
            shapes[int(sid)],
            size_px=image_size,
            color_rgb_255=tuple(shape_cue_color),
            bg_rgb_255=None,
            stroke_rgb_255=tuple(shape_cue_color),
            stroke_width_px=2.0,
            outline_only=True,
            flip=True
        )
    for cid in color_ids:
        preloaded[("color_only", int(cid))] = utils.make_color_gaussian_image(
            color_rgb_255=colors[int(cid)],
            size_px=image_size,
        )

    win = None
    pigpio_pi = None
    experimenter_preview = None
    task_end_notes = "done"
    reward_pulse_s = float(pump_pulse_time_seconds) if pump_pulse_time_seconds is not None else float(reward_pulse_time)
    pump_delay_s = max(0.0, float(pump_delay_time))

    main_screen, experimenter_screen = resolve_task_screens(screen_config, allow_same_screen=True)
    try:
        same_screen = (
            experimenter_screen is not None
            and describe_screen(experimenter_screen) == describe_screen(main_screen)
        )
    except Exception:
        same_screen = False
    if same_screen and not bool(touchscreen):
        experimenter_screen = None
        try:
            msg_logger.log(
                "INFO",
                "experimenter_preview_disabled same_screen=True touchscreen=False",
            )
        except Exception:
            pass
    try:
        msg_logger.log(
            "INFO",
            f"resolved_screens main={describe_screen(main_screen)} experimenter={describe_screen(experimenter_screen)}",
        )
    except Exception:
        pass

    afc_counts: Dict[str, int] = {
        "total": 0,
        "correct": 0,
        "incorrect": 0,
        "no_choice": 0,
        "s2c": 0,
        "c2s": 0,
    }
    last_trial_summary: Dict[str, Any] = {}

    try:
        win = utils.setup_window(
            bg_rgb_255=bg_rgb,
            fullscreen=fullscreen,
            size=win_size,
            screen_info=main_screen,
        )

        is_rig_raw = os.environ.get(IS_RIG_ENV_VAR)
        experimenter_mouse_visible = experimenter_cursor_visible_for_touchscreen(
            touchscreen=bool(touchscreen),
            is_rig=is_rig_raw,
        )
        if experimenter_screen is not None:
            experimenter_preview = ExperimenterPreview(
                experimenter_screen,
                task_label=resolved_config_name,
                start_perf_s=time.perf_counter(),
                update_interval_s=0.1,
                mouse_visible=experimenter_mouse_visible,
            )
        if touchscreen:
            set_window_mouse_visible(win, False)
            try:
                exp_cursor_state = "none"
                if experimenter_preview is not None:
                    exp_cursor_state = "visible" if experimenter_mouse_visible else "hidden"
                msg_logger.log(
                    "INFO",
                    (
                        "touchscreen=True; main mouse cursor hidden "
                        f"experimenter_cursor={exp_cursor_state} "
                        f"{IS_RIG_ENV_VAR}={is_rig_raw if is_rig_raw is not None else 'unset'}"
                    ),
                )
            except Exception:
                pass

        if fixation_size is None:
            fixation_size = 32
        fix = utils.make_fixation_cross(win, size=int(fixation_size)) if bool(show_fixation) else None
        bg_rect = utils.make_bg_rect(win, bg_rgb)
        pylogging.console.setLevel(pylogging.CRITICAL)

        main_scene_size = resolve_scene_size(
            main_screen,
            fullscreen=bool(fullscreen),
            requested_size=win_size,
            realized_size=tuple(win.size),
        )
        try:
            msg_logger.log(
                "INFO",
                (
                    f"resolved_main_scene_size size={main_scene_size[0]}x{main_scene_size[1]} "
                    f"fullscreen={int(bool(fullscreen))} requested_win_size={win_size} "
                    f"realized_win_size={tuple(win.size)}"
                ),
            )
        except Exception:
            pass

        def _feature_key(pair: Tuple[int, int], feature: str) -> Tuple[str, int]:
            sid, cid = pair
            if feature == "shape":
                return ("shape_only", int(sid))
            if feature == "color":
                return ("color_only", int(cid))
            raise ValueError(f"Unknown AFC feature: {feature}")

        def _choice_mapping_for_preview(ttype: str) -> Tuple[str, str]:
            if ttype == "shape_to_color":
                return "shape", "color"
            if ttype == "color_to_shape":
                return "color", "shape"
            raise ValueError(f"Unknown AFC trial_type: {ttype}")

        def _make_stats_panel_image(*, phase: str = "idle"):
            from PIL import Image, ImageDraw, ImageFont

            panel_w, panel_h = 420, 285
            img = Image.new("RGBA", (panel_w, panel_h), (245, 245, 245, 235))
            draw = ImageDraw.Draw(img)
            try:
                font_title = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
                font_body = ImageFont.truetype("DejaVuSans.ttf", 18)
                font_small = ImageFont.truetype("DejaVuSans.ttf", 15)
            except Exception:
                font_title = ImageFont.load_default()
                font_body = ImageFont.load_default()
                font_small = ImageFont.load_default()

            total = int(afc_counts.get("total", 0))
            planned = int(n_blocks)
            remaining = max(0, planned - total)
            correct = int(afc_counts.get("correct", 0))
            incorrect = int(afc_counts.get("incorrect", 0))
            no_choice = int(afc_counts.get("no_choice", 0))
            pct = (100.0 * correct / total) if total > 0 else 0.0
            y = 14
            draw.text((16, y), "AFC CSC1", fill=(0, 0, 0, 255), font=font_title)
            y += 34
            draw.text((16, y), f"phase: {phase}", fill=(40, 40, 40, 255), font=font_small)
            y += 30
            draw.text((16, y), f"trials: {total}/{planned}   remaining: {remaining}", fill=(0, 0, 0, 255), font=font_body)
            y += 24
            draw.text((16, y), f"correct: {correct}   incorrect: {incorrect}", fill=(0, 0, 0, 255), font=font_body)
            y += 24
            draw.text((16, y), f"accuracy: {pct:.1f}%   no choice: {no_choice}", fill=(0, 0, 0, 255), font=font_body)
            y += 24
            draw.text((16, y), f"S2C: {afc_counts.get('s2c', 0)}   C2S: {afc_counts.get('c2s', 0)}", fill=(0, 0, 0, 255), font=font_body)
            y += 31
            if last_trial_summary:
                draw.text((16, y), "last trial", fill=(0, 0, 0, 255), font=font_title)
                y += 28
                draw.text(
                    (16, y),
                    f"#{last_trial_summary.get('trial_num', '')} {last_trial_summary.get('trial_type', '')}",
                    fill=(40, 40, 40, 255),
                    font=font_small,
                )
                y += 22
                draw.text(
                    (16, y),
                    f"target: {last_trial_summary.get('target_index', '')}  chosen: {last_trial_summary.get('chosen_index', '')}",
                    fill=(40, 40, 40, 255),
                    font=font_small,
                )
                y += 22
                draw.text(
                    (16, y),
                    f"result: {last_trial_summary.get('result', '')}",
                    fill=(40, 40, 40, 255),
                    font=font_small,
                )
            else:
                draw.text((16, y), "last trial: none yet", fill=(40, 40, 40, 255), font=font_small)
            return img

        def _status_panel_item(phase: str) -> Optional[Dict[str, Any]]:
            if experimenter_preview is None:
                return None
            panel_img = _make_stats_panel_image(phase=phase)
            payload = serialize_preview_image(panel_img)
            if payload is None:
                return None
            panel_w, panel_h = panel_img.size
            return {
                "image_payload": payload,
                "pos": [
                    float(-main_scene_size[0] / 2.0 + panel_w / 2.0 + 20.0),
                    float(main_scene_size[1] / 2.0 - panel_h / 2.0 - 20.0),
                ],
                "size": [float(panel_w), float(panel_h)],
            }

        def _show_preview_afc_scene(
            *,
            phase: str,
            block_paths_current: Optional[List[Tuple[int, int]]] = None,
            positions_current: Optional[List[Tuple[float, float]]] = None,
            trial_type_current: Optional[str] = None,
            target_index_current: Optional[int] = None,
            chosen_index_current: Optional[int] = None,
            is_correct_current: Optional[bool] = None,
        ) -> None:
            if experimenter_preview is None:
                return

            preview_fixation_size = int(getattr(fix, "height", 0)) if fix is not None else None
            images: List[Dict[str, Any]] = []
            panel = _status_panel_item(phase)
            if panel is not None:
                images.append(panel)

            highlight_box = None
            if block_paths_current and positions_current and trial_type_current:
                cue_feature, choice_feature = _choice_mapping_for_preview(trial_type_current)
                if target_index_current is not None:
                    try:
                        target_pair = tuple(block_paths_current[int(target_index_current) - 1])
                        cue_obj = preloaded.get(_feature_key(target_pair, cue_feature))
                        cue_payload = serialize_preview_image(cue_obj) if cue_obj is not None else None
                        if cue_payload is not None:
                            cue_w, cue_h = tuple(image_size)
                            images.append(
                                {
                                    "image_payload": cue_payload,
                                    "pos": [0.0, 0.0],
                                    "size": [float(cue_w), float(cue_h)],
                                }
                            )
                    except Exception:
                        pass

                for idx, (pair, pos) in enumerate(zip(block_paths_current, positions_current), start=1):
                    pair = tuple(pair)
                    image_obj = preloaded.get(_feature_key(pair, choice_feature))
                    payload = serialize_preview_image(image_obj) if image_obj is not None else None
                    if payload is None:
                        continue
                    images.append(
                        {
                            "image_payload": payload,
                            "pos": [float(pos[0]), float(pos[1])],
                            "size": [float(image_size[0]), float(image_size[1])],
                        }
                    )

                highlight_index = chosen_index_current or target_index_current
                if highlight_index is not None and 1 <= int(highlight_index) <= len(positions_current):
                    hpos = positions_current[int(highlight_index) - 1]
                    if chosen_index_current is None:
                        color = (240, 220, 60)
                    elif bool(is_correct_current):
                        color = (60, 180, 75)
                    else:
                        color = (220, 60, 60)
                    highlight_box = {
                        "pos": [float(hpos[0]), float(hpos[1])],
                        "size": [float(image_size[0]) * 1.15, float(image_size[1]) * 1.15],
                        "color": list(color),
                        "line_width": 6.0,
                    }

            experimenter_preview.show_static_scene(
                bg_rgb_255=bg_rgb,
                main_size=main_scene_size,
                images=images,
                dots=[],
                fixation_size=preview_fixation_size,
                fixation_color=(0, 0, 0),
                reward_counts={},
                highlight_box=highlight_box,
            )

        def _wait_or_abort(duration_s: float) -> bool:
            if duration_s is None or float(duration_s) <= 0:
                return _poll_experimenter_controls()
            if experimenter_preview is not None:
                deadline = time.perf_counter() + max(0.0, float(duration_s))
                while time.perf_counter() < deadline:
                    if _poll_experimenter_controls():
                        return True
                    remaining = deadline - time.perf_counter()
                    if remaining > 0:
                        time.sleep(min(0.05, remaining))
                return _poll_experimenter_controls()
            core.wait(float(duration_s))
            return False

        _show_preview_afc_scene(phase="ready")

        gpio_chip = None
        if raspi:
            try:
                import lgpio

                chip = lgpio.gpiochip_open(0)
                lgpio.gpio_claim_output(chip, int(trial_start_pin))
                lgpio.gpio_claim_output(chip, int(pump_pin))
                lgpio.gpio_claim_output(chip, int(buzz_pin))
                pigpio_pi = chip
                gpio_chip = chip
                msg_logger.log(
                    "INFO",
                    (
                        "lgpio initialized on chip 0, pins claimed: "
                        f"trial_start={trial_start_pin}, pump={pump_pin}, buzz={buzz_pin}"
                    ),
                )
            except Exception as e:
                pigpio_pi = None
                gpio_chip = None
                try:
                    msg_logger.log("WARN", f"lgpio not available or failed to initialize: {e}; raspi disabled")
                except Exception:
                    pass
        else:
            try:
                msg_logger.log("INFO", "raspi=False; GPIO pin signals will not be sent (events will be logged only)")
            except Exception:
                pass

        def _set_pump_pin(value: int, *, context: str) -> None:
            if raspi and gpio_chip is not None:
                try:
                    import lgpio
                    lgpio.gpio_write(gpio_chip, int(pump_pin), int(value))
                except Exception as e:
                    try:
                        msg_logger.log(
                            "ERROR",
                            f"Failed to set pump_pin {'high' if value else 'low'} during {context}: {e}",
                        )
                    except Exception:
                        pass

        def _deliver_manual_reward() -> None:
            pulse_duration = max(0.0, float(reward_pulse_s))
            if pulse_duration <= 0.0:
                try:
                    msg_logger.log("INFO", "manual_reward_request_ignored reason=pulse_duration_zero")
                except Exception:
                    pass
                return
            start_perf = time.perf_counter()
            _set_pump_pin(1, context="manual_reward")
            try:
                logger.log_signal(
                    trial_num=None,
                    event="pump_on",
                    timestamp_perf_s=start_perf,
                    requested_duration=pulse_duration,
                )
                core.wait(pulse_duration)
            finally:
                end_perf = time.perf_counter()
                _set_pump_pin(0, context="manual_reward")
                logger.log_signal(
                    trial_num=None,
                    event="pump_off",
                    timestamp_perf_s=end_perf,
                )

        # Keep exactly one experimenter-control poller. It does not treat poll errors
        # as exits, and it supports manual reward only from the experimenter screen.
        def _poll_experimenter_controls() -> bool:
            if experimenter_preview is None:
                return False
            try:
                if hasattr(experimenter_preview, "consume_manual_reward_request"):
                    if experimenter_preview.consume_manual_reward_request():
                        _deliver_manual_reward()
            except Exception as e:
                try:
                    msg_logger.log("WARN", f"experimenter_manual_reward_poll_failed: {e}")
                except Exception:
                    pass
            try:
                return bool(experimenter_preview.poll())
            except Exception as e:
                try:
                    msg_logger.log("WARN", f"experimenter_preview_poll_failed: {e}")
                except Exception:
                    pass
                return False

        if refresh_rate is not None and float(refresh_rate) > 0:
            fps = float(refresh_rate)
            frame_dur = 1.0 / fps
            msg_logger.log("INFO", f"fps_override refresh_rate={fps:.6f}Hz frame_dur_s={frame_dur:.9f}")
        else:
            fps, frame_dur = utils.detect_frame_rate(win, msg_logger=msg_logger)

        utils.validate_frame_aligned_timings(
            fps,
            {
                "cue_time": cue_time,
                "delay_time": delay_time,
                "isi": isi,
                "choice_time": choice_time,
                "ibi": ibi,
                "timeout_time": timeout_time,
            },
            context="afc_csc1",
            minimum_frames={"cue_time": 1, "choice_time": 1},
            msg_logger=msg_logger,
        )
        quantized = _log_global_timing(
            msg_logger,
            fps,
            frame_dur,
            {
                "cue_time": cue_time,
                "delay_time": delay_time,
                "isi": isi,
                "choice_time": choice_time,
                "ibi": ibi,
                "timeout_time": timeout_time,
            },
        )
        ibi_frames, ibi_s = quantized["ibi"]
        timeout_frames, timeout_s = quantized["timeout_time"]

        onset_stim = utils.make_onset_cue_stim(
            win,
            bg_rgb_255=bg_rgb,
            size_frac=0.075,
            cells=8,
            sigma_frac=0.22,
            zero_threshold=1,
        )
        msg_logger.log(
            "INFO",
            "checkerboard_onset_cue_required source=utils.make_onset_cue_stim "
            f"size={tuple(getattr(onset_stim, 'size', ())) if onset_stim is not None else ''}",
        )

        msg_logger.log("INFO", f"task_ready n_blocks={n_blocks} num_afc={num_afc} delay_time={delay_time:.6f}")

        for trial_num in range(1, int(n_blocks) + 1):
            if _poll_experimenter_controls():
                task_end_notes = "experimenter_exit"
                msg_logger.log("WARN", "experimenter_exit_before_trial_start")
                break

            if trial_type == "random":
                block_trial_type = random.choice(["shape_to_color", "color_to_shape"])
            else:
                block_trial_type = trial_type

            block_paths = _sample_pair_block(
                all_pairs=all_pairs,
                probs=pair_probs,
                num_afc=int(num_afc),
                trial_type=block_trial_type,
                rng=rng,
                msg_logger=msg_logger,
            )
            target_index = int(rng.integers(1, int(num_afc) + 1))
            target_sid, target_cid = block_paths[target_index - 1]
            msg_logger.log(
                "INFO",
                (
                    f"trial_loaded trial_num={trial_num} trial_type={block_trial_type} "
                    f"target_index={target_index} target_shape={target_sid} target_color={target_cid} "
                    f"options={block_paths}"
                ),
            )

            stim_size = tuple(image_size)
            effective_win_size = resolve_scene_size(
                main_screen,
                fullscreen=bool(fullscreen),
                requested_size=win_size,
                realized_size=tuple(win.size),
            )
            sampled_positions, positions = _compute_positions(
                num_afc=int(num_afc),
                stim_size=stim_size,
                effective_win_size=effective_win_size,
                fixed_positions=bool(fixed_positions),
                position_spacing=position_spacing,
                margin=int(margin),
            )
            for idx, (sampled, pos) in enumerate(zip(sampled_positions, positions), start=1):
                msg_logger.log(
                    "INFO",
                    f"position_assigned trial_num={trial_num} idx={idx} sampled={sampled} clamped={pos}",
                )

            trial_meta: Dict[str, Any] = {}
            _show_preview_afc_scene(
                phase="awaiting initiation",
                block_paths_current=block_paths,
                positions_current=positions,
                trial_type_current=block_trial_type,
                target_index_current=target_index,
            )
            aborted, choice_info = utils.present_delayed_afc_trial(
                win=win,
                preloaded=preloaded,
                block_paths=block_paths,
                positions=positions,
                cue_time=float(cue_time),
                delay_time=float(delay_time),
                choice_time=float(choice_time),
                bg_rect=bg_rect,
                fix=fix,
                logger=logger,
                block_idx=trial_num,
                target_index=target_index,
                trial_type=block_trial_type,
                isi=float(isi),
                bg_rgb_255=bg_rgb,
                onset_cue=onset_stim,
                msg_logger=msg_logger,
                fps=fps,
                choice_hitbox_scale=(float(choice_hitbox_scale) * (1.25 if touchscreen else 1.0)),
                trial_meta=trial_meta,
                raspi=bool(raspi and pigpio_pi is not None),
                pigpio_pi=pigpio_pi,
                raspi_pin=trial_start_pin,
                external_abort_checker=_poll_experimenter_controls,
                show_fixation=bool(show_fixation),
            )
            if aborted:
                if task_end_notes == "done" and _poll_experimenter_controls():
                    task_end_notes = "experimenter_exit"
                    msg_logger.log("WARN", f"experimenter_exit_during_trial trial_num={trial_num}")
                elif task_end_notes == "done":
                    task_end_notes = "aborted"
                    msg_logger.log("WARN", f"trial_aborted trial_num={trial_num}")
                break

            chosen_idx_1based = choice_info.get("chosen_index") if choice_info is not None else None
            chosen_idx_zero_based = int(chosen_idx_1based - 1) if chosen_idx_1based is not None else ""
            is_correct = bool(choice_info.get("is_correct")) if choice_info is not None else False

            afc_counts["total"] += 1
            if block_trial_type == "shape_to_color":
                afc_counts["s2c"] += 1
            elif block_trial_type == "color_to_shape":
                afc_counts["c2s"] += 1
            if is_correct:
                afc_counts["correct"] += 1
                trial_result = "correct"
            else:
                afc_counts["incorrect"] += 1
                if choice_info is None:
                    afc_counts["no_choice"] += 1
                    trial_result = "no_choice"
                else:
                    trial_result = "incorrect"
            last_trial_summary.clear()
            last_trial_summary.update(
                {
                    "trial_num": int(trial_num),
                    "trial_type": block_trial_type,
                    "target_index": int(target_index - 1),
                    "chosen_index": chosen_idx_zero_based if chosen_idx_zero_based != "" else "",
                    "result": trial_result,
                }
            )
            _show_preview_afc_scene(
                phase="feedback",
                block_paths_current=block_paths,
                positions_current=positions,
                trial_type_current=block_trial_type,
                target_index_current=target_index,
                chosen_index_current=chosen_idx_1based,
                is_correct_current=is_correct,
            )

            feedback_s = 0.0
            if is_correct and float(reward_pulse_s) > 0.0:
                feedback_s += float(reward_pulse_s)
            if (not is_correct) and float(timeout_s) > 0.0:
                feedback_s += float(timeout_s)

            gray_start_perf = trial_meta.get("gray_flip_perf_s")
            if gray_start_perf is not None:
                logger.log_frame_flip(
                    trial_num=trial_num,
                    event="gray_inter_trial_interval",
                    timestamp_perf_s=float(gray_start_perf),
                    requested_duration=(float(ibi_s) + float(feedback_s)) if (ibi_s + feedback_s) > 0 else None,
                )

            if is_correct and float(reward_pulse_s) > 0.0:
                pulse_duration = max(0.0, float(reward_pulse_s))
                if pump_delay_s > 0.0:
                    if _wait_or_abort(pump_delay_s):
                        task_end_notes = "experimenter_exit"
                        msg_logger.log("WARN", f"experimenter_exit_during_pump_delay trial_num={trial_num}")
                        break

                pulse_start = time.perf_counter()
                _set_pump_pin(1, context="afc_reward")
                aborted_reward = False
                try:
                    logger.log_signal(
                        trial_num=trial_num,
                        event="pump_on",
                        timestamp_perf_s=pulse_start,
                        requested_duration=pulse_duration,
                    )
                    aborted_reward = _wait_or_abort(pulse_duration)
                finally:
                    pulse_end = time.perf_counter()
                    _set_pump_pin(0, context="afc_reward")
                    logger.log_signal(
                        trial_num=trial_num,
                        event="pump_off",
                        timestamp_perf_s=pulse_end,
                    )
                if aborted_reward:
                    task_end_notes = "experimenter_exit"
                    msg_logger.log("WARN", f"experimenter_exit_during_reward_pulse trial_num={trial_num}")
                    break
            elif (not is_correct) and timeout_frames > 0:
                # Silent timeout (no signal logging)
                msg_logger.log(
                    "INFO",
                    f"timeout trial_num={trial_num} duration={float(timeout_s):.3f}s",
                )
                if _wait_or_abort(float(timeout_s)):
                    task_end_notes = "experimenter_exit"
                    msg_logger.log("WARN", f"experimenter_exit_during_timeout trial_num={trial_num}")
                    break

            behavior_row: Dict[str, Any] = {
                "trial_num": int(trial_num),
                "trial_type": block_trial_type,
                "cue_feature": trial_meta.get("cue_feature", ""),
                "choice_feature": trial_meta.get("choice_feature", ""),
                "target_index": int(target_index - 1),
                "target_shape": int(target_sid),
                "target_color": int(target_cid),
                "initiation_time": _fmt_optional(trial_meta.get("initiation_time_s")),
                "choice_made_index": chosen_idx_zero_based,
                "choice_made_shape": "",
                "choice_made_color": "",
                "is_correct": int(is_correct),
                "choice_touch_x": _fmt_optional(choice_info.get("touch_x") if choice_info is not None else ""),
                "choice_touch_y": _fmt_optional(choice_info.get("touch_y") if choice_info is not None else ""),
                "choice_reaction_time": _fmt_optional(choice_info.get("reaction_time_s") if choice_info is not None else ""),
            }
            for idx, (sid, cid) in enumerate(block_paths):
                behavior_row[f"option_{idx}_shape"] = int(sid)
                behavior_row[f"option_{idx}_color"] = int(cid)
            if chosen_idx_1based is not None:
                chosen_sid, chosen_cid = block_paths[int(chosen_idx_1based) - 1]
                behavior_row["choice_made_shape"] = int(chosen_sid)
                behavior_row["choice_made_color"] = int(chosen_cid)
            behavior_logger.writerow(behavior_row)

            if ibi_frames > 0:
                _show_preview_afc_scene(phase="inter-trial interval")
                for _ in range(max(0, int(ibi_frames) - 1)):
                    if _poll_experimenter_controls():
                        task_end_notes = "experimenter_exit"
                        msg_logger.log("WARN", f"experimenter_exit_during_ibi trial_num={trial_num}")
                        break
                    bg_rect.draw()
                    if fix is not None:
                        fix.draw()
                    win.flip()
                if task_end_notes != "done":
                    break
            else:
                _show_preview_afc_scene(phase="ready")

            try:
                session_logs.flush()
            except Exception:
                pass

    finally:
        msg = f"session_end status={task_end_notes}"
        try:
            msg_logger.log("INFO", msg)
        except Exception:
            pass
        try:
            session_logs.close()
        except Exception:
            pass
        try:
            if experimenter_preview is not None:
                experimenter_preview.close()
        except Exception:
            pass
        try:
            if win is not None:
                win.close()
        except Exception:
            pass
        # Do not call core.quit() here; it raises SystemExit and can mask real exceptions.


def main():
    args = parse_args()
    cfg: Dict[str, Any] = {}
    if args.config:
        cfg = load_config(args.config)
        validate_config(cfg, required=["config_name", "colors_tsv", "shapes_tsv", "n"], allow_zero_duration=True)
    else:
        missing = []
        if not args.colors_tsv:
            missing.append("--colors_tsv or config")
        if not args.shapes_tsv:
            missing.append("--shapes_tsv or config")
        if args.n is None:
            missing.append("--n or config")
        if args.duration is None and args.cue_time is None:
            missing.append("--cue_time/--duration or config")
        if missing:
            print(f"ERROR: missing required args: {', '.join(missing)}", file=sys.stderr)
            sys.exit(2)

    def _get(name, default=None):
        val = getattr(args, name, None)
        if val is not None:
            return val
        return cfg.get(name, default)

    screen_config = load_screen_config(
        cfg,
        cli_main=args.main_screen,
        cli_experimenter=args.experimenter_screen,
    )

    duration_raw = _get("duration", cfg.get("duration", None))
    cue_time_raw = _get("cue_time", cfg.get("cue_time", duration_raw))
    if cue_time_raw is None:
        print("ERROR: missing required args: --cue_time/--duration or config", file=sys.stderr)
        sys.exit(2)

    bg_val = _get("bg", cfg.get("bg", None))
    bg = tuple(bg_val) if bg_val is not None else None

    colors_bg_default = cfg.get("colors_tsv_has_bg_row", True)
    colors_tsv_has_bg_row = bool(_get("colors_tsv_has_bg_row", colors_bg_default))

    image_size_val = _get("image_size", cfg.get("image_size", None))
    image_size = tuple(image_size_val) if image_size_val is not None else None
    win_size_val = _get("win_size", cfg.get("win_size", None))
    win_size = tuple(win_size_val) if win_size_val is not None else None
    num_stim_val = _get("num_stim", cfg.get("num_stim", None))
    num_stim = int(num_stim_val) if num_stim_val is not None else None
    pos_spacing_val = _get("position_spacing", cfg.get("position_spacing", None))
    position_spacing = int(pos_spacing_val) if pos_spacing_val is not None else None

    trial_type = _get("trial_type", cfg.get("trial_type", "random"))
    if trial_type not in ("random", "shape_to_color", "color_to_shape"):
        raise ValueError("trial_type must be one of: random, shape_to_color, color_to_shape")

    try:
        run_task(
            colors_tsv=_get("colors_tsv", cfg.get("colors_tsv")),
            shapes_tsv=_get("shapes_tsv", cfg.get("shapes_tsv")),
            n_blocks=int(_get("n", cfg.get("n"))),
            num_afc=int(_get("num_afc", cfg.get("num_afc", 4))),
            cue_time=float(cue_time_raw),
            delay_time=float(_get("delay_time", cfg.get("delay_time", 0.5))),
            choice_time=float(_get("choice_time", cfg.get("choice_time", 0.75))),
            ibi=float(_get("ibi", cfg.get("ibi", 1.0))),
            output_dir=_get("output_dir", cfg.get("output_dir", "./logs")),
            isi=float(_get("isi", cfg.get("isi", 0.0))),
            bg=bg,
            colors_tsv_has_bg_row=colors_tsv_has_bg_row,
            pairing_mode=_get("pairing_mode", cfg.get("pairing_mode", "matched")),
            num_stim=num_stim,
            image_size=image_size,
            shape_cue_color=tuple(_get("shape_cue_color", cfg.get("shape_cue_color", (0, 0, 0)))),
            seed=_get("seed", cfg.get("seed", None)),
            fullscreen=bool(_get("fullscreen", cfg.get("fullscreen", False))),
            win_size=win_size,
            margin=int(_get("margin", cfg.get("margin", 50))),
            debug=bool(_get("debug", cfg.get("debug", False))),
            likelihood_tsv=_get("likelihood_tsv", cfg.get("likelihood_tsv", None)),
            self_initiation=True,  # enforced: every AFC trial starts from checkerboard onset cue
            fixation_size=_get("fixation_size", cfg.get("fixation_size", None)),
            show_fixation=bool(_get("show_fixation", cfg.get("show_fixation", False))),
            refresh_rate=_get("refresh_rate", cfg.get("refresh_rate", cfg.get("refrech_rate", None))),
            touchscreen=bool(_get("touchscreen", cfg.get("touchscreen", False))),
            raspi=bool(_get("raspi", cfg.get("raspi", False))),
            trial_start_pin=int(_get("trial_start_pin", cfg.get("trial_start_pin", cfg.get("raspi_pin", 18)))),
            pump_pin=int(_get("pump_pin", cfg.get("pump_pin", 17))),
            buzz_pin=int(_get("buzz_pin", cfg.get("buzz_pin", 16))),
            fixed_positions=bool(_get("fixed_positions", cfg.get("fixed_positions", False))),
            position_spacing=position_spacing,
            trial_type=trial_type,
            timeout_time=float(_get("timeout_time", cfg.get("timeout_time", 3.0))),
            reward_pulse_time=float(_get("reward_pulse_time", cfg.get("reward_pulse_time", 0.0))),
            reward_inter_pulse_time=float(_get("reward_inter_pulse_time", cfg.get("reward_inter_pulse_time", 0.0))),
            pump_delay_time=float(_get("pump_delay_time", cfg.get("pump_delay_time", 0.0))),
            pump_pulse_time_seconds=(
                None
                if _get("pump_pulse_time_seconds", cfg.get("pump_pulse_time_seconds", None)) is None
                else float(_get("pump_pulse_time_seconds", cfg.get("pump_pulse_time_seconds", None)))
            ),
            inter_pump_interval=(
                None
                if _get("inter_pump_interval", cfg.get("inter_pump_interval", None)) is None
                else float(_get("inter_pump_interval", cfg.get("inter_pump_interval", None)))
            ),
            choice_hitbox_scale=float(_get("choice_hitbox_scale", cfg.get("choice_hitbox_scale", 1.0))),
            config_name=cfg.get("config_name", "afc_csc1"),
            screen_config=screen_config,
        )
    except Exception as e:
        traceback.print_exc()
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()