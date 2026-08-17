"""
AFC trial sequence task.

- Each trial samples `num_afc` unique options without replacement.
- Stimuli are shown one at a time for `duration` seconds at random non-overlapping
  screen positions. After a stimulus disappears a faint dot is left at its location
  (controlled by `dot_size` and `dot_color`). Dots remain visible for the trial.
- After all options have been presented, the dots remain visible for `choice_time`
  seconds. Then dots are cleared, the task waits `iti` seconds, and the next trial
  starts.

Configuration keys used (in addition to common ones):
- num_afc: number of options per trial
- iti: inter-trial interval (seconds)
- dot_size: pixels
- dot_color: [r,g,b] 0-255
- choice_time: seconds to show all dots before clearing
- n: number of trials (overrides previous meaning)

Usage example:
python task/afc_trial_sequence.py --config test_configs/csc_shape_config

"""
import argparse
import math
import sys
import random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from psychopy import logging as pylogging

# Ensure project root on sys.path for local imports (same pattern as other tasks)
_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from bin import utils
from bin.frame_timing import flip_with_timestamps, plan_frame_duration
from bin.logger import SessionLogBundle
from bin.task_lifecycle import USER_EXIT_CODE
from bin.config import load_config, validate_config
from bin.screen import (
    describe_screen,
    load_screen_config,
    resolve_scene_size,
    resolve_task_screens,
)


def parse_args():
    p = argparse.ArgumentParser(description="AFC trial sequence task")
    p.add_argument("--config", help="Path to JSON config file. CLI overrides config keys.")
    p.add_argument("--images_dir", help="Path to images dir (overrides config)")
    p.add_argument("--n", type=int, default=None, help="Number of trials (overrides config n)")
    p.add_argument("--num_afc", type=int, default=None, help="Number of options per trial")
    p.add_argument("--duration", type=float, default=None, help="Stimulus duration (s)")
    p.add_argument("--choice_time", type=float, default=None, help="Choice display duration (s)")
    p.add_argument(
        "--iti",
        "--ibi",
        dest="iti",
        type=float,
        default=None,
        help="Inter-trial interval (s); --ibi is a deprecated compatibility alias",
    )
    p.add_argument("--isi", type=float, default=None, help="Pre-stimulus cue duration (s)")
    p.add_argument("--dot_size", type=int, default=None, help="Dot size in pixels")
    p.add_argument("--dot_color", type=int, nargs=3, default=None, help="Dot RGB color 0-255")
    p.add_argument("--bg", type=int, nargs=3, default=None, help="Background RGB (0-255)")
    p.add_argument("--margin", type=int, default=None, help="Margin from window edge in pixels (overrides default 50)")
    p.add_argument("--output_dir", default=None, help="Output dir for logs")
    p.add_argument("--init_dot_color", type=int, nargs=3, default=None, help="Initial pre-stimulus dot RGB color 0-255")
    p.add_argument("--seed", type=int, default=None, help="Random seed")
    p.add_argument("--fullscreen", action="store_true", default=None, help="Fullscreen")
    p.add_argument("--win_size", type=int, nargs=2, default=None, help="Window size when not fullscreen")
    p.add_argument("--image_size", type=int, nargs=2, default=None, help="Raster draw size (W H)")
    p.add_argument("--debug", action="store_true", default=None, help="Enable debug outputs (write debug images to logs/)")
    p.add_argument("--refresh_rate", type=float, default=None, help="Override detected display refresh rate (Hz); skip auto-detection if provided")
    p.add_argument("--raspi", action="store_true", default=None, help="Enable Raspberry Pi GPIO LED pulses for onset cues")
    p.add_argument("--raspi_pin", type=int, default=None, help="GPIO pin to use for raspi LED pulses (BCM numbering)")
    p.add_argument("--main_screen", default=None, help="Main task screen index or output name")
    p.add_argument("--experimenter_screen", default=None, help="Experimenter screen index or output name")
    # svg_size removed; use --image_size for both rasters and SVG rasterization
    return p.parse_args()


# non-overlap placement helper moved to `bin.utils.sample_non_overlapping_positions`


def run_task(
    images_dir: str,
    n_trials: int,
    num_afc: int,
    duration: float,
    choice_time: float,
    iti: float,
    isi: float,
    dot_size: int,
    dot_color: Tuple[int, int, int],
    bg: Tuple[int, int, int],
    output_dir: str,
    seed: Optional[int] = None,
    fullscreen: bool = True,
    win_size: Optional[Tuple[int, int]] = None,
    image_size: Optional[Tuple[int, int]] = None,
    svg_size: Optional[Tuple[int, int]] = None,
    margin: int = 50,
    init_dot_color: Optional[Tuple[int, int, int]] = None,
    debug: bool = False,
    refresh_rate: Optional[float] = None,
    raspi: bool = False,
    raspi_pin: int = 18,
    config_name: Optional[str] = None,
    screen_config: Optional[Dict[str, Any]] = None,
):
    # Configure debug behavior before any rasterization
    utils.set_debug(debug)
    duration = float(duration)
    choice_time = float(choice_time)
    iti = float(iti)
    isi = float(isi)
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("duration must be a positive finite value")
    if not math.isfinite(choice_time) or choice_time <= 0.0:
        raise ValueError("choice_time must be a positive finite value")
    for label, seconds in (("isi", isi), ("iti", iti)):
        if not math.isfinite(seconds) or seconds < 0.0:
            raise ValueError(f"{label} must be a finite non-negative value")
    if int(n_trials) < 1:
        raise ValueError("n must be at least 1")
    if seed is not None:
        random.seed(seed)

    image_files = utils.find_image_files(images_dir, recursive=False)
    if not image_files:
        raise FileNotFoundError(f"No images found in {images_dir}")
    if num_afc < 1:
        raise ValueError("num_afc must be >= 1")
    if num_afc > len(image_files):
        raise ValueError("num_afc cannot be larger than the number of available images")

    # Preload images; rasterize SVGs to image_size and flatten to bg if needed
    preloaded = utils.load_image_assets(image_files, raster_size=image_size, bg_rgb_255=bg)

    # Window + background + fixation
    main_screen, _ = resolve_task_screens(screen_config, allow_same_screen=True)
    win = utils.setup_window(
        bg_rgb_255=bg,
        fullscreen=fullscreen,
        size=win_size,
        screen_info=main_screen,
    )
    fix = utils.make_fixation_cross(win, size=32)
    bg_rect = utils.make_bg_rect(win, bg)

    resolved_config_name = str(config_name).strip() if config_name else "afc_trial_sequence"
    behavior_fieldnames = ["trial_num"] + [f"stimulus_{idx}" for idx in range(int(num_afc))] + [
        "choice_made_index",
        "choice_touch_x",
        "choice_touch_y",
        "choice_reaction_time",
    ]
    session_logs = SessionLogBundle(
        output_root=output_dir,
        task_name="afc_trial_sequence",
        config_name=resolved_config_name,
        behavior_fieldnames=behavior_fieldnames,
    )
    logger = session_logs.event_logger
    msg_logger = session_logs.message_logger
    behavior_logger = session_logs.behavior_logger
    if behavior_logger is None:
        raise RuntimeError("afc_trial_sequence requires a behavior logger")
    pylogging.console.setLevel(pylogging.CRITICAL)
    msg_logger.log(
        "INFO",
        f"session_start task=afc_trial_sequence config_name={resolved_config_name} session_dir={session_logs.session_dir}",
    )
    msg_logger.log("INFO", f"resolved_screens main={describe_screen(main_screen)}")

    # Initialize lgpio if requested
    pigpio_pi = None  # naming kept for compatibility with presenter API
    if raspi:
        try:
            import lgpio

            chip = lgpio.gpiochip_open(0)  # 0 is the default chip for RPi5
            # Claim the pin as output
            lgpio.gpio_claim_output(chip, raspi_pin)
            pigpio_pi = chip  # store chip handle
            msg_logger.log("INFO", f"lgpio initialized on chip 0, pin {raspi_pin} claimed as output")
        except Exception as e:
            pigpio_pi = None
            try:
                msg_logger.log("WARN", f"lgpio not available or failed to initialize: {e}; raspi disabled")
            except Exception:
                pass

    # Detect or override frame rate once per task
    fps, frame_dur = utils.resolve_frame_rate(
        win,
        refresh_rate,
        msg_logger=msg_logger,
        context="afc_trial_sequence",
    )
    duration_plan = plan_frame_duration(duration, fps, minimum_frames=1)
    isi_plan = plan_frame_duration(isi, fps)
    choice_plan = plan_frame_duration(choice_time, fps, minimum_frames=1)
    iti_plan = plan_frame_duration(iti, fps)
    dur_fr, dur_s = duration_plan.frame_count, duration_plan.scheduled_s
    isi_fr, isi_s = isi_plan.frame_count, isi_plan.scheduled_s
    ch_fr, ch_s = choice_plan.frame_count, choice_plan.scheduled_s
    iti_frames, iti_s = iti_plan.frame_count, iti_plan.scheduled_s
    try:
        msg_logger.log(
            "INFO",
            (
                f"timing_quantization_global fps={fps:.6f} frame_dur_s={frame_dur:.9f} "
                f"duration={duration:.6f}s-> {dur_fr}fr({dur_s:.6f}s) "
                f"isi={isi:.6f}s-> {isi_fr}fr({isi_s:.6f}s) "
                f"choice_time={choice_time:.6f}s-> {ch_fr}fr({ch_s:.6f}s) "
                f"iti={iti:.6f}s-> {iti_frames}fr({iti_s:.6f}s)"
            ),
        )
    except Exception:
        pass

    # Pre-sample independent trial option sets.
    trials = utils.sample_trial_options(image_files, num_afc, n_trials, seed=seed)

    msg_logger.log("INFO", f"task_ready n_trials={n_trials} num_afc={num_afc}")

    # Main trial loop
    aborted_task = False
    for trial_num in range(1, n_trials + 1):
        if isi_fr > 0:
            bg_rect.draw()
            if fix is not None:
                fix.draw()
            cue_timing = flip_with_timestamps(win)
            logger.log_frame_flip(
                trial_num=trial_num,
                event="trial_cue",
                timestamp_perf_s=cue_timing.actual_perf_s,
                requested_timestamp_perf_s=cue_timing.requested_perf_s,
                requested_duration=isi,
            )
            for _ in range(max(0, isi_fr - 1)):
                bg_rect.draw()
                if fix is not None:
                    fix.draw()
                win.flip()

        trial_options = trials[trial_num - 1]
        msg_logger.log("INFO", f"trial_loaded trial_num={trial_num} stimuli={[p.name for p in trial_options]}")

        # compute native stim size from preloaded images (we will use same size for all)
        first_p = trial_options[0]
        pil0 = preloaded[first_p]
        stim_size = pil0.size  # (W,H) in pixels

        effective_win_size = resolve_scene_size(
            main_screen,
            fullscreen=bool(fullscreen),
            requested_size=win_size,
            realized_size=tuple(win.size),
        )

        # Compute non-overlapping positions for all options in this trial.
        sampled_positions = utils.sample_non_overlapping_positions(
            num_afc, stim_size, effective_win_size, margin=margin
        )

        # Defensive clamp via utility (keeps behavior identical but centralizes logic)
        positions = utils.clamp_positions(sampled_positions, stim_size, effective_win_size, margin=margin)

        # Log sampled vs clamped positions for debugging (one row per assign).
        for i, (spos, cpos) in enumerate(zip(sampled_positions, positions), start=1):
            # Non-task diagnostic: log to message logger
            try:
                img = trial_options[i - 1].name if i - 1 < len(trial_options) else ""
            except Exception:
                img = ""
            msg_logger.log("INFO", f"position_assigned trial_num={trial_num} option_num={i} image={img} sampled={spos} clamped={cpos}")

        trial_meta = {}
        aborted, choice_info = utils.present_trial_with_persistent_dots(
            win=win,
            preloaded=preloaded,
            trial_options=trial_options,
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
            bg_rgb_255=bg,
            msg_logger=msg_logger,
            fps=fps,
            raspi=bool(raspi and pigpio_pi is not None),
            pigpio_pi=pigpio_pi,
            raspi_pin=raspi_pin,
            trial_meta=trial_meta,
        )
        if aborted:
            aborted_task = True
            msg_logger.log("WARN", f"trial_aborted trial_num={trial_num}")
            break

        gray_start_perf = trial_meta.get("gray_flip_perf_s", None)
        if gray_start_perf is not None:
            logger.log_frame_flip(
                trial_num=trial_num,
                event="gray_inter_trial_interval",
                timestamp_perf_s=float(gray_start_perf),
                requested_timestamp_perf_s=trial_meta.get(
                    "gray_flip_requested_perf_s"
                ),
                requested_duration=iti if iti > 0.0 else None,
            )

        behavior_row = {"trial_num": trial_num}
        for idx, path in enumerate(trial_options):
            behavior_row[f"stimulus_{idx}"] = path.name
        chosen_idx = choice_info.get("chosen_index") if choice_info is not None else None
        behavior_row["choice_made_index"] = int(chosen_idx - 1) if chosen_idx is not None else ""
        behavior_row["choice_touch_x"] = (
            f"{float(choice_info['touch_x']):.9f}" if choice_info is not None and choice_info.get("touch_x") is not None else ""
        )
        behavior_row["choice_touch_y"] = (
            f"{float(choice_info['touch_y']):.9f}" if choice_info is not None and choice_info.get("touch_y") is not None else ""
        )
        behavior_row["choice_reaction_time"] = (
            f"{float(choice_info['reaction_time_s']):.9f}" if choice_info is not None and choice_info.get("reaction_time_s") is not None else ""
        )
        behavior_logger.writerow(behavior_row)

        if iti_frames > 0:
            msg_logger.log("INFO", f"timing_quantization trial_num={trial_num} iti={iti:.6f}s-> {iti_frames}fr({iti_s:.6f}s)")
            for _f in range(max(0, iti_frames - 1)):
                bg_rect.draw()
                if fix is not None:
                    fix.draw()
                win.flip()

    # finished
    msg_logger.log("INFO", f"session_end status={'aborted' if aborted_task else 'done'}")
    session_logs.close()
    win.close()
    return aborted_task


def main():
    args = parse_args()
    cfg = {}
    if args.config:
        cfg = load_config(args.config)
        # validate some required keys for this task
        validate_config(cfg, required=["config_name", "images_dir", "output_dir", "duration", "n"])  # basic
    else:
        missing = []
        if not args.images_dir:
            missing.append("--images_dir or config")
        if args.n is None:
            missing.append("--n or config")
        if args.duration is None:
            missing.append("--duration or config")
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

    # gather parameters (use config defaults where CLI doesn't override)
    images_dir = _get("images_dir", cfg.get("images_dir"))
    n_trials = int(_get("n", cfg.get("n")))
    num_afc = int(_get("num_afc", cfg.get("num_afc", 2)))
    duration = float(_get("duration", cfg.get("duration")))
    choice_time = float(_get("choice_time", cfg.get("choice_time", 2.0)))
    iti = float(_get("iti", cfg.get("iti", cfg.get("ibi", 1.0))))
    dot_size = int(_get("dot_size", cfg.get("dot_size", 8)))
    dot_color = tuple(_get("dot_color", cfg.get("dot_color", (180, 180, 180))))
    init_dot_color = tuple(_get("init_dot_color", cfg.get("init_dot_color", None))) if _get("init_dot_color", None) else None
    bg = tuple(_get("bg", cfg.get("bg", (128, 128, 128))))
    output_dir = _get("output_dir", cfg.get("output_dir", "./logs"))
    isi = float(_get("isi", cfg.get("isi", 0.0)))
    raw_margin = _get("margin", cfg.get("margin", 50))
    margin = int(raw_margin) if raw_margin is not None else 50
    seed = _get("seed", cfg.get("seed", None))
    fullscreen = bool(_get("fullscreen", cfg.get("fullscreen", True)))
    win_size = tuple(_get("win_size", cfg.get("win_size", None))) if _get("win_size", None) else None
    image_size = tuple(_get("image_size", cfg.get("image_size", None))) if _get("image_size", None) else None
    svg_size = None
    refresh_rate = _get("refresh_rate", cfg.get("refresh_rate", cfg.get("refrech_rate", None)))
    raspi = _get("raspi", cfg.get("raspi", False))
    raspi_pin = int(_get("raspi_pin", cfg.get("raspi_pin", 18)))
    config_name = cfg.get("config_name", "afc_trial_sequence")

    try:
        aborted_by_user = run_task(
            images_dir=images_dir,
            n_trials=n_trials,
            num_afc=num_afc,
            duration=duration,
            choice_time=choice_time,
            iti=iti,
            isi=isi,
            margin=margin,
            init_dot_color=init_dot_color,
            dot_size=dot_size,
            dot_color=dot_color,
            bg=bg,
            output_dir=output_dir,
            seed=seed,
            fullscreen=fullscreen,
            win_size=win_size,
            image_size=image_size,
            svg_size=svg_size,
            refresh_rate=refresh_rate,
            raspi=_get("raspi", cfg.get("raspi", False)),
            raspi_pin=_get("raspi_pin", cfg.get("raspi_pin", 18)),
            config_name=config_name,
            screen_config=screen_config,
        )
        if aborted_by_user:
            sys.exit(USER_EXIT_CODE)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
