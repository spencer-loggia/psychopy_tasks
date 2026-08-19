"""
Run a task that randomly draws N images (raster or SVG) from a directory, centered on a fixation cross,
each shown for a fixed duration on a gray background. Images are preloaded into RAM before presentation.
Events (flip times) are logged in TSV.

Usage example:
python task/random_image_sequence.py \
  --images_dir ./sample_images \
  --n 10 \
  --duration 0.5 \
  --bg 128 128 128 \
  --output_dir ./logs \
  --seed 42 \
  --fullscreen \
  --svg_size 256 256
"""
import argparse
import math
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from psychopy import logging as pylogging, event

# Ensure project root on sys.path for local imports
_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from bin import utils
from bin.frame_timing import flip_with_timestamps, plan_frame_duration
from bin.logger import SessionLogBundle
from bin.task_lifecycle import USER_EXIT_CODE
from bin.config import load_config, validate_config
from bin.screen import describe_screen, load_screen_config


def parse_args():
    parser = argparse.ArgumentParser(description="Random image sequence task with preloading (raster + SVG)")
    parser.add_argument("--config", help="Path to JSON config file. If provided, CLI args override config keys.")
    parser.add_argument("--images_dir", required=False, help="Path to image resources directory")
    parser.add_argument("--n", type=int, default=None, help="Number of images to display")
    parser.add_argument("--duration", type=float, required=False, help="Display duration for each image (seconds)")
    parser.add_argument("--bg", type=int, nargs=3, default=None, help="Background gray RGB (3 ints 0-255)")
    parser.add_argument("--output_dir", required=False, default=None, help="Directory to save event logs")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (optional)")
    parser.add_argument("--fullscreen", action="store_true", default=None, help="Run fullscreen")
    parser.add_argument("--win_size", type=int, nargs=2, default=None, help="Window size when not fullscreen, e.g. --win_size 1024 768")
    parser.add_argument("--fixation_size", type=int, default=None, help="Fixation cross size (px)")
    parser.add_argument("--image_size", type=int, nargs=2, default=None, help="Raster image draw size (W H) in pixels (also used to resize preloaded rasters)")
    # svg_size removed; use --image_size for both rasters and SVG rasterization
    parser.add_argument("--debug", action="store_true", default=None, help="Enable debug outputs (write debug images to logs/)")
    parser.add_argument("--isi", type=float, default=None, help="Inter-stimulus interval in seconds (fixation visible). If omitted, value from --config is used.")
    parser.add_argument("--refresh_rate", type=float, default=None, help="Override detected display refresh rate (Hz); skip auto-detection if provided")
    parser.add_argument("--main_screen", default=None, help="Main task screen index or output name")
    parser.add_argument("--experimenter_screen", default=None, help="Experimenter screen index or output name")
    return parser.parse_args()


def run_task(
    images_dir: str,
    n: int,
    duration: float,
    bg: Tuple[int, int, int],
    output_dir: str,
    seed: int = None,
    fullscreen: bool = True,
    win_size: Optional[Tuple[int, int]] = None,
    fixation_size: int = 40,
    image_size: Optional[Tuple[int, int]] = None,  # raster-only preferred size
    svg_size: Optional[Tuple[int, int]] = None,    # svg rasterization size
    isi: float = 0.0,
    debug: bool = False,
    refresh_rate: Optional[float] = None,
    config_name: Optional[str] = None,
    screen_config: Optional[Dict[str, Any]] = None,
):
    # Enable or disable debug outputs (writing debug PNGs)
    utils.set_debug(debug)
    duration = float(duration)
    isi = float(isi)
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("duration must be a positive finite value")
    if not math.isfinite(isi) or isi < 0.0:
        raise ValueError("isi must be a finite non-negative value")
    if int(n) < 1:
        raise ValueError("n must be at least 1")
    image_files = utils.find_image_files(images_dir, recursive=False)
    if not image_files:
        raise FileNotFoundError(f"No images found in {images_dir}")

    # If SVGs are present we require image_size to be provided so we know what
    # pixel size to rasterize them to. load_image_assets will validate this.

    chosen_paths = utils.sample_images(image_files, n, seed=seed)

    # Preload into RAM (raster resized to image_size; SVG rasterized to svg_size)
    print("Preloading images into RAM (raster + SVG)...")
    # Pass the background color so rasterized SVGs can be flattened to the
    # same background color and avoid platform-specific transparency issues.
    preloaded = utils.load_image_assets(chosen_paths, raster_size=image_size, bg_rgb_255=bg)

    # Resolve and verify the same physical main output used by other tasks.
    win, main_screen, _ = utils.setup_task_window(
        screen_config,
        bg_rgb_255=bg,
        fullscreen=fullscreen,
        size=win_size,
        allow_same_screen=True,
    )
    fix = utils.make_fixation_cross(win, size=fixation_size)
    # Create a full-window background patch so we can reliably show a solid
    # background during ISI periods (some backends may retain previous
    # textures if nothing is drawn).
    bg_rect = utils.make_bg_rect(win, bg)

    resolved_config_name = str(config_name).strip() if config_name else "random_image_sequence"
    session_logs = SessionLogBundle(
        output_root=output_dir,
        task_name="random_image_sequence",
        config_name=resolved_config_name,
        behavior_fieldnames=["trial_num", "stimulus_name", "requested_duration"],
    )
    logger = session_logs.event_logger
    msg_logger = session_logs.message_logger
    behavior_logger = session_logs.behavior_logger
    if behavior_logger is None:
        raise RuntimeError("random_image_sequence requires a behavior logger")
    pylogging.console.setLevel(pylogging.CRITICAL)
    msg_logger.log(
        "INFO",
        f"session_start task=random_image_sequence config_name={resolved_config_name} session_dir={session_logs.session_dir}",
    )
    msg_logger.log("INFO", f"resolved_screens main={describe_screen(main_screen)}")

    # Detect or override frame rate once per task
    fps, frame_dur = utils.resolve_frame_rate(
        win,
        refresh_rate,
        msg_logger=msg_logger,
        context="random_image_sequence",
    )
    # Plan each requested duration to its nearest refresh-locked presentation.
    stim_plan = plan_frame_duration(duration, fps, minimum_frames=1)
    isi_plan = plan_frame_duration(isi, fps)
    final_fix_plan = plan_frame_duration(1.0, fps)
    stim_frames, stim_s = stim_plan.frame_count, stim_plan.scheduled_s
    isi_frames, isi_s = isi_plan.frame_count, isi_plan.scheduled_s
    final_fix_frames, final_fix_s = (
        final_fix_plan.frame_count,
        final_fix_plan.scheduled_s,
    )
    try:
        msg_logger.log(
            "INFO",
            (
                f"timing_quantization_global fps={fps:.6f} frame_dur_s={frame_dur:.9f} "
                f"duration={duration:.6f}s-> {stim_frames}fr({stim_s:.6f}s) "
                f"isi={isi:.6f}s-> {isi_frames}fr({isi_s:.6f}s) "
                f"final_fixation=1.000000s-> {final_fix_frames}fr({final_fix_s:.6f}s)"
            ),
        )
    except Exception:
        pass

    # Convert preloaded PIL images to ImageStim (do NOT pass bg_rgb_255: preserve transparency)
    image_stims = []
    for p in chosen_paths:
        pil_img = preloaded.get(p)
        if pil_img is None:
            raise RuntimeError(f"Preloaded image missing for {p}")
        # We already sized during preload (raster resize or svg rasterization).
        # Draw at native pixel size to avoid double-scaling.
        stim = utils.make_image_stim_from_array(win, pil_img, size=None, bg_rgb_255=None)
        image_stims.append((p.name, stim))

    # initial blank flip draw background
    win.flip()

    # Pre-sequence ISI: show the gray background for `isi` seconds before the
    # first stimulus. This implements the requested sequence: gray(ISI) ->
    # stim(duration) -> gray(ISI) -> stim(duration) ...
    if isi_frames > 0:
        # Pre-sequence ISI for exactly isi_frames frames
        first_flip = True
        for _f in range(isi_frames):
            bg_rect.draw()
            if fix is not None:
                fix.draw()
            isi_timing = flip_with_timestamps(win)
            if first_flip:
                logger.log_frame_flip(
                    trial_num=None,
                    event="gray_pre_sequence",
                    timestamp_perf_s=isi_timing.actual_perf_s,
                    requested_timestamp_perf_s=isi_timing.requested_perf_s,
                    requested_duration=isi,
                )
                first_flip = False

    aborted = False

    # Main loop
    for idx, (img_name, stim) in enumerate(image_stims, start=1):
        first_flip = True
        for _f in range(stim_frames):
            stim.draw()
            if fix is not None:
                fix.draw()  # fixation on top
            stim_timing = flip_with_timestamps(win)
            if first_flip:
                logger.log_frame_flip(
                    trial_num=idx,
                    event="stimulus_on",
                    timestamp_perf_s=stim_timing.actual_perf_s,
                    requested_timestamp_perf_s=stim_timing.requested_perf_s,
                    requested_duration=duration,
                )
                msg_logger.log("INFO", f"stimulus_presented trial_num={idx} stimulus_name={img_name}")
                first_flip = False
            # Abort?
            if event.getKeys(["escape"]):
                msg_logger.log("WARN", f"escape_pressed trial_num={idx}")
                aborted = True
                break
        if aborted:
            break
        is_last_stimulus = idx == len(image_stims)
        # The gray-onset flip is the first frame of a non-zero ISI/final
        # fixation. Skip it between stimuli when the nearest-frame ISI is zero.
        if is_last_stimulus or isi_frames > 0:
            bg_rect.draw()
            if fix is not None:
                fix.draw()
            gray_timing = flip_with_timestamps(win)
            logger.log_frame_flip(
                trial_num=idx,
                event=(
                    "gray_final_fixation"
                    if is_last_stimulus
                    else "gray_inter_stimulus"
                ),
                timestamp_perf_s=gray_timing.actual_perf_s,
                requested_timestamp_perf_s=gray_timing.requested_perf_s,
                requested_duration=1.0 if is_last_stimulus else isi,
            )
        behavior_logger.writerow(
            {
                "trial_num": idx,
                "stimulus_name": img_name,
                "requested_duration": f"{duration:.9f}",
            }
        )
        # ISI between images
        for _f in range(
            max(0, isi_frames - 1) if not is_last_stimulus else 0
        ):
            bg_rect.draw()
            if fix is not None:
                fix.draw()
            win.flip()
        # Abort?
        if event.getKeys(["escape"]):
            msg_logger.log("WARN", f"escape_pressed trial_num={idx}")
            aborted = True
            break

    # Final fixation and cleanup
    if not aborted:
        for _f in range(max(0, final_fix_frames - 1)):
            bg_rect.draw()
            if fix is not None:
                fix.draw()
            win.flip()

    msg_logger.log("INFO", f"session_end status={'aborted' if aborted else 'done'}")
    session_logs.close()
    utils.close_task_window(win)
    print(f"Finished; logs written to {session_logs.session_dir.resolve()}")
    return aborted


def main():
    args = parse_args()
    # Load config if provided
    cfg = {}
    if args.config:
        cfg = load_config(args.config)
        validate_config(cfg, required=["config_name", "images_dir", "output_dir", "duration", "n"])
    else:
        missing = []
        if not args.images_dir:
            missing.append("--images_dir")
        if not args.duration:
            missing.append("--duration")
        if not args.output_dir:
            missing.append("--output_dir")
        if missing:
            print(f"ERROR: missing required arguments (or provide --config): {', '.join(missing)}", file=sys.stderr)
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

    try:
        aborted_by_user = run_task(
            images_dir=_get("images_dir", cfg.get("images_dir")),
            n=int(_get("n", cfg.get("n", 10))),
            duration=float(_get("duration", cfg.get("duration"))),
            bg=tuple(_get("bg", cfg.get("bg", (128, 128, 128)))),
            output_dir=_get("output_dir", cfg.get("output_dir", "./logs")),
            seed=_get("seed", cfg.get("seed", None)),
            fullscreen=bool(_get("fullscreen", cfg.get("fullscreen", True))),
            win_size=tuple(_get("win_size", cfg.get("win_size", None))) if _get("win_size", None) else None,
            fixation_size=int(_get("fixation_size", cfg.get("fixation_size", 40))),
            image_size=tuple(_get("image_size", cfg.get("image_size", None))) if _get("image_size", None) else None,
            svg_size=tuple(_get("svg_size", cfg.get("svg_size", None))) if _get("svg_size", None) else None,
            isi=float(_get("isi", cfg.get("isi", 0.0))),
            debug=bool(_get("debug", cfg.get("debug", False))),
            refresh_rate=_get("refresh_rate", cfg.get("refresh_rate", cfg.get("refrech_rate", None))),
            config_name=cfg.get("config_name", "random_image_sequence"),
            screen_config=screen_config,
        )
        if aborted_by_user:
            sys.exit(USER_EXIT_CODE)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
