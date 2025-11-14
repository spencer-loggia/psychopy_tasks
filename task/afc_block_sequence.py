"""
AFC block sequence task.

- Stimuli are shown in blocks of `num_afc` (from config).
- Each block: sample `num_afc` unique stimuli (without replacement within block).
- Stimuli are shown one at a time for `duration` seconds at random non-overlapping
  screen positions. After a stimulus disappears a faint dot is left at its location
  (controlled by `dot_size` and `dot_color`). Dots remain visible for the block.
- After all stimuli in a block are shown, the dots remain visible for `choice_time`
  seconds (choice period). Then dots are cleared, the task waits `ibi` seconds
  (inter-block interval), and the next block starts.

Configuration keys used (in addition to common ones):
- num_afc: number of stimuli per block
- ibi: inter-block interval (seconds)
- dot_size: pixels
- dot_color: [r,g,b] 0-255
- choice_time: seconds to show all dots before clearing
- n: number of blocks (overrides previous meaning)

Usage example:
python task/afc_block_sequence.py --config test_configs/csc_shape_config

"""
import argparse
import sys
import time
import random
from pathlib import Path
from typing import Tuple, Optional, List

from psychopy import core, logging as pylogging, event

# Ensure project root on sys.path for local imports (same pattern as other tasks)
_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from bin import utils
from bin.logger import EventLogger, MessageLogger
from bin.config import load_config, validate_config


def parse_args():
    p = argparse.ArgumentParser(description="AFC block sequence task")
    p.add_argument("--config", help="Path to JSON config file. CLI overrides config keys.")
    p.add_argument("--images_dir", help="Path to images dir (overrides config)")
    p.add_argument("--n", type=int, default=None, help="Number of blocks (overrides config n)")
    p.add_argument("--num_afc", type=int, default=None, help="Number of stimuli per block")
    p.add_argument("--duration", type=float, default=None, help="Stimulus duration (s)")
    p.add_argument("--choice_time", type=float, default=None, help="Choice display time after block (s)")
    p.add_argument("--ibi", type=float, default=None, help="Inter-block interval (s)")
    p.add_argument("--isi", type=float, default=None, help="Pre-block fixation delay (seconds) before first stimulus)")
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
    # svg_size removed; use --image_size for both rasters and SVG rasterization
    return p.parse_args()


# non-overlap placement helper moved to `bin.utils.sample_non_overlapping_positions`


def run_task(
    images_dir: str,
    n_blocks: int,
    num_afc: int,
    duration: float,
    choice_time: float,
    ibi: float,
    isi: float,
    dot_size: int,
    dot_color: Tuple[int, int, int],
    bg: Tuple[int, int, int],
    output_dir: str,
    seed: Optional[int] = None,
    fullscreen: bool = False,
    win_size: Optional[Tuple[int, int]] = None,
    image_size: Optional[Tuple[int, int]] = None,
    svg_size: Optional[Tuple[int, int]] = None,
    margin: int = 50,
    init_dot_color: Optional[Tuple[int, int, int]] = None,
    debug: bool = False,
    refresh_rate: Optional[float] = None,
    raspi: bool = False,
    raspi_pin: int = 18,
):
    # Configure debug behavior before any rasterization
    utils.set_debug(debug)
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
    win = utils.setup_window(bg_rgb_255=bg, fullscreen=fullscreen, size=win_size)
    fix = utils.make_fixation_cross(win, size=32)
    bg_rect = utils.make_bg_rect(win, bg)

    # Prepare loggers
    logger = EventLogger(output_dir, filename="afc_block_sequence_log.tsv")
    msg_logger = MessageLogger(output_dir, filename="afc_block_sequence_message_log.tsv")
    pylogging.console.setLevel(pylogging.CRITICAL)

    # Initialize pigpio if requested
    pigpio_pi = None
    if raspi:
        try:
            import pigpio as _pigpio

            pigpio_pi = _pigpio.pi()
            if getattr(pigpio_pi, "connected", None) is False:
                try:
                    pigpio_pi.stop()
                except Exception:
                    pass
                pigpio_pi = None
                msg_logger.log("WARN", "pigpio present but not connected; raspi disabled")
        except Exception:
            pigpio_pi = None
            try:
                msg_logger.log("WARN", "pigpio not available; raspi disabled")
            except Exception:
                pass

    # Detect or override frame rate once per task
    if refresh_rate is not None and float(refresh_rate) > 0:
        fps = float(refresh_rate)
        frame_dur = 1.0 / fps
        try:
            msg_logger.log("INFO", f"fps_override refresh_rate={fps:.6f}Hz frame_dur_s={frame_dur:.9f}")
        except Exception:
            pass
    else:
        fps, frame_dur = utils.detect_frame_rate(win, msg_logger=msg_logger)
    # Log global quantization for task timing parameters
    try:
        def _q(seconds: float, at_least_one: bool = False):
            frames = int(round(max(0.0, float(seconds)) * float(fps)))
            if at_least_one:
                frames = max(1, frames)
            return frames, frames / float(fps)

        dur_fr, dur_s = _q(duration, at_least_one=True)
        isi_fr, isi_s = _q(isi, at_least_one=False)
        ch_fr, ch_s = _q(choice_time, at_least_one=False)
        ibi_fr, ibi_s = _q(ibi, at_least_one=False)
        msg_logger.log(
            "INFO",
            (
                f"timing_quantization_global fps={fps:.6f} frame_dur_s={frame_dur:.9f} "
                f"duration={duration:.6f}s-> {dur_fr}fr({dur_s:.6f}s) "
                f"isi={isi:.6f}s-> {isi_fr}fr({isi_s:.6f}s) "
                f"choice_time={choice_time:.6f}s-> {ch_fr}fr({ch_s:.6f}s) "
                f"ibi={ibi:.6f}s-> {ibi_fr}fr({ibi_s:.6f}s)"
            ),
        )
    except Exception:
        pass

    # Pre-sample blocks (each block samples `num_afc` unique stimuli without
    # replacement within the block). Blocks are independent.
    blocks = utils.sample_blocks(image_files, num_afc, n_blocks, seed=seed)

    # Task start
    logger.log("task_start", image_name="", notes=f"n_blocks={n_blocks} num_afc={num_afc}")

    # main block loop
    for block_idx in range(1, n_blocks + 1):
        logger.log(
            "block_start",
            image_name="",
            requested_duration_s=None,
            flip_time_psychopy_s=None,
            flip_time_perf_s=time.perf_counter(),
            end_time_perf_s=None,
            notes=f"block={block_idx}",
        )

        # Block-start fixation cue: draw fixation on background `isi` seconds
        # before the first stimulus of the block so the subject has a cue.
        if isi and isi > 0:
            bg_rect.draw()
            if fix is not None:
                fix.draw()
            cue_flip = win.flip()
            cue_perf = time.perf_counter()
            logger.log(
                "block_cue",
                image_name="",
                requested_duration_s=isi,
                flip_time_psychopy_s=cue_flip,
                flip_time_perf_s=cue_perf,
                end_time_perf_s=cue_perf + isi,
                notes=f"block={block_idx}",
            )
            core.wait(isi)

        # choose num_afc unique stimuli for this block (pre-sampled)
        block_paths = blocks[block_idx - 1]

        # compute native stim size from preloaded images (we will use same size for all)
        first_p = block_paths[0]
        pil0 = preloaded[first_p]
        stim_size = pil0.size  # (W,H) in pixels

        # Determine the logical window size to use for placement. Prefer the
        # user-provided `win_size` (the logical size passed to Window) when
        # available; some backends report a different backing-buffer size via
        # `win.size` (e.g., Retina displays). Using the requested `win_size`
        # keeps coordinates in the expected logical pixel space.
        effective_win_size = tuple(win_size) if win_size is not None else tuple(win.size)

        # compute non-overlapping positions for all items in this block
        sampled_positions = utils.sample_non_overlapping_positions(
            num_afc, stim_size, effective_win_size, margin=margin
        )

        # Defensive clamp via utility (keeps behavior identical but centralizes logic)
        positions = utils.clamp_positions(sampled_positions, stim_size, effective_win_size, margin=margin)

        # Log sampled vs clamped positions for debugging (one row per assign).
        for i, (spos, cpos) in enumerate(zip(sampled_positions, positions), start=1):
            # Non-task diagnostic: log to message logger
            try:
                img = block_paths[i - 1].name if i - 1 < len(block_paths) else ""
            except Exception:
                img = ""
            try:
                msg_logger.log("INFO", f"position_assigned block={block_idx} idx={i} image={img} sampled={spos} clamped={cpos}")
            except Exception:
                pass

        # Present stimuli for this block using the shared utility which
        # draws stimuli one-at-a-time, leaves persistent dots, shows the
        # choice period, and handles abort via escape.
        aborted, choice_info = utils.present_block_with_persistent_dots(
            win=win,
            preloaded=preloaded,
            block_paths=block_paths,
            positions=positions,
            duration=duration,
            choice_time=choice_time,
            dot_size=dot_size,
            dot_color=dot_color,
            bg_rect=bg_rect,
            fix=fix,
            logger=logger,
            block_idx=block_idx,
            isi=isi,
            init_dot_color=init_dot_color,
            bg_rgb_255=bg,
            msg_logger=msg_logger,
            fps=fps,
            raspi=bool(raspi and pigpio_pi is not None),
            pigpio_pi=pigpio_pi,
            raspi_pin=raspi_pin,
        )
        if aborted:
            return

        # Choice logging handled within utils.present_block_with_persistent_dots

        # Inter-block interval (frame-locked)
        if ibi and ibi > 0:
            ibi_frames = int(round(float(ibi) * fps))
            ibi_frames = max(0, ibi_frames)
            ibi_s = ibi_frames / fps
            try:
                msg_logger.log("INFO", f"timing_quantization block={block_idx} ibi={ibi:.6f}s-> {ibi_frames}fr({ibi_s:.6f}s)")
            except Exception:
                pass
            bg_rect.draw()
            if fix is not None:
                fix.draw()
            ibi_flip = win.flip()
            ibi_perf = time.perf_counter()
            logger.log(
                "ibi_start",
                image_name="",
                requested_duration_s=ibi_s,
                flip_time_psychopy_s=ibi_flip,
                flip_time_perf_s=ibi_perf,
                end_time_perf_s=ibi_perf + ibi_s,
                notes=f"after_block={block_idx}",
            )
            for _f in range(max(0, ibi_frames - 1)):
                bg_rect.draw()
                if fix is not None:
                    fix.draw()
                win.flip()

        logger.log("block_end", image_name="", notes=f"block={block_idx}")

    # finished
    logger.log("task_end", image_name="", notes="done")
    logger.close()
    try:
        msg_logger.close()
    except Exception:
        pass
    win.close()
    core.quit()


def main():
    args = parse_args()
    cfg = {}
    if args.config:
        cfg = load_config(args.config)
        # validate some required keys for this task
        validate_config(cfg, required=["images_dir", "output_dir", "duration", "n"])  # basic
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

    # gather parameters (use config defaults where CLI doesn't override)
    images_dir = _get("images_dir", cfg.get("images_dir"))
    n_blocks = int(_get("n", cfg.get("n")))
    num_afc = int(_get("num_afc", cfg.get("num_afc", 2)))
    duration = float(_get("duration", cfg.get("duration")))
    choice_time = float(_get("choice_time", cfg.get("choice_time", 2.0)))
    ibi = float(_get("ibi", cfg.get("ibi", 1.0)))
    dot_size = int(_get("dot_size", cfg.get("dot_size", 8)))
    dot_color = tuple(_get("dot_color", cfg.get("dot_color", (180, 180, 180))))
    init_dot_color = tuple(_get("init_dot_color", cfg.get("init_dot_color", None))) if _get("init_dot_color", None) else None
    bg = tuple(_get("bg", cfg.get("bg", (128, 128, 128))))
    output_dir = _get("output_dir", cfg.get("output_dir", "./logs"))
    isi = float(_get("isi", cfg.get("isi", 0.0)))
    raw_margin = _get("margin", cfg.get("margin", 50))
    margin = int(raw_margin) if raw_margin is not None else 50
    seed = _get("seed", cfg.get("seed", None))
    fullscreen = bool(_get("fullscreen", cfg.get("fullscreen", False)))
    win_size = tuple(_get("win_size", cfg.get("win_size", None))) if _get("win_size", None) else None
    image_size = tuple(_get("image_size", cfg.get("image_size", None))) if _get("image_size", None) else None
    svg_size = None
    refresh_rate = _get("refresh_rate", cfg.get("refresh_rate", cfg.get("refrech_rate", None)))
    raspi = _get("raspi", cfg.get("raspi", False))
    raspi_pin = int(_get("raspi_pin", cfg.get("raspi_pin", 18)))

    try:
        run_task(
            images_dir=images_dir,
            n_blocks=n_blocks,
            num_afc=num_afc,
            duration=duration,
            choice_time=choice_time,
            ibi=ibi,
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
        )
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
