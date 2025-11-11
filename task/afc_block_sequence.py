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
from bin.logger import EventLogger
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
    p.add_argument("--bg", type=int, nargs=3, default=(128, 128, 128), help="Background RGB (0-255)")
    p.add_argument("--margin", type=int, default=None, help="Margin from window edge in pixels (overrides default 50)")
    p.add_argument("--output_dir", default="./logs", help="Output dir for logs")
    p.add_argument("--init_dot_color", type=int, nargs=3, default=None, help="Initial pre-stimulus dot RGB color 0-255")
    p.add_argument("--seed", type=int, default=None, help="Random seed")
    p.add_argument("--fullscreen", action="store_true", help="Fullscreen")
    p.add_argument("--win_size", type=int, nargs=2, default=None, help="Window size when not fullscreen")
    p.add_argument("--image_size", type=int, nargs=2, default=None, help="Raster draw size (W H)")
    p.add_argument("--debug", action="store_true", help="Enable debug outputs (write debug images to logs/)")
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
    from psychopy import visual as _visual
    bg_rect = _visual.Rect(
        win,
        width=win.size[0],
        height=win.size[1],
        fillColor=utils.rgb255_to_psychopy(bg),
        fillColorSpace="rgb",
        lineColor=None,
        units="pix",
    )

    # Prepare logger
    logger = EventLogger(output_dir, filename="afc_block_sequence_log.tsv")
    pylogging.console.setLevel(pylogging.CRITICAL)

    # Pre-sample blocks (each block samples `num_afc` unique stimuli without
    # replacement within the block). Blocks are independent.
    blocks = utils.sample_blocks(image_files, num_afc, n_blocks, seed=seed)

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
            logger.log(
                "position_assigned",
                image_name=block_paths[i - 1].name if i - 1 < len(block_paths) else "",
                notes=f"sampled={spos} clamped={cpos} block={block_idx} idx={i}",
            )

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
        )
        if aborted:
            return

        # Log the canonical single choice result for this block (one row)
        if choice_info is None:
            logger.log("no_choice", image_name="", notes=f"block={block_idx}")
        else:
            ci = choice_info
            img_name = getattr(block_paths[ci["chosen_index"] - 1], "name", str(block_paths[ci["chosen_index"] - 1]))
            click_xy = ci.get("chosen_pos")
            logger.log(
                "choice_registered",
                image_name=img_name,
                requested_duration_s=None,
                flip_time_psychopy_s=None,
                flip_time_perf_s=ci.get("choice_time_perf_s"),
                end_time_perf_s=None,
                notes=f"block={block_idx} idx={ci.get('chosen_index')} click_xy={click_xy}",
            )

        # Inter-block interval
        if ibi and ibi > 0:
            ibi_flip = win.flip()
            ibi_perf = time.perf_counter()
            logger.log(
                "ibi_start",
                image_name="",
                requested_duration_s=ibi,
                flip_time_psychopy_s=ibi_flip,
                flip_time_perf_s=ibi_perf,
                end_time_perf_s=ibi_perf + ibi,
                notes=f"after_block={block_idx}",
            )
            core.wait(ibi)

        logger.log("block_end", image_name="", notes=f"block={block_idx}")

    # finished
    logger.close()
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
        )
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
