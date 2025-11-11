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
import sys
import time
from pathlib import Path
from typing import Tuple, Optional

from psychopy import core, logging as pylogging, event

# Ensure project root on sys.path for local imports
_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from bin import utils
from bin.logger import EventLogger
from bin.config import load_config, validate_config


def parse_args():
    parser = argparse.ArgumentParser(description="Random image sequence task with preloading (raster + SVG)")
    parser.add_argument("--config", help="Path to JSON config file. If provided, CLI args override config keys.")
    parser.add_argument("--images_dir", required=False, help="Path to image resources directory")
    parser.add_argument("--n", type=int, default=10, help="Number of images to display")
    parser.add_argument("--duration", type=float, required=False, help="Display duration for each image (seconds)")
    parser.add_argument("--bg", type=int, nargs=3, default=(128, 128, 128), help="Background gray RGB (3 ints 0-255)")
    parser.add_argument("--output_dir", required=False, help="Directory to save event logs")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (optional)")
    parser.add_argument("--fullscreen", action="store_true", help="Run fullscreen")
    parser.add_argument("--win_size", type=int, nargs=2, default=None, help="Window size when not fullscreen, e.g. --win_size 1024 768")
    parser.add_argument("--fixation_size", type=int, default=40, help="Fixation cross size (px)")
    parser.add_argument("--image_size", type=int, nargs=2, default=None, help="Raster image draw size (W H) in pixels (also used to resize preloaded rasters)")
    # svg_size removed; use --image_size for both rasters and SVG rasterization
    parser.add_argument("--debug", action="store_true", help="Enable debug outputs (write debug images to logs/)")
    parser.add_argument("--isi", type=float, default=None, help="Inter-stimulus interval in seconds (fixation visible). If omitted, value from --config is used.")
    return parser.parse_args()


def run_task(
    images_dir: str,
    n: int,
    duration: float,
    bg: Tuple[int, int, int],
    output_dir: str,
    seed: int = None,
    fullscreen: bool = False,
    win_size: Optional[Tuple[int, int]] = None,
    fixation_size: int = 40,
    image_size: Optional[Tuple[int, int]] = None,  # raster-only preferred size
    svg_size: Optional[Tuple[int, int]] = None,    # svg rasterization size
    isi: float = 0.0,
    debug: bool = False,
):
    # Enable or disable debug outputs (writing debug PNGs)
    utils.set_debug(debug)
    if duration <= 0:
        raise ValueError("duration must be positive")
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

    # Create window & fixation
    win = utils.setup_window(bg_rgb_255=bg, fullscreen=fullscreen, size=win_size)
    fix = utils.make_fixation_cross(win, size=fixation_size)
    # Create a full-window background patch so we can reliably show a solid
    # background during ISI periods (some backends may retain previous
    # textures if nothing is drawn).
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

    # Logger & quiet console
    logger = EventLogger(output_dir, filename="image_sequence_log.tsv")
    pylogging.console.setLevel(pylogging.CRITICAL)

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
    if isi and isi > 0:
        bg_rect.draw()
        fix.draw()
        isi_flip_ps = win.flip()
        isi_perf = time.perf_counter()
        logger.log(
            "isi_start",
            image_name="",
            requested_duration_s=isi,
            flip_time_psychopy_s=isi_flip_ps,
            flip_time_perf_s=isi_perf,
            end_time_perf_s=isi_perf + isi,
            notes="pre-sequence ISI",
        )
        core.wait(isi)

    # Main loop
    for idx, (img_name, stim) in enumerate(image_stims, start=1):
        stim.draw()
        fix.draw()  # fixation on top
        flip_time_ps = win.flip()
        flip_time_perf = time.perf_counter()
        logger.log(
            "image_on",
            image_name=img_name,
            requested_duration_s=duration,
            flip_time_psychopy_s=flip_time_ps,
            flip_time_perf_s=flip_time_perf,
            end_time_perf_s=None,
            notes=f"index={idx}",
        )
        core.wait(duration)

        # Clear the screen to background (fixation on top) so the ISI shows
        # the gray background with fixation. Some OpenGL backends won't
        # clear previous textures when nothing is drawn, so explicitly draw
        # a full-window rect of the background color, draw fixation, and
        # flip that.
        bg_rect.draw()
        fix.draw()
        win.flip()
        end_time_perf = time.perf_counter()
        logger.log(
            "image_off",
            image_name=img_name,
            requested_duration_s=duration,
            flip_time_psychopy_s=None,
            flip_time_perf_s=None,
            end_time_perf_s=end_time_perf,
            notes=f"index={idx}",
        )
        if isi and isi > 0:
            core.wait(isi)

        # Abort?
        if event.getKeys(["escape"]):
            logger.log("abort", image_name="", notes="escape_pressed")
            break

    # Final fixation and cleanup
    fix.draw()
    final_flip_ps = win.flip()
    final_perf = time.perf_counter()
    logger.log("fixation_post_start", image_name="", requested_duration_s=1.0,
               flip_time_psychopy_s=final_flip_ps, flip_time_perf_s=final_perf,
               end_time_perf_s=final_perf + 1.0, notes="post-sequence fixation")
    core.wait(1.0)

    logger.close()
    win.close()
    core.quit()
    print(f"Finished; log written to {Path(output_dir).resolve()}")


def main():
    args = parse_args()
    # Load config if provided
    cfg = {}
    if args.config:
        cfg = load_config(args.config)
        validate_config(cfg, required=["images_dir", "output_dir", "duration", "n"])
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

    try:
        run_task(
            images_dir=_get("images_dir", args.images_dir),
            n=int(_get("n", args.n)),
            duration=float(_get("duration", args.duration)),
            bg=tuple(_get("bg", args.bg)),
            output_dir=_get("output_dir", args.output_dir),
            seed=_get("seed", args.seed),
            fullscreen=bool(_get("fullscreen", args.fullscreen)),
            win_size=tuple(_get("win_size", tuple(args.win_size) if args.win_size else None)) if _get("win_size", None) else None,
            fixation_size=int(_get("fixation_size", args.fixation_size)),
            image_size=tuple(_get("image_size", tuple(args.image_size) if args.image_size else None)) if _get("image_size", None) else None,
            svg_size=tuple(_get("svg_size", tuple(args.svg_size) if args.svg_size else None)) if _get("svg_size", None) else None,
            isi=float(_get("isi", args.isi if hasattr(args, "isi") else 0.0)),
               debug=args.debug,
        )
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
