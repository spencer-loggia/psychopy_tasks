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
from bin.logger import EventLogger, MessageLogger
from bin.config import load_config, validate_config


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
    parser.add_argument("--raspi", action="store_true", default=None, help="Enable Raspberry Pi GPIO features (no-op if unused)")
    parser.add_argument("--raspi_pin", type=int, default=None, help="GPIO pin to use for raspi features (BCM numbering)")
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
    refresh_rate: Optional[float] = None,
    raspi: bool = False,
    raspi_pin: int = 18,
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
    bg_rect = utils.make_bg_rect(win, bg)

    # Loggers & quiet console
    logger = EventLogger(output_dir, filename="image_sequence_log.tsv")
    msg_logger = MessageLogger(output_dir, filename="image_sequence_message_log.tsv")
    pylogging.console.setLevel(pylogging.CRITICAL)

    # Initialize pigpio if requested (harmless if not used here)
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

    # Global timing quantization
    def _q(seconds: float, at_least_one: bool = False):
        frames = int(round(max(0.0, float(seconds)) * float(fps)))
        if at_least_one:
            frames = max(1, frames)
        return frames, frames / float(fps)

    stim_frames, stim_s = _q(duration, at_least_one=True)
    isi_frames, isi_s = _q(isi, at_least_one=False)
    final_fix_frames, final_fix_s = _q(1.0, at_least_one=False)
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
            isi_flip_ps = win.flip()
            if first_flip:
                isi_perf = time.perf_counter()
                logger.log(
                    "isi_start",
                    image_name="",
                    requested_duration_s=isi_s,
                    flip_time_psychopy_s=isi_flip_ps,
                    flip_time_perf_s=isi_perf,
                    end_time_perf_s=isi_perf + isi_s,
                    notes="pre-sequence ISI",
                )
                first_flip = False

    # Task start
    logger.log("task_start", image_name="", notes=f"n={n}")

    # Main loop
    for idx, (img_name, stim) in enumerate(image_stims, start=1):
        first_flip = True
        for _f in range(stim_frames):
            stim.draw()
            if fix is not None:
                fix.draw()  # fixation on top
            flip_time_ps = win.flip()
            if first_flip:
                flip_time_perf = time.perf_counter()
                logger.log(
                    "image_on",
                    image_name=img_name,
                    requested_duration_s=stim_s,
                    flip_time_psychopy_s=flip_time_ps,
                    flip_time_perf_s=flip_time_perf,
                    end_time_perf_s=None,
                    notes=f"index={idx}",
                )
                first_flip = False
            # Abort?
            if event.getKeys(["escape"]):
                logger.log("abort", image_name="", notes="escape_pressed")
                break
        # Clear the screen to background (fixation on top) and log image_off
        bg_rect.draw()
        if fix is not None:
            fix.draw()
        win.flip()
        end_time_perf = time.perf_counter()
        logger.log(
            "image_off",
            image_name=img_name,
            requested_duration_s=stim_s,
            flip_time_psychopy_s=None,
            flip_time_perf_s=None,
            end_time_perf_s=end_time_perf,
            notes=f"index={idx}",
        )
        # ISI between images
        for _f in range(isi_frames):
            bg_rect.draw()
            if fix is not None:
                fix.draw()
            win.flip()
        # Abort?
        if event.getKeys(["escape"]):
            logger.log("abort", image_name="", notes="escape_pressed")
            break

    # Final fixation and cleanup
    if fix is not None:
        fix.draw()
    final_flip_ps = win.flip()
    final_perf = time.perf_counter()
    # Frame-locked post-sequence fixation
    logger.log("fixation_post_start", image_name="", requested_duration_s=final_fix_s,
               flip_time_psychopy_s=final_flip_ps, flip_time_perf_s=final_perf,
               end_time_perf_s=final_perf + final_fix_s, notes="post-sequence fixation")
    for _f in range(max(0, final_fix_frames - 1)):
        if fix is not None:
            fix.draw()
        bg_rect.draw()
        win.flip()

    logger.log("task_end", image_name="", notes="done")
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
            images_dir=_get("images_dir", cfg.get("images_dir")),
            n=int(_get("n", cfg.get("n", 10))),
            duration=float(_get("duration", cfg.get("duration"))),
            bg=tuple(_get("bg", cfg.get("bg", (128, 128, 128)))),
            output_dir=_get("output_dir", cfg.get("output_dir", "./logs")),
            seed=_get("seed", cfg.get("seed", None)),
            fullscreen=bool(_get("fullscreen", cfg.get("fullscreen", False))),
            win_size=tuple(_get("win_size", cfg.get("win_size", None))) if _get("win_size", None) else None,
            fixation_size=int(_get("fixation_size", cfg.get("fixation_size", 40))),
            image_size=tuple(_get("image_size", cfg.get("image_size", None))) if _get("image_size", None) else None,
            svg_size=tuple(_get("svg_size", cfg.get("svg_size", None))) if _get("svg_size", None) else None,
            isi=float(_get("isi", cfg.get("isi", 0.0))),
            debug=bool(_get("debug", cfg.get("debug", False))),
            refresh_rate=_get("refresh_rate", cfg.get("refresh_rate", cfg.get("refrech_rate", None))),
            raspi=_get("raspi", cfg.get("raspi", False)),
            raspi_pin=_get("raspi_pin", cfg.get("raspi_pin", 18)),
        )
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
