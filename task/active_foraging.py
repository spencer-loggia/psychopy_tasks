"""
Active foraging task.

- Loads a color palette TSV (ID, R, G, B)
- Loads a shape-definition TSV (ID, PATH) where PATH must point to an SVG
- Builds shape-color pairs (shape_id, color_id) and rasterizes SVGs recolored
  to each color at `image_size`.
- Presents blocks similar to `afc_block_sequence`: shows stimuli one-at-a-time,
  leaves persistent dots, shows all dots for `choice_time`, then inter-block
  interval `ibi`.

Config keys required/additional:
- colors_tsv: path to TSV file with ID, R, G, B
- shapes_tsv: path to TSV file with ID, PATH (SVG)
- image_size: [W, H]
- num_afc, n, duration, isi, ibi, choice_time, dot_size, dot_color, init_dot_color

"""
import argparse
import sys
import time
import random
from pathlib import Path
from typing import Tuple, Optional, List

from psychopy import core, logging as pylogging, event

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from bin import utils
from bin.logger import EventLogger, MessageLogger
import numpy as np
from bin.config import load_config, validate_config


def parse_args():
    p = argparse.ArgumentParser(description="Active foraging task")
    p.add_argument("--config", help="Path to JSON config file. CLI overrides config keys.")
    p.add_argument("--colors_tsv", help="Path to color TSV (overrides config)")
    p.add_argument("--shapes_tsv", help="Path to shapes TSV (overrides config)")
    p.add_argument("--n", type=int, default=None, help="Number of blocks (overrides config n)")
    p.add_argument("--num_afc", type=int, default=None, help="Number of stimuli per block")
    p.add_argument("--duration", type=float, default=None, help="Stimulus duration (s)")
    p.add_argument("--choice_time", type=float, default=None, help="Choice display time after block (s)")
    p.add_argument("--ibi", type=float, default=None, help="Inter-block interval (s)")
    p.add_argument("--isi", type=float, default=None, help="Pre-block fixation delay before first stim")
    p.add_argument("--dot_size", type=int, default=None, help="Dot size in pixels")
    p.add_argument("--dot_color", type=int, nargs=3, default=None, help="Persistent dot RGB color 0-255")
    p.add_argument("--init_dot_color", type=int, nargs=3, default=None, help="Init pre-stimulus dot RGB color 0-255")
    p.add_argument("--bg", type=int, nargs=3, default=None, help="Background RGB (0-255)")
    p.add_argument("--output_dir", default=None, help="Output dir for logs")
    p.add_argument("--seed", type=int, default=None, help="Random seed")
    p.add_argument("--fullscreen", action="store_true", default=None, help="Fullscreen")
    p.add_argument("--win_size", type=int, nargs=2, default=None, help="Window size when not fullscreen")
    p.add_argument("--image_size", type=int, nargs=2, default=None, help="Raster draw size (W H)")
    p.add_argument("--likelihood_tsv", help="Optional TSV file (color,shape,prob) defining color-shape probabilities")
    p.add_argument("--debug", action="store_true", default=None, help="Enable debug outputs (write debug images to logs/)")
    p.add_argument("--self_initiation", action="store_true", default=None, help="Require participant to self-initiate each block by clicking an onset cue")
    p.add_argument("--fixation_size", type=int, default=None, help="Fixation cross size in pixels; 0 disables fixation")
    p.add_argument("--refresh_rate", type=float, default=None, help="Override detected display refresh rate (Hz); skip auto-detection if provided")
    p.add_argument("--raspi", action="store_true", default=None, help="Enable Raspberry Pi GPIO LED pulses for onset cues")
    p.add_argument("--raspi_pin", type=int, default=None, help="GPIO pin to use for raspi LED pulses (BCM numbering)")
    return p.parse_args()


def run_task(
    colors_tsv: str,
    shapes_tsv: str,
    n_blocks: int,
    num_afc: int,
    duration: float,
    choice_time: float,
    ibi: float,
    isi: float,
    init_dot_color: Optional[Tuple[int, int, int]],
    dot_size: int,
    dot_color: Tuple[int, int, int],
    bg: Tuple[int, int, int],
    output_dir: str,
    seed: Optional[int] = None,
    fullscreen: bool = False,
    win_size: Optional[Tuple[int, int]] = None,
    image_size: Optional[Tuple[int, int]] = None,
    margin: int = 50,
    debug: bool = False,
    likelihood_tsv: Optional[str] = None,
    self_initiation: bool = False,
    fixation_size: Optional[int] = None,
    refresh_rate: Optional[float] = None,
    raspi: bool = False,
    raspi_pin: int = 18,
):
    # Set debug flag before rasterization if requested
    utils.set_debug(debug)

    if seed is not None:
        random.seed(seed)

    colors = utils.load_color_palette(Path(colors_tsv))
    shapes = utils.load_shape_definitions(Path(shapes_tsv))

    # message logger for warnings/debug/info
    msg_logger = MessageLogger(output_dir, filename="active_foraging_message_log.tsv")

    # Build ordered lists of color and shape ids (used to index likelihood matrix)
    color_ids = sorted(colors.keys())
    shape_ids = sorted(shapes.keys())
    n_colors = len(color_ids)
    m_shapes = len(shape_ids)

    # Build all pairs (shape_id, color_id) in the same order as flattened likelihood
    all_pairs: List[Tuple[int, int]] = []
    for sid in shape_ids:
        for cid in color_ids:
            all_pairs.append((sid, cid))

    if num_afc < 1:
        raise ValueError("num_afc must be >= 1")
    if num_afc > len(all_pairs):
        raise ValueError("num_afc cannot be larger than the number of available pairs")

    # Load or construct likelihood distribution (n_colors x m_shapes)
    def load_likelihood(path: Optional[str]):
        # returns numpy array shape (n_colors, m_shapes)
        if path is None:
            arr = np.ones((n_colors, m_shapes), dtype=float)
            total = float(arr.sum())
            msg_logger.log("WARN", f"No likelihood TSV provided; using uniform distribution (sum={total:.6f})")
            arr = arr / total
            return arr

        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Likelihood TSV not found: {path}")

        arr = np.zeros((n_colors, m_shapes), dtype=float)
        import csv

        with p.open("r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            if reader.fieldnames is None:
                # expect three columns without header -> not supported
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
                ci = color_ids.index(cid)
                si = shape_ids.index(sid)
                arr[ci, si] = prob

        total = float(arr.sum())
        if total == 0.0:
            msg_logger.log("WARN", "Likelihood TSV sums to zero; falling back to uniform distribution")
            arr = np.ones((n_colors, m_shapes), dtype=float)
            arr /= arr.sum()
            return arr
        if not np.isclose(total, 1.0):
            msg_logger.log("WARN", f"Likelihood TSV sum is {total:.6f}; normalizing to sum to 1")
            arr = arr / total
        return arr

    likelihood = load_likelihood(likelihood_tsv)

    # Pre-render all pair images (SVG recolored to each color)
    preloaded: dict = {}
    for (sid, cid) in all_pairs:
        svg_path = shapes[sid]
        color = colors[cid]
        pil = utils.rasterize_svg_with_color(svg_path, size_px=tuple(image_size), color_rgb_255=color, bg_rgb_255=bg)
        preloaded[(sid, cid)] = pil

    # Window + background + fixation
    win = utils.setup_window(bg_rgb_255=bg, fullscreen=fullscreen, size=win_size)
    # Measure or override frame rate once per task
    if refresh_rate is not None and float(refresh_rate) > 0:
        fps = float(refresh_rate)
        frame_dur = 1.0 / fps
        try:
            msg_logger.log("INFO", f"fps_override refresh_rate={fps:.6f}Hz frame_dur_s={frame_dur:.9f}")
        except Exception:
            pass
    else:
        fps, frame_dur = utils.detect_frame_rate(win, msg_logger=msg_logger)
    try:
        # Log global timing quantization once based on detected fps
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
    # determine fixation size (allow 0 to disable fixation). If caller
    # didn't provide one, default to 32.
    if fixation_size is None:
        fixation_size = 32
    fix = utils.make_fixation_cross(win, size=fixation_size)

    # If self-initiation requested, build an onset cue ImageStim via utility
    onset_stim = None
    if self_initiation:
        try:
            onset_stim = utils.make_onset_cue_stim(win, bg_rgb_255=bg, size_frac=0.125, cells=8, sigma_frac=0.22, zero_threshold=1)
        except Exception:
            onset_stim = None

    # Create background rectangle via utility
    bg_rect = utils.make_bg_rect(win, bg)

    logger = EventLogger(output_dir, filename="active_foraging_log.tsv")
    pylogging.console.setLevel(pylogging.CRITICAL)

    # Initialize pigpio if requested; do not fail the task if pigpio is unavailable.
    pigpio_pi = None
    if raspi:
        try:
            import pigpio as _pigpio

            pigpio_pi = _pigpio.pi()
            # pigpio.pi() may return a connection object even when daemon isn't running;
            # check for connected attribute if present.
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

    # Pre-sample blocks of pairs according to likelihood distribution
    # Build flat probability list in the same order as all_pairs
    color_to_idx = {cid: i for i, cid in enumerate(color_ids)}
    shape_to_idx = {sid: i for i, sid in enumerate(shape_ids)}
    flat_probs = np.array([likelihood[color_to_idx[cid], shape_to_idx[sid]] for (sid, cid) in all_pairs], dtype=float)
    # numerical safety: ensure non-negative
    flat_probs = np.clip(flat_probs, 0.0, None)
    total_nonzero = np.count_nonzero(flat_probs)
    if flat_probs.sum() <= 0.0:
        msg_logger.log("WARN", "Likelihood distribution sums to zero or all entries are zero; using uniform sampling instead")
        flat_probs = np.ones_like(flat_probs) / float(len(flat_probs))
    else:
        flat_probs = flat_probs / float(flat_probs.sum())

    # seed numpy RNG for reproducibility
    if seed is not None:
        np.random.seed(int(seed))

    blocks = []
    for _ in range(n_blocks):
        if total_nonzero >= num_afc:
            picks = np.random.choice(len(all_pairs), size=num_afc, replace=False, p=flat_probs)
        else:
            # Not enough non-zero entries to sample without replacement; fall back to sampling with replacement
            msg_logger.log("WARN", f"Only {total_nonzero} non-zero pairs available but num_afc={num_afc}; sampling with replacement")
            picks = np.random.choice(len(all_pairs), size=num_afc, replace=True, p=flat_probs)
        blocks.append([all_pairs[int(i)] for i in picks])

    # Task start
    logger.log("task_start", image_name="", notes=f"n_blocks={n_blocks} num_afc={num_afc}")

    for block_idx in range(1, n_blocks + 1):
        logger.log("block_start", image_name="", requested_duration_s=None, flip_time_psychopy_s=None, flip_time_perf_s=time.perf_counter(), end_time_perf_s=None, notes=f"block={block_idx}")

        # No separate pre-block fixation/ISI wait: per-stimulus pre-dot ISI is handled inside the utility.

        block_paths = blocks[block_idx - 1]

        # compute stim size from first pair
        first_pair = block_paths[0]
        pil0 = preloaded[first_pair]
        stim_size = pil0.size

        effective_win_size = tuple(win_size) if win_size is not None else tuple(win.size)

        sampled_positions = utils.sample_non_overlapping_positions(num_afc, stim_size, effective_win_size, margin=margin)
        positions = utils.clamp_positions(sampled_positions, stim_size, effective_win_size, margin=margin)

        for i, (spos, cpos) in enumerate(zip(sampled_positions, positions), start=1):
            # Non-task diagnostic message: use MessageLogger instead of EventLogger
            try:
                msg_logger.log("INFO", f"position_assigned block={block_idx} idx={i} sampled={spos} clamped={cpos}")
            except Exception:
                pass

        # Present this block and handle the single choice using shared utility.
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
            onset_cue=onset_stim,
            msg_logger=msg_logger,
            fps=fps,
            raspi=bool(raspi and pigpio_pi is not None),
            pigpio_pi=pigpio_pi,
            raspi_pin=raspi_pin,
        )
        if aborted:
            return

        # Choice events are already logged inside the utility (choice_made/choice_end);
        # avoid redundant event rows here.

        # ibi (frame-locked)
        if ibi and ibi > 0:
            ibi_frames = int(round(float(ibi) * fps))
            ibi_frames = max(0, ibi_frames)
            ibi_s = ibi_frames / fps
            try:
                msg_logger.log("INFO", f"timing_quantization block={block_idx} ibi={ibi:.6f}s-> {ibi_frames}fr({ibi_s:.6f}s)")
            except Exception:
                pass
            # initial flip
            bg_rect.draw()
            if fix is not None:
                fix.draw()
            ibi_flip = win.flip()
            ibi_perf = time.perf_counter()
            logger.log("ibi_start", image_name="", requested_duration_s=ibi_s, flip_time_psychopy_s=ibi_flip, flip_time_perf_s=ibi_perf, end_time_perf_s=ibi_perf + ibi_s, notes=f"after_block={block_idx}")
            # remaining frames-1
            for _f in range(max(0, ibi_frames - 1)):
                bg_rect.draw()
                if fix is not None:
                    fix.draw()
                win.flip()

        logger.log("block_end", image_name="", notes=f"block={block_idx}")

    # Task end
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
        validate_config(cfg, required=["colors_tsv", "shapes_tsv", "n", "duration"])  # basic
    else:
        missing = []
        if not args.colors_tsv:
            missing.append("--colors_tsv or config")
        if not args.shapes_tsv:
            missing.append("--shapes_tsv or config")
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

    colors_tsv = _get("colors_tsv", cfg.get("colors_tsv"))
    shapes_tsv = _get("shapes_tsv", cfg.get("shapes_tsv"))
    n_blocks = int(_get("n", cfg.get("n")))
    num_afc = int(_get("num_afc", cfg.get("num_afc", 4)))
    duration = float(_get("duration", cfg.get("duration")))
    choice_time = float(_get("choice_time", cfg.get("choice_time", 0.75)))
    ibi = float(_get("ibi", cfg.get("ibi", 1.0)))
    isi = float(_get("isi", cfg.get("isi", 0.5)))
    init_dot_color = tuple(_get("init_dot_color", cfg.get("init_dot_color", None))) if _get("init_dot_color", None) else None
    dot_size = int(_get("dot_size", cfg.get("dot_size", 10)))
    dot_color = tuple(_get("dot_color", cfg.get("dot_color", (155, 155, 155))))
    bg = tuple(_get("bg", cfg.get("bg", (128, 128, 128))))
    output_dir = _get("output_dir", cfg.get("output_dir", "./logs"))
    seed = _get("seed", cfg.get("seed", None))
    fullscreen = bool(_get("fullscreen", cfg.get("fullscreen", False)))
    win_size = tuple(_get("win_size", cfg.get("win_size", None))) if _get("win_size", None) else None
    image_size = tuple(_get("image_size", cfg.get("image_size", None))) if _get("image_size", None) else None
    margin = int(_get("margin", cfg.get("margin", 50)))
    debug = bool(_get("debug", cfg.get("debug", False)))
    likelihood_tsv = _get("likelihood_tsv", cfg.get("likelihood_tsv", None))
    # Accept both 'refresh_rate' and the common misspelling 'refrech_rate' from config
    refresh_rate = _get("refresh_rate", cfg.get("refresh_rate", cfg.get("refrech_rate", None)))
    raspi = _get("raspi", cfg.get("raspi", False))
    raspi_pin = int(_get("raspi_pin", cfg.get("raspi_pin", 18)))
    # stroke options were removed to match utils.rasterize_svg_with_color signature

    try:
        run_task(
            colors_tsv=colors_tsv,
            shapes_tsv=shapes_tsv,
            n_blocks=n_blocks,
            num_afc=num_afc,
            duration=duration,
            choice_time=choice_time,
            ibi=ibi,
            isi=isi,
            init_dot_color=init_dot_color,
            dot_size=dot_size,
            dot_color=dot_color,
            bg=bg,
            output_dir=output_dir,
            seed=seed,
            fullscreen=fullscreen,
            win_size=win_size,
            image_size=image_size,
            margin=margin,
            debug=debug,
            self_initiation=_get("self_initiation", cfg.get("self_initiation", False)),
            fixation_size=_get("fixation_size", cfg.get("fixation_size", None)),
            likelihood_tsv=likelihood_tsv,
            refresh_rate=refresh_rate,
            raspi=_get("raspi", cfg.get("raspi", False)),
            raspi_pin=_get("raspi_pin", cfg.get("raspi_pin", 18)),
        )
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
