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
    p.add_argument("--bg", type=int, nargs=3, default=(128, 128, 128), help="Background RGB (0-255)")
    p.add_argument("--output_dir", default="./logs", help="Output dir for logs")
    p.add_argument("--seed", type=int, default=None, help="Random seed")
    p.add_argument("--fullscreen", action="store_true", help="Fullscreen")
    p.add_argument("--win_size", type=int, nargs=2, default=None, help="Window size when not fullscreen")
    p.add_argument("--image_size", type=int, nargs=2, default=None, help="Raster draw size (W H)")
    p.add_argument("--likelihood_tsv", help="Optional TSV file (color,shape,prob) defining color-shape probabilities")
    p.add_argument("--debug", action="store_true", help="Enable debug outputs (write debug images to logs/)")
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
            # uniform default
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

    logger = EventLogger(output_dir, filename="active_foraging_log.tsv")
    pylogging.console.setLevel(pylogging.CRITICAL)

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

    for block_idx in range(1, n_blocks + 1):
        logger.log("block_start", image_name="", requested_duration_s=None, flip_time_psychopy_s=None, flip_time_perf_s=time.perf_counter(), end_time_perf_s=None, notes=f"block={block_idx}")

        # Pre-block fixation cue
        if isi and isi > 0:
            bg_rect.draw()
            fix.draw()
            cue_flip = win.flip()
            cue_perf = time.perf_counter()
            logger.log("block_cue", image_name="", requested_duration_s=isi, flip_time_psychopy_s=cue_flip, flip_time_perf_s=cue_perf, end_time_perf_s=cue_perf + isi, notes=f"block={block_idx}")
            core.wait(isi)

        block_paths = blocks[block_idx - 1]

        # compute stim size from first pair
        first_pair = block_paths[0]
        pil0 = preloaded[first_pair]
        stim_size = pil0.size

        effective_win_size = tuple(win_size) if win_size is not None else tuple(win.size)

        sampled_positions = utils.sample_non_overlapping_positions(num_afc, stim_size, effective_win_size, margin=margin)
        positions = utils.clamp_positions(sampled_positions, stim_size, effective_win_size, margin=margin)

        for i, (spos, cpos) in enumerate(zip(sampled_positions, positions), start=1):
            logger.log("position_assigned", image_name=str(block_paths[i - 1]), notes=f"sampled={spos} clamped={cpos} block={block_idx} idx={i}")

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
        )
        if aborted:
            return

        # canonical choice logging
        if choice_info is None:
            logger.log("no_choice", image_name="", notes=f"block={block_idx}")
        else:
            ci = choice_info
            sid, cid = block_paths[ci["chosen_index"] - 1]
            img_name = f"shape{sid}_color{cid}"
            logger.log(
                "choice_registered",
                image_name=img_name,
                requested_duration_s=None,
                flip_time_psychopy_s=None,
                flip_time_perf_s=ci.get("choice_time_perf_s"),
                end_time_perf_s=None,
                notes=f"block={block_idx} idx={ci.get('chosen_index')} click_xy={ci.get('chosen_pos')}",
            )

        # ibi
        if ibi and ibi > 0:
            ibi_flip = win.flip()
            ibi_perf = time.perf_counter()
            logger.log("ibi_start", image_name="", requested_duration_s=ibi, flip_time_psychopy_s=ibi_flip, flip_time_perf_s=ibi_perf, end_time_perf_s=ibi_perf + ibi, notes=f"after_block={block_idx}")
            core.wait(ibi)

        logger.log("block_end", image_name="", notes=f"block={block_idx}")

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
        
        )
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
