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
from typing import Tuple, Optional, List, Dict
import pandas as pd

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
    p.add_argument("--fixed_positions", action="store_true", default=None, help="Fix option positions to fixed locations (only supported for 2 or 4 options)")
    p.add_argument("--position_spacing", type=int, default=None, help="Spacing (pixels) to use for fixed positions; ignored if --fixed_positions not set")
    p.add_argument("--is_memory", action="store_true", default=None, help="If set, items are removed and replaced by dots for the choice period (memory task). If not set, config value or default True is used")
    p.add_argument("--sequential", action="store_true", default=None, help="Present stimuli sequentially (one at a time). If not set, config value or default True is used")
    p.add_argument("--refresh_rate", type=float, default=None, help="Override detected display refresh rate (Hz); skip auto-detection if provided")
    p.add_argument("--raspi", action="store_true", default=None, help="Enable Raspberry Pi GPIO LED pulses for onset cues")
    p.add_argument("--trial_start_pin", type=int, default=None, help="GPIO pin to use for trial start pulses (BCM numbering)")
    p.add_argument("--pump_pin", type=int, default=None, help="GPIO pin for pump reward delivery")
    p.add_argument("--buzz_pin", type=int, default=None, help="GPIO pin for timeout buzzer")
    p.add_argument("--freq_space_tsv", help="CSV file defining color-shape pair probabilities")
    p.add_argument("--reward_space_tsv", help="CSV file defining reward levels for color-shape pairs")
    p.add_argument("--pump_pulse_time_seconds", type=float, default=None, help="Duration of pump pulse in seconds")
    p.add_argument("--timeout_duration_seconds", type=float, default=None, help="Duration of timeout period in seconds")
    p.add_argument("--n_colors", type=int, default=None, help="Expected number of base colors (excluding luminance levels)")
    p.add_argument("--n_shapes", type=int, default=None, help="Expected number of shapes")
    p.add_argument("--n_lum_levels", type=int, default=None, help="Expected number of luminance levels per base color")
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
    trial_start_pin: int = 18,
    pump_pin: int = 17,
    buzz_pin: int = 16,
    fixed_positions: bool = False,
    position_spacing: Optional[int] = None,
    sequential: bool = True,
    is_memory: bool = True,
    freq_space_tsv: Optional[str] = None,
    reward_space_tsv: Optional[str] = None,
    pump_pulse_time_seconds: float = 0.25,
    timeout_duration_seconds: float = 3.0,
    reward_to_pulse_map: Optional[Dict[str, int]] = None,
    reward_to_timeout_map: Optional[Dict[str, int]] = None,
    n_colors_expected: Optional[int] = None,
    n_shapes_expected: Optional[int] = None,
    n_lum_levels: Optional[int] = None,
):
    # Set debug flag before rasterization if requested
    utils.set_debug(debug)

    if seed is not None:
        random.seed(seed)

    colors = utils.load_color_palette(Path(colors_tsv))
    shapes = utils.load_shape_definitions(Path(shapes_tsv))

    # message logger for warnings/debug/info
    msg_logger = MessageLogger(output_dir, filename="active_foraging_message_log.tsv")

    # Keep TSV order (do not sort): color ordering defines luminance grouping.
    color_ids = list(colors.keys())
    shape_ids = list(shapes.keys())

    n_color_defs = len(color_ids)
    n_shape_defs = len(shape_ids)

    # Resolve expected dimensions (supports legacy configs by inferring when omitted)
    if n_shapes_expected is None:
        n_shapes_expected = n_shape_defs
    if n_lum_levels is None:
        n_lum_levels = 1
    if n_colors_expected is None:
        if n_color_defs % int(n_lum_levels) != 0:
            raise ValueError(
                f"Cannot infer n_colors: color definitions={n_color_defs} not divisible by n_lum_levels={n_lum_levels}"
            )
        n_colors_expected = int(n_color_defs // int(n_lum_levels))

    n_colors_expected = int(n_colors_expected)
    n_shapes_expected = int(n_shapes_expected)
    n_lum_levels = int(n_lum_levels)

    if n_colors_expected <= 0 or n_shapes_expected <= 0 or n_lum_levels <= 0:
        raise ValueError("n_colors, n_shapes, and n_lum_levels must all be positive integers")

    expected_color_defs = n_colors_expected * n_lum_levels
    if n_color_defs != expected_color_defs:
        raise ValueError(
            f"colors_tsv has {n_color_defs} definitions, expected n_colors*n_lum_levels={n_colors_expected}*{n_lum_levels}={expected_color_defs}"
        )
    if n_shape_defs != n_shapes_expected:
        raise ValueError(
            f"shapes_tsv has {n_shape_defs} definitions, expected n_shapes={n_shapes_expected}"
        )

    # color_ids are ordered in TSV as:
    # (color_1_lum_1,...,color_n_lum_1),...,(color_1_lum_L,...,color_n_lum_L)
    # Build [lum_idx, base_color_idx] -> color_id table.
    color_id_matrix = np.array(color_ids, dtype=int).reshape((n_lum_levels, n_colors_expected))

    # Base pairs (shape_idx, base_color_idx) are what freq/reward spaces index.
    base_pairs: List[Tuple[int, int]] = []
    for shape_idx in range(n_shapes_expected):
        for color_idx in range(n_colors_expected):
            base_pairs.append((shape_idx, color_idx))

    # All displayable full pairs (shape_id, color_id including luminance)
    all_pairs: List[Tuple[int, int]] = []
    for sid in shape_ids:
        for cid in color_ids:
            all_pairs.append((sid, cid))

    if num_afc < 1:
        raise ValueError("num_afc must be >= 1")
    if num_afc > len(base_pairs):
        raise ValueError("num_afc cannot be larger than the number of available base color-shape pairs")

    # Expected number of color-shape pairs for validation
    expected_len = n_colors_expected * n_shapes_expected

    # Load frequency space (determines probability of each color-shape pair)
    # If freq_space_tsv is provided, use it; otherwise fall back to likelihood_tsv
    if freq_space_tsv is not None:
        freq_path = Path(freq_space_tsv)
        if not freq_path.exists():
            raise FileNotFoundError(f"freq_space_tsv not found: {freq_space_tsv}")
        # Load as specified: first column, reshape to (36, 36), transpose
        freq_df = pd.read_csv(freq_path)
        freq_flat = freq_df.iloc[:, 0].values
        # Validate length matches expected pairs
        if len(freq_flat) != expected_len:
            raise ValueError(
                f"freq_space has {len(freq_flat)} entries but expected n_colors*n_shapes={n_colors_expected}*{n_shapes_expected}={expected_len}"
            )
        # Reshape: user specifies reshape((36, 36)).T, but we need to use actual dims
        freq_matrix = freq_flat.reshape((n_shapes_expected, n_colors_expected)).T  # shape (n_colors, n_shapes)
        likelihood = freq_matrix.astype(float)
        # Normalize
        total = float(likelihood.sum())
        if total <= 0.0:
            msg_logger.log("WARN", "freq_space sums to zero; using uniform distribution")
            likelihood = np.ones((n_colors_expected, n_shapes_expected), dtype=float) / float(expected_len)
        else:
            likelihood = likelihood / total
        msg_logger.log("INFO", f"Loaded freq_space from {freq_space_tsv} (sum={total:.6f})")
    elif likelihood_tsv is not None:
        # Legacy: load from likelihood_tsv (previous format)
        def load_likelihood(path: Optional[str]):
            # returns numpy array shape (n_colors, m_shapes)
            if path is None:
                arr = np.ones((n_colors_expected, n_shapes_expected), dtype=float)
                total = float(arr.sum())
                msg_logger.log("WARN", f"No likelihood TSV provided; using uniform distribution (sum={total:.6f})")
                arr = arr / total
                return arr

            p = Path(path)
            if not p.exists():
                raise FileNotFoundError(f"Likelihood TSV not found: {path}")

            arr = np.zeros((n_colors_expected, n_shapes_expected), dtype=float)
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
                    # Legacy likelihood_tsv addresses full color ids; map to base color index.
                    ci_full = color_ids.index(cid)
                    ci = int(ci_full % n_colors_expected)
                    si = shape_ids.index(sid)
                    arr[ci, si] = prob

            total = float(arr.sum())
            if total == 0.0:
                msg_logger.log("WARN", "Likelihood TSV sums to zero; falling back to uniform distribution")
                arr = np.ones((n_colors_expected, n_shapes_expected), dtype=float)
                arr /= arr.sum()
                return arr
            if not np.isclose(total, 1.0):
                msg_logger.log("WARN", f"Likelihood TSV sum is {total:.6f}; normalizing to sum to 1")
                arr = arr / total
            return arr

        likelihood = load_likelihood(likelihood_tsv)
    else:
        # No freq_space or likelihood provided; use uniform
        likelihood = np.ones((n_colors_expected, n_shapes_expected), dtype=float)
        likelihood /= likelihood.sum()
        msg_logger.log("WARN", "No freq_space_tsv or likelihood_tsv provided; using uniform distribution")

    # Load reward space (maps each color-shape pair to a reward level)
    # Initialize default: all pairs have reward 0
    reward_map: Dict[Tuple[int, int], int] = {}
    for pair in all_pairs:
        reward_map[pair] = 0
    
    if reward_space_tsv is not None:
        reward_path = Path(reward_space_tsv)
        if not reward_path.exists():
            raise FileNotFoundError(f"reward_space_tsv not found: {reward_space_tsv}")
        # Load as specified: first column, reshape to (36, 36), transpose
        reward_df = pd.read_csv(reward_path)
        reward_flat = reward_df.iloc[:, 0].values
        # Validate length matches expected pairs
        if len(reward_flat) != expected_len:
            raise ValueError(
                f"reward_space has {len(reward_flat)} entries but expected n_colors*n_shapes={n_colors_expected}*{n_shapes_expected}={expected_len}"
            )
        # Reshape: same as freq_space
        reward_matrix = reward_flat.reshape((n_shapes_expected, n_colors_expected)).T  # shape (n_colors, n_shapes)
        
        # Build reward_map from matrix
        for shape_idx, sid in enumerate(shape_ids):
            for color_idx in range(n_colors_expected):
                reward_level = int(reward_matrix[color_idx, shape_idx])
                for lum_idx in range(n_lum_levels):
                    cid = int(color_id_matrix[lum_idx, color_idx])
                    reward_map[(sid, cid)] = reward_level
        
        # Validate reward levels match reward_to_pulse_map and reward_to_timeout_map
        unique_rewards = set(reward_map.values())
        if reward_to_pulse_map is not None:
            # Convert keys to int for comparison
            pulse_keys = set(int(k) for k in reward_to_pulse_map.keys())
            if not unique_rewards.issubset(pulse_keys):
                missing = unique_rewards - pulse_keys
                raise ValueError(f"reward_space contains reward levels {missing} not defined in reward_to_pulse_map")
        if reward_to_timeout_map is not None:
            # Convert keys to int for comparison
            timeout_keys = set(int(k) for k in reward_to_timeout_map.keys())
            if not unique_rewards.issubset(timeout_keys):
                missing = unique_rewards - timeout_keys
                raise ValueError(f"reward_space contains reward levels {missing} not defined in reward_to_timeout_map")
        
        msg_logger.log("INFO", f"Loaded reward_space from {reward_space_tsv}, unique rewards: {sorted(unique_rewards)}")
    else:
        msg_logger.log("INFO", "No reward_space_tsv provided; all pairs have reward level 0")

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

    # Initialize lgpio if requested; do not fail the task if lgpio is unavailable.
    pigpio_pi = None  # naming kept for compatibility with presenter API
    gpio_chip = None
    if raspi:
        try:
            import lgpio

            chip = lgpio.gpiochip_open(0)  # 0 is the default chip for RPi5
            # Claim all three pins as outputs
            lgpio.gpio_claim_output(chip, trial_start_pin)
            lgpio.gpio_claim_output(chip, pump_pin)
            lgpio.gpio_claim_output(chip, buzz_pin)
            pigpio_pi = chip  # store chip handle
            gpio_chip = chip  # store for later use
            msg_logger.log("INFO", f"lgpio initialized on chip 0, pins claimed: trial_start={trial_start_pin}, pump={pump_pin}, buzz={buzz_pin}")
        except Exception as e:
            pigpio_pi = None
            gpio_chip = None
            try:
                msg_logger.log("WARN", f"lgpio not available or failed to initialize: {e}; raspi disabled")
            except Exception:
                pass
    else:
        msg_logger.log("INFO", "raspi=False; GPIO pin signals will not be sent (events will be logged only)")

    # Pre-sample blocks by base color-shape probabilities.
    # Luminance is sampled uniformly after choosing base pair.
    flat_probs = np.array([likelihood[color_idx, shape_idx] for (shape_idx, color_idx) in base_pairs], dtype=float)
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
    block_base_meta = []
    for _ in range(n_blocks):
        if total_nonzero >= num_afc:
            picks = np.random.choice(len(base_pairs), size=num_afc, replace=False, p=flat_probs)
        else:
            # Not enough non-zero entries to sample without replacement; fall back to sampling with replacement
            msg_logger.log("WARN", f"Only {total_nonzero} non-zero pairs available but num_afc={num_afc}; sampling with replacement")
            picks = np.random.choice(len(base_pairs), size=num_afc, replace=True, p=flat_probs)

        block_display_pairs: List[Tuple[int, int]] = []
        block_meta: List[Tuple[int, int, int]] = []  # (shape_idx, color_idx, lum_idx)
        for i in picks:
            shape_idx, color_idx = base_pairs[int(i)]
            lum_idx = int(np.random.randint(0, n_lum_levels))
            sid = shape_ids[shape_idx]
            cid = int(color_id_matrix[lum_idx, color_idx])
            block_display_pairs.append((sid, cid))
            block_meta.append((shape_idx, color_idx, lum_idx))

        blocks.append(block_display_pairs)
        block_base_meta.append(block_meta)

    # Task start
    logger.log("task_start", image_name="", notes=f"n_blocks={n_blocks} num_afc={num_afc}")

    for block_idx in range(1, n_blocks + 1):
        logger.log("block_start", image_name="", requested_duration_s=None, flip_time_psychopy_s=None, flip_time_perf_s=time.perf_counter(), end_time_perf_s=None, notes=f"block={block_idx}")

        # No separate pre-block fixation/ISI wait: per-stimulus pre-dot ISI is handled inside the utility.

        block_paths = blocks[block_idx - 1]
        block_meta = block_base_meta[block_idx - 1]

        for opt_idx, ((shape_id, color_id), (shape_idx, base_color_idx, lum_idx)) in enumerate(zip(block_paths, block_meta), start=1):
            logger.log(
                "stimulus_selected",
                image_name=str((shape_id, color_id)),
                requested_duration_s=None,
                flip_time_psychopy_s=None,
                flip_time_perf_s=time.perf_counter(),
                end_time_perf_s=None,
                notes=(
                    f"block={block_idx} idx={opt_idx} shape_id={shape_id} shape_idx={shape_idx} "
                    f"color_id={color_id} base_color_idx={base_color_idx} "
                    f"luminance_level={lum_idx + 1} lum_idx={lum_idx}"
                ),
            )

        # compute stim size from first pair
        first_pair = block_paths[0]
        pil0 = preloaded[first_pair]
        stim_size = pil0.size

        effective_win_size = tuple(win_size) if win_size is not None else tuple(win.size)

        # Determine positions: either fixed grid/left-right or randomly sampled non-overlapping
        if fixed_positions:
            # Only support 2 or 4 options for fixed layout
            if num_afc not in (2, 4):
                raise ValueError("fixed_positions only supported for num_afc == 2 or num_afc == 4")
            # default spacing if not provided
            spacing = int(position_spacing) if position_spacing is not None else 300
            sampled_positions = []
            if num_afc == 2:
                # left and right of fixation along the horizontal axis
                sampled_positions = [(-spacing, 0), (spacing, 0)]
            else:
                # four corners: one in each quadrant (x,y) combinations
                sampled_positions = [(-spacing, spacing), (spacing, spacing), (-spacing, -spacing), (spacing, -spacing)]
            positions = utils.clamp_positions(sampled_positions, stim_size, effective_win_size, margin=margin)
        else:
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
            raspi_pin=trial_start_pin,
            sequential=sequential,
            is_memory=is_memory,
        )
        if aborted:
            return

        # Handle reward/timeout based on choice
        if choice_info is not None:
            chosen_idx = choice_info["chosen_index"]  # 1-based
            chosen_pair = block_paths[chosen_idx - 1]  # (shape_id, color_id)
            reward_level = reward_map.get(chosen_pair, 0)
            chosen_shape_idx, chosen_color_idx, chosen_lum_idx = block_base_meta[block_idx - 1][chosen_idx - 1]
            
            # Log the reward level
            logger.log(
                "reward_determined",
                image_name=str(chosen_pair),
                requested_duration_s=None,
                flip_time_psychopy_s=None,
                flip_time_perf_s=time.perf_counter(),
                end_time_perf_s=None,
                notes=(
                    f"block={block_idx} idx={chosen_idx} pair={chosen_pair} "
                    f"base_color_idx={chosen_color_idx} shape_idx={chosen_shape_idx} "
                    f"lum_idx={chosen_lum_idx} reward_level={reward_level}"
                ),
            )
            
            # Deliver pump pulses if reward_to_pulse_map specifies any
            num_pulses = 0
            if reward_to_pulse_map is not None:
                num_pulses = reward_to_pulse_map.get(str(reward_level), 0)
            
            if num_pulses > 0:
                for pulse_num in range(1, num_pulses + 1):
                    pulse_start_perf = time.perf_counter()
                    if raspi and gpio_chip is not None:
                        try:
                            import lgpio
                            lgpio.gpio_write(gpio_chip, pump_pin, 1)
                        except Exception as e:
                            msg_logger.log("ERROR", f"Failed to set pump_pin high: {e}")
                    
                    logger.log(
                        "pump_pulse_start",
                        image_name="",
                        requested_duration_s=pump_pulse_time_seconds,
                        flip_time_psychopy_s=None,
                        flip_time_perf_s=pulse_start_perf,
                        end_time_perf_s=None,
                        notes=f"block={block_idx} reward_level={reward_level} pulse={pulse_num}/{num_pulses}",
                    )
                    
                    # Wait for pulse duration
                    core.wait(pump_pulse_time_seconds)
                    
                    pulse_end_perf = time.perf_counter()
                    if raspi and gpio_chip is not None:
                        try:
                            import lgpio
                            lgpio.gpio_write(gpio_chip, pump_pin, 0)
                        except Exception as e:
                            msg_logger.log("ERROR", f"Failed to set pump_pin low: {e}")
                    
                    logger.log(
                        "pump_pulse_end",
                        image_name="",
                        requested_duration_s=pump_pulse_time_seconds,
                        flip_time_psychopy_s=None,
                        flip_time_perf_s=pulse_end_perf,
                        end_time_perf_s=None,
                        notes=f"block={block_idx} reward_level={reward_level} pulse={pulse_num}/{num_pulses}",
                    )
                    
                    # Inter-pulse interval: wait between pulses (except after the last pulse)
                    if pulse_num < num_pulses:
                        interval_start_perf = time.perf_counter()
                        logger.log(
                            "pump_inter_pulse_interval_start",
                            image_name="",
                            requested_duration_s=pump_pulse_time_seconds,
                            flip_time_psychopy_s=None,
                            flip_time_perf_s=interval_start_perf,
                            end_time_perf_s=None,
                            notes=f"block={block_idx} reward_level={reward_level} after_pulse={pulse_num}/{num_pulses}",
                        )
                        core.wait(pump_pulse_time_seconds)
                        interval_end_perf = time.perf_counter()
                        logger.log(
                            "pump_inter_pulse_interval_end",
                            image_name="",
                            requested_duration_s=pump_pulse_time_seconds,
                            flip_time_psychopy_s=None,
                            flip_time_perf_s=interval_end_perf,
                            end_time_perf_s=None,
                            notes=f"block={block_idx} reward_level={reward_level} after_pulse={pulse_num}/{num_pulses}",
                        )
            
            # Apply timeout if reward_to_timeout_map specifies it
            apply_timeout = 0
            if reward_to_timeout_map is not None:
                apply_timeout = reward_to_timeout_map.get(str(reward_level), 0)
            
            if apply_timeout > 0:
                timeout_start_perf = time.perf_counter()
                if raspi and gpio_chip is not None:
                    try:
                        import lgpio
                        lgpio.gpio_write(gpio_chip, buzz_pin, 1)
                    except Exception as e:
                        msg_logger.log("ERROR", f"Failed to set buzz_pin high: {e}")
                
                logger.log(
                    "timeout_start",
                    image_name="",
                    requested_duration_s=timeout_duration_seconds,
                    flip_time_psychopy_s=None,
                    flip_time_perf_s=timeout_start_perf,
                    end_time_perf_s=None,
                    notes=f"block={block_idx} reward_level={reward_level}",
                )
                
                # Wait for timeout duration
                core.wait(timeout_duration_seconds)
                
                timeout_end_perf = time.perf_counter()
                if raspi and gpio_chip is not None:
                    try:
                        import lgpio
                        lgpio.gpio_write(gpio_chip, buzz_pin, 0)
                    except Exception as e:
                        msg_logger.log("ERROR", f"Failed to set buzz_pin low: {e}")
                
                logger.log(
                    "timeout_end",
                    image_name="",
                    requested_duration_s=timeout_duration_seconds,
                    flip_time_psychopy_s=None,
                    flip_time_perf_s=timeout_end_perf,
                    end_time_perf_s=None,
                    notes=f"block={block_idx} reward_level={reward_level}",
                )

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
    trial_start_pin = int(_get("trial_start_pin", cfg.get("trial_start_pin", 18)))
    pump_pin = int(_get("pump_pin", cfg.get("pump_pin", 17)))
    buzz_pin = int(_get("buzz_pin", cfg.get("buzz_pin", 16)))
    fixed_positions = bool(_get("fixed_positions", cfg.get("fixed_positions", False)))
    # position_spacing may be omitted; leave as None to let run_task choose a default
    pos_spacing_val = _get("position_spacing", cfg.get("position_spacing", None))
    position_spacing = int(pos_spacing_val) if pos_spacing_val is not None else None
    sequential = bool(_get("sequential", cfg.get("sequential", True)))
    is_memory = bool(_get("is_memory", cfg.get("is_memory", True)))
    
    # New reward system parameters
    freq_space_tsv = _get("freq_space_tsv", cfg.get("freq_space_tsv", None))
    reward_space_tsv = _get("reward_space_tsv", cfg.get("reward_space_tsv", None))
    pump_pulse_time_seconds = float(_get("pump_pulse_time_seconds", cfg.get("pump_pulse_time_seconds", 0.25)))
    timeout_duration_seconds = float(_get("timeout_duration_seconds", cfg.get("timeout_duration_seconds", 3.0)))
    reward_to_pulse_map = cfg.get("reward_to_pulse_map", None)
    reward_to_timeout_map = cfg.get("reward_to_timeout_map", None)
    n_colors_expected = _get("n_colors", cfg.get("n_colors", None))
    n_shapes_expected = _get("n_shapes", cfg.get("n_shapes", None))
    n_lum_levels = _get("n_lum_levels", cfg.get("n_lum_levels", None))
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
            trial_start_pin=trial_start_pin,
            pump_pin=pump_pin,
            buzz_pin=buzz_pin,
            fixed_positions=fixed_positions,
            position_spacing=position_spacing,
            sequential=sequential,
            is_memory=is_memory,
            freq_space_tsv=freq_space_tsv,
            reward_space_tsv=reward_space_tsv,
            pump_pulse_time_seconds=pump_pulse_time_seconds,
            timeout_duration_seconds=timeout_duration_seconds,
            reward_to_pulse_map=reward_to_pulse_map,
            reward_to_timeout_map=reward_to_timeout_map,
            n_colors_expected=n_colors_expected,
            n_shapes_expected=n_shapes_expected,
            n_lum_levels=n_lum_levels,
        )
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
