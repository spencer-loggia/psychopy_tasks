#!/usr/bin/env python3
"""Generate a gallery image of all base color/shape pairs with reward labels.

- Uses reward levels from reward_space_tsv.
- Renders only one luminance level (default: 21, i.e. max in CSC configs).
- Produces a single contact sheet PNG plus a CSV manifest.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from bin import utils


def _resolve_path(raw: str, config_path: Path) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    from_cfg = (config_path.parent / p).resolve()
    if from_cfg.exists():
        return from_cfg
    return (_PROJECT_ROOT / p).resolve()


def _load_cfg(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _build_color_id_matrix(color_ids: List[int], n_colors: int, n_lum_levels: int) -> np.ndarray:
    return np.array(color_ids, dtype=int).reshape((n_lum_levels, n_colors))


def _build_reward_matrix(reward_space_tsv: Path, n_colors: int, n_shapes: int) -> np.ndarray:
    reward_df = pd.read_csv(reward_space_tsv)
    reward_flat = reward_df.iloc[:, 0].values
    expected_len = n_colors * n_shapes
    if len(reward_flat) != expected_len:
        raise ValueError(
            f"reward_space has {len(reward_flat)} entries but expected {expected_len} (n_colors*n_shapes)"
        )
    return reward_flat.reshape((n_shapes, n_colors)).T


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate color/shape reward gallery")
    parser.add_argument(
        "--config",
        type=str,
        default="test_configs/csc_task_test.json",
        help="Path to task config JSON",
    )
    parser.add_argument(
        "--lum-level",
        type=int,
        default=21,
        help="1-based luminance level to render (default: 21)",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        nargs=2,
        default=[160, 160],
        metavar=("W", "H"),
        help="Rendered stimulus tile size in pixels",
    )
    parser.add_argument(
        "--output-image",
        type=str,
        default="tests/output/reward_pair_gallery_lum21.png",
        help="Output PNG path",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="tests/output/reward_pair_gallery_lum21.csv",
        help="Output CSV manifest path",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = _load_cfg(config_path)

    n_colors = int(cfg["n_colors"])
    n_shapes = int(cfg["n_shapes"])
    n_lum_levels = int(cfg["n_lum_levels"])
    lum_level = int(args.lum_level)
    if lum_level < 1 or lum_level > n_lum_levels:
        raise ValueError(f"lum-level must be in [1, {n_lum_levels}] but got {lum_level}")

    colors_tsv = _resolve_path(cfg["colors_tsv"], config_path)
    shapes_tsv = _resolve_path(cfg["shapes_tsv"], config_path)
    reward_space_tsv = _resolve_path(cfg["reward_space_tsv"], config_path)

    colors: Dict[int, Tuple[int, int, int]] = utils.load_color_palette(colors_tsv)
    _bg_rgb, colors = utils.split_background_from_palette(colors)
    shapes: Dict[int, Path] = utils.load_shape_definitions(shapes_tsv)

    color_ids = list(colors.keys())
    shape_ids = list(shapes.keys())
    if len(color_ids) != n_colors * n_lum_levels:
        raise ValueError(
            f"colors_tsv entries={len(color_ids)} but expected n_colors*n_lum_levels={n_colors*n_lum_levels}"
        )
    if len(shape_ids) != n_shapes:
        raise ValueError(f"shapes_tsv entries={len(shape_ids)} but expected n_shapes={n_shapes}")

    color_id_matrix = _build_color_id_matrix(color_ids, n_colors, n_lum_levels)
    reward_matrix = _build_reward_matrix(reward_space_tsv, n_colors, n_shapes)

    lum_idx = lum_level - 1
    tile_w, tile_h = int(args.tile_size[0]), int(args.tile_size[1])

    font = ImageFont.load_default()
    text_h = 36
    pad = 8
    cell_w = tile_w + pad * 2
    cell_h = tile_h + text_h + pad * 2

    cols = n_colors
    rows = n_shapes

    sheet = Image.new("RGB", (cell_w * cols, cell_h * rows), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)

    manifest_rows = []
    for shape_idx, sid in enumerate(shape_ids):
        shape_svg = shapes[sid]
        for color_idx in range(n_colors):
            cid = int(color_id_matrix[lum_idx, color_idx])
            rgb = colors[cid]
            reward = int(reward_matrix[color_idx, shape_idx])

            pil = utils.rasterize_svg_with_color(
                shape_svg,
                size_px=(tile_w, tile_h),
                color_rgb_255=rgb,
                bg_rgb_255=(255, 255, 255),
            ).convert("RGB")

            r = shape_idx
            c = color_idx
            x0 = c * cell_w + pad
            y0 = r * cell_h + pad
            sheet.paste(pil, (x0, y0))

            label = f"S{shape_idx+1} C{color_idx+1} R={reward}"
            draw.text((x0, y0 + tile_h + 6), label, fill=(0, 0, 0), font=font)

            manifest_rows.append(
                {
                    "shape_idx_1based": shape_idx + 1,
                    "shape_id": sid,
                    "color_idx_1based": color_idx + 1,
                    "color_id": cid,
                    "lum_level_1based": lum_level,
                    "reward": reward,
                }
            )

    out_img = Path(args.output_image)
    out_csv = Path(args.output_csv)
    out_img.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    sheet.save(out_img)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "shape_idx_1based",
                "shape_id",
                "color_idx_1based",
                "color_id",
                "lum_level_1based",
                "reward",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Saved gallery: {out_img}")
    print(f"Saved manifest: {out_csv}")
    print(f"Rendered {len(manifest_rows)} pairs at luminance level {lum_level}")


if __name__ == "__main__":
    main()
