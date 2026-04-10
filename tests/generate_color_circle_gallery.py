#!/usr/bin/env python3
"""Generate a circular color gallery at the middle luminance level.

- Reads `n_colors`, `n_lum_levels`, and `colors_tsv` from a task config JSON.
- Interprets the first row of `colors_tsv` as background gray.
- Uses only the middle luminance level colors and renders them as small blobs
  arranged on a circle.
- Saves a single PNG in the output folder.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageDraw


_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_path(raw: str, config_path: Path) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    from_cfg = (config_path.parent / p).resolve()
    if from_cfg.exists():
        return from_cfg

    from_root = (_PROJECT_ROOT / p).resolve()
    if from_root.exists():
        return from_root

    # Be tolerant to singular/plural file naming mismatch:
    # test_color_definition.tsv <-> test_color_definitions.tsv
    raw_str = str(p)
    variants = [
        raw_str.replace("definitions.tsv", "definition.tsv"),
        raw_str.replace("definition.tsv", "definitions.tsv"),
    ]
    for raw_variant in variants:
        if raw_variant == raw_str:
            continue
        cand_cfg = (config_path.parent / raw_variant).resolve()
        if cand_cfg.exists():
            return cand_cfg
        cand_root = (_PROJECT_ROOT / raw_variant).resolve()
        if cand_root.exists():
            return cand_root

    return from_root


def _load_cfg(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_colors_with_bg(colors_tsv: Path) -> Tuple[Tuple[int, int, int], List[Tuple[int, int, int]]]:
    with colors_tsv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError("Color TSV must have a header row with ID,R,G,B")

        rows: List[Tuple[int, int, int]] = []
        for row in reader:
            try:
                r = int(row.get("R") or row.get("r") or row[reader.fieldnames[1]])
                g = int(row.get("G") or row.get("g") or row[reader.fieldnames[2]])
                b = int(row.get("B") or row.get("b") or row[reader.fieldnames[3]])
            except Exception as e:
                raise ValueError(f"Invalid color row in {colors_tsv}: {row}") from e
            rows.append((r, g, b))

    if len(rows) < 2:
        raise ValueError("colors_tsv must include 1 background row plus at least 1 color row")

    bg = rows[0]
    palette_rows = rows[1:]
    return bg, palette_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate middle-luminance color circle gallery")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to task config JSON",
    )
    parser.add_argument(
        "--output-image",
        type=str,
        default="tests/outputs/color_circle_mid_lum.png",
        help="Output PNG path",
    )
    parser.add_argument(
        "--canvas-size",
        type=int,
        nargs=2,
        default=[1000, 1000],
        metavar=("W", "H"),
        help="Canvas size in pixels",
    )
    parser.add_argument(
        "--blob-radius",
        type=int,
        default=14,
        help="Blob radius in pixels",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = _load_cfg(config_path)

    n_colors = int(cfg["n_colors"])
    n_lum_levels = int(cfg["n_lum_levels"])
    colors_tsv = _resolve_path(cfg["colors_tsv"], config_path)

    bg_rgb, palette_rows = _load_colors_with_bg(colors_tsv)
    expected = n_colors * n_lum_levels
    if len(palette_rows) != expected:
        raise ValueError(
            f"colors_tsv color rows={len(palette_rows)} but expected n_colors*n_lum_levels={expected}"
        )

    # In even cases there are two center levels; pick the lower middle.
    mid_lum_idx = (n_lum_levels - 1) // 2
    start = mid_lum_idx * n_colors
    mid_colors = palette_rows[start : start + n_colors]

    width, height = int(args.canvas_size[0]), int(args.canvas_size[1])
    cx, cy = width // 2, height // 2
    ring_radius = int(min(width, height) * 0.34)
    blob_r = int(args.blob_radius)

    img = Image.new("RGB", (width, height), bg_rgb)
    draw = ImageDraw.Draw(img)

    for i, rgb in enumerate(mid_colors):
        theta = (2.0 * math.pi * i) / float(n_colors)
        x = cx + ring_radius * math.cos(theta)
        y = cy + ring_radius * math.sin(theta)
        bbox = [x - blob_r, y - blob_r, x + blob_r, y + blob_r]
        draw.ellipse(bbox, fill=rgb)

    out_img = Path(args.output_image)
    out_img.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_img)

    print(f"Saved circle gallery: {out_img}")
    print(f"n_colors={n_colors} middle_luminance_level_1based={mid_lum_idx + 1}")


if __name__ == "__main__":
    main()
