"""Pure file loaders shared by visual tasks and offline tooling."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Tuple


def load_color_palette(tsv_path: str | Path) -> Dict[int, Tuple[int, int, int]]:
    """Load a color TSV and preserve its row order."""
    path = Path(tsv_path)
    if not path.exists():
        raise FileNotFoundError(f"Color TSV not found: {tsv_path}")

    colors: Dict[int, Tuple[int, int, int]] = {}
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError("Color TSV must have header with columns ID,R,G,B")
        for row in reader:
            try:
                color_id = int(
                    row.get("id")
                    or row.get("ID")
                    or row.get("Id")
                    or row.get(reader.fieldnames[0])
                )
                red = int(row.get("r") or row.get("R") or row.get(reader.fieldnames[1]))
                green = int(row.get("g") or row.get("G") or row.get(reader.fieldnames[2]))
                blue = int(row.get("b") or row.get("B") or row.get(reader.fieldnames[3]))
            except Exception as exc:
                raise ValueError(f"Invalid row in color TSV: {row}") from exc
            if color_id in colors:
                raise ValueError(f"Duplicate color ID in TSV: {color_id}")
            colors[color_id] = (red, green, blue)
    return colors


def split_background_from_palette(
    colors: Dict[int, Tuple[int, int, int]],
) -> Tuple[Tuple[int, int, int], Dict[int, Tuple[int, int, int]]]:
    """Return the first palette row as background and all later rows as colors."""
    if not colors:
        raise ValueError(
            "colors_tsv is empty; expected at least background row plus color definitions"
        )
    ordered_items = list(colors.items())
    background = ordered_items[0][1]
    remaining = dict(ordered_items[1:])
    if not remaining:
        raise ValueError(
            "colors_tsv must include at least one color definition after the background row"
        )
    return background, remaining


def load_shape_definitions(tsv_path: str | Path) -> Dict[int, Path]:
    """Load a shape TSV and validate each referenced SVG."""
    path = Path(tsv_path)
    if not path.exists():
        raise FileNotFoundError(f"Shape TSV not found: {tsv_path}")

    shapes: Dict[int, Path] = {}
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError("Shape TSV must have header with columns ID,PATH")
        for row in reader:
            try:
                shape_id = int(row.get("id") or row.get("ID") or row.get(reader.fieldnames[0]))
                path_value = row.get("path") or row.get("PATH") or row.get(reader.fieldnames[1])
                if path_value is None:
                    raise ValueError("Missing path column")
                shape_path = Path(path_value)
                if not shape_path.exists():
                    raise FileNotFoundError(f"Shape file does not exist: {shape_path}")
                if shape_path.suffix.lower() != ".svg":
                    raise ValueError(f"Shape file must be SVG: {shape_path}")
            except Exception as exc:
                raise ValueError(f"Invalid row in shape TSV: {row}") from exc
            if shape_id in shapes:
                raise ValueError(f"Duplicate shape ID in TSV: {shape_id}")
            shapes[shape_id] = shape_path
    return shapes
