"""Shared color/shape stimulus-space helpers for AFC-style tasks."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
from PIL import Image

from bin.stimulus_files import load_color_palette, load_shape_definitions


StimulusKey = Tuple[int, Optional[int]]
StimulusMeta = Tuple[int, Optional[int], Optional[int]]


@dataclass(frozen=True)
class AFCStimulusSpace:
    bg: Tuple[int, int, int]
    shapes: Dict[int, Path]
    colors: Dict[int, Tuple[int, int, int]]
    n_colors: int
    n_lum_levels: int
    color_id_matrix: np.ndarray
    stimuli: Tuple[StimulusKey, ...]
    metadata: Dict[StimulusKey, StimulusMeta]

    @property
    def native_svg_mode(self) -> bool:
        return self.n_colors == 0


def load_afc_stimulus_space(
    *,
    colors_tsv: str | Path,
    shapes_tsv: str | Path,
    n_colors: int,
    n_shapes: int,
    n_lum_levels: int,
) -> AFCStimulusSpace:
    """Load and validate the full displayable stimulus space.

    ``n_colors == 0`` selects native-SVG mode: the color TSV must contain only
    the background row and SVG appearance is preserved as authored.
    """
    n_colors = int(n_colors)
    n_shapes = int(n_shapes)
    n_lum_levels = int(n_lum_levels)
    if n_colors < 0:
        raise ValueError("n_colors must be zero or greater")
    if n_shapes <= 0:
        raise ValueError("n_shapes must be a positive integer")
    if n_lum_levels < 0:
        raise ValueError("n_lum_levels must be zero or greater")

    palette = load_color_palette(Path(colors_tsv))
    if not palette:
        raise ValueError("colors_tsv must include one background row")
    ordered_palette = list(palette.items())
    bg = tuple(ordered_palette[0][1])
    colors = dict(ordered_palette[1:])
    shapes = load_shape_definitions(Path(shapes_tsv))
    if len(shapes) != n_shapes:
        raise ValueError(
            f"shapes_tsv has {len(shapes)} definitions, expected n_shapes={n_shapes}"
        )

    shape_ids = list(shapes)
    metadata: Dict[StimulusKey, StimulusMeta] = {}
    stimuli = []
    if n_colors == 0:
        if colors:
            raise ValueError(
                "n_colors=0 requires colors_tsv to contain only the background row"
            )
        color_id_matrix = np.empty((0, 0), dtype=int)
        for shape_idx, shape_id in enumerate(shape_ids):
            key = (int(shape_id), None)
            stimuli.append(key)
            metadata[key] = (shape_idx, None, None)
        effective_lum_levels = 0
    else:
        if n_lum_levels <= 0:
            raise ValueError("n_lum_levels must be positive when n_colors is positive")
        expected_colors = n_colors * n_lum_levels
        if len(colors) != expected_colors:
            raise ValueError(
                f"colors_tsv has {len(colors)} color definitions after the background row; "
                f"expected n_colors*n_lum_levels={n_colors}*{n_lum_levels}={expected_colors}"
            )
        color_id_matrix = np.array(list(colors), dtype=int).reshape(
            (n_lum_levels, n_colors)
        )
        for shape_idx, shape_id in enumerate(shape_ids):
            for lum_idx in range(n_lum_levels):
                for color_idx in range(n_colors):
                    color_id = int(color_id_matrix[lum_idx, color_idx])
                    key = (int(shape_id), color_id)
                    stimuli.append(key)
                    metadata[key] = (shape_idx, color_idx, lum_idx)
        effective_lum_levels = n_lum_levels

    return AFCStimulusSpace(
        bg=bg,
        shapes=shapes,
        colors=colors,
        n_colors=n_colors,
        n_lum_levels=effective_lum_levels,
        color_id_matrix=color_id_matrix,
        stimuli=tuple(stimuli),
        metadata=metadata,
    )


def render_afc_stimulus(
    stimulus: StimulusKey,
    *,
    shapes: Dict[int, Path],
    colors: Dict[int, Tuple[int, int, int]],
    image_size: Tuple[int, int],
    bg: Tuple[int, int, int],
    stroke_width: Optional[float] = None,
    stroke_color: Optional[Tuple[int, int, int]] = None,
    stroke_linejoin: Optional[str] = None,
    stroke_linecap: Optional[str] = None,
) -> Image.Image:
    """Render one colored stimulus or one native SVG stimulus."""
    from bin import utils

    shape_id, color_id = stimulus
    shape_path = shapes[int(shape_id)]
    if color_id is None:
        return utils.rasterize_svg(
            shape_path,
            size_px=image_size,
            bg_rgb_255=bg,
        )
    return utils.rasterize_svg_with_color(
        shape_path,
        size_px=image_size,
        color_rgb_255=colors[int(color_id)],
        bg_rgb_255=bg,
        stroke_rgb_255=stroke_color,
        stroke_width_px=stroke_width,
        stroke_linejoin=stroke_linejoin,
        stroke_linecap=stroke_linecap,
    )


def render_afc_stimuli(
    stimuli: Iterable[StimulusKey],
    **render_kwargs,
) -> Dict[StimulusKey, np.ndarray]:
    """Render unique stimuli to serializable RGBA arrays."""
    rendered: Dict[StimulusKey, np.ndarray] = {}
    for stimulus in stimuli:
        key = (int(stimulus[0]), None if stimulus[1] is None else int(stimulus[1]))
        if key not in rendered:
            rendered[key] = np.asarray(render_afc_stimulus(key, **render_kwargs))
    return rendered


def stimulus_to_json(stimulus: StimulusKey) -> list[Optional[int]]:
    return [int(stimulus[0]), None if stimulus[1] is None else int(stimulus[1])]


def stimulus_from_json(value) -> StimulusKey:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"Invalid serialized stimulus key: {value!r}")
    return int(value[0]), None if value[1] is None else int(value[1])


def stimulus_storage_key(stimulus: StimulusKey) -> str:
    return f"{int(stimulus[0])}:{'none' if stimulus[1] is None else int(stimulus[1])}"


def stimulus_from_storage_key(value: str) -> StimulusKey:
    shape_token, color_token = str(value).split(":", 1)
    return int(shape_token), None if color_token == "none" else int(color_token)
