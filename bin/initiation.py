"""Shared configuration helpers for self-initiated tasks."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw


INITIATION_CONFIG_KEYS = (
    "initiation_cue_center_position",
    "initiation_cue_style",
    "hold_before_choice",
    "hold_cue_to_init_time_s",
)


@dataclass(frozen=True)
class InitiationConfig:
    cue_center_position: Optional[Tuple[float, float]]
    cue_style: str
    hold_before_choice: bool
    hold_cue_to_init_time_s: float


def build_initiation_cue_image(
    size_px: int,
    *,
    bg_rgb_255: Tuple[int, int, int],
    style: str = "checker",
    cue_color: Optional[Tuple[int, int, int]] = None,
    cells: int = 8,
    sigma_frac: float = 0.22,
    zero_threshold: int = 1,
    pressed: bool = False,
) -> Image.Image:
    """Build the RGBA pixels for a checker or Gaussian blob initiation cue."""
    size_px = max(4, int(size_px))
    style = str(style).strip().lower()
    if style not in {"checker", "blob"}:
        raise ValueError("style must be 'checker' or 'blob'")

    darken_by = 32 if pressed else 0

    def darkened(rgb):
        if len(rgb) != 3:
            raise ValueError("cue colors must contain three RGB values")
        return tuple(max(0, min(255, int(channel)) - darken_by) for channel in rgb)

    if style == "checker":
        cue_image = Image.new("RGB", (size_px, size_px), color=darkened(bg_rgb_255))
        draw = ImageDraw.Draw(cue_image)
        cell = max(2, size_px // int(cells))
        for y in range(0, size_px, cell):
            for x in range(0, size_px, cell):
                xi = x // cell
                yi = y // cell
                fill = (0, 0, 0) if ((xi + yi) % 2 == 0) else darkened((255, 255, 255))
                draw.rectangle([x, y, x + cell - 1, y + cell - 1], fill=fill)
    else:
        cue_image = Image.new(
            "RGB",
            (size_px, size_px),
            color=darkened(cue_color if cue_color is not None else (128, 128, 128)),
        )

    center = (size_px - 1) / 2.0
    sigma = max(2.0, size_px * float(sigma_frac))
    yy, xx = np.mgrid[0:size_px, 0:size_px]
    gaussian = np.exp(
        -0.5 * (((xx - center) / sigma) ** 2 + ((yy - center) / sigma) ** 2)
    )
    mask = np.clip(gaussian * 255.0, 0, 255).astype(np.uint8)
    if zero_threshold is not None and zero_threshold > 0:
        mask[mask <= int(zero_threshold)] = 0
    cue_image.putalpha(Image.fromarray(mask))
    return cue_image


def resolve_initiation_config(
    cfg: Mapping[str, Any],
    *,
    require_keys: bool = True,
) -> InitiationConfig:
    """Validate and normalize the initiation settings shared by AFC tasks."""
    if require_keys:
        for key in INITIATION_CONFIG_KEYS:
            if key not in cfg:
                raise KeyError(f"Missing required config key: '{key}'")

    position_value = cfg.get("initiation_cue_center_position")
    cue_center_position = None
    if position_value is not None:
        if not isinstance(position_value, (list, tuple)) or len(position_value) != 2:
            raise ValueError(
                "Config field 'initiation_cue_center_position' must be null or [x, y]"
            )
        coordinates = []
        for coordinate in position_value:
            if isinstance(coordinate, bool) or not isinstance(coordinate, (int, float)):
                raise ValueError(
                    "Config field 'initiation_cue_center_position' coordinates must be numbers"
                )
            coordinate = float(coordinate)
            if not math.isfinite(coordinate):
                raise ValueError(
                    "Config field 'initiation_cue_center_position' coordinates must be finite"
                )
            coordinates.append(coordinate)
        cue_center_position = (coordinates[0], coordinates[1])

    cue_style_value = cfg.get("initiation_cue_style", "checker")
    if not isinstance(cue_style_value, str):
        raise ValueError("Config field 'initiation_cue_style' must be 'checker' or 'blob'")
    cue_style = cue_style_value.strip().lower()
    if cue_style not in {"checker", "blob"}:
        raise ValueError("Config field 'initiation_cue_style' must be 'checker' or 'blob'")

    hold_before_choice = cfg.get("hold_before_choice", False)
    if not isinstance(hold_before_choice, bool):
        raise ValueError("Config field 'hold_before_choice' must be a boolean")

    hold_time_value = cfg.get("hold_cue_to_init_time_s", 0.0)
    if isinstance(hold_time_value, bool) or not isinstance(hold_time_value, (int, float)):
        raise ValueError("Config field 'hold_cue_to_init_time_s' must be a number")
    hold_cue_to_init_time_s = float(hold_time_value)
    if not math.isfinite(hold_cue_to_init_time_s) or hold_cue_to_init_time_s < 0.0:
        raise ValueError(
            "Config field 'hold_cue_to_init_time_s' must be a finite non-negative number"
        )

    return InitiationConfig(
        cue_center_position=cue_center_position,
        cue_style=cue_style,
        hold_before_choice=hold_before_choice,
        hold_cue_to_init_time_s=hold_cue_to_init_time_s,
    )
