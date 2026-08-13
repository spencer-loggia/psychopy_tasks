"""Shared stimulus sizing and circular placement for AFC-style tasks."""
from __future__ import annotations

import math
import random
from typing import Any, List, Mapping, Optional, Tuple

import numpy as np


StimulusKey = Tuple[int, Optional[int]]


def stimulus_size(
    preloaded: Mapping[StimulusKey, Any],
    stimulus: StimulusKey,
) -> Tuple[int, int]:
    item = preloaded[stimulus]
    if isinstance(item, np.ndarray):
        if item.ndim < 2:
            raise ValueError(f"Invalid preloaded stimulus array shape: {item.shape}")
        return int(item.shape[1]), int(item.shape[0])
    return int(item.size[0]), int(item.size[1])


def resolve_stimulus_circle(
    center_point: Optional[Tuple[float, float]],
    stim_range_radius: Optional[float],
    effective_win_size: Tuple[int, int],
) -> Tuple[Tuple[float, float], float]:
    width_px, height_px = float(effective_win_size[0]), float(effective_win_size[1])
    if width_px <= 0 or height_px <= 0:
        raise ValueError(f"Invalid main screen size for stimulus circle: {effective_win_size}")

    if center_point is None:
        center_px = (width_px / 2.0, height_px / 2.0)
    else:
        if len(center_point) != 2:
            raise ValueError("center_point must contain exactly two pixel coordinates")
        center_px = (float(center_point[0]), float(center_point[1]))

    if not (0.0 <= center_px[0] <= width_px and 0.0 <= center_px[1] <= height_px):
        raise ValueError(
            f"center_point={center_px} is outside the main screen bounds {effective_win_size}"
        )

    closest_edge_px = min(
        center_px[0],
        width_px - center_px[0],
        center_px[1],
        height_px - center_px[1],
    )
    radius_px = closest_edge_px / 2.0 if stim_range_radius is None else float(stim_range_radius)
    if radius_px <= 0.0:
        raise ValueError("stim_range_radius must be greater than 0 pixels")
    if radius_px > closest_edge_px:
        raise ValueError("stim_range_radius places the stimulus circle outside the main screen bounds")
    return center_px, radius_px


def _screen_px_to_psychopy(
    position_px: Tuple[float, float],
    effective_win_size: Tuple[int, int],
) -> Tuple[float, float]:
    width_px, height_px = float(effective_win_size[0]), float(effective_win_size[1])
    return position_px[0] - (width_px / 2.0), (height_px / 2.0) - position_px[1]


def _circle_point(
    center_px: Tuple[float, float],
    radius_px: float,
    angle_rad: float,
) -> Tuple[float, float]:
    return (
        center_px[0] + radius_px * math.cos(angle_rad),
        center_px[1] + radius_px * math.sin(angle_rad),
    )


def _has_overlap(
    position_px: Tuple[float, float],
    placed_px: List[Tuple[float, float]],
    stim_size: Tuple[int, int],
) -> bool:
    stim_w, stim_h = float(stim_size[0]), float(stim_size[1])
    return any(
        abs(position_px[0] - placed_x) < stim_w
        and abs(position_px[1] - placed_y) < stim_h
        for placed_x, placed_y in placed_px
    )


def _assert_non_overlapping_circle_positions(
    positions_px: List[Tuple[float, float]],
    stim_size: Tuple[int, int],
) -> None:
    placed_px: List[Tuple[float, float]] = []
    for position_px in positions_px:
        if _has_overlap(position_px, placed_px, stim_size):
            raise ValueError(
                "Fixed stimulus circle positions overlap; increase stim_range_radius or reduce stimulus size"
            )
        placed_px.append(position_px)


def _sample_non_overlapping_circle_positions(
    count: int,
    center_px: Tuple[float, float],
    radius_px: float,
    stim_size: Tuple[int, int],
    *,
    rng: random.Random,
    max_attempts: int = 2000,
) -> List[Tuple[float, float]]:
    positions_px: List[Tuple[float, float]] = []
    attempts = 0
    while len(positions_px) < count and attempts < max_attempts:
        attempts += 1
        candidate_px = _circle_point(center_px, radius_px, rng.uniform(0.0, 2.0 * math.pi))
        if not _has_overlap(candidate_px, positions_px, stim_size):
            positions_px.append(candidate_px)
    if len(positions_px) < count:
        raise RuntimeError(f"Could not place {count} non-overlapping stimuli on the stimulus circle")
    return positions_px


def compute_afc_positions(
    fixed_positions: bool,
    num_afc: int,
    center_point: Optional[Tuple[float, float]],
    stim_range_radius: Optional[float],
    stim_size: Tuple[int, int],
    effective_win_size: Tuple[int, int],
    *,
    rng: Optional[random.Random] = None,
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """Return screen-pixel and PsychoPy coordinates on a non-overlapping circle."""
    center_px, radius_px = resolve_stimulus_circle(
        center_point,
        stim_range_radius,
        effective_win_size,
    )
    if fixed_positions:
        spacing_angle = (2.0 * math.pi) / float(num_afc)
        start_angle = (math.pi / 2.0) + (spacing_angle / 2.0)
        sampled_positions_px = [
            _circle_point(center_px, radius_px, start_angle + (idx * spacing_angle))
            for idx in range(num_afc)
        ]
        _assert_non_overlapping_circle_positions(sampled_positions_px, stim_size)
    else:
        sampled_positions_px = _sample_non_overlapping_circle_positions(
            num_afc,
            center_px,
            radius_px,
            stim_size,
            rng=rng or random,
        )

    positions = [
        _screen_px_to_psychopy(pos_px, effective_win_size)
        for pos_px in sampled_positions_px
    ]
    return sampled_positions_px, positions
