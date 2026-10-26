"""Shared stimulus sizing and circular placement for AFC-style tasks."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Tuple

import numpy as np

from bin.screen import rotate_centered_point


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
    if not all(math.isfinite(v) and v > 0 for v in (width_px, height_px)):
        raise ValueError(f"Invalid main screen size for stimulus circle: {effective_win_size}")

    if center_point is None:
        center_px = (width_px / 2.0, height_px / 2.0)
    else:
        if len(center_point) != 2:
            raise ValueError("center_point must contain exactly two pixel coordinates")
        center_px = (float(center_point[0]), float(center_point[1]))

    if not all(math.isfinite(v) for v in center_px):
        raise ValueError("center_point coordinates must be finite")

    closest_edge_px = min(
        center_px[0],
        width_px - center_px[0],
        center_px[1],
        height_px - center_px[1],
    )
    if stim_range_radius is None and closest_edge_px <= 0:
        raise ValueError("Set stim_range_radius explicitly when center_point is on or outside the screen edge")
    radius_px = closest_edge_px / 2.0 if stim_range_radius is None else float(stim_range_radius)
    if not math.isfinite(radius_px) or radius_px <= 0.0:
        raise ValueError("stim_range_radius must be finite and greater than 0 pixels")
    return center_px, radius_px


def screen_px_to_psychopy(
    position_px: Tuple[float, float],
    effective_win_size: Tuple[int, int],
    *,
    field_name: str = "position",
) -> Tuple[float, float]:
    width_px, height_px = float(effective_win_size[0]), float(effective_win_size[1])
    if width_px <= 0.0 or height_px <= 0.0:
        raise ValueError(f"Invalid main screen size: {effective_win_size}")
    if len(position_px) != 2:
        raise ValueError(f"{field_name} must contain exactly two pixel coordinates")
    x_px, y_px = float(position_px[0]), float(position_px[1])
    if not (math.isfinite(x_px) and math.isfinite(y_px)):
        raise ValueError(f"{field_name} coordinates must be finite")
    if not (0.0 <= x_px <= width_px and 0.0 <= y_px <= height_px):
        raise ValueError(
            f"{field_name}=({x_px}, {y_px}) is outside the main screen bounds {effective_win_size}"
        )
    return x_px - (width_px / 2.0), (height_px / 2.0) - y_px


def subject_centered_to_screen_px(
    position: Tuple[float, float],
    effective_win_size: Tuple[int, int],
    rotation_degrees: float,
    *,
    field_name: str = "position",
) -> Tuple[float, float]:
    """Convert participant-centered, y-up pixels to native upper-left pixels.

    Circle centers may lie outside the screen. Call ``screen_px_to_psychopy``
    on the result for visible cues that must have an on-screen center.
    """
    width, height = map(float, effective_win_size)
    if not all(math.isfinite(v) and v > 0 for v in (width, height)):
        raise ValueError(f"Invalid main screen size: {effective_win_size}")
    if len(position) != 2:
        raise ValueError(f"{field_name} must contain exactly two pixel coordinates")
    x, y = map(float, position)
    if not all(math.isfinite(v) for v in (x, y)):
        raise ValueError(f"{field_name} coordinates must be finite")
    x, y = rotate_centered_point((x, y), rotation_degrees)
    return width / 2.0 + x, height / 2.0 - y


def _circle_point(
    center_px: Tuple[float, float],
    radius_px: float,
    angle_rad: float,
) -> Tuple[float, float]:
    return (
        center_px[0] + radius_px * math.cos(angle_rad),
        center_px[1] + radius_px * math.sin(angle_rad),
    )


def _valid_circle_arcs(center_px, radius_px, center_bounds):
    """Intersect a circle with the rectangle of allowed stimulus centers.

    Each boundary intersection partitions the circle into intervals that are
    wholly inside or outside. Sampling an accepted interval never needs an
    edge-rejection loop, even when only a very short arc is visible.
    """
    left, top, right, bottom = center_bounds
    if left >= right or top >= bottom:
        return ()
    cx, cy = center_px
    turns = [0.0, math.tau]
    for x in (left, right):
        cosine = (x - cx) / radius_px
        if -1.0 <= cosine <= 1.0:
            angle = math.acos(cosine)
            turns.extend((angle, math.tau - angle))
    for y in (top, bottom):
        sine = (y - cy) / radius_px
        if -1.0 <= sine <= 1.0:
            angle = math.asin(sine)
            turns.extend((angle % math.tau, (math.pi - angle) % math.tau))
    turns = sorted(set(turns))
    arcs = []
    for start, stop in zip(turns, turns[1:]):
        x, y = _circle_point(center_px, radius_px, (start + stop) / 2.0)
        if left <= x <= right and top <= y <= bottom:
            if arcs and arcs[-1][1] == start:
                arcs[-1] = (arcs[-1][0], stop)
            else:
                arcs.append((start, stop))
    return tuple(arcs)


def _sample_arc_angle(arcs, total_length, rng):
    # Weight intervals by angular length, which is proportional to arc length.
    distance = rng.uniform(0.0, total_length)
    for start, stop in arcs:
        length = stop - start
        if distance <= length:
            return start + distance
        distance -= length
    return arcs[-1][1]  # Floating-point roundoff at the final endpoint.


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
    valid_arcs: Tuple[Tuple[float, float], ...],
    rng: random.Random,
    max_attempts: int = 2000,
) -> List[Tuple[float, float]]:
    positions_px: List[Tuple[float, float]] = []
    total_length = sum(stop - start for start, stop in valid_arcs)
    attempts = 0
    while len(positions_px) < count and attempts < max_attempts:
        attempts += 1
        angle = _sample_arc_angle(valid_arcs, total_length, rng)
        candidate_px = _circle_point(center_px, radius_px, angle)
        if not _has_overlap(candidate_px, positions_px, stim_size):
            positions_px.append(candidate_px)
    if len(positions_px) < count:
        raise RuntimeError(f"Could not place {count} non-overlapping stimuli on the valid stimulus circle arcs")
    return positions_px


@dataclass(frozen=True)
class AFCPlacement:
    """Screen-safe geometry prepared once and sampled for successive trials."""

    center_px: Tuple[float, float]
    radius_px: float
    stim_size: Tuple[float, float]
    effective_win_size: Tuple[float, float]
    num_afc: int
    valid_arcs: Tuple[Tuple[float, float], ...]
    fixed_positions_px: Optional[Tuple[Tuple[float, float], ...]]

    def sample(self, *, rng=None):
        if self.fixed_positions_px is not None:
            sampled_positions_px = list(self.fixed_positions_px)
        else:
            sampled_positions_px = _sample_non_overlapping_circle_positions(
                self.num_afc,
                self.center_px,
                self.radius_px,
                self.stim_size,
                valid_arcs=self.valid_arcs,
                rng=rng if rng is not None else random,
            )
        positions = [
            screen_px_to_psychopy(pos_px, self.effective_win_size)
            for pos_px in sampled_positions_px
        ]
        return sampled_positions_px, positions


def prepare_afc_placement(
    fixed_positions: bool,
    num_afc: int,
    center_point: Optional[Tuple[float, float]],
    stim_range_radius: Optional[float],
    stim_size: Tuple[int, int],
    effective_win_size: Tuple[int, int],
    *,
    display_margin: float = 0.0,
) -> AFCPlacement:
    """Precompute allowed arcs using the stimulus's rotated native pixel bounds.

    ``stim_size`` is the axis-aligned footprint after applying the same software
    rotation as the renderer. The screen size and center use native framebuffer
    coordinates; an equal margin at every edge is invariant under screen turns.
    """
    if isinstance(num_afc, bool) or int(num_afc) != num_afc or num_afc <= 0:
        raise ValueError("num_afc must be a positive integer")
    center_px, radius_px = resolve_stimulus_circle(
        center_point,
        stim_range_radius,
        effective_win_size,
    )
    display_margin = float(display_margin)
    if not math.isfinite(display_margin) or display_margin < 0:
        raise ValueError("display_margin must be finite and non-negative")
    stim_w, stim_h = map(float, stim_size)
    if not all(math.isfinite(v) and v > 0 for v in (stim_w, stim_h)):
        raise ValueError("stim_size dimensions must be finite and positive")
    width, height = map(float, effective_win_size)
    left, top = display_margin + stim_w / 2.0, display_margin + stim_h / 2.0
    right, bottom = width - left, height - top
    valid_arcs = _valid_circle_arcs(center_px, radius_px, (left, top, right, bottom))
    if not valid_arcs:
        raise ValueError(
            "No valid stimulus circle arcs: the full rotated stimulus must fit inside "
            "display_margin. Check center_point, stim_range_radius, image_size, and screen size."
        )
    fixed_positions_px = None
    if fixed_positions:
        spacing_angle = (2.0 * math.pi) / float(num_afc)
        start_angle = (math.pi / 2.0) + (spacing_angle / 2.0)
        sampled_positions_px = [
            _circle_point(center_px, radius_px, start_angle + (idx * spacing_angle))
            for idx in range(num_afc)
        ]
        if any(not (left <= x <= right and top <= y <= bottom) for x, y in sampled_positions_px):
            raise ValueError("Fixed stimulus circle positions place part of a stimulus outside display_margin")
        _assert_non_overlapping_circle_positions(sampled_positions_px, stim_size)
        fixed_positions_px = tuple(sampled_positions_px)
    return AFCPlacement(
        center_px=center_px,
        radius_px=radius_px,
        stim_size=(stim_w, stim_h),
        effective_win_size=(width, height),
        num_afc=num_afc,
        valid_arcs=valid_arcs,
        fixed_positions_px=fixed_positions_px,
    )


def compute_afc_positions(
    fixed_positions: bool,
    num_afc: int,
    center_point: Optional[Tuple[float, float]],
    stim_range_radius: Optional[float],
    stim_size: Tuple[int, int],
    effective_win_size: Tuple[int, int],
    *,
    display_margin: float = 0.0,
    rng: Optional[random.Random] = None,
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """Return screen-pixel and PsychoPy coordinates on screen-safe circle arcs."""
    return prepare_afc_placement(
        fixed_positions, num_afc, center_point, stim_range_radius, stim_size,
        effective_win_size, display_margin=display_margin,
    ).sample(rng=rng)
