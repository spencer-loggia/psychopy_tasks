"""Shared stimulus sizing and screen-safe placement for AFC-style tasks."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union


DisplayMargin = Union[float, Sequence[float]]


def normalize_display_margin(value: DisplayMargin) -> Tuple[float, float, float, float]:
    """Return finite, non-negative participant-view left/right/top/bottom margins."""
    values = value if isinstance(value, (list, tuple)) else (value,) * 4
    message = "display_margin must be a non-negative number or four finite, non-negative values (left, right, top, bottom)"
    if len(values) != 4:
        raise ValueError(message)
    try:
        margins = tuple(float(v) for v in values)
    except (TypeError, ValueError):
        raise ValueError(message) from None
    if not all(math.isfinite(v) and v >= 0 for v in margins):
        raise ValueError(message)
    return margins


def normalize_placement_distribution(value: str = "circle") -> str:
    """Validate the mode before callers parse mode-specific geometry."""
    if not isinstance(value, str):
        raise ValueError("placement_distribution must be 'circle' or 'uniform'")
    normalized = value.strip().lower()
    if normalized not in ("circle", "uniform"):
        raise ValueError("placement_distribution must be 'circle' or 'uniform'")
    return normalized


def resolve_stimulus_circle(
    center_point: Optional[Tuple[float, float]],
    stim_range_radius: Optional[float],
    effective_win_size: Tuple[int, int],
) -> Tuple[Tuple[float, float], float]:
    width_px, height_px = float(effective_win_size[0]), float(effective_win_size[1])
    if not all(math.isfinite(v) and v > 0 for v in (width_px, height_px)):
        raise ValueError(f"Invalid main screen size for stimulus circle: {effective_win_size}")

    if center_point is None:
        center_px = (0.0, 0.0)
    else:
        if len(center_point) != 2:
            raise ValueError("center_point must contain exactly two pixel coordinates")
        center_px = (float(center_point[0]), float(center_point[1]))

    if not all(math.isfinite(v) for v in center_px):
        raise ValueError("center_point coordinates must be finite")

    closest_edge_px = min(
        width_px / 2.0 - abs(center_px[0]),
        height_px / 2.0 - abs(center_px[1]),
    )
    if stim_range_radius is None and closest_edge_px <= 0:
        raise ValueError("Set stim_range_radius explicitly when center_point is on or outside the screen edge")
    radius_px = closest_edge_px / 2.0 if stim_range_radius is None else float(stim_range_radius)
    if not math.isfinite(radius_px) or radius_px <= 0.0:
        raise ValueError("stim_range_radius must be finite and greater than 0 pixels")
    return center_px, radius_px


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
    left, bottom, right, top = center_bounds
    if left >= right or bottom >= top:
        return ()
    cx, cy = center_px
    turns = [0.0, math.tau]
    for x in (left, right):
        cosine = (x - cx) / radius_px
        if -1.0 <= cosine <= 1.0:
            angle = math.acos(cosine)
            turns.extend((angle, math.tau - angle))
    for y in (bottom, top):
        sine = (y - cy) / radius_px
        if -1.0 <= sine <= 1.0:
            angle = math.asin(sine)
            turns.extend((angle % math.tau, (math.pi - angle) % math.tau))
    turns = sorted(set(turns))
    arcs = []
    for start, stop in zip(turns, turns[1:]):
        x, y = _circle_point(center_px, radius_px, (start + stop) / 2.0)
        if left <= x <= right and bottom <= y <= top:
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
        abs(position_px[0] - placed_x) <= stim_w
        and abs(position_px[1] - placed_y) <= stim_h
        for placed_x, placed_y in placed_px
    )


def assert_non_overlapping_positions(
    positions_px: List[Tuple[float, float]],
    stim_size: Tuple[int, int],
) -> None:
    """Reject intersecting or touching equal-sized, axis-aligned hit boxes."""
    placed_px: List[Tuple[float, float]] = []
    for position_px in positions_px:
        if _has_overlap(position_px, placed_px, stim_size):
            raise ValueError(
                "Stimulus touch hit boxes overlap; increase spacing or reduce stimulus/hit-box size"
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


def _sample_non_overlapping_uniform_positions(
    count: int,
    center_bounds: Tuple[float, float, float, float],
    stim_size: Tuple[float, float],
    *,
    rng: random.Random,
    max_attempts: int = 2000,
    exclusion_center=None,
    exclusion_radius=0.0,
) -> List[Tuple[float, float]]:
    """Sample the full safe rectangle, rejecting collisions with earlier items.

    Each proposal is uniform in area. Rejection conditions the draw on the
    cue exclusion disk and separation from earlier hit boxes.
    """
    left, bottom, right, top = center_bounds
    positions_px: List[Tuple[float, float]] = []
    exclusion_radius_squared = exclusion_radius * exclusion_radius
    for _ in range(max_attempts):
        candidate = (rng.uniform(left, right), rng.uniform(bottom, top))
        if exclusion_center is not None and (
            (candidate[0] - exclusion_center[0]) ** 2
            + (candidate[1] - exclusion_center[1]) ** 2
        ) < exclusion_radius_squared:
            continue
        if not _has_overlap(candidate, positions_px, stim_size):
            positions_px.append(candidate)
            if len(positions_px) == count:
                return positions_px
    raise RuntimeError(
        f"Could not place {count} non-overlapping stimuli in the uniform placement area; "
        "reduce num_afc, image_size, hit-box size, display_margin, or the exclusion radius"
    )


@dataclass(frozen=True)
class AFCPlacement:
    """Screen-safe geometry prepared once and sampled for successive trials."""

    center_px: Optional[Tuple[float, float]]
    radius_px: Optional[float]
    stim_size: Tuple[float, float]
    effective_win_size: Tuple[float, float]
    num_afc: int
    valid_arcs: Tuple[Tuple[float, float], ...]
    fixed_positions_px: Optional[Tuple[Tuple[float, float], ...]]
    placement_distribution: str = "circle"
    center_bounds: Optional[Tuple[float, float, float, float]] = None
    separation_size: Optional[Tuple[float, float]] = None
    exclusion_center: Optional[Tuple[float, float]] = None
    exclusion_radius: float = 0.0

    def sample(self, *, rng=None, max_attempts=2000):
        if self.placement_distribution == "uniform":
            sampled_positions_px = _sample_non_overlapping_uniform_positions(
                self.num_afc,
                self.center_bounds,
                self.separation_size or self.stim_size,
                rng=rng if rng is not None else random,
                exclusion_center=self.exclusion_center,
                exclusion_radius=self.exclusion_radius,
                max_attempts=max_attempts,
            )
        elif self.fixed_positions_px is not None:
            sampled_positions_px = list(self.fixed_positions_px)
        else:
            sampled_positions_px = _sample_non_overlapping_circle_positions(
                self.num_afc,
                self.center_px,
                self.radius_px,
                self.separation_size or self.stim_size,
                valid_arcs=self.valid_arcs,
                rng=rng if rng is not None else random,
                max_attempts=max_attempts,
            )
        return sampled_positions_px


def prepare_afc_placement(
    fixed_positions: bool,
    num_afc: int,
    center_point: Optional[Tuple[float, float]],
    stim_range_radius: Optional[float],
    stim_size: Tuple[int, int],
    effective_win_size: Tuple[int, int],
    *,
    display_margin: DisplayMargin = 0.0,
    placement_distribution: str = "circle",
    choice_hitbox_scale: float = 1.0,
    initiation_cue_position_px: Optional[Tuple[float, float]] = None,
    fixed_layout_rotation_degrees: float = 0.0,
) -> AFCPlacement:
    """Precompute safe center bounds and, for circle placement, allowed arcs.

    Sizes, centers and returned positions are participant-view pixels. The
    actual screen center is always (0, 0), with X right and Y up.
    ``display_margin`` is one value for every edge or (left, right, top, bottom).
    ``uniform`` excludes centers within ``stim_range_radius`` of the initiation
    cue (screen center by default). Hit-box spacing is separate from image-edge
    clearance, and uses the same scale as the choice presenter.
    """
    placement_distribution = normalize_placement_distribution(placement_distribution)
    if isinstance(num_afc, bool) or int(num_afc) != num_afc or num_afc <= 0:
        raise ValueError("num_afc must be a positive integer")
    center_px = radius_px = None
    if placement_distribution == "circle":
        center_px, radius_px = resolve_stimulus_circle(
            center_point,
            stim_range_radius,
            effective_win_size,
        )
    margin_left, margin_right, margin_top, margin_bottom = normalize_display_margin(display_margin)
    stim_w, stim_h = map(float, stim_size)
    if not all(math.isfinite(v) and v > 0 for v in (stim_w, stim_h)):
        raise ValueError("stim_size dimensions must be finite and positive")
    choice_hitbox_scale = float(choice_hitbox_scale)
    if not math.isfinite(choice_hitbox_scale) or choice_hitbox_scale <= 0:
        raise ValueError("choice_hitbox_scale must be finite and positive")
    separation_size = tuple(max(v, 1.0, v * choice_hitbox_scale) for v in (stim_w, stim_h))
    width, height = map(float, effective_win_size)
    if not all(math.isfinite(v) and v > 0 for v in (width, height)):
        raise ValueError(f"Invalid main screen size for stimulus placement: {effective_win_size}")
    left = -width / 2 + margin_left + stim_w / 2
    right = width / 2 - margin_right - stim_w / 2
    top = height / 2 - margin_top - stim_h / 2
    bottom = -height / 2 + margin_bottom + stim_h / 2
    center_bounds = (left, bottom, right, top)
    if placement_distribution == "uniform":
        if left >= right or bottom >= top:
            raise ValueError(
                "No valid uniform placement area: the full stimulus must fit inside "
                "display_margin. Check image_size and screen size."
            )
        exclusion_center = None
        exclusion_radius = 0.0
        if stim_range_radius is not None:
            exclusion_radius = float(stim_range_radius)
            if not math.isfinite(exclusion_radius) or exclusion_radius < 0:
                raise ValueError("stim_range_radius must be finite and non-negative for uniform placement")
            exclusion_center = tuple(map(float, initiation_cue_position_px or (0, 0)))
            if len(exclusion_center) != 2 or not all(math.isfinite(v) for v in exclusion_center):
                raise ValueError("initiation_cue_position_px must contain two finite coordinates")
            if max((x - exclusion_center[0]) ** 2 + (y - exclusion_center[1]) ** 2
                   for x in (left, right) for y in (bottom, top)) <= exclusion_radius ** 2:
                raise ValueError("No valid uniform placement area outside stim_range_radius of the initiation cue")
        return AFCPlacement(
            center_px=None,
            radius_px=None,
            stim_size=(stim_w, stim_h),
            effective_win_size=(width, height),
            num_afc=num_afc,
            valid_arcs=(),
            fixed_positions_px=None,
            placement_distribution=placement_distribution,
            center_bounds=center_bounds,
            separation_size=separation_size,
            exclusion_center=exclusion_center,
            exclusion_radius=exclusion_radius,
        )
    valid_arcs = _valid_circle_arcs(center_px, radius_px, (left, bottom, right, top))
    if not valid_arcs:
        raise ValueError(
            "No valid stimulus circle arcs: the full stimulus must fit inside "
            "display_margin. Check center_point, stim_range_radius, image_size, and screen size."
        )
    fixed_positions_px = None
    if fixed_positions:
        spacing_angle = (2.0 * math.pi) / float(num_afc)
        # Preserve the established fixed option order/physical layout while
        # defining the positions directly in participant coordinates.
        start_angle = math.radians(fixed_layout_rotation_degrees) - math.pi / 2.0 - spacing_angle / 2.0
        sampled_positions_px = [
            _circle_point(center_px, radius_px, start_angle - (idx * spacing_angle))
            for idx in range(num_afc)
        ]
        if any(not (left <= x <= right and bottom <= y <= top) for x, y in sampled_positions_px):
            raise ValueError("Fixed stimulus circle positions place part of a stimulus outside display_margin")
        assert_non_overlapping_positions(sampled_positions_px, separation_size)
        fixed_positions_px = tuple(sampled_positions_px)
    return AFCPlacement(
        center_px=center_px,
        radius_px=radius_px,
        stim_size=(stim_w, stim_h),
        effective_win_size=(width, height),
        num_afc=num_afc,
        valid_arcs=valid_arcs,
        fixed_positions_px=fixed_positions_px,
        placement_distribution=placement_distribution,
        center_bounds=center_bounds,
        separation_size=separation_size,
    )
