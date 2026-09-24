"""
Static stimulus-set gallery for the Raspberry Pi / PsychoPy task environment.

Each input TSV becomes one static "page" in a single PsychoPy window:
  * every color TSV -> all stimulus colors arranged around a circle
  * every shape TSV -> all shapes arranged around a circle

Controls
--------
  Right arrow / Space / N : next page
  Left arrow / P          : previous page
  1..9                    : jump directly to page 1..9
  Mouse/touch right half  : next page
  Mouse/touch left half   : previous page
  Escape / Q              : quit

Example
-------
python stimulus_set_gallery.py \
    --colors_tsv colors_a.tsv colors_b.tsv \
    --shapes_tsv shapes.tsv \
    --fullscreen \
    --image_size 180 180 \
    --radius 330

The script intentionally reuses the same project utilities as afc_csc1.py for
loading palettes, loading shapes, rasterizing SVGs, making Gaussian color
patches, and creating the PsychoPy window.
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from psychopy import core, event, visual

# Match the project import style used by the AFC task. This assumes this script
# lives one directory below the project root (for example, tasks/ or scripts/).
_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from bin import utils
from bin.config import load_config
from bin.screen import load_screen_config, resolve_scene_size, resolve_task_screens


RGB = Tuple[int, int, int]
Position = Tuple[float, float]


@dataclass
class GalleryPage:
    label: str
    kind: str  # "color" or "shape"
    source: Path
    bg_rgb: RGB
    ids: List[int]
    values: Dict[int, Any]
    high_ids: Optional[List[int]] = None
    low_ids: Optional[List[int]] = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Show complete color/shape stimulus sets as static circular galleries "
            "and toggle between them."
        )
    )
    p.add_argument(
        "--colors_tsv",
        nargs="+",
        default=None,
        metavar="FILE",
        help="One or more color TSVs. Each file becomes one gallery page. JSON config may also use per-file objects with path/background_id/exclude_ids.",
    )
    p.add_argument(
        "--shapes_tsv",
        nargs="+",
        default=None,
        metavar="FILE",
        help="One or more shape TSVs. Each file becomes one gallery page.",
    )
    p.add_argument(
        "--config",
        help="Optional task JSON config; used only for screen selection/window defaults.",
    )
    p.add_argument(
        "--colors_tsv_has_bg_row",
        action="store_true",
        default=None,
        help=(
            "Treat the first row of a plain color TSV as its background and do not "
            "draw that row as a stimulus (default). Per-file background_id in JSON overrides this."
        ),
    )
    p.add_argument(
        "--no_colors_tsv_has_bg_row",
        action="store_false",
        dest="colors_tsv_has_bg_row",
        help="Draw every color row as a stimulus instead of reserving the first as background.",
    )
    p.add_argument(
        "--bg",
        type=int,
        nargs=3,
        metavar=("R", "G", "B"),
        help=(
            "Override the background for every page. If omitted, each color page uses "
            "its own background row; shape pages use the first color-page background "
            "or 128 128 128 if no color page exists."
        ),
    )
    p.add_argument(
        "--shape_color",
        type=int,
        nargs=3,
        default=None,
        metavar=("R", "G", "B"),
        help="RGB used to draw shape outlines (default: 0 0 0).",
    )
    p.add_argument(
        "--image_size",
        type=int,
        nargs=2,
        default=None,
        metavar=("W", "H"),
        help="Stimulus raster/draw size in pixels (default: 180 180).",
    )
    p.add_argument(
        "--radius",
        type=float,
        default=None,
        help="Circle radius in pixels. Default is 34%% of the shorter screen dimension.",
    )
    p.add_argument(
        "--luminance_circle_radius",
        type=float,
        default=None,
        help="Radius in pixels for each high/low luminance circle on color pages.",
    )
    p.add_argument(
        "--luminance_circle_offset_y",
        type=float,
        default=None,
        help="Vertical offset in pixels of the high/low circle centers from screen center.",
    )
    p.add_argument(
        "--start_angle",
        type=float,
        default=None,
        help="Angle in degrees for the first stimulus; 90 places it at the top.",
    )
    p.add_argument(
        "--clockwise",
        action="store_true",
        default=None,
        help="Arrange IDs clockwise instead of counter-clockwise.",
    )
    p.add_argument(
        "--show_labels",
        action="store_true",
        default=None,
        help="Draw stimulus IDs beside the stimuli. Off by default to keep the display clean.",
    )
    p.add_argument(
        "--show_page_title",
        action="store_true",
        default=None,
        help="Draw the current file/page title at the top of the screen.",
    )
    p.add_argument(
        "--show_luminance_labels",
        action="store_true",
        default=None,
        help="Label the top and bottom color circles as High luminance and Low luminance.",
    )
    p.add_argument("--fullscreen", action="store_true", default=None)
    p.add_argument("--win_size", type=int, nargs=2, default=None, metavar=("W", "H"))
    p.add_argument("--main_screen", default=None, help="Main screen index/output name.")
    return p.parse_args()


def _as_rgb(value: Sequence[int]) -> RGB:
    rgb = tuple(int(x) for x in value)
    if len(rgb) != 3:
        raise ValueError(f"Expected 3 RGB values, got {value!r}")
    if any(x < 0 or x > 255 for x in rgb):
        raise ValueError(f"RGB values must be in [0, 255], got {rgb!r}")
    return rgb  # type: ignore[return-value]


def _circle_positions(
    n: int,
    radius: float,
    *,
    start_angle_deg: float = 90.0,
    clockwise: bool = False,
) -> List[Position]:
    if n < 1:
        return []
    direction = -1.0 if clockwise else 1.0
    start = math.radians(float(start_angle_deg))
    step = (2.0 * math.pi) / float(n)
    return [
        (
            float(radius) * math.cos(start + direction * step * i),
            float(radius) * math.sin(start + direction * step * i),
        )
        for i in range(n)
    ]


def _make_image_stim(
    win: visual.Window,
    rendered: Any,
    *,
    size: Tuple[int, int],
    pos: Position,
) -> visual.ImageStim:
    """Display the utility-rendered stimulus directly, without re-normalizing pixels."""
    return visual.ImageStim(
        win,
        image=rendered,
        units="pix",
        size=size,
        pos=pos,
        interpolate=True,
    )

def _load_pages(args: argparse.Namespace) -> List[GalleryPage]:
    pages: List[GalleryPage] = []
    global_bg = _as_rgb(args.bg) if args.bg is not None else None
    first_color_bg: Optional[RGB] = None

    for raw_spec in args.colors_tsv:
        # Backward compatible forms:
        #   "colors.tsv"
        # or, in JSON config:
        #   {"path": "colors.tsv", "background_id": 2, "exclude_ids": [0, 1, 2]}
        #
        # The object form is useful for measurement-style palettes that contain
        # explicit black/white rows before the gray task background.
        if isinstance(raw_spec, dict):
            if "path" not in raw_spec:
                raise ValueError(f"Color-set config is missing 'path': {raw_spec!r}")
            path = Path(raw_spec["path"])
            background_id = raw_spec.get("background_id", None)
            exclude_ids = {int(x) for x in raw_spec.get("exclude_ids", [])}
            page_label = str(raw_spec.get("title", raw_spec.get("label", path.stem)))
            split_luminance = bool(raw_spec.get("split_luminance", True))
            luminance_order = str(raw_spec.get("luminance_order", "high_low")).strip().lower()
            explicit_high_ids = raw_spec.get("high_lum_ids", None)
            explicit_low_ids = raw_spec.get("low_lum_ids", None)
        else:
            path = Path(raw_spec)
            background_id = None
            exclude_ids = set()
            page_label = path.stem
            split_luminance = True
            luminance_order = "high_low"
            explicit_high_ids = None
            explicit_low_ids = None

        if not path.exists():
            raise FileNotFoundError(f"Color TSV not found: {path}")

        palette_all = utils.load_color_palette(path)
        if not palette_all:
            raise ValueError(f"Color TSV contains no colors: {path}")

        if global_bg is not None:
            # Explicit --bg/config bg always wins.
            page_bg = _as_rgb(global_bg)
            colors = dict(palette_all)
            if background_id is not None:
                exclude_ids.add(int(background_id))
        elif background_id is not None:
            bg_id = int(background_id)
            if bg_id not in palette_all:
                raise ValueError(
                    f"background_id={bg_id} is not present in color TSV {path}"
                )
            page_bg = _as_rgb(palette_all[bg_id])
            exclude_ids.add(bg_id)
            colors = dict(palette_all)
        elif args.colors_tsv_has_bg_row:
            bg_from_palette, colors = utils.split_background_from_palette(palette_all)
            page_bg = _as_rgb(bg_from_palette)
        else:
            colors = dict(palette_all)
            page_bg = _as_rgb((128, 128, 128))

        # Remove rows that should define/control the display but should not be
        # shown as circle stimuli (e.g., black, white, and the gray background).
        for excluded_id in exclude_ids:
            colors.pop(int(excluded_id), None)

        if not colors:
            raise ValueError(f"No drawable color stimuli remain after exclusions: {path}")

        if first_color_bg is None:
            first_color_bg = page_bg

        ids = sorted(int(cid) for cid in colors.keys())

        high_ids: Optional[List[int]] = None
        low_ids: Optional[List[int]] = None
        if split_luminance:
            if (explicit_high_ids is None) != (explicit_low_ids is None):
                raise ValueError(
                    f"Color page {path}: provide both high_lum_ids and low_lum_ids, or neither"
                )
            if explicit_high_ids is not None:
                high_ids = [int(x) for x in explicit_high_ids]
                low_ids = [int(x) for x in explicit_low_ids]
                drawable = set(ids)
                unknown = (set(high_ids) | set(low_ids)) - drawable
                if unknown:
                    raise ValueError(
                        f"Color page {path}: luminance ID lists contain excluded/unknown IDs: {sorted(unknown)}"
                    )
                overlap = set(high_ids) & set(low_ids)
                if overlap:
                    raise ValueError(
                        f"Color page {path}: IDs appear in both high and low groups: {sorted(overlap)}"
                    )
                missing = drawable - (set(high_ids) | set(low_ids))
                if missing:
                    raise ValueError(
                        f"Color page {path}: drawable IDs missing from high/low groups: {sorted(missing)}"
                    )
            else:
                if len(ids) % 2 != 0:
                    raise ValueError(
                        f"Color page {path} has {len(ids)} drawable stimuli; automatic high/low splitting "
                        "requires an even count. Add high_lum_ids and low_lum_ids in the config."
                    )
                if luminance_order == "high_low":
                    high_ids = ids[0::2]
                    low_ids = ids[1::2]
                elif luminance_order == "low_high":
                    low_ids = ids[0::2]
                    high_ids = ids[1::2]
                else:
                    raise ValueError(
                        f"Color page {path}: luminance_order must be 'high_low' or 'low_high'"
                    )

        pages.append(
            GalleryPage(
                label=page_label,
                kind="color",
                source=path,
                bg_rgb=page_bg,
                ids=ids,
                values={int(cid): colors[cid] for cid in colors},
                high_ids=high_ids,
                low_ids=low_ids,
            )
        )

    shape_bg = _as_rgb(
        global_bg
        if global_bg is not None
        else (first_color_bg if first_color_bg is not None else (128, 128, 128))
    )
    for raw_spec in args.shapes_tsv:
        if isinstance(raw_spec, dict):
            if "path" not in raw_spec:
                raise ValueError(f"Shape-set config is missing 'path': {raw_spec!r}")
            path = Path(raw_spec["path"])
            page_label = str(raw_spec.get("title", raw_spec.get("label", path.stem)))
        else:
            path = Path(raw_spec)
            page_label = path.stem

        if not path.exists():
            raise FileNotFoundError(f"Shape TSV not found: {path}")

        shapes = utils.load_shape_definitions(path)
        if not shapes:
            raise ValueError(f"Shape TSV contains no shapes: {path}")

        ids = sorted(int(sid) for sid in shapes.keys())
        pages.append(
            GalleryPage(
                label=page_label,
                kind="shape",
                source=path,
                bg_rgb=shape_bg,
                ids=ids,
                values={int(sid): shapes[sid] for sid in shapes},
            )
        )

    if not pages:
        raise ValueError("Provide at least one --colors_tsv or --shapes_tsv file")
    return pages


def _build_page_stims(
    win: visual.Window,
    page: GalleryPage,
    *,
    positions_by_id: Dict[int, Position],
    image_size: Tuple[int, int],
    shape_color: RGB,
    show_labels: bool,
    show_page_title: bool,
    show_luminance_labels: bool,
    scene_size: Tuple[int, int],
    high_center_y: Optional[float] = None,
    low_center_y: Optional[float] = None,
) -> Tuple[Any, List[visual.ImageStim], List[visual.TextStim]]:
    if set(positions_by_id.keys()) != set(page.ids):
        raise ValueError("Position IDs do not match page stimulus IDs")

    bg_rect = utils.make_bg_rect(win, page.bg_rgb)
    image_stims: List[visual.ImageStim] = []
    text_stims: List[visual.TextStim] = []

    for stim_id in page.ids:
        pos = positions_by_id[stim_id]
        if page.kind == "color":
            rendered = utils.make_color_gaussian_image(
                color_rgb_255=page.values[stim_id],
                size_px=image_size,
            )
        elif page.kind == "shape":
            rendered = utils.rasterize_svg_with_color(
                page.values[stim_id],
                size_px=image_size,
                color_rgb_255=shape_color,
                bg_rgb_255=None,
                stroke_rgb_255=shape_color,
                stroke_width_px=2.0,
                outline_only=True,
                flip=True,
            )
        else:
            raise ValueError(f"Unknown page kind: {page.kind}")

        image_stims.append(_make_image_stim(win, rendered, size=image_size, pos=pos))

        if show_labels:
            label_y = float(pos[1]) - (float(image_size[1]) * 0.62)
            text_stims.append(
                visual.TextStim(
                    win,
                    text=str(stim_id),
                    pos=(float(pos[0]), label_y),
                    units="pix",
                    height=22,
                    color=(-1.0, -1.0, -1.0),
                    colorSpace="rgb",
                )
            )

    if show_page_title:
        text_stims.append(
            visual.TextStim(
                win,
                text=page.label,
                pos=(0.0, (float(scene_size[1]) / 2.0) - 42.0),
                units="pix",
                height=30,
                bold=True,
                color=(-1.0, -1.0, -1.0),
                colorSpace="rgb",
            )
        )

    if (
        page.kind == "color"
        and show_luminance_labels
        and page.high_ids is not None
        and page.low_ids is not None
        and high_center_y is not None
        and low_center_y is not None
    ):
        text_stims.append(
            visual.TextStim(
                win,
                text="High luminance",
                pos=(0.0, float(high_center_y)),
                units="pix",
                height=24,
                color=(-1.0, -1.0, -1.0),
                colorSpace="rgb",
            )
        )
        text_stims.append(
            visual.TextStim(
                win,
                text="Low luminance",
                pos=(0.0, float(low_center_y)),
                units="pix",
                height=24,
                color=(-1.0, -1.0, -1.0),
                colorSpace="rgb",
            )
        )

    return bg_rect, image_stims, text_stims


def main() -> None:
    args = parse_args()

    cfg: Dict[str, Any] = {}
    if args.config:
        cfg = load_config(args.config)

    def _get(name: str, default: Any = None) -> Any:
        value = getattr(args, name, None)
        if value is not None:
            return value
        return cfg.get(name, default)

    # Resolve gallery inputs/settings from CLI first, then JSON config.
    colors_cfg = _get("colors_tsv", [])
    shapes_cfg = _get("shapes_tsv", [])
    args.colors_tsv = [colors_cfg] if isinstance(colors_cfg, str) else list(colors_cfg or [])
    args.shapes_tsv = [shapes_cfg] if isinstance(shapes_cfg, str) else list(shapes_cfg or [])
    args.colors_tsv_has_bg_row = bool(_get("colors_tsv_has_bg_row", True))
    args.bg = _get("bg", None)
    args.shape_color = tuple(_get("shape_color", cfg.get("shape_cue_color", [0, 0, 0])))
    args.image_size = tuple(_get("image_size", [180, 180]))
    args.radius = _get("radius", cfg.get("position_spacing", None))
    args.luminance_circle_radius = _get("luminance_circle_radius", None)
    args.luminance_circle_offset_y = _get("luminance_circle_offset_y", None)
    args.start_angle = float(_get("start_angle", 90.0))
    args.clockwise = bool(_get("clockwise", False))
    args.show_labels = bool(_get("show_labels", False))
    args.show_page_title = bool(_get("show_page_title", True))
    args.show_luminance_labels = bool(_get("show_luminance_labels", True))

    pages = _load_pages(args)

    fullscreen = bool(_get("fullscreen", False))
    win_size_raw = _get("win_size", None)
    win_size = tuple(int(x) for x in win_size_raw) if win_size_raw is not None else None

    screen_config = load_screen_config(
        cfg,
        cli_main=_get("main_screen", None),
        cli_experimenter=None,
    )
    main_screen, _ = resolve_task_screens(screen_config, allow_same_screen=True)

    # Initial background only matters before the first page draw.
    win = utils.setup_window(
        bg_rgb_255=pages[0].bg_rgb,
        fullscreen=fullscreen,
        size=win_size,
        screen_info=main_screen,
    )

    mouse = event.Mouse(win=win, visible=True)
    last_mouse_down = False

    try:
        scene_size = resolve_scene_size(
            main_screen,
            fullscreen=fullscreen,
            requested_size=win_size,
            realized_size=tuple(win.size),
        )
        image_size = (int(args.image_size[0]), int(args.image_size[1]))
        radius = (
            float(args.radius)
            if args.radius is not None
            else 0.34 * float(min(scene_size))
        )

        # Pre-render every static page once. Color pages are split into
        # a high-luminance circle above and a low-luminance circle below.
        prepared_pages = []
        for page in pages:
            high_center_y: Optional[float] = None
            low_center_y: Optional[float] = None

            if page.kind == "color" and page.high_ids is not None and page.low_ids is not None:
                lum_radius = (
                    float(args.luminance_circle_radius)
                    if args.luminance_circle_radius is not None
                    else 0.16 * float(min(scene_size))
                )
                lum_offset_y = (
                    float(args.luminance_circle_offset_y)
                    if args.luminance_circle_offset_y is not None
                    else 0.25 * float(scene_size[1])
                )
                high_center_y = lum_offset_y
                low_center_y = -lum_offset_y

                high_positions = _circle_positions(
                    len(page.high_ids),
                    lum_radius,
                    start_angle_deg=float(args.start_angle),
                    clockwise=bool(args.clockwise),
                )
                low_positions = _circle_positions(
                    len(page.low_ids),
                    lum_radius,
                    start_angle_deg=float(args.start_angle),
                    clockwise=bool(args.clockwise),
                )
                positions_by_id: Dict[int, Position] = {}
                for stim_id, (x, y) in zip(page.high_ids, high_positions):
                    positions_by_id[int(stim_id)] = (float(x), float(y) + high_center_y)
                for stim_id, (x, y) in zip(page.low_ids, low_positions):
                    positions_by_id[int(stim_id)] = (float(x), float(y) + low_center_y)
            else:
                positions = _circle_positions(
                    len(page.ids),
                    radius,
                    start_angle_deg=float(args.start_angle),
                    clockwise=bool(args.clockwise),
                )
                positions_by_id = {
                    int(stim_id): pos for stim_id, pos in zip(page.ids, positions)
                }

            prepared_pages.append(
                _build_page_stims(
                    win,
                    page,
                    positions_by_id=positions_by_id,
                    image_size=image_size,
                    shape_color=_as_rgb(args.shape_color),
                    show_labels=bool(args.show_labels),
                    show_page_title=bool(args.show_page_title),
                    show_luminance_labels=bool(args.show_luminance_labels),
                    scene_size=scene_size,
                    high_center_y=high_center_y,
                    low_center_y=low_center_y,
                )
            )

        page_idx = 0
        needs_redraw = True
        print(f"Loaded {len(pages)} gallery pages.")
        print("Controls: left/right or P/N/space; 1-9 jump; click/touch halves; Q/Escape quit.")

        while True:
            if needs_redraw:
                bg_rect, image_stims, text_stims = prepared_pages[page_idx]
                bg_rect.draw()
                for stim in image_stims:
                    stim.draw()
                for stim in text_stims:
                    stim.draw()
                win.flip()
                page = pages[page_idx]
                print(
                    f"Page {page_idx + 1}/{len(pages)}: {page.label} "
                    f"[{page.kind}, n={len(page.ids)}]"
                )
                needs_redraw = False

            keys = event.getKeys()
            if any(k in ("escape", "q") for k in keys):
                break

            if any(k in ("right", "space", "n") for k in keys):
                page_idx = (page_idx + 1) % len(pages)
                needs_redraw = True
            elif any(k in ("left", "p") for k in keys):
                page_idx = (page_idx - 1) % len(pages)
                needs_redraw = True
            else:
                for k in keys:
                    if k.isdigit():
                        requested = int(k) - 1
                        if 0 <= requested < len(pages):
                            page_idx = requested
                            needs_redraw = True
                            break

            # Touchscreen/mouse toggle: trigger only on the transition from up -> down.
            buttons = mouse.getPressed()
            mouse_down = bool(buttons[0])
            if mouse_down and not last_mouse_down:
                x, _ = mouse.getPos()
                if x >= 0:
                    page_idx = (page_idx + 1) % len(pages)
                else:
                    page_idx = (page_idx - 1) % len(pages)
                needs_redraw = True
            last_mouse_down = mouse_down

            core.wait(0.01)

    finally:
        win.close()


if __name__ == "__main__":
    main()
