"""
Utility helpers for PsychoPy tasks (simplified).
Robust transparency on all platforms by using RGB + alpha MASK for ImageStim.
Also supports loading SVG by rasterizing to a requested pixel size (via cairosvg).

Modularity helpers included:
- make_bg_rect: create a full-window background rect in one call.
- make_onset_cue_stim: build a checkerboard ImageStim with a centered 2D Gaussian alpha mask.
"""
from pathlib import Path
import random
from typing import List, Tuple, Optional, Dict, Union
import io

import numpy as np
from PIL import Image
from psychopy import visual, event
import time

# Global debug flag: when True, utilities may write debug files (PNG) to logs/
# Default is False; tasks can enable it via CLI (--debug) or config.
DEBUG = False


def set_debug(value: bool):
    global DEBUG
    DEBUG = bool(value)

# File types
RASTER_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
VECTOR_EXTS = {".svg"}
IMAGE_EXTS = RASTER_EXTS | VECTOR_EXTS  # for discovery


def find_image_files(images_dir: str, recursive: bool = False) -> List[Path]:
    p = Path(images_dir)
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if recursive:
        files = [f for f in p.rglob("*") if f.suffix.lower() in IMAGE_EXTS]
    else:
        files = [f for f in p.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS]
    return sorted(files)


def sample_images(files: List[Path], n: int, seed: Optional[int] = None) -> List[Path]:
    if seed is not None:
        random.seed(seed)
    if not files:
        return []
    return random.sample(files, n) if n <= len(files) else [random.choice(files) for _ in range(n)]


def rgb255_to_psychopy(rgb_255: Tuple[int, int, int]) -> List[float]:
    arr = np.clip(np.array(rgb_255, dtype=float), 0, 255)
    return ((arr / 127.5) - 1).tolist()  # 0->-1, 127.5->0, 255->1


def setup_window(
    bg_rgb_255: Tuple[int, int, int] = (128, 128, 128),
    fullscreen: bool = False,
    size: Optional[Tuple[int, int]] = None,
    monitor: Optional[str] = None,
):
    color = rgb255_to_psychopy(bg_rgb_255)
    win_kwargs = dict(color=color, colorSpace="rgb", units="pix", allowStencil=False)
    if monitor:
        win_kwargs["monitor"] = monitor
    if fullscreen:
        return visual.Window(fullscr=True, **win_kwargs)
    if size is None:
        size = (1024, 768)
    return visual.Window(size=size, fullscr=False, **win_kwargs)


def make_fixation_cross(win: visual.Window, size: int = 40, color: Tuple[int, int, int] = (0, 0, 0)):
    # If size is zero or negative, return None to indicate no fixation should be shown.
    if size is None or size <= 0:
        return None
    return visual.TextStim(win, text="+", height=size, color=rgb255_to_psychopy(color), colorSpace="rgb")


def make_bg_rect(win: visual.Window, bg_rgb_255: Tuple[int, int, int]):
    """Create a full-window background rectangle in pixel units.

    This avoids duplicating rectangle construction logic across tasks.
    """
    return visual.Rect(
        win,
        width=win.size[0],
        height=win.size[1],
        fillColor=rgb255_to_psychopy(bg_rgb_255),
        fillColorSpace="rgb",
        lineColor=None,
        units="pix",
    )


def _to_pil_rgba(obj: Union[Image.Image, Path, np.ndarray]) -> Optional[Image.Image]:
    """Return a PIL RGBA image or None if conversion fails (raster inputs only)."""
    if isinstance(obj, Image.Image):
        return obj.convert("RGBA")
    if isinstance(obj, Path):
        if obj.suffix.lower() in VECTOR_EXTS:
            # handled elsewhere; this helper is raster-only
            return None
        try:
            with Image.open(obj) as im:
                return im.convert("RGBA").copy()
        except Exception:
            return None
    # assume array-like
    try:
        arr = np.asarray(obj)
    except Exception:
        return None
    if arr.dtype.kind == "f":
        arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    if arr.ndim == 2:
        return Image.fromarray(arr, mode="L").convert("RGBA")
    if arr.ndim == 3 and arr.shape[2] in (3, 4):
        mode = "RGBA" if arr.shape[2] == 4 else "RGB"
        return Image.fromarray(arr.astype(np.uint8), mode=mode).convert("RGBA")
    try:
        return Image.fromarray(arr).convert("RGBA")
    except Exception:
        return None


def _rasterize_svg_to_rgba(
    svg_path: Path, size_px: Tuple[int, int], bg_rgb_255: Optional[Tuple[int, int, int]] = None
) -> Image.Image:
    """
    Rasterize an SVG file to a PIL RGBA image using cairosvg.
    size_px: (width, height) in pixels.
    """
    try:
        import cairosvg  # type: ignore
    except Exception as e:
        raise ImportError(
            "SVG support requires 'cairosvg'. Install with: pip install cairosvg"
        ) from e

    if not size_px or len(size_px) != 2 or size_px[0] <= 0 or size_px[1] <= 0:
        raise ValueError("svg_size must be a (width, height) tuple of positive ints")

    # Read file bytes and ask cairosvg to render with a transparent background.
    # Using an explicit transparent color avoids backend defaults that may fill
    # the canvas with an opaque color on some systems.
    svg_bytes = svg_path.read_bytes()
    # If a background color is provided, request cairosvg to rasterize with
    # that opaque background. Otherwise request a transparent background.
    if bg_rgb_255 is not None:
        r, g, b = (int(c) for c in bg_rgb_255)
        bg_token = f"rgb({r},{g},{b})"
    else:
        bg_token = "rgba(0,0,0,0)"

    png_bytes = cairosvg.svg2png(
        bytestring=svg_bytes,
        output_width=int(size_px[0]),
        output_height=int(size_px[1]),
        background_color=bg_token,
    )

    im = Image.open(io.BytesIO(png_bytes)).convert("RGBA")

    # Save a debug copy so it's easy to verify that the rasterized SVG has
    # the expected background. Don't fail if logs/ can't be written; this
    # is purely diagnostic.
    # Save debug raster only when debugging is enabled
    try:
        if DEBUG:
            logs_dir = Path("logs")
            logs_dir.mkdir(parents=True, exist_ok=True)
            debug_path = logs_dir / f"debug_rasterized_svg_{svg_path.stem}.png"
            im.save(debug_path)
    except Exception:
        # ignore failures to write debug file
        pass

    return im


def rasterize_svg_with_color(
    svg_path: Path,
    size_px: Tuple[int, int],
    color_rgb_255: Tuple[int, int, int],
    bg_rgb_255: Optional[Tuple[int, int, int]] = None,
    stroke_rgb_255: Optional[Tuple[int, int, int]] = None,
    stroke_width_px: Optional[float] = None,
) -> Image.Image:
    """Rasterize an SVG and force its fill color to `color_rgb_255`.

    This reads the SVG text, injects a CSS rule to set fill color on common
    shape elements, preserves or sets stroke properties, then rasterizes via
    cairosvg to a PIL RGBA image.

    Parameters:
    - stroke_rgb_255: optional RGB tuple to set the stroke color. If None,
      defaults to black (0,0,0) per project convention.
    - stroke_width_px: optional stroke width in pixels. If None the SVG's
      original stroke-width is left unchanged.
    """
    try:
        import cairosvg  # type: ignore
    except Exception as e:
        raise ImportError(
            "SVG support requires 'cairosvg'. Install with: pip install cairosvg"
        ) from e

    if not size_px or len(size_px) != 2 or size_px[0] <= 0 or size_px[1] <= 0:
        raise ValueError("size_px must be a (width, height) tuple of positive ints")

    svg_text = svg_path.read_text(encoding="utf-8")

    r, g, b = (int(c) for c in color_rgb_255)

    # Determine stroke color (default to black if user didn't provide one).
    if stroke_rgb_255 is None:
        sr, sg, sb = (0, 0, 0)
    else:
        sr, sg, sb = (int(c) for c in stroke_rgb_255)

    # Build CSS rules. We always set fill to the requested color. For stroke
    # we set the requested color; if stroke_width_px is provided we also set
    # stroke-width. If stroke_width_px is None, the SVG's original stroke
    # width is preserved.
    style_rules = [f"fill:rgb({r},{g},{b}) !important", f"stroke:rgb({sr},{sg},{sb}) !important"]
    if stroke_width_px is not None:
        # ensure numeric formatting
        try:
            sw = float(stroke_width_px)
            style_rules.append(f"stroke-width:{sw}px !important")
        except Exception:
            # ignore invalid stroke width and leave it unspecified
            pass

    style_block = f"<style>path,rect,circle,polygon,ellipse,g,polyline{{{';'.join(style_rules)}}}</style>"

    # Find the end of the <svg ...> start tag to inject style immediately after it.
    idx = svg_text.find("<svg")
    if idx == -1:
        # fallback: just prepend the style
        mod_svg = style_block + svg_text
    else:
        # find the next '>' after the <svg
        gt = svg_text.find('>', idx)
        if gt == -1:
            mod_svg = style_block + svg_text
        else:
            mod_svg = svg_text[: gt + 1] + style_block + svg_text[gt + 1 :]

    # If a background color is provided, set the background token; else
    # request transparency.
    if bg_rgb_255 is not None:
        bg_token = f"rgb({int(bg_rgb_255[0])},{int(bg_rgb_255[1])},{int(bg_rgb_255[2])})"
    else:
        bg_token = "rgba(0,0,0,0)"

    png_bytes = cairosvg.svg2png(
        bytestring=mod_svg.encode("utf-8"),
        output_width=int(size_px[0]),
        output_height=int(size_px[1]),
        background_color=bg_token,
    )

    im = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    try:
        if DEBUG:
            logs_dir = Path("logs")
            logs_dir.mkdir(parents=True, exist_ok=True)
            debug_path = logs_dir / f"debug_rasterized_svg_colored_{svg_path.stem}_{r}_{g}_{b}.png"
            im.save(debug_path)
    except Exception:
        pass
    return im


def load_color_palette(tsv_path: Path) -> Dict[int, Tuple[int, int, int]]:
    """Load a TSV with columns ID,R,G,B (header optional) and return mapping id->(r,g,b).

    ID is converted to int. Rows with invalid values raise ValueError.
    """
    import csv

    p = Path(tsv_path)
    if not p.exists():
        raise FileNotFoundError(f"Color TSV not found: {tsv_path}")

    out: Dict[int, Tuple[int, int, int]] = {}
    with p.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        # Accept files that may not have a header by falling back to positional columns
        if reader.fieldnames is None:
            raise ValueError("Color TSV must have header with columns ID,R,G,B")
        names = [n.lower() for n in reader.fieldnames]
        for row in reader:
            try:
                idv = int(row.get("id") or row.get("ID") or row.get("Id") or row.get(reader.fieldnames[0]))
                r = int(row.get("r") or row.get("R") or row.get(reader.fieldnames[1]))
                g = int(row.get("g") or row.get("G") or row.get(reader.fieldnames[2]))
                b = int(row.get("b") or row.get("B") or row.get(reader.fieldnames[3]))
            except Exception as e:
                raise ValueError(f"Invalid row in color TSV: {row}") from e
            out[idv] = (r, g, b)
    return out


def load_shape_definitions(tsv_path: Path) -> Dict[int, Path]:
    """Load a TSV with columns ID,PATH where PATH points to an SVG file.

    Returns mapping id->Path and verifies files exist and are SVGs.
    """
    import csv

    p = Path(tsv_path)
    if not p.exists():
        raise FileNotFoundError(f"Shape TSV not found: {tsv_path}")

    out: Dict[int, Path] = {}
    with p.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError("Shape TSV must have header with columns ID,PATH")
        for row in reader:
            try:
                idv = int(row.get("id") or row.get("ID") or row.get(reader.fieldnames[0]))
                path_str = row.get("path") or row.get("PATH") or row.get(reader.fieldnames[1])
                if path_str is None:
                    raise ValueError("Missing path column")
                sp = Path(path_str)
                if not sp.exists():
                    raise FileNotFoundError(f"Shape file does not exist: {sp}")
                if sp.suffix.lower() != ".svg":
                    raise ValueError(f"Shape file must be SVG: {sp}")
            except Exception as e:
                raise ValueError(f"Invalid row in shape TSV: {row}") from e
            out[idv] = sp
    return out


def load_image_assets(
    files: List[Path],
    raster_size: Optional[Tuple[int, int]] = None,
    bg_rgb_255: Optional[Tuple[int, int, int]] = None,
) -> Dict[Path, Image.Image]:
    """
    Preload a mixed list of raster and SVG images into PIL Images.
    - Raster images are converted to RGBA and optionally resized to raster_size.
    - SVG images are rasterized to svg_size (required if any SVGs are present).
    Returns: dict Path -> PIL.Image.Image (RGBA).
    """
    images: Dict[Path, Image.Image] = {}
    has_svg = any(f.suffix.lower() in VECTOR_EXTS for f in files)
    if has_svg and not raster_size:
        raise ValueError("SVG files detected but no image_size provided. Pass --image_size W H or set in config.")

    for p in files:
        ext = p.suffix.lower()
        if ext in VECTOR_EXTS:
            # Use raster_size as the target rasterization size for SVGs.
            im = _rasterize_svg_to_rgba(p, raster_size, bg_rgb_255=bg_rgb_255)  # type: ignore[arg-type]
            images[p] = im
        else:
            with Image.open(p) as im:
                im = im.convert("RGBA")
                if raster_size is not None:
                    im = im.resize((int(raster_size[0]), int(raster_size[1])), Image.LANCZOS)
                images[p] = im.copy()
    return images


def make_image_stim_from_array(
    win: visual.Window,
    img_obj,
    size: Optional[Tuple[int, int]] = None,
    bg_rgb_255: Optional[Tuple[int, int, int]] = None,
):
    """
    Create an ImageStim from PIL/Path/ndarray.

        - If bg_rgb_255 is given, we pre-composite RGBA onto that solid color (no transparency).
        - Otherwise we pass RGB + a 2D mask in the range [-1, 1] (PsychoPy convention),
            which is robust on macOS and elsewhere.
    """
    pil = _to_pil_rgba(img_obj)
    if pil is None and isinstance(img_obj, Path) and img_obj.suffix.lower() in VECTOR_EXTS:
        # If someone accidentally passes an SVG Path directly here, rasterize on the fly
        raise ValueError("SVG Paths must be pre-rasterized via load_image_assets (provide svg_size).")

    if pil is None:
        # last resort: let PsychoPy try whatever it is
        return visual.ImageStim(win, image=img_obj, size=size, units="pix")

    # If a background color was requested earlier during rasterization,
    # the image may already be fully opaque. In that case, pass the RGB
    # image directly to PsychoPy (no mask) and avoid interpolation which
    # can introduce edge artifacts. Also prefer the image's native pixel
    # size when no explicit `size` is provided to avoid resampling.
    if size is None:
        size = pil.size

    # If a background was requested we probably flattened the SVG; check
    # whether the alpha channel is fully opaque and, if so, send RGB only.
    a = pil.getchannel("A")
    try:
        extrema = a.getextrema()
    except Exception:
        extrema = (255, 255)

    rgb = pil.convert("RGB")
    if extrema == (255, 255):
        # fully opaque: use RGB image (no mask) and no interpolation to
        # prevent border smoothing artifacts
        return visual.ImageStim(win, image=rgb, size=size, units="pix", interpolate=False)

    # Otherwise preserve transparency via RGB + mask. PsychoPy expects masks in
    # the range [-1, 1] where -1 is fully transparent and +1 is fully opaque.
    # Convert the 8-bit alpha channel to that range to avoid unintended 50% opacity.
    mask01 = np.asarray(a, dtype=np.float32) / 255.0  # H x W, 0..1
    mask_pm1 = (mask01 * 2.0) - 1.0                  # H x W, -1..1
    return visual.ImageStim(win, image=rgb, mask=mask_pm1, size=size, units="pix", interpolate=False)


def clear_events():
    event.clearEvents()


def make_onset_cue_stim(
    win: visual.Window,
    bg_rgb_255: Tuple[int, int, int],
    size_frac: float = 0.125,
    cells: int = 8,
    sigma_frac: float = 0.22,
    zero_threshold: int = 1,
):
    """Create a checkerboard onset cue ImageStim with a centered 2D Gaussian alpha mask.

    Parameters:
    - size_frac: fraction of min(window size) for cue edge length
    - cells: number of checkerboard cells per side
    - sigma_frac: sigma expressed as a fraction of cue width
    - zero_threshold: values <= this threshold in [0..255] are set to 0 in the mask
    """
    from PIL import Image, ImageDraw

    w = int(max(4, min(win.size) * float(size_frac)))
    if w <= 0:
        w = 400

    # Build checkerboard RGB on top of background color
    cb = Image.new("RGB", (w, w), color=(int(bg_rgb_255[0]), int(bg_rgb_255[1]), int(bg_rgb_255[2])))
    draw = ImageDraw.Draw(cb)
    cell = max(2, w // int(cells))
    for y in range(0, w, cell):
        for x in range(0, w, cell):
            xi = x // cell
            yi = y // cell
            fill = (0, 0, 0) if ((xi + yi) % 2 == 0) else (255, 255, 255)
            draw.rectangle([x, y, x + cell - 1, y + cell - 1], fill=fill)

    # 2D Gaussian mask centered at cue
    cx = (w - 1) / 2.0
    cy = (w - 1) / 2.0
    sigma = max(2.0, w * float(sigma_frac))
    yy, xx = np.mgrid[0:w, 0:w]
    gauss = np.exp(-0.5 * (((xx - cx) / sigma) ** 2 + ((yy - cy) / sigma) ** 2))
    mask_arr = np.clip(gauss * 255.0, 0, 255)
    mask_u8 = mask_arr.astype(np.uint8)
    if zero_threshold is not None and zero_threshold > 0:
        mask_u8[mask_u8 <= int(zero_threshold)] = 0
    cb.putalpha(Image.fromarray(mask_u8, mode="L"))

    # Convert to ImageStim (preserve alpha; ImageStim builder will provide proper mask)
    stim = make_image_stim_from_array(win, cb, size=(w, w), bg_rgb_255=None)
    try:
        stim.pos = (0, 0)
    except Exception:
        pass
    return stim


def present_block_with_persistent_dots(
    win: visual.Window,
    preloaded: Dict[Union[Path, Tuple[int, int]], Image.Image],
    block_paths: List[Union[Path, Tuple[int, int]]],
    positions: List[Tuple[float, float]],
    duration: float,
    choice_time: float,
    dot_size: int,
    dot_color: Tuple[int, int, int],
    bg_rect,
    fix,
    logger,
    block_idx: int,
    isi: float = 0.0,
    init_dot_color: Optional[Tuple[int, int, int]] = None,
    bg_rgb_255: Optional[Tuple[int, int, int]] = None,
    onset_cue: Optional[visual.ImageStim] = None,
):
    """Present stimuli one at a time, leave faint dots at their locations,
    show all dots for `choice_time`, then clear.

    Returns a tuple (aborted: bool, choice_info: Optional[dict]).
    - aborted: True if the user pressed escape and the task should quit immediately.
    - choice_info: None if no choice was made during the choice period, or a dict with keys:
        - chosen_index: int (1-based index within the block)
        - chosen_pos: (x,y) psychopy pixel coords
        - choice_time_perf_s: perf_counter timestamp when the choice was made
        - choice_time_psychopy_s: None (psychopy flip timestamp not available for mouse click)
        - notes: optional string
    """
    from psychopy import core as _core

    dots: List[visual.Circle] = []
    _visual = visual
    stim_sizes: List[Tuple[float, float]] = []

    # If an onset cue is provided, show it and wait for the participant to
    # self-initiate by clicking/tapping the cue. The function will then fade
    # the cue out and proceed to present stimuli. If the onset cue click
    # triggers an abort (escape), return (True, None).
    from psychopy import event as _event
    mouse = _event.Mouse(win=win)

    if onset_cue is not None:
        # center onset cue by default
        try:
            onset_cue.pos = (0, 0)
            # reset opacity in case previous block hid it
            onset_cue.opacity = 1.0
        except Exception:
            pass
        # show onset cue and wait for click inside its bounding box
        bg_rect.draw()
        onset_cue.draw()
        if fix is not None:
            fix.draw()
        oc_flip = win.flip()
        oc_perf = time.perf_counter()
        logger.log(
            "onset_cue_shown",
            image_name="onset_cue",
            requested_duration_s=None,
            flip_time_psychopy_s=oc_flip,
            flip_time_perf_s=oc_perf,
            end_time_perf_s=None,
            notes=f"block={block_idx}",
        )

        # wait for click within onset cue
        onset_start = time.perf_counter()
        while True:
            if _event.getKeys(["escape"]):
                logger.log("abort", image_name="", notes="escape_pressed")
                win.close()
                _core.quit()
                return True, None

            buttons = mouse.getPressed()
            if any(buttons):
                click_pos = mouse.getPos()
                # determine bounding box from onset_cue.size and pos
                try:
                    oc_w, oc_h = onset_cue.size
                except Exception:
                    oc_w, oc_h = (200, 200)
                oc_x, oc_y = getattr(onset_cue, "pos", (0, 0))
                if abs(click_pos[0] - oc_x) <= oc_w / 2.0 and abs(click_pos[1] - oc_y) <= oc_h / 2.0:
                    click_perf = time.perf_counter()
                    logger.log(
                        "onset_cue_clicked",
                        image_name="onset_cue",
                        requested_duration_s=None,
                        flip_time_psychopy_s=None,
                        flip_time_perf_s=click_perf,
                        end_time_perf_s=None,
                        notes=f"block={block_idx} click_xy=({click_pos[0]:.1f},{click_pos[1]:.1f})",
                    )
                    # Immediately dismiss onset cue (no fade)
                    try:
                        onset_cue.opacity = 0.0
                    except Exception:
                        pass
                    break

            _core.wait(0.01)


    for idx, (p, pos) in enumerate(zip(block_paths, positions), start=1):
        # support two kinds of block items:
        # - Path objects pointing to raster images (used by afc_block_sequence)
        # - (shape_id, color_id) tuples used by active_foraging
        if isinstance(p, tuple) and len(p) == 2:
            sid, cid = p
            name = f"shape{sid}_color{cid}"
            pil_img = preloaded.get((sid, cid))
            if pil_img is None:
                # fall back to string key if user provided stringified keys
                pil_img = preloaded.get(p)
        else:
            name = getattr(p, "name", str(p))
            pil_img = preloaded[p]
        stim = make_image_stim_from_array(win, pil_img, size=None, bg_rgb_255=bg_rgb_255)
        stim.pos = pos
        # record stimulus pixel size for click hit-testing later
        try:
            stim_sizes.append(tuple(stim.size))
        except Exception:
            stim_sizes.append((0.0, 0.0))
        # Optionally show the dot for `isi` seconds before the stimulus
        # appears. We create the dot first (so it persists) and draw it along
        # with existing dots, then wait `isi` seconds before showing the stim.
        if isi and isi > 0:
            # use init_dot_color if provided, otherwise fall back to dot_color
            cue_color = init_dot_color if init_dot_color is not None else dot_color
            dot = _visual.Circle(
                win,
                radius=dot_size / 2.0,
                fillColor=rgb255_to_psychopy(cue_color),
                fillColorSpace="rgb",
                lineColor=None,
                units="pix",
            )
            dot.pos = pos
            dots.append(dot)

            # draw background + all dots (including this new one) + fixation
            bg_rect.draw()
            for d in dots:
                d.draw()
            if fix is not None:
                fix.draw()
            dot_flip = win.flip()
            dot_perf = time.perf_counter()
            logger.log(
                "dot_on",
                image_name=name,
                requested_duration_s=isi,
                flip_time_psychopy_s=dot_flip,
                flip_time_perf_s=dot_perf,
                end_time_perf_s=dot_perf + isi,
                notes=f"block={block_idx} idx={idx}",
            )
            _core.wait(isi)

        # Draw background, existing dots, then current stim and fixation
        bg_rect.draw()
        for d in dots:
            d.draw()
        stim.draw()
        if fix is not None:
            fix.draw()
        flip_ps = win.flip()
        flip_perf = time.perf_counter()
        logger.log(
            "stim_on",
            image_name=name,
            requested_duration_s=duration,
            flip_time_psychopy_s=flip_ps,
            flip_time_perf_s=flip_perf,
            end_time_perf_s=None,
            notes=f"block={block_idx} idx={idx}",
        )

        _core.wait(duration)

        # After stimulus, update the most recent dot's color to the persistent dot_color
        if dots:
            last_dot = dots[-1]
            # set persistent color
            last_dot.fillColor = rgb255_to_psychopy(dot_color)
            last_dot.fillColorSpace = "rgb"

        # draw background + all dots + fixation (so the dot remains visible)
        bg_rect.draw()
        for d in dots:
            d.draw()
        if fix is not None:
            fix.draw()
        off_perf = time.perf_counter()
        win.flip()
        logger.log(
            "stim_off",
            image_name=name,
            requested_duration_s=duration,
            flip_time_psychopy_s=None,
            flip_time_perf_s=None,
            end_time_perf_s=off_perf,
            notes=f"block={block_idx} idx={idx}",
        )

        # small safety: check for abort
        if event.getKeys(["escape"]):
            logger.log("abort", image_name="", notes="escape_pressed")
            win.close()
            _core.quit()
            return True, None

    # After all stimuli in this block were shown, enter the choice period.
    # We'll draw the dots and wait up to `choice_time` seconds, but poll for
    # mouse clicks so a choice can end the choice period immediately.
    from psychopy import event as _event

    mouse = _event.Mouse(win=win)

    # choice loop
    start = time.perf_counter()
    choice_made = False
    chosen_info = None

    # log choice start (no psychopy flip timestamp available for a mouse click; we record perf times)
    bg_rect.draw()
    for d in dots:
        d.draw()
    if fix is not None:
        fix.draw()
    choice_flip = win.flip()
    choice_perf = time.perf_counter()
    logger.log(
        "choice_start",
        image_name="",
        requested_duration_s=choice_time,
        flip_time_psychopy_s=choice_flip,
        flip_time_perf_s=choice_perf,
        end_time_perf_s=choice_perf + choice_time,
        notes=f"block={block_idx}",
    )

    # Note: click hitboxes are based on stimulus size (not the dot size).
    # We'll compute per-stim hit radius later when a click candidate is found.

    # convert draw positions to a list for easy indexing
    pos_list = list(positions)

    while True:
        now = time.perf_counter()
        elapsed = now - start

        # draw background, dots and fixation each frame so the display stays active
        bg_rect.draw()
        for d in dots:
            d.draw()
        if fix is not None:
            fix.draw()
        win.flip()

        # check for escape abort
        if _event.getKeys(["escape"]):
            logger.log("abort", image_name="", notes="escape_pressed")
            win.close()
            _core.quit()
            return True, None

        # check for mouse clicks (any button)
        buttons = mouse.getPressed()
        if any(buttons):
            # get click position in window coordinates
            click_pos = mouse.getPos()
            # find nearest stimulus position
            chosen_idx = None
            best_dist = None
            for i, ppos in enumerate(pos_list, start=1):
                dx = click_pos[0] - ppos[0]
                dy = click_pos[1] - ppos[1]
                dist = (dx * dx + dy * dy) ** 0.5
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    chosen_idx = i

            if best_dist is not None:
                # determine hit radius based on the chosen stimulus native size
                try:
                    w, h = stim_sizes[chosen_idx - 1]
                    candidate_radius = max(24.0, min(w, h) / 2.0)
                except Exception:
                    candidate_radius = 64.0
                if best_dist <= candidate_radius:
                    # register the choice (1-based index)
                    choice_time_perf = time.perf_counter()
                    chosen_info = {
                        "chosen_index": int(chosen_idx),
                        "chosen_pos": tuple(pos_list[chosen_idx - 1]),
                        "choice_time_perf_s": float(choice_time_perf),
                        "choice_time_psychopy_s": None,
                        "notes": f"block={block_idx}",
                    }
                    logger.log(
                        "choice_made",
                        image_name=getattr(block_paths[chosen_idx - 1], "name", str(block_paths[chosen_idx - 1])),
                        requested_duration_s=None,
                        flip_time_psychopy_s=None,
                        flip_time_perf_s=choice_time_perf,
                        end_time_perf_s=None,
                        notes=f"block={block_idx} idx={chosen_idx} dist={best_dist:.1f} click_xy=({click_pos[0]:.1f},{click_pos[1]:.1f})",
                    )
                    choice_made = True
                    break

        # no click: check timeout
        if elapsed >= choice_time:
            # choice period ended without a valid selection
            logger.log("choice_end", image_name="", notes=f"block={block_idx} no_choice")
            break

        # small sleep to avoid busy loop
        _core.wait(0.01)

    # clear dots
    bg_rect.draw()
    if fix is not None:
        fix.draw()
    win.flip()

    return False, chosen_info


def sample_non_overlapping_positions(
    count: int,
    stim_size: Tuple[int, int],
    win_size: Tuple[int, int],
    max_attempts: int = 2000,
    margin: int = 50,
) -> List[Tuple[float, float]]:
    """Public helper: wrapper around the non-overlap placement algorithm.

    Returns a list of (x, y) positions in PsychoPy pixel coords (centered at 0,0).
    """
    w_win, h_win = win_size
    w_stim, h_stim = stim_size
    half_w = w_win / 2.0
    half_h = h_win / 2.0

    # Enforce margin from window edges
    min_x = -half_w + w_stim / 2.0 + margin
    max_x = half_w - w_stim / 2.0 - margin
    min_y = -half_h + h_stim / 2.0 + margin
    max_y = half_h - h_stim / 2.0 - margin

    if min_x > max_x or min_y > max_y:
        raise ValueError("Stimulus size is larger than window; cannot place stimuli")

    rects: List[Tuple[float, float, float, float]] = []
    positions: List[Tuple[float, float]] = []
    attempts = 0
    while len(positions) < count and attempts < max_attempts:
        attempts += 1
        x = random.uniform(min_x, max_x)
        y = random.uniform(min_y, max_y)
        ok = True
        for (cx, cy, ww, hh) in rects:
            if abs(x - cx) < (w_stim + ww) / 2.0 and abs(y - cy) < (h_stim + hh) / 2.0:
                ok = False
                break
        if ok:
            rects.append((x, y, w_stim, h_stim))
            positions.append((x, y))

    if len(positions) < count:
        raise RuntimeError(f"Could not place {count} non-overlapping stimuli after {max_attempts} attempts")
    return positions


def clamp_positions(
    positions: List[Tuple[float, float]],
    stim_size: Tuple[int, int],
    win_size: Tuple[int, int],
    margin: int = 50,
) -> List[Tuple[float, float]]:
    """Clamp a list of center positions so stimuli remain within window margins.

    Returns a new list of (x,y) positions.
    """
    half_w = win_size[0] / 2.0
    half_h = win_size[1] / 2.0
    w_stim, h_stim = stim_size
    min_x = -half_w + w_stim / 2.0 + margin
    max_x = half_w - w_stim / 2.0 - margin
    min_y = -half_h + h_stim / 2.0 + margin
    max_y = half_h - h_stim / 2.0 - margin

    out: List[Tuple[float, float]] = []
    for (x, y) in positions:
        cx = min(max(x, min_x), max_x)
        cy = min(max(y, min_y), max_y)
        out.append((cx, cy))
    return out


def sample_blocks(files: List[Path], num_afc: int, n_blocks: int, seed: Optional[int] = None) -> List[List[Path]]:
    """Sample stimuli for each block.

    For each block, sample `num_afc` unique stimuli without replacement within that block.
    Blocks are independent (the same image may appear in different blocks).
    """
    if seed is not None:
        random.seed(seed)
    if num_afc < 1:
        raise ValueError("num_afc must be >= 1")
    if num_afc > len(files):
        raise ValueError("num_afc cannot be larger than the number of available images")

    blocks: List[List[Path]] = []
    for _ in range(n_blocks):
        blocks.append(random.sample(files, num_afc))
    return blocks