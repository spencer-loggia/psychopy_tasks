"""
Utility helpers for PsychoPy tasks (simplified).
Robust transparency on all platforms by using RGB + alpha MASK for ImageStim.
Also supports loading SVG by rasterizing to a requested pixel size (via cairosvg).

Modularity helpers included:
- make_bg_rect: create a full-window background rect in one call.
- make_onset_cue_stim: build a checkerboard ImageStim with a centered 2D Gaussian alpha mask.
"""
from pathlib import Path
import datetime as dt
import random
from typing import List, Tuple, Optional, Dict, Union, Callable, Any
import io
import sys
import multiprocessing as mp
import queue
import traceback

import numpy as np
from PIL import Image
import threading
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
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".mpeg", ".mpg", ".m4v", ".wmv"}


def find_image_files(images_dir: str, recursive: bool = False) -> List[Path]:
    p = Path(images_dir)
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if recursive:
        files = [f for f in p.rglob("*") if f.suffix.lower() in IMAGE_EXTS]
    else:
        files = [f for f in p.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS]
    return sorted(files)


def find_video_files(videos_dir: str, recursive: bool = False) -> List[Path]:
    p = Path(videos_dir)
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"Video directory not found: {videos_dir}")
    if recursive:
        files = [f for f in p.rglob("*") if f.suffix.lower() in VIDEO_EXTS]
    else:
        files = [f for f in p.iterdir() if f.is_file() and f.suffix.lower() in VIDEO_EXTS]
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


def compute_cover_size(
    content_size: Tuple[float, float],
    container_size: Tuple[float, float],
) -> Tuple[float, float]:
    content_w = float(content_size[0])
    content_h = float(content_size[1])
    container_w = float(container_size[0])
    container_h = float(container_size[1])

    if content_w <= 0 or content_h <= 0:
        raise ValueError(f"Invalid content size: {content_size}")
    if container_w <= 0 or container_h <= 0:
        raise ValueError(f"Invalid container size: {container_size}")

    content_aspect = content_w / content_h
    container_aspect = container_w / container_h
    if content_aspect >= container_aspect:
        target_h = container_h
        target_w = target_h * content_aspect
    else:
        target_w = container_w
        target_h = target_w / content_aspect
    return float(target_w), float(target_h)


def play_video_fill_screen(
    win: visual.Window,
    video_path: Union[str, Path],
    logger=None,
    bg_rect=None,
    msg_logger=None,
    allow_escape: bool = True,
    no_audio: bool = False,
) -> Dict[str, Any]:
    video_file = Path(video_path)
    if not video_file.exists():
        raise FileNotFoundError(f"Video file not found: {video_file}")

    movie = visual.MovieStim(
        win,
        filename=str(video_file),
        units="pix",
        size=None,
        pos=(0.0, 0.0),
        loop=False,
        autoStart=False,
        noAudio=bool(no_audio),
    )

    video_size = tuple(movie.videoSize)
    target_size = compute_cover_size(video_size, tuple(win.size))
    movie.size = target_size
    movie.pos = (0.0, 0.0)

    if msg_logger is not None:
        try:
            msg_logger.log(
                "INFO",
                (
                    f"video_scaling file={video_file.name} "
                    f"video_size={video_size} win_size={tuple(win.size)} draw_size={target_size}"
                ),
            )
        except Exception:
            pass

    first_flip_ps = None
    first_flip_perf = None
    end_perf = None
    prev_frame_idx = None
    dropped_frames = 0
    aborted = False
    movie.play(log=False)

    try:
        while True:
            if allow_escape and event.getKeys(["escape"]):
                aborted = True
                if logger is not None:
                    logger.log("abort", image_name=video_file.name, notes="escape_pressed_during_video")
                break

            if bg_rect is not None:
                bg_rect.draw()
            movie.draw()
            flip_ps = win.flip()
            flip_perf = time.perf_counter()
            frame_idx = movie.frameIndex

            if first_flip_ps is None:
                first_flip_ps = flip_ps
                first_flip_perf = flip_perf
                if logger is not None:
                    logger.log(
                        "video_start",
                        image_name=video_file.name,
                        requested_duration_s=movie.duration if movie.duration and movie.duration > 0 else None,
                        flip_time_psychopy_s=first_flip_ps,
                        flip_time_perf_s=first_flip_perf,
                        end_time_perf_s=None,
                        notes=(
                            f"video_size={video_size} draw_size=({target_size[0]:.1f},{target_size[1]:.1f}) "
                            f"dropped_frames=0"
                        ),
                    )

            if prev_frame_idx is not None and frame_idx is not None and frame_idx > prev_frame_idx + 1:
                dropped_now = int(frame_idx - prev_frame_idx - 1)
                dropped_frames += dropped_now
                if logger is not None:
                    logger.log(
                        "video_frames_dropped",
                        image_name=video_file.name,
                        requested_duration_s=None,
                        flip_time_psychopy_s=flip_ps,
                        flip_time_perf_s=flip_perf,
                        end_time_perf_s=None,
                        notes=(
                            f"dropped_now={dropped_now} total_dropped={dropped_frames} "
                            f"prev_frame_idx={prev_frame_idx} frame_idx={frame_idx}"
                        ),
                    )

            if frame_idx is not None:
                prev_frame_idx = int(frame_idx)

            if movie.isFinished:
                break

        if bg_rect is not None:
            bg_rect.draw()
        clear_flip_ps = win.flip()
        end_perf = time.perf_counter()
        if logger is not None:
            logger.log(
                "video_end",
                image_name=video_file.name,
                requested_duration_s=movie.duration if movie.duration and movie.duration > 0 else None,
                flip_time_psychopy_s=clear_flip_ps,
                flip_time_perf_s=None,
                end_time_perf_s=end_perf,
                notes=f"dropped_frames={dropped_frames} aborted={int(aborted)}",
            )
    finally:
        try:
            movie.stop(log=False)
        except Exception:
            pass
        try:
            movie.unload(log=False)
        except Exception:
            pass

    return {
        "video_name": video_file.name,
        "video_path": video_file,
        "start_flip_psychopy_s": first_flip_ps,
        "start_flip_perf_s": first_flip_perf,
        "end_time_perf_s": end_perf,
        "dropped_frames": int(dropped_frames),
        "aborted": bool(aborted),
        "video_size": tuple(video_size),
        "draw_size": tuple(target_size),
    }


def detect_frame_rate(win: visual.Window, msg_logger=None) -> Tuple[float, float]:
    """Detect the display refresh rate and return (fps, frameDur_s).

    Attempts to use Window.getActualFrameRate(); falls back to 60 Hz if unavailable.
    Logs detection to the optional message logger.
    """
    fps = None
    try:
        fps = win.getActualFrameRate(nIdentical=20, nMaxFrames=120, nWarmUpFrames=10, threshold=1)
        # getActualFrameRate may return a float; if it's nonsensical (<= 0), ignore.
        if fps is not None and fps <= 0:
            fps = None
    except Exception:
        fps = None
    if fps is None:
        fps = 60.0
        if msg_logger is not None:
            try:
                msg_logger.log("WARN", f"Frame rate detection failed; using fallback fps={fps:.3f}")
            except Exception:
                pass
    frame_dur = 1.0 / float(fps)
    if msg_logger is not None:
        try:
            msg_logger.log("INFO", f"frame_timing fps={fps:.6f} frame_dur_s={frame_dur:.9f}")
        except Exception:
            pass
    return float(fps), float(frame_dur)


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
            if idv in out:
                raise ValueError(f"Duplicate color ID in TSV: {idv}")
            out[idv] = (r, g, b)
    return out


def split_background_from_palette(
    colors: Dict[int, Tuple[int, int, int]]
) -> Tuple[Tuple[int, int, int], Dict[int, Tuple[int, int, int]]]:
    """Split first TSV row (background) from subsequent color definitions.

    `colors` must preserve file row order (as produced by `load_color_palette`).
    Returns (bg_rgb, remaining_colors).
    """
    if not colors:
        raise ValueError("colors_tsv is empty; expected at least background row plus color definitions")

    ordered_items = list(colors.items())
    _bg_id, bg_rgb = ordered_items[0]
    remaining = dict(ordered_items[1:])
    if not remaining:
        raise ValueError("colors_tsv must include at least one color definition after the background row")
    return bg_rgb, remaining


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


def _send_led_pulse_on_flip(chip, pin: int, duration_us: int):
    """GPIO pulse callback executed by PsychoPy at flip time.
    
    This is called by PsychoPy's callOnFlip mechanism, ensuring the GPIO write
    happens at the exact moment the frame is presented, minimizing latency.
    
    Args:
        chip: lgpio chip handle
        pin: GPIO pin number (BCM numbering)
        duration_us: pulse duration in microseconds
    """
    import lgpio
    
    # Set pin HIGH immediately (called at flip time by PsychoPy)
    lgpio.gpio_write(chip, pin, 1)
    
    # Use hardware-timed pulse to turn it off after duration
    result = lgpio.tx_pulse(chip, pin, 0, duration_us, 0, 1)
    
    # If pulse fails, at least turn the pin back off
    if result < 0:
        try:
            lgpio.gpio_write(chip, pin, 0)
        except Exception:
            pass
        raise RuntimeError(f"Hardware pulse failed with code {result} during flip callback")


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
    msg_logger=None,
    fps: Optional[float] = None,
    raspi: bool = False,
    pigpio_pi=None,
    raspi_pin: int = 18,
    sequential: bool = True,
    is_memory: bool = True,
    choice_hitbox_scale: float = 1.0,
    trial_meta: Optional[Dict[str, Any]] = None,
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
    # stims to potentially keep visible during the choice period when is_memory is False
    stims_for_choice: List[visual.ImageStim] = []
    # Establish frame timing
    if fps is None:
        fps, frame_dur = detect_frame_rate(win, msg_logger=msg_logger)
    else:
        frame_dur = 1.0 / float(fps)

    def _q_to_frames(seconds: float, at_least_one: bool = True) -> Tuple[int, float]:
        frames = int(round(max(0.0, float(seconds)) * float(fps)))
        if at_least_one:
            frames = max(1, frames)
        actual_s = frames * frame_dur
        return frames, actual_s

    def _set_initiation_time(perf_s: Optional[float] = None):
        if trial_meta is None or "initiation_time_iso" in trial_meta:
            return
        trial_meta["initiation_time_iso"] = dt.datetime.now().isoformat(timespec="milliseconds")
        if perf_s is not None:
            trial_meta["initiation_time_perf_s"] = float(perf_s)

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
        
        # If Raspberry Pi GPIO is requested, register callback to send pulse at flip time.
        # This ensures minimal latency between visual flip and GPIO pulse.
        if raspi and pigpio_pi is not None:
            try:
                pulse_frames, pulse_s = _q_to_frames(0.25, at_least_one=True)
                duration_us = int(pulse_s * 1_000_000)
                # Register callback to execute at flip time
                win.callOnFlip(_send_led_pulse_on_flip, pigpio_pi, raspi_pin, duration_us)
                if msg_logger is not None:
                    try:
                        msg_logger.log("INFO", f"raspi_pulse_registered block={block_idx} duration_s={pulse_s:.6f}")
                    except Exception:
                        pass
            except Exception as e:
                # Failed to register pulse callback
                error_msg = f"CRITICAL: Failed to register GPIO pulse callback: {e}. Task cannot continue."
                if msg_logger is not None:
                    try:
                        msg_logger.log("ERROR", error_msg)
                    except Exception:
                        pass
                print(f"\n{error_msg}", file=sys.stderr)
                logger.log("abort", image_name="", notes=f"gpio_callback_registration_failed: {e}")
                win.close()
                _core.quit()
                return True, None
        
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
        
        # Log successful pulse execution if raspi was enabled
        if raspi and pigpio_pi is not None:
            if msg_logger is not None:
                try:
                    msg_logger.log("INFO", f"raspi_pulse_executed block={block_idx} at_flip_time")
                except Exception:
                    pass

        # wait for click within onset cue
        onset_start = time.perf_counter()
        while True:
            if _event.getKeys(["escape"]):
                logger.log("abort", image_name="", notes="escape_pressed")
                win.close()
                _core.quit()
                return True, None

            # Get position first to ensure touchscreen events are processed synchronously
            click_pos = mouse.getPos()
            buttons = mouse.getPressed()
            if any(buttons):
                # determine bounding box from onset_cue.size and pos
                try:
                    oc_w, oc_h = onset_cue.size
                except Exception:
                    oc_w, oc_h = (200, 200)
                oc_x, oc_y = getattr(onset_cue, "pos", (0, 0))
                if abs(click_pos[0] - oc_x) <= oc_w / 2.0 and abs(click_pos[1] - oc_y) <= oc_h / 2.0:
                    click_perf = time.perf_counter()
                    _set_initiation_time(click_perf)
                    logger.log(
                        "onset_cue_clicked",
                        image_name="onset_cue",
                        requested_duration_s=None,
                        flip_time_psychopy_s=None,
                        flip_time_perf_s=click_perf,
                        end_time_perf_s=None,
                        notes=f"block={block_idx} click_xy=({click_pos[0]:.1f},{click_pos[1]:.1f})",
                    )
                    # Remove onset cue on the next flip so we can tie a flip timestamp
                    bg_rect.draw()
                    if fix is not None:
                        fix.draw()
                    rem_flip = win.flip()
                    rem_perf = time.perf_counter()
                    logger.log(
                        "onset_cue_removed",
                        image_name="onset_cue",
                        requested_duration_s=None,
                        flip_time_psychopy_s=rem_flip,
                        flip_time_perf_s=rem_perf,
                        end_time_perf_s=None,
                        notes=f"block={block_idx} click_to_flip_delay_s={(rem_perf - click_perf):.6f}",
                    )
                    break

            _core.wait(0.01)


    # Quantize durations to frames and log rounding in message logger
    stim_frames, stim_s = _q_to_frames(duration, at_least_one=True)
    isi_frames, isi_s = _q_to_frames(isi, at_least_one=False)
    choice_frames, choice_s = _q_to_frames(choice_time, at_least_one=False)
    if msg_logger is not None:
        try:
            msg_logger.log(
                "INFO",
                (
                    f"timing_quantization block={block_idx} "
                    f"stim_duration={duration:.6f}s-> {stim_frames}fr({stim_s:.6f}s) "
                    f"isi={isi:.6f}s-> {isi_frames}fr({isi_s:.6f}s) "
                    f"choice_time={choice_time:.6f}s-> {choice_frames}fr({choice_s:.6f}s)"
                ),
            )
        except Exception:
            pass

    chosen_info = None
    click_registered = False
    click_perf_capture = None
    click_meta = None
    prev_touch_down = False
    poll_interval_s = 0.002
    touch_acquire_window_s = 0.050
    choice_started = False
    choice_flip = None
    choice_perf = None
    choice_window_s = float(choice_s)
    choice_deadline = None
    pos_list = list(positions)
    stims: List[visual.ImageStim] = []
    names: List[str] = []
    choice_hit_targets: List[visual.Rect] = []

    def _build_choice_hit_targets():
        if choice_hit_targets:
            return
        if len(stim_sizes) < len(pos_list):
            return
        for ppos, stim_size in zip(pos_list, stim_sizes):
            try:
                w = max(1.0, float(stim_size[0]) * float(choice_hitbox_scale))
                h = max(1.0, float(stim_size[1]) * float(choice_hitbox_scale))
            except Exception:
                w, h = (64.0, 64.0)
            target = visual.Rect(
                win,
                width=w,
                height=h,
                pos=ppos,
                units="pix",
                fillColor=None,
                lineColor=None,
                opacity=0.0,
            )
            choice_hit_targets.append(target)

    def _match_choice_target(click_pos: Tuple[float, float]) -> Optional[int]:
        _build_choice_hit_targets()
        for i, target in enumerate(choice_hit_targets, start=1):
            try:
                if target.contains(click_pos):
                    return i
            except Exception:
                pass
        return None

    def _start_choice_window(start_flip_ps, start_perf: float, window_s: float, notes_suffix: str = ""):
        nonlocal choice_started, choice_flip, choice_perf, choice_window_s, choice_deadline, prev_touch_down
        if choice_started:
            return
        choice_started = True
        choice_flip = start_flip_ps
        choice_perf = float(start_perf)
        choice_window_s = max(0.0, float(window_s))
        choice_deadline = choice_perf + choice_window_s
        prev_touch_down = any(mouse.getPressed())
        logger.log(
            "choice_start",
            image_name="",
            requested_duration_s=choice_window_s,
            flip_time_psychopy_s=choice_flip,
            flip_time_perf_s=choice_perf,
            end_time_perf_s=choice_deadline,
            notes=f"block={block_idx}{notes_suffix}",
        )

    def _poll_choice_until(deadline_perf: float) -> bool:
        nonlocal click_registered, click_perf_capture, click_meta, prev_touch_down, chosen_info
        while time.perf_counter() < deadline_perf and not click_registered:
            keys = _event.getKeys(["escape"])
            if keys:
                logger.log("abort", image_name="", notes="escape_pressed")
                win.close()
                _core.quit()
                return True

            click_pos = mouse.getPos()
            buttons = mouse.getPressed()
            touch_down = any(buttons)
            touch_started = touch_down and (not prev_touch_down)
            prev_touch_down = touch_down

            if touch_down and choice_perf is not None:
                touch_onset_perf = time.perf_counter()
                chosen_idx = _match_choice_target(click_pos)

                # Some touch backends report button-down before the final touch
                # coordinates are visible through Mouse.getPos(). Allow a brief
                # acquisition window to pick up the coordinate update for the same tap.
                if chosen_idx is None and touch_started and choice_deadline is not None:
                    acquire_deadline = min(choice_deadline, touch_onset_perf + touch_acquire_window_s)
                    while time.perf_counter() < acquire_deadline:
                        _event.getKeys([])
                        click_pos = mouse.getPos()
                        chosen_idx = _match_choice_target(click_pos)
                        if chosen_idx is not None:
                            break
                        remaining_acquire = acquire_deadline - time.perf_counter()
                        if remaining_acquire > 0:
                            _core.wait(min(poll_interval_s, remaining_acquire))

                if touch_started and msg_logger is not None:
                    try:
                        msg_logger.log(
                            "INFO",
                            f"choice_touch_attempt block={block_idx} click_xy=({click_pos[0]:.1f},{click_pos[1]:.1f}) matched_idx={chosen_idx}",
                        )
                    except Exception:
                        pass

                if chosen_idx is not None:
                    click_perf_capture = touch_onset_perf
                    chosen_info = {
                        "chosen_index": int(chosen_idx),
                        "chosen_pos": tuple(pos_list[chosen_idx - 1]),
                        "choice_start_perf_s": float(choice_perf),
                        "choice_time_perf_s": float(click_perf_capture),
                        "reaction_time_s": float(click_perf_capture - choice_perf),
                        "choice_time_psychopy_s": None,
                        "notes": f"block={block_idx}",
                    }
                    logger.log(
                        "choice_made",
                        image_name=getattr(block_paths[chosen_idx - 1], "name", str(block_paths[chosen_idx - 1])),
                        requested_duration_s=None,
                        flip_time_psychopy_s=None,
                        flip_time_perf_s=click_perf_capture,
                        end_time_perf_s=None,
                        notes=f"block={block_idx} idx={chosen_idx} click_xy=({click_pos[0]:.1f},{click_pos[1]:.1f})",
                    )
                    click_meta = {"idx": chosen_idx}
                    click_registered = True
                    break

            remaining = deadline_perf - time.perf_counter()
            if remaining > 0:
                _core.wait(min(poll_interval_s, remaining))
        return False

    # If sequential: original behavior (per-stimulus loop)
    if sequential:
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
            # keep a reference in case we need to show the stimulus during choice (is_memory False)
            stims_for_choice.append(stim)
            # record stimulus pixel size for click hit-testing later
            try:
                stim_sizes.append(tuple(stim.size))
            except Exception:
                stim_sizes.append((0.0, 0.0))
            # Optionally show the dot for `isi` seconds before the stimulus
            # appears. We create the dot first (so it persists) and draw it along
            # with existing dots, then wait `isi` seconds before showing the stim.
            if isi_frames > 0:
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

                # Pre-stimulus dot/cue for exactly isi_frames frames
                first_flip = True
                dot_on_perf = None
                for _f in range(isi_frames if isi_frames > 0 else 0):
                    bg_rect.draw()
                    for d in dots:
                        d.draw()
                    if fix is not None:
                        fix.draw()
                    dot_flip = win.flip()
                    if first_flip:
                        dot_on_perf = time.perf_counter()
                        _set_initiation_time(dot_on_perf)
                        logger.log(
                            "dot_on",
                            image_name=name,
                            requested_duration_s=isi_s,
                            flip_time_psychopy_s=dot_flip,
                            flip_time_perf_s=dot_on_perf,
                            end_time_perf_s=(dot_on_perf + isi_s) if dot_on_perf is not None else None,
                            notes=f"block={block_idx} idx={idx}",
                        )
                        first_flip = False

            # Draw background, existing dots, then current stim and fixation
            first_flip = True
            for _f in range(stim_frames):
                bg_rect.draw()
                for d in dots:
                    d.draw()
                stim.draw()
                if fix is not None:
                    fix.draw()
                flip_ps = win.flip()
                if first_flip:
                    flip_perf = time.perf_counter()
                    _set_initiation_time(flip_perf)
                    logger.log(
                        "stim_on",
                        image_name=name,
                        requested_duration_s=stim_s,
                        flip_time_psychopy_s=flip_ps,
                        flip_time_perf_s=flip_perf,
                        end_time_perf_s=None,
                        notes=f"block={block_idx} idx={idx}",
                    )
                    first_flip = False

            # After stimulus: either persist a dot (memory task) or keep the stimulus visible
            if dots and is_memory:
                last_dot = dots[-1]
                # set persistent color
                last_dot.fillColor = rgb255_to_psychopy(dot_color)
                last_dot.fillColorSpace = "rgb"
            elif dots and not is_memory:
                # remove the last pre-stimulus cue dot so it doesn't persist
                try:
                    dots.pop()
                except Exception:
                    pass

            # draw background + either all dots or kept stimuli + fixation (so the correct cue remains visible)
            bg_rect.draw()
            if is_memory:
                for d in dots:
                    d.draw()
            else:
                for s in stims_for_choice:
                    s.draw()
            if fix is not None:
                fix.draw()
            off_perf = time.perf_counter()
            off_flip = win.flip()
            logger.log(
                "stim_off",
                image_name=name,
                requested_duration_s=stim_s,
                flip_time_psychopy_s=None,
                flip_time_perf_s=None,
                end_time_perf_s=off_perf,
                notes=f"block={block_idx} idx={idx}",
            )

            if (not is_memory) and idx == len(block_paths):
                _start_choice_window(off_flip, off_perf, choice_s, notes_suffix=" response_from=all_visible")

            # small safety: check for abort
            if event.getKeys(["escape"]):
                logger.log("abort", image_name="", notes="escape_pressed")
                win.close()
                _core.quit()
                return True, None

        # After all stimuli in this block were shown, enter the choice period.
    else:
        # Non-sequential: present all items at once.
        stims = []
        names = []
        for idx, (p, pos) in enumerate(zip(block_paths, positions), start=1):
            if isinstance(p, tuple) and len(p) == 2:
                sid, cid = p
                name = f"shape{sid}_color{cid}"
                pil_img = preloaded.get((sid, cid))
                if pil_img is None:
                    pil_img = preloaded.get(p)
            else:
                name = getattr(p, "name", str(p))
                pil_img = preloaded[p]
            stim = make_image_stim_from_array(win, pil_img, size=None, bg_rgb_255=bg_rgb_255)
            stim.pos = pos
            stims.append(stim)
            names.append(name)
            try:
                stim_sizes.append(tuple(stim.size))
            except Exception:
                stim_sizes.append((0.0, 0.0))

        # Optionally show pre-stimulus dots for isi duration
        if isi_frames > 0:
            cue_color = init_dot_color if init_dot_color is not None else dot_color
            for pos in positions:
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

            first_flip = True
            dot_on_perf = None
            for _f in range(isi_frames if isi_frames > 0 else 0):
                bg_rect.draw()
                for d in dots:
                    d.draw()
                if fix is not None:
                    fix.draw()
                dot_flip = win.flip()
                if first_flip:
                    dot_on_perf = time.perf_counter()
                    _set_initiation_time(dot_on_perf)
                    # log dot_on for each stim
                    for idx, name in enumerate(names, start=1):
                        logger.log(
                            "dot_on",
                            image_name=name,
                            requested_duration_s=isi_s,
                            flip_time_psychopy_s=dot_flip,
                            flip_time_perf_s=dot_on_perf,
                            end_time_perf_s=(dot_on_perf + isi_s) if dot_on_perf is not None else None,
                            notes=f"block={block_idx} idx={idx}",
                        )
                    first_flip = False

        # Show all stimuli simultaneously for stim_frames
        first_flip = True
        for _f in range(stim_frames):
            bg_rect.draw()
            for d in dots:
                d.draw()
            for s in stims:
                s.draw()
            if fix is not None:
                fix.draw()
            flip_ps = win.flip()
            flip_perf = time.perf_counter()
            if first_flip:
                _set_initiation_time(flip_perf)
                for idx, name in enumerate(names, start=1):
                    logger.log(
                        "stim_on",
                        image_name=name,
                        requested_duration_s=stim_s,
                        flip_time_psychopy_s=flip_ps,
                        flip_time_perf_s=flip_perf,
                        end_time_perf_s=None,
                        notes=f"block={block_idx} idx={idx}",
                    )
                if not is_memory:
                    _build_choice_hit_targets()
                    _start_choice_window(flip_ps, flip_perf, stim_s + choice_s, notes_suffix=" response_from=stim_on")
                first_flip = False
            if choice_started and choice_deadline is not None:
                if _poll_choice_until(min(choice_deadline, flip_perf + frame_dur)):
                    return True, None
                if click_registered:
                    break

        # After stimuli: either persist dots (memory) or keep stimuli visible
        if click_registered:
            off_perf = click_perf_capture if click_perf_capture is not None else time.perf_counter()
            for idx, name in enumerate(names, start=1):
                logger.log(
                    "stim_off",
                    image_name=name,
                    requested_duration_s=stim_s,
                    flip_time_psychopy_s=None,
                    flip_time_perf_s=None,
                    end_time_perf_s=off_perf,
                    notes=f"block={block_idx} idx={idx} early_choice=1",
                )
        else:
            if is_memory:
                for d in dots:
                    d.fillColor = rgb255_to_psychopy(dot_color)
                    d.fillColorSpace = "rgb"
                # draw background + all dots + fixation and log stim_off for each
                bg_rect.draw()
                for d in dots:
                    d.draw()
                if fix is not None:
                    fix.draw()
                off_perf = time.perf_counter()
                win.flip()
            else:
                # draw background + all kept stimuli + fixation and log stim_off for each
                bg_rect.draw()
                for s in stims:
                    s.draw()
                if fix is not None:
                    fix.draw()
                off_perf = time.perf_counter()
                win.flip()
            for idx, name in enumerate(names, start=1):
                logger.log(
                    "stim_off",
                    image_name=name,
                    requested_duration_s=stim_s,
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
    # If a response window has not started yet, start it on the first frame
    # where the selectable targets for this mode are actually visible.
    if not click_registered:
        if not choice_started:
            bg_rect.draw()
            if is_memory:
                for d in dots:
                    d.draw()
            else:
                for s in (stims_for_choice if sequential else stims):
                    s.draw()
            if fix is not None:
                fix.draw()
            choice_flip = win.flip()
            choice_perf_now = time.perf_counter()
            _build_choice_hit_targets()
            _start_choice_window(choice_flip, choice_perf_now, choice_s)

        if choice_deadline is not None:
            if _poll_choice_until(choice_deadline):
                return True, None

    if click_registered:
        # Clear dots on the next flip and log the flip timestamp
        bg_rect.draw()
        if fix is not None:
            fix.draw()
        clr_flip = win.flip()
        clr_perf = time.perf_counter()
        logger.log(
            "choice_cleared",
            image_name="",
            requested_duration_s=None,
            flip_time_psychopy_s=clr_flip,
            flip_time_perf_s=clr_perf,
            end_time_perf_s=None,
            notes=f"block={block_idx} idx={click_meta.get('idx') if click_meta else ''} click_to_flip_delay_s={(clr_perf - (click_perf_capture or clr_perf)):.6f}",
        )
    else:
        # Timed out without a valid selection
        logger.log("choice_end", image_name="", notes=f"block={block_idx} no_choice")
        # Clear dots on the next flip and log
        bg_rect.draw()
        if fix is not None:
            fix.draw()
        clr_flip = win.flip()
        clr_perf = time.perf_counter()
        logger.log(
            "choice_cleared",
            image_name="",
            requested_duration_s=None,
            flip_time_psychopy_s=clr_flip,
            flip_time_perf_s=clr_perf,
            end_time_perf_s=None,
            notes=f"block={block_idx} timeout_to_flip_delay_s={frame_dur:.6f}",
        )

    return False, chosen_info

    # --- simultaneous (non-sequential) branch ---
    # Note: unreachable when sequential branch returns; keep for clarity


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


# -----------------------------------------------------------------------------------------
# Trial Buffer Manager (for background trial generation with multiprocessing)
# -----------------------------------------------------------------------------------------

def _trial_buffer_worker_generic(
    trial_generator_func: Callable[[int, dict], dict],
    config: dict,
    trial_queue: Any,
    stop_event: Any,
    start_idx: int = 0
):
    """
    Generic worker process that generates trials in the background.
    
    Args:
        trial_generator_func: Callable that takes (trial_idx, config) and returns trial dict
        config: Configuration dictionary to pass to the generator
        trial_queue: Queue to push generated trials into
        stop_event: Event to signal worker to stop
        start_idx: Starting trial index
    """
    trial_idx = start_idx
    while not stop_event.is_set():
        try:
            trial_data = trial_generator_func(trial_idx, config)
            while not stop_event.is_set():
                try:
                    trial_queue.put(trial_data, timeout=0.1)
                    break
                except queue.Full:
                    continue
            trial_idx += 1
        except Exception as e:
            # Put error into queue for main process to handle
            error_text = str(e) or repr(e)
            error_trace = traceback.format_exc()
            while not stop_event.is_set():
                try:
                    trial_queue.put(
                        {
                            "type": "error",
                            "error": error_text,
                            "traceback": error_trace,
                            "trial_idx": trial_idx,
                        },
                        timeout=0.1,
                    )
                    break
                except queue.Full:
                    continue
            break


class TrialBufferManager:
    """
    Generic trial buffer manager that uses multiprocessing to pre-generate trials
    on a separate core. Works with any task paradigm by taking a user-defined
    trial generation callable.
    
    The trial_generator_func should be a function that takes:
        - trial_idx: int (the index of the trial to generate)
        - config: dict (containing any parameters needed for generation)
    
    And returns a dictionary representing the trial data.
    
    Example usage:
        def my_trial_generator(trial_idx: int, config: dict) -> dict:
            # Generate trial based on config
            return {"trial_idx": trial_idx, "stimuli": [...], ...}
        
        buffer_mgr = TrialBufferManager(
            trial_generator_func=my_trial_generator,
            config={"param1": value1, "param2": value2},
            buffer_size=5
        )
        
        # In your task loop:
        trial_data = buffer_mgr.get_next_trial()
        # Use trial_data...
        
        # When done:
        buffer_mgr.close()
    """
    
    def __init__(
        self, 
        trial_generator_func: Callable[[int, dict], dict],
        config: dict,
        buffer_size: int = 5,
        start_idx: int = 0
    ):
        """
        Initialize the trial buffer manager.
        
        Args:
            trial_generator_func: A callable that generates trial data.
                                  Must take (trial_idx: int, config: dict) -> dict
            config: Dictionary of configuration parameters to pass to generator
            buffer_size: Maximum number of trials to buffer ahead (default: 5)
            start_idx: Starting trial index (default: 0)
        """
        self.trial_generator_func = trial_generator_func
        self.config = config
        self.buffer_size = buffer_size
        self.start_idx = start_idx
        self.next_trial_idx = start_idx
        
        # Set up multiprocessing with spawn context (required for some libraries like PsychoPy)
        ctx = mp.get_context('spawn')
        self.trial_queue = ctx.Queue(maxsize=buffer_size)
        self.stop_event = ctx.Event()
        
        # Start the worker process
        self.worker = ctx.Process(
            target=_trial_buffer_worker_generic,
            args=(trial_generator_func, config, self.trial_queue, self.stop_event, start_idx)
        )
        self.worker.start()
        self.is_closed = False
    
    def get_next_trial(self) -> dict:
        """
        Get the next pre-generated trial from the buffer.
        
        Returns:
            Dictionary containing the trial data generated by trial_generator_func
            
        Raises:
            RuntimeError: If the buffer manager has been closed or worker encountered an error
        """
        if self.is_closed:
            raise RuntimeError("TrialBufferManager has been closed")
        
        try:
            trial_data = self.trial_queue.get(timeout=30.0)
            
            # Check if worker sent an error
            if isinstance(trial_data, dict) and trial_data.get("type") == "error":
                error_text = trial_data.get("error", "Unknown worker error")
                trial_idx = trial_data.get("trial_idx")
                trace_text = trial_data.get("traceback")
                details = f"Trial generation error at trial_idx={trial_idx}: {error_text}"
                if trace_text:
                    details = f"{details}\n{trace_text}"
                raise RuntimeError(details)
            
            self.next_trial_idx += 1
            return trial_data
            
        except Exception as e:
            self.close()
            raise RuntimeError(f"Failed to get next trial: {e}")
    
    def close(self):
        """
        Clean up the worker process and release resources.
        Should be called when done using the buffer manager.
        """
        if self.is_closed:
            return
            
        self.is_closed = True
        self.stop_event.set()
        
        # Give worker time to finish cleanly
        self.worker.join(timeout=2.0)
        
        # Force terminate if still alive
        if self.worker.is_alive():
            self.worker.terminate()
            self.worker.join(timeout=1.0)
    
    def __del__(self):
        """Destructor to ensure cleanup happens even if close() not called."""
        self.close()
