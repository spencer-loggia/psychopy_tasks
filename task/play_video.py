"""
Play randomly selected videos from a directory and log playback timing.

By default this task selects from `task/resources/cropped_videos`, where clips
have already been cropped, downsampled, stripped of audio, and converted to a
playback-friendly HEVC/H.265 yuv420p format.
"""
import argparse
import datetime as dt
import random
import sys
from pathlib import Path
from typing import Optional, Tuple

from psychopy import core, event, logging as pylogging

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from bin import utils
from bin.config import load_config, validate_config
from bin.logger import EventLogger, MessageLogger, build_run_log_filename


def parse_args():
    parser = argparse.ArgumentParser(description="Play a random video stimulus")
    parser.add_argument("--config", help="Path to JSON config file. CLI overrides config keys.")
    parser.add_argument("--videos_dir", default=None, help="Directory containing preprocessed video files")
    parser.add_argument("--output_dir", default=None, help="Directory to save logs")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--fullscreen", action="store_true", default=None, help="Run fullscreen")
    parser.add_argument("--win_size", type=int, nargs=2, default=None, help="Window size when not fullscreen")
    parser.add_argument("--bg", type=int, nargs=3, default=None, help="Background RGB color")
    parser.add_argument("--refresh_rate", type=float, default=None, help="Override detected display refresh rate (Hz)")
    parser.add_argument("--ffprobe", default=None, help="Path to ffprobe for codec probing")
    return parser.parse_args()


def run_task(
    videos_dir: str,
    output_dir: str,
    seed: Optional[int] = None,
    fullscreen: bool = True,
    win_size: Optional[Tuple[int, int]] = None,
    bg: Tuple[int, int, int] = (0, 0, 0),
    refresh_rate: Optional[float] = None,
    config_name: Optional[str] = None,
    ffprobe_bin: str = "ffprobe",
):
    if seed is not None:
        random.seed(seed)

    video_files = utils.find_video_files(videos_dir, recursive=False)
    if not video_files:
        raise FileNotFoundError(f"No videos found in {videos_dir}")

    win = utils.setup_window(bg_rgb_255=bg, fullscreen=fullscreen, size=win_size)
    bg_rect = utils.make_bg_rect(win, bg)
    mouse = event.Mouse(win=win)

    run_started_dt = dt.datetime.now()
    resolved_config_name = str(config_name).strip() if config_name else "play_video"
    logger = EventLogger(
        output_dir,
        filename=build_run_log_filename(
            resolved_config_name,
            "play_video_log",
            when=run_started_dt,
            in_progress=True,
        ),
    )
    msg_logger = MessageLogger(
        output_dir,
        filename=build_run_log_filename(
            resolved_config_name,
            "play_video_message_log",
            when=run_started_dt,
            in_progress=True,
        ),
    )
    pylogging.console.setLevel(pylogging.CRITICAL)

    if refresh_rate is not None and float(refresh_rate) > 0:
        fps = float(refresh_rate)
        frame_dur = 1.0 / fps
        try:
            msg_logger.log("INFO", f"fps_override refresh_rate={fps:.6f}Hz frame_dur_s={frame_dur:.9f}")
        except Exception:
            pass
    else:
        fps, frame_dur = utils.detect_frame_rate(win, msg_logger=msg_logger)

    logger.log(
        "task_start",
        image_name="",
        requested_duration_s=None,
        flip_time_psychopy_s=None,
        flip_time_perf_s=None,
        end_time_perf_s=None,
        notes=f"videos_dir={Path(videos_dir).resolve()} fps={fps:.6f} n_videos={len(video_files)}",
    )

    try:
        event.clearEvents(eventType="mouse")
        mouse.clickReset()
    except Exception:
        pass
    playback_info = None
    played_videos = 0
    stop_reason = "mouse_click"
    while True:
        try:
            is_pressed = any(mouse.getPressed())
        except Exception:
            is_pressed = False
        if is_pressed:
            stop_reason = "mouse_click"
            break

        chosen_video = random.choice(video_files)
        playback_info = utils.play_video_fill_screen(
            win=win,
            video_path=chosen_video,
            logger=logger,
            bg_rect=bg_rect,
            msg_logger=msg_logger,
            allow_escape=True,
            stop_on_mouse_click=True,
            mouse=mouse,
            ffprobe_bin=ffprobe_bin,
        )
        played_videos += 1
        if playback_info["aborted"]:
            stop_reason = playback_info.get("abort_reason") or "aborted"
            break

    task_end_dt = dt.datetime.now()
    logger.log(
        "task_end",
        image_name=(playback_info["video_name"] if playback_info is not None else ""),
        requested_duration_s=None,
        flip_time_psychopy_s=None,
        flip_time_perf_s=None,
        end_time_perf_s=None,
        notes=(
            f"played_videos={played_videos} stop_reason={stop_reason} "
            f"aborted={int(playback_info['aborted']) if playback_info is not None else 0} "
            f"dropped_frames={playback_info['dropped_frames'] if playback_info is not None else 0} "
            f"backend={playback_info['backend_used'] if playback_info is not None else 'n/a'} "
            f"backend_dropped_frames={playback_info['backend_dropped_frames'] if playback_info is not None else 'n/a'}"
        ),
    )
    logger.finalize(build_run_log_filename(resolved_config_name, "play_video_log", when=task_end_dt))
    try:
        msg_logger.finalize(build_run_log_filename(resolved_config_name, "play_video_message_log", when=task_end_dt))
    except Exception:
        pass
    win.close()
    core.quit()


def main():
    args = parse_args()
    cfg = {}
    if args.config:
        cfg = load_config(args.config)
        validate_config(cfg, required=["config_name", "output_dir"])

    def _get(name, default=None):
        val = getattr(args, name, None)
        if val is not None:
            return val
        return cfg.get(name, default)

    try:
        run_task(
            videos_dir=_get("videos_dir", "./task/resources/cropped_videos"),
            output_dir=_get("output_dir", "./logs"),
            seed=_get("seed", None),
            fullscreen=bool(_get("fullscreen", cfg.get("fullscreen", True))),
            win_size=tuple(_get("win_size", cfg.get("win_size", None))) if _get("win_size", None) else None,
            bg=tuple(_get("bg", cfg.get("bg", (0, 0, 0)))),
            refresh_rate=_get("refresh_rate", cfg.get("refresh_rate", cfg.get("refrech_rate", None))),
            config_name=_get("config_name", cfg.get("config_name", "play_video")),
            ffprobe_bin=_get("ffprobe", cfg.get("ffprobe", "ffprobe")),
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
