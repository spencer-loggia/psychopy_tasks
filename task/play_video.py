"""
Play randomly selected videos from a directory and log playback timing.

By default this task selects from `task/resources/cropped_videos`, where clips
have already been cropped, downsampled, stripped of audio, and converted to a
playback-friendly HEVC/H.265 yuv420p format.
"""
import argparse
import random
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

from psychopy import core, event, logging as pylogging

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from bin import utils
from bin.config import load_config, validate_config
from bin.logger import SessionLogBundle
from bin.screen import (
    ExperimenterPreview,
    describe_screen,
    load_screen_config,
    resolve_scene_size,
    resolve_task_screens,
)


def _resolve_videos_dir(videos_dir: str) -> tuple[Path, Optional[str]]:
    requested = Path(videos_dir)
    if requested.exists() and requested.is_dir():
        return requested, None

    # Be tolerant of setups that haven't run preprocessing yet. The checked-in
    # config and docstring prefer `cropped_videos`, but some local repos still
    # only have the source `video` directory.
    if requested.name == "cropped_videos":
        fallback = requested.with_name("video")
        if fallback.exists() and fallback.is_dir():
            return fallback, (
                f"videos_dir_missing requested={requested} fallback_to={fallback}"
            )

    raise FileNotFoundError(f"Video directory not found: {videos_dir}")


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
    parser.add_argument("--main_screen", default=None, help="Main task screen index or output name")
    parser.add_argument("--experimenter_screen", default=None, help="Experimenter screen index or output name")
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
    screen_config=None,
):
    if seed is not None:
        random.seed(seed)

    resolved_videos_dir, videos_dir_warning = _resolve_videos_dir(videos_dir)
    video_files = utils.find_video_files(str(resolved_videos_dir), recursive=False)
    if not video_files:
        raise FileNotFoundError(f"No videos found in {resolved_videos_dir}")

    main_screen, experimenter_screen = resolve_task_screens(screen_config)
    win = utils.setup_window(bg_rgb_255=bg, fullscreen=fullscreen, size=win_size, screen_info=main_screen)
    bg_rect = utils.make_bg_rect(win, bg)
    mouse = event.Mouse(win=win)
    experimenter_preview = None
    main_scene_size = resolve_scene_size(
        main_screen,
        fullscreen=bool(fullscreen),
        requested_size=win_size,
        realized_size=tuple(win.size),
    )

    resolved_config_name = str(config_name).strip() if config_name else "play_video"
    session_logs = SessionLogBundle(
        output_root=output_dir,
        task_name="play_video",
        config_name=resolved_config_name,
        behavior_fieldnames=["trial_num", "video_name", "expected_duration", "aborted", "stop_reason", "dropped_frames"],
    )
    logger = session_logs.event_logger
    msg_logger = session_logs.message_logger
    behavior_logger = session_logs.behavior_logger
    if behavior_logger is None:
        raise RuntimeError("play_video requires a behavior logger")
    if experimenter_screen is not None:
        experimenter_preview = ExperimenterPreview(
            experimenter_screen,
            task_label=resolved_config_name,
            start_perf_s=time.perf_counter(),
            update_interval_s=0.1,
        )
        experimenter_preview.clear_scene(bg_rgb_255=bg, main_size=main_scene_size)
    pylogging.console.setLevel(pylogging.CRITICAL)
    try:
        msg_logger.log(
            "INFO",
            f"session_start task=play_video config_name={resolved_config_name} session_dir={session_logs.session_dir}",
        )
        msg_logger.log(
            "INFO",
            f"resolved_screens main={describe_screen(main_screen)} experimenter={describe_screen(experimenter_screen)}",
        )
        if videos_dir_warning:
            msg_logger.log("WARN", videos_dir_warning)
        msg_logger.log(
            "INFO",
            f"resolved_main_scene_size size={main_scene_size[0]}x{main_scene_size[1]} fullscreen={int(bool(fullscreen))} requested_win_size={win_size} realized_win_size={tuple(win.size)}",
        )
    except Exception:
        pass

    if refresh_rate is not None and float(refresh_rate) > 0:
        fps = float(refresh_rate)
        frame_dur = 1.0 / fps
        try:
            msg_logger.log("INFO", f"fps_override refresh_rate={fps:.6f}Hz frame_dur_s={frame_dur:.9f}")
        except Exception:
            pass
    else:
        fps, frame_dur = utils.detect_frame_rate(win, msg_logger=msg_logger)
    msg_logger.log("INFO", f"task_ready videos_dir={resolved_videos_dir.resolve()} fps={fps:.6f} n_videos={len(video_files)}")

    try:
        event.clearEvents(eventType="mouse")
        mouse.clickReset()
    except Exception:
        pass
    playback_info = None
    played_videos = 0
    stop_reason = "mouse_click"
    while True:
        if experimenter_preview is not None and experimenter_preview.poll():
            stop_reason = "experimenter_exit"
            msg_logger.log("WARN", "experimenter_exit_before_video")
            break
        try:
            is_pressed = any(mouse.getPressed())
        except Exception:
            is_pressed = False
        if is_pressed:
            stop_reason = "mouse_click"
            break

        chosen_video = random.choice(video_files)
        if experimenter_preview is not None:
            experimenter_preview.play_video(str(chosen_video), bg_rgb_255=bg, main_size=main_scene_size)
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
            external_abort_checker=(experimenter_preview.poll if experimenter_preview is not None else None),
            trial_num=played_videos + 1,
        )
        played_videos += 1
        behavior_logger.writerow(
            {
                "trial_num": played_videos,
                "video_name": playback_info["video_name"],
                "expected_duration": (
                    f"{float(playback_info['expected_duration_s']):.9f}"
                    if playback_info.get("expected_duration_s") is not None
                    else ""
                ),
                "aborted": int(playback_info["aborted"]),
                "stop_reason": playback_info.get("abort_reason") or "completed",
                "dropped_frames": playback_info["dropped_frames"],
            }
        )
        if experimenter_preview is not None:
            experimenter_preview.clear_scene(bg_rgb_255=bg, main_size=main_scene_size)
        if playback_info["aborted"]:
            stop_reason = playback_info.get("abort_reason") or "aborted"
            break

    msg_logger.log(
        "INFO",
        (
            f"session_end status={stop_reason} played_videos={played_videos} "
            f"stop_reason={stop_reason} "
            f"last_video={playback_info['video_name'] if playback_info is not None else ''}"
        ),
    )
    session_logs.close()
    try:
        if experimenter_preview is not None:
            experimenter_preview.close()
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

    screen_config = load_screen_config(
        cfg,
        cli_main=args.main_screen,
        cli_experimenter=args.experimenter_screen,
    )

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
            screen_config=screen_config,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
