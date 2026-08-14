"""Play random fixed-duration clips from explicitly configured video files."""
import argparse
import math
import random
import sys
import time
from pathlib import Path
from typing import Optional, Sequence, Tuple

from psychopy import core, event, logging as pylogging

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from bin import utils
from bin.config import load_config, validate_config
from bin.logger import SessionLogBundle
from bin.task_lifecycle import USER_EXIT_CODE
from bin.video_playback import (
    RandomFramePulseSchedule,
    SharedVideoFrameBuffer,
    find_pi_hevc_decoder_device,
    is_raspberry_pi,
    select_random_video_clip,
    validate_hevc_stream,
    video_duration_seconds,
)
from bin.screen import (
    ExperimenterPreview,
    describe_screen,
    load_screen_config,
    resolve_scene_size,
    resolve_task_screens,
)


def _resolve_video_files(video_files: Sequence[str]) -> list[Path]:
    if isinstance(video_files, (str, bytes)) or not isinstance(video_files, Sequence):
        raise ValueError("video_files must be a JSON list of video file paths")
    if not video_files:
        raise ValueError("video_files must contain at least one video file path")

    resolved: list[Path] = []
    for index, value in enumerate(video_files):
        if not isinstance(value, (str, Path)) or not str(value).strip():
            raise ValueError(f"video_files[{index}] must be a non-empty path")
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = path.absolute()
        if not path.is_file():
            raise FileNotFoundError(f"Video file not found: {path}")
        resolved.append(path)
    return resolved


def parse_args():
    parser = argparse.ArgumentParser(description="Play a random video stimulus")
    parser.add_argument("--config", help="Path to JSON config file. CLI overrides config keys.")
    parser.add_argument("--video_files", nargs="+", default=None, help="Explicit paths to source video files")
    parser.add_argument("--clip_duration_seconds", type=float, default=None, help="Duration of each randomly selected clip/trial")
    parser.add_argument("--seek_timeout_seconds", type=float, default=None, help="Maximum wait for a network source seek and first decoded frame")
    parser.add_argument("--output_dir", default=None, help="Directory to save logs")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--fullscreen", action="store_true", default=None, help="Run fullscreen")
    parser.add_argument("--win_size", type=int, nargs=2, default=None, help="Window size when not fullscreen")
    parser.add_argument("--bg", type=int, nargs=3, default=None, help="Background RGB color")
    parser.add_argument("--refresh_rate", type=float, default=None, help="Override detected display refresh rate (Hz)")
    parser.add_argument("--ffprobe", default=None, help="Path to ffprobe for codec probing")
    parser.add_argument("--raspi", action="store_true", default=None, help="Enable Pi 5 hardware validation and frame sync GPIO")
    parser.add_argument("--no_raspi", action="store_false", dest="raspi", help="Disable Raspberry Pi hardware behavior")
    parser.add_argument("--sync_pin", type=int, default=None, help="BCM GPIO pin for frame-locked video sync pulses")
    parser.add_argument("--sync_interval_frames", type=int, nargs=2, default=None, metavar=("MIN", "MAX"), help="Inclusive randomized interval between sync-pulse onsets, in display frames")
    parser.add_argument("--sync_pulse_frames", type=int, default=None, help="Sync pulse width in display frames")
    parser.add_argument("--main_screen", default=None, help="Main task screen index or output name")
    parser.add_argument("--experimenter_screen", default=None, help="Experimenter screen index or output name")
    return parser.parse_args()


def run_task(
    video_files: Sequence[str],
    clip_duration_seconds: Optional[float],
    output_dir: str,
    seed: Optional[int] = None,
    fullscreen: bool = True,
    win_size: Optional[Tuple[int, int]] = None,
    bg: Tuple[int, int, int] = (0, 0, 0),
    refresh_rate: Optional[float] = None,
    config_name: Optional[str] = None,
    ffprobe_bin: str = "ffprobe",
    screen_config=None,
    raspi: bool = False,
    sync_pin: int = 18,
    sync_interval_frames: Tuple[int, int] = (100, 300),
    sync_pulse_frames: int = 1,
    seek_timeout_seconds: float = 30.0,
):
    if clip_duration_seconds is None:
        raise ValueError("clip_duration_seconds is required")
    clip_duration_seconds = float(clip_duration_seconds)
    if not math.isfinite(clip_duration_seconds) or clip_duration_seconds <= 0.0:
        raise ValueError("clip_duration_seconds must be a positive finite value")
    seek_timeout_seconds = float(seek_timeout_seconds)
    if not math.isfinite(seek_timeout_seconds) or seek_timeout_seconds <= 0.0:
        raise ValueError("seek_timeout_seconds must be a positive finite value")
    selection_rng = random.Random(seed)
    sync_rng = random.Random(None if seed is None else int(seed) + 0x51A7)
    if len(sync_interval_frames) != 2:
        raise ValueError("sync_interval_frames must contain exactly MIN and MAX")
    sync_interval_min = int(sync_interval_frames[0])
    sync_interval_max = int(sync_interval_frames[1])
    sync_pulse_frames = int(sync_pulse_frames)
    sync_pin = int(sync_pin)
    if raspi:
        # Validate all timing parameters before opening windows or GPIO.
        RandomFramePulseSchedule(
            sync_interval_min,
            sync_interval_max,
            pulse_width_frames=sync_pulse_frames,
            rng=random.Random(0),
        )
        if sync_pin < 0:
            raise ValueError("sync_pin must be a non-negative BCM GPIO number")

    resolved_video_files = _resolve_video_files(video_files)

    video_streams = {}
    maximum_frame_bytes = 0
    for video_path in dict.fromkeys(resolved_video_files):
        stream = utils.probe_video_stream(video_path, ffprobe_bin=ffprobe_bin)
        if not stream:
            raise RuntimeError(f"Could not probe video stream: {video_path}")
        validate_hevc_stream(
            video_path,
            stream,
            require_pi5_compatible=bool(raspi),
        )
        source_duration_s = video_duration_seconds(stream)
        if source_duration_s <= 0.0:
            raise ValueError(f"Video duration is missing or invalid: {video_path}")
        if source_duration_s + 1e-9 < clip_duration_seconds:
            raise ValueError(
                f"Video is shorter than clip_duration_seconds: {video_path} "
                f"({source_duration_s:.6f}s < {clip_duration_seconds:.6f}s)"
            )
        video_streams[video_path] = stream
        maximum_frame_bytes = max(
            maximum_frame_bytes,
            int(stream["width"]) * int(stream["height"]) * 4,
        )

    hevc_decoder_device = None
    if raspi:
        hevc_decoder_device = find_pi_hevc_decoder_device()
        if hevc_decoder_device is None:
            raise RuntimeError(
                "Raspberry Pi HEVC hardware decoder is unavailable or inaccessible. "
                "Expected an HEVC/rpivid V4L2 device (normally /dev/video19); "
                "check Raspberry Pi OS drivers and membership in the video group."
            )

    main_screen, experimenter_screen = resolve_task_screens(screen_config)
    win = utils.setup_window(bg_rgb_255=bg, fullscreen=fullscreen, size=win_size, screen_info=main_screen)
    bg_rect = utils.make_bg_rect(win, bg)
    mouse = event.Mouse(win=win)
    experimenter_preview = None
    frame_publisher = None
    sync_lgpio = None
    sync_gpio_chip = None
    reusable_movie = None
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
        behavior_fieldnames=[
            "trial_num",
            "source_video_path",
            "source_video_name",
            "source_duration_seconds",
            "source_clip_start_seconds",
            "source_clip_end_seconds",
            "requested_clip_duration_seconds",
            "actual_source_start_seconds",
            "actual_source_last_frame_seconds",
            "first_frame_time_since_session_start",
            "last_frame_end_time_since_session_start",
            "displayed_duration_seconds",
            "display_frames",
            "aborted",
            "stop_reason",
            "dropped_frames",
            "sync_pulses",
        ],
        auto_flush=False,
    )
    logger = session_logs.event_logger
    msg_logger = session_logs.message_logger
    behavior_logger = session_logs.behavior_logger
    if behavior_logger is None:
        raise RuntimeError("play_video requires a behavior logger")
    pylogging.console.setLevel(pylogging.CRITICAL)
    try:
        if experimenter_screen is not None:
            frame_publisher = SharedVideoFrameBuffer(maximum_frame_bytes)
            experimenter_preview = ExperimenterPreview(
                experimenter_screen,
                task_label=resolved_config_name,
                start_perf_s=time.perf_counter(),
                update_interval_s=0.1,
            )
            experimenter_preview.clear_scene(bg_rgb_255=bg, main_size=main_scene_size)

        if raspi:
            import lgpio

            sync_lgpio = lgpio
            sync_gpio_chip = lgpio.gpiochip_open(0)
            if isinstance(sync_gpio_chip, int) and sync_gpio_chip < 0:
                raise RuntimeError(
                    f"Could not open GPIO chip 0; lgpio={sync_gpio_chip}"
                )
            claim_result = lgpio.gpio_claim_output(sync_gpio_chip, sync_pin)
            if isinstance(claim_result, int) and claim_result < 0:
                raise RuntimeError(
                    f"Could not claim sync GPIO {sync_pin}; lgpio={claim_result}"
                )
            write_result = lgpio.gpio_write(sync_gpio_chip, sync_pin, 0)
            if isinstance(write_result, int) and write_result < 0:
                raise RuntimeError(
                    f"Could not initialize sync GPIO {sync_pin} low; lgpio={write_result}"
                )

        msg_logger.log(
            "INFO",
            f"session_start task=play_video config_name={resolved_config_name} session_dir={session_logs.session_dir}",
        )
        msg_logger.log(
            "INFO",
            f"resolved_screens main={describe_screen(main_screen)} experimenter={describe_screen(experimenter_screen)}",
        )
        msg_logger.log(
            "INFO",
            (
                f"video_requirements codec=hevc profile=Main pix_fmt=yuv420p "
                f"probed_once=1 n_video_paths={len(resolved_video_files)} "
                f"n_unique_videos={len(video_streams)} clip_duration_s={clip_duration_seconds:.6f} "
                f"seek_timeout_s={seek_timeout_seconds:.3f}"
            ),
        )
        if raspi:
            msg_logger.log(
                "INFO",
                (
                    f"pi5_video_hardware decoder_device={hevc_decoder_device} "
                    f"sync_pin={sync_pin} interval_frames={sync_interval_min}-{sync_interval_max} "
                    f"pulse_width_frames={sync_pulse_frames}"
                ),
            )
        if frame_publisher is not None:
            msg_logger.log(
                "INFO",
                (
                    f"experimenter_video_mirror mode=single_decode_latest_frame_wins "
                    f"shared_memory={frame_publisher.name} capacity_bytes={maximum_frame_bytes}"
                    f" slots={frame_publisher.slot_count} publish_interval_s=0"
                ),
            )
        msg_logger.log(
            "INFO",
            f"resolved_main_scene_size size={main_scene_size[0]}x{main_scene_size[1]} fullscreen={int(bool(fullscreen))} requested_win_size={win_size} realized_win_size={tuple(win.size)}",
        )
        if refresh_rate is not None and float(refresh_rate) > 0:
            fps = float(refresh_rate)
            frame_dur = 1.0 / fps
            msg_logger.log("INFO", f"fps_override refresh_rate={fps:.6f}Hz frame_dur_s={frame_dur:.9f}")
        else:
            fps, frame_dur = utils.detect_frame_rate(win, msg_logger=msg_logger)
        msg_logger.log(
            "INFO",
            f"task_ready fps={fps:.6f} n_video_paths={len(resolved_video_files)} clip_duration_s={clip_duration_seconds:.6f}",
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

            chosen_video = selection_rng.choice(resolved_video_files)
            chosen_stream = video_streams[chosen_video]
            selected_clip = select_random_video_clip(
                chosen_stream,
                clip_duration_seconds,
                rng=selection_rng,
            )
            if experimenter_preview is not None and frame_publisher is not None:
                experimenter_preview.play_shared_video(
                    shared_frame_buffer=frame_publisher.descriptor(),
                    minimum_sequence=frame_publisher.sequence + 1,
                    video_size=(int(chosen_stream["width"]), int(chosen_stream["height"])),
                    bg_rgb_255=bg,
                    main_size=main_scene_size,
                )
            sync_schedule = (
                RandomFramePulseSchedule(
                    sync_interval_min,
                    sync_interval_max,
                    pulse_width_frames=sync_pulse_frames,
                    rng=sync_rng,
                )
                if raspi
                else None
            )
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
                stream_info=chosen_stream,
                frame_publisher=frame_publisher,
                sync_schedule=sync_schedule,
                sync_gpio_module=sync_lgpio,
                sync_gpio_chip=sync_gpio_chip,
                sync_pin=sync_pin,
                frame_duration_s=frame_dur,
                frame_publish_interval_s=0.0,
                clip_start_s=selected_clip.start_s,
                clip_duration_s=selected_clip.duration_s,
                movie_stim=reusable_movie,
                keep_movie_loaded=True,
                seek_timeout_s=seek_timeout_seconds,
            )
            reusable_movie = playback_info["movie_stim"]
            played_videos += 1
            first_frame_time = (
                logger.seconds_since_session_start(playback_info["start_flip_perf_s"])
                if playback_info.get("start_flip_perf_s") is not None
                else None
            )
            last_frame_end_time = (
                logger.seconds_since_session_start(playback_info["last_frame_end_perf_s"])
                if playback_info.get("last_frame_end_perf_s") is not None
                else None
            )
            behavior_logger.writerow(
                {
                    "trial_num": played_videos,
                    "source_video_path": str(playback_info["video_path"]),
                    "source_video_name": playback_info["video_name"],
                    "source_duration_seconds": f"{float(playback_info['source_duration_s']):.9f}",
                    "source_clip_start_seconds": f"{float(playback_info['clip_start_s']):.9f}",
                    "source_clip_end_seconds": f"{float(playback_info['clip_end_s']):.9f}",
                    "requested_clip_duration_seconds": f"{float(playback_info['clip_duration_s']):.9f}",
                    "actual_source_start_seconds": (
                        f"{float(playback_info['actual_source_start_s']):.9f}"
                        if playback_info.get("actual_source_start_s") is not None
                        else ""
                    ),
                    "actual_source_last_frame_seconds": (
                        f"{float(playback_info['actual_source_last_frame_s']):.9f}"
                        if playback_info.get("actual_source_last_frame_s") is not None
                        else ""
                    ),
                    "first_frame_time_since_session_start": (
                        f"{float(first_frame_time):.9f}" if first_frame_time is not None else ""
                    ),
                    "last_frame_end_time_since_session_start": (
                        f"{float(last_frame_end_time):.9f}" if last_frame_end_time is not None else ""
                    ),
                    "displayed_duration_seconds": (
                        f"{float(playback_info['displayed_duration_s']):.9f}"
                        if playback_info.get("displayed_duration_s") is not None
                        else ""
                    ),
                    "display_frames": playback_info["frames_presented"],
                    "aborted": int(playback_info["aborted"]),
                    "stop_reason": playback_info.get("abort_reason") or "completed",
                    "dropped_frames": playback_info["dropped_frames"],
                    "sync_pulses": playback_info["sync_pulses"],
                }
            )
            if experimenter_preview is not None:
                experimenter_preview.clear_scene(bg_rgb_255=bg, main_size=main_scene_size)
            session_logs.flush()
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
        return stop_reason
    finally:
        if sync_lgpio is not None and sync_gpio_chip is not None:
            try:
                sync_lgpio.gpio_write(sync_gpio_chip, sync_pin, 0)
            except Exception:
                pass
            try:
                sync_lgpio.gpiochip_close(sync_gpio_chip)
            except Exception:
                pass
        if reusable_movie is not None:
            try:
                reusable_movie.stop(log=False)
            except Exception:
                pass
        if experimenter_preview is not None:
            try:
                experimenter_preview.close()
            except Exception:
                pass
        if frame_publisher is not None:
            try:
                frame_publisher.close()
            except Exception:
                pass
        try:
            session_logs.close()
        except Exception:
            pass
        try:
            win.close()
        except Exception:
            pass


def main():
    args = parse_args()
    cfg = {}
    if args.config:
        cfg = load_config(args.config)
        validate_config(
            cfg,
            required=[
                "config_name",
                "output_dir",
                "video_files",
                "clip_duration_seconds",
            ],
        )

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
        stop_reason = run_task(
            video_files=_get("video_files", []),
            clip_duration_seconds=_get("clip_duration_seconds", None),
            output_dir=_get("output_dir", "./logs"),
            seed=_get("seed", None),
            fullscreen=bool(_get("fullscreen", cfg.get("fullscreen", True))),
            win_size=tuple(_get("win_size", cfg.get("win_size", None))) if _get("win_size", None) else None,
            bg=tuple(_get("bg", cfg.get("bg", (0, 0, 0)))),
            refresh_rate=_get("refresh_rate", cfg.get("refresh_rate", cfg.get("refrech_rate", None))),
            config_name=_get("config_name", cfg.get("config_name", "play_video")),
            ffprobe_bin=_get("ffprobe", cfg.get("ffprobe", "ffprobe")),
            screen_config=screen_config,
            raspi=bool(_get("raspi", is_raspberry_pi())),
            sync_pin=int(_get("sync_pin", cfg.get("sync_pin", 18))),
            sync_interval_frames=tuple(
                int(value)
                for value in _get(
                    "sync_interval_frames",
                    cfg.get("sync_interval_frames", (100, 300)),
                )
            ),
            sync_pulse_frames=int(
                _get("sync_pulse_frames", cfg.get("sync_pulse_frames", 1))
            ),
            seek_timeout_seconds=float(
                _get("seek_timeout_seconds", cfg.get("seek_timeout_seconds", 30.0))
            ),
        )
        if stop_reason != "done":
            sys.exit(USER_EXIT_CODE)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
