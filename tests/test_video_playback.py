import ctypes
import importlib.util
import random
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from bin.video_playback import (
    RandomFramePulseSchedule,
    SharedVideoFrameBuffer,
    SharedVideoFrameReader,
    center_crop_bounds,
    find_pi_hevc_decoder_device,
    next_video_frame_slot,
    parse_frame_rate,
    prepare_vlc_clip,
    select_random_video_clip,
    validate_hevc_stream,
    video_duration_seconds,
)


class VideoPlaybackTests(unittest.TestCase):
    def test_parse_frame_rate_accepts_ffprobe_fraction(self):
        self.assertAlmostEqual(parse_frame_rate("30000/1001"), 29.97002997)

    def test_hevc_main_yuv420p_is_pi5_compatible(self):
        validate_hevc_stream(
            "clip.mp4",
            {
                "codec_name": "hevc",
                "profile": "Main",
                "pix_fmt": "yuv420p",
                "width": 720,
                "height": 1280,
                "avg_frame_rate": "30/1",
            },
            require_pi5_compatible=True,
        )

    def test_non_hevc_video_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "required hevc/H.265"):
            validate_hevc_stream(
                "clip.mp4",
                {
                    "codec_name": "h264",
                    "profile": "High",
                    "pix_fmt": "yuv420p",
                    "width": 720,
                    "height": 1280,
                    "avg_frame_rate": "30/1",
                },
            )

    def test_random_clip_is_frame_aligned_and_within_source(self):
        stream = {
            "duration": "120.0",
            "avg_frame_rate": "30/1",
        }
        clip = select_random_video_clip(
            stream,
            10.0,
            rng=random.Random(17),
        )

        self.assertEqual(clip.duration_s, 10.0)
        self.assertGreaterEqual(clip.start_s, 0.0)
        self.assertLessEqual(clip.end_s, 120.0)
        self.assertAlmostEqual(clip.end_s - clip.start_s, 10.0)
        self.assertAlmostEqual(clip.start_s * 30.0, clip.start_frame)
        self.assertEqual(clip.frame_count, 300)
        self.assertEqual(clip.frame_rate, 30.0)

    def test_configured_frame_rate_is_authoritative_for_clip_selection(self):
        clip = select_random_video_clip(
            {"duration": "10.0", "avg_frame_rate": "24000/1001"},
            0.11,
            rng=random.Random(4),
            frame_rate=30.0,
        )

        self.assertEqual(clip.frame_count, 3)
        self.assertAlmostEqual(clip.duration_s, 0.1)
        self.assertAlmostEqual(clip.start_s * 30.0, clip.start_frame)
        self.assertAlmostEqual(clip.requested_duration_s, 0.11)

    def test_absolute_video_schedule_skips_expired_slots_without_lag(self):
        slot, skipped = next_video_frame_slot(
            first_flip_perf_s=100.0,
            next_slot=3,
            now_perf_s=100.2,
            frame_rate=30.0,
            request_lead_s=0.015,
        )

        self.assertEqual(slot, 7)
        self.assertEqual(skipped, 4)

    def test_random_clip_rejects_source_shorter_than_requested_clip(self):
        with self.assertRaisesRegex(ValueError, "exceeds source duration"):
            select_random_video_clip(
                {"duration": "4.0", "avg_frame_rate": "30/1"},
                5.0,
            )

    def test_video_duration_rejects_nonfinite_metadata(self):
        self.assertEqual(video_duration_seconds({"duration": "nan"}), 0.0)

    def test_prepare_vlc_clip_seeks_to_zero_when_reusing_source(self):
        class FakePlayer:
            def is_seekable(self):
                return True

        class FakeMovie:
            def __init__(self):
                self._player = FakePlayer()
                self._frameCounter = 12
                self.isPlaying = False
                self.isPaused = True
                self.isFinished = False
                self.source_time = 42.0
                self.seek_calls = []

            def play(self, log=False):
                self.isPlaying = True
                self.isPaused = False

            def seek(self, timestamp, log=False):
                self.seek_calls.append(timestamp)
                self.source_time = float(timestamp)
                self._frameCounter += 1

            def pause(self, log=False):
                self.isPlaying = False
                self.isPaused = True

            def getFPS(self):
                return 30.0

            def getCurrentFrameTime(self):
                return self.source_time

        movie = FakeMovie()
        callback_states = []
        actual_start = prepare_vlc_clip(
            movie,
            0.0,
            0.1,
            ready_callback=lambda: callback_states.append(movie.isPaused),
        )

        self.assertEqual(movie.seek_calls, [0.0])
        self.assertEqual(actual_start, 0.0)
        self.assertTrue(movie.isPaused)
        self.assertEqual(callback_states, [True])

    def test_random_frame_pulses_have_frame_locked_edges(self):
        schedule = RandomFramePulseSchedule(
            100,
            300,
            pulse_width_frames=1,
            rng=random.Random(7),
        )
        edges = []
        for frame_index in range(700):
            edges.extend(schedule.edges_for_frame(frame_index))

        on_edges = [edge for edge in edges if edge.level == 1]
        off_edges = [edge for edge in edges if edge.level == 0]
        self.assertGreaterEqual(len(on_edges), 2)
        self.assertEqual(len(off_edges), len(on_edges))
        for on_edge, off_edge in zip(on_edges, off_edges):
            self.assertGreaterEqual(on_edge.interval_frames, 100)
            self.assertLessEqual(on_edge.interval_frames, 300)
            self.assertEqual(off_edge.frame_index, on_edge.frame_index + 1)

    def test_shared_frame_buffer_publishes_latest_rgba_without_decoding(self):
        width, height = 4, 3
        source_array = np.arange(width * height * 4, dtype=np.uint8)
        source_type = ctypes.c_ubyte * source_array.size
        source = source_type(*source_array.tolist())
        shared = SharedVideoFrameBuffer(source_array.size)
        reader = SharedVideoFrameReader(shared.name, source_array.size)
        try:
            sequence = shared.publish_rgba(
                source,
                width=width,
                height=height,
                source_frame_index=17,
                source_media_time_s=12.5,
                main_display_flip_perf_s=345.25,
                trial_num=8,
            )
            frame = reader.read_latest(0)

            self.assertEqual(shared.descriptor()["slot_count"], 4)
            self.assertEqual(frame.sequence, sequence)
            self.assertEqual(frame.source_frame_index, 17)
            self.assertEqual(frame.source_media_time_s, 12.5)
            self.assertEqual(frame.main_display_flip_perf_s, 345.25)
            self.assertEqual(frame.trial_num, 8)
            self.assertEqual(frame.rgba.shape, (height, width, 4))
            np.testing.assert_array_equal(frame.rgba.reshape(-1), source_array)
            self.assertIsNone(reader.read_latest(sequence))
        finally:
            reader.close()
            shared.close()

    def test_shared_frame_ring_skips_to_latest_after_slot_rollover(self):
        width, height = 2, 2
        frame_bytes = width * height * 4
        source_type = ctypes.c_ubyte * frame_bytes
        shared = SharedVideoFrameBuffer(frame_bytes)
        reader = SharedVideoFrameReader(
            shared.name,
            frame_bytes,
            slot_count=shared.slot_count,
        )
        try:
            for frame_index in range(1, 8):
                source = source_type(*([frame_index] * frame_bytes))
                shared.publish_rgba(
                    source,
                    width=width,
                    height=height,
                    source_frame_index=frame_index,
                    source_media_time_s=frame_index / 30.0,
                    main_display_flip_perf_s=100.0 + frame_index / 60.0,
                    trial_num=3,
                )

            frame = reader.read_latest(0)

            self.assertEqual(frame.sequence, 7)
            self.assertEqual(frame.source_frame_index, 7)
            self.assertEqual(frame.trial_num, 3)
            np.testing.assert_array_equal(
                frame.rgba,
                np.full((height, width, 4), 7, dtype=np.uint8),
            )
        finally:
            reader.close()
            shared.close()

    def test_center_crop_bounds_match_target_aspect(self):
        self.assertEqual(
            center_crop_bounds((1920, 1080), (1080, 1920)),
            (656, 0, 1264, 1080),
        )
        self.assertEqual(
            center_crop_bounds((1080, 1920), (1920, 1080)),
            (0, 656, 1080, 1264),
        )

    def test_find_pi_hevc_decoder_device_uses_named_v4l2_device(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sys_root = root / "sys"
            dev_root = root / "dev"
            (sys_root / "video19").mkdir(parents=True)
            dev_root.mkdir()
            (sys_root / "video19" / "name").write_text("rpivid-hevc-dec\n")
            (dev_root / "video19").touch()

            self.assertEqual(
                find_pi_hevc_decoder_device(sys_root, dev_root),
                dev_root / "video19",
            )


class PlayVideoTaskRotationTests(unittest.TestCase):
    @staticmethod
    def _load_task_module(fake_utils, fake_screen, fake_psychopy):
        import bin as bin_package

        task_path = Path(__file__).resolve().parents[1] / "task" / "play_video.py"
        spec = importlib.util.spec_from_file_location(
            "_test_play_video_task_rotation",
            task_path,
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load task module from {task_path}")
        module = importlib.util.module_from_spec(spec)
        with (
            patch.dict(
                sys.modules,
                {
                    "psychopy": fake_psychopy,
                    "bin.utils": fake_utils,
                    "bin.screen": fake_screen,
                },
            ),
            patch.object(bin_package, "utils", fake_utils, create=True),
            patch.object(bin_package, "screen", fake_screen, create=True),
        ):
            spec.loader.exec_module(module)
        return module

    def test_run_task_propagates_native_geometry_and_clockwise_rotation(self):
        mouse = types.SimpleNamespace(
            getPressed=Mock(return_value=(True, False, False)),
            clickReset=Mock(),
        )
        fake_event = types.SimpleNamespace(
            Mouse=Mock(return_value=mouse),
            clearEvents=Mock(),
        )
        fake_psychopy = types.ModuleType("psychopy")
        fake_psychopy.event = fake_event
        fake_psychopy.logging = types.SimpleNamespace(
            CRITICAL=50,
            console=types.SimpleNamespace(setLevel=Mock()),
        )

        main_screen = types.SimpleNamespace(
            index=0,
            x=0,
            y=0,
            width=2560,
            height=1600,
            name="MAIN",
            rotation="normal",
        )
        experimenter_screen = types.SimpleNamespace(
            index=1,
            x=2560,
            y=0,
            width=1920,
            height=1080,
            name="PREVIEW",
            rotation="normal",
        )
        resolve_scene_size = Mock(return_value=(2560, 1600))
        software_stimulus_rotation = Mock(return_value=90)
        oriented_size = Mock(return_value=(1600, 2560))

        preview_instances = []

        class FakePreview:
            def __init__(self, *args, **kwargs):
                self.clear_calls = []
                self.play_calls = []
                self.closed = False
                preview_instances.append(self)

            def clear_scene(self, **kwargs):
                self.clear_calls.append(kwargs)

            def play_shared_video(self, **kwargs):
                self.play_calls.append(kwargs)

            def poll(self):
                return False

            def close(self):
                self.closed = True

        fake_screen = types.ModuleType("bin.screen")
        fake_screen.ExperimenterPreview = FakePreview
        fake_screen.describe_screen = lambda screen: screen.name
        fake_screen.load_screen_config = Mock()
        fake_screen.oriented_size = oriented_size
        fake_screen.resolve_scene_size = resolve_scene_size
        fake_screen.software_stimulus_rotation = software_stimulus_rotation

        stream = {
            "codec_name": "hevc",
            "profile": "Main",
            "pix_fmt": "yuv420p",
            "width": 1920,
            "height": 1080,
            "duration": "60.0",
            "avg_frame_rate": "30/1",
        }
        win = types.SimpleNamespace(
            size=(2560, 1600),
            _neuro_tasks_refresh_sync_requested=True,
            close=Mock(),
        )
        movie = types.SimpleNamespace(stop=Mock())
        playback_result = {
            "movie_stim": movie,
            "start_flip_perf_s": 10.0,
            "last_frame_end_perf_s": 12.0,
            "video_path": Path("clip.mp4"),
            "video_name": "clip.mp4",
            "source_duration_s": 60.0,
            "clip_start_s": 1.0,
            "clip_end_s": 3.0,
            "clip_duration_s": 2.0,
            "actual_source_start_s": 1.0,
            "actual_source_last_frame_s": 3.0,
            "displayed_duration_s": 2.0,
            "frames_presented": 120,
            "aborted": False,
            "abort_reason": "",
            "dropped_frames": 0,
            "sync_pulses": 0,
        }
        play_video_fill_screen = Mock(return_value=playback_result)
        fake_utils = types.ModuleType("bin.utils")
        fake_utils.probe_video_stream = Mock(return_value=stream)
        fake_utils.setup_task_window = Mock(
            return_value=(win, main_screen, experimenter_screen)
        )
        fake_utils.make_bg_rect = Mock(return_value=object())
        fake_utils.resolve_frame_rate = Mock(return_value=(60.0, 1.0 / 60.0))
        fake_utils.play_video_fill_screen = play_video_fill_screen

        module = self._load_task_module(
            fake_utils,
            fake_screen,
            fake_psychopy,
        )
        with self.assertRaisesRegex(ValueError, "num_clips must be a positive integer"):
            module.run_task([], 2.0, "unused", num_clips=0)

        class FakeFramePublisher:
            name = "raw-frame-buffer"
            slot_count = 4
            sequence = 0

            def __init__(self, capacity_bytes):
                self.capacity_bytes = capacity_bytes
                self.closed = False

            def descriptor(self):
                return {
                    "name": self.name,
                    "capacity_bytes": self.capacity_bytes,
                    "slot_count": self.slot_count,
                }

            def close(self):
                self.closed = True

        class FakeEventLogger:
            @staticmethod
            def seconds_since_session_start(timestamp):
                return timestamp

        class FakeBehaviorLogger:
            def __init__(self):
                self.rows = []

            def writerow(self, row):
                self.rows.append(row)

        class FakeMessageLogger:
            def __init__(self):
                self.messages = []

            def log(self, level, message):
                self.messages.append((level, message))

        class FakeSessionLogs:
            def __init__(self):
                self.event_logger = FakeEventLogger()
                self.message_logger = FakeMessageLogger()
                self.behavior_logger = FakeBehaviorLogger()
                self.session_dir = Path("test-session")
                self.closed = False

            def flush(self):
                pass

            def close(self):
                self.closed = True

        session_logs = FakeSessionLogs()
        module.SessionLogBundle = Mock(return_value=session_logs)
        module.SharedVideoFrameBuffer = FakeFramePublisher
        module.build_main_and_worker_affinity_plan = Mock(
            return_value={
                "supported": False,
                "reason": "test",
                "current_affinity": None,
                "main_cpu_affinity": None,
                "worker_cpu_affinity": None,
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "clip.mp4"
            video_path.touch()
            stop_reason = module.run_task(
                video_files=[str(video_path)],
                clip_duration_seconds=2.0,
                output_dir=tmpdir,
                num_clips=2,
                screen_config={"main": "MAIN", "experimenter": "PREVIEW"},
            )

        self.assertEqual(stop_reason, "done")
        fake_utils.setup_task_window.assert_called_once_with(
            {"main": "MAIN", "experimenter": "PREVIEW"},
            bg_rgb_255=(0, 0, 0),
            fullscreen=True,
            size=None,
            allow_same_screen=True,
        )
        software_stimulus_rotation.assert_called_once_with("normal")
        oriented_size.assert_called_once_with((2560, 1600), 90)

        self.assertEqual(play_video_fill_screen.call_count, 2)
        playback_kwargs = play_video_fill_screen.call_args.kwargs
        self.assertFalse(playback_kwargs["stop_on_mouse_click"])
        self.assertEqual(playback_kwargs["native_target_size"], (2560, 1600))
        self.assertEqual(playback_kwargs["stimulus_rotation_degrees"], 90)
        self.assertEqual(playback_kwargs["video_frame_rate"], 30.0)
        self.assertEqual(playback_kwargs["video_frame_count"], 60)
        self.assertEqual(playback_kwargs["flip_request_lead_s"], 0.015)

        preview = preview_instances[0]
        self.assertEqual(len(preview.play_calls), 2)
        self.assertEqual(preview.play_calls[0]["video_size"], (1920, 1080))
        self.assertEqual(preview.play_calls[0]["main_size"], (2560, 1600))
        self.assertEqual(preview.play_calls[0]["main_rotation_deg"], 90)
        self.assertEqual(len(preview.clear_calls), 3)
        for clear_call in preview.clear_calls:
            self.assertEqual(clear_call["main_size"], (2560, 1600))
            self.assertEqual(clear_call["main_rotation_deg"], 90)

        messages = [message for _, message in session_logs.message_logger.messages]
        geometry_message = next(
            message for message in messages if message.startswith("resolved_main_scene_size")
        )
        self.assertIn("native_size=2560x1600", geometry_message)
        self.assertIn("subject_size=1600x2560", geometry_message)
        self.assertIn("stimulus_rotation_deg=90", geometry_message)
        self.assertFalse(any("swap_interval" in message for message in messages))

if __name__ == "__main__":
    unittest.main()
