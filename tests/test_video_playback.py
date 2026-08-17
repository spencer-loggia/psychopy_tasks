from collections import deque
import importlib.util
import queue
import random
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from bin.buffered_video import (
    BufferedVideoFrameStream,
    DEFAULT_BUFFER_BYTES,
    VideoBufferUnderrun,
    copy_ffpyplayer_rgb24,
    plan_video_chunks,
)
from bin.video_playback import (
    RandomFramePulseSchedule,
    SharedVideoFrameBuffer,
    SharedVideoFrameReader,
    center_crop_bounds,
    parse_frame_rate,
    select_random_video_clip,
    validate_hevc_stream,
    video_duration_seconds,
)


class VideoPlaybackTests(unittest.TestCase):
    @staticmethod
    def _make_buffered_stream_state_machine():
        frame_bytes = 2 * 1 * 3
        layout = plan_video_chunks(
            (2, 1),
            frame_count=8,
            frame_rate=2.0,
            memory_budget_bytes=3 * 2 * frame_bytes,
        )
        stream = object.__new__(BufferedVideoFrameStream)
        stream.layout = layout
        stream._started = True
        stream._closed = False
        stream._pending_chunks = deque()
        stream._current_chunk = None
        stream._current_chunk_offset = 0
        stream._next_frame_index = 0
        stream._producer_eof = False
        stream._ready_chunks = queue.Queue()
        stream._free_slots = queue.Queue(maxsize=layout.slot_count)
        stream._shm = types.SimpleNamespace(buf=bytearray(layout.total_bytes))
        return stream

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

    def test_chunk_plan_buffers_a_small_clip_in_one_slot(self):
        layout = plan_video_chunks(
            (4, 3),
            frame_count=5,
            frame_rate=30.0,
            memory_budget_bytes=5 * 4 * 3 * 3,
        )

        self.assertEqual(layout.frame_bytes, 4 * 3 * 3)
        self.assertEqual(layout.frames_per_chunk, 5)
        self.assertEqual(layout.slot_count, 1)
        self.assertEqual(layout.slot_bytes, 5 * 4 * 3 * 3)
        self.assertEqual(layout.total_bytes, layout.slot_bytes)
        self.assertEqual(layout.total_chunks, 1)
        self.assertEqual(layout.preload_chunks, 1)

    def test_chunk_plan_uses_bounded_three_slot_ring_for_large_clip(self):
        frame_bytes = 4 * 3 * 3
        layout = plan_video_chunks(
            (4, 3),
            frame_count=100,
            frame_rate=10.0,
            memory_budget_bytes=3 * 4 * frame_bytes,
        )

        self.assertEqual(layout.frames_per_chunk, 4)
        self.assertEqual(layout.slot_count, 3)
        self.assertEqual(layout.slot_bytes, 4 * frame_bytes)
        self.assertEqual(layout.total_bytes, 3 * 4 * frame_bytes)
        self.assertEqual(layout.total_chunks, 25)
        self.assertEqual(layout.preload_chunks, 2)

    def test_random_clip_rejects_source_shorter_than_requested_clip(self):
        with self.assertRaisesRegex(ValueError, "exceeds source duration"):
            select_random_video_clip(
                {"duration": "4.0", "avg_frame_rate": "30/1"},
                5.0,
            )

    def test_video_duration_rejects_nonfinite_metadata(self):
        self.assertEqual(video_duration_seconds({"duration": "nan"}), 0.0)

    def test_copy_ffpyplayer_rgb24_removes_row_padding(self):
        width, height = 2, 2
        line_size = 8
        padded_rows = bytearray(
            [
                1,
                2,
                3,
                4,
                5,
                6,
                250,
                251,
                7,
                8,
                9,
                10,
                11,
                12,
                252,
                253,
            ]
        )

        class FakeImage:
            @staticmethod
            def get_pixel_format():
                return "rgb24"

            @staticmethod
            def get_size():
                return width, height

            @staticmethod
            def get_linesizes(*, keep_align):
                self.assertTrue(keep_align)
                return (line_size,)

            @staticmethod
            def to_memoryview(*, keep_align):
                self.assertTrue(keep_align)
                return (types.SimpleNamespace(memview=memoryview(padded_rows)),)

        target = np.empty((height, width, 3), dtype=np.uint8)
        copy_ffpyplayer_rgb24(FakeImage(), target)

        np.testing.assert_array_equal(
            target,
            np.array(
                [
                    [[1, 2, 3], [4, 5, 6]],
                    [[7, 8, 9], [10, 11, 12]],
                ],
                dtype=np.uint8,
            ),
        )

    def test_buffered_stream_crosses_ready_chunk_boundary_without_waiting(self):
        stream = self._make_buffered_stream_state_machine()
        stream._process_message(("chunk", 0, 0, 2, (10.0, 10.5)))
        stream._ready_chunks.put_nowait(
            ("chunk", 1, 2, 2, (11.0, 11.5))
        )
        stream._receive_message = Mock(
            side_effect=AssertionError("presentation must not wait for a chunk")
        )

        shared_slots = np.ndarray(
            (
                stream.layout.slot_count,
                stream.layout.frames_per_chunk,
                stream.layout.height,
                stream.layout.width,
                3,
            ),
            dtype=np.uint8,
            buffer=stream._shm.buf,
        )
        shared_slots[0].fill(10)
        shared_slots[1].fill(20)

        first = stream.next_frame()
        second = stream.next_frame()
        third = stream.next_frame()

        self.assertEqual(
            [first.frame_index, second.frame_index, third.frame_index],
            [0, 1, 2],
        )
        self.assertEqual(third.source_pts_s, 11.0)
        np.testing.assert_array_equal(
            third.rgb,
            np.full((1, 2, 3), 20, dtype=np.uint8),
        )
        self.assertEqual(stream._free_slots.get_nowait(), 0)
        stream._receive_message.assert_not_called()

    def test_buffered_stream_raises_immediate_underrun_at_chunk_boundary(self):
        stream = self._make_buffered_stream_state_machine()
        stream._process_message(("chunk", 0, 0, 2, (10.0, 10.5)))
        stream._receive_message = Mock(
            side_effect=AssertionError("presentation must not wait for a chunk")
        )

        self.assertEqual(stream.next_frame().frame_index, 0)
        self.assertEqual(stream.next_frame().frame_index, 1)
        with self.assertRaisesRegex(
            VideoBufferUnderrun,
            "prepared frame 2 was unavailable",
        ):
            stream.next_frame()

        self.assertEqual(stream._free_slots.get_nowait(), 0)
        stream._receive_message.assert_not_called()

    def test_buffered_stream_rejects_invalid_chunk_slot(self):
        stream = self._make_buffered_stream_state_machine()

        with self.assertRaisesRegex(RuntimeError, "invalid slot 3"):
            stream._process_message(
                ("chunk", stream.layout.slot_count, 0, 2, (10.0, 10.5))
            )

        self.assertFalse(stream._pending_chunks)

    def test_buffered_stream_rejects_inconsistent_chunk_metadata(self):
        stream = self._make_buffered_stream_state_machine()

        with self.assertRaisesRegex(RuntimeError, "metadata is inconsistent"):
            stream._process_message(("chunk", 0, 0, 2, (10.0,)))

        self.assertFalse(stream._pending_chunks)

    def test_buffered_stream_rejects_noncontiguous_next_chunk(self):
        stream = self._make_buffered_stream_state_machine()
        stream._process_message(("chunk", 0, 1, 2, (10.0, 10.5)))

        with self.assertRaisesRegex(
            RuntimeError,
            "non-contiguous prepared chunk starts at 1; expected 0",
        ):
            stream.next_frame()

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

    def test_shared_frame_buffer_publishes_latest_rgb_without_decoding(self):
        width, height = 4, 3
        source = np.arange(width * height * 3, dtype=np.uint8).reshape(
            height,
            width,
            3,
        )
        shared = SharedVideoFrameBuffer(source.nbytes)
        reader = SharedVideoFrameReader(shared.name, source.nbytes)
        try:
            sequence = shared.publish_rgb(
                source,
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
            self.assertEqual(frame.rgb.shape, (height, width, 3))
            np.testing.assert_array_equal(frame.rgb, source)
            self.assertIsNone(reader.read_latest(sequence))
        finally:
            reader.close()
            shared.close()

    def test_shared_frame_ring_skips_to_latest_after_slot_rollover(self):
        width, height = 2, 2
        frame_bytes = width * height * 3
        shared = SharedVideoFrameBuffer(frame_bytes)
        reader = SharedVideoFrameReader(
            shared.name,
            frame_bytes,
            slot_count=shared.slot_count,
        )
        try:
            for frame_index in range(1, 8):
                source = np.full(
                    (height, width, 3),
                    frame_index,
                    dtype=np.uint8,
                )
                shared.publish_rgb(
                    source,
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
                frame.rgb,
                np.full((height, width, 3), 7, dtype=np.uint8),
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


class PlayVideoTaskTests(unittest.TestCase):
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

    def _make_validation_harness(self, *, source_rate, monitor_rate):
        mouse = types.SimpleNamespace(
            getPressed=Mock(return_value=(False, False, False)),
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
        win = types.SimpleNamespace(
            size=(2560, 1600),
            _neuro_tasks_refresh_sync_requested=True,
            close=Mock(),
        )
        fake_screen = types.ModuleType("bin.screen")
        fake_screen.ExperimenterPreview = Mock()
        fake_screen.describe_screen = lambda screen: screen.name if screen else "none"
        fake_screen.load_screen_config = Mock()
        fake_screen.oriented_size = Mock(return_value=(1600, 2560))
        fake_screen.resolve_scene_size = Mock(return_value=(2560, 1600))
        fake_screen.software_stimulus_rotation = Mock(return_value=90)

        stream = {
            "codec_name": "hevc",
            "profile": "Main",
            "pix_fmt": "yuv420p",
            "width": 1920,
            "height": 1080,
            "duration": "60.0",
            "avg_frame_rate": source_rate,
        }
        fake_utils = types.ModuleType("bin.utils")
        fake_utils.probe_video_stream = Mock(return_value=stream)
        fake_utils.setup_task_window = Mock(
            return_value=(win, main_screen, None)
        )
        fake_utils.make_bg_rect = Mock(return_value=object())
        fake_utils.resolve_frame_rate = Mock(
            return_value=(float(monitor_rate), 1.0 / float(monitor_rate))
        )
        fake_utils.play_video_fill_screen = Mock()

        module = self._load_task_module(
            fake_utils,
            fake_screen,
            fake_psychopy,
        )
        messages = []
        session_logs = types.SimpleNamespace(
            event_logger=types.SimpleNamespace(
                seconds_since_session_start=lambda timestamp: timestamp
            ),
            message_logger=types.SimpleNamespace(
                log=lambda level, message: messages.append((level, message))
            ),
            behavior_logger=types.SimpleNamespace(writerow=Mock()),
            session_dir=Path("test-session"),
            flush=Mock(),
            close=Mock(),
        )
        module.SessionLogBundle = Mock(return_value=session_logs)
        module.build_main_and_worker_affinity_plan = Mock(
            return_value={
                "supported": False,
                "reason": "test",
                "current_affinity": None,
                "main_cpu_affinity": None,
                "worker_cpu_affinity": None,
            }
        )
        return module, fake_utils, messages

    def test_run_task_rejects_source_frame_rate_mismatch_before_window(self):
        module, fake_utils, _ = self._make_validation_harness(
            source_rate="24000/1001",
            monitor_rate=60.0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "clip.mp4"
            video_path.touch()
            with self.assertRaisesRegex(
                ValueError,
                "does not match configured frame_rate 30.000000",
            ):
                module.run_task(
                    video_files=[str(video_path)],
                    clip_duration_seconds=2.0,
                    output_dir=tmpdir,
                    num_clips=1,
                    frame_rate=30.0,
                )

        fake_utils.setup_task_window.assert_not_called()
        fake_utils.play_video_fill_screen.assert_not_called()

    def test_run_task_rejects_noninteger_monitor_cadence_before_decode(self):
        module, fake_utils, messages = self._make_validation_harness(
            source_rate="30/1",
            monitor_rate=59.94,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "clip.mp4"
            video_path.touch()
            with self.assertRaisesRegex(
                ValueError,
                "Monitor rate 59.940000 Hz is not an integer-compatible multiple",
            ):
                module.run_task(
                    video_files=[str(video_path)],
                    clip_duration_seconds=2.0,
                    output_dir=tmpdir,
                    num_clips=1,
                    frame_rate=30.0,
                )

        fake_utils.setup_task_window.assert_called_once()
        fake_utils.play_video_fill_screen.assert_not_called()
        self.assertTrue(
            any(
                level == "ERROR"
                and message.startswith("video_monitor_cadence_mismatch")
                for level, message in messages
            )
        )

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
        playback_result = {
            "start_flip_perf_s": 10.0,
            "last_frame_end_perf_s": 12.0,
            "video_path": Path("clip.mp4"),
            "video_name": "clip.mp4",
            "source_duration_s": 60.0,
            "clip_start_s": 1.0,
            "clip_end_s": 3.0,
            "clip_duration_s": 2.0,
            "actual_source_start_s": 1.0,
            "actual_source_last_frame_s": 1.0 + (59.0 / 30.0),
            "displayed_duration_s": 2.0,
            "frames_presented": 60,
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

            def __init__(self, maximum_frame_bytes):
                self.maximum_frame_bytes = maximum_frame_bytes
                self.closed = False

            def descriptor(self):
                return {
                    "name": self.name,
                    "maximum_frame_bytes": self.maximum_frame_bytes,
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
        self.assertEqual(
            playback_kwargs["video_buffer_bytes"],
            DEFAULT_BUFFER_BYTES,
        )

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
