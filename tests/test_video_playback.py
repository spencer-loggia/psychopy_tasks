from collections import deque
import importlib.util
import math
from multiprocessing import shared_memory
import queue
import random
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from bin.buffered_video import (
    BufferedVideoFrameStream,
    DEFAULT_BUFFER_BYTES,
    VideoBufferUnderrun,
    _decode_worker,
    copy_ffpyplayer_rgb24,
    plan_video_chunks,
)
from bin.video_playback import (
    RandomFramePulseSchedule,
    SharedVideoFrameBuffer,
    SharedVideoFrameReader,
    center_crop_bounds,
    parse_frame_rate,
    plan_video_refresh_cadence,
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
            frame_rate=8.0,
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
        stream._slot_views = tuple(
            np.ndarray(
                (
                    layout.frames_per_chunk,
                    layout.height,
                    layout.width,
                    3,
                ),
                dtype=np.uint8,
                buffer=stream._shm.buf,
                offset=slot_index * layout.slot_bytes,
            )
            for slot_index in range(layout.slot_count)
        )
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
        self.assertEqual(clip.source_time_origin_s, 0.0)
        self.assertGreaterEqual(clip.start_s, 0.0)
        self.assertLessEqual(clip.end_s, 120.0)
        self.assertAlmostEqual(clip.end_s - clip.start_s, 10.0)
        self.assertAlmostEqual(
            (clip.start_s - clip.source_time_origin_s) * 30.0,
            clip.start_frame,
        )
        self.assertEqual(clip.frame_count, 300)
        self.assertEqual(clip.frame_rate, 30.0)

    def test_random_clip_preserves_nonzero_source_time_origin(self):
        rng = Mock()
        rng.randint.return_value = 3

        clip = select_random_video_clip(
            {
                "duration": "10.0",
                "start_time": "4.25",
                "avg_frame_rate": "2/1",
            },
            2.0,
            rng=rng,
            frame_rate=2.0,
        )

        rng.randint.assert_called_once_with(0, 16)
        self.assertEqual(clip.source_time_origin_s, 4.25)
        self.assertEqual(clip.start_frame, 3)
        self.assertEqual(clip.start_s, 5.75)
        self.assertEqual(clip.end_s, 7.75)
        self.assertEqual(clip.duration_s, 2.0)

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

        self.assertEqual(layout.frames_per_chunk, 3)
        self.assertEqual(layout.slot_count, 3)
        self.assertEqual(layout.slot_bytes, 3 * frame_bytes)
        self.assertEqual(layout.total_bytes, 3 * 3 * frame_bytes)
        self.assertEqual(layout.total_chunks, 34)
        self.assertEqual(layout.preload_chunks, 2)

    def test_chunk_plan_streams_long_clip_even_when_it_fits_memory(self):
        frame_bytes = 4 * 3 * 3
        layout = plan_video_chunks(
            (4, 3),
            frame_count=60,
            frame_rate=30.0,
            memory_budget_bytes=60 * frame_bytes,
        )

        self.assertEqual(layout.frames_per_chunk, 8)
        self.assertEqual(layout.slot_count, 3)
        self.assertEqual(layout.preload_chunks, 2)
        self.assertEqual(layout.total_bytes, 24 * frame_bytes)

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

    def test_decode_worker_seeks_before_consuming_and_discards_stale_pts(self):
        layout = plan_video_chunks(
            (2, 2),
            frame_count=3,
            frame_rate=2.0,
            memory_budget_bytes=3 * 2 * 2 * 3,
        )

        class FakeImage:
            def __init__(self, value):
                self._pixels = bytearray([value] * (2 * 2 * 3))

            @staticmethod
            def get_pixel_format():
                return "rgb24"

            @staticmethod
            def get_size():
                return (2, 2)

            @staticmethod
            def get_linesizes(*, keep_align):
                return (6,)

            def to_memoryview(self, *, keep_align):
                return (
                    types.SimpleNamespace(memview=memoryview(self._pixels)),
                )

        player_calls = []
        decoded_frames = deque(
            [
                ("stale", FakeImage(99), 10.5),
                ("target", FakeImage(10), 10.0),
                ("next", FakeImage(20), 10.5),
                ("last", FakeImage(30), 11.0),
            ]
        )

        class FakeMediaPlayer:
            def __init__(self):
                self.metadata_calls = 0
                self.seek_args = None

            def get_metadata(self):
                player_calls.append("metadata")
                self.metadata_calls += 1
                return {
                    "src_vid_size": (
                        (0, 0) if self.metadata_calls == 1 else (2, 2)
                    )
                }

            def seek(self, *args, **kwargs):
                player_calls.append("seek")
                self.seek_args = (args, kwargs)

            def get_frame(self):
                label, image, pts = decoded_frames.popleft()
                player_calls.append(f"frame:{label}")
                return (image, pts), 0.0

            def close_player(self):
                player_calls.append("close")

        player = FakeMediaPlayer()
        media_player_factory = Mock(return_value=player)
        fake_player_module = types.ModuleType("ffpyplayer.player")
        fake_player_module.MediaPlayer = media_player_factory
        fake_ffpyplayer = types.ModuleType("ffpyplayer")
        fake_ffpyplayer.__path__ = []
        fake_ffpyplayer.player = fake_player_module

        owner = shared_memory.SharedMemory(
            create=True,
            size=layout.total_bytes,
        )
        free_slots = queue.Queue()
        for slot_index in range(layout.slot_count):
            free_slots.put(slot_index)
        ready_chunks = queue.Queue()
        try:
            with (
                patch.dict(
                    sys.modules,
                    {
                        "ffpyplayer": fake_ffpyplayer,
                        "ffpyplayer.player": fake_player_module,
                    },
                ),
                patch("bin.buffered_video.time.sleep") as sleep,
            ):
                _decode_worker(
                    video_path="unused.mp4",
                    shared_memory_name=owner.name,
                    layout=layout,
                    crop_bounds=(0, 0, 2, 2),
                    clip_start_s=10.0,
                    seek_timeout_s=1.0,
                    free_slots=free_slots,
                    ready_chunks=ready_chunks,
                    stop_event=threading.Event(),
                )

            messages = []
            while not ready_chunks.empty():
                messages.append(ready_chunks.get_nowait())

            self.assertEqual(
                player_calls,
                [
                    "metadata",
                    "metadata",
                    "seek",
                    "frame:stale",
                    "frame:target",
                    "frame:next",
                    "frame:last",
                    "close",
                ],
            )
            sleep.assert_called_once_with(0.002)
            self.assertEqual(
                player.seek_args,
                (
                    (10.0,),
                    {
                        "relative": False,
                        "seek_by_bytes": False,
                        "accurate": True,
                    },
                ),
            )
            player_options = media_player_factory.call_args.kwargs["ff_opts"]
            self.assertEqual(
                player_options["vf"],
                ["crop=2:2:0:0"],
            )
            self.assertEqual(
                messages,
                [
                    (
                        "info",
                        {
                            "video_filter": "crop=2:2:0:0",
                            "pixel_format": "rgb24",
                        },
                    ),
                    ("chunk", 0, 0, 1, (10.0,)),
                    ("chunk", 1, 1, 1, (10.5,)),
                    ("chunk", 2, 2, 1, (11.0,)),
                    ("eof", 3),
                ],
            )

            shared_frames = np.ndarray(
                (3, 2, 2, 3),
                dtype=np.uint8,
                buffer=owner.buf,
            )
            self.assertEqual(
                [int(frame[0, 0, 0]) for frame in shared_frames],
                [10, 20, 30],
            )
        finally:
            owner.close()
            owner.unlink()

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

    def test_source_boundary_presenter_recovers_one_adjacent_vbl_miss(self):
        video_utils = self._load_utils_without_psychopy_runtime()

        class FakeClock:
            def __init__(self):
                self.anchor_s = 100.0
                self.now_s = self.anchor_s

            def perf_counter(self):
                return self.now_s

            def sleep(self, duration_s):
                self.now_s += float(duration_s)

        class FakeWindow:
            def __init__(self, clock):
                self.clock = clock
                self.size = (2, 2)
                self.flip_count = 0
                self.refresh_index = 0
                self.recordFrameIntervals = False
                self.refreshThreshold = 0.1
                self.nDroppedFrames = 0
                self._flip_callbacks = []

            def callOnFlip(self, callback, *args, **kwargs):
                self._flip_callbacks.append((callback, args, kwargs))

            def flip(self):
                self.flip_count += 1
                earliest_refresh = math.ceil(
                    (self.clock.now_s - self.clock.anchor_s) * 60.0
                    - 1e-9
                )
                self.refresh_index = max(
                    self.refresh_index + 1,
                    earliest_refresh,
                )
                # Delay source frame 1 by one VBL. The absolute schedule must
                # recover on frame 2 without skipping either source frame.
                if self.flip_count == 4:
                    self.refresh_index += 1
                self.clock.now_s = (
                    self.clock.anchor_s + self.refresh_index / 60.0
                )
                callbacks, self._flip_callbacks = self._flip_callbacks, []
                for callback, args, kwargs in callbacks:
                    callback(*args, **kwargs)
                return self.flip_count

        clock = FakeClock()
        win = FakeWindow(clock)
        frames = [
            types.SimpleNamespace(
                frame_index=frame_index,
                source_pts_s=frame_index / 24.0,
                rgb=np.full((2, 2, 3), frame_index, dtype=np.uint8),
            )
            for frame_index in range(3)
        ]
        frame_stream = types.SimpleNamespace(
            layout=types.SimpleNamespace(
                slot_count=1,
                frames_per_chunk=3,
                preload_chunks=1,
                total_bytes=sum(frame.rgb.nbytes for frame in frames),
            ),
            startup_wait_s=0.0,
            wait_until_ready=Mock(),
            next_frame=Mock(side_effect=frames),
            close=Mock(),
        )
        stimulus = types.SimpleNamespace(current_frame=None)
        drawn_frames = []
        stimulus.draw = Mock(
            side_effect=lambda: drawn_frames.append(stimulus.current_frame)
        )
        uploaded_frames = []

        def record_upload(target_stimulus, rgb):
            frame_index = int(rgb[0, 0, 0])
            self.assertIs(target_stimulus, stimulus)
            target_stimulus.current_frame = frame_index
            uploaded_frames.append(frame_index)

        class FakeGPIO:
            def __init__(self):
                self.writes = []

            def gpio_write(self, chip, pin, level):
                self.writes.append((win.refresh_index, chip, pin, level))
                return 0

        gpio = FakeGPIO()
        sync_schedule = RandomFramePulseSchedule(
            2,
            2,
            pulse_width_frames=1,
            rng=random.Random(0),
        )
        frame_publisher = Mock()
        event_logger = Mock()
        message_logger = Mock()
        onset_callback = Mock()
        background = Mock()
        # Deliberately reproduce a noisy measured rate while the fake panel's
        # actual VBLs remain 60 Hz. Absolute source-time targets must not drift
        # when this estimate is slightly wrong.
        measured_refresh_rate = 60.342108

        with (
            patch.object(
                video_utils,
                "BufferedVideoFrameStream",
                return_value=frame_stream,
            ),
            patch.object(
                video_utils.visual,
                "ImageStim",
                return_value=stimulus,
            ) as image_stim,
            patch.object(
                video_utils,
                "upload_rgb_texture",
                side_effect=record_upload,
            ) as upload_texture,
            patch.object(
                video_utils.time,
                "perf_counter",
                side_effect=clock.perf_counter,
            ),
            patch.object(
                video_utils.time,
                "sleep",
                side_effect=clock.sleep,
            ) as sleep,
        ):
            result = video_utils.play_video_fill_screen(
                win=win,
                video_path="unused.mp4",
                logger=event_logger,
                msg_logger=message_logger,
                bg_rect=background,
                allow_escape=False,
                stream_info={
                    "duration": "1.0",
                    "start_time": "0.0",
                    "width": 2,
                    "height": 2,
                },
                frame_publisher=frame_publisher,
                sync_schedule=sync_schedule,
                sync_gpio_module=gpio,
                sync_gpio_chip="chip",
                sync_pin=18,
                frame_publish_interval_s=0.0,
                clip_start_s=0.0,
                clip_duration_s=0.125,
                requested_clip_duration_s=0.125,
                video_frame_rate=24.0,
                video_frame_count=3,
                display_refresh_rate=measured_refresh_rate,
                native_target_size=(2, 2),
                video_onset_callback=onset_callback,
            )

        self.assertEqual(frame_stream.next_frame.call_count, 3)
        self.assertEqual(upload_texture.call_count, 4)
        self.assertEqual(uploaded_frames, [0, 0, 1, 2])
        self.assertEqual(drawn_frames, [0, 0, 0, 1, 2])
        self.assertEqual(stimulus.draw.call_count, 5)
        self.assertEqual(win.flip_count, 6)
        image_stim.assert_called_once()
        self.assertEqual(sleep.call_count, 3)
        # Opaque aspect-cover video does not redraw the hidden full-screen
        # background; it is drawn only for the final clear/offset flip.
        background.draw.assert_called_once_with()
        frame_stream.close.assert_called_once()

        flip_time = lambda flip_index: clock.anchor_s + flip_index / 60.0
        self.assertEqual(result["start_flip_psychopy_s"], 3)
        self.assertAlmostEqual(result["start_flip_perf_s"], flip_time(3))
        self.assertAlmostEqual(result["last_frame_on_perf_s"], flip_time(8))
        self.assertNotIn("final_repeat_flip_perf_s", result)
        self.assertAlmostEqual(result["clip_offset_perf_s"], flip_time(11))
        self.assertAlmostEqual(result["displayed_duration_s"], 8.0 / 60.0)
        self.assertAlmostEqual(
            result["clear_flip_submitted_perf_s"],
            flip_time(3)
            + 3.0 / 24.0
            - 0.49 / measured_refresh_rate,
        )
        self.assertAlmostEqual(
            result["end_requested_perf_s"],
            flip_time(3)
            + 3.0 / 24.0
            - 0.49 / measured_refresh_rate,
        )
        self.assertAlmostEqual(
            result["requested_end_perf_s"],
            flip_time(3) + 3.0 / 24.0,
        )
        self.assertEqual(result["frames_presented"], 3)
        self.assertEqual(result["source_frame_holds_completed"], 3)
        self.assertEqual(result["display_refreshes_presented"], 8)
        self.assertEqual(result["display_warmup_flips"], 2)
        self.assertEqual(result["dropped_frames"], 0)
        self.assertEqual(result["late_frame_count"], 1)
        onset_callback.assert_called_once_with(flip_time(3))
        warning_messages = [
            call.args[1]
            for call in message_logger.log.call_args_list
            if call.args[0] == "WARN"
        ]
        self.assertTrue(
            any(
                "video_frame_boundary_missed" in message
                and "action=continue_absolute_schedule" in message
                for message in warning_messages
            )
        )

        self.assertEqual(frame_publisher.publish_rgb.call_count, 3)
        publish_calls = frame_publisher.publish_rgb.call_args_list
        self.assertEqual(
            [call.kwargs["source_frame_index"] for call in publish_calls],
            [0, 1, 2],
        )
        self.assertEqual(
            [
                call.kwargs["main_display_flip_perf_s"]
                for call in publish_calls
            ],
            [flip_time(3), flip_time(7), flip_time(8)],
        )

        # The pulse begins on source frame 2's boundary. The unchanged front
        # buffer holds it through the intervening VBLs, then clear forces low.
        self.assertEqual(
            gpio.writes,
            [(8, "chip", 18, 1), (11, "chip", 18, 0)],
        )
        self.assertEqual(result["sync_pulses"], 1)

        end_flip_call = next(
            call
            for call in event_logger.log_frame_flip.call_args_list
            if call.kwargs["event"] == "video_clip_end"
        )
        self.assertAlmostEqual(
            end_flip_call.kwargs["requested_timestamp_perf_s"],
            flip_time(3)
            + 3.0 / 24.0
            - 0.49 / measured_refresh_rate,
        )
        self.assertAlmostEqual(
            end_flip_call.kwargs["timestamp_perf_s"],
            flip_time(11),
        )

    @staticmethod
    def _load_utils_without_psychopy_runtime():
        fake_psychopy = types.ModuleType("psychopy")
        fake_psychopy.visual = types.SimpleNamespace(
            Window=object,
            ImageStim=object,
            Circle=object,
            Rect=object,
            TextStim=object,
        )
        fake_psychopy.event = types.SimpleNamespace(
            Mouse=object,
            clearEvents=Mock(),
            getKeys=Mock(return_value=[]),
        )
        with patch.dict(sys.modules, {"psychopy": fake_psychopy}):
            import bin.utils as video_utils

        return video_utils

    def test_refresh_cadence_is_uniform_for_exact_rate_multiple(self):
        cadence = plan_video_refresh_cadence(4, 30.0, 60.0)

        self.assertEqual(cadence.frame_refresh_counts, (2, 2, 2, 2))
        self.assertEqual(cadence.refresh_boundaries, (0, 2, 4, 6, 8))
        self.assertEqual(cadence.refresh_count_histogram, ((2, 4),))
        self.assertEqual(cadence.total_refreshes, 8)
        self.assertEqual(cadence.final_phase_error_s, 0.0)

    def test_refresh_cadence_absorbs_5994_rate_without_phase_drift(self):
        cadence = plan_video_refresh_cadence(1800, 30.0, 59.94)

        self.assertEqual(cadence.refresh_count_histogram, ((1, 4), (2, 1796)))
        self.assertEqual(cadence.total_refreshes, 3596)
        self.assertEqual(min(cadence.frame_refresh_counts), 1)
        self.assertEqual(max(cadence.frame_refresh_counts), 2)
        self.assertLessEqual(
            cadence.maximum_absolute_phase_error_s,
            (0.5 / 59.94) + 1e-12,
        )
        self.assertLessEqual(
            abs(cadence.final_phase_error_s),
            (0.5 / 59.94) + 1e-12,
        )

    def test_refresh_cadence_uses_balanced_three_two_pulldown(self):
        cadence = plan_video_refresh_cadence(6, 24.0, 60.0)

        self.assertEqual(cadence.frame_refresh_counts, (3, 2, 3, 2, 3, 2))
        self.assertEqual(cadence.total_refreshes, 15)

    def test_refresh_cadence_rejects_any_zero_refresh_source_frame(self):
        with self.assertRaisesRegex(
            ValueError,
            "would receive zero physical refreshes",
        ):
            plan_video_refresh_cadence(30, 30.0, 20.0)


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

    def test_run_task_accepts_noninteger_monitor_cadence_without_drift(self):
        module, fake_utils, messages = self._make_validation_harness(
            source_rate="30/1",
            monitor_rate=59.94,
        )
        fake_utils.play_video_fill_screen.return_value = {
            "video_name": "clip.mp4",
            "frames_presented": 60,
            "aborted": False,
            "abort_reason": "",
            "dropped_frames": 0,
            "sync_pulses": 0,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "clip.mp4"
            video_path.touch()
            stop_reason = module.run_task(
                video_files=[str(video_path)],
                clip_duration_seconds=2.0,
                output_dir=tmpdir,
                num_clips=1,
                frame_rate=30.0,
            )

        self.assertEqual(stop_reason, "done")
        fake_utils.setup_task_window.assert_called_once()
        fake_utils.play_video_fill_screen.assert_called_once()
        self.assertTrue(
            any(
                level == "INFO"
                and message.startswith("video_cadence")
                and "schedule=absolute_source_time_nearest_vbl" in message
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
                self.video_start_calls = []
                self.closed = False
                preview_instances.append(self)

            def clear_scene(self, **kwargs):
                self.clear_calls.append(kwargs)

            def play_shared_video(self, **kwargs):
                self.play_calls.append(kwargs)

            def mark_video_started(self, onset_perf_s):
                self.video_start_calls.append(float(onset_perf_s))

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
        self.assertEqual(playback_kwargs["display_refresh_rate"], 60.0)
        cadence = playback_kwargs["refresh_cadence"]
        self.assertEqual(cadence.frame_refresh_counts, (2,) * 60)
        self.assertEqual(cadence.total_refreshes, 120)
        self.assertNotIn("flip_request_lead_s", playback_kwargs)
        self.assertEqual(
            playback_kwargs["video_buffer_bytes"],
            DEFAULT_BUFFER_BYTES,
        )

        preview = preview_instances[0]
        onset_callback = playback_kwargs["video_onset_callback"]
        self.assertTrue(callable(onset_callback))
        onset_callback(123.456)
        self.assertEqual(preview.video_start_calls, [123.456])
        self.assertEqual(len(preview.play_calls), 2)
        self.assertEqual(preview.play_calls[0]["video_size"], (674, 1080))
        self.assertEqual(
            preview.play_calls[0]["shared_frame_buffer"][
                "maximum_frame_bytes"
            ],
            674 * 1080 * 3,
        )
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
