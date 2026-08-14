import ctypes
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np

from bin.video_playback import (
    RandomFramePulseSchedule,
    SharedVideoFrameBuffer,
    SharedVideoFrameReader,
    center_crop_bounds,
    find_pi_hevc_decoder_device,
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
        actual_start = prepare_vlc_clip(movie, 0.0, 0.1)

        self.assertEqual(movie.seek_calls, [0.0])
        self.assertEqual(actual_start, 0.0)
        self.assertTrue(movie.isPaused)

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


if __name__ == "__main__":
    unittest.main()
