"""Exercise bounded, cancellable clip playback without a decoder or display."""
from contextlib import ExitStack
from pathlib import Path
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from bin import eye_calibration_video as video
from bin.buffered_video import _build_video_filter


def wait_for(predicate):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(0.002)
    raise AssertionError("Video worker did not reach the expected state")


class CalibrationVideoTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(
            video, "resolve_video_files", return_value=[Path("clip.mp4")],
        ))
        self.stack.enter_context(patch.object(video, "probe_video_stream", return_value={
            "codec_name": "hevc", "profile": "Main", "pix_fmt": "yuv420p",
            "width": 8, "height": 4, "duration": 5, "avg_frame_rate": "20/1",
        }))

    def make_player(self):
        player = video.CalibrationVideoClips(
            video_files=["clip.mp4"], clip_duration_seconds=0.1, frame_rate=20,
            seek_timeout_seconds=0.1, seed=1,
        )
        self.addCleanup(player.close)
        return player

    def test_repeats_clips_and_owns_pixels_until_toggled_off(self):
        opened, closed = [], []
        pixels = np.zeros((100, 100, 3), dtype=np.uint8)

        class Decoder:
            def __init__(self, **kwargs):
                opened.append(kwargs)
                self.index = 0

            def __enter__(self):
                return self

            def close(self):
                closed.append(True)

            def wait_until_ready(self, **_kwargs):
                pass

            def next_frame(self, **_kwargs):
                pixels.fill(self.index + 1)
                frame = SimpleNamespace(frame_index=self.index, source_pts_s=self.index / 20, rgb=pixels)
                self.index += 1
                return frame

        self.stack.enter_context(patch.object(video, "BufferedVideoFrameStream", Decoder))
        player = self.make_player()
        self.assertIsNone(player.frame())
        self.assertTrue(player.toggle())
        first = wait_for(player.frame)
        self.assertFalse(np.shares_memory(first.rgb, pixels))
        wait_for(lambda: len(closed) >= 2)
        self.assertGreaterEqual(len(opened), 3)
        self.assertEqual(opened[0]["output_size"], (100, 100))
        self.assertEqual(opened[0]["crop_bounds"], (2, 0, 6, 4))
        self.assertEqual(opened[0]["frame_count"], 2)
        self.assertFalse(player.toggle())
        self.assertIsNone(player.frame())
        player.close()
        self.assertEqual(len(opened), len(closed))
        self.assertFalse(player._thread.is_alive())

    def test_toggle_off_cancels_buffering_and_quick_reenable_discards_old_generation(self):
        preparing = threading.Event()
        closed = []

        class Decoder:
            def __init__(self, **_kwargs):
                pass

            def __enter__(self):
                return self

            def close(self):
                closed.append(True)

            def wait_until_ready(self, *, abort_checker):
                preparing.set()
                wait_for(abort_checker)
                raise video.VideoPreparationAborted("cancelled")

        self.stack.enter_context(patch.object(video, "BufferedVideoFrameStream", Decoder))
        player = self.make_player()
        player.toggle()
        self.assertTrue(preparing.wait(1))
        player.toggle()
        player.toggle()
        wait_for(lambda: closed)
        self.assertIsNone(player.frame())
        self.assertTrue(player.enabled)
        player.close()
        self.assertIsNone(player.error)

    def test_optional_media_failure_disables_video_and_can_be_retried(self):
        with patch.object(video, "probe_video_stream", return_value={}):
            player = self.make_player()
            player.toggle()
            wait_for(lambda: player.error)
            self.assertFalse(player.enabled)
            self.assertIsNone(player.frame())
            self.assertIn("Could not probe", player.error)

    def test_current_decoder_failure_cancels_pending_seek_and_closes_both(self):
        preparing_next = threading.Event()
        opened, closed = [], []

        class Decoder:
            def __init__(self, **_kwargs):
                self.index = len(opened)
                opened.append(self)

            def wait_until_ready(self, *, abort_checker):
                if self.index:
                    preparing_next.set()
                    wait_for(abort_checker)
                    raise video.VideoPreparationAborted("cancelled")

            def next_frame(self, **_kwargs):
                assert preparing_next.wait(1)
                raise RuntimeError("decoder failed")

            def close(self):
                closed.append(self.index)

        self.stack.enter_context(patch.object(video, "BufferedVideoFrameStream", Decoder))
        player = self.make_player()
        player.toggle()
        wait_for(lambda: player.error)
        player.close()
        self.assertIn("decoder failed", player.error)
        self.assertFalse(player.enabled)
        self.assertEqual(sorted(closed), [0, 1])

    def test_resize_filter_leaves_full_screen_playback_unchanged(self):
        self.assertEqual(_build_video_filter((2, 0, 6, 4)), "crop=4:4:2:0")
        self.assertEqual(_build_video_filter((2, 0, 6, 4), (4, 4)), "crop=4:4:2:0")
        self.assertEqual(
            _build_video_filter((2, 0, 6, 4), (100, 100)),
            "crop=4:4:2:0,scale=100:100",
        )

    def test_invalid_box_and_timing_rejected_before_start(self):
        for kwargs in (
            {"video_size": (0, 100)}, {"video_size": (100.5, 100)},
            {"clip_duration_seconds": 0}, {"frame_rate": float("nan")},
            {"seek_timeout_seconds": -1}, {"video_buffer_megabytes": 0},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                video.CalibrationVideoClips(**kwargs)
