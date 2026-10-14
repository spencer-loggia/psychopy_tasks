"""Interruptible attention clips using the same decoder as play_video.

One coordinator plays a clip while one preparer buffers the next clip. Display
code only reads a bounded latest-frame snapshot; file probing, seeking,
buffering and decoder shutdown never run in the calibration loop.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import math
import random
import threading
import time

import numpy as np

from .buffered_video import (
    BufferedVideoFrameStream, DEFAULT_BUFFER_BYTES, VideoBufferUnderrun,
    VideoPreparationAborted,
)
from .video_playback import (
    center_crop_bounds, probe_video_stream, resolve_video_files,
    select_random_video_clip, validate_video_source,
)


@dataclass(frozen=True)
class CalibrationVideoFrame:
    sequence: int
    clip_number: int
    path: str
    source_frame_index: int
    source_pts_s: float
    rgb: np.ndarray


class CalibrationVideoClips:
    def __init__(
        self, *, video_files=(), video_size=(100, 100), clip_duration_seconds=5.0,
        frame_rate=30.0, seek_timeout_seconds=30.0,
        video_buffer_megabytes=DEFAULT_BUFFER_BYTES / (1024 * 1024),
        ffprobe_bin="ffprobe", seed=None, raspi=False,
    ):
        if len(video_size) != 2 or any(
            isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in video_size
        ):
            raise ValueError("video_size must contain two positive pixel dimensions")
        for name, value in (
            ("clip_duration_seconds", clip_duration_seconds), ("frame_rate", frame_rate),
            ("seek_timeout_seconds", seek_timeout_seconds),
            ("video_buffer_megabytes", video_buffer_megabytes),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        self.video_files = video_files
        self.video_size = tuple(video_size)
        self.clip_duration_seconds = clip_duration_seconds
        self.frame_rate = frame_rate
        self.seek_timeout_seconds = seek_timeout_seconds
        self.buffer_bytes = int(video_buffer_megabytes * 1024 * 1024)
        self.ffprobe_bin = ffprobe_bin
        self.raspi = raspi
        self._rng = random.Random(seed)
        self._condition = threading.Condition()
        self._enabled = self._closed = False
        self._generation = self._sequence = self._clip_number = 0
        self._frame = None
        self._error = None
        self._thread = None

    @property
    def enabled(self):
        with self._condition:
            return self._enabled

    @property
    def error(self):
        with self._condition:
            return self._error

    def toggle(self):
        with self._condition:
            if self._closed:
                raise RuntimeError("Calibration video is closed")
            self._enabled = not self._enabled
            self._generation += 1
            self._frame = self._error = None
            if self._enabled and self._thread is None:
                self._thread = threading.Thread(
                    target=self._run, name="calibration-video", daemon=True,
                )
                self._thread.start()
            self._condition.notify_all()
            return self._enabled

    def frame(self):
        with self._condition:
            frame = self._frame
            if not self._enabled or frame is None:
                return None
            return frame

    def _cancelled(self, generation):
        with self._condition:
            return self._closed or not self._enabled or generation != self._generation

    def _wait(self, seconds, generation):
        with self._condition:
            return self._condition.wait_for(
                lambda: self._cancelled(generation), timeout=max(0.0, seconds),
            )

    def _run(self):
        streams = None
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._enabled or self._closed)
                if self._closed:
                    return
                generation = self._generation
            try:
                if streams is None:
                    # Missing/invalid optional media is reported in the UI;
                    # calibration and UART observations continue to run.
                    streams = {}
                    for path in resolve_video_files(self.video_files):
                        if self._cancelled(generation):
                            break
                        stream = probe_video_stream(
                            path, ffprobe_bin=self.ffprobe_bin,
                            timeout_s=self.seek_timeout_seconds,
                        )
                        if not stream:
                            raise RuntimeError(f"Could not probe video stream: {path}")
                        validate_video_source(
                            path, stream, clip_duration_seconds=self.clip_duration_seconds,
                            frame_rate=self.frame_rate, require_pi5_compatible=self.raspi,
                        )
                        streams[path] = stream
                    if self._cancelled(generation):
                        streams = None
                        continue
                # At most two decoders exist: current and preloaded next.
                # Waiting for the next seek never blocks the display thread.
                with ThreadPoolExecutor(max_workers=1, thread_name_prefix="clip-preload") as pool:
                    pending = pool.submit(self._prepare_clip, streams, generation)
                    try:
                        while not self._cancelled(generation):
                            prepared = pending.result()
                            pending = None
                            try:
                                if self._cancelled(generation):
                                    break
                                pending = pool.submit(self._prepare_clip, streams, generation)
                                self._play_clip(*prepared, generation)
                            finally:
                                prepared[2].close()
                    except Exception as exc:
                        # Cancel a pending seek before waiting for its cleanup.
                        # Otherwise a failed current clip could leave the error
                        # hidden until the next clip's seek timeout elapses.
                        self._fail(exc, generation)
                        raise
                    finally:
                        if pending is not None:
                            pending.result()[2].close()
            except VideoPreparationAborted:
                pass
            except Exception as exc:
                streams = None  # Permit a retry after paths/configuration are repaired.
                self._fail(exc, generation)

    def _fail(self, error, generation):
        with self._condition:
            if not self._cancelled(generation):
                self._error = f"{type(error).__name__}: {error}".splitlines()[0]
                self._enabled = False
                self._frame = None
                self._condition.notify_all()

    def _prepare_clip(self, streams, generation):
        if self._cancelled(generation):
            raise VideoPreparationAborted("video toggled off")
        path = self._rng.choice(list(streams))
        stream = streams[path]
        selection = select_random_video_clip(
            stream, self.clip_duration_seconds, rng=self._rng, frame_rate=self.frame_rate,
        )
        size = (int(stream["width"]), int(stream["height"]))
        decoder = BufferedVideoFrameStream(
            video_path=path, source_size=size,
            crop_bounds=center_crop_bounds(size, self.video_size, alignment=2),
            output_size=self.video_size, clip_start_s=selection.start_s,
            frame_count=selection.frame_count, frame_rate=self.frame_rate,
            seek_timeout_s=self.seek_timeout_seconds, memory_budget_bytes=self.buffer_bytes // 2,
        )
        try:
            decoder.wait_until_ready(abort_checker=lambda: self._abort_reason(generation))
            return path, selection, decoder
        except BaseException:
            decoder.close()
            raise

    def _abort_reason(self, generation):
        return "video toggled off" if self._cancelled(generation) else None

    def _play_clip(self, path, selection, decoder, generation):
        cancelled = lambda: self._abort_reason(generation)
        start_s = time.perf_counter()
        end_s = start_s + selection.duration_s
        self._clip_number += 1
        for index in range(selection.frame_count):
            if self._wait(start_s + index / self.frame_rate - time.perf_counter(), generation):
                return
            while True:
                try:
                    frame = decoder.next_frame(abort_checker=cancelled)
                    break
                except VideoBufferUnderrun:
                    if time.perf_counter() >= end_s or self._wait(0.005, generation):
                        return
            if frame is None:
                return
            # Attention clips follow elapsed time. If the display falls
            # behind, retain only the newest frame, never a growing queue.
            if time.perf_counter() >= start_s + (index + 1) / self.frame_rate:
                continue
            rgb = frame.rgb.copy()  # Own pixels before the decoder recycles its slot.
            with self._condition:
                if self._cancelled(generation):
                    return
                self._sequence += 1
                self._frame = CalibrationVideoFrame(
                    self._sequence, self._clip_number, str(path), frame.frame_index,
                    frame.source_pts_s, rgb,
                )
        self._wait(end_s - time.perf_counter(), generation)

    def close(self):
        with self._condition:
            self._closed = True
            self._enabled = False
            self._frame = None
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=self.seek_timeout_seconds + 5)
            if self._thread.is_alive():
                raise RuntimeError("Calibration video worker did not stop")
