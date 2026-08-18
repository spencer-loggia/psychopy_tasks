"""Bounded, process-isolated ffpyplayer frame preparation.

The decoder owns no display resources. It writes complete RGB24 chunks into a
parent-owned shared-memory pool and sends only small ownership descriptors over
multiprocessing queues. The presentation process reads the shared arrays in
place, so changing chunks does not copy pixels.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import multiprocessing as mp
from multiprocessing import shared_memory
from pathlib import Path
import queue
import time
import traceback
from typing import Any, Callable, Optional, Sequence

import numpy as np


RGB_CHANNELS = 3
DEFAULT_BUFFER_BYTES = 512 * 1024 * 1024
DEFAULT_CHUNK_SLOTS = 3
TARGET_CHUNK_SECONDS = 0.25


@dataclass(frozen=True)
class VideoChunkLayout:
    """Fixed shared-memory layout for one prepared clip."""

    width: int
    height: int
    frame_count: int
    frame_rate: float
    frame_bytes: int
    frames_per_chunk: int
    slot_count: int
    slot_bytes: int
    total_bytes: int
    total_chunks: int
    preload_chunks: int


def plan_video_chunks(
    frame_size: Sequence[int],
    frame_count: int,
    frame_rate: float,
    *,
    memory_budget_bytes: int = DEFAULT_BUFFER_BYTES,
) -> VideoChunkLayout:
    """Plan either a whole-clip buffer or a bounded three-slot chunk ring."""
    width, height = int(frame_size[0]), int(frame_size[1])
    frame_count = int(frame_count)
    frame_rate = float(frame_rate)
    memory_budget_bytes = int(memory_budget_bytes)
    if width <= 0 or height <= 0:
        raise ValueError("prepared video dimensions must be positive")
    if frame_count <= 0:
        raise ValueError("prepared video frame_count must be positive")
    if not math.isfinite(frame_rate) or frame_rate <= 0.0:
        raise ValueError("prepared video frame_rate must be positive and finite")

    frame_bytes = width * height * RGB_CHANNELS
    if memory_budget_bytes < frame_bytes:
        raise ValueError(
            f"video buffer budget {memory_budget_bytes} bytes cannot hold one "
            f"RGB24 frame requiring {frame_bytes} bytes"
        )

    target_frames_per_chunk = max(
        1,
        int(math.ceil(frame_rate * TARGET_CHUNK_SECONDS)),
    )
    whole_clip_bytes = frame_count * frame_bytes
    # Only fully preload clips no longer than one low-latency chunk. Waiting
    # for a longer clip merely because it fits in RAM delays onset and prevents
    # decode/display overlap.
    if (
        frame_count <= target_frames_per_chunk
        and whole_clip_bytes <= memory_budget_bytes
    ):
        frames_per_chunk = frame_count
        slot_count = 1
    else:
        slot_count = DEFAULT_CHUNK_SLOTS
        maximum_frames_per_slot = memory_budget_bytes // (
            slot_count * frame_bytes
        )
        if maximum_frames_per_slot < 1:
            raise ValueError(
                f"video buffer budget {memory_budget_bytes} bytes cannot hold "
                f"{slot_count} RGB24 staging frames of {frame_bytes} bytes each"
            )
        frames_per_chunk = min(
            target_frames_per_chunk,
            int(maximum_frames_per_slot),
        )

    slot_bytes = frames_per_chunk * frame_bytes
    total_chunks = int(math.ceil(frame_count / frames_per_chunk))
    return VideoChunkLayout(
        width=width,
        height=height,
        frame_count=frame_count,
        frame_rate=frame_rate,
        frame_bytes=frame_bytes,
        frames_per_chunk=frames_per_chunk,
        slot_count=slot_count,
        slot_bytes=slot_bytes,
        total_bytes=slot_count * slot_bytes,
        total_chunks=total_chunks,
        preload_chunks=(1 if slot_count == 1 else min(2, total_chunks)),
    )


@dataclass(frozen=True)
class PreparedChunk:
    slot_index: int
    first_frame_index: int
    frame_count: int
    source_pts: tuple[float, ...]


@dataclass(frozen=True)
class PreparedVideoFrame:
    """A view valid until the stream is advanced to the next frame."""

    frame_index: int
    source_pts_s: float
    rgb: np.ndarray


class VideoPreparationAborted(RuntimeError):
    def __init__(self, reason: str):
        self.reason = str(reason or "aborted")
        super().__init__(self.reason)


class VideoBufferUnderrun(RuntimeError):
    """Raised when a prepared chunk is unavailable during presentation."""


def _shared_slot_view(
    shm: shared_memory.SharedMemory,
    layout: VideoChunkLayout,
    slot_index: int,
) -> np.ndarray:
    slot_index = int(slot_index)
    if slot_index < 0 or slot_index >= layout.slot_count:
        raise ValueError(f"invalid video chunk slot {slot_index}")
    return np.ndarray(
        (
            layout.frames_per_chunk,
            layout.height,
            layout.width,
            RGB_CHANNELS,
        ),
        dtype=np.uint8,
        buffer=shm.buf,
        offset=slot_index * layout.slot_bytes,
    )


def copy_ffpyplayer_rgb24(image: Any, target: np.ndarray) -> None:
    """Copy one possibly padded ffpyplayer RGB24 image into a packed array."""
    if target.dtype != np.uint8 or target.ndim != 3 or target.shape[2] != 3:
        raise ValueError("target must be a contiguous HxWx3 uint8 array")
    if not target.flags.c_contiguous:
        raise ValueError("target RGB24 array must be contiguous")
    expected_size = (int(target.shape[1]), int(target.shape[0]))
    if str(image.get_pixel_format()).lower() != "rgb24":
        raise RuntimeError(
            f"ffpyplayer returned {image.get_pixel_format()!r}, expected rgb24"
        )
    if tuple(int(value) for value in image.get_size()) != expected_size:
        raise RuntimeError(
            f"ffpyplayer returned frame size {image.get_size()}, "
            f"expected {expected_size}"
        )

    plane = image.to_memoryview(keep_align=True)[0]
    if plane is None:
        raise RuntimeError("ffpyplayer returned an empty RGB24 plane")
    buffer_view = getattr(plane, "memview", plane)
    line_size = int(image.get_linesizes(keep_align=True)[0])
    row_bytes = expected_size[0] * RGB_CHANNELS
    if line_size < row_bytes:
        raise RuntimeError(
            f"ffpyplayer RGB24 stride {line_size} is shorter than {row_bytes}"
        )
    source = np.frombuffer(buffer_view, dtype=np.uint8)
    required_bytes = line_size * expected_size[1]
    if source.size < required_bytes:
        raise RuntimeError(
            f"ffpyplayer RGB24 plane has {source.size} bytes, "
            f"expected at least {required_bytes}"
        )
    source_rows = source[:required_bytes].reshape(expected_size[1], line_size)
    np.copyto(
        target.reshape(expected_size[1], row_bytes),
        source_rows[:, :row_bytes],
        casting="no",
    )


def _queue_put_until_stopped(output_queue, message, stop_event) -> bool:
    while not stop_event.is_set():
        try:
            output_queue.put(message, timeout=0.1)
            return True
        except queue.Full:
            continue
    return False


def _queue_get_until_stopped(input_queue, stop_event) -> Optional[int]:
    while not stop_event.is_set():
        try:
            return int(input_queue.get(timeout=0.1))
        except queue.Empty:
            continue
    return None


def _build_video_filter(
    crop_bounds: Sequence[int],
) -> str:
    left, top, right, bottom = (int(value) for value in crop_bounds)
    width = right - left
    height = bottom - top
    if min(left, top, width, height) < 0 or width <= 0 or height <= 0:
        raise ValueError(f"invalid video crop bounds {tuple(crop_bounds)}")
    if any(value % 2 for value in (left, top, width, height)):
        raise ValueError(
            f"yuv420p crop bounds must have even offsets and dimensions: "
            f"{tuple(crop_bounds)}"
        )
    return f"crop={width}:{height}:{left}:{top}"


def _decode_worker(
    *,
    video_path: str,
    shared_memory_name: str,
    layout: VideoChunkLayout,
    crop_bounds: tuple[int, int, int, int],
    clip_start_s: float,
    seek_timeout_s: float,
    free_slots,
    ready_chunks,
    stop_event,
) -> None:
    shm = None
    player = None
    slot = None
    try:
        from ffpyplayer.player import MediaPlayer

        shm = shared_memory.SharedMemory(name=shared_memory_name, create=False)
        video_filter = _build_video_filter(crop_bounds)
        ff_opts = {
            "an": True,
            "sn": True,
            "sync": "video",
            "framedrop": False,
            "paused": False,
            "out_fmt": "rgb24",
            "autorotate": False,
            # ffpyplayer accepts either a string or a list here, but some
            # aarch64/Cython builds encode a bare string to ``bytes`` and then
            # iterate it as integers while constructing the C filter list
            # ("expected bytes, int found"). A one-item list is the documented
            # equivalent and keeps each encoded filter as one bytes object.
            "vf": [video_filter],
        }
        player = MediaPlayer(
            video_path,
            ff_opts=ff_opts,
            lib_opts={"threads": "auto"},
            loglevel="warning",
        )

        metadata_deadline = time.monotonic() + float(seek_timeout_s)
        while not stop_event.is_set():
            metadata = player.get_metadata()
            if tuple(metadata.get("src_vid_size") or (0, 0)) != (0, 0):
                break
            if time.monotonic() >= metadata_deadline:
                raise RuntimeError(
                    f"ffpyplayer did not open {Path(video_path).name} within "
                    f"{float(seek_timeout_s):.1f}s"
                )
            # Metadata is updated by ffpyplayer's reader thread. Do not consume
            # a frame merely to open the file, because doing so can leave a
            # pre-seek image queued ahead of the requested clip.
            time.sleep(0.002)
        if stop_event.is_set():
            return

        player.seek(
            float(clip_start_s),
            relative=False,
            seek_by_bytes=False,
            accurate=True,
        )
        frame_period_s = 1.0 / layout.frame_rate
        pts_tolerance_s = max(1e-4, 0.05 * frame_period_s)
        frame_deadline = time.monotonic() + float(seek_timeout_s)
        produced = 0
        info_sent = False

        while produced < layout.frame_count and not stop_event.is_set():
            slot_index = _queue_get_until_stopped(free_slots, stop_event)
            if slot_index is None:
                return
            slot = _shared_slot_view(shm, layout, slot_index)
            first_frame_index = produced
            pts_values: list[float] = []
            chunk_capacity = min(
                layout.frames_per_chunk,
                layout.frame_count - produced,
            )

            while len(pts_values) < chunk_capacity and not stop_event.is_set():
                absolute_frame_index = produced + len(pts_values)
                expected_pts = (
                    float(clip_start_s)
                    + (absolute_frame_index * frame_period_s)
                )
                if absolute_frame_index == 0 and time.monotonic() >= frame_deadline:
                    raise RuntimeError(
                        f"ffpyplayer did not reach requested clip start within "
                        f"{float(seek_timeout_s):.1f}s"
                    )
                decoded, status = player.get_frame()
                if status == "eof":
                    raise RuntimeError(
                        f"ffpyplayer reached EOF after {produced + len(pts_values)} "
                        f"of {layout.frame_count} requested frames"
                    )
                if decoded is None:
                    if time.monotonic() >= frame_deadline:
                        raise RuntimeError(
                            f"ffpyplayer produced no frame for "
                            f"{float(seek_timeout_s):.1f}s"
                        )
                    time.sleep(0.001)
                    continue

                image, raw_pts = decoded
                pts = float(raw_pts)
                if not math.isfinite(pts):
                    raise RuntimeError("ffpyplayer returned a non-finite frame PTS")
                if absolute_frame_index == 0:
                    # Accurate seek decodes forward from an earlier seek point,
                    # and some builds can briefly expose a frame queued before
                    # the seek completed. Ignore every non-target PTS until the
                    # exact requested first frame arrives or the seek deadline
                    # expires. Later frames remain strictly contiguous.
                    if abs(pts - expected_pts) > pts_tolerance_s:
                        continue
                if abs(pts - expected_pts) > pts_tolerance_s:
                    raise RuntimeError(
                        f"ffpyplayer frame {absolute_frame_index} PTS {pts:.6f}s "
                        f"does not match expected {expected_pts:.6f}s"
                    )

                copy_ffpyplayer_rgb24(image, slot[len(pts_values)])
                if not info_sent:
                    if not _queue_put_until_stopped(
                        ready_chunks,
                        (
                            "info",
                            {
                                "video_filter": video_filter,
                                "pixel_format": "rgb24",
                            },
                        ),
                        stop_event,
                    ):
                        return
                    info_sent = True
                pts_values.append(pts)
                frame_deadline = time.monotonic() + float(seek_timeout_s)

            if stop_event.is_set():
                return
            produced += len(pts_values)
            message = (
                "chunk",
                int(slot_index),
                int(first_frame_index),
                len(pts_values),
                tuple(pts_values),
            )
            if not _queue_put_until_stopped(ready_chunks, message, stop_event):
                return
            slot = None

        if not stop_event.is_set():
            _queue_put_until_stopped(
                ready_chunks,
                ("eof", int(produced)),
                stop_event,
            )
    except BaseException as exc:
        _queue_put_until_stopped(
            ready_chunks,
            (
                "error",
                f"{type(exc).__name__}: {exc}",
                traceback.format_exc(),
            ),
            stop_event,
        )
    finally:
        if player is not None:
            try:
                player.close_player()
            except Exception:
                pass
        if shm is not None:
            try:
                slot = None
                shm.close()
            except Exception:
                pass


class BufferedVideoFrameStream:
    """Prepare and consume one clip through a bounded shared-memory ring."""

    def __init__(
        self,
        *,
        video_path: str | Path,
        source_size: Sequence[int],
        crop_bounds: Sequence[int],
        clip_start_s: float,
        frame_count: int,
        frame_rate: float,
        seek_timeout_s: float,
        memory_budget_bytes: int = DEFAULT_BUFFER_BYTES,
        context_name: str = "spawn",
    ):
        self.video_path = Path(video_path)
        self.source_size = (int(source_size[0]), int(source_size[1]))
        self.crop_bounds = tuple(int(value) for value in crop_bounds)
        left, top, right, bottom = self.crop_bounds
        if (
            left < 0
            or top < 0
            or right > self.source_size[0]
            or bottom > self.source_size[1]
            or right <= left
            or bottom <= top
        ):
            raise ValueError(
                f"crop bounds {self.crop_bounds} exceed source {self.source_size}"
            )
        self.clip_start_s = float(clip_start_s)
        self.seek_timeout_s = float(seek_timeout_s)
        if not math.isfinite(self.clip_start_s) or self.clip_start_s < 0.0:
            raise ValueError("clip_start_s must be finite and non-negative")
        if not math.isfinite(self.seek_timeout_s) or self.seek_timeout_s <= 0.0:
            raise ValueError("seek_timeout_s must be positive and finite")

        self.layout = plan_video_chunks(
            (right - left, bottom - top),
            frame_count,
            frame_rate,
            memory_budget_bytes=memory_budget_bytes,
        )
        self._context = mp.get_context(context_name)
        shm = None
        free_slots = None
        ready_chunks = None
        try:
            shm = shared_memory.SharedMemory(
                create=True,
                size=self.layout.total_bytes,
            )
            free_slots = self._context.Queue(maxsize=self.layout.slot_count)
            ready_chunks = self._context.Queue(
                maxsize=self.layout.slot_count + 4
            )
            stop_event = self._context.Event()
            for slot_index in range(self.layout.slot_count):
                free_slots.put(slot_index)
            process = self._context.Process(
                target=_decode_worker,
                kwargs={
                    "video_path": str(self.video_path),
                    "shared_memory_name": shm.name,
                    "layout": self.layout,
                    "crop_bounds": self.crop_bounds,
                    "clip_start_s": self.clip_start_s,
                    "seek_timeout_s": self.seek_timeout_s,
                    "free_slots": free_slots,
                    "ready_chunks": ready_chunks,
                    "stop_event": stop_event,
                },
                name="ffpyplayer-video-preparer",
            )
        except BaseException:
            for managed_queue in (free_slots, ready_chunks):
                if managed_queue is None:
                    continue
                try:
                    managed_queue.cancel_join_thread()
                except Exception:
                    pass
                try:
                    managed_queue.close()
                except Exception:
                    pass
            if shm is not None:
                try:
                    shm.close()
                finally:
                    try:
                        shm.unlink()
                    except FileNotFoundError:
                        pass
            raise

        self._shm = shm
        self._slot_views = tuple(
            _shared_slot_view(shm, self.layout, slot_index)
            for slot_index in range(self.layout.slot_count)
        )
        self._free_slots = free_slots
        self._ready_chunks = ready_chunks
        self._stop_event = stop_event
        self._process = process
        self._started = False
        self._closed = False
        self._pending_chunks: deque[PreparedChunk] = deque()
        self._current_chunk: Optional[PreparedChunk] = None
        self._current_chunk_offset = 0
        self._next_frame_index = 0
        self._producer_eof = False
        self.info: dict[str, Any] = {}
        self.startup_wait_s = 0.0

    @property
    def frame_size(self) -> tuple[int, int]:
        return self.layout.width, self.layout.height

    @property
    def shared_memory_name(self) -> str:
        return self._shm.name

    def start(self) -> None:
        if self._closed:
            raise RuntimeError("buffered video stream is closed")
        if self._started:
            return
        try:
            self._process.start()
        except BaseException:
            self.close()
            raise
        self._started = True

    @staticmethod
    def _abort_reason(abort_checker: Optional[Callable[[], Any]]) -> str:
        if abort_checker is None:
            return ""
        try:
            result = abort_checker()
        except Exception:
            return ""
        if not result:
            return ""
        return str(result) if isinstance(result, str) else "external_abort"

    def _process_message(self, message) -> None:
        message_type = str(message[0])
        if message_type == "info":
            self.info.update(dict(message[1]))
            return
        if message_type == "chunk":
            chunk = PreparedChunk(
                slot_index=int(message[1]),
                first_frame_index=int(message[2]),
                frame_count=int(message[3]),
                source_pts=tuple(float(value) for value in message[4]),
            )
            if not 0 <= chunk.slot_index < self.layout.slot_count:
                raise RuntimeError(
                    f"prepared video chunk used invalid slot {chunk.slot_index}"
                )
            if not 1 <= chunk.frame_count <= self.layout.frames_per_chunk:
                raise RuntimeError(
                    f"prepared video chunk has invalid length {chunk.frame_count}"
                )
            if chunk.frame_count != len(chunk.source_pts):
                raise RuntimeError("prepared video chunk metadata is inconsistent")
            if (
                chunk.first_frame_index < 0
                or chunk.first_frame_index + chunk.frame_count
                > self.layout.frame_count
            ):
                raise RuntimeError("prepared video chunk exceeds the requested clip")
            self._pending_chunks.append(chunk)
            return
        if message_type == "eof":
            self._producer_eof = True
            if int(message[1]) != self.layout.frame_count:
                raise RuntimeError(
                    f"ffpyplayer prepared {int(message[1])} frames, "
                    f"expected {self.layout.frame_count}"
                )
            return
        if message_type == "error":
            raise RuntimeError(
                f"ffpyplayer preparation failed: {message[1]}\n{message[2]}"
            )
        raise RuntimeError(f"unknown ffpyplayer preparation message {message_type!r}")

    def _receive_message(
        self,
        *,
        deadline_s: float,
        abort_checker: Optional[Callable[[], Any]],
    ) -> None:
        while True:
            reason = self._abort_reason(abort_checker)
            if reason:
                raise VideoPreparationAborted(reason)
            remaining = deadline_s - time.monotonic()
            if remaining <= 0.0:
                raise RuntimeError(
                    f"ffpyplayer chunk preparation timed out after "
                    f"{self.seek_timeout_s:.1f}s"
                )
            try:
                message = self._ready_chunks.get(timeout=min(0.02, remaining))
            except queue.Empty:
                if self._started and not self._process.is_alive():
                    try:
                        message = self._ready_chunks.get_nowait()
                    except queue.Empty:
                        raise RuntimeError(
                            "ffpyplayer preparation process exited without "
                            "completing the requested clip"
                        )
                else:
                    continue

            self._process_message(message)
            return

    def wait_until_ready(
        self,
        *,
        abort_checker: Optional[Callable[[], Any]] = None,
    ) -> None:
        self.start()
        started_waiting = time.monotonic()
        deadline = started_waiting + self.seek_timeout_s
        while len(self._pending_chunks) < self.layout.preload_chunks:
            self._receive_message(
                deadline_s=deadline,
                abort_checker=abort_checker,
            )
            if self._producer_eof and (
                len(self._pending_chunks) < self.layout.preload_chunks
            ):
                raise RuntimeError("ffpyplayer ended before the preload completed")
        self.startup_wait_s = time.monotonic() - started_waiting

    def _release_current_chunk(self) -> None:
        if self._current_chunk is None:
            return
        try:
            self._free_slots.put_nowait(self._current_chunk.slot_index)
        except queue.Full as exc:
            raise RuntimeError(
                "prepared-video slot ownership was released twice"
            ) from exc
        self._current_chunk = None
        self._current_chunk_offset = 0

    def _acquire_next_chunk(
        self,
        *,
        abort_checker: Optional[Callable[[], Any]],
    ) -> None:
        self._release_current_chunk()
        reason = self._abort_reason(abort_checker)
        if reason:
            raise VideoPreparationAborted(reason)
        while True:
            try:
                message = self._ready_chunks.get_nowait()
            except queue.Empty:
                break
            self._process_message(message)
        if not self._pending_chunks:
            raise VideoBufferUnderrun(
                f"prepared frame {self._next_frame_index} was unavailable; "
                "playback was stopped without skipping it"
            )
        chunk = self._pending_chunks.popleft()
        if chunk.first_frame_index != self._next_frame_index:
            raise RuntimeError(
                f"non-contiguous prepared chunk starts at {chunk.first_frame_index}; "
                f"expected {self._next_frame_index}"
            )
        self._current_chunk = chunk
        self._current_chunk_offset = 0

    def next_frame(
        self,
        *,
        abort_checker: Optional[Callable[[], Any]] = None,
    ) -> Optional[PreparedVideoFrame]:
        if not self._started:
            raise RuntimeError("call wait_until_ready() before consuming frames")
        if self._next_frame_index >= self.layout.frame_count:
            self._release_current_chunk()
            return None
        if (
            self._current_chunk is None
            or self._current_chunk_offset >= self._current_chunk.frame_count
        ):
            self._acquire_next_chunk(abort_checker=abort_checker)
        chunk = self._current_chunk
        if chunk is None:
            raise RuntimeError("prepared video chunk acquisition failed")
        frame_index = self._next_frame_index
        chunk_offset = self._current_chunk_offset
        slot = self._slot_views[chunk.slot_index]
        result = PreparedVideoFrame(
            frame_index=frame_index,
            source_pts_s=chunk.source_pts[chunk_offset],
            rgb=slot[chunk_offset],
        )
        self._current_chunk_offset += 1
        self._next_frame_index += 1
        return result

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop_event.set()
        if self._started:
            self._process.join(timeout=2.0)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=2.0)
        for managed_queue in (self._free_slots, self._ready_chunks):
            try:
                managed_queue.cancel_join_thread()
            except Exception:
                pass
            try:
                managed_queue.close()
            except Exception:
                pass
        # Release cached NumPy exports before closing the SharedMemory mmap.
        self._slot_views = ()
        try:
            self._shm.close()
        finally:
            try:
                self._shm.unlink()
            except FileNotFoundError:
                pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.close()
