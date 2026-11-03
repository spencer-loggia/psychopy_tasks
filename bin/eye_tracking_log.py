"""Queue each received camera measurement once; write CSV off the UART thread."""

from __future__ import annotations

import csv
import math
from pathlib import Path
import queue
import threading
import time

from .eye_tracking import fraction_to_pixels
from .screen import oriented_size, rotate_centered_point


EYE_DATA_FILENAME = "eye_tracking.csv"
EYE_DATA_FIELDS = (
    "task_time_ns", "camera_time_ns", "x_screen_px", "y_screen_px",
    "pupil_diameter_px", "eye_valid", "screen_ready", "rejection_reason",
    "session_id", "frame_sequence",
)


class EyeDataRecorder:
    """Buffer acquisition snapshots without polling the latest state.

    Screen pixels use native PsychoPy coordinates, centered and right/up
    positive. The camera clock is preserved verbatim and is not synchronized
    with the experiment's task clock.
    """

    def __init__(self, path, *, calibration, queue_size=8192):
        self.path = Path(path)
        self._calibration = calibration
        self._geometry = None
        self._lock = threading.Lock()
        self._queue = queue.Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self._error = None
        self._last_key = None
        self._file = self.path.open("w", newline="", encoding="utf-8", buffering=65536)
        try:
            self._writer = csv.writer(self._file)
            self._writer.writerow(EYE_DATA_FIELDS)
            self._file.flush()
            self._thread = threading.Thread(target=self._run, name="eye-csv", daemon=True)
            self._thread.start()
        except Exception:
            self._file.close()
            raise

    def set_screen_geometry(self, native_size, rotation_degrees=0):
        """Bind the realized task window and its subject-to-window rotation."""
        size = tuple(float(v) for v in native_size)
        if len(size) != 2 or not all(math.isfinite(v) and v > 0 for v in size):
            raise ValueError("Eye recording screen size must contain two positive dimensions")
        if not math.isfinite(rotation_degrees):
            raise ValueError("Eye recording screen rotation must be finite")
        with self._lock:
            self._geometry = (oriented_size(size, rotation_degrees), rotation_degrees)

    def submit(self, state):
        """Enqueue a measurement without waiting for disk or queue capacity.

        Freeze the calibration and geometry in effect at acquisition, rather
        than applying subsequent screen/calibration edits when draining CSV.
        """
        key = (state.session_id, state.frame_sequence, state.camera_time_ns)
        with self._lock:
            if self._stop.is_set() or self._error is not None or not state.camera_time_ns:
                return
            if key == self._last_key:
                return
            self._last_key = key
            fraction = self._calibration.map_position(*state.smoothed_position) if state.valid else None
            try:
                self._queue.put_nowait((state, self._geometry, fraction))
            except queue.Full:
                self._error = RuntimeError("Eye CSV queue overflow; recording is incomplete")

    def _write_sample(self, state, geometry, fraction):
        x = y = pupil = math.nan
        if state.valid:
            pupil = state.pupil_diameter_px
            if geometry is not None:
                subject_size, rotation = geometry
                x, y = rotate_centered_point(fraction_to_pixels(fraction, subject_size), rotation)
        self._writer.writerow((
            state.task_time_ns, state.camera_time_ns,
            f"{x:.6f}", f"{y:.6f}", f"{pupil:.6f}",
            int(state.valid), int(geometry is not None), state.rejection_reason or "",
            state.session_id, state.frame_sequence,
        ))

    def _run(self):
        try:
            with self._file:
                next_flush = time.monotonic() + 1.0
                while not self._stop.is_set() or not self._queue.empty():
                    try:
                        sample = self._queue.get(timeout=0.1)
                    except queue.Empty:
                        pass
                    else:
                        self._write_sample(*sample)
                    now = time.monotonic()
                    if now >= next_flush:
                        self._file.flush()
                        next_flush = now + 1.0
        except Exception as exc:
            self._error = exc

    def check_error(self):
        if self._error is not None:
            raise RuntimeError(f"Eye CSV recording failed: {self._error}") from self._error

    def stop_accepting(self):
        # Serialize against submit so a concurrent callback cannot enqueue after
        # the writer has drained and exited. No serial I/O is performed here.
        with self._lock:
            self._stop.set()

    def close(self):
        self.stop_accepting()
        self._thread.join(timeout=5.0)
        if self._thread.is_alive():
            raise RuntimeError("Eye CSV writer did not stop")
        self.check_error()
