"""Continuously sample calibrated gaze to CSV, independently of rendering/UART.

Only this worker writes the file. It reads the UART reader's latest immutable
state without serial I/O and retains repeated camera frames at the recording
cadence. Source frame IDs/times distinguish those repeats from new measurements.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
import threading
import time

from .eye_tracking import fraction_to_pixels
from .screen import oriented_size, rotate_centered_point


EYE_DATA_FILENAME = "eye_tracking.csv"
EYE_DATA_FIELDS = (
    "timestamp_ms", "x_screen_px", "y_screen_px", "pupil_diameter_px",
    "eye_valid", "screen_ready", "rejection_reason", "sample_timestamp_ms",
    "sample_index", "frame_sequence",
)


class EyeDataRecorder:
    """Record at an independent fixed cadence using the event log's time origin.

    Screen pixels are centered native-window coordinates (right/up positive),
    matching task stimulus positions. Pupil diameter stays in camera pixels;
    screen calibration cannot convert a pupil measurement to physical units.
    """

    def __init__(
        self, path, *, get_state, calibration, start_perf_s, rate_hz=60.0,
        clock=time.perf_counter,
    ):
        self.path = Path(path)
        self._get_state = get_state
        self._calibration = calibration
        self._start_perf_s = start_perf_s
        if not math.isfinite(rate_hz) or rate_hz <= 0:
            raise ValueError("eye_tracking.record_rate_hz must be positive and finite")
        self._period = 1.0 / rate_hz
        self._clock = clock
        self._geometry = None
        self._geometry_lock = threading.Lock()
        self._stop = threading.Event()
        self._error = None
        # A fixed file buffer and periodic flush bound memory and preserve
        # completed data during long blocks without per-sample disk flushes.
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
        with self._geometry_lock:
            self._geometry = (oriented_size(size, rotation_degrees), rotation_degrees)

    def _write_sample(self):
        state = self._get_state()
        # Timestamp the observation, never the later buffer flush or a screen
        # flip. The remote sensor clock is not synchronized to this task's clock.
        observed_s = self._clock()
        with self._geometry_lock:
            geometry = self._geometry
        x = y = pupil = math.nan
        if state.valid:
            pupil = state.pupil_diameter_px
            if geometry is not None:
                subject_size, rotation = geometry
                fraction = self._calibration.map_position(*state.smoothed_position)
                x, y = rotate_centered_point(
                    fraction_to_pixels(fraction, subject_size), rotation
                )
        sample_ms = (
            (state.timestamp_perf_s - self._start_perf_s) * 1000
            if state.timestamp_perf_s > 0 else math.nan
        )
        self._writer.writerow((
            f"{(observed_s - self._start_perf_s) * 1000:.6f}",
            f"{x:.6f}", f"{y:.6f}", f"{pupil:.6f}",
            int(state.valid), int(geometry is not None), state.rejection_reason or "",
            f"{sample_ms:.6f}", state.sample_index, state.frame_sequence,
        ))

    def _run(self):
        try:
            with self._file:
                next_sample = self._clock()
                next_flush = next_sample + 1.0
                while not self._stop.is_set():
                    self._write_sample()
                    now = self._clock()
                    if now >= next_flush:
                        self._file.flush()
                        next_flush = now + 1.0
                    next_sample += self._period
                    now = self._clock()
                    if next_sample < now:
                        # Slow scheduling/disk writes leave honest timestamp
                        # gaps. Never backfill invented observations in a burst.
                        missed = math.floor((now - next_sample) / self._period) + 1
                        next_sample += missed * self._period
                    self._stop.wait(max(0.0, next_sample - self._clock()))
        except Exception as exc:
            self._error = exc

    def check_error(self):
        if self._error is not None:
            raise RuntimeError(f"Eye CSV recording failed: {self._error}") from self._error

    def close(self):
        self._stop.set()
        self._thread.join(timeout=5.0)
        if self._thread.is_alive():
            raise RuntimeError("Eye CSV writer did not stop")
        self.check_error()
