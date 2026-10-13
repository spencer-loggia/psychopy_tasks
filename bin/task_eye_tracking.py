"""Shared UART gaze access, continuous CSV recording and experimenter preview.

Serial I/O stays in the tracker worker. Main tasks read its latest immutable
state without waiting for the UART. Preview gaze has its own shared mailbox so
updating it cannot replace a movie or a stimulus-scene command. Recording owns
another worker so serial acquisition and rendering never wait on CSV writes.
"""

from __future__ import annotations

import atexit
import multiprocessing as mp
from pathlib import Path
import threading
import time

from .eye_tracking import UARTEyeTracker
from .eye_tracking_log import EYE_DATA_FILENAME, EyeDataRecorder


def read_preview_gaze(shared, *, now_s=None):
    """Return oriented-scene fractions, or None for invalid/stale/offscreen gaze."""
    if shared is None:
        return None
    with shared.get_lock():
        timestamp, x, y, valid, max_age = shared[:]
    now = time.perf_counter() if now_s is None else now_s
    if (
        not valid
        or now - timestamp > max_age
        or not (-0.5 <= x <= 0.5 and -0.5 <= y <= 0.5)
    ):
        return None
    return (x, y)


class TaskEyeTrackingSession:
    """Own an optional Eye 0 tracker, recorder, and calibrated preview mailbox."""

    def __init__(self, settings, *, output_dir=None, start_perf_s=None):
        self.settings = settings
        self.preview_state = (
            mp.get_context("spawn").Array(
                "d",
                [0, 0, 0, 0, settings.uart.max_sample_age_s],
                lock=True,
            )
            if settings.show_preview
            else None
        )
        self._last_state = None
        self._closed = False
        self._preview_lock = threading.Lock()
        self.tracker = UARTEyeTracker(
            uart_config=settings.uart,
            filter_config=settings.filter,
            on_state=self._update_preview,
        )
        self.recorder = None
        try:
            self.tracker.start()
            if output_dir is not None:
                self.recorder = EyeDataRecorder(
                    Path(output_dir) / EYE_DATA_FILENAME,
                    get_state=self.tracker.get_state,
                    calibration=settings.calibration,
                    start_perf_s=(
                        time.perf_counter() if start_perf_s is None else start_perf_s
                    ),
                    rate_hz=settings.record_rate_hz,
                )
        except Exception:
            self.tracker.stop()
            raise
        # Normal cleanup goes through SessionLogBundle. Also stop tracking if a
        # legacy task exits before its main loop's finally block is established.
        atexit.register(self.close)

    def _update_preview(self, state):
        with self._preview_lock:
            if self._closed or state == self._last_state or self.preview_state is None:
                return
            self._last_state = state
            fraction = (
                self.settings.calibration.map_position(*state.smoothed_position)
                if state.valid
                else None
            )
            with self.preview_state.get_lock():
                self.preview_state[:] = [
                    state.timestamp_perf_s,
                    fraction[0] if fraction else 0,
                    fraction[1] if fraction else 0,
                    int(state.valid),
                    self.settings.uart.max_sample_age_s,
                ]

    def get_state(self):
        """Read raw/smoothed crop pixels, validity, frame ID and sample timestamp."""
        self.check_recording_error()
        return self.tracker.get_state()

    def set_screen_geometry(self, native_size, rotation_degrees=0):
        if self.recorder is not None:
            self.recorder.set_screen_geometry(native_size, rotation_degrees)

    def check_recording_error(self):
        if self.recorder is not None:
            self.recorder.check_error()

    def get_position(self):
        """Return calibrated, unclamped subject-scene fractions or None.

        Safe on the main task thread: this reads a short-lived memory lock,
        never serial I/O. None includes startup, blink/loss, stale data, errors
        and service restarts. Current tasks use gaze only for the preview.
        """
        if self._closed:
            return None
        state = self.get_state()
        return (
            self.settings.calibration.map_position(*state.smoothed_position)
            if state.valid
            else None
        )

    def close(self):
        if self._closed:
            return
        try:
            # End CSV observations before a potentially slow remote STOP ACK.
            try:
                if self.recorder is not None:
                    self.recorder.close()
            finally:
                self.tracker.stop()
        finally:
            with self._preview_lock:
                self._closed = True
                if self.preview_state is not None:
                    with self.preview_state.get_lock():
                        self.preview_state[3] = 0
            atexit.unregister(self.close)
