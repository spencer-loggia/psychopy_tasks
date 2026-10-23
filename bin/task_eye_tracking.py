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

from .task_clock import TaskClock
from .eye_tracking import UARTEyeTracker
from .eye_tracking_log import EYE_DATA_FILENAME, EyeDataRecorder
from .eye_tracker_warnings import EyeTrackerWarnings


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

    def __init__(self, settings, *, output_dir=None, task_clock=None, on_state=None):
        self.settings = settings
        self._on_state = on_state
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
        self.warnings = EyeTrackerWarnings()
        self.tracker = UARTEyeTracker(
            uart_config=settings.uart,
            filter_config=settings.filter,
            on_state=self._update_preview,
            on_measurement=self._record_measurement,
            task_clock=task_clock or TaskClock(),
        )
        self.recorder = None
        try:
            if output_dir is not None:
                self.recorder = EyeDataRecorder(
                    Path(output_dir) / EYE_DATA_FILENAME,
                    calibration=settings.calibration,
                )
        except Exception as exc:
            self.warnings.warn("recording", f"Could not start eye data acquisition/CSV recording: {exc}")
        try:
            self.tracker.start()
        except Exception as exc:
            self.warnings.warn("tracking", f"Could not start eye acquisition: {exc}")
        # Normal cleanup goes through SessionLogBundle. Also stop tracking if a
        # legacy task exits before its main loop's finally block is established.
        atexit.register(self.close)

    def _record_measurement(self, state):
        if self.recorder is not None:
            self.recorder.submit(state)

    def _update_preview(self, state):
        with self._preview_lock:
            if self._closed:
                return
            # Fixation history follows acquisition, even with no preview or
            # while the presentation thread is busy decoding/flipping video.
            if self._on_state is not None:
                self._on_state(state)
            if state == self._last_state or self.preview_state is None:
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
            try:
                self.recorder.check_error()
            except Exception as exc:
                self.warnings.warn("recording", str(exc))

    def get_position(self):
        """Return calibrated, unclamped subject-scene fractions or None.

        Safe on the main task thread: this reads a short-lived memory lock,
        never serial I/O. None includes startup, blink/loss, stale data, errors
        and service restarts.
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
            # Stop accepting measurements and drain CSV before a slow STOP ACK.
            try:
                if self.recorder is not None:
                    self.recorder.close()
            except Exception as exc:
                self.warnings.warn("recording", str(exc))
            finally:
                try:
                    self.tracker.stop()
                except Exception as exc:
                    self.warnings.warn("stop", f"Eye tracker cleanup failed: {exc}")
        finally:
            with self._preview_lock:
                self._closed = True
                if self.preview_state is not None:
                    with self.preview_state.get_lock():
                        self.preview_state[3] = 0
            atexit.unregister(self.close)
