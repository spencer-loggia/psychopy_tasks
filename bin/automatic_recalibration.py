"""Small offset corrections from accepted dot fixations; no fitting or I/O."""

from collections import deque
from copy import copy
import math


class OffsetRecalibration:
    LOOKBACK_S = 0.050
    ADJUSTMENT_FRACTION = 0.05

    def __init__(self, calibration, *, max_gap_s):
        # Publish a new complete calibration per correction. Readers can keep
        # using the previous snapshot without locks or mixed x/y offsets.
        self.calibration = copy(calibration)
        self.max_gap_s = max_gap_s
        self.update_count = 0
        self.reset_history()

    def reset_history(self):
        self._history = deque(maxlen=256)
        self._last_key = None

    def observe(self, state):
        if not state.valid or not all(math.isfinite(v) for v in (
            state.timestamp_perf_s, *state.smoothed_position,
        )):
            self.reset_history()
            return
        key = (state.session_id, state.frame_sequence, state.camera_time_ns)
        if key == self._last_key:
            return
        if self._history and (
            state.session_id != self._history[-1].session_id
            or not 0 < state.timestamp_perf_s - self._history[-1].timestamp_perf_s <= self.max_gap_s
        ):
            self.reset_history()
        self._last_key = key
        self._history.append(state)
        cutoff = state.timestamp_perf_s - self.LOOKBACK_S - self.max_gap_s
        while len(self._history) > 1 and self._history[1].timestamp_perf_s <= cutoff:
            self._history.popleft()

    def accept(self, target, *, now_s, fixation_start_s):
        """Use the sample holding at acceptance minus 50 ms, if trustworthy.

        The gate calls this once per displayed target while holding its history
        lock. Short fixations, gaps and invalid samples cannot seed a correction.
        """
        reference_s = now_s - self.LOOKBACK_S
        sample = next((state for state in reversed(self._history)
                       if state.timestamp_perf_s <= reference_s + 1e-9), None)
        if sample is None or not (
            fixation_start_s <= sample.timestamp_perf_s
            and -1e-9 <= reference_s - sample.timestamp_perf_s <= self.max_gap_s
            and 0 <= now_s - self._history[-1].timestamp_perf_s <= self.max_gap_s
        ):
            return
        calibration = copy(self.calibration)
        position = calibration.map_position(*sample.smoothed_position)
        offsets = (
            calibration.x_offset + self.ADJUSTMENT_FRACTION * (target[0] - position[0]),
            calibration.y_offset + self.ADJUSTMENT_FRACTION * (target[1] - position[1]),
        )
        if not all(math.isfinite(value) for value in offsets):
            return
        calibration.x_offset, calibration.y_offset = offsets
        self.calibration = calibration
        self.update_count += 1
