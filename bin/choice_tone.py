"""A choice-onset buzzer pulse with a nonblocking, independently timed offset."""

import queue
import threading
import time


def validate_choice_tone(value):
    if not isinstance(value, bool):
        raise ValueError("choice_period_idicator_tone must be true or false")
    return value


class ChoicePeriodTone:
    duration_s = 0.050

    def __init__(self, set_buzzer):
        self._set_buzzer = set_buzzer
        self._lock = threading.Lock()
        self._timer = None
        self._generation = 0
        self._trial_num = None
        self._events = queue.SimpleQueue()
        self._errors = queue.SimpleQueue()

    def _edge(self, active):
        requested_ns = time.perf_counter_ns()
        self._set_buzzer(active)
        actual_ns = time.perf_counter_ns()
        self._events.put(dict(
            trial_num=self._trial_num,
            event="buzzer_on" if active else "buzzer_off",
            requested_timestamp_perf_ns=requested_ns,
            timestamp_perf_ns=actual_ns,
            requested_duration=self.duration_s if active else None,
        ))
        return actual_ns

    def start(self, trial_num):
        """Raise the buzzer in the choice-opening flip callback; never wait."""
        with self._lock:
            self._stop_locked()
            self._generation += 1
            self._trial_num = trial_num
            try:
                onset_ns = self._edge(True)
                delay_s = max(0.0, self.duration_s - (time.perf_counter_ns() - onset_ns) / 1e9)
                self._timer = threading.Timer(delay_s, self._finish, args=(self._generation,))
                self._timer.daemon = True
                self._timer.start()
            except BaseException:
                self._stop_locked()
                raise

    def _finish(self, generation):
        with self._lock:
            if generation != self._generation:
                return
            try:
                self._stop_locked()
            except Exception as error:
                # Report timer-thread failures on the main task thread.
                self._errors.put(error)

    def _stop_locked(self):
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        if self._trial_num is not None:
            self._edge(False)
            self._trial_num = None

    def stop(self):
        """Cancel before a timeout or task exit; stale timers cannot clear it."""
        with self._lock:
            self._generation += 1
            self._stop_locked()

    def flush_events(self, logger):
        """Keep logging outside flip callbacks and off the timer thread."""
        while True:
            try:
                event = self._events.get_nowait()
            except queue.Empty:
                break
            logger.log_signal(**event)
        try:
            error = self._errors.get_nowait()
        except queue.Empty:
            return
        raise RuntimeError(f"Choice-period buzzer failed: {error}") from error
