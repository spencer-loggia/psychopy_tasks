"""Best-effort experiment commands, ordered independently of the launcher UI."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import datetime as dt
from pathlib import Path

from bin.eye_tracking import UARTEyeTrackerConfig
from bin.eye_tracker_warnings import EyeTrackerWarnings
from bin.eye_tracker_wire.controller import EyeTrackerController, SerialPacketChannel
from bin.eye_tracker_wire.protocol import StatusFlag


# Preserve START/STOP ordering even if a new experiment is opened immediately.
_COMMANDS = ThreadPoolExecutor(max_workers=1, thread_name_prefix="experiment-uart")


class ExperimentEyeTracker:
    def __init__(self, config, *, subject: str, started_at: dt.datetime, directory: Path):
        self.warnings = EyeTrackerWarnings()
        self.pending_path = directory / ".eye_tracker_start_pending"
        self.subject = subject
        stamp = started_at.astimezone(dt.timezone.utc)
        delta = stamp - dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)
        self.timestamp_ns = (
            (delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds
        ) * 1000
        self.uart = None
        self._finished = False
        self.start_future = None
        self.stop_future = None
        try:
            options = config.get("eye_tracker_experiment", {})
            if not isinstance(options, dict):
                raise ValueError("eye_tracker_experiment must be a JSON object")
            enabled = options.get("enabled", False)
            if not isinstance(enabled, bool):
                raise ValueError("eye_tracker_experiment.enabled must be true or false")
            if not enabled:
                return
            uart = options.get("uart", {})
            if not isinstance(uart, dict):
                raise ValueError("eye_tracker_experiment.uart must be a JSON object")
            self.uart = UARTEyeTrackerConfig(**uart)
            if self.uart.simulate:
                return
            self.pending_path.touch()
            self.start_future = _COMMANDS.submit(self._send, start=True)
        except Exception as exc:
            self.uart = None
            self.warnings.warn("config", f"Experiment synchronization is unavailable: {exc}")

    def finish(self):
        if self._finished:
            return
        self._finished = True
        if self.uart is not None and not self.uart.simulate:
            try:
                self.stop_future = _COMMANDS.submit(self._send, start=False)
            except Exception as exc:
                self.warnings.warn("end", f"Could not queue STOP_EXPERIMENT: {exc}")

    def _send(self, *, start: bool):
        action = "start" if start else "end"
        try:
            with SerialPacketChannel(self.uart.device, baud=self.uart.baud) as channel:
                controller = EyeTrackerController(
                    channel,
                    response_timeout_s=self.uart.response_timeout_s,
                    state_timeout_s=self.uart.state_timeout_s,
                    retries=self.uart.state_retries,
                )
                status = (
                    controller.start_experiment(self.subject, timestamp_ns=self.timestamp_ns)
                    if start else controller.stop_experiment()
                )
                if status.flags & StatusFlag.ERROR:
                    self.warnings.warn(action, f"The eye tracker reported an error while trying to {action} the experiment.")
                elif start and not status.flags & StatusFlag.EXPERIMENT_ACTIVE:
                    self.warnings.warn(action, "The eye tracker did not report an active experiment after START_EXPERIMENT.")
                elif not start and status.flags & (
                    StatusFlag.EXPERIMENT_ACTIVE | StatusFlag.TRACKING
                    | StatusFlag.RECORDING | StatusFlag.CALIBRATING
                ):
                    self.warnings.warn(action, "The eye tracker is still active after STOP_EXPERIMENT; recording may not have been flushed.")
        except Exception as exc:
            self.warnings.warn(action, f"Could not {action} the eye tracker experiment for {self.subject}: {exc}")
        finally:
            if start:
                # Task UART workers wait here, never the task's rendering thread.
                try:
                    self.pending_path.unlink(missing_ok=True)
                except OSError as exc:
                    self.warnings.warn("handoff", f"Could not release the eye tracker startup marker: {exc}")
