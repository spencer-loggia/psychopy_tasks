"""Record the Arducam B0267 aggregate stream with minimal Python overhead.

The camera and encoder stay in ``rpicam-vid``. Python only starts and stops that
process; no frame is copied through Python.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Sequence


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.json")
RAW_H264_SUFFIXES = {".264", ".h264"}


class RecordingError(RuntimeError):
    """Raised when the recorder cannot start or exits unsuccessfully."""


@dataclass(frozen=True)
class RecorderConfig:
    executable: str = "rpicam-vid"
    camera: int = 0
    sensor_mode: str = "5120:720:8"
    output_width: int = 3840
    output_height: int = 540
    framerate: int = 30
    bitrate: int = 8_000_000
    intra_period: int = 60
    denoise: str = "cdn_off"
    low_latency: bool = True
    inline_headers: bool = True
    cpu_affinity: tuple[int, ...] = (1, 2, 3)
    nice: int = 5
    startup_check_seconds: float = 0.35
    extra_args: tuple[str, ...] = ()

    @classmethod
    def load(cls, path: str | Path = DEFAULT_CONFIG_PATH) -> "RecorderConfig":
        config_path = Path(path).expanduser()
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise RecordingError(f"Recorder config does not exist: {config_path}") from exc
        except json.JSONDecodeError as exc:
            raise RecordingError(f"Invalid JSON in recorder config {config_path}: {exc}") from exc

        if not isinstance(raw, dict):
            raise RecordingError(f"Recorder config must be a JSON object: {config_path}")

        known = {field.name for field in fields(cls)}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise RecordingError(
                f"Unknown recorder config option(s): {', '.join(unknown)}"
            )

        values = dict(raw)
        if "cpu_affinity" in values:
            affinity = values["cpu_affinity"]
            if not isinstance(affinity, list):
                raise RecordingError("cpu_affinity must be a JSON list")
            values["cpu_affinity"] = tuple(affinity)
        if "extra_args" in values:
            extra_args = values["extra_args"]
            if not isinstance(extra_args, list):
                raise RecordingError("extra_args must be a JSON list")
            values["extra_args"] = tuple(extra_args)

        try:
            config = cls(**values)
        except TypeError as exc:
            raise RecordingError(f"Invalid recorder config {config_path}: {exc}") from exc
        config.validate()
        return config

    def validate(self) -> None:
        if not isinstance(self.executable, str) or not self.executable.strip():
            raise RecordingError("executable must be a non-empty string")
        if isinstance(self.camera, bool) or not isinstance(self.camera, int) or self.camera < 0:
            raise RecordingError("camera must be a non-negative integer")
        if not isinstance(self.sensor_mode, str) or not self.sensor_mode.strip():
            raise RecordingError("sensor_mode must be a non-empty string")
        for name, value in (
            ("output_width", self.output_width),
            ("output_height", self.output_height),
            ("framerate", self.framerate),
            ("bitrate", self.bitrate),
            ("intra_period", self.intra_period),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise RecordingError(f"{name} must be a positive integer")
        if self.output_width % 2 or self.output_height % 2:
            raise RecordingError("output_width and output_height must both be even")
        if self.denoise not in {"auto", "off", "cdn_off", "cdn_fast", "cdn_hq"}:
            raise RecordingError(f"Unsupported denoise mode: {self.denoise}")
        if not isinstance(self.low_latency, bool):
            raise RecordingError("low_latency must be true or false")
        if not isinstance(self.inline_headers, bool):
            raise RecordingError("inline_headers must be true or false")
        if isinstance(self.nice, bool) or not isinstance(self.nice, int) or not 0 <= self.nice <= 19:
            raise RecordingError("nice must be an integer from 0 through 19")
        if not isinstance(self.startup_check_seconds, (int, float)) or isinstance(
            self.startup_check_seconds, bool
        ):
            raise RecordingError("startup_check_seconds must be a non-negative number")
        if not math.isfinite(float(self.startup_check_seconds)) or self.startup_check_seconds < 0:
            raise RecordingError("startup_check_seconds must be a non-negative number")
        if any(
            isinstance(cpu, bool) or not isinstance(cpu, int) or cpu < 0
            for cpu in self.cpu_affinity
        ):
            raise RecordingError("cpu_affinity entries must be non-negative integers")
        if len(set(self.cpu_affinity)) != len(self.cpu_affinity):
            raise RecordingError("cpu_affinity must not contain duplicate CPU numbers")
        if any(not isinstance(arg, str) or not arg for arg in self.extra_args):
            raise RecordingError("extra_args entries must be non-empty strings")

    def command(
        self,
        destination: str | Path,
        duration_seconds: float | None = None,
    ) -> list[str]:
        """Build the exact command used for recording without starting it."""
        output_path = _validate_destination(destination)
        timeout_ms = _duration_to_milliseconds(duration_seconds)

        command: list[str] = []
        if self.cpu_affinity:
            command.extend(
                ["taskset", "--cpu-list", ",".join(map(str, self.cpu_affinity))]
            )
        if self.nice:
            command.extend(["nice", "-n", str(self.nice)])

        command.extend(
            [
                self.executable,
                "--camera",
                str(self.camera),
                "--timeout",
                str(timeout_ms),
                "--nopreview",
                "--mode",
                self.sensor_mode,
                "--width",
                str(self.output_width),
                "--height",
                str(self.output_height),
                "--framerate",
                str(self.framerate),
                "--codec",
                "h264",
                "--bitrate",
                str(self.bitrate),
                "--intra",
                str(self.intra_period),
                "--denoise",
                self.denoise,
            ]
        )
        if self.low_latency:
            command.append("--low-latency")
        if self.inline_headers:
            command.append("--inline")
        command.extend(self.extra_args)
        command.extend(["--output", str(output_path)])
        return command


def _duration_to_milliseconds(duration_seconds: float | None) -> int:
    if duration_seconds is None:
        return 0
    if isinstance(duration_seconds, bool) or not isinstance(duration_seconds, (int, float)):
        raise RecordingError("duration_seconds must be a positive number or None")
    if not math.isfinite(float(duration_seconds)) or duration_seconds <= 0:
        raise RecordingError("duration_seconds must be a positive number or None")
    return max(1, round(float(duration_seconds) * 1000))


def _estimated_recording_bytes(duration_seconds: float, bitrate: int) -> int:
    """Estimate output size with 20% headroom for bitrate/container variance."""
    return math.ceil(float(duration_seconds) * bitrate / 8 * 1.20)


def _validate_destination(destination: str | Path) -> Path:
    output_path = Path(destination).expanduser().resolve()
    if output_path.suffix.lower() not in RAW_H264_SUFFIXES:
        raise RecordingError(
            "The low-overhead recorder writes a raw H.264 stream; destination must "
            "end in .h264 or .264"
        )
    return output_path


def _require_executable(name: str) -> None:
    if shutil.which(name) is None:
        raise RecordingError(f"Required executable was not found on PATH: {name}")


class Recording:
    """A running ``rpicam-vid`` child process.

    Call :meth:`stop` in a ``finally`` block (or use this object as a context
    manager) so the encoder flushes its final frames.
    """

    def __init__(
        self,
        process: subprocess.Popen[str],
        destination: Path,
        command: Sequence[str],
    ) -> None:
        self.process = process
        self.destination = destination
        self.command = tuple(command)
        self._stderr_lines: deque[str] = deque(maxlen=100)
        self._stop_lock = threading.Lock()
        self._stop_requested = False
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr,
            name="rpicam-vid-stderr",
            daemon=True,
        )
        self._stderr_thread.start()

    def _drain_stderr(self) -> None:
        stream = self.process.stderr
        if stream is None:
            return
        try:
            for line in stream:
                self._stderr_lines.append(line.rstrip())
        finally:
            stream.close()

    @property
    def pid(self) -> int:
        return self.process.pid

    @property
    def is_running(self) -> bool:
        return self.process.poll() is None

    @property
    def stderr_tail(self) -> str:
        return "\n".join(self._stderr_lines)

    def _error(self, prefix: str) -> RecordingError:
        detail = self.stderr_tail.strip()
        if detail:
            return RecordingError(f"{prefix}\nrpicam-vid output:\n{detail}")
        return RecordingError(prefix)

    def _validate_completed(self, allow_interrupted: bool = False) -> Path:
        return_code = self.process.returncode
        acceptable = {0}
        if allow_interrupted:
            acceptable.update({-signal.SIGINT, -signal.SIGTERM})
        if return_code not in acceptable:
            raise self._error(f"Recorder exited with status {return_code}")
        try:
            size = self.destination.stat().st_size
        except FileNotFoundError as exc:
            raise self._error(
                f"Recorder exited without creating {self.destination}"
            ) from exc
        if size == 0:
            raise self._error(f"Recorder created an empty file: {self.destination}")
        return self.destination

    def wait(self, timeout: float | None = None) -> Path:
        """Wait for a duration-limited recording to finish."""
        self.process.wait(timeout=timeout)
        self._stderr_thread.join(timeout=1)
        return self._validate_completed(allow_interrupted=self._stop_requested)

    def stop(self, timeout: float = 15.0) -> Path:
        """Gracefully stop and flush the recording, escalating only if stuck."""
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        with self._stop_lock:
            if self.process.poll() is not None:
                self._stderr_thread.join(timeout=1)
                return self._validate_completed(allow_interrupted=self._stop_requested)

            self._stop_requested = True
            try:
                os.killpg(self.process.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
            try:
                self.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(self.process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(self.process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    self.process.wait()
                    self._stderr_thread.join(timeout=1)
                    raise self._error("Recorder did not stop cleanly and was killed")

            self._stderr_thread.join(timeout=1)
            return self._validate_completed(allow_interrupted=True)

    close = stop

    def __enter__(self) -> "Recording":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        try:
            self.stop()
        except Exception:
            if exc_type is None:
                raise
        return False


def record(
    destination: str | Path,
    duration_seconds: float | None = None,
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    overwrite: bool = False,
) -> Recording:
    """Start recording and return immediately with a stoppable handle.

    ``destination`` must use a ``.h264`` or ``.264`` suffix. With no duration,
    recording continues until :meth:`Recording.stop` is called. Supplying a
    duration makes ``rpicam-vid`` stop itself, after which :meth:`wait` returns
    the completed path.
    """
    config = RecorderConfig.load(config_path)
    output_path = _validate_destination(destination)
    _duration_to_milliseconds(duration_seconds)
    if output_path.exists() and not overwrite:
        raise RecordingError(f"Refusing to overwrite existing recording: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if duration_seconds is not None:
        required_bytes = _estimated_recording_bytes(duration_seconds, config.bitrate)
        free_bytes = shutil.disk_usage(output_path.parent).free
        if output_path.exists() and overwrite:
            free_bytes += output_path.stat().st_size
        if free_bytes < required_bytes:
            required_gib = required_bytes / (1024**3)
            free_gib = free_bytes / (1024**3)
            raise RecordingError(
                f"Insufficient free space for the requested duration: need about "
                f"{required_gib:.1f} GiB including headroom, have {free_gib:.1f} GiB"
            )

    _require_executable(config.executable)
    if config.cpu_affinity:
        _require_executable("taskset")
    if config.nice:
        _require_executable("nice")

    command = config.command(output_path, duration_seconds)
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
    except OSError as exc:
        raise RecordingError(f"Could not start recorder: {exc}") from exc

    recording = Recording(process, output_path, command)
    if config.startup_check_seconds:
        time.sleep(config.startup_check_seconds)
    if process.poll() not in {None, 0}:
        recording._stderr_thread.join(timeout=1)
        raise recording._error(f"Recorder failed during startup (status {process.returncode})")
    return recording


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record the four-camera Arducam aggregate as low-overhead H.264"
    )
    parser.add_argument("destination", type=Path, help="Output .h264 or .264 path")
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Seconds to record; omit to continue until Ctrl-C",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Recorder JSON config path",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command without accessing the camera or destination",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    config = RecorderConfig.load(args.config)
    if args.dry_run:
        print(shlex.join(config.command(args.destination, args.duration)))
        return 0

    recording = record(
        args.destination,
        args.duration,
        config_path=args.config,
        overwrite=args.overwrite,
    )
    print(f"Recording to {recording.destination} (pid {recording.pid})", flush=True)
    try:
        completed = recording.wait()
    except KeyboardInterrupt:
        completed = recording.stop()
    print(f"Saved {completed}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RecordingError as exc:
        print(f"recording error: {exc}", file=sys.stderr)
        raise SystemExit(1)
