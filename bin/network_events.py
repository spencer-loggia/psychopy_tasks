"""Best-effort log streaming; task callbacks only enqueue already-built rows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import queue
import threading
import time
from typing import Any, Mapping
from urllib.parse import urlsplit


@dataclass(frozen=True)
class NetworkEventsConfig:
    endpoint: str = "tcp://*:5556"
    queue_size: int = 8192
    send_high_water_mark: int = 8192
    subject: str | None = None
    experiment_number: int | None = None
    config_file_name: str | None = None

    def __post_init__(self):
        try:
            address = urlsplit(self.endpoint)
            valid = (
                address.scheme == "tcp" and address.hostname is not None
                and address.port is not None and 1 <= address.port <= 65535
                and not (address.path or address.query or address.fragment or address.username)
            )
        except (TypeError, ValueError, AttributeError):
            valid = False
        if not valid:
            raise ValueError("send_network_events bind address must be tcp://ADDRESS:PORT")
        for key in ("queue_size", "send_high_water_mark"):
            value = getattr(self, key)
            if type(value) is not int or value < 1:
                raise ValueError(f"send_network_events.{key} must be a positive integer")
        if self.experiment_number is not None and (
            type(self.experiment_number) is not int or self.experiment_number < 0
        ):
            raise ValueError("experiment_number must be a non-negative integer")


def load_network_event_settings(
    config: Mapping[str, Any], *, config_path: str | Path | None = None,
) -> NetworkEventsConfig | None:
    setting = config.get("send_network_events", False)
    options = setting if isinstance(setting, Mapping) else {}
    enabled = options.get("enabled", False) if isinstance(setting, Mapping) else setting
    if type(enabled) is not bool:
        raise ValueError("send_network_events must be a boolean or an object with a boolean enabled field")
    if not enabled:
        return None
    port = options.get("port", 5556)
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("send_network_events.port must be an integer from 1 to 65535")
    address = options.get("bind_address", "*")
    if not isinstance(address, str) or not address.strip():
        raise ValueError("send_network_events.bind_address must be a non-empty string")
    source = config.get("source_config_file", config_path)
    subject = config.get("subject")
    return NetworkEventsConfig(
        endpoint=f"tcp://{address}:{port}",
        queue_size=options.get("queue_size", 8192),
        send_high_water_mark=options.get("send_high_water_mark", 8192),
        subject=None if subject is None else str(subject).strip(),
        experiment_number=config.get("experiment_number"),
        config_file_name=Path(source).name if source else None,
    )


class NetworkEventSender:
    """One socket-owning worker, bounded memory, and no receiver acknowledgments.

    PUB binds a port for any number of SUB receivers. Every send uses DONTWAIT;
    a slow or absent subscriber never holds up the sender. The application queue
    drops new messages when full; ZeroMQ can silently drop subscriber messages.
    """

    def __init__(self, config: NetworkEventsConfig, metadata: Mapping[str, Any]):
        self.config = config
        self._metadata = dict(metadata)
        self._queue = queue.Queue(maxsize=config.queue_size)
        self._stop = threading.Event()
        self._close_deadline = float("inf")
        self._next_sequence = 0
        self.queue_dropped = 0
        self.send_dropped = 0
        self.sent = 0
        self.error = None
        # Metadata always precedes the block_start log row on this stream.
        self.submit("block_start", self._metadata)
        self._thread = threading.Thread(target=self._run, name="network-events", daemon=True)
        try:
            self._thread.start()
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"

    def submit(self, kind: str, row, fieldnames=None) -> None:
        """Transfer an owned row snapshot; never serialize or touch a socket here."""
        sequence = self._next_sequence
        self._next_sequence += 1
        if self._stop.is_set() or self.error is not None:
            self.queue_dropped += 1
            return
        try:
            self._queue.put_nowait((sequence, kind, row, fieldnames))
        except queue.Full:
            self.queue_dropped += 1

    def _run(self):
        context = socket = None
        try:
            # Disabled sessions do not import PyZMQ or create network resources.
            import zmq

            context = zmq.Context()
            socket = context.socket(zmq.PUB)
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt(zmq.SNDHWM, self.config.send_high_water_mark)
            socket.bind(self.config.endpoint)
            # Give already-running subscribers a chance to reconnect to this
            # block's publisher. This is a fixed worker-only delay, never a
            # receipt/connection check, and never a wait on the task thread.
            self._stop.wait(0.2)
            while time.monotonic() < self._close_deadline:
                try:
                    sequence, kind, row, fieldnames = self._queue.get(timeout=0.02)
                except queue.Empty:
                    if self._stop.is_set():
                        break
                    continue
                message = {
                    "version": 1, "type": kind, "sequence": sequence,
                    "experiment_id": self._metadata.get("experiment_id"),
                    "block_id": self._metadata["block_id"],
                }
                if kind == "block_start":
                    message["metadata"] = row
                else:
                    values = row if fieldnames is None else zip(fieldnames, row)
                    # CSV writes None as an empty cell and stringifies scalars.
                    # Preserve exactly those cell values, including native ns.
                    items = values.items() if isinstance(values, Mapping) else values
                    message["row"] = {key: "" if value is None else str(value) for key, value in items}
                try:
                    socket.send_json(message, flags=zmq.DONTWAIT, separators=(",", ":"))
                    self.sent += 1
                except zmq.Again:
                    self.send_dropped += 1
                # Yield between small batches if a backlog could otherwise keep
                # Python's GIL busy. Event latency is secondary to task timing.
                if (self.sent + self.send_dropped) % 64 == 0:
                    time.sleep(0)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            if socket is not None:
                socket.close(linger=0)
            if context is not None:
                context.term()

    def close(self) -> None:
        """Allow only a bounded local drain after task timing has ended."""
        if self._stop.is_set():
            return
        self._close_deadline = time.monotonic() + 0.25
        self._stop.set()
        if self._thread.ident is not None:
            self._thread.join(timeout=0.3)

    def summary(self) -> str:
        # 'sent' means accepted by ZeroMQ, never confirmed by the receiver.
        return (
            f"network_events sent={self.sent} queue_dropped={self.queue_dropped} "
            f"send_dropped={self.send_dropped} pending={self._queue.qsize()} "
            f"worker_alive={int(self._thread.is_alive())} error={self.error or 'none'}"
        )
