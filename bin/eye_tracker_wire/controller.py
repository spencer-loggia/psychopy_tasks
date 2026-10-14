from __future__ import annotations

import os
import secrets
import select
import termios
import threading
import time
from collections.abc import Callable
from pathlib import Path
from types import TracebackType
from typing import Protocol

from .protocol import (
    TRANSACTION_SIZE,
    CommandCode,
    CommandPacket,
    DecodedStatus,
    ProtocolError,
    StatusCode,
    decode_command,
    decode_status,
    encode_command,
    validate_subject_name,
)
from .uart import configure_raw_uart, open_exclusive_uart

DEFAULT_BAUD = 460_800
DEFAULT_RESPONSE_TIMEOUT_S = 0.25
DEFAULT_STATE_TIMEOUT_S = 10.0
DEFAULT_RETRY_DELAY_S = 0.02
DEFAULT_INTER_BYTE_TIMEOUT_S = 0.10
DEFAULT_POLL_RATE_HZ = 100.0


class ControllerError(RuntimeError):
    """Base class for controller-side transport and command errors."""


class ResponseTimeout(ControllerError):
    """The eye-tracker did not return one complete status packet in time."""


class AcknowledgementTimeout(ControllerError):
    def __init__(self, sequence: int, attempts: int) -> None:
        super().__init__(
            f"No acknowledgement for command sequence {sequence} after {attempts} attempt(s)"
        )
        self.sequence = sequence
        self.attempts = attempts


class CommandRejected(ControllerError):
    def __init__(self, sequence: int, status: StatusCode) -> None:
        super().__init__(f"Command sequence {sequence} was rejected: {status.name}")
        self.sequence = sequence
        self.status = status


class PacketChannel(Protocol):
    """Injectable request/response channel for one 64-byte transaction."""

    def exchange(self, packet: bytes, timeout_s: float) -> bytes: ...

    def close(self) -> None: ...


class SerialPacketChannel:
    """Raw 8-N-1 UART channel implemented with the Python standard library."""

    def __init__(self, path: str | Path, *, baud: int = DEFAULT_BAUD) -> None:
        self.path = Path(path)
        self.baud = baud
        self._lock = threading.Lock()
        self._closed = False
        try:
            self._fd = open_exclusive_uart(self.path)
        except OSError as exc:
            raise ControllerError(f"Could not open UART {self.path}: {exc}") from exc
        try:
            self._original_attributes = termios.tcgetattr(self._fd)
            configure_raw_uart(self._fd, baud, read_timeout_deciseconds=0)
        except Exception:
            os.close(self._fd)
            raise
        self._pending = bytearray()

    def exchange(self, packet: bytes, timeout_s: float) -> bytes:
        _validate_request(packet, timeout_s)
        expected_sequence = decode_command(packet).sequence
        with self._lock:
            if self._closed:
                raise ControllerError("UART packet channel is closed")
            deadline = time.monotonic() + timeout_s
            try:
                _write_fd_exact(self._fd, packet, deadline)
                while True:
                    response = _read_status_packet(self._fd, self._pending, deadline)
                    if _response_is_current_or_protocol_error(response, expected_sequence):
                        return response
            except OSError as exc:
                raise ControllerError(f"UART transaction failed: {exc}") from exc

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                termios.tcsetattr(self._fd, termios.TCSANOW, self._original_attributes)
            finally:
                os.close(self._fd)

    def __enter__(self) -> SerialPacketChannel:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        self.close()


class EyeTrackerController:
    """Reliable command and telemetry client for the eye-tracker service."""

    def __init__(
        self,
        channel: PacketChannel,
        *,
        initial_sequence: int | None = None,
        response_timeout_s: float = DEFAULT_RESPONSE_TIMEOUT_S,
        state_timeout_s: float = DEFAULT_STATE_TIMEOUT_S,
        retries: int = 3,
        retry_delay_s: float = DEFAULT_RETRY_DELAY_S,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if initial_sequence is None:
            initial_sequence = secrets.randbelow(0xFFFFFFFF) + 1
        if not 1 <= initial_sequence <= 0xFFFFFFFF:
            raise ValueError("initial_sequence must be in [1, 2^32-1]")
        if response_timeout_s <= 0.0:
            raise ValueError("response_timeout_s must be positive")
        if state_timeout_s <= 0.0:
            raise ValueError("state_timeout_s must be positive")
        if retries < 0:
            raise ValueError("retries cannot be negative")
        if retry_delay_s < 0.0:
            raise ValueError("retry_delay_s cannot be negative")
        self.channel = channel
        self.response_timeout_s = response_timeout_s
        self.state_timeout_s = state_timeout_s
        self.retries = retries
        self.retry_delay_s = retry_delay_s
        self._next_sequence = initial_sequence
        self._sleep = sleep

    def poll(self, *, timeout_s: float | None = None) -> DecodedStatus:
        packet = encode_command(CommandPacket(CommandCode.POLL, self._take_sequence()))
        return self._exchange(packet, timeout_s)

    def send_state_command(self, code: CommandCode) -> DecodedStatus:
        if code is CommandCode.POLL:
            raise ValueError("Use poll() for POLL commands")
        if code is CommandCode.START_EXPERIMENT:
            raise ValueError("Use start_experiment() with a subject for START_EXPERIMENT")
        return self._send_state_packet(
            CommandPacket(
                code=code,
                sequence=self._take_sequence(),
                controller_timestamp_ns=time.monotonic_ns(),
            )
        )

    def start_experiment(
        self, subject: str, *, timestamp_ns: int | None = None
    ) -> DecodedStatus:
        """Open a Pi Tracker experiment using Unix UTC nanoseconds."""
        validate_subject_name(subject)
        return self._send_state_packet(
            CommandPacket(
                code=CommandCode.START_EXPERIMENT,
                sequence=self._take_sequence(),
                controller_timestamp_ns=time.time_ns() if timestamp_ns is None else timestamp_ns,
                subject=subject,
            )
        )

    def stop_experiment(self) -> DecodedStatus:
        return self.send_state_command(CommandCode.STOP_EXPERIMENT)

    def _send_state_packet(self, command: CommandPacket) -> DecodedStatus:
        sequence = command.sequence
        command_packet = encode_command(command)
        attempts = self.retries + 1
        for attempt in range(attempts):
            try:
                status = self._exchange(command_packet, self.state_timeout_s)
            except (ProtocolError, ResponseTimeout):
                status = None
            acknowledged = self._matching_ack(sequence, status)
            if acknowledged is not None:
                return acknowledged
            # A state action is synchronous. If its response was lost or corrupt,
            # repeat the exact command: the service's sequence cache makes this
            # idempotent and preserves a rejection outcome. A POLL is never used
            # as proof that a state command succeeded.
            if attempt + 1 < attempts and self.retry_delay_s:
                self._sleep(self.retry_delay_s)
        raise AcknowledgementTimeout(sequence, attempts)

    def _exchange(self, packet: bytes, timeout_s: float | None = None) -> DecodedStatus:
        timeout = self.response_timeout_s if timeout_s is None else timeout_s
        response = self.channel.exchange(packet, timeout)
        if len(response) != TRANSACTION_SIZE:
            raise ProtocolError(
                f"Expected a {TRANSACTION_SIZE}-byte status, received {len(response)}"
            )
        status = decode_status(response)
        expected_sequence = decode_command(packet).sequence
        if status.request_sequence != expected_sequence:
            raise ProtocolError(
                "Status echoed request sequence "
                f"{status.request_sequence}, expected {expected_sequence}"
            )
        return status

    @staticmethod
    def _matching_ack(sequence: int, status: DecodedStatus | None) -> DecodedStatus | None:
        if status is None or status.request_sequence != sequence:
            return None
        if status.status is not StatusCode.OK:
            raise CommandRejected(sequence, status.status)
        return status

    def _take_sequence(self) -> int:
        sequence = self._next_sequence
        self._next_sequence = (sequence + 1) & 0xFFFFFFFF
        if self._next_sequence == 0:
            self._next_sequence = 1
        return sequence


def _validate_request(packet: bytes, timeout_s: float) -> None:
    if len(packet) != TRANSACTION_SIZE:
        raise ValueError(f"Transactions must be exactly {TRANSACTION_SIZE} bytes")
    if timeout_s <= 0.0:
        raise ValueError("timeout_s must be positive")


def _response_is_current_or_protocol_error(packet: bytes, expected_sequence: int) -> bool:
    status = decode_status(packet)
    return status.request_sequence == expected_sequence or (
        status.request_sequence == 0 and status.status is not StatusCode.OK
    )


def _wait_fd(fd: int, *, readable: bool, deadline: float) -> None:
    remaining = deadline - time.monotonic()
    if remaining <= 0.0:
        raise ResponseTimeout("Timed out waiting for eye-tracker status")
    readers = [fd] if readable else []
    writers = [] if readable else [fd]
    ready_read, ready_write, _exceptional = select.select(readers, writers, [], remaining)
    if not ready_read and not ready_write:
        raise ResponseTimeout("Timed out waiting for eye-tracker status")


def _write_fd_exact(fd: int, packet: bytes, deadline: float) -> None:
    remaining = memoryview(packet)
    while remaining:
        _wait_fd(fd, readable=False, deadline=deadline)
        try:
            written = os.write(fd, remaining)
        except BlockingIOError:
            continue
        if written <= 0:
            raise ControllerError("UART write made no progress")
        remaining = remaining[written:]


def _extract_status_packet(pending: bytearray) -> bytes | None:
    """Return one validated status record, discarding corrupt stream bytes."""
    discarded_complete_record = False
    while True:
        start = pending.find(b"ET")
        if start < 0:
            pending[:] = b"E" if pending.endswith(b"E") else b""
            if discarded_complete_record:
                raise ProtocolError("Discarded a corrupt UART status record")
            return None
        if start:
            del pending[:start]
        if len(pending) < TRANSACTION_SIZE:
            if discarded_complete_record:
                raise ProtocolError("Discarded a corrupt UART status record")
            return None
        candidate = bytes(pending[:TRANSACTION_SIZE])
        try:
            decode_status(candidate)
        except ProtocolError:
            # The magic may have occurred in noise or a damaged record. Advance
            # one byte, then search again so the next intact response can frame.
            del pending[0]
            discarded_complete_record = True
            continue
        del pending[:TRANSACTION_SIZE]
        return candidate


def _read_status_packet(
    fd: int,
    pending: bytearray,
    deadline: float,
    *,
    inter_byte_timeout_s: float = DEFAULT_INTER_BYTE_TIMEOUT_S,
) -> bytes:
    if inter_byte_timeout_s <= 0.0:
        raise ValueError("inter_byte_timeout_s must be positive")
    read_deadline = (
        min(deadline, time.monotonic() + inter_byte_timeout_s) if pending else deadline
    )
    while True:
        packet = _extract_status_packet(pending)
        if packet is not None:
            return packet
        _wait_fd(fd, readable=True, deadline=read_deadline)
        try:
            chunk = os.read(fd, TRANSACTION_SIZE * 4)
        except BlockingIOError:
            continue
        if not chunk:
            continue
        pending.extend(chunk)
        # At 460800 baud a complete record takes about 1.4 ms. Once a
        # response begins, a much shorter idle deadline distinguishes a lost
        # byte from a camera action that legitimately takes several seconds.
        read_deadline = min(deadline, time.monotonic() + inter_byte_timeout_s)
