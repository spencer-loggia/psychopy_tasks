from __future__ import annotations

import enum
import re
import struct
from dataclasses import dataclass


@dataclass(frozen=True)
class EyeMeasurement:
    eye_id: int
    x: float
    y: float
    pupil_diameter: float
    confidence: float
    valid: bool
    blink: bool


MAGIC = b"ET"
PROTOCOL_VERSION = 1
TRANSACTION_SIZE = 64
_HEADER = struct.Struct("<2sBBHHIIQ")
_EYE = struct.Struct("<iiHBB")
_TAIL = struct.Struct("<IHHHBB")
_CRC = struct.Struct("<I")
assert _HEADER.size + 2 * _EYE.size + _TAIL.size + _CRC.size == TRANSACTION_SIZE


class ProtocolError(ValueError):
    pass


class PacketType(enum.IntEnum):
    COMMAND = 1
    STATUS = 2


class CommandCode(enum.IntEnum):
    POLL = 0
    START_TRACKING = 1
    STOP_TRACKING = 2
    START_RECORDING = 3
    STOP_RECORDING = 4
    CLEAR_ERROR = 5
    START_EXPERIMENT = 6
    STOP_EXPERIMENT = 7
    SHUTDOWN = 8


class StatusCode(enum.IntEnum):
    OK = 0
    BAD_MAGIC = 1
    BAD_VERSION = 2
    BAD_PACKET_TYPE = 3
    BAD_CRC = 4
    UNKNOWN_COMMAND = 5
    INVALID_STATE = 6
    INTERNAL_ERROR = 7
    SEQUENCE_CONFLICT = 8


class StatusFlag(enum.IntFlag):
    TRACKING = 1 << 0
    RECORDING = 1 << 1
    CONFIGURED = 1 << 2
    ERROR = 1 << 3
    RESULT_STALE = 1 << 4
    EXPERIMENT_ACTIVE = 1 << 5
    CALIBRATING = 1 << 6
    SHUTTING_DOWN = 1 << 7


class EyeFlag(enum.IntFlag):
    VALID = 1 << 0
    BLINK = 1 << 1
    LOST = 1 << 2


@dataclass(frozen=True)
class CommandPacket:
    code: CommandCode
    sequence: int
    controller_timestamp_ns: int = 0
    flags: int = 0
    subject: str | None = None
    experiment_number: int | None = None
    block_number: int | None = None


@dataclass(frozen=True)
class DecodedStatus:
    status: StatusCode
    flags: StatusFlag
    frame_sequence: int
    sensor_timestamp_ns: int
    request_sequence: int
    session_id: int
    configured_eye_count: int
    eyes: tuple[EyeMeasurement, ...]
    dropped_analysis_frames: int
    processing_time_us: int
    result_age_ms: int


def crc32c(data: bytes, initial: int = 0xFFFFFFFF) -> int:
    """CRC-32C/Castagnoli with the conventional initial/final XOR."""
    crc = initial & 0xFFFFFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ 0x82F63B78 if crc & 1 else crc >> 1
    return (crc ^ 0xFFFFFFFF) & 0xFFFFFFFF


def _checked_u32(value: int, name: str) -> int:
    if not 0 <= value <= 0xFFFFFFFF:
        raise ValueError(f"{name} must fit in uint32")
    return value


def validate_subject_name(subject: str) -> str:
    """Use the tracker's subject identifier unchanged on the wire."""
    if (
        not isinstance(subject, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,31}", subject, flags=re.ASCII) is None
    ):
        raise ValueError(
            "Subject must be 1–32 ASCII letters, digits, underscores, dots, or "
            "hyphens, starting with a letter or digit"
        )
    return subject


def validate_run_number(value: int | None) -> int | None:
    """Accept automatic numbering (None) or a positive uint32 experiment/block ID."""
    if value is not None and (
        isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 0xFFFFFFFF
    ):
        raise ValueError("Experiment/block number must be an integer in [1, 2^32-1]")
    return value


def encode_command(command: CommandPacket) -> bytes:
    sequence = _checked_u32(command.sequence, "command sequence")
    if sequence == 0:
        raise ValueError("command sequence must be nonzero")
    if not 0 <= command.flags <= 0xFFFF:
        raise ValueError("command flags must fit in uint16")
    if not 0 <= command.controller_timestamp_ns <= 0xFFFFFFFFFFFFFFFF:
        raise ValueError("controller_timestamp_ns must fit in uint64")
    subject_bytes = b""
    experiment_number = validate_run_number(command.experiment_number)
    block_number = validate_run_number(command.block_number)
    if experiment_number is not None and command.code != CommandCode.START_EXPERIMENT:
        raise ValueError("Only START_EXPERIMENT accepts an experiment number")
    if block_number is not None and command.code not in (
        CommandCode.START_TRACKING, CommandCode.START_RECORDING,
    ):
        raise ValueError("Only START_TRACKING/START_RECORDING accept a block number")
    if command.code == CommandCode.START_EXPERIMENT:
        if command.controller_timestamp_ns == 0:
            raise ValueError("START_EXPERIMENT requires a nonzero Unix UTC timestamp")
        subject_bytes = validate_subject_name(command.subject).encode("ascii")
    elif command.subject is not None:
        raise ValueError("Only START_EXPERIMENT accepts a subject")
    packet = bytearray(TRANSACTION_SIZE)
    _HEADER.pack_into(
        packet,
        0,
        MAGIC,
        PROTOCOL_VERSION,
        PacketType.COMMAND,
        command.flags,
        int(command.code),
        sequence,
        experiment_number or block_number or 0,
        command.controller_timestamp_ns,
    )
    if subject_bytes:
        packet[24] = len(subject_bytes)
        packet[25 : 25 + len(subject_bytes)] = subject_bytes
    _CRC.pack_into(packet, TRANSACTION_SIZE - _CRC.size, crc32c(packet[:-4]))
    return bytes(packet)


def decode_command(packet: bytes) -> CommandPacket:
    if len(packet) != TRANSACTION_SIZE:
        raise ProtocolError(
            f"Expected a {TRANSACTION_SIZE}-byte transaction, received {len(packet)}"
        )
    expected_crc = _CRC.unpack_from(packet, TRANSACTION_SIZE - 4)[0]
    if crc32c(packet[:-4]) != expected_crc:
        raise ProtocolError("BAD_CRC")
    magic, version, packet_type, flags, code, sequence, related, timestamp_ns = (
        _HEADER.unpack_from(packet)
    )
    if magic != MAGIC:
        raise ProtocolError("BAD_MAGIC")
    if version != PROTOCOL_VERSION:
        raise ProtocolError("BAD_VERSION")
    if packet_type != PacketType.COMMAND:
        raise ProtocolError("BAD_PACKET_TYPE")
    if sequence == 0:
        raise ProtocolError("BAD_SEQUENCE")
    try:
        command_code = CommandCode(code)
    except ValueError as exc:
        raise ProtocolError("UNKNOWN_COMMAND") from exc
    if related and command_code not in (
        CommandCode.START_EXPERIMENT, CommandCode.START_TRACKING, CommandCode.START_RECORDING,
    ):
        raise ProtocolError("BAD_RESERVED")
    subject = None
    if command_code is CommandCode.START_EXPERIMENT:
        if timestamp_ns == 0:
            raise ProtocolError("BAD_TIMESTAMP")
        subject_length = packet[24]
        if not 1 <= subject_length <= 32:
            raise ProtocolError("BAD_SUBJECT")
        try:
            subject = validate_subject_name(packet[25 : 25 + subject_length].decode("ascii"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise ProtocolError("BAD_SUBJECT") from exc
        if any(packet[25 + subject_length : TRANSACTION_SIZE - _CRC.size]):
            raise ProtocolError("BAD_RESERVED")
    elif any(packet[_HEADER.size : TRANSACTION_SIZE - _CRC.size]):
        raise ProtocolError("BAD_RESERVED")
    return CommandPacket(
        code=command_code,
        sequence=sequence,
        controller_timestamp_ns=timestamp_ns,
        flags=flags,
        subject=subject,
        experiment_number=(related or None)
        if command_code is CommandCode.START_EXPERIMENT else None,
        block_number=(related or None)
        if command_code in (CommandCode.START_TRACKING, CommandCode.START_RECORDING) else None,
    )


def decode_status(packet: bytes) -> DecodedStatus:
    if len(packet) != TRANSACTION_SIZE:
        raise ProtocolError(
            f"Expected a {TRANSACTION_SIZE}-byte transaction, received {len(packet)}"
        )
    expected_crc = _CRC.unpack_from(packet, TRANSACTION_SIZE - 4)[0]
    if crc32c(packet[:-4]) != expected_crc:
        raise ProtocolError("BAD_CRC")
    magic, version, packet_type, raw_flags, code, sequence, related, timestamp_ns = (
        _HEADER.unpack_from(packet)
    )
    if magic != MAGIC:
        raise ProtocolError("BAD_MAGIC")
    if version != PROTOCOL_VERSION:
        raise ProtocolError("BAD_VERSION")
    if packet_type != PacketType.STATUS:
        raise ProtocolError("BAD_PACKET_TYPE")
    try:
        status = StatusCode(code)
    except ValueError as exc:
        raise ProtocolError(f"Unknown status code: {code}") from exc
    known_status_flags = int(
        StatusFlag.TRACKING
        | StatusFlag.RECORDING
        | StatusFlag.CONFIGURED
        | StatusFlag.ERROR
        | StatusFlag.RESULT_STALE
        | StatusFlag.EXPERIMENT_ACTIVE
        | StatusFlag.CALIBRATING
        | StatusFlag.SHUTTING_DOWN
    )
    if raw_flags & ~known_status_flags:
        raise ProtocolError("Status contains unknown flag bits")
    flags = StatusFlag(raw_flags)
    eyes: list[EyeMeasurement] = []
    for eye_id in (0, 1):
        x_raw, y_raw, diameter_raw, confidence_raw, eye_flags_raw = _EYE.unpack_from(
            packet, _HEADER.size + eye_id * _EYE.size
        )
        eye_flags = EyeFlag(eye_flags_raw)
        known_eye_flags = int(EyeFlag.VALID | EyeFlag.BLINK | EyeFlag.LOST)
        if eye_flags_raw & ~known_eye_flags:
            raise ProtocolError(f"Eye {eye_id} contains unknown flag bits")
        valid = bool(eye_flags & EyeFlag.VALID)
        blink = bool(eye_flags & EyeFlag.BLINK)
        lost = bool(eye_flags & EyeFlag.LOST)
        if sum((valid, blink, lost)) != 1:
            raise ProtocolError("Eye record must set exactly one state flag")
        if not valid and diameter_raw != 0:
            raise ProtocolError("Blink/lost eye record must have zero diameter")
        if valid and diameter_raw == 0:
            raise ProtocolError("Valid eye record must have a positive diameter")
        eyes.append(
            EyeMeasurement(
                eye_id=eye_id,
                x=x_raw / 65536.0,
                y=y_raw / 65536.0,
                pupil_diameter=diameter_raw / 16.0 if valid and not blink else 0.0,
                confidence=confidence_raw / 255.0,
                valid=valid,
                blink=blink,
            )
        )
    session_id, dropped, processing_us, age_ms, eye_count, _reserved = _TAIL.unpack_from(
        packet, _HEADER.size + 2 * _EYE.size
    )
    if eye_count > 2:
        raise ProtocolError("Configured eye count must be 0, 1, or 2")
    if _reserved != 0:
        raise ProtocolError("Status reserved byte must be zero")
    return DecodedStatus(
        status=status,
        flags=flags,
        frame_sequence=sequence,
        sensor_timestamp_ns=timestamp_ns,
        request_sequence=related,
        session_id=session_id,
        configured_eye_count=eye_count,
        eyes=tuple(eyes),
        dropped_analysis_frames=dropped,
        processing_time_us=processing_us,
        result_age_ms=age_ms,
    )
