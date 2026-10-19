"""Wire compatibility and real stream framing, without a Pi or camera."""

import json
import os
from pathlib import Path
import pty
import struct
import threading
import time
import unittest
from unittest.mock import Mock, patch

from bin.eye_tracker_wire.controller import (
    EyeTrackerController,
    SerialPacketChannel,
    ResponseTimeout,
    CommandRejected,
    AcknowledgementTimeout,
    _extract_status_packet,
)
from bin.eye_tracker_wire.protocol import (
    CommandCode,
    CommandPacket,
    ProtocolError,
    StatusCode,
    StatusFlag,
    crc32c,
    encode_command,
    decode_command,
    decode_status,
)

FIXTURE = json.loads((Path(__file__).parent / "fixtures/eye_uart_v1.json").read_text())
STATUS_BYTES = bytes.fromhex(FIXTURE["status"])


def status_bytes(*, sequence=42, code=0, edits=()):
    packet = bytearray(STATUS_BYTES)
    struct.pack_into("<I", packet, 12, sequence)
    struct.pack_into("<H", packet, 6, code)
    for offset, value in edits:
        packet[offset] = value
    struct.pack_into("<I", packet, 60, crc32c(packet[:60]))
    return bytes(packet)


class UARTProtocolTests(unittest.TestCase):
    def test_shutdown_matches_upstream_and_retries_identical_command(self):
        channel = Mock()
        channel.exchange.side_effect = [ResponseTimeout("lost ACK"), bytes.fromhex(FIXTURE["shutdown_status"])]
        controller = EyeTrackerController(channel, initial_sequence=42, retries=1, retry_delay_s=0)
        with patch("bin.eye_tracker_wire.controller.time.monotonic_ns", return_value=111222333):
            status = controller.shutdown()
        self.assertEqual(status.flags, StatusFlag.SHUTTING_DOWN)
        calls = channel.exchange.call_args_list
        self.assertEqual(calls[0].args[0].hex(), FIXTURE["shutdown_command"])
        self.assertEqual(calls[0].args[0], calls[1].args[0])
        packet = decode_command(calls[0].args[0])
        self.assertEqual(packet.code, CommandCode.SHUTDOWN)
        self.assertEqual(packet.flags, 0)
        self.assertIsNone(packet.subject)
        self.assertEqual(calls[0].args[0][24:60], bytes(36))

    def test_experiment_commands_match_upstream_packets_and_new_status_flags(self):
        start = CommandPacket(CommandCode.START_EXPERIMENT, 42, 1791988200123456000, subject="Yuri")
        self.assertEqual(encode_command(start).hex(), FIXTURE["start_experiment"])
        self.assertEqual(decode_command(bytes.fromhex(FIXTURE["start_experiment"])), start)
        stop = CommandPacket(CommandCode.STOP_EXPERIMENT, 43, 111222333)
        self.assertEqual(encode_command(stop).hex(), FIXTURE["stop_experiment"])
        status = decode_status(bytes.fromhex(FIXTURE["experiment_status"]))
        self.assertEqual(status.flags, StatusFlag.EXPERIMENT_ACTIVE | StatusFlag.CALIBRATING)

    def test_experiment_start_validation_and_padding(self):
        for subject in (None, "", "Subject One", "../x", "café", "A" * 33):
            with self.subTest(subject=subject), self.assertRaises(ValueError):
                encode_command(CommandPacket(CommandCode.START_EXPERIMENT, 1, 1, subject=subject))
        for timestamp in (0, -1, 1 << 64):
            with self.subTest(timestamp=timestamp), self.assertRaises(ValueError):
                encode_command(CommandPacket(CommandCode.START_EXPERIMENT, 1, timestamp, subject="A"))
        packet = encode_command(CommandPacket(CommandCode.START_EXPERIMENT, 1, 1, subject="A"))
        for offset, value in ((24, 0), (24, 33), (25, 255), (26, 1), (59, 1), (16, 0)):
            malformed = bytearray(packet)
            malformed[offset] = value
            struct.pack_into("<I", malformed, 60, crc32c(malformed[:60]))
            with self.subTest(offset=offset), self.assertRaises(ProtocolError):
                decode_command(bytes(malformed))

    def test_experiment_start_retries_original_utc_timestamp_and_subject(self):
        channel = Mock()
        channel.exchange.side_effect = [ResponseTimeout("lost ACK"), status_bytes(), status_bytes(sequence=43)]
        controller = EyeTrackerController(channel, initial_sequence=42, retries=1, retry_delay_s=0)
        with patch("bin.eye_tracker_wire.controller.time.time_ns", return_value=1791988200123456000) as clock:
            controller.start_experiment("Yuri", experiment_number=7)
            clock.assert_called_once_with()
        controller.stop_experiment()
        calls = channel.exchange.call_args_list
        self.assertEqual(calls[0].args[0], bytes.fromhex(FIXTURE["numbered_experiment"]))
        self.assertEqual(calls[0].args[0], calls[1].args[0])
        self.assertEqual(decode_command(calls[2].args[0]).code, CommandCode.STOP_EXPERIMENT)
        self.assertEqual(calls[2].args[0][24:60], bytes(36))

    def test_numbered_starts_match_upstream_and_retry_identically(self):
        for method, key, code in (
            ("start_tracking", "numbered_tracking", CommandCode.START_TRACKING),
            ("start_recording", "numbered_recording", CommandCode.START_RECORDING),
        ):
            with self.subTest(method=method):
                channel = Mock()
                channel.exchange.side_effect = [ResponseTimeout("lost ACK"), status_bytes()]
                controller = EyeTrackerController(channel, initial_sequence=42, retries=1, retry_delay_s=0)
                with patch("bin.eye_tracker_wire.controller.time.monotonic_ns", return_value=111222333) as clock:
                    getattr(controller, method)(block_number=0x01020304)
                clock.assert_called_once_with()
                packets = [call.args[0] for call in channel.exchange.call_args_list]
                self.assertEqual(packets, [bytes.fromhex(FIXTURE[key])] * 2)
                self.assertEqual(decode_command(packets[0]), CommandPacket(code, 42, 111222333, block_number=0x01020304))
                self.assertEqual(len(packets[0]), 64)
                self.assertEqual(packets[0][12:16], b"\x04\x03\x02\x01")
                self.assertEqual(packets[0][24:60], bytes(36))

    def test_start_numbers_validate_uint32_and_round_trip(self):
        for code in (CommandCode.START_EXPERIMENT, CommandCode.START_TRACKING, CommandCode.START_RECORDING):
            field = "experiment_number" if code == CommandCode.START_EXPERIMENT else "block_number"
            payload = {"subject": "Yuri"} if code == CommandCode.START_EXPERIMENT else {}
            for number in (None, 1, 0xFFFFFFFF):
                command = CommandPacket(code, 42, 1, **payload, **{field: number})
                with self.subTest(code=code, number=number):
                    packet = encode_command(command)
                    self.assertEqual(struct.unpack_from("<I", packet, 12)[0], number or 0)
                    self.assertEqual(decode_command(packet), command)
            for number in (0, -1, True, 1.5, "2", 1 << 32):
                with self.subTest(code=code, number=number), self.assertRaises(ValueError):
                    encode_command(CommandPacket(code, 42, 1, **payload, **{field: number}))

    def test_numbers_are_rejected_on_other_commands(self):
        for code in CommandCode:
            for field, allowed in (
                ("experiment_number", (CommandCode.START_EXPERIMENT,)),
                ("block_number", (CommandCode.START_TRACKING, CommandCode.START_RECORDING)),
            ):
                if code not in allowed:
                    with self.subTest(code=code, field=field), self.assertRaises(ValueError):
                        encode_command(CommandPacket(code, 42, 1, **{field: 7}))
            if code not in (CommandCode.START_EXPERIMENT, CommandCode.START_TRACKING, CommandCode.START_RECORDING):
                packet = bytearray(encode_command(CommandPacket(code, 42)))
                struct.pack_into("<I", packet, 12, 7)
                struct.pack_into("<I", packet, 60, crc32c(packet[:60]))
                with self.subTest(code=code), self.assertRaisesRegex(ProtocolError, "BAD_RESERVED"):
                    decode_command(bytes(packet))

    def test_codec_matches_upstream_golden_packets(self):
        self.assertEqual(crc32c(b"123456789"), 0xE3069283)
        self.assertEqual(
            encode_command(CommandPacket(CommandCode.START_TRACKING, 42, 111222333)),
            bytes.fromhex(FIXTURE["start_command"]),
        )
        status = decode_status(STATUS_BYTES)
        self.assertEqual(status.eyes[0].x, 123.25)
        self.assertEqual(status.eyes[0].y, -45.5)
        self.assertEqual(status.eyes[0].pupil_diameter, 12.5)
        self.assertEqual(status.eyes[1].x, 999)
        self.assertEqual(status.session_id, 321)
        self.assertEqual(status.result_age_ms, 2)

    def test_corruption_reserved_and_invalid_flags_rejected(self):
        for edits in (
            ((2, 2),),
            ((3, 1),),
            ((5, 1),),
            ((35, 3),),
            ((35, 0),),
            ((32, 0), (33, 0)),
            ((58, 3),),
            ((59, 1),),
        ):
            with self.subTest(edits=edits), self.assertRaises(ProtocolError):
                decode_status(status_bytes(edits=edits))
        corrupt = bytearray(STATUS_BYTES)
        corrupt[28] ^= 1
        with self.assertRaises(ProtocolError):
            decode_status(bytes(corrupt))

    def test_stream_resynchronizes_after_noise_corruption_and_split_magic(self):
        pending = bytearray(b"noiseE")
        self.assertIsNone(_extract_status_packet(pending))
        self.assertEqual(pending, b"E")
        pending.extend(STATUS_BYTES[1:20])
        self.assertIsNone(_extract_status_packet(pending))
        pending.extend(STATUS_BYTES[20:])
        self.assertEqual(_extract_status_packet(pending), STATUS_BYTES)
        corrupt = bytearray(STATUS_BYTES)
        corrupt[22] ^= 1
        pending.extend(corrupt + STATUS_BYTES)
        self.assertEqual(_extract_status_packet(pending), STATUS_BYTES)
        self.assertFalse(pending)

    def test_state_retry_is_byte_identical_and_poll_uses_next_sequence(self):
        channel = Mock()
        channel.exchange.side_effect = [
            ResponseTimeout("lost ACK"),
            status_bytes(),
            status_bytes(sequence=43),
        ]
        controller = EyeTrackerController(
            channel, initial_sequence=42, retries=1, retry_delay_s=0
        )
        controller.send_state_command(CommandCode.START_TRACKING)
        controller.poll()
        calls = channel.exchange.call_args_list
        self.assertEqual(calls[0].args[0], calls[1].args[0])
        self.assertEqual(calls[0].args[1], 10)
        self.assertEqual(decode_command(calls[2].args[0]).sequence, 43)

    def test_rejected_state_and_mismatched_ack_are_not_success(self):
        channel = Mock()
        channel.exchange.return_value = status_bytes(code=StatusCode.INVALID_STATE)
        controller = EyeTrackerController(channel, initial_sequence=42, retries=0)
        with self.assertRaises(CommandRejected):
            controller.send_state_command(CommandCode.START_TRACKING)
        channel.exchange.return_value = status_bytes(sequence=999)
        with self.assertRaises(AcknowledgementTimeout):
            controller.send_state_command(CommandCode.STOP_TRACKING)

    def test_sequence_zero_protocol_error_is_surfaced_without_waiting(self):
        channel = Mock()
        channel.exchange.return_value = status_bytes(
            sequence=0, code=StatusCode.BAD_CRC
        )
        controller = EyeTrackerController(channel, initial_sequence=42)
        with self.assertRaises(ProtocolError):
            controller.poll()

    def test_serial_exchange_handles_fragmented_response_and_late_ack(self):
        master, slave = pty.openpty()
        path = os.ttyname(slave)
        channel = None
        worker_errors = []
        try:
            # Pseudo-terminals use a portable baud; wire pacing is not simulated.
            channel = SerialPacketChannel(path, baud=9600)

            def peer():
                try:
                    request = bytearray()
                    while len(request) < 64:
                        request.extend(os.read(master, 64 - len(request)))
                    self.assertEqual(decode_command(bytes(request)).sequence, 42)
                    response = (
                        b"noise" + status_bytes(sequence=41) + status_bytes(sequence=42)
                    )
                    for start in range(0, len(response), 7):
                        os.write(master, response[start : start + 7])
                        time.sleep(0.001)
                except BaseException as exc:
                    worker_errors.append(exc)

            worker = threading.Thread(target=peer, daemon=True)
            worker.start()
            response = channel.exchange(
                encode_command(CommandPacket(CommandCode.POLL, 42)), 1
            )
            worker.join(1)
            self.assertEqual(decode_status(response).request_sequence, 42)
            self.assertFalse(worker_errors)
        finally:
            if channel is not None:
                channel.close()
            os.close(slave)
            os.close(master)


if __name__ == "__main__":
    unittest.main()
