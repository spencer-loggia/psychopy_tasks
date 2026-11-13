import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from video.recorder import (
    Recording, RecordingError, RecorderConfig, _PictureProbe, _file_signature, wait_for_fresh_picture,
)


HEADERS = b"\x00\x00\x00\x01\x67\x42\x00\x1e\x00\x00\x01\x68\xce"
PICTURE = b"\x00\x00\x01\x65\x88\x84\x21"
NEXT_PICTURE = b"\x00\x00\x00\x01\x41\x9a"


class PictureProbeTests(unittest.TestCase):
    def test_headers_and_incomplete_picture_are_not_ready(self):
        probe = _PictureProbe()
        self.assertFalse(probe.feed(HEADERS))
        self.assertFalse(probe.feed(PICTURE))
        self.assertTrue(probe.feed(NEXT_PICTURE))

    def test_both_start_code_lengths_can_cross_every_chunk_boundary(self):
        data = HEADERS + PICTURE + NEXT_PICTURE
        for split in range(1, len(data)):
            with self.subTest(split=split):
                probe = _PictureProbe()
                ready = probe.feed(data[:split])
                ready = ready or probe.feed(data[split:])
                self.assertTrue(ready)
        probe = _PictureProbe()
        self.assertTrue(any(probe.feed(bytes([byte])) for byte in data))

    def test_empty_or_invalid_picture_nal_is_not_ready(self):
        for header in (b"\x65", b"\xe5"):
            with self.subTest(header=header):
                data = b"\x00\x00\x01" + header + b"\x00\x00\x00\x01\x67\x42"
                self.assertFalse(_PictureProbe().feed(data))

    def test_large_picture_does_not_accumulate_in_memory(self):
        probe = _PictureProbe()
        self.assertFalse(probe.feed(PICTURE))
        for _ in range(100):
            self.assertFalse(probe.feed(b"\x88" * 32768))
            self.assertLessEqual(len(probe.tail), 5)
        self.assertTrue(probe.feed(NEXT_PICTURE))


class RecordingReadinessTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.destination = Path(self.directory.name) / "camera.h264"
        self.process = Mock(stderr=None)
        self.process.poll.return_value = None

    def recording(self, **kwargs):
        return Recording(self.process, self.destination, ["rpicam-vid"], **kwargs)

    def test_waits_for_picture_then_leaves_recording_running(self):
        self.destination.write_bytes(HEADERS)
        recording = self.recording()

        def append_picture(_delay):
            with self.destination.open("ab") as stream:
                stream.write(PICTURE + NEXT_PICTURE)

        with patch("video.recorder.time.sleep", side_effect=append_picture):
            self.assertEqual(recording.wait_until_ready(), self.destination)
        self.process.wait.assert_not_called()

    def test_missing_file_or_headers_only_time_out(self):
        for data in (None, HEADERS, HEADERS + PICTURE):
            with self.subTest(data=data):
                if data is not None:
                    self.destination.write_bytes(data)
                recording = self.recording()
                with patch("video.recorder.time.monotonic", side_effect=[0, 2]):
                    with self.assertRaisesRegex(RecordingError, "encoded camera picture"):
                        recording.wait_until_ready(timeout=1)

    def test_existing_output_does_not_count_as_new_acquisition(self):
        self.destination.write_bytes(HEADERS + PICTURE + NEXT_PICTURE)
        recording = self.recording(initial_signature=_file_signature(self.destination))
        with patch("video.recorder.time.monotonic", side_effect=[0, 2]):
            with self.assertRaisesRegex(RecordingError, "Timed out"):
                recording.wait_until_ready(timeout=1)

    def test_exited_process_is_never_ready_even_with_picture_output(self):
        self.destination.write_bytes(HEADERS + PICTURE + NEXT_PICTURE)
        for status in (0, 1):
            with self.subTest(status=status):
                self.process.poll.return_value = status
                with self.assertRaisesRegex(RecordingError, "exited before acquisition"):
                    self.recording().wait_until_ready()

    def test_validates_timeout(self):
        for timeout in (0, -1, float("inf"), float("nan"), True):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                self.recording().wait_until_ready(timeout)

    def test_recorder_flushes_encoded_buffers_for_prompt_readiness(self):
        self.assertIn("--flush", RecorderConfig().command(self.destination))

    def test_fresh_check_rejects_stalled_recording_with_old_pictures(self):
        self.destination.write_bytes(HEADERS + PICTURE + NEXT_PICTURE)
        with patch("video.recorder.time.monotonic", side_effect=[0, 2]):
            with self.assertRaisesRegex(RecordingError, "Timed out"):
                wait_for_fresh_picture(self.destination, timeout=1)

    def test_fresh_check_waits_for_new_picture_after_existing_data(self):
        self.destination.write_bytes(HEADERS + PICTURE + NEXT_PICTURE)

        def append_picture(_delay):
            with self.destination.open("ab") as stream:
                stream.write(PICTURE + NEXT_PICTURE)

        with patch("video.recorder.time.sleep", side_effect=append_picture) as sleep:
            self.assertEqual(wait_for_fresh_picture(self.destination), self.destination)
        sleep.assert_called_once()

    def test_fresh_check_fails_for_missing_output(self):
        with self.assertRaisesRegex(RecordingError, "Cannot check fresh"):
            wait_for_fresh_picture(self.destination)


if __name__ == "__main__":
    unittest.main()
