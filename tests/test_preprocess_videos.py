import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bin import preprocess_videos


class VideoPreprocessingTests(unittest.TestCase):
    @staticmethod
    def _existing_output_call(output_path: Path) -> str:
        return preprocess_videos.preprocess_video(
            ffmpeg_bin="ffmpeg",
            ffprobe_bin="ffprobe",
            input_path=Path("input.mp4"),
            input_codec_name="hevc",
            output_path=output_path,
            filter_chain="unused",
            codecs=["libx265"],
            preset="veryfast",
            crf=20,
            tune="fastdecode",
            gop_frames=60,
            overwrite=False,
            expected_size=(1280, 720),
            expected_frame_rate=30.0,
        )

    @staticmethod
    def _valid_stream(**overrides):
        stream = {
            "codec_name": "hevc",
            "profile": "Main",
            "pix_fmt": "yuv420p",
            "width": 1280,
            "height": 720,
            "avg_frame_rate": "30/1",
            "start_time": "0.0",
        }
        stream.update(overrides)
        return stream

    def test_existing_valid_output_is_validated_then_skipped(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "clip.mp4"
            output_path.write_bytes(b"existing")

            with (
                patch.object(
                    preprocess_videos,
                    "probe_video",
                    return_value=self._valid_stream(),
                ) as probe,
                patch.object(preprocess_videos, "_run_checked") as encode,
                patch("builtins.print"),
            ):
                result = self._existing_output_call(output_path)

        self.assertEqual(result, "skipped")
        probe.assert_called_once_with("ffprobe", output_path)
        encode.assert_not_called()

    def test_existing_invalid_output_is_rejected_without_encoding(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "clip.mp4"
            output_path.write_bytes(b"existing")

            with (
                patch.object(
                    preprocess_videos,
                    "probe_video",
                    return_value=self._valid_stream(start_time="0.25"),
                ),
                patch.object(preprocess_videos, "_run_checked") as encode,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "Existing output does not satisfy.*rerun with --overwrite",
                ):
                    self._existing_output_call(output_path)

            self.assertTrue(output_path.exists())
            encode.assert_not_called()

    def test_filter_normalizes_rate_before_rebasing_pts(self):
        filter_chain = preprocess_videos.build_filter(
            1920,
            1080,
            1280,
            720,
            frame_rate=30.0,
        )

        self.assertTrue(
            filter_chain.endswith(
                ",fps=fps=30:round=near,setpts=PTS-STARTPTS"
            )
        )


if __name__ == "__main__":
    unittest.main()
