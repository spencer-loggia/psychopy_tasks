import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from uuid import UUID

from bin.logger import SessionLogBundle
from bin.run_identity import BLOCK_ID_ENV, EXPERIMENT_ID_ENV, prepare_block_identity


class RunIdentityTests(unittest.TestCase):
    def test_standalone_block_persists_id_before_eye_tracking_starts(self):
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(os.environ, {}, clear=True), patch(
            "bin.task_eye_tracking.TaskEyeTrackingSession"
        ) as eye_session:
            def start(settings, **kwargs):
                identity = json.loads((kwargs["output_dir"] / "block.json").read_text())
                self.assertEqual(identity["block_id"], kwargs["block_id"])
                self.assertEqual(UUID(identity["block_id"]).version, 4)
                return None

            eye_session.side_effect = start
            from bin.eye_tracking_config import EyeTrackingSettings
            with_session = SessionLogBundle(output_root=tmpdir, task_name="active_foraging",
                                           config_name="identity", eye_tracking_config=EyeTrackingSettings())
            with_session.close()
            saved = json.loads((with_session.session_dir / "block.json").read_text())
            self.assertIsNone(saved["experiment_id"])
            self.assertEqual(prepare_block_identity(with_session.session_dir), saved)
            self.assertEqual(eye_session.call_count, 1)

    def test_launcher_identity_is_verified_and_never_replaced(self):
        first = "a0000000-0000-4000-8000-000000000001"
        second = "b0000000-0000-4000-8000-000000000002"
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(os.environ, {
            EXPERIMENT_ID_ENV: first, BLOCK_ID_ENV: second,
        }):
            directory = Path(tmpdir)
            identity = prepare_block_identity(directory)
            self.assertEqual(identity, {"experiment_id": first, "block_id": second})
            with patch.dict(os.environ, {BLOCK_ID_ENV: first}), self.assertRaisesRegex(ValueError, "block_id"):
                prepare_block_identity(directory)
            self.assertEqual(json.loads((directory / "block.json").read_text()), identity)
