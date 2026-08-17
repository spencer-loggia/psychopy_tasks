import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = ROOT / "task"
SUBJECT_WINDOW_TASKS = {
    "active_foraging.py",
    "afc_csc1.py",
    "afc_trial_sequence.py",
    "calibrate_eye_tracker.py",
    "match2cue.py",
    "play_video.py",
    "random_image_sequence.py",
}


def _call_name(node: ast.Call) -> str:
    parts = []
    value = node.func
    while isinstance(value, ast.Attribute):
        parts.append(value.attr)
        value = value.value
    if isinstance(value, ast.Name):
        parts.append(value.id)
    return ".".join(reversed(parts))


class ScreenPathPolicyTests(unittest.TestCase):
    def test_every_subject_window_uses_the_privileged_task_factory(self):
        for filename in sorted(SUBJECT_WINDOW_TASKS):
            path = TASK_ROOT / filename
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            calls = [_call_name(node) for node in ast.walk(tree) if isinstance(node, ast.Call)]
            self.assertEqual(
                calls.count("utils.setup_task_window"),
                1,
                f"{filename} must open its subject window exactly once through utils.setup_task_window",
            )

    def test_task_code_cannot_bypass_screen_resolution_or_window_verification(self):
        forbidden_names = {
            "open_psychopy_window",
            "initialize_psychopy_window",
            "resolve_task_screens",
            "Window",
        }
        for path in sorted(TASK_ROOT.glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            calls = [_call_name(node) for node in ast.walk(tree) if isinstance(node, ast.Call)]
            if path.name != "system_diagnostic.py":
                bypasses = [
                    name for name in calls if name.rsplit(".", 1)[-1] in forbidden_names
                ]
                self.assertEqual(
                    bypasses,
                    [],
                    f"{path.name} bypasses the privileged screen path: {bypasses}",
                )

            low_level_calls = [
                name for name in calls if name.rsplit(".", 1)[-1] == "setup_window"
            ]
            expected = ["utils.setup_window"] if path.name == "calibrate_eye_tracker.py" else []
            self.assertEqual(
                low_level_calls,
                expected,
                f"{path.name} may not open a task window through low-level setup_window",
            )


if __name__ == "__main__":
    unittest.main()
