from contextlib import ExitStack
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

from PIL import Image
import bin
from bin.participant_view import ParticipantView
from bin.screen import ScreenGeometry
from tests.test_active_foraging_fixation import Display, load_presenter, Stimulus
from tests.test_active_foraging_timing import FakeClock


def load_task(name):
    utils, psychopy = load_presenter(FakeClock())
    psychopy.logging = types.SimpleNamespace(console=Mock(), CRITICAL=50)
    path = Path(__file__).resolve().parents[1] / "task" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_{name}_under_test", path)
    task = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"psychopy": psychopy, "bin.utils": utils}), patch.object(bin, "utils", utils, create=True):
        spec.loader.exec_module(task)
    return task, utils


class RelatedAFCTaskTests(unittest.TestCase):
    def test_delayed_presenter_uses_participant_touch_and_option_positions(self):
        for rotation in (0, 90, 180, 270):
            with self.subTest(rotation=rotation):
                clock = FakeClock()
                utils, psychopy = load_presenter(clock)
                window = Display(clock)
                view = ParticipantView(rotation)
                cue_position = (50, -150)
                positions = [(-100, 60), (100, -60)]
                # The older CSC1 artwork keeps its existing native orientation.
                # Touch inside a rectangular image, away from its center.
                dx, dy = view.from_native((15, 40))
                touch_position = (positions[0][0] + dx, positions[0][1] + dy)
                mouse = psychopy.event.Mouse.return_value

                def pressed(getTime=False):
                    elapsed = clock.seconds() - 10
                    buttons = (.04 <= elapsed < .06 or .14 <= elapsed < .16, False, False)
                    return (buttons, (0, 0, 0)) if getTime else buttons

                mouse.getPressed.side_effect = pressed
                mouse.getPos.side_effect = lambda: view.to_native(
                    cue_position if clock.seconds() < 10.08 else touch_position
                )
                images = {(14, None): Image.new("RGB", (40, 100)),
                          (15, None): Image.new("RGB", (40, 100))}
                with patch.dict(sys.modules, {"psychopy": psychopy}), patch(
                    "time.perf_counter", side_effect=clock.seconds,
                ), patch("time.perf_counter_ns", side_effect=lambda: clock.ns), patch(
                    "time.sleep", side_effect=clock.sleep,
                ), patch.object(utils, "make_image_stim_from_array", side_effect=lambda *a, **kw: Stimulus(
                    win=window, size=(40, 100), is_image=True,
                )):
                    aborted, choice = utils.present_delayed_afc_trial(
                        win=window, preloaded=images, trial_options=list(images),
                        positions=positions, target_index=1,
                        trial_type="shape_to_shape", cue_time=.05, delay_time=0,
                        choice_time=.5, bg_rect=Stimulus(), fix=None, logger=Mock(),
                        trial_num=1, fps=60, stimulus_rotation_degrees=rotation,
                        onset_cue=view.stimulus(Stimulus(size=(100, 100)), pos=cue_position),
                        external_abort_checker=lambda: clock.seconds() > 11,
                    )
                self.assertFalse(aborted)
                self.assertEqual(choice["chosen_pos"], positions[0])
                self.assertEqual((choice["touch_x"], choice["touch_y"]), touch_position)
                self.assertTrue(any(view.to_native(positions[0]) in scene for scene in window.scenes))

    def test_sequence_and_csc1_finish_only_after_registered_choices(self):
        for name in ("afc_trial_sequence", "afc_csc1"):
            with self.subTest(task=name):
                task, utils = load_task(name)
                logs = Mock()
                preview = Mock()
                preview.poll.return_value = False
                preview.consume_manual_reward_request.return_value = False
                files = [Path("a.png"), Path("b.png")]
                choices = (None, 1, None, 2)
                attempts = []

                def present(**kwargs):
                    attempts.append(kwargs)
                    index = choices[kwargs["trial_num"] - 1]
                    kwargs["trial_meta"].update(gray_flip_perf_s=0, gray_flip_perf_ns=0)
                    return False, ({"chosen_index": index, "is_correct": index == 1} if index is not None else None)

                replacements = {
                    "setup_task_window": Mock(return_value=(
                        types.SimpleNamespace(size=(800, 600)),
                        ScreenGeometry(0, 0, 0, 800, 600, rotation="normal"),
                        ScreenGeometry(1, 800, 0, 800, 600),
                    )),
                    "resolve_frame_rate": Mock(return_value=(60, 1 / 60)),
                    "make_bg_rect": Mock(), "make_fixation_cross": Mock(return_value=None),
                    "make_onset_cue_stim": Mock(return_value=Stimulus()),
                    "configure_sync_bar": Mock(), "service_sync_bar": Mock(),
                    "finish_block_sync": Mock(), "close_task_window": Mock(),
                    "verify_task_window_vblank": Mock(),
                    "serialize_preview_image": Mock(return_value=None),
                    "find_image_files": Mock(return_value=files),
                    "load_image_assets": Mock(return_value={p: Image.new("RGB", (40, 20)) for p in files}),
                    "load_color_palette": Mock(return_value={1: (128, 128, 128), 2: (255, 0, 0)}),
                    "load_shape_definitions": Mock(return_value={i: Path("shape.svg") for i in range(14, 28)}),
                    "rasterize_svg_with_color": Mock(return_value=Image.new("RGB", (40, 20))),
                    "present_trial_with_persistent_dots": Mock(side_effect=present),
                    "present_delayed_afc_trial": Mock(side_effect=present),
                }
                with ExitStack() as stack:
                    for attr, value in replacements.items():
                        stack.enter_context(patch.object(utils, attr, value))
                    stack.enter_context(patch.object(task, "SessionLogBundle", return_value=logs))
                    stack.enter_context(patch.object(task, "ExperimenterPreview", return_value=preview))
                    if name == "afc_trial_sequence":
                        result = task.run_task(
                            "images", n_trials=2, num_afc=2, duration=.1, choice_time=.5, iti=0, isi=0,
                            dot_size=10, dot_color=(50, 50, 50), bg=(128, 128, 128), output_dir="unused",
                        )
                        self.assertFalse(result)
                    else:
                        stack.enter_context(patch.object(task, "serialize_preview_image", return_value=None))
                        result = task.run_task(
                            colors_tsv="colors", shapes_tsv="shapes", n_trials=2, num_afc=2,
                            cue_time=.1, delay_time=0, choice_time=.5, iti=0, output_dir="unused",
                            image_size=(40, 20), trial_type="shape_to_shape", timeout_time=0,
                        )
                        self.assertEqual(result, "done")
                self.assertEqual([a["trial_num"] for a in attempts], [1, 2, 3, 4])
                rows = [c.args[0] for c in logs.behavior_logger.writerow.call_args_list]
                self.assertEqual([r["trial_num"] for r in rows], [1, 2, 3, 4])
                self.assertEqual([r["choice_made_index"] for r in rows], ["", 0, "", 1])
                preview.set_trial_progress.assert_called_with(4, 2, completed_trials=2)


if __name__ == "__main__":
    unittest.main()
