import ctypes
import sys
import types
import unittest
from unittest.mock import Mock, patch

from bin import frame_timing
from bin.frame_timing import arm_flip_timestamp
from bin.sync_bar import SyncBar, SyncBarConfig
from bin.sync_display import PendingSceneBuffer, SyncDisplay, configure_sync_bar
from bin.touch_input import MousePressTracker


class FakeWindow:
    size = (1000, 600)

    def __init__(self):
        self._toCall = []
        self._toLog = ["pending log"]
        self._toDraw = ["pending autoDraw"]
        self._toDrawDepths = [0]
        self.scene = "upcoming task scene"
        self.front = "currently visible scene"

    def callOnFlip(self, callback, *args):
        self._toCall.append({"function": callback, "args": args, "kwargs": {}})

    def flip(self):
        self.front = self.scene
        callbacks, self._toCall = self._toCall, []
        for callback in callbacks:
            callback["function"](*callback["args"], **callback["kwargs"])
        self.scene = "cleared back buffer"
        return 42.0


class FakeSceneBuffer:
    def __init__(self, win):
        self.win = win
        self.closed = False

    def save_and_show_front(self):
        self.pending_scene = self.win.scene
        self.win.scene = self.win.front

    def restore(self):
        self.win.scene = self.pending_scene

    def close(self):
        self.closed = True


class PendingSceneBufferTests(unittest.TestCase):
    def test_gpu_copies_preserve_pending_scene_and_scissor_for_fbo_and_backbuffer(self):
        for use_fbo in (False, True):
            for scissor_enabled in (False, True):
                with self.subTest(use_fbo=use_fbo, scissor_enabled=scissor_enabled):
                    self._check_gpu_round_trip(use_fbo, scissor_enabled)

    def _check_gpu_round_trip(self, use_fbo, scissor_enabled):
        names = ("GL_FRAMEBUFFER", "GL_RENDERBUFFER", "GL_RGBA8", "GL_COLOR_ATTACHMENT0",
                 "GL_FRAMEBUFFER_COMPLETE", "GL_BACK", "GL_FRONT", "GL_READ_FRAMEBUFFER",
                 "GL_DRAW_FRAMEBUFFER", "GL_COLOR_BUFFER_BIT", "GL_NEAREST",
                 "GL_SCISSOR_TEST", "GL_SCISSOR_BOX")
        gl = types.SimpleNamespace(**{name: index + 1 for index, name in enumerate(names)})
        gl.GLuint, gl.GLint = ctypes.c_uint, ctypes.c_int
        state = {"scissor": scissor_enabled, "box": (13, 17, 300, 200)}
        buffers, copies = {}, []

        def identifier(value):
            return value.value if hasattr(value, "value") else value

        def bind(target, framebuffer):
            framebuffer = identifier(framebuffer)
            if target in (gl.GL_FRAMEBUFFER, gl.GL_READ_FRAMEBUFFER):
                state["read"] = framebuffer
            if target in (gl.GL_FRAMEBUFFER, gl.GL_DRAW_FRAMEBUFFER):
                state["draw"] = framebuffer

        def blit(*args):
            self.assertFalse(state["scissor"], "A scissor-clipped blit can erase pending task pixels")
            source = (state["read"], state["read_buffer"])
            target = (state["draw"], state["draw_buffer"])
            copies.append((source, target, args))
            buffers[target] = buffers[source]

        def get_integers(key, destination):
            self.assertEqual(key, gl.GL_SCISSOR_BOX)
            values = ctypes.cast(destination, ctypes.POINTER(ctypes.c_int))
            for index, value in enumerate(state["box"]):
                values[index] = value

        gl.glGenFramebuffers = Mock(side_effect=lambda n, pointer: setattr(pointer._obj, "value", 41))
        gl.glGenRenderbuffers = Mock(side_effect=lambda n, pointer: setattr(pointer._obj, "value", 42))
        gl.glBindFramebuffer = Mock(side_effect=bind)
        gl.glBindRenderbuffer = Mock()
        gl.glRenderbufferStorage = Mock()
        gl.glFramebufferRenderbuffer = Mock()
        gl.glCheckFramebufferStatus = Mock(return_value=gl.GL_FRAMEBUFFER_COMPLETE)
        gl.glReadBuffer = Mock(side_effect=lambda value: state.__setitem__("read_buffer", value))
        gl.glDrawBuffer = Mock(side_effect=lambda value: state.__setitem__("draw_buffer", value))
        gl.glBlitFramebuffer = Mock(side_effect=blit)
        gl.glIsEnabled = Mock(side_effect=lambda flag: state["scissor"])
        gl.glGetIntegerv = Mock(side_effect=get_integers)
        gl.glDisable = Mock(side_effect=lambda flag: state.__setitem__("scissor", False))
        gl.glEnable = Mock(side_effect=lambda flag: state.__setitem__("scissor", True))
        gl.glScissor = Mock(side_effect=lambda *box: state.__setitem__("box", box))
        gl.glDeleteFramebuffers, gl.glDeleteRenderbuffers = Mock(), Mock()
        win = types.SimpleNamespace(size=(1000, 600), useFBO=use_fbo, frameBuffer=17,
                                    _setCurrent=Mock())
        task_buffer = (17, gl.GL_COLOR_ATTACHMENT0) if use_fbo else (0, gl.GL_BACK)
        scratch_buffer = (41, gl.GL_COLOR_ATTACHMENT0)
        front_buffer = (0, gl.GL_FRONT)
        buffers[task_buffer], buffers[front_buffer] = "future trial scene", "visible trial scene"
        with patch.dict(sys.modules, {"pyglet": types.SimpleNamespace(gl=gl)}):
            buffer = PendingSceneBuffer(win)
        buffer.save_and_show_front()
        self.assertEqual(buffers[task_buffer], "visible trial scene")
        # PsychoPy's swap changes the active scissor and clears its back buffer.
        state.update(scissor=True, box=(0, 0, 1, 1))
        buffers[task_buffer] = "cleared after sync swap"
        buffer.restore()
        self.assertEqual(buffers[task_buffer], "future trial scene")
        self.assertEqual(state["scissor"], scissor_enabled)
        self.assertEqual(state["box"], (13, 17, 300, 200))
        self.assertEqual([(source, target) for source, target, _ in copies],
                         [(task_buffer, scratch_buffer), (front_buffer, task_buffer),
                          (scratch_buffer, task_buffer)])
        self.assertTrue(all(args[:8] == (0, 0, 1000, 600, 0, 0, 1000, 600)
                            for _, _, args in copies))
        self.assertEqual((state["read"], state["read_buffer"]), task_buffer)
        self.assertEqual((state["draw"], state["draw_buffer"]), task_buffer)
        buffer.close()
        gl.glDeleteFramebuffers.assert_called_once()
        gl.glDeleteRenderbuffers.assert_called_once()


class SyncDisplayTests(unittest.TestCase):
    def setUp(self):
        self.win = FakeWindow()
        self.original_flip = self.win.flip
        self.buffer = FakeSceneBuffer(self.win)
        self.bar = Mock()
        self.bar.next_flip_perf_s = 11.0
        self.display = SyncDisplay(self.win, self.bar, 0.1, scene_buffer=self.buffer)
        self.display.last_flip_perf_s = 10.9

    def tearDown(self):
        self.display.close()

    def test_sparse_swap_preserves_upcoming_scene_and_task_callbacks(self):
        task_callback = Mock()
        self.win.callOnFlip(task_callback)
        pending = {name: getattr(self.win, name) for name in
                   ("_toCall", "_toLog", "_toDraw", "_toDrawDepths")}
        with patch("bin.sync_display.time.perf_counter", return_value=10.96):
            self.display.service()
        self.assertEqual(self.win.front, "currently visible scene")
        self.assertEqual(self.win.scene, "upcoming task scene")
        task_callback.assert_not_called()
        for name, value in pending.items():
            self.assertIs(getattr(self.win, name), value)
        self.assertEqual(self.display.extra_flip_count, 1)
        self.bar.flush_events.assert_called_once_with()
        with patch("bin.sync_display.time.perf_counter", return_value=11.01):
            self.assertEqual(self.win.flip(), 42.0)
        self.assertEqual(self.win.front, "upcoming task scene")
        task_callback.assert_called_once_with()

    def test_coincident_task_deadline_does_not_consume_extra_refresh(self):
        with patch("bin.sync_display.time.perf_counter", return_value=10.96):
            self.display.service(deadline_perf_s=11.0)
        self.assertEqual(self.display.extra_flip_count, 0)
        self.bar.prepare_flip.assert_not_called()
        self.assertEqual(self.win.scene, "upcoming task scene")

    def test_adjacent_task_refresh_retains_preparation_time(self):
        with patch("bin.sync_display.time.perf_counter", return_value=10.96):
            self.display.service(deadline_perf_s=11.1)
        self.assertEqual(self.display.extra_flip_count, 0)
        self.bar.prepare_flip.assert_not_called()
        self.assertEqual(self.win.front, "currently visible scene")
        self.assertEqual(self.win.scene, "upcoming task scene")

    def test_deferred_bar_and_gpio_edges_share_actual_task_transition(self):
        now = [10_000_000_000]
        writes = []
        logger = Mock()

        def gpio_write(chip, pin, level):
            # FakeWindow publishes the scene before invoking flip callbacks.
            self.assertEqual(self.win.front, f"task phase {level}")
            writes.append((now[0], chip, pin, level))
            now[0] += 500
            return 0

        bar = SyncBar(
            self.win, SyncBarConfig(True, 20, 1, 0),
            rect_factory=lambda *a, **k: Mock(), clock=lambda: now[0],
            gpio_module=types.SimpleNamespace(gpio_write=gpio_write),
            gpio_chip=0, sync_pin=18, event_logger=logger,
        )
        bar.start()
        self.display.bar = bar

        with patch("bin.sync_display.time.perf_counter", side_effect=lambda: now[0] / 1e9):
            for requested_ns, actual_ns, level in (
                (11_000_000_000, 11_100_000_000, 1),
                (12_100_000_000, 12_200_000_000, 0),
            ):
                with self.subTest(level=level):
                    self.assertEqual(round(bar.next_flip_perf_s * 1e9), requested_ns)
                    self.win.scene = f"task phase {level}"
                    previous_count = len(writes)
                    # Reserve the next task refresh, including after the
                    # original sync deadline has arrived.
                    for poll_ns in (requested_ns - 40_000_000, requested_ns):
                        now[0] = poll_ns
                        self.display.service(deadline_perf_s=actual_ns / 1e9)
                        self.assertEqual(len(writes), previous_count)
                        self.assertEqual(logger.log_event.call_count, 2 * previous_count)
                        self.assertEqual(bar.phase, level ^ 1)

                    def task_swap():
                        # Preparing/drawing the bar cannot send an early edge.
                        self.assertEqual(len(writes), previous_count)
                        now[0] = actual_ns
                        return self.original_flip()

                    self.display.original_flip = task_swap
                    now[0] = actual_ns - 40_000_000
                    self.win.flip()
                    self.assertEqual(writes[-1], (actual_ns, 0, 18, level))
                    self.assertEqual(len(writes), previous_count + 1)
                    self.assertEqual(bar.phase, level)
                    flip_event, signal_event = [
                        call.kwargs for call in logger.log_event.call_args_list[-2:]
                    ]
                    self.assertEqual(flip_event["event"], "sync_bar_flip")
                    self.assertEqual(flip_event["requested_timestamp_perf_ns"], requested_ns)
                    self.assertEqual(flip_event["timestamp_perf_ns"], actual_ns)
                    self.assertEqual(signal_event["event"], "sync_signal_on" if level else "sync_signal_off")
                    self.assertEqual(signal_event["requested_timestamp_perf_ns"], actual_ns)
                    self.assertEqual(signal_event["timestamp_perf_ns"], actual_ns + 500)
        self.assertEqual(self.display.extra_flip_count, 0)

    def test_distant_task_refresh_allows_sync_only_swap(self):
        with patch("bin.sync_display.time.perf_counter", return_value=10.96):
            self.display.service(deadline_perf_s=11.2)
        self.assertEqual(self.display.extra_flip_count, 1)
        self.bar.prepare_flip.assert_called_once()

    def test_no_swap_before_submission_window(self):
        with patch("bin.sync_display.time.perf_counter", return_value=10.94):
            self.display.service()
        self.bar.prepare_flip.assert_not_called()
        self.assertEqual(self.display.extra_flip_count, 0)

    def test_refresh_phase_uses_next_refresh_after_elapsed_idle_frames(self):
        self.assertAlmostEqual(self.display._next_refresh(11.11), 11.2)
        self.assertAlmostEqual(self.display._next_refresh(10.91), 11.0)

    def test_request_is_rounded_to_nearest_refresh_including_early_refresh(self):
        now = [10_000_000_000]
        bar = SyncBar(self.win, SyncBarConfig(True, 20, 1.04, 0),
                      rect_factory=lambda *a, **k: Mock(), clock=lambda: now[0])
        bar.start()
        self.display.bar = bar
        now[0] = 10_995_000_000
        original = self.display.original_flip

        def swap():
            now[0] = 11_000_000_000
            return original()

        self.display.original_flip = swap
        with patch("bin.sync_display.time.perf_counter", side_effect=lambda: now[0] / 1e9):
            self.display.service()
        self.assertEqual(bar.phase, 1)
        self.assertAlmostEqual(bar.next_flip_perf_s, 12.04)
        self.assertEqual(self.display.last_flip_perf_s, 11)

    def test_timestamp_and_gpio_precede_slow_task_callbacks(self):
        now = [10_000_000_000]
        gpio = Mock()

        def gpio_write(*args):
            now[0] += 2_000_000
            return 0

        gpio.gpio_write.side_effect = gpio_write
        bar = SyncBar(self.win, SyncBarConfig(True, 20, 1, 0),
                      rect_factory=lambda *a, **k: Mock(), clock=lambda: now[0],
                      gpio_module=gpio, gpio_chip=0, sync_pin=18)
        bar.start()
        self.display.bar = bar
        self.win.callOnFlip(lambda: now.__setitem__(0, now[0] + 30_000_000))
        capture = arm_flip_timestamp(self.win)
        original = self.display.original_flip

        def swap():
            now[0] = 11_000_000_000
            return original()

        self.display.original_flip = swap
        now[0] = 10_960_000_000
        with (
            patch("bin.sync_display.time.perf_counter", side_effect=lambda: now[0] / 1e9),
            patch("bin.frame_timing.time.perf_counter_ns", side_effect=lambda: now[0]),
        ):
            self.win.flip()
        self.assertEqual(capture["actual_perf_ns"], 11_000_000_000)
        self.assertEqual(self.display.last_flip_perf_s, 11)
        self.assertEqual(bar.next_flip_perf_s, 12)
        gpio.gpio_write.assert_called_once_with(0, 18, 1)
        self.assertEqual(now[0], 11_032_000_000)

    def test_video_frame_swap_includes_due_bar_transition_without_extra_swap(self):
        now = [10_000_000_000]
        bar = SyncBar(self.win, SyncBarConfig(True, 20, 1, 0),
                      rect_factory=lambda *a, **k: Mock(), clock=lambda: now[0])
        bar.start()
        self.display.bar = bar
        original = self.display.original_flip

        def video_swap():
            now[0] = 11_000_000_000
            return original()

        self.display.original_flip = video_swap
        now[0] = 10_960_000_000
        with patch("bin.sync_display.time.perf_counter", side_effect=lambda: now[0] / 1e9):
            self.win.flip()
        self.assertEqual(bar.phase, 1)
        self.assertEqual(self.display.extra_flip_count, 0)
        self.assertEqual(self.win.front, "upcoming task scene")
        self.assertFalse(hasattr(self.buffer, "pending_scene"))

    def test_completed_short_tap_keeps_source_timestamp_across_sync_only_swap(self):
        mouse = Mock()
        mouse.getPressed.side_effect = [
            ([0, 0, 0], [0.0, 0.0, 0.0]),
            ([0, 0, 0], [0.006, 0.0, 0.0]),
            ([0, 0, 0], [0.0, 0.0, 0.0]),
        ]
        mouse.getPos.return_value = (25, -30)
        tracker = MousePressTracker(mouse)
        with patch("bin.touch_input.time.perf_counter_ns", return_value=10_940_000_000):
            tracker.reset()
        self.win.callOnFlip(tracker.reset)  # Belongs to the upcoming task scene.
        with patch("bin.sync_display.time.perf_counter", return_value=10.96):
            self.display.service()
        mouse.clickReset.assert_called_once_with()
        with patch("bin.touch_input.time.perf_counter_ns", return_value=11_000_000_000):
            sample = tracker.poll()
            next_sample = tracker.poll()
        self.assertTrue(sample.buffered_press)
        self.assertTrue(sample.press_started)
        self.assertFalse(sample.down)
        self.assertEqual(sample.timestamp_perf_ns, 10_946_000_000)
        self.assertEqual(sample.observed_perf_ns, 11_000_000_000)
        self.assertEqual(sample.position, (25, -30))
        self.assertFalse(next_sample.press_started)

    def test_failure_restores_pending_scene_and_callbacks(self):
        callback = Mock()
        self.win.callOnFlip(callback)
        pending = self.win._toCall
        self.bar.prepare_flip.side_effect = RuntimeError("drawing failed")
        with patch("bin.sync_display.time.perf_counter", return_value=10.96):
            with self.assertRaisesRegex(RuntimeError, "drawing failed"):
                self.display.service()
        self.assertIs(self.win._toCall, pending)
        self.assertEqual(self.win.scene, "upcoming task scene")
        callback.assert_not_called()

    def test_close_restores_flip_and_releases_gpio_even_if_buffer_close_fails(self):
        gpio = Mock()
        self.display.gpio_module = gpio
        self.display.gpio_chip = 9
        self.buffer.close = Mock(side_effect=RuntimeError("context lost"))
        with self.assertRaisesRegex(RuntimeError, "context lost"):
            self.display.close()
        self.assertEqual(self.win.flip, self.original_flip)
        self.bar.close.assert_called_once_with()
        gpio.gpiochip_close.assert_called_once_with(9)


class SyncDisplayConfigurationTests(unittest.TestCase):
    def test_effective_pin_is_claimed_and_passed_to_the_bar(self):
        for options, legacy_pin, expected in (
            ({"sync_pin": 23}, 24, 23), ({"sync_pin": 0}, 24, 0),
            ({"sync_pin": None}, 24, None), ({}, 24, 24), ({}, None, None), ({}, 18, 18),
        ):
            with self.subTest(options=options, legacy_pin=legacy_pin):
                gpio = types.SimpleNamespace(gpiochip_open=Mock(return_value=4),
                                             gpio_claim_output=Mock(return_value=0),
                                             gpiochip_close=Mock())
                with (
                    patch.dict(sys.modules, {"lgpio": gpio}),
                    patch("bin.sync_display.SyncDisplay"),
                    patch("bin.sync_display.SyncBar") as bar,
                    patch("bin.sync_display.frame_timing.set_background_presenter"),
                ):
                    configure_sync_bar(FakeWindow(), {"enabled": True, **options}, sync_pin=legacy_pin)
                self.assertEqual(bar.call_args.kwargs["sync_pin"], expected)
                self.assertEqual(bar.call_args.args[1].sync_pin, expected)
                if expected is None:
                    gpio.gpiochip_open.assert_not_called()
                    gpio.gpio_claim_output.assert_not_called()
                    self.assertIsNone(bar.call_args.kwargs["gpio_module"])
                else:
                    gpio.gpio_claim_output.assert_called_once_with(4, expected, 0)

    def test_disabled_never_allocates_gpu_or_gpio(self):
        with patch("bin.sync_display.SyncDisplay") as display, patch("bin.sync_display.SyncBar") as bar:
            self.assertIsNone(configure_sync_bar(object(), {"enabled": False}))
        display.assert_not_called()
        bar.assert_not_called()

    def test_unavailable_gpio_keeps_visual_sync_and_preview(self):
        win, preview, logger = FakeWindow(), Mock(), Mock()
        with (
            patch.dict(sys.modules, {"lgpio": None}),
            patch("bin.sync_display.SyncDisplay") as display,
            patch("bin.sync_display.SyncBar") as bar,
            patch("bin.sync_display.frame_timing.set_background_presenter") as install,
        ):
            result = configure_sync_bar(win, {"enabled": True}, experimenter_preview=preview,
                                        rotation_degrees=90, msg_logger=logger)
        self.assertIs(result, display.return_value)
        self.assertIsNone(bar.call_args.kwargs["gpio_module"])
        preview.configure_sync_bar.assert_called_once_with(20, win.size, 90)
        self.assertEqual(bar.call_args.kwargs["preview_callback"], preview.set_sync_bar_phase)
        logger.log.assert_called_once()
        install.assert_called_once_with(result)

    def test_failed_gpio_claim_releases_chip(self):
        gpio = types.SimpleNamespace(gpiochip_open=Mock(return_value=4),
                                     gpio_claim_output=Mock(return_value=-1),
                                     gpiochip_close=Mock())
        with (
            patch.dict(sys.modules, {"lgpio": gpio}),
            patch("bin.sync_display.SyncDisplay"),
            patch("bin.sync_display.SyncBar") as bar,
            patch("bin.sync_display.frame_timing.set_background_presenter"),
        ):
            configure_sync_bar(FakeWindow(), {"enabled": True})
        gpio.gpiochip_close.assert_called_once_with(4)
        self.assertIsNone(bar.call_args.kwargs["gpio_module"])

    def test_failed_overlay_creation_releases_claimed_gpio(self):
        gpio = types.SimpleNamespace(gpiochip_open=Mock(return_value=4),
                                     gpio_claim_output=Mock(return_value=0),
                                     gpiochip_close=Mock())
        with (
            patch.dict(sys.modules, {"lgpio": gpio}),
            patch("bin.sync_display.SyncBar", side_effect=RuntimeError("unsupported display")),
        ):
            with self.assertRaisesRegex(RuntimeError, "unsupported display"):
                configure_sync_bar(FakeWindow(), {"enabled": True})
        gpio.gpiochip_close.assert_called_once_with(4)

    def test_driver_specific_exception_keeps_visual_sync(self):
        class DriverError(Exception):
            pass

        gpio = types.SimpleNamespace(gpiochip_open=Mock(return_value=4),
                                     gpio_claim_output=Mock(side_effect=DriverError("pin busy")),
                                     gpiochip_close=Mock())
        logger = Mock()
        with (
            patch.dict(sys.modules, {"lgpio": gpio}),
            patch("bin.sync_display.SyncDisplay"),
            patch("bin.sync_display.SyncBar") as bar,
            patch("bin.sync_display.frame_timing.set_background_presenter"),
        ):
            configure_sync_bar(FakeWindow(), {"enabled": True}, msg_logger=logger)
        gpio.gpiochip_close.assert_called_once_with(4)
        self.assertIsNone(bar.call_args.kwargs["gpio_module"])
        self.assertIn("pin busy", logger.log.call_args.args[1])


class BackgroundPresenterDeadlineTests(unittest.TestCase):
    def tearDown(self):
        frame_timing.set_background_presenter(None)

    def test_poll_callback_cannot_override_the_earlier_task_deadline(self):
        presenter = Mock()
        frame_timing.set_background_presenter(presenter)

        def cancel():
            frame_timing.service_background_presenter(deadline_perf_s=20.0)
            return True

        self.assertFalse(frame_timing.wait_until(10.0, poll_callback=cancel))
        self.assertEqual([call.kwargs["deadline_perf_s"] for call in presenter.service.call_args_list],
                         [10.0])
        frame_timing.service_background_presenter(deadline_perf_s=20.0)
        presenter.service.assert_called_with(deadline_perf_s=20.0)

    def test_cancelled_task_poll_does_not_submit_a_sync_only_swap(self):
        presenter = Mock()
        frame_timing.set_background_presenter(presenter)
        self.assertFalse(frame_timing.wait_until(10.0, poll_callback=lambda: True))
        presenter.service.assert_not_called()


if __name__ == "__main__":
    unittest.main()
