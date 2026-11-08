"""Integrate a sync bar with sparse PsychoPy scene presentation.

Sync-only swaps preserve the pending task scene entirely on the GPU. They also
isolate pending callOnFlip/autoDraw work, so an upcoming task transition cannot
accidentally happen on an earlier sync refresh.
"""
from __future__ import annotations

import atexit
import ctypes
import math
import time

from . import frame_timing
from .sync_bar import SyncBar, SyncBarConfig
from .video_playback import is_raspberry_pi


class PendingSceneBuffer:
    """One reusable GPU buffer; no screenshots, readback, or texture uploads."""

    def __init__(self, win):
        from pyglet import gl

        self.gl = gl
        self.win = win
        self.width, self.height = map(int, win.size)
        self.framebuffer = gl.GLuint()
        self.renderbuffer = gl.GLuint()
        win._setCurrent()
        self.target = getattr(win, "frameBuffer", 0) if win.useFBO else 0
        self.target_buffer = gl.GL_COLOR_ATTACHMENT0 if win.useFBO else gl.GL_BACK
        try:
            gl.glGenFramebuffers(1, ctypes.byref(self.framebuffer))
            gl.glGenRenderbuffers(1, ctypes.byref(self.renderbuffer))
            gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, self.renderbuffer)
            gl.glRenderbufferStorage(gl.GL_RENDERBUFFER, gl.GL_RGBA8, self.width, self.height)
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.framebuffer)
            gl.glFramebufferRenderbuffer(
                gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0,
                gl.GL_RENDERBUFFER, self.renderbuffer,
            )
            if gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) != gl.GL_FRAMEBUFFER_COMPLETE:
                raise RuntimeError("Sync bar requires a complete GPU scene buffer")
        except BaseException:
            self.close()
            raise
        finally:
            gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, 0)
            self._bind_task_buffer()

    def _bind_task_buffer(self):
        gl = self.gl
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.target)
        gl.glReadBuffer(self.target_buffer)
        gl.glDrawBuffer(self.target_buffer)

    def _copy(self, source, source_buffer, target, target_buffer):
        gl = self.gl
        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, source)
        gl.glReadBuffer(source_buffer)
        gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, target)
        gl.glDrawBuffer(target_buffer)
        gl.glBlitFramebuffer(
            0, 0, self.width, self.height, 0, 0, self.width, self.height,
            gl.GL_COLOR_BUFFER_BIT, gl.GL_NEAREST,
        )

    def save_and_show_front(self):
        gl = self.gl
        self.win._setCurrent()
        self._scissor = bool(gl.glIsEnabled(gl.GL_SCISSOR_TEST))
        self._scissor_box = (gl.GLint * 4)()
        gl.glGetIntegerv(gl.GL_SCISSOR_BOX, self._scissor_box)
        gl.glDisable(gl.GL_SCISSOR_TEST)
        self._copy(self.target, self.target_buffer, self.framebuffer, gl.GL_COLOR_ATTACHMENT0)
        self._copy(0, gl.GL_FRONT, self.target, self.target_buffer)
        self._bind_task_buffer()

    def restore(self):
        gl = self.gl
        gl.glDisable(gl.GL_SCISSOR_TEST)
        self._copy(self.framebuffer, gl.GL_COLOR_ATTACHMENT0, self.target, self.target_buffer)
        self._bind_task_buffer()
        gl.glScissor(*self._scissor_box)
        if self._scissor:
            gl.glEnable(gl.GL_SCISSOR_TEST)
        else:
            gl.glDisable(gl.GL_SCISSOR_TEST)

    def close(self):
        self.win._setCurrent()
        gl = self.gl
        gl.glDeleteFramebuffers(1, ctypes.byref(self.framebuffer))
        gl.glDeleteRenderbuffers(1, ctypes.byref(self.renderbuffer))


class SyncDisplay:
    def __init__(self, win, bar, frame_duration_s, *, scene_buffer=None, gpio_module=None, gpio_chip=None):
        self.win = win
        self.bar = bar
        self.frame_duration_s = float(frame_duration_s)
        if not math.isfinite(self.frame_duration_s) or self.frame_duration_s <= 0:
            raise ValueError("frame_duration_s must be positive and finite")
        self.buffer = PendingSceneBuffer(win) if scene_buffer is None else scene_buffer
        self.gpio_module, self.gpio_chip = gpio_module, gpio_chip
        self.original_flip = win.flip
        self.last_flip_perf_s = None
        self.extra_flip_count = 0
        self.closed = False
        win.flip = self.flip
        atexit.register(self.close)

    def _capture_flip(self):
        self.last_flip_perf_s = time.perf_counter()

    def _next_refresh(self, now):
        if self.last_flip_perf_s is None:
            return now + self.frame_duration_s
        frames = max(1, math.floor((now - self.last_flip_perf_s) / self.frame_duration_s) + 1)
        return self.last_flip_perf_s + frames * self.frame_duration_s

    def flip(self, *args, **kwargs):
        queued = len(self.win._toCall)
        self.bar.prepare_flip(self._next_refresh(time.perf_counter()) + self.frame_duration_s / 2)
        self.win.callOnFlip(self._capture_flip)
        # Keep the display timestamp and GPIO edge ahead of task callbacks
        # (which may reset input devices or do more expensive bookkeeping).
        sync_callbacks = self.win._toCall[queued:]
        self.win._toCall[queued:] = []
        captures = [item for item in self.win._toCall
                    if item["function"] is frame_timing.capture_perf_counter_on_flip]
        self.win._toCall[:] = [item for item in self.win._toCall
                              if item["function"] is not frame_timing.capture_perf_counter_on_flip]
        self.win._toCall[:0] = captures + sync_callbacks[-1:] + sync_callbacks[:-1]
        result = self.original_flip(*args, **kwargs)
        self.bar.flush_events()
        return result

    def service(self, *, deadline_perf_s=None):
        if self.closed or self.last_flip_perf_s is None:
            return
        target = self.bar.next_flip_perf_s
        if target is None:
            return
        now = time.perf_counter()
        if now < target - frame_timing.DEFAULT_FLIP_SUBMISSION_LEAD_FRACTION * self.frame_duration_s:
            return
        refresh = self._next_refresh(now)
        # Reserve a full refresh before task/input/reward deadlines for task
        # preparation. A due bar inversion joins the ordinary task swap.
        if deadline_perf_s is not None and refresh + self.frame_duration_s >= deadline_perf_s:
            return
        pending = {}
        for name in ("_toCall", "_toLog", "_toDraw", "_toDrawDepths"):
            if hasattr(self.win, name):
                pending[name] = getattr(self.win, name)
                setattr(self.win, name, [])
        try:
            self.buffer.save_and_show_front()
            try:
                self.flip()
                self.extra_flip_count += 1
            finally:
                self.buffer.restore()
        finally:
            for name, value in pending.items():
                setattr(self.win, name, value)

    def close(self):
        if self.closed:
            return
        self.closed = True
        atexit.unregister(self.close)
        self.win.flip = self.original_flip
        frame_timing.set_background_presenter(None)
        try:
            self.bar.close()
        finally:
            try:
                self.buffer.close()
            finally:
                if self.gpio_module is not None:
                    self.gpio_module.gpiochip_close(self.gpio_chip)


def configure_sync_bar(
    win, config=None, *, sync_pin=18, event_logger=None,
    experimenter_preview=None, rotation_degrees=0, frame_duration_s=1 / 60,
    msg_logger=None,
):
    config = SyncBarConfig.from_mapping(config, sync_pin=sync_pin)
    sync_pin = config.sync_pin
    if not config.enabled:
        return None
    if getattr(win, "_neuro_tasks_sync_display", None) is not None:
        raise RuntimeError("Sync bar is already configured for this window")
    gpio, chip = None, None
    if sync_pin is not None:
        try:
            import lgpio
            chip = lgpio.gpiochip_open(0)
            if chip < 0:
                raise RuntimeError(f"gpiochip_open returned {chip}")
            gpio = lgpio
            result = gpio.gpio_claim_output(chip, sync_pin, 0)
            if result < 0:
                raise RuntimeError(f"gpio_claim_output returned {result}")
        except Exception as exc:
            if gpio is not None:
                gpio.gpiochip_close(chip)
            gpio, chip = None, None
            if is_raspberry_pi():
                raise RuntimeError(
                    f"Sync bar requires GPIO output on Raspberry Pi; "
                    f"could not initialize BCM pin {sync_pin}: {exc}"
                ) from exc
            if msg_logger is not None:
                msg_logger.log("WARN", f"sync_gpio_unavailable pin={sync_pin} error={exc}")
        else:
            if msg_logger is not None:
                msg_logger.log("INFO", f"sync_gpio_ready pin={sync_pin} chip=0")
    try:
        preview_callback = None
        if experimenter_preview is not None:
            experimenter_preview.configure_sync_bar(config.height, win.size, rotation_degrees)
            preview_callback = experimenter_preview.set_sync_bar_phase
        bar = SyncBar(
            win, config, rotation_degrees=rotation_degrees,
            event_logger=event_logger, gpio_module=gpio, gpio_chip=chip,
            sync_pin=sync_pin, preview_callback=preview_callback,
        )
        display = SyncDisplay(win, bar, frame_duration_s, gpio_module=gpio, gpio_chip=chip)
    except BaseException:
        if gpio is not None:
            gpio.gpiochip_close(chip)
        raise
    win._neuro_tasks_sync_display = display
    frame_timing.set_background_presenter(display)
    return display
