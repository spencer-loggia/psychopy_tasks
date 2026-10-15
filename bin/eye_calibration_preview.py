"""Interactive calibration renderer hosted by the shared ExperimenterPreview process.

Only this process owns the secondary window and its mouse. Control values cross
back as one atomic snapshot; rendering updates use the preview's latest-only
mailbox. There is no UART or reward hardware access in this component.
"""

from __future__ import annotations

import math
import time
from typing import Sequence, Tuple
from psychopy import event, visual
from bin import utils
from bin.eye_calibration_keys import (
    CALIBRATION_KEYS,
    FIXATION_STEP_FRACTION,
    fixation_key_steps,
)
from bin.eye_tracking import clamp_fraction
from bin.preview_text import set_preview_text
from bin.screen import compute_centered_aspect_fit
from bin.video_playback import (
    SharedVideoFrameReader, make_rgb_video_stim, upload_rgb_texture,
)


class Slider:
    def __init__(
        self,
        win,
        *,
        orientation: str,
        pos: Tuple[float, float],
        length: float,
        value_min: float,
        value_max: float,
        value: float,
        label: str,
    ):
        self.win = win
        self.orientation = orientation
        self.pos = (float(pos[0]), float(pos[1]))
        self.length = max(1.0, float(length))
        self.value_min = float(value_min)
        self.value_max = float(value_max)
        self.set_value(value)
        self.label = str(label)
        self.track = visual.Rect(
            win,
            width=self.length if self.orientation == "horizontal" else 6.0,
            height=6.0 if self.orientation == "horizontal" else self.length,
            pos=self.pos,
            fillColor=utils.rgb255_to_psychopy((70, 70, 70)),
            fillColorSpace="rgb",
            lineColor=utils.rgb255_to_psychopy((210, 210, 210)),
            lineColorSpace="rgb",
            units="pix",
            autoLog=False,
        )
        self.knob = visual.Rect(
            win,
            width=24.0,
            height=24.0,
            pos=self._knob_pos(),
            fillColor=utils.rgb255_to_psychopy((245, 245, 245)),
            fillColorSpace="rgb",
            lineColor=utils.rgb255_to_psychopy((30, 30, 30)),
            lineColorSpace="rgb",
            units="pix",
            autoLog=False,
        )
        self.text = visual.TextStim(
            win,
            text="",
            units="pix",
            height=18.0,
            color=utils.rgb255_to_psychopy((245, 245, 245)),
            colorSpace="rgb",
            pos=(0, 0),
            autoLog=False,
        )

    def set_value(self, value: float) -> None:
        """Display fitted gains exactly, extending the manual range if needed."""
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("Calibration gain must be finite")
        self.value_min = min(self.value_min, value * 1.1 if value < 0 else value)
        self.value_max = max(self.value_max, value * 1.1 if value > 0 else value)
        self.value = value

    def _fraction(self) -> float:
        span = self.value_max - self.value_min
        if span == 0:
            return 0.5
        return max(0.0, min(1.0, (float(self.value) - self.value_min) / span))

    def _knob_pos(self) -> Tuple[float, float]:
        frac = self._fraction()
        if self.orientation == "horizontal":
            return (
                self.pos[0] - (self.length * 0.5) + (frac * self.length),
                self.pos[1],
            )
        return (self.pos[0], self.pos[1] - (self.length * 0.5) + (frac * self.length))

    def contains(self, point: Sequence[float]) -> bool:
        px, py = float(point[0]), float(point[1])
        kx, ky = self._knob_pos()
        if abs(px - kx) <= 24.0 and abs(py - ky) <= 24.0:
            return True
        if self.orientation == "horizontal":
            return (
                abs(py - self.pos[1]) <= 18.0
                and abs(px - self.pos[0]) <= self.length * 0.5
            )
        return (
            abs(px - self.pos[0]) <= 18.0 and abs(py - self.pos[1]) <= self.length * 0.5
        )

    def set_from_pos(self, point: Sequence[float]) -> None:
        if self.orientation == "horizontal":
            frac = (float(point[0]) - (self.pos[0] - self.length * 0.5)) / self.length
        else:
            frac = (float(point[1]) - (self.pos[1] - self.length * 0.5)) / self.length
        frac = max(0.0, min(1.0, frac))
        self.value = self.value_min + (frac * (self.value_max - self.value_min))

    def nudge(self, steps: int) -> bool:
        """Adjust gain by 0.0001 per step, clamping to the slider limits."""
        value = max(self.value_min, min(self.value_max, self.value + steps * 0.0001))
        changed = value != self.value
        self.value = value
        return changed

    def draw(self) -> None:
        self.knob.pos = self._knob_pos()
        self.track.draw()
        self.knob.draw()
        if self.orientation == "horizontal":
            label_text = f"{self.label} {self.value:.4f}"
            self.text.pos = (self.pos[0], self.pos[1] + 30.0)
        else:
            label_text = f"{self.label}\n{self.value:.4f}"
            self.text.pos = (self.pos[0], self.pos[1] + self.length / 2 + 35)
        # Assign only the final string: the vertical slider previously toggled
        # single/multiline text every frame even when the value was unchanged.
        set_preview_text(self.text, label_text)
        self.text.draw()


def _make_fixation_cross(win, *, size: float, weight: float, color_rgb: Sequence[int]):
    size = max(1.0, float(size))
    weight = max(1.0, min(size, float(weight)))
    color = utils.rgb255_to_psychopy(tuple(int(v) for v in color_rgb))
    horizontal = visual.Rect(
        win,
        width=size,
        height=weight,
        fillColor=color,
        fillColorSpace="rgb",
        lineColor=None,
        units="pix",
        autoLog=False,
    )
    vertical = visual.Rect(
        win,
        width=weight,
        height=size,
        fillColor=color,
        fillColorSpace="rgb",
        lineColor=None,
        units="pix",
        autoLog=False,
    )
    for part in (horizontal, vertical):
        # Keep the original dimensions so repeated attention requests never
        # compound the scale or allocate new graphics objects.
        part._calibration_base_size = (part.width, part.height)
        part._calibration_scale = 1.0
        part._calibration_base_color = color
        part._calibration_color = color
    return (horizontal, vertical)


_RAINBOW_COLORS = (
    (255, 0, 0), (255, 127, 0), (255, 255, 0), (0, 255, 0),
    (0, 0, 255), (75, 0, 130), (148, 0, 211),
)


def _draw_fixation_cross(
    parts, pos: Sequence[float], *, attention_started_s=None, now_s=None
) -> None:
    """Draw a normal cross, or seven doubled rainbow flashes in one second.

    Both displays use the main task's monotonic start timestamp. Rendering
    follows elapsed time without sleeps, frame counters or new stimuli.
    """
    scale, visible, rainbow_index = 1.0, True, None
    if attention_started_s is not None:
        now_s = time.perf_counter() if now_s is None else now_s
        elapsed = now_s - attention_started_s
        if 0 <= elapsed < 1.0:
            scale = 2.0
            phase = int(elapsed * 14)  # Seven colors, each followed by darkness.
            visible = phase % 2 == 0
            rainbow_index = phase // 2
    for part in parts:
        if part._calibration_scale != scale:
            part.size = tuple(value * scale for value in part._calibration_base_size)
            part._calibration_scale = scale
        color = (
            part._calibration_base_color if rainbow_index is None else
            utils.rgb255_to_psychopy(_RAINBOW_COLORS[rainbow_index])
        )
        if part._calibration_color != color:
            part.fillColor = color
            part._calibration_color = color
        part.pos = (float(pos[0]), float(pos[1]))
        if visible:
            part.draw()


def _make_button(
    win, *, text: str, pos: Tuple[float, float], size: Tuple[float, float], color_rgb
):
    rect = visual.Rect(
        win,
        width=float(size[0]),
        height=float(size[1]),
        pos=pos,
        fillColor=utils.rgb255_to_psychopy(color_rgb),
        fillColorSpace="rgb",
        lineColor=utils.rgb255_to_psychopy((20, 20, 20)),
        lineColorSpace="rgb",
        units="pix",
        autoLog=False,
    )
    label = visual.TextStim(
        win,
        text=text,
        units="pix",
        height=max(18.0, min(float(size[0]), float(size[1])) * 0.45),
        pos=pos,
        color=utils.rgb255_to_psychopy((255, 255, 255)),
        colorSpace="rgb",
        autoLog=False,
    )
    return rect, label


def _draw_button(button) -> None:
    rect, label = button
    rect.draw()
    label.draw()


def _compute_preview_layout(exp_size: Sequence[int], main_size: Sequence[int]) -> dict:
    exp_w, exp_h = max(float(exp_size[0]), 1.0), max(float(exp_size[1]), 1.0)
    fit = compute_centered_aspect_fit(
        (max(100, exp_w - 260), max(100, exp_h - 240)), main_size
    )
    fit["left_margin"] = fit["right_margin"] = (exp_w - fit["box_size"][0]) / 2
    fit["top_margin"] = fit["bottom_margin"] = (exp_h - fit["box_size"][1]) / 2
    box_center = fit["box_center"]
    box_size = fit["box_size"]
    box_left = float(box_center[0]) - (float(box_size[0]) * 0.5)
    box_right = float(box_center[0]) + (float(box_size[0]) * 0.5)
    box_top = float(box_center[1]) + (float(box_size[1]) * 0.5)
    box_bottom = float(box_center[1]) - (float(box_size[1]) * 0.5)
    return {
        "box_center": box_center,
        "box_size": box_size,
        "left_margin": fit["left_margin"],
        "right_margin": fit["right_margin"],
        "top_margin": fit["top_margin"],
        "bottom_margin": fit["bottom_margin"],
        "box_left": box_left,
        "box_right": box_right,
        "box_top": box_top,
        "box_bottom": box_bottom,
    }


def _fraction_to_preview(
    fraction_pos: Sequence[float],
    *,
    box_center: Sequence[float],
    box_size: Sequence[float],
    clamp: bool = True,
) -> Tuple[float, float]:
    fx = float(fraction_pos[0])
    fy = float(fraction_pos[1])
    if clamp:
        fx = clamp_fraction(fx)
        fy = clamp_fraction(fy)
    return (
        float(box_center[0]) + (fx * float(box_size[0])),
        float(box_center[1]) + (fy * float(box_size[1])),
    )


def _preview_to_fraction(
    point: Sequence[float],
    *,
    box_center: Sequence[float],
    box_size: Sequence[float],
) -> Tuple[float, float]:
    return (
        clamp_fraction(
            (float(point[0]) - float(box_center[0])) / max(float(box_size[0]), 1.0)
        ),
        clamp_fraction(
            (float(point[1]) - float(box_center[1])) / max(float(box_size[1]), 1.0)
        ),
    )


class EyeCalibrationControls:
    def __init__(
        self, win, canvas_size, config, controls_shared, reward_event, exit_event,
        actions_queue=None,
    ):
        from bin.eye_tracking import EyeCalibration

        self.win = exp_win = win
        self.controls_shared = controls_shared
        self.actions_queue = actions_queue
        self.reward_event = reward_event
        self.exit_event = exit_event
        exp_size = canvas_size
        main_size = config["subject_size"]
        bg = config["bg"]
        fixation_color = config["fixation_color"]
        fixation_cross_size = config["fixation_cross_size"]
        fixation_cross_weight = config["fixation_cross_weight"]
        calibration = EyeCalibration(**config["calibration"])
        x_scale_limits = config["x_scale_limits"]
        y_scale_limits = config["y_scale_limits"]
        layout = _compute_preview_layout(exp_size, main_size)
        box_center, box_size = layout["box_center"], layout["box_size"]
        exp_bg = utils.make_bg_rect(exp_win, (30, 30, 30))
        preview_box = visual.Rect(
            exp_win,
            width=box_size[0],
            height=box_size[1],
            pos=box_center,
            fillColor=utils.rgb255_to_psychopy(bg),
            fillColorSpace="rgb",
            lineColor=utils.rgb255_to_psychopy((150, 150, 150)),
            lineColorSpace="rgb",
            lineWidth=2,
            units="pix",
            autoLog=False,
        )
        exp_cross = _make_fixation_cross(
            exp_win,
            size=max(
                6.0,
                float(fixation_cross_size)
                * (float(box_size[0]) / max(float(main_size[0]), 1.0)),
            ),
            weight=max(
                2.0,
                float(fixation_cross_weight)
                * (float(box_size[0]) / max(float(main_size[0]), 1.0)),
            ),
            color_rgb=fixation_color,
        )
        eye_dot = visual.Circle(
            exp_win,
            radius=max(4.0, min(float(exp_size[0]), float(exp_size[1])) * 0.008),
            fillColor=utils.rgb255_to_psychopy((40, 120, 255)),
            fillColorSpace="rgb",
            lineColor=utils.rgb255_to_psychopy((255, 255, 255)),
            lineColorSpace="rgb",
            units="pix",
            autoLog=False,
        )
        calibration_dot = visual.Circle(
            exp_win,
            radius=max(3.0, min(float(exp_size[0]), float(exp_size[1])) * 0.006),
            fillColor=utils.rgb255_to_psychopy((255, 0, 0)),
            fillColorSpace="rgb",
            lineColor=None,
            units="pix",
            autoLog=False,
        )

        # Fixed gutters keep the preview and controls disjoint even when the
        # two displays have the same aspect ratio. The image itself remains an
        # aspect-preserving fit of the oriented subject scene.
        exp_w, exp_h = float(exp_size[0]), float(exp_size[1])
        screen_top = exp_h / 2
        button_w, button_h = 90.0, 48.0
        reward_pos = (-exp_w / 2 + 65, screen_top - 45)
        exit_pos = (exp_w / 2 - 65, screen_top - 45)
        zero_pos = (-exp_w / 2 + 65, -exp_h / 2 + 50)
        x_slider_pos = (0, -exp_h / 2 + 55)
        y_slider_pos = (-exp_w / 2 + 65, 0)

        reward_button = _make_button(
            exp_win,
            text="reward",
            pos=reward_pos,
            size=(button_w, button_h),
            color_rgb=(50, 150, 80),
        )
        exit_button = _make_button(
            exp_win,
            text="exit",
            pos=exit_pos,
            size=(button_w, button_h),
            color_rgb=(200, 55, 55),
        )
        zero_button = _make_button(
            exp_win,
            text="add (Z)",
            pos=zero_pos,
            size=(90.0, 48.0),
            color_rgb=(75, 75, 75),
        )

        x_slider = Slider(
            exp_win,
            orientation="horizontal",
            pos=x_slider_pos,
            length=box_size[0] * 0.85,
            value_min=x_scale_limits[0],
            value_max=x_scale_limits[1],
            value=calibration.x_gain,
            label="subject X gain",
        )
        y_slider = Slider(
            exp_win,
            orientation="vertical",
            pos=y_slider_pos,
            length=box_size[1] * 0.85,
            value_min=y_scale_limits[0],
            value_max=y_scale_limits[1],
            value=calibration.y_gain,
            label="subject Y gain",
        )
        status_text = visual.TextStim(
            exp_win,
            text="",
            units="pix",
            height=18.0,
            pos=(0, screen_top - 48),
            wrapWidth=max(150, exp_w - 320),
            color=utils.rgb255_to_psychopy((245, 245, 245)),
            colorSpace="rgb",
            autoLog=False,
        )

        self.bg, self.preview_box = exp_bg, preview_box
        self.cross, self.eye_dot = exp_cross, eye_dot
        self.calibration_dot = calibration_dot
        self.x_slider, self.y_slider = x_slider, y_slider
        self.reward_button, self.exit_button, self.zero_button = (
            reward_button,
            exit_button,
            zero_button,
        )
        self.status_text = status_text
        self.box_center, self.box_size = box_center, box_size
        self.mouse = event.Mouse(win=win)
        self.last_down = False
        self.dragging = None
        self.fixation = (0.0, 0.0)
        self.zero_count = 0
        self.delete_count = 0
        self.attention_count = 0
        self.auto_reward_count = 0
        self.video_toggle_count = 0
        self.revision = 0
        self.video_size = tuple(config.get("video_size", (100, 100)))
        self.video_preview_size = tuple(
            pixels * preview_pixels / subject_pixels
            for pixels, preview_pixels, subject_pixels in zip(
                self.video_size, box_size, main_size,
            )
        )
        self.video_reader = self.video_stim = None
        self.video_sequence = 0
        # Mask pixels outside the subject scene when the fixation reaches an
        # edge. Controls are drawn later, on top of these fixed border masks.
        w, h = exp_size
        bw, bh = box_size
        self.video_masks = tuple(
            visual.Rect(
                win, width=mw, height=mh, pos=pos, units="pix",
                fillColor=utils.rgb255_to_psychopy((30, 30, 30)), lineColor=None,
                autoLog=False,
            )
            for mw, mh, pos in (
                ((w - bw) / 2, h, (-(w + bw) / 4, 0)),
                ((w - bw) / 2, h, ((w + bw) / 4, 0)),
                (bw, (h - bh) / 2, (0, -(h + bh) / 4)),
                (bw, (h - bh) / 2, (0, (h + bh) / 4)),
            )
        )

    def _apply_pending_gains_locked(self):
        """Consume the reliable fit mailbox without moving the target or revision."""
        shared = self.controls_shared
        if shared[16] == shared[17]:
            return
        self.x_slider.set_value(shared[14])
        self.y_slider.set_value(shared[15])
        shared[:2] = [self.x_slider.value, self.y_slider.value]
        shared[17] = shared[16]

    def _publish_controls(self, *, calibration_changed=True):
        if calibration_changed:
            self.revision += 1
        with self.controls_shared.get_lock():
            # A fit may arrive between polling and publishing. Apply it before
            # sending the UI snapshot so older slider values never undo a fit.
            self._apply_pending_gains_locked()
            # Leave pending main-window gain/target steps untouched. Only this
            # process writes control values, avoiding stale UI overwrites.
            self.controls_shared[:7] = [
                self.x_slider.value,
                self.y_slider.value,
                *self.fixation,
                self.revision,
                self.zero_count,
                self.attention_count,
            ]
            self.controls_shared[9] = self.auto_reward_count
            self.controls_shared[12] = self.video_toggle_count
            self.controls_shared[13] = self.delete_count

    def _set_fixation(self, position):
        """Share click/key bounds and notify the task to reset fixation history."""
        position = tuple(clamp_fraction(value) for value in position)
        if position != self.fixation:
            self.fixation = position
            self._publish_controls()

    def _request_point_action(self, action):
        """Retain input order and the target displayed when each point was added."""
        if self.actions_queue is not None:
            point_action = {"action": action}
            if action == "add":
                point_action["target_fraction"] = list(self.fixation)
            self.actions_queue.put_nowait(point_action)
        if action == "add":
            self.zero_count += 1
        else:
            self.delete_count += 1
        self._publish_controls()

    def poll_controls(self):
        keys = event.getKeys(keyList=("escape", *CALIBRATION_KEYS))
        with self.controls_shared.get_lock():
            self._apply_pending_gains_locked()
            x_steps, y_steps = self.controls_shared[7:9]
            self.controls_shared[7:9] = [0, 0]
            fix_x_steps, fix_y_steps = self.controls_shared[10:12]
            self.controls_shared[10:12] = [0, 0]
        # Use the same sliders for both focus paths and retain repeated keys
        # between preview frames. Signed scales change numerically, even below 0.
        x_changed = self.x_slider.nudge(
            int(x_steps) + keys.count("right") - keys.count("left")
        )
        y_changed = self.y_slider.nudge(
            int(y_steps) + keys.count("up") - keys.count("down")
        )
        if x_changed or y_changed:
            self._publish_controls()
        # Work in subject-scene fractions, as clicks do; the main task applies
        # display rotation when drawing. Preview resizing never changes a step.
        self._set_fixation((
            self.fixation[0] + fix_x_steps * FIXATION_STEP_FRACTION,
            self.fixation[1] + fix_y_steps * FIXATION_STEP_FRACTION,
        ))
        # Separate counters cannot distinguish Delete then Z from Z then
        # Delete. Preserve point actions and target movement in keyboard order.
        for key in keys:
            local_x, local_y = fixation_key_steps((key,))
            if local_x or local_y:
                self._set_fixation((
                    self.fixation[0] + local_x * FIXATION_STEP_FRACTION,
                    self.fixation[1] + local_y * FIXATION_STEP_FRACTION,
                ))
            elif key == "z":
                self._request_point_action("add")
            elif key in ("delete", "backspace"):
                self._request_point_action("delete")
        if "escape" in keys:
            self.exit_event.set()
        if "r" in keys:
            self.reward_event.set()
        if "f" in keys:
            self.attention_count += 1
            # A visual cue must not reset fixation acceptance or change gain,
            # offset, or target position. Keep the calibration revision intact.
            self._publish_controls(calibration_changed=False)
        if "a" in keys:
            self.auto_reward_count += keys.count("a")
            self._publish_controls(calibration_changed=False)
        if "v" in keys:
            self.video_toggle_count += keys.count("v")
            self._publish_controls(calibration_changed=False)
        down = bool(self.mouse.getPressed()[0])
        pos = self.mouse.getPos()
        if down and not self.last_down:
            if self.exit_button[0].contains(pos):
                self.exit_event.set()
            elif self.reward_button[0].contains(pos):
                self.reward_event.set()
            elif self.zero_button[0].contains(pos):
                self._request_point_action("add")
            elif self.x_slider.contains(pos):
                self.dragging = self.x_slider
            elif self.y_slider.contains(pos):
                self.dragging = self.y_slider
            elif self.preview_box.contains(pos):
                self._set_fixation(_preview_to_fraction(
                    pos, box_center=self.box_center, box_size=self.box_size
                ))
        if down and self.dragging is not None:
            self.dragging.set_from_pos(pos)
            self._publish_controls()
        if not down:
            self.dragging = None
        self.last_down = down

    def draw(self, payload):
        self.bg.draw()
        self.preview_box.draw()
        fixation_pos = _fraction_to_preview(
            self.fixation, box_center=self.box_center, box_size=self.box_size,
        )
        descriptor = payload.get("video_descriptor")
        if descriptor is not None and self.video_reader is None:
            self.video_reader = SharedVideoFrameReader(**descriptor)
            self.video_stim = make_rgb_video_stim(
                self.win, self.video_size, size=self.video_preview_size, visual_module=visual,
            )
        video_visible = bool(payload.get("video_visible")) and self.video_reader is not None
        if video_visible:
            frame = self.video_reader.read_latest(
                self.video_sequence, minimum_sequence=payload.get("video_sequence", 1),
            )
            if frame is not None:
                upload_rgb_texture(self.video_stim, frame.rgb)
                self.video_sequence = frame.sequence
            # Before the first decoded frame the texture is black. Retain the
            # last video frame during a delayed seek; only V restores the cross.
        if video_visible:
            self.video_stim.pos = fixation_pos
            self.video_stim.draw()
            for mask in self.video_masks:
                mask.draw()
        else:
            _draw_fixation_cross(
                self.cross, fixation_pos, attention_started_s=payload.get("attention_started_s"),
            )
        # Hide invalid gaze instead of displaying last reliable coordinates as
        # a live sample. The status text explains blink/loss/staleness/errors.
        if payload.get("eye_valid", False):
            self.eye_dot.pos = _fraction_to_preview(
                payload["eye_fraction"],
                box_center=self.box_center,
                box_size=self.box_size,
            )
            self.eye_dot.draw()
        # Draw retained targets above gaze so a correctly fixated point still
        # shows its red center rather than being covered by the larger eye dot.
        for point in payload.get("calibration_points", ()):
            self.calibration_dot.pos = _fraction_to_preview(
                point, box_center=self.box_center, box_size=self.box_size,
            )
            self.calibration_dot.draw()
        self.x_slider.draw()
        self.y_slider.draw()
        for button in (self.reward_button, self.exit_button, self.zero_button):
            _draw_button(button)
        set_preview_text(
            self.status_text, payload.get("status_text", "Waiting for Eye 0 UART data")
        )
        self.status_text.draw()

    def close(self):
        if self.video_reader is not None:
            self.video_reader.close()
