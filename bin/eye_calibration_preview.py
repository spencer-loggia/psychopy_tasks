"""Interactive calibration renderer hosted by the shared ExperimenterPreview process.

Only this process owns the secondary window and its mouse. Control values cross
back as one atomic snapshot; rendering updates use the preview's latest-only
mailbox. There is no UART or reward hardware access in this component.
"""

from __future__ import annotations

from typing import Sequence, Tuple
from psychopy import event, visual
from bin import utils
from bin.eye_tracking import clamp_fraction
from bin.screen import compute_centered_aspect_fit


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
        self.value = float(value)
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
        )
        self.text = visual.TextStim(
            win,
            text="",
            units="pix",
            height=18.0,
            color=utils.rgb255_to_psychopy((245, 245, 245)),
            colorSpace="rgb",
            pos=(0, 0),
        )

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

    def draw(self) -> None:
        self.knob.pos = self._knob_pos()
        self.track.draw()
        self.knob.draw()
        self.text.text = f"{self.label} {self.value:.4f}"
        if self.orientation == "horizontal":
            self.text.pos = (self.pos[0], self.pos[1] + 30.0)
        else:
            self.text.text = f"{self.label}\n{self.value:.4f}"
            self.text.pos = (self.pos[0], self.pos[1] + self.length / 2 + 35)
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
    )
    vertical = visual.Rect(
        win,
        width=weight,
        height=size,
        fillColor=color,
        fillColorSpace="rgb",
        lineColor=None,
        units="pix",
    )
    return (horizontal, vertical)


def _draw_fixation_cross(parts, pos: Sequence[float]) -> None:
    for part in parts:
        part.pos = (float(pos[0]), float(pos[1]))
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
    )
    label = visual.TextStim(
        win,
        text=text,
        units="pix",
        height=max(18.0, min(float(size[0]), float(size[1])) * 0.45),
        pos=pos,
        color=utils.rgb255_to_psychopy((255, 255, 255)),
        colorSpace="rgb",
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
        self, win, canvas_size, config, controls_shared, reward_event, exit_event
    ):
        from bin.eye_tracking import EyeCalibration

        self.win = exp_win = win
        self.controls_shared = controls_shared
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
            text="zero",
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
            value=calibration.x_scale,
            label="x scale",
        )
        y_slider = Slider(
            exp_win,
            orientation="vertical",
            pos=y_slider_pos,
            length=box_size[1] * 0.85,
            value_min=y_scale_limits[0],
            value_max=y_scale_limits[1],
            value=calibration.y_scale,
            label="y scale",
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
        )

        self.bg, self.preview_box = exp_bg, preview_box
        self.cross, self.eye_dot = exp_cross, eye_dot
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
        self.revision = 0

    def _publish_controls(self):
        self.revision += 1
        with self.controls_shared.get_lock():
            self.controls_shared[:] = [
                self.x_slider.value,
                self.y_slider.value,
                *self.fixation,
                self.revision,
                self.zero_count,
            ]

    def poll_controls(self):
        keys = event.getKeys(keyList=["escape", "r", "z"])
        if "escape" in keys:
            self.exit_event.set()
        if "r" in keys:
            self.reward_event.set()
        if "z" in keys:
            self.zero_count += 1
            self._publish_controls()
        down = bool(self.mouse.getPressed()[0])
        pos = self.mouse.getPos()
        if down and not self.last_down:
            if self.exit_button[0].contains(pos):
                self.exit_event.set()
            elif self.reward_button[0].contains(pos):
                self.reward_event.set()
            elif self.zero_button[0].contains(pos):
                self.zero_count += 1
                self._publish_controls()
            elif self.x_slider.contains(pos):
                self.dragging = self.x_slider
            elif self.y_slider.contains(pos):
                self.dragging = self.y_slider
            elif self.preview_box.contains(pos):
                self.fixation = _preview_to_fraction(
                    pos, box_center=self.box_center, box_size=self.box_size
                )
                self._publish_controls()
        if down and self.dragging is not None:
            self.dragging.set_from_pos(pos)
            self._publish_controls()
        if not down:
            self.dragging = None
        self.last_down = down

    def draw(self, payload):
        self.bg.draw()
        self.preview_box.draw()
        _draw_fixation_cross(
            self.cross,
            _fraction_to_preview(
                self.fixation, box_center=self.box_center, box_size=self.box_size
            ),
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
        self.x_slider.draw()
        self.y_slider.draw()
        for button in (self.reward_button, self.exit_button, self.zero_button):
            _draw_button(button)
        self.status_text.text = payload.get(
            "status_text", "Waiting for Eye 0 UART data"
        )
        self.status_text.draw()
