"""Small numeric controls rendered only in the experimenter preview process.

This module imports no GUI toolkit. The caller supplies its existing PsychoPy
visual module and a settings model, so tasks share the editor without creating
windows, waiting for input, or doing UI work on their stimulus thread.
"""
from __future__ import annotations

from .preview_text import set_preview_text


SETTINGS_STRIP_HEIGHT = 60.0


class SettingsPanel:
    """Preallocated setting buttons and one nonblocking numeric input field."""

    def __init__(self, win, settings, *, visual):
        self.settings = settings
        self.specs = tuple(settings.specs)
        self.active_spec = None
        self.text = ""
        self.error = ""
        self.replace_text = False
        self.buttons = {}
        self.labels = {}
        for spec in self.specs:
            self.buttons[spec.name] = visual.Rect(
                win, width=124, height=56, units="pix",
                fillColor=(-0.65, -0.65, -0.65), lineColor=(0.3, 0.3, 0.3),
                autoLog=False,
            )
            self.labels[spec.name] = visual.TextStim(
                win, text="", height=12, wrapWidth=116, units="pix", autoLog=False,
            )
        # Retain the single-setting editor's useful inspection attributes.
        self.button = self.buttons[self.specs[0].name] if self.specs else None
        self.label = self.labels[self.specs[0].name] if self.specs else None
        self.panel = visual.Rect(
            win, width=360, height=154, units="pix", fillColor=(-0.85,) * 3,
            lineColor=(0.6,) * 3, autoLog=False,
        )
        self.title = visual.TextStim(
            win, text="", height=13, units="pix", autoLog=False,
        )
        self.field = visual.Rect(
            win, width=312, height=28, units="pix", fillColor=(-1,) * 3,
            lineColor=(0.4, 0.7, 1), autoLog=False,
        )
        self.input = visual.TextStim(
            win, text="", height=18, units="pix", autoLog=False,
        )
        self.hint = visual.TextStim(
            win, text="", height=11, units="pix", autoLog=False,
        )
        self._popup((0, 0), (360, 154), narrow=True)

    @property
    def editing(self):
        return self.active_spec is not None

    def _popup(self, pos, size, *, narrow):
        x, y = pos
        width, height = size
        self.panel.pos = pos
        self.panel.width, self.panel.height = size
        self.title.wrapWidth = self.hint.wrapWidth = width - 12
        self.field.width = width - 20 if narrow else min(280, width - 20)
        self.field.height = 28 if narrow else 20
        self.input.height = 14 if width < 150 else 18
        self.title.pos = (x, y + (height * 0.32 if narrow else 19))
        self.field.pos = self.input.pos = (x, y + (height * 0.04 if narrow else 0))
        self.hint.pos = (x, y - (height * 0.29 if narrow else 19))
        self._narrow = narrow

    def layout_strip(self, canvas_size):
        """Reserve the bottom 60 pixels, including the open input field."""
        width, height = map(float, canvas_size)
        count = max(1, len(self.specs))
        button_width = min(180.0, (width - 16) / count - 8)
        row_width = count * (button_width + 8) - 8
        y = -height / 2 + SETTINGS_STRIP_HEIGHT / 2
        for index, spec in enumerate(self.specs):
            pos = (-row_width / 2 + button_width / 2 + index * (button_width + 8), y)
            button, label = self.buttons[spec.name], self.labels[spec.name]
            button.width, button.height = button_width, 44
            button.pos = label.pos = pos
            label.wrapWidth = button_width - 8
        self._popup((0, y), (min(620, width - 16), 58), narrow=False)

    def layout_gutter(self, *, left, right, top, bottom):
        """Stack settings and place the editor inside the calibration gutter."""
        width = max(40.0, float(right) - float(left))
        x = (float(left) + float(right)) / 2
        for index, spec in enumerate(self.specs):
            button, label = self.buttons[spec.name], self.labels[spec.name]
            button.width, button.height = width, 44
            button.pos = label.pos = (x, float(top) - 22 - 50 * index)
            label.wrapWidth = width - 8
        popup_height = min(182.0, float(top) - float(bottom))
        # The popup replaces the same setting rows instead of covering scene
        # pixels. Reward/Exit remain visible above it.
        self._popup((x, float(top) - popup_height / 2), (width, popup_height), narrow=True)

    def click(self, pos):
        if self.editing:
            return True
        for spec in self.specs:
            if self.buttons[spec.name].contains(pos):
                self.active_spec = spec
                self.text = str(self.settings.snapshot()[spec.name][1])
                self.error = ""
                self.replace_text = True
                return True
        return False

    def handle_keys(self, keys):
        if not self.editing:
            return keys
        for key in keys:
            if key == "escape":
                self.active_spec = None
                break
            if key in ("return", "enter", "num_enter"):
                try:
                    self.settings.request(self.active_spec.name, self.text)
                except ValueError as exc:
                    self.error = str(exc)
                else:
                    self.active_spec = None
                    break
            elif key in ("backspace", "delete"):
                self.text = "" if self.replace_text or key == "delete" else self.text[:-1]
                self.replace_text = False
                self.error = ""
            else:
                character = {
                    "period": ".", "decimal": ".", "num_decimal": ".",
                    "minus": "-", "num_subtract": "-", "plus": "+", "num_add": "+",
                }.get(key, key.removeprefix("num_"))
                if len(character) == 1 and character in "0123456789.e-+":
                    self.text = ("" if self.replace_text else self.text) + character
                    self.replace_text = False
                    self.error = ""
        # A confirmation/cancellation consumes the whole keyboard batch, so
        # trailing reward/exit/calibration keys cannot escape into task input.
        return []

    def _visible_input_text(self):
        # Numeric glyphs fit within one font em. Reserve that conservative
        # width per character, including the cursor and overflow indicator;
        # keep the complete input buffer for validation and confirmation.
        capacity = max(2, int((self.field.width - 12) / self.input.height))
        if len(self.text) + 1 <= capacity:
            return self.text + "|"
        return "<" + self.text[-(capacity - 2):] + "|" if capacity > 2 else "<|"

    def draw(self, pos=None):
        if pos is not None and self.button is not None:
            self.button.pos = self.label.pos = pos
        values = self.settings.snapshot()
        for spec in self.specs:
            current, requested = values[spec.name]
            suffix = f" {spec.units}" if spec.units else ""
            label = f"{spec.label}: {current:g}{suffix}"
            if current != requested:
                label += f"\nNext: {requested:g}{suffix}"
            set_preview_text(self.labels[spec.name], label)
            self.buttons[spec.name].draw()
            self.labels[spec.name].draw()
        if self.editing:
            self.panel.draw()
            set_preview_text(self.title, self.active_spec.title)
            self.title.draw()
            self.field.draw()
            set_preview_text(self.input, self._visible_input_text())
            self.input.draw()
            hint = getattr(self.settings, "edit_hint", None) or f"Applies next {self.settings.boundary}"
            separator = "\n" if self._narrow else "    "
            set_preview_text(self.hint, self.error or hint + separator + "Enter: confirm  Esc: cancel")
            self.hint.draw()
