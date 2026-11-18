"""Participant-coordinate adapters at the PsychoPy drawing/input boundary.

Task positions and hit tests stay in screen-centered, right/up pixels. A
stimulus's native position/orientation is prepared when it changes, never on
each draw. The display driver alone sees the rotated framebuffer coordinates.
"""

from .screen import _quarter_turn_degrees, _rotate_centered_quarter_turn


class ParticipantView:
    def __init__(self, rotation_degrees=0):
        self.rotation = _quarter_turn_degrees(rotation_degrees)
        self._inverse_rotation = (-self.rotation) % 360

    def to_native(self, point):
        return _rotate_centered_quarter_turn(point, self.rotation)

    def from_native(self, point):
        return _rotate_centered_quarter_turn(point, self._inverse_rotation)

    def stimulus(self, native_stimulus, *, pos=(0, 0), ori=0):
        return ParticipantStimulus(native_stimulus, self, pos=pos, ori=ori)

    def mouse(self, native_mouse):
        return ParticipantMouse(native_mouse, self)


class ParticipantStimulus:
    def __init__(self, native_stimulus, view, *, pos=(0, 0), ori=0):
        object.__setattr__(self, "_native", native_stimulus)
        object.__setattr__(self, "_view", view)
        self.pos = pos
        self.ori = ori

    def __getattr__(self, name):
        return getattr(self._native, name)

    def __setattr__(self, name, value):
        if name == "pos":
            position = tuple(map(float, value))
            object.__setattr__(self, name, position)
            self._native.pos = self._view.to_native(position)
        elif name == "ori":
            object.__setattr__(self, name, float(value))
            object.__setattr__(self, "_local_view", ParticipantView(-float(value)))
            self._native.ori = float(value) + self._view.rotation
        else:
            setattr(self._native, name, value)

    def contains(self, point, *args, **kwargs):
        if hasattr(point, "getPos"):
            point = point.getPos()
        x, y = self._local_view.to_native((point[0] - self.pos[0], point[1] - self.pos[1]))
        if hasattr(self._native, "radius"):
            return x * x + y * y <= float(self._native.radius) ** 2
        if hasattr(self._native, "width") and hasattr(self._native, "height"):
            width, height = self._native.width, self._native.height
        else:
            width, height = self._native.size
        return abs(x) <= float(width) / 2 and abs(y) <= float(height) / 2


class ParticipantMouse:
    def __init__(self, native_mouse, view):
        self._native = native_mouse
        self._view = view

    def __getattr__(self, name):
        return getattr(self._native, name)

    def getPos(self):
        return self._view.from_native(self._native.getPos())

    def setPos(self, position):
        return self._native.setPos(self._view.to_native(position))
