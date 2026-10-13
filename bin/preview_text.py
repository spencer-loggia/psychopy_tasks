"""Update diagnostic text without accumulating pyglet labels/GPU buffers.

PsychoPy 2025.1.1 TextStim._setTextShaders creates a new pyglet Label for
every changed string, without explicitly deleting the previous layout. Its
text setter documents the associated leak. A continuously changing gaze
readout must instead update the existing Label's document/layout in place.

Keep this compatibility detail here, shared by calibration and task previews.
These labels use fixed styling and left-to-right text, not participant stimuli.
Upstream: https://github.com/psychopy/psychopy/blob/2025.1.1/psychopy/visual/text.py
"""


def set_preview_text(stim, text: str) -> None:
    """Set preview text, reusing its native layout whenever styling is unchanged.

    Call on the window's rendering thread, just like TextStim.draw(). PsychoPy
    still handles positioning, colors, and drawing. Font/wrap changes, other
    backends and language shaping use its usual setter; release any replaced
    pyglet layout explicitly in that case.
    """
    text = str(text)
    if stim.text == text:
        return
    label = getattr(stim, "_pygletTextObj", None)
    if (
        label is not None
        and not getattr(stim, "_needSetText", False)
        and getattr(stim, "languageStyle", "LTR").lower() == "ltr"
    ):
        label.text = text
        # Mirror TextStim's bookkeeping without invoking its label-allocating
        # text descriptor. No automatic log entry is needed for preview text.
        stim.__dict__["text"] = text
        stim.width = label.width
        stim._fontHeightPix = label.height
        stim._needUpdate = True
    else:
        stim.text = text
        if label is not None and label is not getattr(stim, "_pygletTextObj", None):
            label.delete()
