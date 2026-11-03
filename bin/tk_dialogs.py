"""Touch-sized dialogs with consistent keyboard navigation on every platform."""

import tkinter as tk
from tkinter.scrolledtext import ScrolledText


def bind_dialog_keys(dialog, buttons, *, cancel, extra_widgets=()):
    """Use arrows to select, Enter to activate, and Escape to dismiss.

    Bind on buttons before their Tk class bindings to avoid invoking twice.
    A dialog's own toplevel bindings keep these keys out of the launcher menu.
    This helper never grabs input, so it also works for nonmodal warnings.
    """
    selected = 0

    def select(index, *, focus=True):
        nonlocal selected
        selected = index % len(buttons)
        for i, button in enumerate(buttons):
            button.configure(
                highlightthickness=3, highlightbackground="#2563eb" if i == selected else "#dddddd",
                highlightcolor="#2563eb", default="active" if i == selected else "normal",
            )
        if focus:
            buttons[selected].focus_set()

    def key(event):
        if event.keysym == "Escape":
            cancel()
        elif event.keysym in ("Left", "Up", "Right", "Down"):
            select(selected + (-1 if event.keysym in ("Left", "Up") else 1))
        else:
            buttons[selected].invoke()
        return "break"

    for i, button in enumerate(buttons):
        button.bind("<FocusIn>", lambda event, index=i: select(index, focus=False), add="+")
    for widget in (dialog, *buttons, *extra_widgets):
        for sequence in ("<Left>", "<Right>", "<Up>", "<Down>", "<Return>", "<KP_Enter>", "<Escape>"):
            widget.bind(sequence, key)
    dialog.protocol("WM_DELETE_WINDOW", cancel)
    select(0)
    dialog.after_idle(buttons[0].focus_set)


def choose_dialog(parent, title, message, choices, *, cancel_value=None):
    """Return the selected value; closing/Escape returns ``cancel_value``."""
    dialog = tk.Toplevel(parent)
    dialog.title(title)
    dialog.transient(parent)
    width = max(320, min(760, parent.winfo_width() - 40))
    height = max(260, min(560, parent.winfo_height() - 60))
    dialog.geometry(f"{width}x{height}+{parent.winfo_rootx() + 20}+{parent.winfo_rooty() + 20}")
    previous_focus = parent.focus_get()
    previous_grab = parent.grab_current()
    result = cancel_value

    def choose(value):
        nonlocal result
        result = value
        dialog.destroy()

    tk.Label(dialog, text=title, font=("Helvetica", 20, "bold"), pady=12).pack(fill="x")
    buttons_frame = tk.Frame(dialog)
    buttons_frame.pack(side="bottom", fill="x", padx=14, pady=14)
    body = ScrolledText(dialog, wrap="word", font=("Helvetica", 14), padx=12, pady=12)
    body.insert("1.0", str(message))
    body.configure(state="disabled")
    body.pack(fill="both", expand=True, padx=14)
    buttons = []
    for i, (label, value) in enumerate(choices):
        button = tk.Button(
            buttons_frame, text=label, command=lambda v=value: choose(v),
            font=("Helvetica", 16, "bold"), height=2, pady=16,
            wraplength=max(80, width // len(choices) - 35),
        )
        button.grid(row=0, column=i, sticky="ew", padx=4)
        buttons_frame.grid_columnconfigure(i, weight=1, uniform="choices")
        buttons.append(button)
    bind_dialog_keys(dialog, buttons, cancel=lambda: choose(cancel_value), extra_widgets=(body,))
    dialog.grab_set()
    try:
        parent.wait_window(dialog)
    finally:
        # Restore a surrounding dialog, if any, and keyboard navigation in the
        # launcher. Either widget may have been destroyed during the callback.
        for widget, method in ((previous_grab, "grab_set"), (previous_focus, "focus_set")):
            try:
                if widget is not None and widget.winfo_exists():
                    getattr(widget, method)()
            except tk.TclError:
                pass
    return result


def showinfo(title, message, *, parent=None):
    parent = parent if parent is not None else tk._default_root
    return choose_dialog(parent, title, message, [("OK", "ok")], cancel_value="ok")


showwarning = showinfo
showerror = showinfo
