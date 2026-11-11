"""Touch controls for hardware pulses, independent of experiment lifecycle."""
from __future__ import annotations

import queue
import sys
import time
import tkinter as tk

from bin.debug_outputs import OutputPin
from interface.debug_hardware_process import DebugHardwareController


def debug_settings(cfg):
    """Bad optional settings must not prevent access to either output backend."""
    settings = cfg.get("debug_mode")
    settings = settings if isinstance(settings, dict) else {}
    diagnostic = cfg.get("system_diagnostic")
    diagnostic = diagnostic if isinstance(diagnostic, dict) else {}

    def number(name, fallback, maximum):
        try:
            value = int(settings.get(name, fallback))
            return value if 0 <= value <= maximum else 0
        except (TypeError, ValueError, OverflowError):
            return 0

    return {
        "gpio_chip": number("gpio_chip", 0, 255),
        "daq_address": number("daq_address", diagnostic.get("daq_address", 0), 7),
        "daq_module": str(settings.get("daq_module") or diagnostic.get("daq_module") or "piplates.DAQC2plate"),
    }


class HardwareDebugPanel(tk.Frame):
    """Overlay the experimenter menu without opening or resolving task screens."""

    def __init__(self, root, cfg, *, scroll_frame_class, on_close, config_dir=None):
        super().__init__(root, bg="#e9ecef")
        settings = debug_settings(cfg)
        self.controller = DebugHardwareController(
            cfg, config_dir=config_dir, daq_module=settings["daq_module"],
        )
        self.on_close = on_close
        self.backend = "gpio"
        self.gpio_chip = settings["gpio_chip"]
        self.daq_address = settings["daq_address"]
        self.closing = False
        self.releasing = False
        self.close_started = 0.0
        self.buttons = {}
        self.states = {}
        self.status = tk.StringVar(value="Tap a pin for a 5-second pulse. GPIO uses BCM numbering.")
        self.grid(row=0, column=0, rowspan=3, sticky="nsew")
        self.tkraise()
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        header = tk.Frame(self, bg="#e9ecef")
        header.grid(row=0, column=0, sticky="ew", padx=8, pady=6)
        header.grid_columnconfigure(0, weight=1)
        tk.Label(header, text="Debug Mode · 5 s pulses", font=("Helvetica", 20, "bold"), bg="#e9ecef").grid(row=0, column=0, sticky="w")
        self._button(header, "Back", self.close).grid(row=0, column=1, padx=4)

        selectors = tk.Frame(self, bg="#e9ecef")
        selectors.grid(row=1, column=0, sticky="ew", padx=8)
        for index, (label, backend) in enumerate((("GPIO", "gpio"), ("DAQC2 DOUT", "daqc2"))):
            self._button(selectors, label, lambda b=backend: self._select(b)).grid(row=0, column=index, padx=3, sticky="ew")
            selectors.grid_columnconfigure(index, weight=1)
        self.device_button = self._button(selectors, "", self._next_device)
        self.device_button.grid(row=0, column=2, padx=3, sticky="ew")

        self.scroller = scroll_frame_class(self, bg="#e9ecef")
        self.scroller.grid(row=2, column=0, sticky="nsew", padx=8, pady=6)
        footer = tk.Frame(self, bg="#e9ecef")
        footer.grid(row=3, column=0, sticky="ew", padx=8, pady=6)
        footer.grid_columnconfigure(0, weight=1)
        label = tk.Label(
            footer, textvariable=self.status, font=("Helvetica", 14),
            bg="#e9ecef", anchor="w", justify="left", wraplength=500, height=3,
        )
        label.grid(row=0, column=0, sticky="ew")
        label.bind("<Configure>", lambda event: label.configure(wraplength=max(100, event.width - 12)))
        self._button(footer, "Stop All", self.stop_all, bg="#b91c1c", fg="white").grid(row=0, column=1, padx=6)
        self._render_pins()
        self.after_id = self.after(50, self._poll)

    @staticmethod
    def _button(parent, text, command, **kwargs):
        button = tk.Button(
            parent, text=text, command=command, font=("Helvetica", 16, "bold"),
            height=2, padx=8, takefocus=True, **kwargs,
        )

        def activate(_event):
            button.invoke()
            return "break"

        # Back/Stop/selectors must consume Enter before the launcher's root
        # binding could interpret it as a request to pulse the selected pin.
        for sequence in ("<Return>", "<KP_Enter>"):
            button.bind(sequence, activate)
        return button

    def _select(self, backend):
        self.backend = backend
        self._render_pins()

    def _next_device(self):
        if self.backend == "gpio":
            self.gpio_chip = (self.gpio_chip + 1) % 8
        else:
            self.daq_address = (self.daq_address + 1) % 8
        self._render_pins()

    def _render_pins(self):
        self.scroller.reset_scroll()
        for child in self.scroller.inner.winfo_children():
            child.destroy()
        self.buttons.clear()
        gpio = self.backend == "gpio"
        device = self.gpio_chip if gpio else self.daq_address
        self.device_button.configure(text=f"{'Chip' if gpio else 'Address'}: {device} ▸")
        note = "BCM GPIO 0–27 · HIGH then LOW" if gpio else "DOUT 0–7 · ON pulls low; OFF releases output"
        tk.Label(self.scroller.inner, text=note, font=("Helvetica", 14), bg="#e9ecef", wraplength=440).grid(row=0, column=0, columnspan=4, sticky="ew", pady=4)
        for pin in range(28 if gpio else 8):
            target = OutputPin(self.backend, pin, device)
            button = self._button(self.scroller.inner, "", lambda t=target: self._pulse(t))
            button.grid(row=pin // 4 + 1, column=pin % 4, sticky="ew", padx=3, pady=3)
            self.scroller.inner.grid_columnconfigure(pin % 4, weight=1, uniform="pins")
            self.scroller.register_button(button)
            for sequence in ("<Up>", "<Down>", "<Return>", "<KP_Enter>"):
                button.bind(sequence, self._key)
            self.buttons[target] = button
            self._update_button(target)

    def _key(self, event):
        if event.keysym in {"Up", "Down"}:
            self.scroller.move_selection(-1 if event.keysym == "Up" else 1)
        else:
            self.scroller.invoke_selection()
        return "break"

    def _update_button(self, target):
        button = self.buttons.get(target)
        if button is None:
            return
        state, _detail = self.states.get(target, ("ready", ""))
        caption = {
            "ready": "5 s pulse", "starting": "Opening…", "active": "ON · 5 s",
            "done": "5 s pulse", "error": "Retry 5 s", "uncleared": "OFF failed",
            "cleanup": "Release failed",
        }[state]
        button.configure(
            text=f"{'GPIO' if target.backend == 'gpio' else 'DOUT'} {target.pin}\n{caption}",
            state="disabled" if self.closing or state in {"starting", "active", "uncleared", "cleanup"} else "normal",
            bg="#bbf7d0" if state == "active" else "#fecaca" if state in {"error", "uncleared", "cleanup"} else "#f7f7f7",
        )

    def _pulse(self, target):
        if not self.closing and self.controller.pulse(target):
            self.states[target] = ("starting", "Opening output…")
            self._update_button(target)

    def stop_all(self):
        self.controller.stop_all()
        self.status.set("Stopping pulses and retrying any failed OFF commands…")

    def close(self):
        self.closing = True
        self.close_started = time.monotonic()
        self.stop_all()
        for target in self.buttons:
            self._update_button(target)

    def _poll(self):
        while True:
            try:
                target, state, detail = self.controller.events.get_nowait()
            except queue.Empty:
                break
            self.states[target] = (state, detail)
            if state in {"error", "uncleared", "cleanup"}:
                print(f"Debug Mode — {target.label}: {detail}", file=sys.stderr)
            # A verbose driver exception must not push the other pins offscreen.
            summary = detail if len(detail) <= 200 else detail[:197] + "…"
            self.status.set(f"{target.label}: {summary}")
            self._update_button(target)
        if self.closing and not self.releasing and self.controller.busy and time.monotonic() - self.close_started >= 2.0:
            self.closing = False
            self.status.set("A driver is still busy stopping. Other pins remain available; tap Back to retry.")
            for target in self.buttons:
                self._update_button(target)
        if self.closing and not self.controller.busy:
            if self.controller.uncleared:
                # Keep failed OFF visible and retryable; other pins stay usable.
                self.closing = False
                for target in self.buttons:
                    self._update_button(target)
            else:
                if not self.releasing:
                    self.releasing = True
                    self.controller.close()
                if self.controller.busy:
                    self.after_id = self.after(50, self._poll)
                    return
                self.destroy()
                self.on_close()
                return
        self.after_id = self.after(50, self._poll)
