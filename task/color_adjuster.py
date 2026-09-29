#!/usr/bin/env python3
"""
RGB Color Adjuster for Raspberry Pi

Features
--------
- Reads CSV/TSV color files with required columns: id, r, g, b
- Can be launched with a JSON config containing a colors_tsv list
- Opens fullscreen on the configured experimenter display
- Shows only the live adjusted color (not a side-by-side original)
- Uses large touch-friendly RGB sliders
- Saves adjusted RGB and RGB-delta files after every Save & Next

Run:
    python3 rgb_color_adjuster.py rgb_adjuster_config.json

Config screen behavior:
    "screens": {
        "experimenter": null
    }

experimenter may be:
- null or "auto": second connected monitor if available, otherwise primary
- an integer such as 0 or 1
- a monitor name such as "HDMI-1"
"""

import csv
import json
import re
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox


def clamp(value, low=0, high=255):
    return max(low, min(high, int(value)))


def rgb_to_hex(r, g, b):
    return f"#{r:02x}{g:02x}{b:02x}"


def normalize_fieldnames(fieldnames):
    return {str(f).strip().lower(): f for f in (fieldnames or [])}


def get_required_column(fieldnames, name):
    normalized = normalize_fieldnames(fieldnames)
    if name not in normalized:
        raise ValueError(f"Missing required column: {name}")
    return normalized[name]


def read_color_file(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Color file not found:\n{path}")

    if path.suffix.lower() == ".tsv":
        delimiter = "\t"
    elif path.suffix.lower() == ".csv":
        delimiter = ","
    else:
        delimiter = None

    last_error = None
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            with path.open("r", newline="", encoding=encoding) as f:
                if delimiter is None:
                    sample = f.read(4096)
                    f.seek(0)
                    try:
                        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
                        use_delimiter = dialect.delimiter
                    except csv.Error:
                        use_delimiter = "\t"
                else:
                    use_delimiter = delimiter

                reader = csv.DictReader(f, delimiter=use_delimiter)
                fieldnames = reader.fieldnames or []

                id_col = get_required_column(fieldnames, "id")
                r_col = get_required_column(fieldnames, "r")
                g_col = get_required_column(fieldnames, "g")
                b_col = get_required_column(fieldnames, "b")

                colors = []
                for row_number, row in enumerate(reader, start=2):
                    color_id = str(row.get(id_col, "")).strip()
                    if not color_id:
                        raise ValueError(f"Missing id on row {row_number}.")

                    try:
                        r = clamp(round(float(row[r_col])))
                        g = clamp(round(float(row[g_col])))
                        b = clamp(round(float(row[b_col])))
                    except (ValueError, TypeError, KeyError):
                        raise ValueError(f"Invalid RGB value on row {row_number}.")

                    colors.append({"id": color_id, "r": r, "g": g, "b": b})

                return colors, use_delimiter

        except UnicodeDecodeError as exc:
            last_error = exc
            continue

    if last_error:
        raise last_error

    return [], delimiter or "\t"


def load_config(config_path):
    config_path = Path(config_path).resolve()

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    entries = config.get("colors_tsv")
    if not isinstance(entries, list) or not entries:
        raise ValueError('Config must contain a non-empty "colors_tsv" array.')

    resolved_entries = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"colors_tsv entry {index + 1} must be an object.")

        rel_path = entry.get("path")
        if not rel_path:
            raise ValueError(f'colors_tsv entry {index + 1} is missing "path".')

        color_path = Path(rel_path)
        if not color_path.is_absolute():
            color_path = (config_path.parent / color_path).resolve()

        resolved = dict(entry)
        resolved["_resolved_path"] = color_path
        resolved["_index"] = index
        resolved_entries.append(resolved)

    return config, resolved_entries


def get_connected_monitors(root):
    """Return monitor dictionaries with name, x, y, width, height, primary."""
    monitors = []

    try:
        result = subprocess.run(
            ["xrandr", "--listmonitors"],
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                match = re.match(
                    r"^\s*(\d+):\s+([^\s]+).*?(\d+)/\d+x(\d+)/\d+\+(-?\d+)\+(-?\d+)\s+(\S+)\s*$",
                    line,
                )
                if match:
                    index, flags, width, height, x, y, name = match.groups()
                    monitors.append({
                        "index": int(index),
                        "name": name,
                        "x": int(x),
                        "y": int(y),
                        "width": int(width),
                        "height": int(height),
                        "primary": "*" in flags,
                    })
    except (OSError, subprocess.SubprocessError):
        pass

    if not monitors:
        try:
            result = subprocess.run(
                ["xrandr", "--query"],
                capture_output=True,
                text=True,
                check=False,
                timeout=2,
            )
            if result.returncode == 0:
                index = 0
                for line in result.stdout.splitlines():
                    match = re.match(
                        r"^(\S+)\s+connected(?:\s+(primary))?\s+(\d+)x(\d+)\+(-?\d+)\+(-?\d+)",
                        line,
                    )
                    if match:
                        name, primary, width, height, x, y = match.groups()
                        monitors.append({
                            "index": index,
                            "name": name,
                            "x": int(x),
                            "y": int(y),
                            "width": int(width),
                            "height": int(height),
                            "primary": bool(primary),
                        })
                        index += 1
        except (OSError, subprocess.SubprocessError):
            pass

    if not monitors:
        monitors = [{
            "index": 0,
            "name": "default",
            "x": 0,
            "y": 0,
            "width": root.winfo_screenwidth(),
            "height": root.winfo_screenheight(),
            "primary": True,
        }]

    return monitors


def choose_experimenter_monitor(root, config):
    monitors = get_connected_monitors(root)
    setting = (config or {}).get("screens", {}).get("experimenter")

    if setting is None or str(setting).strip().lower() in {"auto", "second", "secondary"}:
        if len(monitors) > 1:
            return monitors[1]
        primary = next((m for m in monitors if m.get("primary")), None)
        return primary or monitors[0]

    if isinstance(setting, int):
        if 0 <= setting < len(monitors):
            return monitors[setting]
        raise ValueError(
            f"Experimenter screen index {setting} does not exist. "
            f"Connected monitors: {[m['name'] for m in monitors]}"
        )

    setting_text = str(setting).strip()
    if setting_text.isdigit():
        idx = int(setting_text)
        if 0 <= idx < len(monitors):
            return monitors[idx]

    for monitor in monitors:
        if monitor["name"].lower() == setting_text.lower():
            return monitor

    raise ValueError(
        f"Experimenter screen '{setting}' was not found. "
        f"Connected monitors: {[m['name'] for m in monitors]}"
    )


def place_fullscreen_on_monitor(root, monitor, fullscreen=True):
    """Move the window to one monitor, then fullscreen it there."""
    root.attributes("-fullscreen", False)
    root.overrideredirect(False)
    root.geometry(
        f"{monitor['width']}x{monitor['height']}+{monitor['x']}+{monitor['y']}"
    )
    root.update_idletasks()
    root.update()

    if fullscreen:
        # On Raspberry Pi / X11 this normally fullscreens on the monitor the
        # window was moved to. If the WM rejects it, the exact monitor-sized
        # geometry still leaves the application filling that display.
        try:
            root.attributes("-fullscreen", True)
        except tk.TclError:
            root.overrideredirect(True)
            root.geometry(
                f"{monitor['width']}x{monitor['height']}+{monitor['x']}+{monitor['y']}"
            )


class DatasetChooser(tk.Toplevel):
    def __init__(self, parent, entries, config_name=""):
        super().__init__(parent)
        self.entries = entries
        self.result = None

        self.title("Choose Color Set")
        self.geometry("760x480")
        self.transient(parent)
        self.grab_set()

        outer = tk.Frame(self, padx=24, pady=24)
        outer.pack(fill="both", expand=True)

        heading = "Choose a color dataset"
        if config_name:
            heading += f" - {config_name}"

        tk.Label(
            outer,
            text=heading,
            font=("Arial", 22, "bold"),
            anchor="w",
        ).pack(fill="x", pady=(0, 18))

        self.listbox = tk.Listbox(
            outer,
            font=("Arial", 18),
            activestyle="dotbox",
        )
        self.listbox.pack(fill="both", expand=True)

        for entry in entries:
            title = entry.get("title") or Path(entry["_resolved_path"]).name
            self.listbox.insert("end", title)

        self.listbox.selection_set(0)
        self.listbox.activate(0)
        self.listbox.bind("<Double-Button-1>", lambda _e: self.choose())
        self.listbox.bind("<Return>", lambda _e: self.choose())

        buttons = tk.Frame(outer)
        buttons.pack(fill="x", pady=(18, 0))

        tk.Button(
            buttons,
            text="Cancel",
            command=self.cancel,
            font=("Arial", 18),
            padx=24,
            pady=14,
        ).pack(side="left")

        tk.Button(
            buttons,
            text="Open",
            command=self.choose,
            font=("Arial", 18, "bold"),
            padx=32,
            pady=14,
        ).pack(side="right")

        self.protocol("WM_DELETE_WINDOW", self.cancel)

    def choose(self):
        selected = self.listbox.curselection()
        if not selected:
            return
        self.result = self.entries[selected[0]]
        self.destroy()

    def cancel(self):
        self.result = None
        self.destroy()


class ColorAdjuster:
    def __init__(
        self,
        root,
        input_path,
        dataset_title=None,
        config=None,
        config_path=None,
        config_entry=None,
    ):
        self.root = root
        self.input_path = Path(input_path).resolve()
        self.dataset_title = dataset_title or self.input_path.stem
        self.config = config or {}
        self.config_path = Path(config_path).resolve() if config_path else None
        self.config_entry = config_entry or {}

        self.colors, self.delimiter = read_color_file(self.input_path)
        if not self.colors:
            raise ValueError("No colors were found in the input file.")

        ext = ".tsv" if self.delimiter == "\t" else ".csv"
        stem = self.input_path.stem
        self.adjusted_path = self.input_path.with_name(f"{stem}_adjusted_rgb{ext}")
        self.delta_path = self.input_path.with_name(f"{stem}_delta_rgb{ext}")

        self.results = [None] * len(self.colors)
        self.current_index = 0

        self.root.title(f"RGB Color Adjuster - {self.dataset_title}")
        self.root.configure(bg="black")
        self.root.bind("<Escape>", self.exit_fullscreen)
        self.root.bind("<F11>", self.toggle_fullscreen)
        self.root.bind("<Return>", lambda _e: self.save_and_next())

        self.fullscreen = bool(self.config.get("fullscreen", True))
        self.monitor = choose_experimenter_monitor(self.root, self.config)

        self.build_ui()
        self.load_current_color()

        # Move first, then fullscreen after Tk has drawn the controls.
        self.root.after(
            100,
            lambda: place_fullscreen_on_monitor(
                self.root, self.monitor, self.fullscreen
            ),
        )

    def exit_fullscreen(self, _event=None):
        self.fullscreen = False
        try:
            self.root.attributes("-fullscreen", False)
        except tk.TclError:
            pass
        self.root.overrideredirect(False)

    def toggle_fullscreen(self, _event=None):
        self.fullscreen = not self.fullscreen
        place_fullscreen_on_monitor(self.root, self.monitor, self.fullscreen)

    def build_ui(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        # Compact status bar. No original-color swatch/value is shown.
        top = tk.Frame(self.root, padx=24, pady=14)
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(1, weight=1)

        self.progress_label = tk.Label(
            top,
            text="",
            font=("Arial", 22, "bold"),
            anchor="w",
        )
        self.progress_label.grid(row=0, column=0, sticky="w")

        self.id_label = tk.Label(
            top,
            text="",
            font=("Arial", 24, "bold"),
            anchor="e",
        )
        self.id_label.grid(row=0, column=1, sticky="e")

        # The adjusted color occupies almost all available screen area.
        self.adjusted_swatch = tk.Frame(
            self.root,
            bg="black",
            relief="flat",
            borderwidth=0,
        )
        self.adjusted_swatch.grid(
            row=1,
            column=0,
            sticky="nsew",
            padx=18,
            pady=(0, 12),
        )

        controls = tk.Frame(self.root, padx=28, pady=18)
        controls.grid(row=2, column=0, sticky="ew")
        controls.columnconfigure(1, weight=1)

        self.r_var = tk.IntVar()
        self.g_var = tk.IntVar()
        self.b_var = tk.IntVar()

        self.make_slider(controls, 0, "R", self.r_var)
        self.make_slider(controls, 1, "G", self.g_var)
        self.make_slider(controls, 2, "B", self.b_var)

        info = tk.Frame(controls)
        info.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(12, 12))
        info.columnconfigure(0, weight=1)
        info.columnconfigure(1, weight=1)

        self.adjusted_value_label = tk.Label(
            info,
            text="",
            font=("Courier", 20, "bold"),
            anchor="w",
        )
        self.adjusted_value_label.grid(row=0, column=0, sticky="w")

        self.delta_label = tk.Label(
            info,
            text="Delta: R +0   G +0   B +0",
            font=("Courier", 20, "bold"),
            anchor="e",
        )
        self.delta_label.grid(row=0, column=1, sticky="e")

        buttons = tk.Frame(controls)
        buttons.grid(row=4, column=0, columnspan=3, sticky="ew")
        buttons.columnconfigure(0, weight=1)
        buttons.columnconfigure(1, weight=2)

        self.reset_button = tk.Button(
            buttons,
            text="Reset",
            command=self.reset_current,
            font=("Arial", 22, "bold"),
            padx=24,
            pady=18,
        )
        self.reset_button.grid(row=0, column=0, padx=(0, 12), sticky="ew")

        self.next_button = tk.Button(
            buttons,
            text="Save & Next",
            command=self.save_and_next,
            font=("Arial", 22, "bold"),
            padx=28,
            pady=18,
        )
        self.next_button.grid(row=0, column=1, padx=(12, 0), sticky="ew")

    def make_slider(self, parent, row, label, variable):
        tk.Label(
            parent,
            text=label,
            font=("Arial", 26, "bold"),
            width=2,
        ).grid(row=row, column=0, sticky="w", pady=6)

        slider = tk.Scale(
            parent,
            from_=0,
            to=255,
            orient="horizontal",
            variable=variable,
            showvalue=False,
            resolution=1,
            width=48,
            sliderlength=82,
            borderwidth=3,
            highlightthickness=0,
            command=lambda _value: self.update_preview(),
        )
        slider.grid(row=row, column=1, sticky="ew", padx=18, pady=6)

        tk.Label(
            parent,
            textvariable=variable,
            width=4,
            font=("Courier", 24, "bold"),
        ).grid(row=row, column=2, pady=6)

    def load_current_color(self):
        color = self.colors[self.current_index]

        self.progress_label.config(
            text=f"Color {self.current_index + 1} of {len(self.colors)}"
        )
        self.id_label.config(text=f"ID: {color['id']}")

        existing = self.results[self.current_index]
        if existing is None:
            r, g, b = color["r"], color["g"], color["b"]
        else:
            r, g, b = existing["r"], existing["g"], existing["b"]

        self.r_var.set(r)
        self.g_var.set(g)
        self.b_var.set(b)
        self.update_preview()

        if self.current_index == len(self.colors) - 1:
            self.next_button.config(text="Save & Finish")
        else:
            self.next_button.config(text="Save & Next")

    def update_preview(self):
        r = clamp(self.r_var.get())
        g = clamp(self.g_var.get())
        b = clamp(self.b_var.get())

        adjusted_hex = rgb_to_hex(r, g, b)
        self.adjusted_swatch.config(bg=adjusted_hex)
        self.adjusted_value_label.config(
            text=f"RGB({r}, {g}, {b})   {adjusted_hex.upper()}"
        )

        original = self.colors[self.current_index]
        dr = r - original["r"]
        dg = g - original["g"]
        db = b - original["b"]

        self.delta_label.config(text=f"Delta: R {dr:+d}   G {dg:+d}   B {db:+d}")

    def reset_current(self):
        color = self.colors[self.current_index]
        self.r_var.set(color["r"])
        self.g_var.set(color["g"])
        self.b_var.set(color["b"])
        self.update_preview()

    def save_current_result(self):
        original = self.colors[self.current_index]

        r = clamp(self.r_var.get())
        g = clamp(self.g_var.get())
        b = clamp(self.b_var.get())

        self.results[self.current_index] = {
            "id": original["id"],
            "r": r,
            "g": g,
            "b": b,
            "delta_r": r - original["r"],
            "delta_g": g - original["g"],
            "delta_b": b - original["b"],
        }

        self.write_output_files()

    def write_output_files(self):
        completed = [result for result in self.results if result is not None]

        with self.adjusted_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["id", "r", "g", "b"],
                delimiter=self.delimiter,
            )
            writer.writeheader()
            for result in completed:
                writer.writerow({
                    "id": result["id"],
                    "r": result["r"],
                    "g": result["g"],
                    "b": result["b"],
                })

        with self.delta_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["id", "delta_r", "delta_g", "delta_b"],
                delimiter=self.delimiter,
            )
            writer.writeheader()
            for result in completed:
                writer.writerow({
                    "id": result["id"],
                    "delta_r": result["delta_r"],
                    "delta_g": result["delta_g"],
                    "delta_b": result["delta_b"],
                })

    def save_and_next(self):
        self.save_current_result()

        if self.current_index < len(self.colors) - 1:
            self.current_index += 1
            self.load_current_color()
            return

        # Avoid opening a dialog on another display at the end. The app simply
        # leaves the final color visible and disables editing buttons.
        self.next_button.config(state="disabled", text="Finished")
        self.reset_button.config(state="disabled")


def choose_input_file(root):
    return filedialog.askopenfilename(
        parent=root,
        title="Choose config or color file",
        filetypes=[
            ("Config / color files", "*.json *.tsv *.csv"),
            ("JSON config", "*.json"),
            ("TSV colors", "*.tsv"),
            ("CSV colors", "*.csv"),
            ("All files", "*.*"),
        ],
    )


def select_dataset(root, config, entries):
    if len(entries) == 1:
        return entries[0]

    chooser = DatasetChooser(
        root,
        entries,
        config_name=str(config.get("config_name", "")),
    )
    root.wait_window(chooser)
    return chooser.result


def main():
    root = tk.Tk()
    root.withdraw()

    input_arg = sys.argv[1] if len(sys.argv) >= 2 else choose_input_file(root)

    if not input_arg:
        root.destroy()
        return

    input_path = Path(input_arg)

    try:
        if input_path.suffix.lower() == ".json":
            config, entries = load_config(input_path)
            selected = select_dataset(root, config, entries)

            if selected is None:
                root.destroy()
                return

            color_path = selected["_resolved_path"]
            title = selected.get("title") or color_path.stem

            root.deiconify()
            ColorAdjuster(
                root,
                color_path,
                dataset_title=title,
                config=config,
                config_path=input_path,
                config_entry=selected,
            )
        else:
            root.deiconify()
            ColorAdjuster(root, input_path, config={"fullscreen": True})

        root.mainloop()

    except Exception as exc:
        try:
            messagebox.showerror("Error", str(exc))
        finally:
            root.destroy()
        raise


if __name__ == "__main__":
    main()
