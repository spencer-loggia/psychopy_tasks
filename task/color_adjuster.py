#!/usr/bin/env python3
"""
HSV Color Adjuster for Raspberry Pi

- Reads CSV/TSV with required columns: id, r, g, b
- Can load a JSON config containing colors_tsv entries
- Shows only the adjusted color
- Uses Hue / Saturation / Brightness sliders
- Converts HSV back to 8-bit RGB for saving
- Saves adjusted RGB and delta RGB after every Save & Next
- Opens fullscreen on the configured MAIN display
- Throttles preview redraws to reduce Raspberry Pi slider lag
"""

import colorsys
import csv
import json
import re
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

PREVIEW_DELAY_MS = 30


def clamp(value, low=0, high=255):
    return max(low, min(high, int(round(float(value)))))


def rgb_to_hex(r, g, b):
    return f"#{r:02x}{g:02x}{b:02x}"


def rgb_to_hsv(r, g, b):
    h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    return h * 360.0, s * 100.0, v * 100.0


def hsv_to_rgb(h, s, v):
    h = float(h) % 360.0
    s = max(0.0, min(100.0, float(s)))
    v = max(0.0, min(100.0, float(v)))
    r, g, b = colorsys.hsv_to_rgb(h / 360.0, s / 100.0, v / 100.0)
    return clamp(r * 255.0), clamp(g * 255.0), clamp(b * 255.0)


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
                        use_delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t").delimiter
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
                        r = clamp(row[r_col])
                        g = clamp(row[g_col])
                        b = clamp(row[b_col])
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
    monitors = []
    try:
        result = subprocess.run(
            ["xrandr", "--listmonitors"], capture_output=True, text=True,
            check=False, timeout=2,
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
                        "index": int(index), "name": name, "x": int(x), "y": int(y),
                        "width": int(width), "height": int(height), "primary": "*" in flags,
                    })
    except (OSError, subprocess.SubprocessError):
        pass

    if not monitors:
        try:
            result = subprocess.run(
                ["xrandr", "--query"], capture_output=True, text=True,
                check=False, timeout=2,
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
                            "index": index, "name": name, "x": int(x), "y": int(y),
                            "width": int(width), "height": int(height), "primary": bool(primary),
                        })
                        index += 1
        except (OSError, subprocess.SubprocessError):
            pass

    if not monitors:
        monitors = [{
            "index": 0, "name": "default", "x": 0, "y": 0,
            "width": root.winfo_screenwidth(), "height": root.winfo_screenheight(),
            "primary": True,
        }]

    return monitors


def choose_main_monitor(root, config):
    monitors = get_connected_monitors(root)
    setting = (config or {}).get("screens", {}).get("main")

    if setting is None or str(setting).strip().lower() in {"auto", "primary"}:
        primary = next((m for m in monitors if m.get("primary")), None)
        return primary or monitors[0]

    if isinstance(setting, int):
        if 0 <= setting < len(monitors):
            return monitors[setting]
        raise ValueError(
            f"Main screen index {setting} does not exist. Connected monitors: {[m['name'] for m in monitors]}"
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
        f"Main screen '{setting}' was not found. Connected monitors: {[m['name'] for m in monitors]}"
    )


def place_fullscreen_on_monitor(root, monitor, fullscreen=True):
    root.attributes("-fullscreen", False)
    root.overrideredirect(False)
    root.geometry(
        f"{monitor['width']}x{monitor['height']}+{monitor['x']}+{monitor['y']}"
    )
    root.update_idletasks()
    root.update()
    if fullscreen:
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
        tk.Label(outer, text=heading, font=("Arial", 22, "bold"), anchor="w").pack(fill="x", pady=(0, 18))

        self.listbox = tk.Listbox(outer, font=("Arial", 18), activestyle="dotbox")
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
        tk.Button(buttons, text="Cancel", command=self.cancel, font=("Arial", 18), padx=24, pady=14).pack(side="left")
        tk.Button(buttons, text="Open", command=self.choose, font=("Arial", 18, "bold"), padx=32, pady=14).pack(side="right")
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


class HSVColorAdjuster:
    def __init__(self, root, input_path, dataset_title=None, config=None, config_path=None, config_entry=None):
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
        self.preview_job = None

        self.root.title(f"HSV Color Adjuster - {self.dataset_title}")
        self.root.configure(bg="black")
        self.root.bind("<Escape>", self.exit_fullscreen)
        self.root.bind("<F11>", self.toggle_fullscreen)
        self.root.bind("<Return>", lambda _e: self.save_and_next())

        self.fullscreen = bool(self.config.get("fullscreen", True))
        self.monitor = choose_main_monitor(self.root, self.config)

        self.build_ui()
        self.load_current_color()
        self.root.after(100, lambda: place_fullscreen_on_monitor(self.root, self.monitor, self.fullscreen))

    def exit_fullscreen(self, _event=None):
        self.fullscreen = False
        try:
            self.root.attributes("-fullscreen", False)
        except tk.TclError:
            pass
        self.root.overrideredirect(False)

    def close_app(self):
        """Close the color adjuster cleanly."""
        if self.preview_job is not None:
            try:
                self.root.after_cancel(self.preview_job)
            except tk.TclError:
                pass
            self.preview_job = None
        self.root.destroy()

    def toggle_fullscreen(self, _event=None):
        self.fullscreen = not self.fullscreen
        place_fullscreen_on_monitor(self.root, self.monitor, self.fullscreen)

    def build_ui(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        top = tk.Frame(self.root, padx=24, pady=14)
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(1, weight=1)

        self.progress_label = tk.Label(top, text="", font=("Arial", 22, "bold"), anchor="w")
        self.progress_label.grid(row=0, column=0, sticky="w")
        self.id_label = tk.Label(top, text="", font=("Arial", 24, "bold"), anchor="e")
        self.id_label.grid(row=0, column=1, sticky="e")

        self.adjusted_swatch = tk.Frame(self.root, bg="black", relief="flat", borderwidth=0)
        self.adjusted_swatch.grid(row=1, column=0, sticky="nsew", padx=18, pady=(0, 12))

        controls = tk.Frame(self.root, padx=28, pady=18)
        controls.grid(row=2, column=0, sticky="ew")
        controls.columnconfigure(1, weight=1)

        self.h_var = tk.DoubleVar()
        self.s_var = tk.DoubleVar()
        self.v_var = tk.DoubleVar()

        self.make_slider(controls, 0, "H", self.h_var, 0, 359, "°")
        self.make_slider(controls, 1, "S", self.s_var, 0, 100, "%")
        self.make_slider(controls, 2, "B", self.v_var, 0, 100, "%")

        info = tk.Frame(controls)
        info.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(12, 12))
        info.columnconfigure(0, weight=1)
        info.columnconfigure(1, weight=1)

        self.adjusted_value_label = tk.Label(info, text="", font=("Courier", 20, "bold"), anchor="w")
        self.adjusted_value_label.grid(row=0, column=0, sticky="w")
        self.delta_label = tk.Label(info, text="Delta: R +0   G +0   B +0", font=("Courier", 20, "bold"), anchor="e")
        self.delta_label.grid(row=0, column=1, sticky="e")

        buttons = tk.Frame(controls)
        buttons.grid(row=4, column=0, columnspan=3, sticky="ew")
        buttons.columnconfigure(0, weight=1)
        buttons.columnconfigure(1, weight=1)
        buttons.columnconfigure(2, weight=2)

        self.exit_button = tk.Button(
            buttons,
            text="Exit",
            command=self.close_app,
            font=("Arial", 22, "bold"),
            padx=24,
            pady=18,
        )
        self.exit_button.grid(row=0, column=0, padx=(0, 12), sticky="ew")

        self.reset_button = tk.Button(
            buttons,
            text="Reset",
            command=self.reset_current,
            font=("Arial", 22, "bold"),
            padx=24,
            pady=18,
        )
        self.reset_button.grid(row=0, column=1, padx=12, sticky="ew")

        self.next_button = tk.Button(
            buttons,
            text="Save & Next",
            command=self.save_and_next,
            font=("Arial", 22, "bold"),
            padx=28,
            pady=18,
        )
        self.next_button.grid(row=0, column=2, padx=(12, 0), sticky="ew")

    def make_slider(self, parent, row, label, variable, from_value, to_value, suffix):
        tk.Label(parent, text=label, font=("Arial", 26, "bold"), width=2).grid(row=row, column=0, sticky="w", pady=7)

        slider = tk.Scale(
            parent, from_=from_value, to=to_value, orient="horizontal",
            variable=variable, showvalue=False, resolution=1,
            width=48, sliderlength=90, borderwidth=3, highlightthickness=0,
            command=self.slider_changed,
        )
        slider.grid(row=row, column=1, sticky="ew", padx=18, pady=7)

        value_label = tk.Label(parent, text="", width=6, font=("Courier", 24, "bold"))
        value_label.grid(row=row, column=2, pady=7)

        def update_text(*_args):
            value_label.config(text=f"{int(round(variable.get()))}{suffix}")

        variable.trace_add("write", update_text)
        update_text()
        slider.bind("<ButtonRelease-1>", lambda _e: self.update_preview())

    def slider_changed(self, _value=None):
        if self.preview_job is not None:
            try:
                self.root.after_cancel(self.preview_job)
            except tk.TclError:
                pass
        self.preview_job = self.root.after(PREVIEW_DELAY_MS, self.update_preview)

    def load_current_color(self):
        color = self.colors[self.current_index]
        self.progress_label.config(text=f"Color {self.current_index + 1} of {len(self.colors)}")
        self.id_label.config(text=f"ID: {color['id']}")

        existing = self.results[self.current_index]
        if existing is None:
            r, g, b = color["r"], color["g"], color["b"]
        else:
            r, g, b = existing["r"], existing["g"], existing["b"]

        h, s, v = rgb_to_hsv(r, g, b)
        self.h_var.set(round(h))
        self.s_var.set(round(s))
        self.v_var.set(round(v))
        self.update_preview()

        self.next_button.config(text="Save & Finish" if self.current_index == len(self.colors) - 1 else "Save & Next")

    def get_current_rgb(self):
        return hsv_to_rgb(self.h_var.get(), self.s_var.get(), self.v_var.get())

    def update_preview(self):
        self.preview_job = None
        r, g, b = self.get_current_rgb()
        adjusted_hex = rgb_to_hex(r, g, b)
        self.adjusted_swatch.config(bg=adjusted_hex)
        self.adjusted_value_label.config(text=f"RGB({r}, {g}, {b})   {adjusted_hex.upper()}")

        original = self.colors[self.current_index]
        dr = r - original["r"]
        dg = g - original["g"]
        db = b - original["b"]
        self.delta_label.config(text=f"Delta: R {dr:+d}   G {dg:+d}   B {db:+d}")

    def reset_current(self):
        color = self.colors[self.current_index]
        h, s, v = rgb_to_hsv(color["r"], color["g"], color["b"])
        self.h_var.set(round(h))
        self.s_var.set(round(s))
        self.v_var.set(round(v))
        self.update_preview()

    def save_current_result(self):
        if self.preview_job is not None:
            try:
                self.root.after_cancel(self.preview_job)
            except tk.TclError:
                pass
            self.preview_job = None

        r, g, b = self.get_current_rgb()
        original = self.colors[self.current_index]
        self.results[self.current_index] = {
            "id": original["id"], "r": r, "g": g, "b": b,
            "delta_r": r - original["r"], "delta_g": g - original["g"], "delta_b": b - original["b"],
        }
        self.write_output_files()

    def write_output_files(self):
        completed = [result for result in self.results if result is not None]

        with self.adjusted_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "r", "g", "b"], delimiter=self.delimiter)
            writer.writeheader()
            for result in completed:
                writer.writerow({"id": result["id"], "r": result["r"], "g": result["g"], "b": result["b"]})

        with self.delta_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "delta_r", "delta_g", "delta_b"], delimiter=self.delimiter)
            writer.writeheader()
            for result in completed:
                writer.writerow({
                    "id": result["id"], "delta_r": result["delta_r"],
                    "delta_g": result["delta_g"], "delta_b": result["delta_b"],
                })

    def save_and_next(self):
        self.save_current_result()
        if self.current_index < len(self.colors) - 1:
            self.current_index += 1
            self.load_current_color()
            return
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
    chooser = DatasetChooser(root, entries, config_name=str(config.get("config_name", "")))
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
            HSVColorAdjuster(
                root, color_path, dataset_title=title, config=config,
                config_path=input_path, config_entry=selected,
            )
        else:
            root.deiconify()
            HSVColorAdjuster(
                root, input_path,
                config={"fullscreen": True, "screens": {"main": None}},
            )

        root.mainloop()

    except Exception as exc:
        try:
            messagebox.showerror("Error", str(exc))
        finally:
            root.destroy()
        raise


if __name__ == "__main__":
    main()
