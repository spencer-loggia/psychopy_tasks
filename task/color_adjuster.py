#!/usr/bin/env python3
"""
RGB Color Adjuster for Raspberry Pi

Usage:
    python3 rgb_color_adjuster.py
    python3 rgb_color_adjuster.py colors.tsv
    python3 rgb_color_adjuster.py gallery_config.json

Supported direct color files:
    CSV or TSV with required columns:
        id, r, g, b

Supported JSON config:
    {
      "config_name": "gallery",
      "colors_tsv": [
        {
          "path": "./task/resources/colors.tsv",
          "title": "My Colors",
          ...
        }
      ]
    }

If the JSON config contains multiple colors_tsv entries, the app asks which
dataset to open.

Relative paths in the JSON config are resolved relative to the config file.

Outputs:
    <input_stem>_adjusted_rgb.csv/tsv
    <input_stem>_delta_rgb.csv/tsv

Each click of "Save & Next" immediately updates both output files.
"""

import csv
import json
import re
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
                        raise ValueError(
                            f"Invalid RGB value on row {row_number}."
                        )

                    colors.append({
                        "id": color_id,
                        "r": r,
                        "g": g,
                        "b": b,
                    })

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
        raise ValueError(
            'Config must contain a non-empty "colors_tsv" array.'
        )

    resolved_entries = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(
                f'colors_tsv entry {index + 1} must be an object.'
            )

        rel_path = entry.get("path")
        if not rel_path:
            raise ValueError(
                f'colors_tsv entry {index + 1} is missing "path".'
            )

        color_path = Path(rel_path)
        if not color_path.is_absolute():
            color_path = (config_path.parent / color_path).resolve()

        resolved = dict(entry)
        resolved["_resolved_path"] = color_path
        resolved["_index"] = index
        resolved_entries.append(resolved)

    return config, resolved_entries


class DatasetChooser(tk.Toplevel):
    def __init__(self, parent, entries, config_name=""):
        super().__init__(parent)
        self.parent = parent
        self.entries = entries
        self.result = None

        self.title("Choose Color Set")
        self.geometry("620x360")
        self.resizable(True, True)
        self.transient(parent)
        self.grab_set()

        outer = tk.Frame(self, padx=16, pady=16)
        outer.pack(fill="both", expand=True)

        heading = "Choose a color dataset"
        if config_name:
            heading += f" — {config_name}"

        tk.Label(
            outer,
            text=heading,
            font=("Arial", 16, "bold"),
            anchor="w",
        ).pack(fill="x", pady=(0, 12))

        self.listbox = tk.Listbox(
            outer,
            font=("Arial", 12),
            activestyle="dotbox",
        )
        self.listbox.pack(fill="both", expand=True)

        for entry in entries:
            title = entry.get("title") or Path(entry["_resolved_path"]).name
            path_text = entry.get("path", "")
            self.listbox.insert("end", f"{title}    [{path_text}]")

        self.listbox.selection_set(0)
        self.listbox.activate(0)
        self.listbox.bind("<Double-Button-1>", lambda _e: self.choose())
        self.listbox.bind("<Return>", lambda _e: self.choose())

        buttons = tk.Frame(outer)
        buttons.pack(fill="x", pady=(12, 0))

        tk.Button(
            buttons,
            text="Cancel",
            command=self.cancel,
            font=("Arial", 12),
            padx=16,
            pady=8,
        ).pack(side="left")

        tk.Button(
            buttons,
            text="Open",
            command=self.choose,
            font=("Arial", 12, "bold"),
            padx=24,
            pady=8,
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
        config_path=None,
        config_entry=None,
    ):
        self.root = root
        self.input_path = Path(input_path).resolve()
        self.dataset_title = dataset_title or self.input_path.stem
        self.config_path = Path(config_path).resolve() if config_path else None
        self.config_entry = config_entry or {}

        self.colors, self.delimiter = read_color_file(self.input_path)

        if not self.colors:
            raise ValueError("No colors were found in the input file.")

        ext = ".tsv" if self.delimiter == "\t" else ".csv"
        stem = self.input_path.stem
        self.adjusted_path = self.input_path.with_name(
            f"{stem}_adjusted_rgb{ext}"
        )
        self.delta_path = self.input_path.with_name(
            f"{stem}_delta_rgb{ext}"
        )

        self.results = [None] * len(self.colors)
        self.current_index = 0

        self.root.title(f"RGB Color Adjuster — {self.dataset_title}")
        self.root.geometry("900x680")
        self.root.minsize(760, 600)

        self.build_ui()
        self.load_current_color()

    def build_ui(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        top = tk.Frame(self.root, padx=16, pady=12)
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(1, weight=1)

        self.progress_label = tk.Label(
            top, text="", font=("Arial", 14, "bold"), anchor="w"
        )
        self.progress_label.grid(row=0, column=0, sticky="w")

        self.name_label = tk.Label(
            top, text="", font=("Arial", 18, "bold"), anchor="e"
        )
        self.name_label.grid(row=0, column=1, sticky="e")

        if self.dataset_title:
            tk.Label(
                top,
                text=self.dataset_title,
                font=("Arial", 11),
                anchor="w",
            ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 0))

        preview = tk.Frame(self.root, padx=16, pady=8)
        preview.grid(row=1, column=0, sticky="nsew")
        preview.columnconfigure(0, weight=1)
        preview.columnconfigure(1, weight=1)
        preview.rowconfigure(1, weight=1)

        tk.Label(preview, text="Original", font=("Arial", 13, "bold")).grid(
            row=0, column=0, pady=(0, 6)
        )
        tk.Label(preview, text="Adjusted", font=("Arial", 13, "bold")).grid(
            row=0, column=1, pady=(0, 6)
        )

        self.original_swatch = tk.Frame(
            preview, relief="solid", borderwidth=2, width=300, height=300
        )
        self.original_swatch.grid(
            row=1, column=0, padx=(0, 8), sticky="nsew"
        )
        self.original_swatch.grid_propagate(False)

        self.adjusted_swatch = tk.Frame(
            preview, relief="solid", borderwidth=2, width=300, height=300
        )
        self.adjusted_swatch.grid(
            row=1, column=1, padx=(8, 0), sticky="nsew"
        )
        self.adjusted_swatch.grid_propagate(False)

        self.original_value_label = tk.Label(
            preview, text="", font=("Courier", 12)
        )
        self.original_value_label.grid(row=2, column=0, pady=(8, 0))

        self.adjusted_value_label = tk.Label(
            preview, text="", font=("Courier", 12)
        )
        self.adjusted_value_label.grid(row=2, column=1, pady=(8, 0))

        controls = tk.Frame(self.root, padx=20, pady=12)
        controls.grid(row=2, column=0, sticky="ew")
        controls.columnconfigure(1, weight=1)

        self.r_var = tk.IntVar()
        self.g_var = tk.IntVar()
        self.b_var = tk.IntVar()

        self.make_slider(controls, 0, "R", self.r_var)
        self.make_slider(controls, 1, "G", self.g_var)
        self.make_slider(controls, 2, "B", self.b_var)

        self.delta_label = tk.Label(
            controls,
            text="Delta: R +0   G +0   B +0",
            font=("Courier", 12, "bold"),
            anchor="center",
        )
        self.delta_label.grid(
            row=3, column=0, columnspan=3, sticky="ew", pady=(6, 10)
        )

        buttons = tk.Frame(controls)
        buttons.grid(row=4, column=0, columnspan=3, sticky="ew")
        buttons.columnconfigure(0, weight=1)
        buttons.columnconfigure(1, weight=1)

        self.reset_button = tk.Button(
            buttons,
            text="Reset",
            command=self.reset_current,
            font=("Arial", 13),
            padx=16,
            pady=10,
        )
        self.reset_button.grid(row=0, column=0, padx=(0, 8), sticky="ew")

        self.next_button = tk.Button(
            buttons,
            text="Save & Next",
            command=self.save_and_next,
            font=("Arial", 13, "bold"),
            padx=16,
            pady=10,
        )
        self.next_button.grid(row=0, column=1, padx=(8, 0), sticky="ew")

        footer_text = (
            f"Input: {self.input_path.name}    "
            f"Outputs: {self.adjusted_path.name}, {self.delta_path.name}"
        )
        footer = tk.Label(
            self.root,
            text=footer_text,
            anchor="w",
            padx=16,
            pady=8,
        )
        footer.grid(row=3, column=0, sticky="ew")

    def make_slider(self, parent, row, label, variable):
        tk.Label(
            parent,
            text=label,
            font=("Arial", 14, "bold"),
            width=2,
        ).grid(row=row, column=0, sticky="w")

        slider = tk.Scale(
            parent,
            from_=0,
            to=255,
            orient="horizontal",
            variable=variable,
            showvalue=False,
            resolution=1,
            command=lambda _value: self.update_preview(),
        )
        slider.grid(row=row, column=1, sticky="ew", padx=8)

        tk.Label(
            parent,
            textvariable=variable,
            width=4,
            font=("Courier", 12),
        ).grid(row=row, column=2)

    def load_current_color(self):
        color = self.colors[self.current_index]

        self.progress_label.config(
            text=f"Color {self.current_index + 1} of {len(self.colors)}"
        )
        self.name_label.config(text=f"ID: {color['id']}")

        original_hex = rgb_to_hex(color["r"], color["g"], color["b"])
        self.original_swatch.config(bg=original_hex)
        self.original_value_label.config(
            text=f"RGB({color['r']}, {color['g']}, {color['b']})   "
                 f"{original_hex.upper()}"
        )

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

        self.delta_label.config(
            text=f"Delta: R {dr:+d}   G {dg:+d}   B {db:+d}"
        )

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

        messagebox.showinfo(
            "Finished",
            "All colors have been saved.\n\n"
            f"Adjusted RGB:\n{self.adjusted_path}\n\n"
            f"RGB deltas:\n{self.delta_path}",
        )
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
                config_path=input_path,
                config_entry=selected,
            )
        else:
            root.deiconify()
            ColorAdjuster(root, input_path)

        root.mainloop()

    except Exception as exc:
        messagebox.showerror("Error", str(exc))
        root.destroy()
        raise


if __name__ == "__main__":
    main()
