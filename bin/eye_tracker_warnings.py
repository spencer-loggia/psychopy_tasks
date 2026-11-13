"""Nonmodal tracker warnings, isolated from task rendering and UART workers."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import threading


def confirm_without_eye_tracking(message: str, *, screen_config=None) -> bool:
    """Require an explicit choice on the experimenter display; failure cancels."""
    import json
    try:
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--confirm", message,
             json.dumps(screen_config)],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
        )
        return result.returncode == 0
    except Exception as exc:
        print(f"Could not confirm unavailable eye tracking: {exc}", file=sys.stderr)
        return False


def _confirm_main(message, screen_config):
    import tkinter as tk
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from bin.screen import load_screen_config, resolve_task_screens
    from bin.tk_dialogs import choose_dialog

    root = tk.Tk()
    try:
        main, secondary = resolve_task_screens(
            screen_config if screen_config is not None else load_screen_config({}),
            allow_same_screen=True,
        )
        screen = secondary or main
        root.geometry(f"760x460+{screen.x + 20}+{screen.y + 20}")
        root.title("Eye acquisition unavailable")
        root.attributes("-topmost", True)
        root.update_idletasks()
        accepted = choose_dialog(
            root, "Eye acquisition unavailable",
            message + "\n\nEye tracking/recording could not be confirmed. "
            "Continue this block with potentially missing eye data?",
            [("Cancel block", False), ("Continue without confirmed eye data", True)],
            cancel_value=False,
        )
        return 0 if accepted else 1
    finally:
        root.destroy()


def show_eye_tracker_warning(message: str) -> None:
    """Log immediately and open a dismissible dialog without waiting for it."""
    print(f"Eye tracker warning: {message}", file=sys.stderr)

    def launch():
        try:
            subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), message],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                start_new_session=True,
            )
        except Exception as exc:
            print(f"Could not display eye tracker warning: {exc}", file=sys.stderr)

    threading.Thread(target=launch, name="eye-warning", daemon=True).start()


class EyeTrackerWarnings:
    """Show each kind of failure once per session, even at high polling rates."""

    def __init__(self):
        self._shown: set[str] = set()
        self._lock = threading.Lock()

    def warn(self, key: str, message: str) -> None:
        self.warn_many({key: message})

    def warn_many(self, problems: dict[str, str]) -> None:
        with self._lock:
            new = {key: value for key, value in problems.items() if key not in self._shown}
            if not new:
                return
            self._shown.update(new)
        show_eye_tracker_warning("\n\n".join(new.values()))


def _main(message: str) -> None:
    import tkinter as tk
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from bin.tk_dialogs import bind_dialog_keys

    root = tk.Tk()
    root.title("Eye Tracker Warning")
    root.resizable(False, False)
    root.attributes("-topmost", True)
    tk.Label(
        root, text=message + "\n\nThe experiment and task can continue.",
        wraplength=500, justify="left", padx=24, pady=24, font=("Helvetica", 16),
    ).pack()
    dismiss = tk.Button(
        root, text="Dismiss", command=root.destroy,
        padx=24, pady=12, font=("Helvetica", 16),
    )
    dismiss.pack(pady=(0, 20))
    bind_dialog_keys(root, [dismiss], cancel=root.destroy)
    # Place the dialog on the launcher's experimenter display when available.
    try:
        from bin.screen import load_screen_config, resolve_task_screens

        main, secondary = resolve_task_screens(load_screen_config({}), allow_same_screen=True)
        screen = secondary or main
        root.geometry(f"+{screen.x + 40}+{screen.y + 40}")
    except Exception:
        pass
    # No grab, wait_window, or modal messagebox: dismissing is never required.
    root.mainloop()


if __name__ == "__main__":
    try:
        if sys.argv[1] == "--confirm":
            import json
            sys.exit(_confirm_main(sys.argv[2], json.loads(sys.argv[3])))
        else:
            _main(sys.argv[1])
    except Exception as exc:
        print(f"Could not display eye tracker warning: {exc}", file=sys.stderr)
        sys.exit(1)
