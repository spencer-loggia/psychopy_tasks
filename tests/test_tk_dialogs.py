"""Exercise keyboard selection and dismissal without requiring a display."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from bin.tk_dialogs import bind_dialog_keys, choose_dialog, showerror, showwarning


def callbacks(widget):
    return {call.args[0]: call.args[1] for call in widget.bind.call_args_list}


class DialogKeyboardTests(unittest.TestCase):
    def test_arrows_select_buttons_and_enter_activates_once(self):
        dialog, cancel = Mock(), Mock()
        buttons = [Mock(), Mock(), Mock()]
        bind_dialog_keys(dialog, buttons, cancel=cancel)
        events = callbacks(buttons[0])
        self.assertEqual(events["<Right>"](SimpleNamespace(keysym="Right")), "break")
        self.assertEqual(events["<KP_Enter>"](SimpleNamespace(keysym="KP_Enter")), "break")
        buttons[1].invoke.assert_called_once_with()
        buttons[0].invoke.assert_not_called()
        buttons[2].invoke.assert_not_called()
        # Focus changes from mouse or Tab also change the keyboard selection.
        callbacks(buttons[2])["<FocusIn>"](None)
        events["<Return>"](SimpleNamespace(keysym="Return"))
        buttons[2].invoke.assert_called_once_with()
        cancel.assert_not_called()

    def test_all_arrow_directions_and_escape_work_without_a_modal_grab(self):
        dialog, cancel = Mock(), Mock()
        buttons = [Mock(), Mock(), Mock()]
        bind_dialog_keys(dialog, buttons, cancel=cancel)
        events = callbacks(dialog)
        for key in ("Left", "Down", "Up", "Right"):
            self.assertEqual(events[f"<{key}>"](SimpleNamespace(keysym=key)), "break")
        events["<Return>"](SimpleNamespace(keysym="Return"))
        buttons[0].invoke.assert_called_once_with()
        self.assertEqual(events["<Escape>"](SimpleNamespace(keysym="Escape")), "break")
        cancel.assert_called_once_with()
        dialog.grab_set.assert_not_called()

    def test_choice_dialog_close_cancels_and_restores_parent_focus(self):
        parent, dialog, body = Mock(), Mock(), Mock()
        parent.winfo_width.return_value = 800
        parent.winfo_height.return_value = 480
        parent.winfo_rootx.return_value = 1920
        parent.winfo_rooty.return_value = 0
        parent.grab_current.return_value = None

        def dismiss(_dialog):
            callbacks(body)["<Escape>"](SimpleNamespace(keysym="Escape"))

        parent.wait_window.side_effect = dismiss
        with patch("bin.tk_dialogs.tk.Toplevel", return_value=dialog), patch(
            "bin.tk_dialogs.tk.Label"
        ), patch("bin.tk_dialogs.tk.Frame"), patch(
            "bin.tk_dialogs.tk.Button", side_effect=lambda *args, **kwargs: Mock()
        ), patch("bin.tk_dialogs.ScrolledText", return_value=body):
            result = choose_dialog(parent, "Shutdown Cleanup", "Permission denied",
                                   [("Retry", True), ("Shut down anyway", False)], cancel_value=None)
        self.assertIsNone(result)
        dialog.destroy.assert_called_once_with()
        parent.focus_get.return_value.focus_set.assert_called_once_with()
        dialog.geometry.assert_called_once_with("760x420+1940+20")
        body.insert.assert_called_once_with("1.0", "Permission denied")

    def test_error_and_warning_use_keyboard_capable_dialog(self):
        parent = Mock()
        with patch("bin.tk_dialogs.choose_dialog", return_value="ok") as choose:
            for show in (showerror, showwarning):
                self.assertEqual(show("Failure", "Details", parent=parent), "ok")
                choose.assert_called_with(parent, "Failure", "Details", [("OK", "ok")], cancel_value="ok")


if __name__ == "__main__":
    unittest.main()
