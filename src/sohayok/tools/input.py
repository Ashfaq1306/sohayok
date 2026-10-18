"""Keyboard and mouse control."""

from __future__ import annotations

import time

from pynput.mouse import Button, Controller

from sohayok import winapi
from sohayok.tools.registry import ToolError, integer, string, tool

_mouse = Controller()
_BUTTONS = {"left": Button.left, "right": Button.right, "middle": Button.middle}


@tool(
    "Type text into the focused window, exactly as given (Bangla, English, or symbols). "
    "Use for dictation: 'write ...', 'লেখো ...'.",
    {
        "text": string("The text to type"),
        "press_enter": {"type": "boolean", "description": "Press Enter after"},
    },
    required=["text"],
)
def type_text(text: str, press_enter: bool = False) -> str:
    note = _prepare_target()
    if len(text) > 40 or any(ord(c) > 127 for c in text):
        # Pasting keeps Bangla conjuncts intact; per-character Unicode keystrokes can be
        # reordered or dropped by apps that are still busy.
        paste_text(text)
    else:
        winapi.type_unicode(text)
    if press_enter:
        winapi.press("enter")
    return f"typed {len(text)} characters into '{note}'"


def _prepare_target() -> str:
    """Describe the focused control and collapse any selection, so typing never replaces it."""
    import uiautomation as auto

    with auto.UIAutomationInitializerInThread():
        focused = auto.GetFocusedControl()
        if focused is None:
            raise ToolError("nothing has keyboard focus; click a text field first")
        label = focused.Name or focused.ControlTypeName.removesuffix("Control")
        try:
            pattern = focused.GetPattern(auto.PatternId.TextPattern)
            if pattern and any(r.GetText(5) for r in pattern.GetSelection()):
                winapi.press("right")  # cursor to the end of the selection, nothing deleted
        except Exception:
            pass
        window = focused.GetTopLevelControl()
        return f"{label} in {window.Name}" if window and window.Name else label


def _clipboard_text() -> str | None:
    import win32clipboard

    win32clipboard.OpenClipboard()
    try:
        if win32clipboard.IsClipboardFormatAvailable(win32clipboard.CF_UNICODETEXT):
            return win32clipboard.GetClipboardData(win32clipboard.CF_UNICODETEXT)
        return None
    finally:
        win32clipboard.CloseClipboard()


def _set_clipboard(text: str) -> None:
    import win32clipboard

    for attempt in range(10):  # another app may hold the clipboard for a moment
        try:
            win32clipboard.OpenClipboard()
            break
        except Exception:
            if attempt == 9:
                raise
            time.sleep(0.05)
    try:
        win32clipboard.EmptyClipboard()
        win32clipboard.SetClipboardText(text, win32clipboard.CF_UNICODETEXT)
    finally:
        win32clipboard.CloseClipboard()


def paste_text(text: str) -> None:
    """Paste through the clipboard, then put the user's previous clipboard text back."""
    try:
        previous = _clipboard_text()
    except Exception:
        previous = None
    _set_clipboard(text)
    winapi.press("ctrl+v")
    time.sleep(0.3)  # the target reads the clipboard asynchronously
    if previous is not None:
        _set_clipboard(previous)


@tool(
    "Press a key or shortcut such as 'enter', 'ctrl+c', 'ctrl+shift+t', 'alt+tab', 'win+e', "
    "'f5', 'volumeup'. Use repeat for several presses.",
    {"keys": string("Key combination joined with +"), "repeat": integer("How many times (1-50)")},
    required=["keys"],
)
def press_keys(keys: str, repeat: int = 1) -> str:
    combo = {k.strip().lower() for k in keys.replace("-", "+").split("+")}
    if {"ctrl", "alt"} <= combo and combo & {"del", "delete"}:
        raise ToolError("Ctrl+Alt+Delete is reserved by Windows and cannot be pressed for you")
    try:
        winapi.press(keys, max(1, min(int(repeat), 50)))
    except ValueError as exc:
        raise ToolError(str(exc)) from exc
    return f"pressed {keys}" + (f" ×{repeat}" if repeat > 1 else "")


@tool(
    "Click the mouse at screen pixel coordinates (from describe_screen or a screenshot).",
    {
        "x": integer("X in physical pixels"),
        "y": integer("Y in physical pixels"),
        "button": string("Mouse button", ["left", "right", "middle"]),
        "double": {"type": "boolean", "description": "Double-click"},
    },
    required=["x", "y"],
)
def click(x: int, y: int, button: str = "left", double: bool = False) -> str:
    width, height = winapi.screen_size()
    if not (0 <= x < width * 2 and 0 <= y < height * 2):
        raise ToolError(f"({x}, {y}) is off screen")
    _mouse.position = (int(x), int(y))
    time.sleep(0.05)
    _mouse.click(_BUTTONS.get(button, Button.left), 2 if double else 1)
    return f"{'double-' if double else ''}clicked {button} at ({x}, {y})"


@tool(
    "Scroll the mouse wheel in the window under the pointer.",
    {
        "direction": string("Direction", ["up", "down", "left", "right"]),
        "amount": integer("Number of wheel steps (1-30)"),
    },
    required=["direction"],
)
def scroll(direction: str, amount: int = 5) -> str:
    amount = max(1, min(int(amount), 30))
    dx, dy = {
        "up": (0, amount),
        "down": (0, -amount),
        "left": (-amount, 0),
        "right": (amount, 0),
    }.get(direction, (0, -amount))
    _mouse.scroll(dx, dy)
    return f"scrolled {direction} {amount}"


@tool(
    "Move the mouse pointer to screen coordinates.",
    {"x": integer("X in physical pixels"), "y": integer("Y in physical pixels")},
)
def move_mouse(x: int, y: int) -> str:
    _mouse.position = (int(x), int(y))
    return f"moved to ({x}, {y})"
