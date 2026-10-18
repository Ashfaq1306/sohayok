"""List, switch, arrange, and close windows."""

from __future__ import annotations

from dataclasses import dataclass

import psutil
import win32con
import win32gui
import win32process
from rapidfuzz import fuzz, process

from sohayok import winapi
from sohayok.lang import normalize
from sohayok.tools.registry import ToolError, string, tool

SKIP_TITLES = {"Program Manager", "Windows Input Experience"}


@dataclass
class Window:
    hwnd: int
    title: str
    process: str
    pid: int


def _is_app_window(hwnd: int) -> bool:
    if not win32gui.IsWindowVisible(hwnd) or not win32gui.GetWindowText(hwnd):
        return False
    if win32gui.GetWindow(hwnd, win32con.GW_OWNER):
        return False
    ex_style = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
    if ex_style & win32con.WS_EX_TOOLWINDOW:
        return False
    # Suspended Store apps keep invisible "cloaked" frames; DWMWA_CLOAKED = 14.
    import ctypes

    cloaked = ctypes.c_int(0)
    ctypes.windll.dwmapi.DwmGetWindowAttribute(hwnd, 14, ctypes.byref(cloaked), 4)
    return cloaked.value == 0


def list_window_handles() -> list[Window]:
    found: list[Window] = []

    def visit(hwnd: int, _: object) -> bool:
        if _is_app_window(hwnd):
            title = win32gui.GetWindowText(hwnd)
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            try:
                name = psutil.Process(pid).name()
            except psutil.Error:
                name = "?"
            if title not in SKIP_TITLES:
                found.append(Window(hwnd, title, name, pid))
        return True

    win32gui.EnumWindows(visit, None)
    return found


def find_window(query: str | None) -> Window:
    """The window whose title or program best matches `query`; the active one if empty."""
    windows = list_window_handles()
    if not query or normalize(query) in {"this", "current", "active", "এই", "এটা", "এটি", "বর্তমান"}:
        hwnd = winapi.foreground_window()
        for win in windows:
            if win.hwnd == hwnd:
                return win
        if windows:
            return windows[0]
        raise ToolError("no window is open")
    labels = [f"{w.title} — {w.process.removesuffix('.exe')}" for w in windows]
    best = process.extractOne(normalize(query), labels, scorer=fuzz.WRatio, processor=normalize)
    if not best or best[1] < 60:
        raise ToolError(f"no open window matches {query!r}")
    return windows[best[2]]


def close_handle(hwnd: int) -> None:
    win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)


@tool("List the open windows (title and program).")
def list_windows() -> list[dict[str, str]]:
    return [{"title": w.title, "program": w.process} for w in list_window_handles()]


@tool(
    "Bring a window to the front by (part of) its title or program name.",
    {"query": string("Window title or program, e.g. 'Chrome', 'Downloads', 'Word'")},
)
def switch_to_window(query: str) -> str:
    win = find_window(query)
    winapi.force_foreground(win.hwnd)
    return f"switched to {win.title}"


@tool(
    "Minimize, maximize, restore, or close a window. Without a query, acts on the active window.",
    {
        "action": string("What to do", ["minimize", "maximize", "restore", "close"]),
        "query": string("Window title or program; empty for the active window"),
    },
    required=["action"],
)
def window_action(action: str, query: str = "") -> str:
    win = find_window(query)
    if action == "close":
        close_handle(win.hwnd)
    else:
        command = {
            "minimize": win32con.SW_MINIMIZE,
            "maximize": win32con.SW_MAXIMIZE,
            "restore": win32con.SW_RESTORE,
        }.get(action)
        if command is None:
            raise ToolError(f"unknown window action {action!r}")
        win32gui.ShowWindow(win.hwnd, command)
    return f"{action}d {win.title}"


@tool(
    "Snap the active window to the left or right half of the screen.",
    {"side": string("Which half", ["left", "right"])},
)
def snap_window(side: str) -> str:
    winapi.press(f"win+{side}")
    return f"snapped to the {side}"


@tool("Show the desktop by minimizing all windows (again to bring them back).")
def show_desktop() -> str:
    winapi.press("win+d")
    return "toggled the desktop"


@tool("Open the Windows task view to see all windows.")
def task_view() -> str:
    winapi.press("win+tab")
    return "opened task view"
