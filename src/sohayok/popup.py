"""The desktop shell: a tray icon and a floating pop-up window, like Siri.

Sohayok runs in the background from sign-in until shutdown. The window stays hidden until
you say the wake word, press the hotkey, or click the tray icon; it then appears above the
taskbar, listens, answers, and slides away again when you are done.
"""

from __future__ import annotations

import ctypes
import logging
import threading
from collections.abc import Callable
from typing import Any

log = logging.getLogger(__name__)
TITLE = "Sohayok"
WIDTH, HEIGHT = 420, 660


def work_area() -> tuple[int, int, int, int]:
    """The primary monitor's usable area (left, top, right, bottom), taskbar excluded."""
    from ctypes import wintypes

    rect = wintypes.RECT()
    ctypes.windll.user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(rect), 0)  # SPI_GETWORKAREA
    return rect.left, rect.top, rect.right, rect.bottom


def dpi_scale() -> float:
    try:
        return ctypes.windll.user32.GetDpiForSystem() / 96
    except AttributeError:
        return 1.0


class Bridge:
    """Methods the page can call through window.pywebview.api."""

    def __init__(self, shell: DesktopShell) -> None:
        self._shell = shell

    def hide(self) -> None:
        self._shell.hide()

    def quit(self) -> None:
        self._shell.quit()

    def is_desktop(self) -> bool:
        return True


class DesktopShell:
    def __init__(
        self, url: str, on_listen: Callable[[], None], on_quit: Callable[[], None]
    ) -> None:
        self.url = url
        self.on_listen = on_listen
        self.on_quit = on_quit
        self.window: Any = None
        self.tray: Any = None
        self.visible = False
        self._hwnd = 0

    # ---- window ---------------------------------------------------------------------------
    def _position(self) -> tuple[int, int]:
        left, top, right, bottom = work_area()
        scale = dpi_scale()
        margin = int(16 * scale)
        return (int((right - margin) / scale) - WIDTH, int((bottom - margin) / scale) - HEIGHT)

    def _style(self) -> None:
        """Round the corners (Windows 11) and give the window the face icon. Called once the
        window is on screen; changing its styles earlier breaks the WinForms window."""
        import win32con
        import win32gui

        hwnd = win32gui.FindWindow(None, TITLE)
        if not hwnd:
            return
        self._hwnd = hwnd
        try:  # DWMWA_WINDOW_CORNER_PREFERENCE = 33, DWMWCP_ROUND = 2
            preference = ctypes.c_int(2)
            ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, 33, ctypes.byref(preference), 4)
        except OSError:
            pass
        try:
            from importlib import resources

            ico = str(resources.files("sohayok").joinpath("ui/icon.ico"))
            flags = win32con.LR_LOADFROMFILE
            big = win32gui.LoadImage(0, ico, win32con.IMAGE_ICON, 48, 48, flags)
            small = win32gui.LoadImage(0, ico, win32con.IMAGE_ICON, 16, 16, flags)
            win32gui.PostMessage(hwnd, win32con.WM_SETICON, win32con.ICON_BIG, big)
            win32gui.PostMessage(hwnd, win32con.WM_SETICON, win32con.ICON_SMALL, small)
        except Exception:
            log.debug("could not set the window icon", exc_info=True)

    def show(self) -> None:
        if self.window is None:
            return
        x, y = self._position()
        self.window.move(x, y)
        self.window.show()
        self.visible = True
        if self._hwnd:
            from sohayok import winapi

            winapi.force_foreground(self._hwnd)
        self.window.evaluate_js("window.sohayokShown && window.sohayokShown()")

    def hide(self) -> None:
        if self.window is not None:
            self.window.hide()
            self.visible = False

    def toggle(self) -> None:
        self.hide() if self.visible else self.show()

    def quit(self) -> None:
        log.info("quitting")
        try:
            if self.tray is not None:
                self.tray.stop()
        finally:
            self.on_quit()
            if self.window is not None:
                self.window.destroy()

    # ---- tray -------------------------------------------------------------------------------
    def _start_tray(self) -> None:
        import pystray

        from sohayok.icon import face

        def listen(icon, item):
            self.show()
            self.on_listen()

        menu = pystray.Menu(
            pystray.MenuItem("Sohayok দেখাও · Show", lambda icon, item: self.show(), default=True),
            pystray.MenuItem("শোনো · Listen now", listen),
            pystray.MenuItem("লুকাও · Hide", lambda icon, item: self.hide()),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("বন্ধ করো · Quit Sohayok", lambda icon, item: self.quit()),
        )
        self.tray = pystray.Icon("sohayok", face(64), "Sohayok — say “Tammy”", menu)
        threading.Thread(target=self.tray.run, name="tray", daemon=True).start()

    def run(self, start_hidden: bool = True) -> None:
        """Create the window and tray; blocks on the GUI loop until Quit."""
        import webview

        x, y = self._position()
        self.window = webview.create_window(
            TITLE,
            self.url,
            js_api=Bridge(self),
            width=WIDTH,
            height=HEIGHT,
            x=x,
            y=y,
            frameless=True,
            easy_drag=False,
            on_top=True,
            hidden=start_hidden,
            background_color="#04060D",
            min_size=(360, 480),
        )
        self.visible = not start_hidden
        self._start_tray()

        self.window.events.shown += self._style
        webview.start()
