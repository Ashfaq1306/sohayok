"""Find, open, and close installed applications."""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time

import psutil
from rapidfuzz import fuzz, process

from sohayok.lang import normalize
from sohayok.tools.registry import ToolError, string, tool

# Spoken names (Bangla and English) for apps whose Start-menu name differs.
ALIASES = {
    "chrome": "Google Chrome",
    "google chrome": "Google Chrome",
    "ক্রোম": "Google Chrome",
    "গুগল ক্রোম": "Google Chrome",
    "ক্রম": "Google Chrome",  # how speech recognition often spells "Chrome"
    "browser": "Google Chrome",
    "ব্রাউজার": "Google Chrome",
    "edge": "Microsoft Edge",
    "এজ": "Microsoft Edge",
    "vs code": "Visual Studio Code",
    "vscode": "Visual Studio Code",
    "code": "Visual Studio Code",
    "ভিএস কোড": "Visual Studio Code",
    "ভি এস কোড": "Visual Studio Code",
    "notepad": "Notepad",
    "নোটপ্যাড": "Notepad",
    "calculator": "Calculator",
    "ক্যালকুলেটর": "Calculator",
    "file explorer": "File Explorer",
    "explorer": "File Explorer",
    "ফাইল এক্সপ্লোরার": "File Explorer",
    "ফাইল": "File Explorer",
    "settings": "Settings",
    "সেটিংস": "Settings",
    "সেটিং": "Settings",
    "task manager": "Task Manager",
    "টাস্ক ম্যানেজার": "Task Manager",
    "terminal": "Terminal",
    "টার্মিনাল": "Terminal",
    "powershell": "Windows PowerShell",
    "cmd": "Command Prompt",
    "command prompt": "Command Prompt",
    "paint": "Paint",
    "পেইন্ট": "Paint",
    "word": "Word",
    "ওয়ার্ড": "Word",
    "excel": "Excel",
    "এক্সেল": "Excel",
    "powerpoint": "PowerPoint",
    "পাওয়ারপয়েন্ট": "PowerPoint",
    "camera": "Camera",
    "ক্যামেরা": "Camera",
    "photos": "Photos",
    "ফটো": "Photos",
    "spotify": "Spotify",
    "স্পটিফাই": "Spotify",
    "telegram": "Telegram",
    "টেলিগ্রাম": "Telegram",
    "whatsapp": "WhatsApp",
    "হোয়াটসঅ্যাপ": "WhatsApp",
    "discord": "Discord",
    "ডিসকর্ড": "Discord",
    "snipping tool": "Snipping Tool",
    "স্নিপিং টুল": "Snipping Tool",
    "clock": "Clock",
    "ঘড়ি": "Clock",
    "store": "Microsoft Store",
    "মাইক্রোসফট স্টোর": "Microsoft Store",
    "control panel": "Control Panel",
    "কন্ট্রোল প্যানেল": "Control Panel",
}
ALIASES = {normalize(k): v for k, v in ALIASES.items()}  # lookups use normalized text
# Built into Windows; launched directly so they work even if the index is stale.
BUILTIN = {
    "File Explorer": ["explorer.exe"],
    "Settings": ["explorer.exe", "ms-settings:"],
    "Task Manager": ["taskmgr.exe"],
    "Notepad": ["notepad.exe"],
    "Calculator": ["explorer.exe", "calculator:"],
    "Command Prompt": ["cmd.exe", "/k"],
    "Windows PowerShell": ["powershell.exe", "-NoExit"],
    "Control Panel": ["control.exe"],
    "Paint": ["mspaint.exe"],
}
# Process names to close for apps whose window title does not contain their name.
PROCESSES = {
    "Google Chrome": "chrome.exe",
    "Microsoft Edge": "msedge.exe",
    "Visual Studio Code": "Code.exe",
    "Notepad": "Notepad.exe",
    "Spotify": "Spotify.exe",
    "Telegram": "Telegram.exe",
    "Discord": "Discord.exe",
    "Word": "WINWORD.EXE",
    "Excel": "EXCEL.EXE",
    "PowerPoint": "POWERPNT.EXE",
    "Paint": "mspaint.exe",
    "Calculator": "CalculatorApp.exe",
    "Task Manager": "Taskmgr.exe",
}

_cache: dict[str, str] = {}
_cache_time = 0.0
_lock = threading.Lock()
NO_WINDOW = 0x08000000


def start_apps(refresh: bool = False) -> dict[str, str]:
    """Map Start-menu app names to AppIDs (desktop programs and Store apps alike)."""
    global _cache_time
    with _lock:
        if _cache and not refresh and time.time() - _cache_time < 600:
            return _cache
        out = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-Command",
                "[Console]::OutputEncoding=[Text.Encoding]::UTF8;"
                "Get-StartApps | Select-Object Name,AppID | ConvertTo-Json -Compress",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
            creationflags=NO_WINDOW,
        )
        data = json.loads(out.stdout or "[]")
        if isinstance(data, dict):
            data = [data]
        _cache.clear()
        for item in data:
            if item.get("Name") and item.get("AppID"):
                _cache.setdefault(item["Name"], item["AppID"])
        _cache_time = time.time()
        return _cache


def resolve_app(name: str, apps: dict[str, str] | None = None) -> str | None:
    """Turn a spoken app name into a Start-menu name, or None if nothing is close enough."""
    spoken = normalize(name).removesuffix(" app").removesuffix(" অ্যাপ")
    target = ALIASES.get(spoken)
    apps = start_apps() if apps is None else apps
    if target:
        if target in apps or target in BUILTIN:
            return target
        spoken = normalize(target)
    names = list(apps)
    exact = {normalize(n): n for n in names}
    if spoken in exact:
        return exact[spoken]
    best = process.extractOne(spoken, names, scorer=fuzz.WRatio, processor=normalize)
    if best and best[1] >= 86:
        return best[0]
    return target if target in BUILTIN else None


def launch(app_name: str, apps: dict[str, str]) -> None:
    app_id = apps.get(app_name)
    if app_id:
        # Every Start-menu entry, desktop or Store, launches through the AppsFolder shell.
        subprocess.Popen(["explorer.exe", f"shell:AppsFolder\\{app_id}"], creationflags=NO_WINDOW)
    elif app_name in BUILTIN:
        command = BUILTIN[app_name]
        if command[0] in {"cmd.exe", "powershell.exe"}:
            subprocess.Popen(command, creationflags=subprocess.CREATE_NEW_CONSOLE)
        else:
            subprocess.Popen(command, creationflags=NO_WINDOW)
    else:
        raise ToolError(f"cannot launch {app_name}")


@tool(
    "Open (launch) an installed application by name, e.g. 'Google Chrome', 'Notepad', "
    "'Visual Studio Code', 'Settings', 'File Explorer'. Accepts Bangla names too.",
    {"name": string("Application name as the user said it")},
)
def open_app(name: str) -> str:
    from sohayok.tools.files import drive, open_path, shell_place

    if shell_place(name) or drive(name):  # "This PC", "D drive": File Explorer places
        return open_path(name)
    apps = start_apps()
    match = resolve_app(name, apps)
    if match is None:
        apps = start_apps(refresh=True)
        match = resolve_app(name, apps)
    if match is None:
        close = process.extract(normalize(name), list(apps), scorer=fuzz.WRatio, limit=3)
        hint = ", ".join(c[0] for c in close)
        raise ToolError(f"no installed app matches {name!r}. Closest: {hint}")
    window = launch_and_wait(match, apps)
    if window is None:
        return f"opened {match} (its window has not appeared yet)"
    return f"opened {match}; its window '{window}' is in front and ready"


def _app_windows(app_name: str) -> dict[int, str]:
    from sohayok.tools import windows

    exe = PROCESSES.get(app_name, "").lower()
    words = normalize(app_name).split()
    found = {}
    for win in windows.list_window_handles():
        title = normalize(win.title)
        if (exe and win.process.lower() == exe) or (words and words[-1] in title):
            found[win.hwnd] = win.title
    return found


def launch_and_wait(app_name: str, apps: dict[str, str], timeout: float = 8.0) -> str | None:
    """Launch, then wait until the app's window is in front so typing lands in it."""
    from sohayok import winapi

    before = _app_windows(app_name)
    launch(app_name, apps)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        time.sleep(0.25)
        current = _app_windows(app_name)
        fresh = [h for h in current if h not in before] or list(current)
        if fresh:
            hwnd = fresh[0]
            winapi.force_foreground(hwnd)
            time.sleep(0.6)  # let the app finish drawing and take keyboard focus
            title = _app_windows(app_name).get(hwnd, current[hwnd])
            if app_name == "Notepad" and not title.lower().startswith(("untitled", "*untitled")):
                # Notepad reopens the last session's files; start a blank tab instead of
                # letting dictation land in an old document.
                winapi.press("ctrl+n")
                time.sleep(0.4)
                title = _app_windows(app_name).get(hwnd, title)
            return title
    return None


@tool(
    "Search installed applications by name and return the closest matches.",
    {"query": string("Part of an application name")},
)
def find_apps(query: str) -> list[str]:
    apps = start_apps()
    return [
        m[0]
        for m in process.extract(
            normalize(query), list(apps), scorer=fuzz.WRatio, processor=normalize, limit=8
        )
        if m[1] >= 60
    ]


@tool(
    "Close an application gracefully (like clicking its X; it may ask to save work).",
    {"name": string("Application name, e.g. 'Chrome' or 'Notepad'")},
)
def close_app(name: str) -> str:
    from sohayok.tools import windows

    resolved = resolve_app(name, start_apps()) or name
    exe = PROCESSES.get(resolved, "").lower()
    closed = 0
    for win in windows.list_window_handles():
        if (exe and win.process.lower() == exe) or normalize(name) in normalize(win.title):
            windows.close_handle(win.hwnd)
            closed += 1
    if not closed:
        raise ToolError(f"no open window found for {name!r}")
    return f"asked {closed} window(s) of {resolved} to close"


def running_processes() -> list[str]:
    names = {p.info["name"] for p in psutil.process_iter(["name"]) if p.info.get("name")}
    return sorted(names, key=str.lower)


def app_paths_executable(exe: str) -> str | None:
    """Look up an executable in the registry's App Paths (where Chrome registers itself)."""
    import winreg

    for root in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
        try:
            with winreg.OpenKey(
                root, rf"Software\Microsoft\Windows\CurrentVersion\App Paths\{exe}"
            ) as k:
                path = winreg.QueryValue(k, None)
                if path and os.path.exists(path.strip('"')):
                    return path.strip('"')
        except OSError:
            continue
    return None
