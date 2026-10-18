"""Clock, system status, settings pages, locking, and power."""

from __future__ import annotations

import ctypes
import shutil
import subprocess
import time
from datetime import datetime

import psutil

from sohayok import winapi
from sohayok.tools.registry import Risk, ToolError, string, tool

NO_WINDOW = 0x08000000
BN_MONTHS = [
    "জানুয়ারি",
    "ফেব্রুয়ারি",
    "মার্চ",
    "এপ্রিল",
    "মে",
    "জুন",
    "জুলাই",
    "আগস্ট",
    "সেপ্টেম্বর",
    "অক্টোবর",
    "নভেম্বর",
    "ডিসেম্বর",
]
BN_DAYS = ["সোমবার", "মঙ্গলবার", "বুধবার", "বৃহস্পতিবার", "শুক্রবার", "শনিবার", "রবিবার"]
SETTINGS = {
    "home": "",
    "wifi": "network-wifi",
    "network": "network",
    "bluetooth": "bluetooth",
    "display": "display",
    "sound": "sound",
    "notifications": "notifications",
    "power": "powersleep",
    "battery": "batterysaver",
    "storage": "storagesense",
    "apps": "appsfeatures",
    "default apps": "defaultapps",
    "startup": "startupapps",
    "accounts": "yourinfo",
    "time": "dateandtime",
    "language": "regionlanguage",
    "keyboard": "typing",
    "mouse": "mousetouchpad",
    "update": "windowsupdate",
    "privacy": "privacy",
    "microphone": "privacy-microphone",
    "camera": "privacy-webcam",
    "personalization": "personalization",
    "background": "personalization-background",
    "themes": "themes",
    "night light": "nightlight",
    "printers": "printers",
    "about": "about",
    "vpn": "network-vpn",
    "hotspot": "network-mobilehotspot",
}


@tool(
    "Get the current local date and time (answer 'what time is it', 'আজ কত তারিখ').",
    {"language": string("Reply language", ["bn", "en"])},
    required=[],
)
def current_time(language: str = "en") -> str:
    now = datetime.now()
    if language == "bn":
        from sohayok.lang import to_bangla_digits as d

        hour = now.hour % 12 or 12
        part = (
            "রাত"
            if now.hour < 4
            else "ভোর"
            if now.hour < 6
            else "সকাল"
            if now.hour < 12
            else "দুপুর"
            if now.hour < 15
            else "বিকাল"
            if now.hour < 18
            else "সন্ধ্যা"
            if now.hour < 20
            else "রাত"
        )
        return (
            f"{part} {d(hour)}টা {d(f'{now.minute:02d}')} মিনিট, {BN_DAYS[now.weekday()]}, "
            f"{d(now.day)} {BN_MONTHS[now.month - 1]} {d(now.year)}"
        )
    return now.strftime("%I:%M %p, %A %d %B %Y").lstrip("0")


def gpu_status() -> str | None:
    smi = shutil.which("nvidia-smi")
    if not smi:
        return None
    out = subprocess.run(
        [
            smi,
            "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=10,
        creationflags=NO_WINDOW,
    ).stdout.strip()
    if not out:
        return None
    name, util, used, total, temp = [part.strip() for part in out.splitlines()[0].split(",")]
    return f"{name}: {util}% busy, {used}/{total} MiB memory, {temp}°C"


@tool("Report CPU, memory, GPU, disk, battery, and uptime.")
def system_status() -> dict[str, str]:
    memory = psutil.virtual_memory()
    status = {
        "cpu": f"{psutil.cpu_percent(interval=0.5):.0f}% of {psutil.cpu_count()} threads",
        "memory": f"{memory.used / 2**30:.1f} of {memory.total / 2**30:.1f} GB used "
        f"({memory.percent:.0f}%)",
        "uptime": f"{(time.time() - psutil.boot_time()) / 3600:.1f} hours",
    }
    gpu = gpu_status()
    if gpu:
        status["gpu"] = gpu
    for part in psutil.disk_partitions():
        if "fixed" in part.opts or part.fstype:
            try:
                usage = psutil.disk_usage(part.mountpoint)
            except OSError:
                continue
            status[f"disk {part.device}"] = (
                f"{usage.free / 2**30:.0f} GB free of {usage.total / 2**30:.0f} GB"
            )
    battery = psutil.sensors_battery()
    if battery:
        plugged = "charging" if battery.power_plugged else "on battery"
        status["battery"] = f"{battery.percent:.0f}% ({plugged})"
    return status


@tool(
    "Open a Windows Settings page: " + ", ".join(SETTINGS) + ".",
    {"page": string("Settings page", list(SETTINGS))},
    required=[],
)
def open_settings(page: str = "home") -> str:
    target = SETTINGS.get(page.lower().strip())
    if target is None:
        raise ToolError(f"unknown settings page {page!r}")
    subprocess.Popen(["explorer.exe", f"ms-settings:{target}"], creationflags=NO_WINDOW)
    return f"opened {page} settings"


@tool("Lock the computer (the user signs back in with their own PIN or password).")
def lock_screen() -> str:
    winapi.lock_workstation()
    return "locked"


POWER_SUMMARY = {
    "en": {
        "shutdown": "Shut down the computer",
        "restart": "Restart the computer",
        "signout": "Sign out of Windows",
        "sleep": "Put the computer to sleep",
    },
    "bn": {
        "shutdown": "কম্পিউটার বন্ধ করব",
        "restart": "কম্পিউটার রিস্টার্ট করব",
        "signout": "উইন্ডোজ থেকে সাইন আউট করব",
        "sleep": "কম্পিউটার স্লিপে পাঠাব",
    },
}
_POWER = {
    "shutdown": ["shutdown.exe", "/s", "/t", "20"],
    "restart": ["shutdown.exe", "/r", "/t", "20"],
    "signout": ["shutdown.exe", "/l"],
}


@tool(
    "Shut down, restart, sign out, or sleep the computer. Shut down and restart wait 20 "
    "seconds and can be cancelled with cancel_shutdown.",
    {"action": string("Power action", ["shutdown", "restart", "signout", "sleep"])},
    risk=Risk.CONFIRM,
    summary=lambda a, lang: POWER_SUMMARY[lang if lang == "bn" else "en"].get(
        a.get("action", ""), f"Power: {a.get('action')}"
    ),
)
def power(action: str) -> str:
    if action == "sleep":
        # SetSuspendState(hibernate=False, force=False, wakeup_events_disabled=False)
        ctypes.windll.powrprof.SetSuspendState(0, 0, 0)
        return "sleeping"
    command = _POWER.get(action)
    if command is None:
        raise ToolError(f"unknown power action {action!r}")
    subprocess.run(command, check=True, creationflags=NO_WINDOW)
    return f"{action} scheduled in 20 seconds" if action != "signout" else "signing out"


@tool("Cancel a pending shutdown or restart.")
def cancel_shutdown() -> str:
    result = subprocess.run(
        ["shutdown.exe", "/a"], capture_output=True, text=True, creationflags=NO_WINDOW
    )
    return "cancelled the shutdown" if result.returncode == 0 else "no shutdown was pending"
