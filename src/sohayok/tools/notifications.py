"""Read Windows notifications aloud (WhatsApp messages, reminders, anything that pops up).

Windows keeps recent notifications in a local SQLite database; it is read-only here. The sender
of the last message read out is remembered so "reply …" knows whom to answer.
"""

from __future__ import annotations

import os
import sqlite3
import xml.etree.ElementTree as ET
from datetime import UTC, datetime, timedelta

from sohayok.tools.registry import ToolError, integer, string, tool

DATABASE = r"%LOCALAPPDATA%\Microsoft\Windows\Notifications\wpndatabase.db"
# The sender ("Piyas Bhai") and app of the last message read out, for "reply …".
LAST = {"sender": "", "app": ""}
APP_NAMES = {
    "whatsapp": "WhatsApp",
    "telegram": "Telegram",
    "outlook": "Outlook",
    "teams": "Teams",
    "chrome": "Chrome",
    "msedge": "Edge",
    "messenger": "Messenger",
    "nvidia": "NVIDIA",
    "securitycenter": "Windows Security",
    "systemtoast": "Windows",
}


def app_name(handler: str) -> str:
    low = handler.lower()
    for key, name in APP_NAMES.items():
        if key in low:
            return name
    tail = handler.split("!")[0].split("_")[0].split(".")[-1]
    return tail or handler


def filetime(value: int) -> datetime:
    return datetime(1601, 1, 1, tzinfo=UTC) + timedelta(microseconds=value / 10)


def toast_lines(payload: bytes | str) -> list[str]:
    try:
        root = ET.fromstring(payload)
    except ET.ParseError:
        return []
    return [t.text.strip() for t in root.iter("text") if t.text and t.text.strip()]


def recent(limit: int = 5, app: str = "") -> list[dict]:
    path = os.path.expandvars(DATABASE)
    if not os.path.exists(path):
        raise ToolError("Windows has no notification history here")
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=2)
    try:
        rows = con.execute(
            "select h.PrimaryId, n.Payload, n.ArrivalTime from Notification n "
            "join NotificationHandler h on n.HandlerId = h.RecordId "
            "where n.Type = 'toast' order by n.ArrivalTime desc limit 200"
        ).fetchall()
    finally:
        con.close()
    found = []
    for handler, payload, arrived in rows:
        name = app_name(handler)
        if app and app.lower() not in name.lower():
            continue
        lines = toast_lines(payload)
        if not lines or any(f["app"] == name and f["title"] == lines[0] for f in found):
            continue  # the same notification shown again
        found.append(
            {
                "app": name,
                "title": lines[0],
                "text": " ".join(lines[1:])[:300],
                "time": filetime(arrived).astimezone().strftime("%H:%M"),
            }
        )
        if len(found) >= limit:
            break
    return found


@tool(
    "Read the latest Windows notifications (WhatsApp messages and others), newest first. "
    "Use when the user asks what a notification or new message says.",
    {
        "app": string("Only this app, e.g. 'WhatsApp'; empty for all"),
        "limit": integer("How many (default 3)"),
    },
    required=[],
)
def read_notifications(app: str = "", limit: int = 3) -> dict:
    items = recent(max(1, min(int(limit or 3), 10)), app.strip())
    if not items:
        return {
            "notifications": [],
            "spoken_bn": "নতুন কোনো নোটিফিকেশন নেই।",
            "spoken_en": "There are no new notifications.",
        }
    first = items[0]
    LAST.update(sender=first["title"], app=first["app"])
    bn = " ".join(f"{i['app']}-এ {i['title']}: {i['text']}।" for i in items)
    en = " ".join(f"{i['app']}, {i['title']}: {i['text']}." for i in items)
    return {"notifications": items, "spoken_bn": bn, "spoken_en": en}


def _clear_button(app: str = ""):
    """The notification centre's "Clear all" button, or the one for `app` ("WhatsApp")."""
    import subprocess
    import time

    import uiautomation as auto

    subprocess.Popen(["explorer.exe", "ms-actioncenter:"])
    deadline = time.monotonic() + 4
    while time.monotonic() < deadline:
        time.sleep(0.4)
        center = auto.WindowControl(searchDepth=1, Name="Notification Center")
        if not center.Exists(0.2):
            continue
        if not app:
            button = center.ButtonControl(AutomationId="ClearAllButton")
            if button.Exists(0.5):
                return button
            continue
        wanted = f"clear all notifications for {app.lower()}"
        for control, _depth in auto.WalkControl(center, maxDepth=12):
            if control.ControlTypeName == "ButtonControl" and (
                control.Name or ""
            ).lower().startswith(wanted):
                return control
    return None


@tool(
    "Clear (dismiss) Windows notifications: all of them, or only one app's. The messages "
    "themselves stay in the app.",
    {"app": string("Only this app's notifications, e.g. 'WhatsApp'; empty for all")},
    required=[],
)
def dismiss_notifications(app: str = "") -> dict:
    import uiautomation as auto

    app = APP_NAMES.get(app.strip().lower(), app.strip())
    button = _clear_button(app)
    if button is None:
        auto.SendKeys("{Esc}")
        return {
            "cleared": False,
            "spoken_bn": "মুছে ফেলার মতো কোনো নোটিফিকেশন নেই।",
            "spoken_en": "There are no notifications to clear.",
        }
    button.GetInvokePattern().Invoke()
    auto.SendKeys("{Esc}")
    return {
        "cleared": True,
        "spoken_bn": f"{app}-এর নোটিফিকেশন মুছে দিলাম।" if app else "সব নোটিফিকেশন মুছে দিলাম।",
        "spoken_en": f"Cleared the {app} notifications." if app else "Cleared all notifications.",
    }
