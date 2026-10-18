"""See the screen: screenshots and the accessibility tree of the active window."""

from __future__ import annotations

import io
import time
from datetime import datetime
from pathlib import Path

import mss
from PIL import Image
from rapidfuzz import fuzz, process

from sohayok.lang import normalize
from sohayok.tools.registry import ToolError, integer, string, tool

INTERACTIVE = {
    "ButtonControl",
    "CheckBoxControl",
    "ComboBoxControl",
    "EditControl",
    "HyperlinkControl",
    "ListItemControl",
    "MenuItemControl",
    "RadioButtonControl",
    "TabItemControl",
    "TreeItemControl",
    "SplitButtonControl",
    "DocumentControl",
    "DataItemControl",
}


def grab(monitor: int = 1) -> Image.Image:
    """Capture one monitor (1 = primary) as an RGB image in physical pixels."""
    with mss.MSS() as sct:
        monitors = sct.monitors
        shot = sct.grab(monitors[monitor if monitor < len(monitors) else 1])
        return Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")


def png_bytes(image: Image.Image, max_long_edge: int | None = None) -> tuple[bytes, float]:
    """Encode as PNG, optionally downscaled; returns the bytes and the scale applied."""
    scale = 1.0
    if max_long_edge and max(image.size) > max_long_edge:
        scale = max_long_edge / max(image.size)
        image = image.resize(
            (round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS
        )
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue(), scale


def screenshot_dir() -> Path:
    path = Path.home() / "Pictures" / "Sohayok"
    path.mkdir(parents=True, exist_ok=True)
    return path


@tool("Take a screenshot of the primary screen and save it to Pictures\\Sohayok.")
def take_screenshot() -> str:
    path = screenshot_dir() / f"screenshot-{datetime.now():%Y%m%d-%H%M%S}.png"
    grab().save(path)
    return f"saved {path}"


def _elements(max_elements: int, max_seconds: float) -> list[dict]:
    import uiautomation as auto

    with auto.UIAutomationInitializerInThread():
        window = auto.GetForegroundControl()
        if window is None:
            raise ToolError("no active window")
        top = window.GetTopLevelControl() or window
        found = [{"window": top.Name, "type": "Window"}]
        deadline = time.monotonic() + max_seconds
        for control, _depth in auto.WalkControl(top, maxDepth=25):
            if time.monotonic() > deadline or len(found) > max_elements:
                break
            kind = control.ControlTypeName
            name = (control.Name or "").strip()
            if kind not in INTERACTIVE or (not name and kind != "EditControl"):
                continue
            try:
                if control.IsOffscreen:
                    continue
                rect = control.BoundingRectangle
            except Exception:
                continue
            if rect.width() <= 0 or rect.height() <= 0:
                continue
            found.append(
                {
                    "name": name[:80],
                    "type": kind.removesuffix("Control"),
                    "x": rect.xcenter(),
                    "y": rect.ycenter(),
                }
            )
        return found


@tool(
    "Describe the active window: its title and the visible buttons, links, fields, tabs and "
    "menu items with their names and click coordinates. Use before clicking things.",
    {"max_elements": integer("Maximum elements to list (default 80)")},
    required=[],
)
def describe_screen(max_elements: int = 80) -> list[dict]:
    return _elements(max(10, min(int(max_elements), 200)), max_seconds=4.0)


@tool(
    "Click a named control (button, link, tab, menu item, field) in the active window.",
    {
        "name": string("Visible label of the control, e.g. 'Save', 'Sign in', 'Downloads'"),
        "kind": string("Optional control type, e.g. Button, Hyperlink, Edit, TabItem"),
    },
    required=["name"],
)
def click_element(name: str, kind: str = "") -> str:
    from sohayok.tools.input import click

    candidates = [
        e
        for e in _elements(300, max_seconds=5.0)[1:]
        if not kind or e["type"].lower() == kind.lower()
    ]
    if not candidates:
        raise ToolError("no clickable controls found in the active window")
    labels = [e["name"] for e in candidates]
    best = process.extractOne(normalize(name), labels, scorer=fuzz.WRatio, processor=normalize)
    if not best or best[1] < 70:
        raise ToolError(f"no control named {name!r}; visible: {', '.join(labels[:25])}")
    element = candidates[best[2]]
    click(element["x"], element["y"])
    return f"clicked {element['type']} '{element['name']}'"
