"""Bluetooth and Wi-Fi on/off, and connecting or disconnecting a Bluetooth device.

The radios are switched through Windows' radio API. Connecting a device (headphones, a
speaker) presses that device's Connect / Disconnect button in Settings > Bluetooth & devices,
which is what a person would do; the device can be named ("JBL", "WF-1000XM5") or described
("headphones", "speaker", "হেডফোন").
"""

from __future__ import annotations

import asyncio
import os
import re
import time

from rapidfuzz import fuzz

from sohayok.lang import normalize
from sohayok.tools.registry import Risk, ToolError, boolean, string, tool
from sohayok.translit import to_latin

# Spoken kinds of device -> words in the Settings category ("Category Speakers").
KINDS = {
    "headphone": ("headphones", "communications", "headset"),
    "headphones": ("headphones", "communications", "headset"),
    "headset": ("headset", "headphones", "communications"),
    "earbuds": ("headphones", "communications", "headset"),
    "earphone": ("headphones", "communications", "headset"),
    "speaker": ("speakers", "audio"),
    "speakers": ("speakers", "audio"),
    "mouse": ("mouse",),
    "keyboard": ("keyboard",),
    "হেডফোন": ("headphones", "communications", "headset"),
    "হেডসেট": ("headset", "headphones", "communications"),
    "ইয়ারবাড": ("headphones", "communications", "headset"),
    "ইয়ারফোন": ("headphones", "communications", "headset"),
    "এয়ারবাড": ("headphones", "communications", "headset"),
    "স্পিকার": ("speakers", "audio"),
    "মাউস": ("mouse",),
    "কিবোর্ড": ("keyboard",),
}
KINDS = {normalize(k): v for k, v in KINDS.items()}
# Brands whose devices are named by model ("WF-1000XM5" is Sony's).
BRANDS = {
    "sony": ("wf-", "wh-", "ult", "srs-"),
    "সনি": ("wf-", "wh-", "ult", "srs-"),
    "apple": ("airpods",),
    "samsung": ("galaxy buds",),
}


def _radio(kind: str):
    from winrt.windows.devices.radios import Radio, RadioKind

    async def find():
        await Radio.request_access_async()
        for radio in await Radio.get_radios_async():
            if radio.kind == getattr(RadioKind, kind.upper().replace("-", "_")):
                return radio
        return None

    return asyncio.run(find())


def _set_radio(kind: str, on: bool) -> None:
    from winrt.windows.devices.radios import RadioAccessStatus, RadioState

    radio = _radio(kind)
    if radio is None:
        raise ToolError(f"this PC has no {kind} radio")
    status = asyncio.run(radio.set_state_async(RadioState.ON if on else RadioState.OFF))
    if status != RadioAccessStatus.ALLOWED:
        raise ToolError(f"Windows did not allow switching {kind} ({status})")


@tool(
    "Turn Bluetooth on or off.",
    {"on": boolean("True to turn Bluetooth on, false to turn it off")},
)
def bluetooth_power(on: bool) -> dict:
    _set_radio("bluetooth", bool(on))
    word_bn, word_en = ("চালু", "on") if on else ("বন্ধ", "off")
    return {"spoken_bn": f"ব্লুটুথ {word_bn} করলাম।", "spoken_en": f"Bluetooth is {word_en}."}


@tool(
    "Turn Wi-Fi on or off. Turning it off also cuts Sohayok's own internet.",
    {"on": boolean("True to turn Wi-Fi on, false to turn it off")},
    risk=Risk.CONFIRM,
    summary=lambda a, lang: (
        ("ওয়াইফাই চালু করব" if a.get("on") else "ওয়াইফাই বন্ধ করব (ইন্টারনেট চলে যাবে)")
        if lang == "bn"
        else ("Turn Wi-Fi on" if a.get("on") else "Turn Wi-Fi off (the internet will go)")
    ),
)
def wifi_power(on: bool) -> dict:
    _set_radio("wifi", bool(on))
    word_bn, word_en = ("চালু", "on") if on else ("বন্ধ", "off")
    return {"spoken_bn": f"ওয়াইফাই {word_bn} করলাম।", "spoken_en": f"Wi-Fi is {word_en}."}


def _settings_window():
    import uiautomation as auto

    os.startfile("ms-settings:bluetooth")  # noqa: S606 - Windows' own settings page
    window = auto.WindowControl(
        searchDepth=1, ClassName="ApplicationFrameWindow", RegexName="Settings"
    )
    if not window.Exists(10):
        raise ToolError("Settings did not open")
    time.sleep(1.5)  # the device list fills in after the page appears
    return window


def device_cards(window) -> list[tuple[str, str, object | None]]:
    """(device name, category text, Connect/Disconnect button or None) for each device."""
    import uiautomation as auto

    cards = []
    for control, _ in auto.WalkControl(window, maxDepth=12):
        name = control.Name or ""
        if control.ControlTypeName != "GroupControl" or ", Category " not in name:
            continue
        button = None
        for child, _ in auto.WalkControl(control, maxDepth=4):
            if child.ControlTypeName == "ButtonControl" and child.Name in {"Connect", "Disconnect"}:
                button = child
                break
        device, _, rest = name.partition(", Category ")
        cards.append((device, rest.lower(), button))
    return cards


def pick_device(cards: list, wanted: str, action: str):
    """The card the user means, among those that can do `action` ("connect"/"disconnect")."""
    label = "Connect" if action == "connect" else "Disconnect"
    usable = [c for c in cards if c[2] is not None and c[2].Name == label]
    if not usable:
        return None
    key = normalize(wanted).strip(" .!?।")
    key = re.sub(r"^(?:my|the|আমার)\s+", "", key)
    kind = next((v for k, v in KINDS.items() if k in key), None)
    if kind:  # "headphones": the device of that kind
        for word in kind:
            for card in usable:
                if word in card[1]:
                    return card
    latin = to_latin(key) if re.search(r"[ঀ-৿]", key) else key
    for brand, models in BRANDS.items():  # "Sony" -> its model names ("WF-1000XM5")
        if brand in latin or brand in key:
            for card in usable:
                if card[0].lower().startswith(models):
                    return card
    scored = [
        (max(fuzz.WRatio(latin, c[0].lower()), fuzz.partial_ratio(latin, c[0].lower())), c)
        for c in usable
    ]
    score, card = max(scored, key=lambda pair: pair[0])
    if score >= 70:
        return card
    if not key or len(usable) == 1 and key in {"bluetooth", "ব্লুটুথ", "device", "ডিভাইস"}:
        return usable[0]
    return None


@tool(
    "Connect or disconnect a paired Bluetooth device such as headphones, earbuds, or a "
    "speaker, by its name or kind.",
    {
        "device": string("Device name or kind, e.g. 'JBL', 'WF-1000XM5', 'headphones'"),
        "action": string("connect or disconnect", ["connect", "disconnect"]),
    },
)
def bluetooth_device(device: str, action: str) -> dict:
    action = "disconnect" if action.lower().startswith("dis") else "connect"
    window = _settings_window()
    try:
        cards = device_cards(window)
        card = pick_device(cards, device, action)
        if card is None:
            names = ", ".join(c[0] for c in cards if c[2] is not None) or "none"
            raise ToolError(
                f"no paired device {device!r} that can {action}; devices: {names}",
                spoken_bn=f"{device} নামে কোনো ডিভাইস পাইনি।",
                spoken_en=f"I couldn't find {device} among your Bluetooth devices.",
            )
        name, _, button = card
        button.GetInvokePattern().Invoke()
        time.sleep(1.0)
    finally:
        try:
            window.GetWindowPattern().Close()
        except Exception:
            pass
    done_bn = "কানেক্ট করছি" if action == "connect" else "ডিসকানেক্ট করলাম"
    done_en = "Connecting" if action == "connect" else "Disconnected"
    return {"device": name, "spoken_bn": f"{name} {done_bn}।", "spoken_en": f"{done_en} {name}."}
