"""WhatsApp Desktop: open a chat, send a message, or start a voice or video call.

WhatsApp for Windows is WhatsApp Web inside a WebView, so the controls are found with UI
Automation by their accessible names (search box, message box, call buttons). Sending and
calling always ask the user first, and a chat is only used when its name matches the contact
the user said (contacts saved in English letters match names said in Bangla).
"""

from __future__ import annotations

import re
import time

from rapidfuzz import fuzz

from sohayok.lang import normalize
from sohayok.tools.registry import Risk, ToolError, boolean, string, tool
from sohayok.translit import to_latin

NOT_SIGNED_IN = (
    "WhatsApp is not signed in on this PC. Open WhatsApp and link it with your phone "
    "(WhatsApp on the phone → Linked devices → scan the QR code)."
)
SEARCH = re.compile(r"search", re.I)
COMPOSE = re.compile(r"type a message|type to |message", re.I)
VOICE_CALL = re.compile(r"^(?:voice call|audio call|call)$", re.I)
VIDEO_CALL = re.compile(r"^video call$", re.I)
MATCH = 70  # minimum fuzzy score for a chat name to count as the contact


def name_score(spoken: str, chat: str) -> float:
    """How well a chat name matches a spoken contact name, across Bangla and English letters."""
    spoken, chat = normalize(spoken), normalize(chat)
    if not spoken or not chat:
        return 0.0
    options = [(spoken, chat), (to_latin(spoken), to_latin(chat))]
    return max(max(fuzz.WRatio(a, b), fuzz.ratio(a, b)) for a, b in options)


def _window():
    import uiautomation as auto

    from sohayok.tools.apps import launch, start_apps

    window = auto.WindowControl(searchDepth=1, Name="WhatsApp")
    if not window.Exists(0.3):
        launch("WhatsApp", start_apps())
        if not window.Exists(15):
            raise ToolError("WhatsApp did not open")
    window.SetActive()
    window.SetFocus()
    return window


def _document(window):
    document = window.DocumentControl(searchDepth=25, AutomationId="RootWebArea")
    if not document.Exists(5):
        raise ToolError("WhatsApp's window did not load")
    return document


def _find(document, control_type: str, pattern: re.Pattern[str], timeout: float = 4.0):
    import uiautomation as auto

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for control, _ in auto.WalkControl(document, maxDepth=40):
            if control.ControlTypeName == control_type and pattern.search(control.Name or ""):
                return control
        time.sleep(0.3)
    return None


def _signed_in(document) -> bool:
    import uiautomation as auto

    for control, _ in auto.WalkControl(document, maxDepth=12):
        if control.ControlTypeName == "ButtonControl" and (control.Name or "") == "Log in":
            return False
    return True


def open_chat(contact: str) -> tuple[object, str]:
    """Search for the contact and open the best-matching chat; returns (document, chat name)."""
    import uiautomation as auto

    from sohayok import winapi

    document = _document(_window())
    if not _signed_in(document):
        raise ToolError(NOT_SIGNED_IN)
    search = _find(document, "EditControl", SEARCH)
    if search is None:
        raise ToolError("could not find WhatsApp's search box")
    best_name, best_score, best_item = "", 0.0, None
    for query in dict.fromkeys([contact, to_latin(contact)]):  # as said, then in English letters
        search.Click(simulateMove=False)
        winapi.press("ctrl+a")
        winapi.press("backspace")
        winapi.type_unicode(query)
        time.sleep(1.2)  # results appear as you type
        for control, _ in auto.WalkControl(document, maxDepth=40):
            if control.ControlTypeName not in {"ListItemControl", "DataItemControl"}:
                continue
            name = (control.Name or "").split("\n")[0]
            score = name_score(contact, name.split(",")[0])
            if score > best_score:
                best_name, best_score, best_item = name, score, control
        if best_score >= MATCH:
            break
    if best_item is None or best_score < MATCH:
        winapi.press("escape")
        raise ToolError(f"no WhatsApp chat matches {contact!r}")
    best_item.Click(simulateMove=False)
    time.sleep(0.8)
    return document, best_name.split(",")[0].strip()


@tool(
    "Open a WhatsApp chat with a contact or group.",
    {"contact": string("Contact or group name as the user said it")},
)
def whatsapp_open_chat(contact: str) -> dict:
    _, chat = open_chat(contact)
    return {
        "chat": chat,
        "spoken_bn": f"হোয়াটসঅ্যাপে {chat}-এর চ্যাট খুলেছি।",
        "spoken_en": f"Opened your WhatsApp chat with {chat}.",
    }


@tool(
    "Send a WhatsApp message to a contact or group.",
    {
        "contact": string("Contact or group name as the user said it"),
        "message": string("The message text, in the user's words"),
    },
    risk=Risk.CONFIRM,
    summary=lambda a, lang: (
        f"{a.get('contact')}-কে হোয়াটসঅ্যাপে পাঠাব: “{a.get('message')}”"
        if lang == "bn"
        else f"Send “{a.get('message')}” to {a.get('contact')} on WhatsApp"
    ),
)
def whatsapp_send(contact: str, message: str) -> dict:
    from sohayok import winapi

    if not message.strip():
        raise ToolError("the message is empty")
    document, chat = open_chat(contact)
    box = _find(document, "EditControl", COMPOSE)
    if box is None or SEARCH.search(box.Name or ""):
        raise ToolError("could not find the message box")
    box.Click(simulateMove=False)
    winapi.type_unicode(message.strip())
    time.sleep(0.2)
    winapi.press("enter")
    return {
        "sent_to": chat,
        "spoken_bn": f"{chat}-কে মেসেজ পাঠিয়েছি।",
        "spoken_en": f"Sent to {chat}.",
    }


@tool(
    "Start a WhatsApp voice call (or video call) with a contact.",
    {
        "contact": string("Contact name as the user said it"),
        "video": boolean("True for a video call"),
    },
    required=["contact"],
    risk=Risk.CONFIRM,
    summary=lambda a, lang: (
        f"{a.get('contact')}-কে হোয়াটসঅ্যাপে {'ভিডিও ' if a.get('video') else ''}কল দেব"
        if lang == "bn"
        else f"{'Video' if a.get('video') else 'Voice'} call {a.get('contact')} on WhatsApp"
    ),
)
def whatsapp_call(contact: str, video: bool = False) -> dict:
    document, chat = open_chat(contact)
    button = _find(document, "ButtonControl", VIDEO_CALL if video else VOICE_CALL)
    if button is None:
        raise ToolError("could not find the call button in this chat")
    button.Click(simulateMove=False)
    return {
        "calling": chat,
        "spoken_bn": f"{chat}-কে কল দিচ্ছি।",
        "spoken_en": f"Calling {chat}.",
    }
