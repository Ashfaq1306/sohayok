"""WhatsApp for Windows: open a chat, send or reply, read messages, voice or video call.

WhatsApp for Windows is WhatsApp Web inside a WebView. Its controls are found with Windows'
native UI Automation search (one call, ~0.03 s; walking the ~700 controls one by one took
5-10 s on a busy PC): the "Search or start a new chat" box, the "Chat list" grid, the
"Type a message to <name>" box, and the "Voice call" / "Video call" buttons. Message and chat
text is read through the page's text pattern. Contacts are usually saved in English letters
while the recognizer writes names in Bangla, so searches use English spellings first
("পিয়াস ভাই" -> "pias", "piyas", …). Sending and calling ask the user first.
"""

from __future__ import annotations

import re
import time

from rapidfuzz import fuzz

from sohayok.lang import normalize
from sohayok.tools.notifications import LAST
from sohayok.tools.registry import Risk, ToolError, boolean, integer, string, tool
from sohayok.translit import to_latin

NOT_SIGNED_IN = ToolError(
    "WhatsApp is not signed in on this PC",
    spoken_bn="হোয়াটসঅ্যাপ এই পিসিতে লগইন করা নেই। হোয়াটসঅ্যাপ খুলে ফোন দিয়ে কিউআর কোড স্ক্যান করুন।",
    spoken_en="WhatsApp isn't signed in on this PC. Open WhatsApp and scan its QR code with "
    "your phone.",
)
SEARCH = re.compile(r"^search", re.I)
COMPOSE = re.compile(r"^type a message", re.I)
VOICE_CALL = re.compile(r"^(?:voice call|audio call)$", re.I)
VIDEO_CALL = re.compile(r"^video call$", re.I)
UNREAD = re.compile(r"(\d+)\s+unread messages?", re.I)
STATUS = re.compile(r"\b(?:delivered|read|sent|pending)\b", re.I)
TIME = re.compile(r"^\d{1,2}:\d{2}(?:\s*[ap]m)?$", re.I)
MATCH = 70  # minimum fuzzy score for a chat name to count as the contact
# Relationship words people add to names; the saved contact may or may not have them.
HONORIFICS = {
    "ভাই", "ভাইয়া", "ভাইয়ে", "ভাইকে", "আপু", "আপা", "ভাবি", "মামা", "চাচা", "খালা", "স্যার",
    "bhai", "vai", "bhaiya", "apu", "apa", "sir", "mama", "chacha", "khala",
}  # fmt: skip
HONORIFICS = {normalize(w) for w in HONORIFICS}


# --- matching names ---------------------------------------------------------------------------
def name_score(spoken: str, chat: str) -> float:
    """How well a chat name matches a spoken contact name, across Bangla and English letters."""
    spoken, chat = normalize(spoken), normalize(chat)
    if not spoken or not chat:
        return 0.0
    bare = " ".join(w for w in spoken.split() if w not in HONORIFICS) or spoken
    chat_bare = " ".join(w for w in chat.split() if w not in HONORIFICS) or chat
    pairs = []
    for said in {spoken, bare}:
        pairs += [(said, chat), (to_latin(said), to_latin(chat))]
    variants = {to_latin(bare)}
    for change in SPELLINGS:  # "piyaj" may be "pias": compare the usual spellings too
        variants |= {change(v) for v in variants}
    pairs += [(v, to_latin(chat_bare)) for v in variants]
    return max(max(fuzz.WRatio(a, b), fuzz.ratio(a, b)) for a, b in pairs)


# Common English spellings of the same Bangla sounds (piyas / pias / piash, rohim / rahim),
# and জ heard for স ("পিয়াজ").
SPELLINGS = (
    lambda v: v.replace("y", ""),
    lambda v: v.replace("j", "s"),
    lambda v: v.replace("o", "a", 1),
    lambda v: v.replace("s", "sh"),
    lambda v: v.replace("ph", "f"),
)


def search_queries(contact: str) -> list[str]:
    """What to type into WhatsApp's search, most likely to find the saved name first."""
    words = [w for w in normalize(contact).split() if w not in HONORIFICS] or contact.split()
    first = words[0] if words else contact
    if len(words) > 1 and len(to_latin(first)) <= 2:
        first = words[0] + words[1]  # the recognizer split a name: "পি আশ" -> "পিআশ"
    queries = []
    if re.search(r"[ঀ-৿]", first):
        latin = to_latin(first)
        # common English spellings of the same sound: piyas / pias / piash, rohim / rahim;
        # জ is sometimes heard for স ("পিয়াজ")
        variants = {latin}
        for change in SPELLINGS:  # combined: "piyaj" -> "piaj" -> "pias"
            variants |= {change(v) for v in variants}
        queries += sorted(variants, key=len)
        queries.append(first)  # saved in Bangla
        queries += [latin[:3], latin[:2]]  # last resort: the first sounds
    else:
        queries += [first, first[:3]]
    return list(dict.fromkeys(q for q in queries if q))


# --- the WhatsApp window ----------------------------------------------------------------------
def _find_all(root, control_type: int) -> list:
    """Every descendant of one control type, in one native UI Automation call."""
    import uiautomation as auto
    from uiautomation.uiautomation import _AutomationClient

    client = _AutomationClient.instance().IUIAutomation
    condition = client.CreatePropertyCondition(auto.PropertyId.ControlTypeProperty, control_type)
    found = root.Element.FindAll(4, condition)  # 4: TreeScope_Descendants
    return [auto.Control.CreateControlFromElement(found.GetElement(i)) for i in range(found.Length)]


def _name(control) -> str:
    """A control's name, or "" if WhatsApp just replaced the control (it redraws often)."""
    try:
        return control.Name or ""
    except Exception:
        return ""


def _click(find) -> bool:
    """Click what `find()` returns, finding it again if WhatsApp redrew it in between."""
    for _ in range(4):
        control = find()
        if control is None:
            time.sleep(0.3)
            continue
        try:
            control.Click(simulateMove=False)
            return True
        except Exception:
            time.sleep(0.3)
    return False


def _id(control) -> tuple:
    try:
        return tuple(control.GetRuntimeId())
    except Exception:
        return ()


def _first(root, property_id: int, value):
    """The first descendant with a property value, in one native call (or None)."""
    import uiautomation as auto
    from uiautomation.uiautomation import _AutomationClient

    client = _AutomationClient.instance().IUIAutomation
    found = root.Element.FindFirst(4, client.CreatePropertyCondition(property_id, value))
    return auto.Control.CreateControlFromElement(found) if found else None


_CACHE: dict = {"document": None, "at": 0.0}  # WhatsApp's page, reused for a few seconds


def _named(root, control_type: int, pattern: re.Pattern[str], timeout: float = 3.0):
    deadline = time.monotonic() + timeout
    while True:
        for control in _find_all(root, control_type):
            if pattern.search(_name(control)):
                return control
        if time.monotonic() > deadline:
            return None
        time.sleep(0.25)


def _text(document, control) -> str:
    """The visible text inside a control (through the page's text pattern)."""
    try:
        return document.GetTextPattern().pattern.RangeFromChild(control.Element).GetText(-1)
    except Exception:
        return _name(control)


def _window(activate: bool = True):
    import uiautomation as auto

    from sohayok.tools.apps import launch, start_apps

    window = auto.WindowControl(searchDepth=1, Name="WhatsApp")
    if not window.Exists(0.3):
        launch("WhatsApp", start_apps())
        if not window.Exists(15):
            raise ToolError("WhatsApp did not open")
    if activate:
        window.ShowWindow(auto.SW.Restore)
        window.SetActive()
    return window


def signed_in_document(activate: bool = True):
    """WhatsApp's page, or NOT_SIGNED_IN (about 0.1 s)."""
    import uiautomation as auto

    cached = _CACHE.get("document")
    if cached is not None and time.monotonic() - _CACHE["at"] < 20:
        try:
            _window(activate)
            if _search_box(cached) is not None:
                _CACHE["at"] = time.monotonic()
                return cached
        except Exception:
            pass
    window = _window(activate)
    document = None
    deadline = time.monotonic() + 8
    while document is None:
        document = _first(window, auto.PropertyId.AutomationIdProperty, "RootWebArea")
        if document is None:
            if time.monotonic() > deadline:
                raise ToolError("WhatsApp's window did not load")
            time.sleep(0.3)
    deadline = time.monotonic() + 6
    while _search_box(document) is None:
        signed_out = any(
            _name(b) == "Log in" for b in _find_all(document, auto.ControlType.ButtonControl)
        )
        if signed_out or time.monotonic() > deadline:
            raise NOT_SIGNED_IN
        time.sleep(0.3)  # still loading
    _CACHE.update(document=document, at=time.monotonic())
    return document


def _search_box(document):
    """The chat search box: named "Search or start a new chat" while empty, unnamed with text
    in it; either way the text box that is not the message box."""
    import uiautomation as auto

    edits = _find_all(document, auto.ControlType.EditControl)
    for edit in edits:
        if SEARCH.search(_name(edit)):
            return edit
    return next((e for e in edits if not COMPOSE.search(_name(e))), None)


def check_signed_in(args: dict) -> None:
    signed_in_document(activate=False)


def _clear_search(document) -> None:
    """Empty the search box so the side list shows all chats again."""
    from sohayok import winapi

    search = _search_box(document)
    if search is not None and not SEARCH.search(_name(search)):  # it has text in it
        _click(lambda: _search_box(document))
        winapi.press("ctrl+a")
        winapi.press("backspace")
        time.sleep(0.5)


def _chat_rows(document) -> list[tuple[str, object]]:
    """(chat name, row) for the chats or search results in the side list."""
    import uiautomation as auto

    for _ in range(3):  # the list redraws while search results arrive
        rows = []
        try:
            for grid in _find_all(document, auto.ControlType.DataGridControl):
                for row in _find_all(grid, auto.ControlType.DataItemControl):
                    try:
                        name = _name(row).split("\n")[0].strip()
                    except Exception:  # this row was just replaced
                        continue
                    if name and not any(name == seen for seen, _ in rows):
                        rows.append((name, row))
            return rows
        except Exception:
            time.sleep(0.3)
    return []


def open_chat(contact: str) -> tuple[object, str]:
    """Search for the contact and open the best-matching chat; returns (document, chat name)."""
    import uiautomation as auto

    from sohayok import winapi

    document = signed_in_document()
    search = _search_box(document)
    if search is None:
        raise NOT_SIGNED_IN
    best_name, best_score, best_row = "", 0.0, None
    for query in search_queries(contact):
        _click(lambda: _search_box(document))
        winapi.press("ctrl+a")
        winapi.press("backspace")
        winapi.type_unicode(query)
        time.sleep(1.0)  # results appear as you type
        for name, row in _chat_rows(document):
            score = name_score(contact, name)
            if score > best_score:
                best_name, best_score, best_row = name, score, row
        if best_score >= MATCH:
            break
    if best_row is None or best_score < MATCH:
        winapi.press("escape")
        raise ToolError(
            f"no WhatsApp chat matches {contact!r}",
            spoken_bn=f"হোয়াটসঅ্যাপে {contact} নামে কাউকে পাইনি।",
            spoken_en=f"I couldn't find {contact} in WhatsApp.",
        )
    wanted = best_name
    if not _click(lambda: next((r for n, r in _chat_rows(document) if n == wanted), None)):
        raise ToolError("could not open the chat")
    compose = lambda: _named(document, auto.ControlType.EditControl, COMPOSE, timeout=4)  # noqa: E731
    box = compose()
    if box is not None:  # "Type a message to Pias Vai" names the chat that really opened
        best_name = re.sub(COMPOSE, "", _name(box)).removeprefix(" to ").strip() or best_name
    _clear_search(document)
    _click(compose)  # ready to type in the opened chat
    return document, best_name


# --- tools ------------------------------------------------------------------------------------
@tool(
    "Open a WhatsApp chat with a contact or group.",
    {"contact": string("Contact or group name as the user said it")},
    check=check_signed_in,
)
def whatsapp_open_chat(contact: str) -> dict:
    _, chat = open_chat(contact)
    LAST.update(sender=chat, app="WhatsApp")
    return {
        "chat": chat,
        "spoken_bn": f"হোয়াটসঅ্যাপে {chat}-এর চ্যাট খুলেছি।",
        "spoken_en": f"Opened your WhatsApp chat with {chat}.",
    }


def _send(contact: str, message: str) -> str:
    import uiautomation as auto

    from sohayok import winapi

    if not message.strip():
        raise ToolError("the message is empty")
    document, chat = open_chat(contact)
    box = _named(document, auto.ControlType.EditControl, COMPOSE)
    if box is None:
        raise ToolError("could not find the message box")
    if not _click(lambda: _named(document, auto.ControlType.EditControl, COMPOSE)):
        raise ToolError("could not find the message box")
    winapi.type_unicode(message.strip())
    time.sleep(0.2)
    winapi.press("enter")
    LAST.update(sender=chat, app="WhatsApp")
    return chat


@tool(
    "Send a WhatsApp message to a contact or group.",
    {
        "contact": string("Contact or group name as the user said it"),
        "message": string("The message text, in the user's words"),
    },
    risk=Risk.CONFIRM,
    check=check_signed_in,
    summary=lambda a, lang: (
        f"{a.get('contact')}-কে হোয়াটসঅ্যাপে পাঠাব: “{a.get('message')}”"
        if lang == "bn"
        else f"Send “{a.get('message')}” to {a.get('contact')} on WhatsApp"
    ),
)
def whatsapp_send(contact: str, message: str) -> dict:
    chat = _send(contact, message)
    return {"sent_to": chat, "spoken_bn": f"{chat}-কে পাঠিয়েছি।", "spoken_en": f"Sent to {chat}."}


@tool(
    "Reply on WhatsApp to the person whose message was just read out (or the open chat).",
    {"message": string("The reply, in the user's words")},
    risk=Risk.CONFIRM,
    check=check_signed_in,
    summary=lambda a, lang: (
        f"{LAST['sender'] or 'এই চ্যাটে'}-কে উত্তর পাঠাব: “{a.get('message')}”"
        if lang == "bn"
        else f"Reply “{a.get('message')}” to {LAST['sender'] or 'this chat'}"
    ),
)
def whatsapp_reply(message: str) -> dict:
    if not LAST["sender"]:
        raise ToolError(
            "no message has been read out yet, so it is unclear whom to reply to",
            spoken_bn="কাকে উত্তর দেব বুঝিনি। নাম বলুন।",
            spoken_en="Who should I reply to?",
        )
    chat = _send(LAST["sender"], message)
    return {
        "sent_to": chat,
        "spoken_bn": f"{chat}-কে উত্তর পাঠিয়েছি।",
        "spoken_en": f"Replied to {chat}.",
    }


def unread_chats(document) -> list[dict]:
    unread = []
    for name, row in _chat_rows(document):
        found = UNREAD.search(_text(document, row))
        if found:
            unread.append({"chat": name, "count": int(found.group(1))})
    return unread


@tool("List WhatsApp chats with unread messages.", {}, check=check_signed_in)
def whatsapp_unread() -> dict:
    document = signed_in_document()
    _clear_search(document)
    unread = unread_chats(document)
    if not unread:
        return {
            "unread": [],
            "spoken_bn": "হোয়াটসঅ্যাপে নতুন কোনো মেসেজ নেই।",
            "spoken_en": "No unread WhatsApp messages.",
        }
    LAST.update(sender=unread[0]["chat"], app="WhatsApp")
    bn = ", ".join(f"{u['chat']} {u['count']}টা" for u in unread[:5])
    en = ", ".join(f"{u['chat']} ({u['count']})" for u in unread[:5])
    return {"unread": unread, "spoken_bn": f"নতুন মেসেজ: {bn}।", "spoken_en": f"Unread: {en}."}


def message_lines(text: str) -> str:
    """A message row's text without its time and the icon glyphs after it."""
    lines = []
    for line in text.split("\n"):
        line = line.strip()
        if TIME.match(line):
            break  # the time ends the message; what follows are status icons
        if len(line) > 1 and not STATUS.fullmatch(line):
            lines.append(line)
    return " ".join(lines)


def chat_messages(document) -> list[tuple[str, str]]:
    """("in" or "out", text) for the messages shown in the open chat, oldest first."""
    import uiautomation as auto

    in_lists = set()
    for grid in _find_all(document, auto.ControlType.DataGridControl):
        in_lists |= {_id(r) for r in _find_all(grid, auto.ControlType.DataItemControl)}
    messages = []
    for row in _find_all(document, auto.ControlType.DataItemControl):
        if _id(row) in in_lists:
            continue
        text = message_lines(_text(document, row))
        if not text:
            continue
        mine = any(STATUS.search(_name(b)) for b in _find_all(row, auto.ControlType.ButtonControl))
        messages.append(("out" if mine else "in", text[:300]))
    return messages


@tool(
    "Read the latest messages of a WhatsApp chat aloud.",
    {
        "contact": string("Contact or group name; empty for the chat whose message was last read"),
        "count": integer("How many recent messages (default 3)"),
    },
    required=[],
    check=check_signed_in,
)
def whatsapp_read_chat(contact: str = "", count: int = 3) -> dict:
    contact = contact or LAST["sender"]
    if not contact:
        raise ToolError("which chat?", spoken_bn="কার চ্যাট পড়ব?", spoken_en="Whose chat?")
    document, chat = open_chat(contact)
    time.sleep(0.5)
    messages = chat_messages(document)[-max(1, min(int(count or 3), 10)) :]
    if not messages:
        raise ToolError("could not read this chat's messages")
    LAST.update(sender=chat, app="WhatsApp")
    bn = " ".join(f"{chat if who == 'in' else 'আপনি'}: {text}।" for who, text in messages)
    en = " ".join(f"{chat if who == 'in' else 'You'}: {text}." for who, text in messages)
    return {"chat": chat, "messages": messages, "spoken_bn": bn, "spoken_en": en}


@tool(
    "Start a WhatsApp voice call (or video call) with a contact.",
    {
        "contact": string("Contact name as the user said it"),
        "video": boolean("True for a video call"),
    },
    required=["contact"],
    risk=Risk.CONFIRM,
    check=check_signed_in,
    summary=lambda a, lang: (
        f"{a.get('contact')}-কে হোয়াটসঅ্যাপে {'ভিডিও ' if a.get('video') else ''}কল দেব"
        if lang == "bn"
        else f"{'Video' if a.get('video') else 'Voice'} call {a.get('contact')} on WhatsApp"
    ),
)
def whatsapp_call(contact: str, video: bool = False) -> dict:
    import uiautomation as auto

    document, chat = open_chat(contact)
    button = _named(document, auto.ControlType.ButtonControl, VIDEO_CALL if video else VOICE_CALL)
    if button is None:
        raise ToolError("could not find the call button in this chat")
    try:
        button.GetInvokePattern().Invoke()
    except Exception:
        button.Click(simulateMove=False)
    return {"calling": chat, "spoken_bn": f"{chat}-কে কল দিচ্ছি।", "spoken_en": f"Calling {chat}."}
