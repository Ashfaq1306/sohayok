"""Find, open, read, create, and recycle files and folders."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from pathlib import Path

from rapidfuzz import fuzz

from sohayok.lang import normalize
from sohayok.tools.registry import Risk, ToolError, string, tool

HOME = Path.home()
KNOWN = {
    "desktop": HOME / "Desktop",
    "ডেস্কটপ": HOME / "Desktop",
    "documents": HOME / "Documents",
    "ডকুমেন্ট": HOME / "Documents",
    "ডকুমেন্টস": HOME / "Documents",
    "downloads": HOME / "Downloads",
    "ডাউনলোড": HOME / "Downloads",
    "ডাউনলোডস": HOME / "Downloads",
    "pictures": HOME / "Pictures",
    "ছবি": HOME / "Pictures",
    "music": HOME / "Music",
    "গান": HOME / "Music",
    "videos": HOME / "Videos",
    "ভিডিও": HOME / "Videos",
    "home": HOME,
    "screenshots": HOME / "Pictures" / "Screenshots",
    "স্ক্রিনশট": HOME / "Pictures" / "Screenshots",
}
KNOWN = {normalize(k): v for k, v in KNOWN.items()}  # lookups use normalized text
SKIP_DIRS = {
    "node_modules",
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "AppData",
    ".cache",
    "site-packages",
    "$Recycle.Bin",
    ".gradle",
    ".pub-cache",
}


# Places that are not folders on disk; File Explorer opens them by their shell name.
SHELL_PLACES = {
    "this pc": "shell:MyComputerFolder",
    "my computer": "shell:MyComputerFolder",
    "computer": "shell:MyComputerFolder",
    "my pc": "shell:MyComputerFolder",
    "দিস পিসি": "shell:MyComputerFolder",
    "ডিস পিসি": "shell:MyComputerFolder",
    "এই পিসি": "shell:MyComputerFolder",
    "মাই পিসি": "shell:MyComputerFolder",
    "মাই কম্পিউটার": "shell:MyComputerFolder",
    "কম্পিউটার": "shell:MyComputerFolder",
    "recycle bin": "shell:RecycleBinFolder",
    "রিসাইকেল বিন": "shell:RecycleBinFolder",
    "control panel": "shell:ControlPanelFolder",
    "কন্ট্রোল প্যানেল": "shell:ControlPanelFolder",
}
SHELL_PLACES = {normalize(k): v for k, v in SHELL_PLACES.items()}
# Drive letters as spoken in Bangla ("ডি ড্রাইভ").
BANGLA_LETTERS = {
    "সি": "c",
    "ডি": "d",
    "ই": "e",
    "এফ": "f",
    "জি": "g",
    "এইচ": "h",
    "আই": "i",
    "জে": "j",
    "কে": "k",
    "এল": "l",
    "এম": "m",
    "এন": "n",
    "পি": "p",
    "কিউ": "q",
    "এক্স": "x",
    "জেড": "z",
}


def shell_place(name: str) -> str | None:
    key = normalize(name).strip(" .!?।").removesuffix(" folder").removesuffix(" ফোল্ডার")
    return SHELL_PLACES.get(key)


def drive(name: str) -> Path | None:
    """ "D drive", "drive D", "D:", "ডি ড্রাইভ" -> D:\\ (only drives that exist)."""
    key = normalize(name).strip(" .!?।").replace("-", " ")
    key = key.replace("ড্রাইভে", "ড্রাইভ").replace("ড্রাইভের", "ড্রাইভ")
    match = re.fullmatch(
        r"(?:(?P<a>[a-z]|[ঀ-৿]+)\s*(?:drive|ড্রাইভ)|(?:drive|ড্রাইভ)\s*(?P<b>[a-z])|(?P<c>[a-z]):\\?)",
        key,
    )
    if not match:
        return None
    letter = match.group("a") or match.group("b") or match.group("c")
    letter = BANGLA_LETTERS.get(letter, letter)
    if len(letter) != 1:
        return None
    root = Path(f"{letter.upper()}:\\")
    return root if root.exists() else None


def explorer_windows() -> list:
    """Open File Explorer windows (Shell.Application), front-most first when known."""
    import win32com.client

    from sohayok import winapi

    windows = []
    for window in win32com.client.Dispatch("Shell.Application").Windows():
        try:
            if Path(window.FullName).name.lower() == "explorer.exe":
                windows.append(window)
        except Exception:
            continue
    front = winapi.foreground_window()
    windows.sort(key=lambda w: int(w.HWND) != front)
    return windows


def explorer_folder() -> Path | None:
    """The folder shown in the File Explorer window in front, if it is a folder on disk."""
    from sohayok import winapi

    try:
        windows = explorer_windows()
    except Exception:
        return None
    if not windows or int(windows[0].HWND) != winapi.foreground_window():
        return None
    try:
        folder = Path(windows[0].Document.Folder.Self.Path)
    except Exception:
        return None
    return folder if folder.is_dir() else None


def show_in_explorer(target: str | Path) -> str:
    """Show a folder or shell place in the Explorer window in front, or open a new one."""
    from sohayok import winapi

    try:
        windows = explorer_windows()
    except Exception:
        windows = []
    if windows and int(windows[0].HWND) == winapi.foreground_window():
        windows[0].Navigate(str(target))
        return "here"
    subprocess.Popen(["explorer.exe", str(target)])
    return "new window"


def known_folder(name: str) -> Path | None:
    key = normalize(name).removesuffix(" folder").removesuffix(" ফোল্ডার")
    folder = KNOWN.get(key)
    if folder is None:
        return None
    # OneDrive can redirect Desktop/Documents/Pictures.
    onedrive = HOME / "OneDrive" / folder.name
    return folder if folder.exists() or not onedrive.exists() else onedrive


def resolve(path: str) -> Path:
    folder = known_folder(path) or drive(path)
    if folder:
        return folder
    expanded = Path(os.path.expandvars(os.path.expanduser(path.strip().strip('"'))))
    return expanded if expanded.is_absolute() else HOME / expanded


def search_roots(extra: list[str] | None = None) -> list[Path]:
    roots = [
        HOME / n
        for n in ("Desktop", "Documents", "Downloads", "Pictures", "Music", "Videos", "OneDrive")
    ]
    roots += [Path(p) for p in extra or []]
    return [r for r in roots if r.exists()]


def find(query: str, roots: list[Path], limit: int = 10, seconds: float = 4.0) -> list[dict]:
    """Fuzzy-match file and folder names under `roots`, best first, within a time budget."""
    wanted = normalize(query)
    deadline = time.monotonic() + seconds
    scored: list[tuple[float, Path, bool]] = []
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
            for name, is_dir in [(d, True) for d in dirnames] + [(f, False) for f in filenames]:
                stem = normalize(Path(name).stem if not is_dir else name)
                score = 100.0 if wanted == stem else fuzz.partial_ratio(wanted, stem)
                if wanted in stem:
                    score = max(score, 95 - min(len(stem) - len(wanted), 15))
                if score >= 80:
                    scored.append((score, Path(dirpath) / name, is_dir))
            if time.monotonic() > deadline:
                break
        if time.monotonic() > deadline:
            break
    scored.sort(key=lambda item: (-item[0], len(str(item[1]))))
    return [{"path": str(p), "kind": "folder" if d else "file"} for _, p, d in scored[:limit]]


@tool(
    "Find files or folders by name in Desktop, Documents, Downloads, Pictures, Music, Videos "
    "and OneDrive (plus any extra folders configured).",
    {"query": string("Part of the file or folder name")},
)
def find_files(query: str) -> list[dict]:
    from sohayok.config import current

    results = find(query, search_roots(current().files.extra_search_roots))
    if not results:
        raise ToolError(f"nothing named like {query!r} was found")
    return results


@tool(
    "Open a file with its default app, or a folder in File Explorer (in the Explorer window "
    'in front, if there is one). Accepts a full path, a drive ("D drive"), This PC, '
    "Recycle Bin, or a known folder: desktop, documents, downloads, pictures, music, videos, "
    "screenshots.",
    {"path": string("Path, drive, or place name")},
)
def open_path(path: str) -> str:
    place = shell_place(path)
    if place:
        show_in_explorer(place)
        return f"opened {path}"
    target = resolve(path)
    if not target.exists():
        raise ToolError(f"{target} does not exist")
    if target.is_dir():
        show_in_explorer(target)
    else:
        os.startfile(target)  # noqa: S606 - opening the user's own file is the point
    return f"opened {target}"


@tool(
    "Read the beginning of a text file (txt, md, csv, json, code) so you can summarize it.",
    {"path": string("Full path to the file")},
)
def read_text_file(path: str) -> str:
    target = resolve(path)
    if not target.is_file():
        raise ToolError(f"{target} is not a file")
    if target.stat().st_size > 5_000_000:
        raise ToolError("file is larger than 5 MB")
    text = target.read_text(encoding="utf-8", errors="replace")
    return text[:6000] + ("\n…(truncated)" if len(text) > 6000 else "")


def unused_name(folder: Path, name: str) -> Path:
    """`name`, or "name (2)", "name (3)", … if it is taken, like File Explorer."""
    target, number = folder / name, 2
    while target.exists():
        target, number = folder / f"{name} ({number})", number + 1
    return target


@tool(
    "Create a new folder. With no location it goes in the folder open in File Explorer (if "
    "Explorer is in front), otherwise on the Desktop.",
    {
        "name": string('Folder name (empty: "New folder"), or a full path'),
        "location": string("Where: a drive, known folder, or full path; empty for the default"),
    },
)
def create_folder(name: str = "", location: str = "") -> str | dict:
    name = name.strip().strip('"')
    if name and (":" in name or name.startswith(("~", "%"))):
        target = resolve(name)
        target.mkdir(parents=True, exist_ok=True)
        return f"created {target}"
    if location:
        folder = resolve(location)
        if not folder.is_dir():
            raise ToolError(f"{folder} is not a folder")
    else:
        folder = explorer_folder() or known_folder("desktop")
    target = unused_name(folder, name or "New folder")
    target.mkdir(parents=True)
    place = "Desktop" if folder == known_folder("desktop") else (folder.name or str(folder))
    place_bn = "ডেস্কটপ" if place == "Desktop" else place
    return {
        "created": str(target),
        "spoken_bn": f"{place_bn}-এ “{target.name}” ফোল্ডার বানালাম।",
        "spoken_en": f"Created the folder “{target.name}” in {place}.",
    }


@tool(
    "Write text to a new file (or replace an existing one, after confirmation).",
    {
        "path": string("Full path, or a file name to create on the Desktop"),
        "content": string("Text to write"),
    },
    risk=Risk.CONFIRM,
    summary=lambda a, lang: (
        f"{a.get('path')} ফাইলে লিখব"
        if lang == "bn"
        else f"Write {len(a.get('content', ''))} characters to {a.get('path')}"
    ),
)
def write_text_file(path: str, content: str) -> str:
    target = (
        resolve(path)
        if (":" in path or path.startswith(("~", "%")))
        else known_folder("desktop") / path
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"wrote {target}"


@tool(
    "Move a file or folder to the Recycle Bin (recoverable).",
    {"path": string("Full path of the file or folder")},
    risk=Risk.CONFIRM,
    summary=lambda a, lang: (
        f"{a.get('path')} রিসাইকেল বিনে সরাব"
        if lang == "bn"
        else f"Move {a.get('path')} to the Recycle Bin"
    ),
)
def recycle(path: str) -> str:
    from send2trash import send2trash

    target = resolve(path)
    if not target.exists():
        raise ToolError(f"{target} does not exist")
    if target in {HOME, Path(target.anchor)} or target in KNOWN.values():
        raise ToolError(f"refusing to recycle {target}")
    send2trash(str(target))
    return f"moved {target} to the Recycle Bin"


@tool(
    "Move or rename a file or folder.",
    {"source": string("Current full path"), "destination": string("New full path or folder")},
    risk=Risk.CONFIRM,
    summary=lambda a, lang: (
        f"{a.get('source')} সরিয়ে {a.get('destination')} এ রাখব"
        if lang == "bn"
        else f"Move {a.get('source')} to {a.get('destination')}"
    ),
)
def move_path(source: str, destination: str) -> str:
    src, dst = resolve(source), resolve(destination)
    if not src.exists():
        raise ToolError(f"{src} does not exist")
    final = shutil.move(str(src), str(dst))
    return f"moved to {final}"


@tool(
    "Save a note (appended with the date to Documents\\Sohayok\\notes.md).",
    {"text": string("The note")},
)
def take_note(text: str) -> str:
    folder = (known_folder("documents") or HOME / "Documents") / "Sohayok"
    folder.mkdir(parents=True, exist_ok=True)
    notes = folder / "notes.md"
    with notes.open("a", encoding="utf-8") as handle:
        handle.write(f"- {time.strftime('%Y-%m-%d %H:%M')} — {text}\n")
    return f"noted in {notes}"
