"""Run PowerShell commands, read the clipboard, and hand big jobs to Claude Code."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from sohayok.tools.registry import Risk, ToolError, integer, string, tool

NO_WINDOW = 0x08000000


def _clip(text: str, limit: int = 4000) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[: limit // 2] + "\n…\n" + text[-limit // 2 :]


@tool(
    "Run a PowerShell command and return its output. For anything no other tool covers: "
    "installing software with winget, git, network checks, file operations in bulk.",
    {
        "command": string("PowerShell command"),
        "timeout": integer("Seconds before giving up (default 120)"),
    },
    required=["command"],
    risk=Risk.CONFIRM,
    summary=lambda a, lang: (
        f"এই পাওয়ারশেল কমান্ড চালাব: {a.get('command')}"
        if lang == "bn"
        else f"Run PowerShell: {a.get('command')}"
    ),
)
def run_powershell(command: str, timeout: int = 120) -> str:
    try:
        result = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "[Console]::OutputEncoding=[Text.Encoding]::UTF8;" + command,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=max(5, min(int(timeout), 1800)),
            cwd=Path.home(),
            creationflags=NO_WINDOW,
        )
    except subprocess.TimeoutExpired as exc:
        raise ToolError(f"timed out after {exc.timeout} s") from exc
    output = _clip(result.stdout + ("\n" + result.stderr if result.stderr.strip() else ""))
    return f"exit code {result.returncode}\n{output or '(no output)'}"


@tool("Read the text currently on the clipboard.")
def get_clipboard() -> str:
    import win32clipboard

    win32clipboard.OpenClipboard()
    try:
        if not win32clipboard.IsClipboardFormatAvailable(win32clipboard.CF_UNICODETEXT):
            return "(the clipboard has no text)"
        return _clip(win32clipboard.GetClipboardData(win32clipboard.CF_UNICODETEXT), 6000)
    finally:
        win32clipboard.CloseClipboard()


@tool("Put text on the clipboard.", {"text": string("Text to copy")})
def set_clipboard(text: str) -> str:
    import win32clipboard

    win32clipboard.OpenClipboard()
    try:
        win32clipboard.EmptyClipboard()
        win32clipboard.SetClipboardText(text, win32clipboard.CF_UNICODETEXT)
    finally:
        win32clipboard.CloseClipboard()
    return f"copied {len(text)} characters"


def gh_path() -> str | None:
    found = shutil.which("gh")
    default = Path(r"C:\Program Files\GitHub CLI\gh.exe")
    return found or (str(default) if default.exists() else None)


def gh(*args: str, timeout: int = 60) -> str:
    exe = gh_path()
    if not exe:
        raise ToolError("GitHub CLI (gh) is not installed")
    result = subprocess.run(
        [exe, *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        creationflags=NO_WINDOW,
    )
    if result.returncode != 0:
        raise ToolError(_clip(result.stderr or result.stdout, 800))
    return result.stdout


@tool(
    "Summarize the user's GitHub: unread notifications, open pull requests they authored or "
    "must review, and issues assigned to them (uses the signed-in GitHub CLI).",
)
def github_overview() -> dict:
    def search(kind: str, query: str) -> list[str]:
        out = gh("search", kind, *query.split(), "--json", "title,repository,url", "--limit", "10")
        return [f"{i['repository']['nameWithOwner']}: {i['title']}" for i in json.loads(out)]

    notifications = json.loads(
        gh(
            "api",
            "notifications",
            "--jq",
            "[.[] | {repo: .repository.full_name, title: .subject.title}]",
        )
    )
    return {
        "unread_notifications": notifications[:10],
        "my_open_prs": search("prs", "--author=@me --state=open"),
        "review_requests": search("prs", "--review-requested=@me --state=open"),
        "assigned_issues": search("issues", "--assignee=@me --state=open"),
    }


@tool(
    "List the user's GitHub repositories (most recently updated first).",
    {"limit": integer("How many (default 15)")},
    required=[],
)
def github_repos(limit: int = 15) -> list[str]:
    out = gh(
        "repo",
        "list",
        "--json",
        "nameWithOwner,description,updatedAt,isPrivate",
        "--limit",
        str(max(1, min(int(limit), 100))),
    )
    return [
        f"{r['nameWithOwner']}{' (private)' if r['isPrivate'] else ''}: "
        f"{r.get('description') or ''}".strip()
        for r in json.loads(out)
    ]


@tool(
    "Hand a large or open-ended computer task to Claude Code (the coding agent installed on "
    "this PC), e.g. 'fix the failing tests in E:\\projects\\x', 'summarize this repository'. "
    "Slow (minutes); use only when the other tools cannot do it.",
    {
        "task": string("Full description of the task, in English or Bangla"),
        "folder": string("Working folder (default: home)"),
    },
    required=["task"],
    risk=Risk.CONFIRM,
    summary=lambda a, lang: (
        f"Claude Code-কে এই কাজ দেব: {a.get('task')}"
        if lang == "bn"
        else f"Ask Claude Code to: {a.get('task')} (in {a.get('folder') or '~'})"
    ),
)
def ask_claude_code(task: str, folder: str = "") -> str:
    from sohayok.config import current

    exe = shutil.which("claude")
    if not exe:
        raise ToolError("Claude Code (claude) is not installed")
    cwd = Path(folder).expanduser() if folder else Path.home()
    if not cwd.is_dir():
        raise ToolError(f"{cwd} is not a folder")
    settings = current().claude_code
    command = [exe, "-p", task, "--output-format", "json"]
    if settings.allowed_tools:
        command += ["--allowedTools", *settings.allowed_tools]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=settings.timeout_seconds,
            cwd=cwd,
            creationflags=NO_WINDOW,
        )
    except subprocess.TimeoutExpired as exc:
        raise ToolError(f"Claude Code did not finish within {exc.timeout} s") from exc
    try:
        payload = json.loads(result.stdout)
        return _clip(str(payload.get("result", "")), 3000)
    except json.JSONDecodeError:
        return _clip(result.stdout or result.stderr, 3000)
