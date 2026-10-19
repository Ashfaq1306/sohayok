"""Cloud backend: Claude through the Anthropic API, optionally with computer use."""

from __future__ import annotations

import base64
import logging
import time
from collections.abc import Callable
from typing import Any

from sohayok.brain.base import CONFIRM_TOOL, Reply, system_prompt, user_message
from sohayok.safety import Executor

log = logging.getLogger(__name__)

COMPUTER_TOOLSET = {"type": "computer_toolset_20260801"}
# Claude 5.5 models accept screenshots up to 2576 px on the long edge.
MAX_LONG_EDGE = 2576
CLAUDE_KEYS = {
    "return": "enter",
    "super": "win",
    "super_l": "win",
    "page_down": "pagedown",
    "page_up": "pageup",
    "escape": "esc",
    "back_space": "backspace",
    "kp_enter": "enter",
    "prior": "pageup",
    "next": "pagedown",
    "meta": "win",
}


class ComputerUse:
    """Carries out the computer toolset's member tools on the real desktop."""

    def __init__(self, max_long_edge: int = MAX_LONG_EDGE) -> None:
        self.scale = 1.0  # screenshot pixels per physical pixel
        self.max_long_edge = max_long_edge

    def _xy(self, coordinate: list[int] | None) -> tuple[int, int] | None:
        if not coordinate:
            return None
        return round(coordinate[0] / self.scale), round(coordinate[1] / self.scale)

    def screenshot_block(self) -> dict[str, Any]:
        from sohayok.tools.screen import grab, png_bytes

        data, self.scale = png_bytes(grab(), self.max_long_edge)
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": base64.b64encode(data).decode(),
            },
        }

    def run(self, name: str, args: dict[str, Any]) -> list[dict[str, Any]]:
        from pynput.mouse import Controller

        from sohayok import winapi
        from sohayok.tools.screen import grab, png_bytes

        mouse = Controller()
        # Clicks, drags, and scrolls take held keys ("shift", "ctrl+shift") in `text`.
        modifiers = (
            [k for k in str(args.get("text") or "").split("+") if k]
            if name.endswith(("click", "click_drag")) or name == "scroll"
            else []
        )
        held = [winapi.vk_for(CLAUDE_KEYS.get(m.lower(), m)) for m in modifiers]
        point = self._xy(args.get("coordinate"))
        if name == "screenshot":
            return [self.screenshot_block()]
        if name == "zoom":
            x0, y0, x1, y1 = args["region"]
            box = tuple(round(v / self.scale) for v in (x0, y0, x1, y1))
            data, _ = png_bytes(grab().crop(box), self.max_long_edge)
            return [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": base64.b64encode(data).decode(),
                    },
                }
            ]
        if name == "cursor_position":
            x, y = mouse.position
            return [
                {"type": "text", "text": f"X={round(x * self.scale)}, Y={round(y * self.scale)}"}
            ]
        if point and name != "left_click_drag":
            mouse.position = point
            time.sleep(0.05)
        for vk in held:
            winapi._send([winapi._key(vk, False)])
        try:
            return self._act(name, args, mouse, point)
        finally:
            for vk in reversed(held):
                winapi._send([winapi._key(vk, True)])

    def _act(self, name: str, args: dict[str, Any], mouse: Any, point: Any) -> list[dict]:
        from pynput.mouse import Button

        from sohayok import winapi

        if name in {"left_click", "right_click", "middle_click", "double_click", "triple_click"}:
            button = {"right_click": Button.right, "middle_click": Button.middle}.get(
                name, Button.left
            )
            count = {"double_click": 2, "triple_click": 3}.get(name, 1)
            mouse.click(button, count)
        elif name == "left_click_drag":
            start = self._xy(args.get("start_coordinate")) or mouse.position
            end = point or mouse.position
            mouse.position = start
            time.sleep(0.05)
            mouse.press(Button.left)
            # Move in steps: apps like Paint ignore a drag that jumps straight to the end.
            for i in range(1, 13):
                mouse.position = (
                    round(start[0] + (end[0] - start[0]) * i / 12),
                    round(start[1] + (end[1] - start[1]) * i / 12),
                )
                time.sleep(0.015)
            time.sleep(0.05)
            mouse.release(Button.left)
        elif name == "left_mouse_down":
            mouse.press(Button.left)
        elif name == "left_mouse_up":
            mouse.release(Button.left)
        elif name == "mouse_move":
            pass
        elif name == "scroll":
            amount = int(args.get("scroll_amount", 3))
            dx, dy = {
                "up": (0, amount),
                "down": (0, -amount),
                "left": (-amount, 0),
                "right": (amount, 0),
            }[args.get("scroll_direction", "down")]
            mouse.scroll(dx, dy)
        elif name == "type":
            winapi.type_unicode(str(args.get("text", "")))
        elif name in {"key", "hold_key"}:
            combo = "+".join(CLAUDE_KEYS.get(k.lower(), k) for k in str(args["text"]).split("+"))
            if name == "key":
                winapi.press(combo, int(args.get("repeat", 1)))
            else:
                keys = [winapi.vk_for(k) for k in combo.split("+")]
                winapi._send([winapi._key(vk, False) for vk in keys])
                time.sleep(min(float(args.get("duration", 1)), 10))
                winapi._send([winapi._key(vk, True) for vk in reversed(keys)])
        elif name == "wait":
            time.sleep(min(float(args.get("duration", 1)), 30))
        else:
            return [{"type": "text", "text": f"unsupported action {name}"}]
        return [{"type": "text", "text": "ok"}]


class ClaudeBrain:
    def __init__(
        self,
        model: str = "claude-sonnet-5-5",
        name: str = "Sohayok",
        city: str = "Dhaka",
        max_steps: int = 25,
        computer_use: bool = True,
        client: Any = None,
    ) -> None:
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self.client = client
        self.model = model
        self.name = name
        self.city = city
        self.max_steps = max_steps
        self.computer = ComputerUse() if computer_use else None

    def respond(
        self,
        text: str,
        language: str,
        executor: Executor,
        history: list[dict[str, str]],
        context: dict[str, str],
        on_text: Callable[[str], None] | None = None,
    ) -> Reply:
        tools: list[dict[str, Any]] = [
            {"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
            for t in executor.registry.schemas() + [CONFIRM_TOOL]
        ]
        if self.computer:
            tools.append(COMPUTER_TOOLSET)
        system = system_prompt(self.name, self.city)
        request = user_message(text, language, context.get("window", ""))
        messages: list[dict[str, Any]] = [*history, {"role": "user", "content": request}]
        calls: list[dict[str, Any]] = []
        for step in range(1, self.max_steps + 1):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                system=system,
                tools=tools,
                messages=messages,
            )
            content = [block.model_dump(exclude_none=True) for block in response.content]
            messages.append({"role": "assistant", "content": content})
            uses = [block for block in content if block.get("type") == "tool_use"]
            spoken = " ".join(b.get("text", "") for b in content if b.get("type") == "text").strip()
            if not uses:
                return Reply(spoken, calls, step)
            if spoken and on_text:
                on_text(spoken)
            results = []
            for use in uses:
                name, args = use["name"], use.get("input") or {}
                result: dict[str, Any] = {"type": "tool_result", "tool_use_id": use["id"]}
                if use.get("toolset_name") == "computer" and self.computer:
                    result["toolset_name"] = "computer"
                    executor.prepare("computer")
                    try:
                        result["content"] = self.computer.run(name, args)
                    except Exception as exc:  # report to Claude instead of crashing
                        result["content"] = [{"type": "text", "text": f"error: {exc}"}]
                        result["is_error"] = True
                    executor.on_tool(f"computer.{name}", args, "ok")
                    calls.append({"name": f"computer.{name}", "args": args, "result": "ok"})
                elif name == "confirm_with_user":
                    answer = "yes" if executor.ask_user(str(args.get("question", ""))) else "no"
                    result["content"] = answer
                    calls.append({"name": name, "args": args, "result": answer})
                else:
                    output = executor.run(name, args)
                    result["content"] = output
                    calls.append({"name": name, "args": args, "result": output})
                results.append(result)
            messages.append({"role": "user", "content": results})
        log.warning("stopped after %d steps", self.max_steps)
        return Reply("", calls, self.max_steps)
