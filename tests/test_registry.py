from sohayok.tools import load_all
from sohayok.tools.registry import Registry, Risk, ToolError, string, tool


def test_decorator_builds_schema_and_required():
    registry = Registry()

    @tool("Greet.", {"name": string("Who"), "loud": {"type": "boolean"}}, registry=registry)
    def greet(name: str, loud: bool = False) -> str:
        return name.upper() if loud else name

    schema = registry.get("greet").schema()
    assert schema["parameters"]["required"] == ["name"]
    assert registry.call("greet", {"name": "ami", "loud": True}) == "AMI"


def test_call_ignores_invented_args_and_reports_missing_ones():
    registry = Registry()

    @tool("Echo.", {"text": string("x")}, registry=registry)
    def echo(text: str) -> dict:
        return {"text": text}

    assert registry.call("echo", {"text": "হ্যাঁ", "bogus": 1}) == '{"text": "হ্যাঁ"}'
    assert registry.call("echo", {}).startswith("error: missing argument")
    assert registry.call("nope", {}).startswith("error: unknown tool")


def test_errors_become_text():
    registry = Registry()

    @tool("Fail.", registry=registry)
    def fail() -> None:
        raise ToolError("no such window")

    @tool("Crash.", registry=registry)
    def crash() -> None:
        raise RuntimeError("boom")

    assert registry.call("fail", {}) == "error: no such window"
    assert registry.call("crash", {}) == "error: RuntimeError: boom"


def test_all_tools_have_valid_schemas():
    registry = load_all()
    assert len(registry.tools) >= 40
    for name, item in registry.tools.items():
        params = item.parameters
        assert params["type"] == "object", name
        assert set(params["required"]) <= set(params["properties"]), name
        assert item.description.strip(), name
    risky = {n for n, t in registry.tools.items() if t.risk is Risk.CONFIRM}
    assert {
        "power",
        "recycle",
        "run_powershell",
        "ask_claude_code",
        "move_path",
        "write_text_file",
    } <= risky
    assert "open_app" not in risky and "volume" not in risky
