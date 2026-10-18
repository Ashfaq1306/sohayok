"""Print Markdown tables from saved benchmark outputs (for docs/benchmarks.md)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from sohayok.bench.brain import summarize
from sohayok.bench.rescore import rescore


def stt_table(folder: Path) -> str:
    scores = rescore(folder)
    names = sorted({key.rsplit(".", 1)[0] for key in scores})
    lines = [
        "| Model | Bangla WER | Bangla CER | English WER | English CER |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name in names:
        bn, en = scores.get(f"{name}.bn", {}), scores.get(f"{name}.en", {})

        def cell(block, key):
            return f"{block[key] * 100:.1f}%" if block else "—"

        lines.append(
            f"| {name} | {cell(bn, 'wer')} | {cell(bn, 'cer')} | {cell(en, 'wer')} | "
            f"{cell(en, 'cer')} |"
        )
    return "\n".join(lines)


def brain_table(folder: Path, suffix: str = "") -> str:
    lines = [
        "| Pipeline | Cases | Right tool | Right tool + args | Reply language | Median s |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for path in sorted(folder.glob(f"*{suffix}.jsonl")):
        if suffix == "" and ".model-only" in path.name:
            continue
        rows = [json.loads(line) for line in path.read_text("utf-8").splitlines() if line]
        for scope, subset in [
            ("all", rows),
            ("Bangla", [r for r in rows if r["language"] == "bn"]),
            ("English", [r for r in rows if r["language"] == "en"]),
            ("MASSIVE", [r for r in rows if r["source"] == "massive"]),
            ("desktop", [r for r in rows if r["source"] == "desktop"]),
        ]:
            block = summarize(subset)["all"]
            lines.append(
                f"| {path.stem} · {scope} | {block['n']} | {block['tool_accuracy'] * 100:.1f}% | "
                f"{block['args_accuracy'] * 100:.1f}% | {block['reply_language'] * 100:.1f}% | "
                f"{block['median_seconds']} |"
            )
    return "\n".join(lines)


if __name__ == "__main__":
    root = Path(sys.argv[1])
    print("## Speech recognition\n\n" + stt_table(root / "stt"))
    print("\n## Command understanding\n\n" + brain_table(root / "brain"))
