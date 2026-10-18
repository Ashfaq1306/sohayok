"""Recompute WER/CER from saved transcripts (bench.stt writes one JSONL per model and language)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from sohayok.bench.stt import error_rates


def rescore(folder: Path) -> dict[str, dict[str, float]]:
    scores = {}
    for path in sorted(folder.glob("*.jsonl")):
        rows = [json.loads(line) for line in path.read_text("utf-8").splitlines() if line]
        rates = error_rates([r["reference"] for r in rows], [r["hypothesis"] for r in rows])
        scores[path.stem] = {"n": len(rows), **{k: round(v, 4) for k, v in rates.items()}}
    return scores


if __name__ == "__main__":
    print(json.dumps(rescore(Path(sys.argv[1])), indent=2))
