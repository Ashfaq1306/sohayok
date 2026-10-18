import json
import os
import shutil
from pathlib import Path

import pytest

from sohayok.models import build_tokenizer_json, resolve_bangla_model

SOURCE = Path(
    os.environ.get("SOHAYOK_BUZZASR_SOURCE", "E:/projects/sohayok-data/models/BuzzASR-bengali")
)
TOKENIZER_FILES = [
    "added_tokens.json",
    "generation_config.json",
    "merges.txt",
    "normalizer.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "vocab.json",
]


@pytest.mark.skipif(not (SOURCE / "vocab.json").exists(), reason="BuzzASR files not downloaded")
def test_bangla_tokenizer_ids_match_the_model(tmp_path):
    for name in TOKENIZER_FILES:
        shutil.copy(SOURCE / name, tmp_path / name)
    expected = build_tokenizer_json(tmp_path)
    assert expected["<|bn|>"] == 51910 and expected["<|transcribe|>"] == 51968

    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tmp_path / "tokenizer.json"))
    text = "আমাদের এলাকায় তিন দিন ধরে পানি নেই।"
    assert tokenizer.decode(tokenizer.encode(text, add_special_tokens=False).ids) == text
    data = json.loads((tmp_path / "tokenizer.json").read_text("utf-8"))
    assert len(data["model"]["vocab"]) == 51865


def test_resolve_bangla_model(tmp_path, monkeypatch):
    import sohayok.models as models

    monkeypatch.setattr(models, "bangla_model_dir", lambda: tmp_path)
    assert resolve_bangla_model("auto") == ""
    (tmp_path / "model.bin").write_bytes(b"x")
    assert resolve_bangla_model("auto") == str(tmp_path)
    assert resolve_bangla_model("") == ""
    assert resolve_bangla_model("D:/m") == "D:/m"
