"""Download and prepare the speech models (pinned revisions, verified tokenizer)."""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path

from sohayok.config import app_dir

log = logging.getLogger(__name__)

# BuzzASR Bengali: Whisper large-v3 fine-tuned for Bengali (MIT; arXiv 2609.09554).
BANGLA_REPO = "BuzzASR/bengali"
BANGLA_REVISION = "79f1010fece241070161a9b72ea644f70e71a070"
BANGLA_FILES = [
    "added_tokens.json",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors",
    "normalizer.json",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "vocab.json",
]


def bangla_model_dir() -> Path:
    return app_dir() / "models" / "BuzzASR-bengali-ct2"


def resolve_bangla_model(setting: str) -> str:
    """'auto' means the prepared model if `sohayok setup` has been run, else none."""
    if setting != "auto":
        return setting
    path = bangla_model_dir()
    return str(path) if (path / "model.bin").exists() else ""


def build_tokenizer_json(source: Path) -> dict:
    """Write a fast-tokenizer file whose special-token ids match the model's training.

    The repository's vocab.json also lists ~1,500 special strings ("<|endoftext|>",
    "<|transcribe|>", ...) at low ids, while added_tokens.json places them at 51865+.
    A naive conversion gives those tokens the low ids and shifts every language token by
    two (<|bn|> becomes 51908 instead of 51910), so the model would be prompted wrongly.
    Here the clashing vocab entries are renamed (ids kept) and the added tokens take
    their declared ids, which is checked against generation_config.json.
    """
    from transformers import WhisperTokenizerFast

    added = json.loads((source / "added_tokens.json").read_text("utf-8"))
    generation = json.loads((source / "generation_config.json").read_text("utf-8"))
    scratch = source / "_plain_tokenizer"
    WhisperTokenizerFast.from_pretrained(source).save_pretrained(scratch)
    data = json.loads((scratch / "tokenizer.json").read_text("utf-8"))
    shutil.rmtree(scratch, ignore_errors=True)
    vocab = data["model"]["vocab"]
    if len(vocab) != min(added.values()):
        raise ValueError("unexpected vocabulary layout; refusing to guess token ids")
    for content in [c for c in vocab if c in added]:
        index = vocab.pop(content)
        vocab[f"<|unused_{index}|>"] = index
    data["added_tokens"] = [
        {
            "id": index,
            "content": content,
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        }
        for content, index in sorted(added.items(), key=lambda item: item[1])
    ]

    def remap(node):
        if isinstance(node, dict):
            if "ids" in node and "id" in node and node["id"] in added:
                node["ids"] = [added[node["id"]]]
            for value in node.values():
                remap(value)
        elif isinstance(node, list):
            for value in node:
                remap(value)

    remap(data.get("post_processor"))
    (source / "tokenizer.json").write_text(json.dumps(data, ensure_ascii=False), "utf-8")

    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(source / "tokenizer.json"))
    expected = {
        "<|bn|>": generation["lang_to_id"]["<|bn|>"],
        "<|transcribe|>": generation["task_to_id"]["transcribe"],
        "<|notimestamps|>": generation["no_timestamps_token_id"],
        "<|startoftranscript|>": generation["decoder_start_token_id"],
        "<|endoftext|>": generation["eos_token_id"],
    }
    wrong = {
        t: (tokenizer.token_to_id(t), i)
        for t, i in expected.items()
        if tokenizer.token_to_id(t) != i
    }
    if wrong:
        raise ValueError(f"tokenizer ids do not match the model: {wrong}")
    return expected


def set_language_ids(model_dir: Path) -> list[int]:
    """Record the language tokens in the CTranslate2 config.

    The converter finds none for this tokenizer, and CTranslate2 then treats the model as
    English-only: faster-whisper drops the <|bn|> prompt and transcribes as English.
    """
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    start = tokenizer.token_to_id("<|startoftranscript|>")
    translate = tokenizer.token_to_id("<|translate|>")
    ids = list(range(start + 1, translate))
    names = [tokenizer.id_to_token(i) for i in ids]
    if not all(n.startswith("<|") and n.endswith("|>") and len(n) <= 7 for n in names):
        raise ValueError(f"unexpected language tokens: {names[:5]}…")
    if tokenizer.token_to_id("<|bn|>") not in ids:
        raise ValueError("<|bn|> is not a language token")
    config_path = model_dir / "config.json"
    config = json.loads(config_path.read_text("utf-8"))
    config["lang_ids"] = ids
    config_path.write_text(json.dumps(config), "utf-8")
    return ids


def prepare_bangla_model(destination: Path | None = None, keep_source: bool = False) -> Path:
    """Download BuzzASR Bengali, fix its tokenizer, and convert it to CTranslate2 int8."""
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # plain HTTPS downloads resume reliably
    from ctranslate2.converters import TransformersConverter
    from huggingface_hub import hf_hub_download

    destination = destination or bangla_model_dir()
    source = destination.parent / "BuzzASR-bengali-source"
    source.mkdir(parents=True, exist_ok=True)
    for name in BANGLA_FILES:
        if not (source / name).exists():
            log.info("downloading %s", name)
            hf_hub_download(BANGLA_REPO, name, revision=BANGLA_REVISION, local_dir=source)
    build_tokenizer_json(source)
    TransformersConverter(
        str(source),
        copy_files=["tokenizer.json", "preprocessor_config.json"],
        low_cpu_mem_usage=True,
    ).convert(str(destination), quantization="int8_float16", force=True)
    set_language_ids(destination)
    (destination / "SOURCE.txt").write_text(
        f"{BANGLA_REPO}@{BANGLA_REVISION}, converted to CTranslate2 int8_float16 by sohayok\n",
        "utf-8",
    )
    if not keep_source:
        shutil.rmtree(source, ignore_errors=True)
    return destination
