"""Rebuild Sohayok's action decider from scratch: download MASSIVE, build data, fine-tune."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

MASSIVE_REPO = "mteb/amazon_massive_intent"
MASSIVE_REVISION = "940fd47a81eaa7f2cc7b129674d945d618ac38c2"


def train_decider(app_dir: Path) -> Path:
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import hf_hub_download, snapshot_download

    from sohayok.decide import LAYA_REPO, LAYA_REVISION
    from sohayok.training.actions_data import build
    from sohayok.training.finetune_decider import finetune

    work = app_dir / "training"
    massive = work / "massive"
    massive.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation"):
        for language in ("bn", "en"):
            path = hf_hub_download(
                MASSIVE_REPO,
                f"{split}/{language}.json.gz",
                repo_type="dataset",
                revision=MASSIVE_REVISION,
            )
            shutil.copy(path, massive / f"{split}.{language}.json.gz")
    build(massive, work / "decider-data")
    base = snapshot_download(LAYA_REPO, revision=LAYA_REVISION)
    output = app_dir / "models" / "laya-sohayok-actions"
    staging = work / "laya-sohayok-actions"
    shutil.rmtree(staging, ignore_errors=True)
    finetune(base, work / "decider-data", staging)
    shutil.rmtree(output, ignore_errors=True)
    shutil.move(str(staging), str(output))
    return output
