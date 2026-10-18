"""Fine-tune Laya-Bangla to pick Sohayok's desktop action.

Adapted from laya-bangla's trainer (github.com/shamspias/laya-bangla): upstream `laya.train`
does the training; this adds a fixed validation split for epoch selection and temperature
calibration, and freezes the token embedding so full fine-tuning fits in 8 GB.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

TRAINER = dict(
    epochs=2, micro_batch=32, grad_accum=2, encoder_lr=3e-5, head_lr=1e-4, min_lr=1e-6,
    weight_decay=0.01, loss="soft-ce", shuffle_options=("choice",), label_smoothing=0.05,
    max_len=512, head_max_len=320, amp=True, gradient_checkpointing=True, log_every=100,
    seed=20261018,
)


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]


def accuracy_nll(records) -> dict:
    correct, nll = 0, 0.0
    for _qtype, row, target, k in records:
        logits = [float(v) for v in row[:k]]
        top = max(range(k), key=logits.__getitem__)
        best = max(range(k), key=lambda i: target[i])
        correct += top == best
        peak = max(logits)
        nll -= logits[best] - peak - math.log(sum(math.exp(v - peak) for v in logits))
    return {"accuracy": correct / len(records), "nll": nll / len(records)}


def finetune(base: str, data: Path, output: Path) -> dict:
    import torch
    from laya.calibrate import fit_temperature_map
    from laya.train import (
        TrainConfig,
        calibration_records,
        items_from_rows,
        load_checkpoint,
        save_checkpoint,
        train_model,
    )

    trainer = TrainConfig(**TRAINER)
    trainer.validate()
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(TRAINER["seed"])
    model, tok, cfg = load_checkpoint(base)
    items = {}
    for split in ("train", "validation"):
        items[split], skipped = items_from_rows(
            tok, read_rows(data / f"{split}.jsonl"), trainer.max_len, trainer.head_max_len,
            label_smoothing=trainer.label_smoothing if split == "train" else 0.0,
        )
        print(f"{split}: {len(items[split])} items, skipped {skipped}", flush=True)
    model.encoder.get_input_embeddings().weight.requires_grad_(False)
    model.to(device)
    history, best = [], {"nll": math.inf}

    def evaluate(epoch, loss=None):
        records = calibration_records(model, tok, items["validation"], device, trainer.max_len,
                                      trainer.head_max_len)
        metrics = dict(accuracy_nll(records), epoch=epoch, train_loss=loss)
        history.append(metrics)
        print(f"validation after epoch {epoch}: {metrics}", flush=True)
        if epoch > 0 and metrics["nll"] < best["nll"]:
            best.update(metrics, state={k: v.detach().cpu() for k, v in model.state_dict().items()})
        model.train()

    evaluate(0)
    started = time.perf_counter()
    train_model(model, tok, items["train"], trainer, device, trainer.max_len,
                trainer.head_max_len, on_epoch_end=lambda e, loss: evaluate(e + 1, loss))
    seconds = time.perf_counter() - started
    model.load_state_dict(best.pop("state"))
    model.to(device).eval()
    records = calibration_records(model, tok, items["validation"], device, trainer.max_len,
                                  trainer.head_max_len)
    fitted = fit_temperature_map(records)
    out_cfg = dict(cfg, max_len=trainer.max_len, head_max_len=trainer.head_max_len,
                   fine_tuned=True, temperature=fitted["temperature"],
                   temperature_by_options=fitted["temperature_by_options"])
    save_checkpoint(model, tok, out_cfg, str(output))
    report = {
        "base": base, "finished_at": datetime.now(UTC).isoformat(),
        "train_seconds": round(seconds, 1), "trainer": asdict(trainer),
        "validation_history": history, "selected_epoch": best["epoch"],
        "temperature": fitted["temperature"],
        "peak_gpu_memory_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)
        if torch.cuda.is_available() else None,
    }
    (output / "train_report.json").write_text(json.dumps(report, indent=2), "utf-8")
    return report


if __name__ == "__main__":
    import sys

    print(json.dumps(finetune(sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])), indent=2))
