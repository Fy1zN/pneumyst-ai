"""Train the TB or NIH model with AMP, gradient accumulation, freeze-then-unfreeze and resumable checkpoints.

Usage: python -m ml.training.train --task tb [--epochs 20] [--batch-size 16] [--accum 2] [--workers 4]
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ml.datasets.targets import NIH_FINDINGS, TB, label_column
from ml.training.augment import AugConfig
from ml.training.data import XrayDataset, load_processed
from ml.training.metrics import multilabel_auroc, safe_auprc, safe_auroc
from ml.training.model import build_model, set_backbone_trainable

REPO_ROOT = Path(__file__).resolve().parents[2]
TASKS = {
    "tb": {"source": "tbx11k", "labels": [label_column(TB)], "pad_p": 0.5},
    "nih": {"source": "nih", "labels": [label_column(f) for f in NIH_FINDINGS], "pad_p": 0.3},
}


@dataclass
class TrainConfig:
    task: str = "tb"                  # "tb" or "nih"
    run_name: str = "tb_densenet121"
    epochs: int = 20
    frozen_epochs: int = 2            # train only the new output layer first
    batch_size: int = 16
    accum_steps: int = 2              # effective batch = batch_size * accum_steps
    lr_head: float = 1e-3
    lr_backbone: float = 1e-4
    weight_decay: float = 1e-4
    warmup_epochs: float = 1.0
    patience: int = 5                 # early stopping on the validation metric (after the frozen stage)
    pos_weight_cap: float = 50.0      # limits the loss weight of very rare findings (Hernia)
    workers: int = 4
    amp: bool = True
    memory_efficient: bool = True
    pretrained: bool = True
    seed: int = 42
    limit: int | None = None          # use only N train and N val images (smoke runs)
    pad_p: float | None = None        # None = task default (TB 0.5, NIH 0.3)


# ---------- loss ----------

def compute_pos_weight(labels: np.ndarray, cap: float) -> torch.Tensor:
    """negatives / positives per label over known labels, capped."""
    pos = np.nansum(labels == 1, axis=0).astype(float)
    neg = np.nansum(labels == 0, axis=0).astype(float)
    return torch.tensor(np.clip(neg / np.maximum(pos, 1), 1.0, cap), dtype=torch.float32)


def masked_bce(logits: torch.Tensor, y: torch.Tensor, pos_weight: torch.Tensor) -> torch.Tensor:
    """BCE that ignores unknown (NaN) labels."""
    known = ~torch.isnan(y)
    loss = F.binary_cross_entropy_with_logits(logits, torch.nan_to_num(y), pos_weight=pos_weight, reduction="none")
    return (loss * known).sum() / known.sum().clamp(min=1)


# ---------- data ----------

def make_splits(cfg: TrainConfig, root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_processed(root, TASKS[cfg.task]["source"])
    train, val = df[df["split"] == "train"], df[df["split"] == "val"]
    if cfg.limit:
        train = train.sample(min(cfg.limit, len(train)), random_state=cfg.seed)
        val = val.sample(min(cfg.limit, len(val)), random_state=cfg.seed)
    return train.reset_index(drop=True), val.reset_index(drop=True)


def make_loaders(cfg: TrainConfig, root: Path, train: pd.DataFrame, val: pd.DataFrame) -> tuple[DataLoader, DataLoader]:
    labels = TASKS[cfg.task]["labels"]
    pad_p = TASKS[cfg.task]["pad_p"] if cfg.pad_p is None else cfg.pad_p
    common = {"num_workers": cfg.workers, "pin_memory": torch.cuda.is_available(),
              "persistent_workers": cfg.workers > 0}
    train_dl = DataLoader(XrayDataset(train, root, labels, AugConfig(pad_p=pad_p)), batch_size=cfg.batch_size,
                          shuffle=True, drop_last=len(train) > cfg.batch_size, **common)
    val_dl = DataLoader(XrayDataset(val, root, labels), batch_size=cfg.batch_size * 2, shuffle=False, **common)
    return train_dl, val_dl


# ---------- train / predict ----------

def train_one_epoch(model, loader, opt, sched, scaler, pos_weight, device, cfg: TrainConfig, frozen: bool) -> float:
    model.train()
    if frozen:
        model.features.eval()  # keep pretrained BatchNorm statistics while the backbone is frozen
    use_amp = cfg.amp and device.type == "cuda"
    total, seen = 0.0, 0
    opt.zero_grad(set_to_none=True)
    for step, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(x)
        loss = masked_bce(logits.float(), y, pos_weight)
        scaler.scale(loss / cfg.accum_steps).backward()
        if (step + 1) % cfg.accum_steps == 0 or step + 1 == len(loader):
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
        total += loss.item() * len(x)
        seen += len(x)
    return total / max(seen, 1)


@torch.no_grad()
def predict(model, loader, device, amp: bool = True) -> np.ndarray:
    model.eval()
    out = []
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            out.append(torch.sigmoid(model(x).float()).cpu().numpy())
    return np.concatenate(out)


def score(task: str, val: pd.DataFrame, probs: np.ndarray) -> dict:
    """Validation metrics; 'metric' is what model selection uses."""
    labels = TASKS[task]["labels"]
    y = val[labels].to_numpy(dtype=float)
    if task == "tb":
        p, yt = probs[:, 0], y[:, 0]
        hard = val["original_label"].isin(["sick", "tb"]).to_numpy()
        return {"metric": safe_auroc(yt, p), "auroc": safe_auroc(yt, p), "auprc": safe_auprc(yt, p),
                "auroc_sick_vs_tb": safe_auroc(yt[hard], p[hard])}
    per = multilabel_auroc(y, probs, NIH_FINDINGS)
    return {"metric": float(per.mean()), "mean_auroc": float(per.mean()),
            **{f"auroc_{k}": round(float(v), 4) for k, v in per.items()}}


def save_predictions(path: Path, task: str, val: pd.DataFrame, probs: np.ndarray) -> None:
    labels = TASKS[task]["labels"]
    out = val[["image_id"] + labels].copy()
    for i, col in enumerate(labels):
        out[col.replace("label_", "prob_")] = probs[:, i]
    out.to_csv(path, index=False)


def make_optimizer(model, cfg: TrainConfig, steps_per_epoch: int):
    opt = torch.optim.AdamW([{"params": model.features.parameters(), "lr": cfg.lr_backbone},
                             {"params": model.classifier.parameters(), "lr": cfg.lr_head}],
                            weight_decay=cfg.weight_decay)
    total = max(1, cfg.epochs * steps_per_epoch)
    warmup = max(1, int(cfg.warmup_epochs * steps_per_epoch))

    def factor(step: int) -> float:  # linear warmup, then cosine down to 1% of the start rate
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total - warmup))
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress))

    return opt, torch.optim.lr_scheduler.LambdaLR(opt, factor)


def fit(cfg: TrainConfig, root: Path, run_dir: Path | None = None, device: str | None = None, log=print) -> dict:
    """Train with resume: rerunning with the same run_dir continues from last.pt."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.backends.cudnn.benchmark = True
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    run_dir = Path(run_dir or REPO_ROOT / "ml" / "checkpoints" / cfg.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    train, val = make_splits(cfg, root)
    train_dl, val_dl = make_loaders(cfg, root, train, val)
    labels = TASKS[cfg.task]["labels"]
    pos_weight = compute_pos_weight(train[labels].to_numpy(dtype=float), cfg.pos_weight_cap).to(dev)
    log(f"{cfg.task}: train {len(train):,} | val {len(val):,} | device {dev} | "
        f"pos_weight {np.round(pos_weight.cpu().numpy(), 1).tolist()}")

    model = build_model(len(labels), cfg.pretrained, cfg.memory_efficient).to(dev)
    steps_per_epoch = math.ceil(len(train_dl) / cfg.accum_steps)
    opt, sched = make_optimizer(model, cfg, steps_per_epoch)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and dev.type == "cuda")

    state = {"epoch": 0, "best": -1.0, "bad_epochs": 0, "history": [], "stopped": False}
    last = run_dir / "last.pt"
    if last.exists():
        ckpt = torch.load(last, map_location=dev, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        sched.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        state = ckpt["state"]
        log(f"resumed from epoch {state['epoch']} (best {state['best']:.4f})")

    for epoch in range(state["epoch"], cfg.epochs):
        if state["stopped"]:
            break
        frozen = epoch < cfg.frozen_epochs
        set_backbone_trainable(model, not frozen)
        t0 = time.time()
        train_loss = train_one_epoch(model, train_dl, opt, sched, scaler, pos_weight, dev, cfg, frozen)
        probs = predict(model, val_dl, dev, cfg.amp)
        val_loss = float(masked_bce(torch.logit(torch.tensor(probs).clamp(1e-6, 1 - 1e-6)),
                                    torch.tensor(val[labels].to_numpy(dtype=np.float32)), pos_weight.cpu()))
        metrics = score(cfg.task, val, probs)
        row = {"epoch": epoch + 1, "frozen": frozen, "train_loss": round(train_loss, 4), "val_loss": round(val_loss, 4),
               "lr_backbone": opt.param_groups[0]["lr"], "minutes": round((time.time() - t0) / 60, 2), **metrics}
        state["history"].append(row)

        if metrics["metric"] > state["best"]:
            state["best"], state["bad_epochs"] = metrics["metric"], 0
            torch.save({"model": model.state_dict(), "config": asdict(cfg), "epoch": epoch + 1, "metrics": metrics,
                        "labels": labels}, run_dir / "best.pt")
            save_predictions(run_dir / "val_predictions.csv", cfg.task, val, probs)
            mark = " *best*"
        else:
            state["bad_epochs"] += 0 if frozen else 1
            mark = ""
        log(f"epoch {epoch + 1}/{cfg.epochs}{' (frozen)' if frozen else ''} | train {train_loss:.4f} | "
            f"val {val_loss:.4f} | metric {metrics['metric']:.4f} | {row['minutes']} min{mark}")

        state["epoch"] = epoch + 1
        if state["bad_epochs"] >= cfg.patience:
            state["stopped"] = True
            log(f"early stop: no improvement for {cfg.patience} epochs")
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                    "scaler": scaler.state_dict(), "state": state, "config": asdict(cfg)}, last)
        (run_dir / "history.json").write_text(json.dumps(state["history"], indent=2, default=str))

    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    return {"best": state["best"], "history": pd.DataFrame(state["history"]), "run_dir": run_dir}


def find_batch_size(cfg: TrainConfig, candidates=(32, 24, 16, 12, 8), size: int = 320) -> int | None:
    """Largest batch that survives a full forward + backward pass on the GPU (unfrozen, AMP)."""
    if not torch.cuda.is_available():
        return None
    dev = torch.device("cuda")
    model = build_model(len(TASKS[cfg.task]["labels"]), False, cfg.memory_efficient).to(dev)
    opt = torch.optim.AdamW(model.parameters())
    best, x, loss = None, None, None
    for bs in candidates:
        try:
            x = torch.randn(bs, 3, size, size, device=dev)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=cfg.amp):
                loss = model(x).float().mean()
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            peak = torch.cuda.max_memory_allocated() / 1e9
            print(f"batch {bs}: ok, peak {peak:.2f} GB")
            best = bs
            break
        except torch.cuda.OutOfMemoryError:
            print(f"batch {bs}: out of memory")
        finally:
            x = loss = None
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    del model, opt
    torch.cuda.empty_cache()
    return best


def benchmark_speed(configs: list[dict], steps: int = 15, size: int = 320, device: str | None = None) -> pd.DataFrame:
    """Images per second of a full training step (forward + backward + optimizer) for each
    {"batch_size": ..., "memory_efficient": ...}. Random input, so data loading is not included."""
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.backends.cudnn.benchmark = True
    rows = []
    for c in configs:
        bs, mem_eff = c["batch_size"], c["memory_efficient"]
        row = {"batch_size": bs, "memory_efficient": mem_eff, "images_per_s": float("nan"), "peak_gb": float("nan"),
               "status": "ok"}
        model = build_model(14, False, mem_eff).to(dev)
        opt = torch.optim.AdamW(model.parameters())
        scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
        x = torch.randn(bs, 3, size, size, device=dev)
        y = torch.randint(0, 2, (bs, 14), device=dev).float()
        logits = None
        try:
            if dev.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            for i in range(steps + 3):  # 3 warm-up steps are not timed
                if i == 3:
                    if dev.type == "cuda":
                        torch.cuda.synchronize()
                    t0 = time.time()
                with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                    logits = model(x)
                scaler.scale(F.binary_cross_entropy_with_logits(logits.float(), y)).backward()
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            if dev.type == "cuda":
                torch.cuda.synchronize()
                row["peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
            row["images_per_s"] = round(steps * bs / (time.time() - t0), 1)
        except torch.cuda.OutOfMemoryError:
            row["status"] = "out of memory"
        finally:
            del model, opt, x, y, logits
            if dev.type == "cuda":
                torch.cuda.empty_cache()
        print(f"batch {bs:>2} | memory_efficient {mem_eff!s:>5} | {row['images_per_s']} img/s | "
              f"peak {row['peak_gb']} GB | {row['status']}")
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    from ml.config import data_root

    parser = argparse.ArgumentParser(description="Train the PNEUMYST AI TB or NIH model.")
    parser.add_argument("--task", choices=list(TASKS), default="tb")
    parser.add_argument("--run-name")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--accum", type=int, dest="accum_steps")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--limit", type=int)
    args = {k: v for k, v in vars(parser.parse_args()).items() if v is not None}
    args.setdefault("run_name", f"{args['task']}_densenet121")
    cfg = replace(TrainConfig(), **args)
    result = fit(cfg, data_root())
    print(f"best metric {result['best']:.4f} -> {result['run_dir']}")


if __name__ == "__main__":
    main()