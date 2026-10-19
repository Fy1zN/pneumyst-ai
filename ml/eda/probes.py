"""Shortcut probes, normalization comparison and NIH duplicate search."""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import ml.eda.run_eda as eda
from ml.datasets.common import _POPCOUNT, hash_files, lsh_pairs, to_8bit_gray
from ml.datasets.targets import TB, label_column

TB_COL = label_column(TB)
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
NORMALIZATIONS = ["none", "percentile", "clahe"]


# ---------- image features ----------

def load_thumb(path: Path, size: int = 256) -> np.ndarray | None:
    try:
        with Image.open(path) as im:
            g = to_8bit_gray(im)
            g.thumbnail((size, size))
            return np.asarray(g, dtype=np.uint8)
    except OSError:
        return None


def normalize(px: np.ndarray, method: str) -> np.ndarray:
    if method == "percentile":
        lo, hi = np.percentile(px, (1, 99))
        if hi <= lo:
            return px
        return np.clip((px.astype(np.float32) - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    if method == "clahe":
        return _CLAHE.apply(px)
    return px


def features(px: np.ndarray) -> dict:
    """Global, lung-agnostic statistics a shortcut could exploit."""
    busy_r = np.where(px.std(axis=1) > 4)[0]
    busy_c = np.where(px.std(axis=0) > 4)[0]
    content = (np.ptp(busy_r) + 1) * (np.ptp(busy_c) + 1) if len(busy_r) and len(busy_c) else 0
    p1, p50, p99 = np.percentile(px, (1, 50, 99))
    hist = np.histogram(px, bins=16, range=(0, 256))[0] / px.size
    out = {"mean": px.mean(), "std": px.std(), "p1": p1, "p50": p50, "p99": p99,
           "sharpness": np.log10(cv2.Laplacian(px, cv2.CV_64F).var() + 1),
           "border_frac": 1 - content / px.size,
           "black_frac": (px <= 5).mean(), "white_frac": (px >= 250).mean()}
    out.update({f"h{i:02d}": v for i, v in enumerate(hist)})
    return {k: float(v) for k, v in out.items()}


def feature_table(rows: pd.DataFrame, root: Path, group_col: str, methods=("none",), desc="features") -> pd.DataFrame:
    recs = []
    for r in tqdm(rows.itertuples(index=False), total=len(rows), desc=desc):
        px = load_thumb(root / r.raw_path)
        if px is None:
            continue
        for m in methods:
            recs.append({"image_id": r.image_id, "group": getattr(r, group_col), "method": m, **features(normalize(px, m))})
    return pd.DataFrame(recs)


def probe_auc(feats: pd.DataFrame, a: str, b: str, seed: int = 42) -> float | None:
    """Cross-validated AUROC of a linear model separating group a from b using global stats only."""
    part = feats[feats["group"].isin([a, b])]
    y = (part["group"] == b).astype(int).to_numpy()
    if min(np.bincount(y, minlength=2)) < 5:
        return None
    x = part.drop(columns=["image_id", "group", "method"]).to_numpy()
    folds = StratifiedKFold(n_splits=min(5, int(min(np.bincount(y)))), shuffle=True, random_state=seed)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    p = cross_val_predict(model, x, y, cv=folds, method="predict_proba")[:, 1]
    return round(float(roc_auc_score(y, p)), 3)


# ---------- 20. TBX11K class shortcut probe ----------

def tbx_class_probe(tbx: pd.DataFrame, root: Path, out: Path, n_per_class: int = 800, seed: int = 42,
                    methods: tuple = ("none",)) -> tuple[pd.DataFrame, pd.DataFrame]:
    pick = pd.concat([part.sample(min(n_per_class, len(part)), random_state=seed)
                      for _, part in tbx.groupby("original_label")])
    all_feats = feature_table(pick, root, "original_label", methods=methods, desc="TBX11K classes")
    feats = all_feats[all_feats["method"] == methods[0]]
    stats = feats.groupby("group")[["mean", "std", "p1", "p99", "sharpness", "border_frac", "black_frac", "white_frac"]].median().round(3)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, col in zip(axes, ["mean", "std", "sharpness", "black_frac"]):
        feats.boxplot(column=col, by="group", ax=ax)
        ax.set_title(col)
        ax.set_xlabel("")
    fig.suptitle("TBX11K: global image stats by class")
    eda.save(fig, out / "20_tbx11k_class_stats.png")

    recs = [{"comparison": f"{a} vs {b}", "method": m,
             "probe_auroc": probe_auc(all_feats[all_feats["method"] == m], a, b, seed)}
            for m in methods for a, b in (("health", "tb"), ("sick", "tb"), ("health", "sick"))]
    aucs = pd.DataFrame(recs).pivot(index="comparison", columns="method", values="probe_auroc")[list(methods)]
    return stats, aucs


# ---------- 21 + 22. Hospital fingerprint and normalization ----------

def origin_probe(dfs: dict, root: Path, out: Path, n: int = 400, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for name in ("tbx11k", "shenzhen", "montgomery"):
        if name in dfs:
            normal = dfs[name][dfs[name][TB_COL] == 0]
            rows.append(normal.sample(min(n, len(normal)), random_state=seed))
    pick = pd.concat(rows)
    feats = feature_table(pick, root, "source", methods=NORMALIZATIONS, desc="normals x 3 methods")

    recs = []
    for m in NORMALIZATIONS:
        fm = feats[feats["method"] == m]
        for a, b in combinations(sorted(fm["group"].unique()), 2):
            recs.append({"method": m, "comparison": f"{a} vs {b}", "probe_auroc": probe_auc(fm, a, b, seed)})
    table = pd.DataFrame(recs).pivot(index="comparison", columns="method", values="probe_auroc")[NORMALIZATIONS]

    fig, axes = plt.subplots(1, len(NORMALIZATIONS), figsize=(15, 4), sharey=True)
    for ax, m in zip(axes, NORMALIZATIONS):
        for src, part in feats[feats["method"] == m].groupby("group"):
            ax.plot(part[[c for c in part.columns if c.startswith("h")]].mean().to_numpy(), label=src)
        ax.set_title(f"Mean histogram · {m}")
        ax.set_xlabel("Intensity bin (16)")
        ax.legend(fontsize=8)
    eda.save(fig, out / "21_22_hospital_fingerprint_by_normalization.png")
    return table, feats


def normalization_preview(dfs: dict, root: Path, local: Path, per_source: int = 2, seed: int = 42) -> None:
    items = []
    for name in ("tbx11k", "shenzhen", "montgomery", "nih"):
        if name not in dfs:
            continue
        for p in dfs[name].sample(min(per_source, len(dfs[name])), random_state=seed)["raw_path"]:
            px = load_thumb(root / p)
            if px is not None:
                items.append((name, [normalize(px, m) for m in NORMALIZATIONS]))
    if not items:
        return
    fig, axes = plt.subplots(len(items), len(NORMALIZATIONS), figsize=(9, 3 * len(items)), squeeze=False)
    for row, (name, imgs) in zip(axes, items):
        for ax, img, m in zip(row, imgs, NORMALIZATIONS):
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            ax.set_title(f"{name} · {m}", fontsize=8)
            ax.axis("off")
    eda.save(fig, local / "normalization_preview.png")


# ---------- 23. NIH hashing + LSH duplicate search ----------

def nih_hashes(nih: pd.DataFrame, root: Path, cache: Path, workers: int = 4) -> pd.DataFrame:
    if cache.exists():
        cached = pd.read_csv(cache, dtype=str, keep_default_na=False)
        if len(cached) == len(nih):
            return cached
    results = hash_files([str(root / p) for p in nih["raw_path"]], workers, desc="hashing NIH")
    out = pd.DataFrame({"image_id": nih["image_id"].to_numpy(),
                        "sha256": [r[0] for r in results], "phash": [r[1] for r in results]})
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache, index=False)
    return out


def nih_duplicates(nih: pd.DataFrame, hashes: pd.DataFrame, out_csv: Path, max_distance: int = 6) -> tuple[pd.DataFrame, dict]:
    h = nih[["image_id", "patient_id", "split"]].merge(hashes, on="image_id")
    h = h[h["phash"] != ""].reset_index(drop=True)
    pairs, skipped = lsh_pairs(h["phash"].tolist(), max_distance)
    recs = []
    for i, j, d in pairs:
        a, b = h.loc[i], h.loc[j]
        same_patient = a["patient_id"] == b["patient_id"]
        cross = a["split"] != b["split"]
        kind = ("LEAK: different patients, different splits" if cross and not same_patient else
                "different patients, same split" if not same_patient else "same patient")
        recs.append({"image_a": a["image_id"], "patient_a": a["patient_id"], "split_a": a["split"],
                     "image_b": b["image_id"], "patient_b": b["patient_id"], "split_b": b["split"],
                     "phash_distance": d, "exact_match": a["sha256"] == b["sha256"], "kind": kind})
    report = pd.DataFrame(recs, columns=["image_a", "patient_a", "split_a", "image_b", "patient_b", "split_b",
                                         "phash_distance", "exact_match", "kind"])
    report.to_csv(out_csv, index=False)
    summary = {"hashed": int(len(h)), "unreadable": int(len(nih) - len(h)), "pairs": int(len(report)),
               "oversized_buckets_skipped": skipped, "by_kind": report["kind"].value_counts().to_dict()}
    return report, summary


def duplicate_grid(report: pd.DataFrame, nih: pd.DataFrame, root: Path, path: Path, max_pairs: int = 8) -> None:
    leaks = report[report["kind"].str.startswith("LEAK")].sort_values("phash_distance").head(max_pairs)
    if leaks.empty:
        return
    raw = nih.set_index("image_id")["raw_path"]
    fig, axes = plt.subplots(len(leaks), 2, figsize=(6, 3 * len(leaks)), squeeze=False)
    for row, r in zip(axes, leaks.itertuples()):
        for ax, img_id, split in ((row[0], r.image_a, r.split_a), (row[1], r.image_b, r.split_b)):
            px = load_thumb(root / raw[img_id])
            if px is not None:
                ax.imshow(px, cmap="gray", vmin=0, vmax=255)
            ax.set_title(f"{img_id.split(':')[1]} [{split}] d={r.phash_distance}", fontsize=8)
            ax.axis("off")
    eda.save(fig, path)
