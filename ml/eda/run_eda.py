"""Exploratory data analysis on the manifests.

Usage: python -m ml.eda.run_eda [--sample-per-source 300]

Charts and summary -> docs/figures/eda/ (repo). Sample X-ray grids -> DATA_ROOT/processed/eda/ (local only).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from ml.datasets.common import to_8bit_gray
from ml.datasets.manifest import finalize
from ml.datasets.targets import DISPLAY_NAMES, NIH_FINDINGS, TB, label_column

SOURCES = ["tbx11k", "shenzhen", "montgomery", "nih"]
SPLITS = ["train", "val", "test", "external_test"]
NIH_COLS = [label_column(f) for f in NIH_FINDINGS]
NIH_NAMES = [DISPLAY_NAMES[f] for f in NIH_FINDINGS]
TB_COL = label_column(TB)
REPO_ROOT = Path(__file__).resolve().parents[2]
SHOW = False  # set True in notebooks to display figures inline


def load_manifests(mdir: Path) -> dict[str, pd.DataFrame]:
    out = {}
    for name in SOURCES:
        path = mdir / f"{name}_manifest.csv"
        if path.exists():
            raw = pd.read_csv(path, dtype=str, keep_default_na=False).replace("", pd.NA)
            out[name] = finalize(raw.to_dict("records"))
    return out


def save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    if SHOW:
        plt.show()
    plt.close(fig)


def plot_split_sizes(dfs: dict, out: Path) -> dict:
    table = pd.DataFrame({n: d["split"].value_counts() for n, d in dfs.items()}).reindex(SPLITS).fillna(0).astype(int)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    table.T.plot(kind="bar", stacked=True, ax=ax, logy=True)
    ax.set_ylabel("Images (log scale)")
    ax.set_title("Images per dataset and split")
    ax.tick_params(axis="x", rotation=0)
    save(fig, out / "01_split_sizes.png")
    return table.to_dict()


def plot_tb_balance(dfs: dict, out: Path) -> dict:
    rows = []
    for name in ("tbx11k", "shenzhen", "montgomery"):
        if name in dfs:
            for split, part in dfs[name].groupby("split"):
                rows.append({"set": f"{name}\n{split}", "TB": int((part[TB_COL] == 1).sum()),
                             "Not TB": int((part[TB_COL] == 0).sum())})
    if not rows:
        return {}
    table = pd.DataFrame(rows).set_index("set")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    table[["Not TB", "TB"]].plot(kind="bar", stacked=True, ax=ax, color=["#9aa5b1", "#c0392b"])
    ax.set_ylabel("Images")
    ax.set_title("Tuberculosis label balance")
    ax.tick_params(axis="x", rotation=0)
    save(fig, out / "02_tb_balance.png")
    return {k.replace("\n", "/"): v for k, v in table.to_dict("index").items()}


def plot_nih_prevalence(nih: pd.DataFrame, out: Path) -> dict:
    prev = nih.groupby("split")[NIH_COLS].mean().reindex(["train", "val", "test"]) * 100
    prev.columns = NIH_NAMES
    fig, ax = plt.subplots(figsize=(9, 6))
    prev.T.sort_values("train").plot(kind="barh", ax=ax)
    ax.set_xlabel("% of images positive")
    ax.set_title("NIH finding prevalence by split")
    save(fig, out / "03_nih_prevalence_by_split.png")
    ratio = (prev.loc["test"] / prev.loc["train"]).round(2)
    return {"percent": prev.round(2).to_dict("index"), "test_to_train_ratio": ratio.to_dict()}


def plot_nih_cooccurrence(train: pd.DataFrame, out: Path) -> list[dict]:
    x = train[NIH_COLS].astype(int).to_numpy()
    co = x.T @ x
    counts = np.diag(co).astype(float)
    cond = np.divide(co, counts[:, None], out=np.zeros_like(co, dtype=float), where=counts[:, None] > 0)
    np.fill_diagonal(cond, np.nan)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cond * 100, cmap="viridis")
    ax.set_xticks(range(len(NIH_NAMES)), NIH_NAMES, rotation=60, ha="right")
    ax.set_yticks(range(len(NIH_NAMES)), NIH_NAMES)
    ax.set_xlabel("...also has this finding")
    ax.set_ylabel("Image with this finding...")
    ax.set_title("NIH co-occurrence (train)\n% of row-finding images that also have the column finding")
    fig.colorbar(im, ax=ax, label="%")
    save(fig, out / "04_nih_cooccurrence.png")
    pairs = [{"finding": NIH_NAMES[i], "also": NIH_NAMES[j], "percent": round(cond[i, j] * 100, 1)}
             for i in range(len(NIH_NAMES)) for j in range(len(NIH_NAMES)) if i != j]
    return sorted(pairs, key=lambda p: -p["percent"])[:10]


def plot_labels_per_image(train: pd.DataFrame, out: Path) -> dict:
    counts = train[NIH_COLS].sum(axis=1).astype(int).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(counts.index.astype(str), counts.values)
    ax.set_xlabel("Findings per image (0 = No Finding)")
    ax.set_ylabel("Images")
    ax.set_title("NIH findings per image (train)")
    save(fig, out / "05_nih_findings_per_image.png")
    return {int(k): int(v) for k, v in counts.items()}


def plot_view_shortcut(train: pd.DataFrame, out: Path) -> dict:
    overall = float((train["view"] == "AP").mean() * 100)
    share = {DISPLAY_NAMES[f]: float((train.loc[train[label_column(f)] == 1, "view"] == "AP").mean() * 100)
             for f in NIH_FINDINGS}
    s = pd.Series(share).sort_values()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(s.index, s.values)
    ax.axvline(overall, color="#c0392b", linestyle="--", label=f"All train images: {overall:.0f}% AP")
    ax.set_xlabel("% AP view among positive images")
    ax.set_title("AP-view share per finding (shortcut risk if far above the line)")
    ax.legend()
    save(fig, out / "06_nih_ap_view_by_finding.png")
    return {"overall_ap_percent": round(overall, 1), "ap_percent_among_positives": {k: round(v, 1) for k, v in share.items()}}


def plot_demographics(dfs: dict, out: Path) -> dict:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    summary = {}
    for name, df in dfs.items():
        ages = df["age"].dropna()
        if len(ages):
            a1.hist(ages, bins=range(0, 101, 5), density=True, histtype="step", linewidth=1.5, label=name)
        sexes = df["sex"].replace("", "unknown").value_counts()
        summary[name] = {"age_median": None if ages.empty else float(ages.median()),
                         "age_known": int(len(ages)), "sex": sexes.to_dict()}
    a1.set_xlabel("Age (years)")
    a1.set_ylabel("Density")
    a1.set_title("Age distribution")
    a1.legend()
    sex = pd.DataFrame({n: s["sex"] for n, s in summary.items()}).fillna(0)
    (sex / sex.sum() * 100).T.plot(kind="bar", stacked=True, ax=a2)
    a2.set_ylabel("% of images")
    a2.set_title("Sex")
    a2.tick_params(axis="x", rotation=0)
    save(fig, out / "07_demographics.png")
    return summary


def plot_followups(nih: pd.DataFrame, out: Path) -> dict:
    per_patient = nih.groupby("patient_id").size()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(per_patient.clip(upper=50), bins=50, log=True)
    ax.set_xlabel("Images per patient (capped at 50)")
    ax.set_ylabel("Patients (log scale)")
    ax.set_title("NIH images per patient")
    save(fig, out / "08_nih_images_per_patient.png")
    return {"patients": int(per_patient.size), "median": float(per_patient.median()),
            "max": int(per_patient.max()), "patients_with_10plus": int((per_patient >= 10).sum())}


def sample_image_stats(dfs: dict, root: Path, n: int, seed: int) -> pd.DataFrame:
    rows = []
    for name, df in dfs.items():
        pick = df.sample(min(n, len(df)), random_state=seed)
        for r in tqdm(pick.itertuples(index=False), total=len(pick), desc=f"sample {name}"):
            try:
                with Image.open(root / r.raw_path) as im:
                    w, h, mode = im.width, im.height, im.mode
                    g = to_8bit_gray(im)
                    g.thumbnail((256, 256))
                    px = np.asarray(g, dtype=np.uint8)
            except OSError:
                continue
            hist = np.bincount(px.ravel(), minlength=256) / px.size
            rows.append({"source": name, "width": w, "height": h, "mode": mode,
                         "mean": float(px.mean()), "std": float(px.std()), "hist": hist})
    return pd.DataFrame(rows)


def plot_image_stats(stats: pd.DataFrame, out: Path) -> dict:
    if stats.empty:
        return {}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for name, part in stats.groupby("source"):
        axes[0].plot(np.mean(np.stack(part["hist"]), axis=0), label=name)
        axes[1].scatter(part["mean"], part["std"], s=6, alpha=0.5, label=name)
    axes[0].set_title("Mean pixel-intensity histogram (8-bit)")
    axes[0].set_xlabel("Pixel value")
    axes[1].set_title("Brightness vs contrast per image")
    axes[1].set_xlabel("Mean")
    axes[1].set_ylabel("Std")
    for ax in axes[:2]:
        ax.legend()
    sizes = stats.assign(long_side=stats[["width", "height"]].max(axis=1))
    sizes.boxplot(column="long_side", by="source", ax=axes[2])
    axes[2].set_title("Original long side (px)")
    axes[2].set_xlabel("")
    fig.suptitle("")
    save(fig, out / "09_image_properties.png")
    return {name: {"n": int(len(p)), "modes": p["mode"].value_counts().to_dict(),
                   "width_range": [int(p["width"].min()), int(p["width"].max())],
                   "height_range": [int(p["height"].min()), int(p["height"].max())],
                   "mean_intensity": round(float(p["mean"].mean()), 1),
                   "std_intensity": round(float(p["std"].mean()), 1)}
            for name, p in stats.groupby("source")}


def sample_grids(dfs: dict, root: Path, local: Path, seed: int) -> None:
    for name, df in dfs.items():
        if name == "nih":
            pick = df.sample(min(16, len(df)), random_state=seed)
            titles = [str(t).replace("|", ", ")[:28] for t in pick["original_label"]]
        else:
            pos = df[df[TB_COL] == 1].sample(min(8, int((df[TB_COL] == 1).sum())), random_state=seed)
            neg = df[df[TB_COL] == 0].sample(min(8, int((df[TB_COL] == 0).sum())), random_state=seed)
            pick = pd.concat([pos, neg])
            titles = ["TB" if v == 1 else str(o) for v, o in zip(pick[TB_COL], pick["original_label"])]
        fig, axes = plt.subplots(4, 4, figsize=(10, 10))
        for ax in axes.ravel():
            ax.axis("off")
        for ax, path, title in zip(axes.ravel(), pick["raw_path"], titles):
            try:
                with Image.open(root / path) as im:
                    g = to_8bit_gray(im)
                    g.thumbnail((256, 256))
                ax.imshow(g, cmap="gray")
                ax.set_title(title, fontsize=8)
            except OSError:
                ax.set_title("unreadable", fontsize=8)
        fig.suptitle(f"{name}: random samples")
        save(fig, local / f"samples_{name}.png")


def run_eda(root: Path, out: Path, local: Path, sample_per_source: int = 300, seed: int = 42) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    local.mkdir(parents=True, exist_ok=True)
    dfs = load_manifests(root / "processed" / "manifests")
    if not dfs:
        raise FileNotFoundError("No manifests found; run python -m ml.datasets.build_manifests first")

    summary = {"split_sizes": plot_split_sizes(dfs, out), "tb_balance": plot_tb_balance(dfs, out),
               "demographics": plot_demographics(dfs, out)}
    if "nih" in dfs:
        nih = dfs["nih"]
        train = nih[nih["split"] == "train"]
        summary["nih_prevalence"] = plot_nih_prevalence(nih, out)
        summary["nih_top_cooccurrence"] = plot_nih_cooccurrence(train, out)
        summary["nih_findings_per_image"] = plot_labels_per_image(train, out)
        summary["nih_view"] = plot_view_shortcut(train, out)
        summary["nih_patients"] = plot_followups(nih, out)

    stats = sample_image_stats(dfs, root, sample_per_source, seed)
    summary["image_properties"] = plot_image_stats(stats, out)
    sample_grids(dfs, root, local, seed)

    (out / "eda_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    from ml.config import data_root

    parser = argparse.ArgumentParser(description="Run EDA on the dataset manifests.")
    parser.add_argument("--sample-per-source", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "docs" / "figures" / "eda")
    args = parser.parse_args()

    root = data_root()
    summary = run_eda(root, args.out_dir, root / "processed" / "eda", args.sample_per_source, args.seed)
    print(json.dumps({k: summary[k] for k in ("nih_view", "image_properties") if k in summary}, indent=2))
    print(f"\nCharts: {args.out_dir}\nSample grids (local only): {root / 'processed' / 'eda'}")


if __name__ == "__main__":
    main()
