"""Extra EDA checks that inform preprocessing and the report."""
from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import ml.eda.run_eda as eda
from ml.datasets.common import to_8bit_gray
from ml.datasets.targets import DISPLAY_NAMES, NIH_FINDINGS, TB, label_column

TB_COL = label_column(TB)
AGE_BINS = [0, 20, 40, 60, 80, 200]
AGE_LABELS = ["0-19", "20-39", "40-59", "60-79", "80+"]
FOLLOWUP_BINS = [0, 1, 2, 5, 10, 20, 10_000]
FOLLOWUP_LABELS = ["0", "1", "2-4", "5-9", "10-19", "20+"]


def _thumb(im: Image.Image, size: int = 256) -> np.ndarray:
    g = to_8bit_gray(im)
    g.thumbnail((size, size))
    return np.asarray(g, dtype=np.uint8)


def _grid(items: list[tuple[np.ndarray, str]], path: Path, cols: int = 4, title: str = "") -> None:
    if not items:
        return
    rows = int(np.ceil(len(items) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3.2 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (px, label) in zip(axes.ravel(), items):
        ax.imshow(px, cmap="gray", vmin=0, vmax=255)
        ax.set_title(label, fontsize=8)
    fig.suptitle(title)
    eda.save(fig, path)


def _sample_paths(dfs: dict, n: int, seed: int) -> list[tuple[str, str]]:
    out = []
    for name, df in dfs.items():
        out += [(name, p) for p in df.sample(min(n, len(df)), random_state=seed)["raw_path"]]
    return out


# 11. Image mode census + are color images really gray?
def mode_census(dfs: dict, root: Path) -> pd.DataFrame:
    rows = []
    for name, df in dfs.items():
        for p in tqdm(df["raw_path"], desc=f"modes {name}"):
            try:
                with Image.open(root / p) as im:
                    rows.append((name, im.mode, p))
            except OSError:
                rows.append((name, "unreadable", p))
    return pd.DataFrame(rows, columns=["source", "mode", "raw_path"])


def channel_check(census: pd.DataFrame, root: Path, out: Path, n: int = 200, seed: int = 42) -> pd.DataFrame:
    color = census[census["mode"].isin(["RGB", "RGBA", "P"])]
    rows = []
    for (src, mode), part in color.groupby(["source", "mode"]):
        for p in tqdm(part.sample(min(n, len(part)), random_state=seed)["raw_path"], desc=f"channels {src}/{mode}"):
            with Image.open(root / p) as im:
                alpha_min = int(np.asarray(im.getchannel("A")).min()) if im.mode == "RGBA" else 255
                rgb = im.convert("RGB")
                rgb.thumbnail((256, 256))
                a = np.asarray(rgb, dtype=np.int16)
            diff = int(max(np.abs(a[..., 0] - a[..., 1]).max(), np.abs(a[..., 1] - a[..., 2]).max()))
            rows.append({"source": src, "mode": mode, "raw_path": p, "max_channel_diff": diff, "alpha_min": alpha_min})
    res = pd.DataFrame(rows, columns=["source", "mode", "raw_path", "max_channel_diff", "alpha_min"])
    if len(res):
        res["is_gray"] = res["max_channel_diff"] <= 2
        share = res.groupby(["source", "mode"])["is_gray"].mean() * 100
        fig, ax = plt.subplots(figsize=(7, 3.5))
        share.plot(kind="barh", ax=ax)
        ax.set_xlim(0, 100)
        ax.set_xlabel("% of sampled images whose 3 channels are identical")
        ax.set_title("Are color-mode images really grayscale?")
        eda.save(fig, out / "11_color_mode_check.png")
    return res


# 12. Inverted images
def inversion_check(census: pd.DataFrame, root: Path, out: Path, local: Path, n: int = 200, seed: int = 42) -> pd.DataFrame:
    rows = []
    for (src, mode), part in census[census["mode"] != "unreadable"].groupby(["source", "mode"]):
        for p in tqdm(part.sample(min(n, len(part)), random_state=seed)["raw_path"], desc=f"inversion {src}/{mode}"):
            with Image.open(root / p) as im:
                palette_inverted = None
                if im.mode == "P" and im.getpalette():
                    lut = np.array(im.getpalette()).reshape(-1, 3)[:, 0]
                    palette_inverted = bool(lut[0] > lut[-1])
                px = _thumb(im)
            h, w = px.shape
            c = max(4, min(h, w) // 10)
            corners = np.concatenate([px[:c, :c].ravel(), px[:c, -c:].ravel(), px[-c:, :c].ravel(), px[-c:, -c:].ravel()])
            center = px[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
            rows.append({"source": src, "mode": mode, "raw_path": p, "palette_inverted": palette_inverted,
                         "corner_mean": float(corners.mean()), "center_mean": float(center.mean())})
    res = pd.DataFrame(rows)
    if res.empty:
        return res
    res["suspect_inverted"] = res["corner_mean"] > res["center_mean"] + 20
    fig, ax = plt.subplots(figsize=(6, 5))
    for src, part in res.groupby("source"):
        ax.scatter(part["center_mean"], part["corner_mean"], s=8, alpha=0.6, label=src)
    ax.plot([0, 255], [20, 275], "r--", linewidth=1, label="suspect above this line")
    ax.set_xlim(0, 255)
    ax.set_ylim(0, 255)
    ax.set_xlabel("Center brightness")
    ax.set_ylabel("Corner brightness")
    ax.set_title("Inversion check (inverted images have bright corners)")
    ax.legend(fontsize=8)
    eda.save(fig, out / "12_inversion_check.png")
    items = []
    for r in res[res["suspect_inverted"]].head(12).itertuples():
        with Image.open(root / r.raw_path) as im:
            items.append((_thumb(im), f"{r.source}: {Path(r.raw_path).name}"))
    _grid(items, local / "suspect_inverted.png", title="Suspected inverted images")
    return res


# 13. Borders and aspect ratio
def border_check(dfs: dict, root: Path, out: Path, n: int = 200, seed: int = 42) -> pd.DataFrame:
    rows = []
    for src, p in tqdm(_sample_paths(dfs, n, seed), desc="borders"):
        try:
            with Image.open(root / p) as im:
                w, h = im.size
                px = _thumb(im)
        except OSError:
            continue
        busy_r = np.where(px.std(axis=1) > 4)[0]
        busy_c = np.where(px.std(axis=0) > 4)[0]
        content = (np.ptp(busy_r) + 1) * (np.ptp(busy_c) + 1) if len(busy_r) and len(busy_c) else 0
        rows.append({"source": src, "aspect": w / h, "border_frac": 1 - content / px.size,
                     "black_frac": float((px <= 5).mean()), "white_frac": float((px >= 250).mean())})
    res = pd.DataFrame(rows)
    if res.empty:
        return res
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, col, title in zip(axes, ["border_frac", "black_frac", "white_frac", "aspect"],
                              ["Flat border share", "Near-black pixel share", "Near-white pixel share", "Aspect ratio (w / h)"]):
        res.boxplot(column=col, by="source", ax=ax)
        ax.set_title(title)
        ax.set_xlabel("")
    fig.suptitle("")
    eda.save(fig, out / "13_borders_and_aspect.png")
    return res


# 14. NIH bounding-box sizes vs training resolution
def bbox_sizes(root: Path, out: Path) -> pd.DataFrame:
    path = root / "processed" / "manifests" / "nih_bboxes.csv"
    if not path.exists():
        return pd.DataFrame()
    b = pd.read_csv(path)
    b["side_1024"] = np.sqrt(b["w"] * b["h"])
    b["side_224"] = b["side_1024"] * 224 / 1024
    b["side_320"] = b["side_1024"] * 320 / 1024
    order = b.groupby("finding")["side_224"].median().sort_values().index
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.boxplot([b.loc[b["finding"] == f, "side_224"] for f in order])
    ax.set_xticks(range(1, len(order) + 1), [DISPLAY_NAMES.get(f, f) for f in order])
    ax.axhline(10, color="#c0392b", linestyle="--", linewidth=1, label="10 px")
    ax.set_ylabel("Box side at 224 px input (px)")
    ax.set_title("How big lesions are at the training resolution (NIH boxes)")
    ax.tick_params(axis="x", rotation=30)
    ax.legend()
    eda.save(fig, out / "14_bbox_size_at_224.png")
    return (b.groupby("finding").agg(boxes=("x", "size"), median_side_1024=("side_1024", "median"),
                                     median_side_224=("side_224", "median"), median_side_320=("side_320", "median"),
                                     min_side_224=("side_224", "min"))
             .round(1).sort_values("median_side_224"))


# 15. Duplicate pairs for human review
def duplicate_review(dfs: dict, root: Path, local: Path, max_pairs: int = 10) -> pd.DataFrame:
    path = root / "processed" / "manifests" / "tb_duplicates.csv"
    if not path.exists():
        return pd.DataFrame()
    d = pd.read_csv(path)
    paths = pd.concat([df[["image_id", "raw_path"]] for df in dfs.values()]).set_index("image_id")["raw_path"].to_dict()
    raw_dirs = {"tbx11k": "raw/tbx11k/TBX11K/imgs", "shenzhen": "raw/shenzhen/CXR_png", "montgomery": "raw/montgomery/CXR_png"}
    removed = d["action"].str.startswith("removed")
    pick = d[(removed & (d["phash_distance"] > 0)) | (~removed & (d["split_a"] != d["split_b"]))]
    pick = pick.sort_values("phash_distance").head(max_pairs)
    items = []
    for r in pick.itertuples():
        for img_id, split in ((r.image_a, r.split_a), (r.image_b, r.split_b)):
            src, name = img_id.split(":", 1)
            label = f"{name} [{split}] d={r.phash_distance}"
            p = paths.get(img_id)
            if p is None:
                p, label = f"{raw_dirs.get(src, '')}/{name}", f"{label} (removed)"
            try:
                with Image.open(root / p) as im:
                    items.append((_thumb(im), label))
            except OSError:
                items.append((np.zeros((64, 64), np.uint8), f"{label} (not found)"))
    _grid(items, local / "duplicate_pairs.png", cols=2, title="Duplicate pairs (left/right = same pair)")
    return pick


# 16. Prevalence by age band, sex and view
def prevalence_by_group(train: pd.DataFrame, out: Path) -> dict:
    t = train.assign(age_band=pd.cut(train["age"], AGE_BINS, right=False, labels=AGE_LABELS))
    tables = {}
    fig, axes = plt.subplots(1, 3, figsize=(17, 6), gridspec_kw={"width_ratios": [5, 2, 2]})
    for ax, col in zip(axes, ["age_band", "sex", "view"]):
        tab = t.groupby(col, observed=True)[eda.NIH_COLS].mean().astype(float).T * 100
        tab.index = eda.NIH_NAMES
        tables[col] = tab.round(2)
        im = ax.imshow(tab.to_numpy(), aspect="auto", cmap="magma")
        ax.set_xticks(range(tab.shape[1]), tab.columns)
        ax.set_yticks(range(tab.shape[0]), tab.index if col == "age_band" else [""] * tab.shape[0])
        ax.set_title(f"% positive by {col.replace('_', ' ')}")
        for (i, j), v in np.ndenumerate(tab.to_numpy()):
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7, color="white" if v < tab.to_numpy().max() / 2 else "black")
        fig.colorbar(im, ax=ax, fraction=0.05)
    eda.save(fig, out / "16_prevalence_by_age_sex_view.png")
    return tables


# 17. Positivity vs follow-up number
def followup_effect(nih: pd.DataFrame, out: Path) -> pd.DataFrame:
    fu = nih["original_subtypes"].str.extract(r"follow_up=(\d+)")[0].astype(float)
    t = nih.assign(follow_up_bin=pd.cut(fu, FOLLOWUP_BINS, right=False, labels=FOLLOWUP_LABELS),
                   any_finding=(nih[eda.NIH_COLS].astype(float).sum(axis=1) > 0).astype(int))
    cols = {"any_finding": "Any finding"} | {label_column(f): DISPLAY_NAMES[f]
                                              for f in ("Effusion", "Pneumothorax", "Infiltration", "Atelectasis")}
    tab = t.groupby("follow_up_bin", observed=True)[list(cols)].mean().astype(float) * 100
    tab = tab.rename(columns=cols)
    tab["images"] = t.groupby("follow_up_bin", observed=True).size()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    tab.drop(columns="images").plot(marker="o", ax=ax)
    ax.set_xlabel("Follow-up number of the scan")
    ax.set_ylabel("% positive")
    ax.set_title("Later follow-up scans are more often abnormal")
    eda.save(fig, out / "17_positivity_vs_followup.png")
    return tab.round(1)


# 18. Sharpness and exposure
def quality_stats(dfs: dict, root: Path, out: Path, n: int = 300, seed: int = 42) -> tuple[pd.DataFrame, dict]:
    rows = []
    for src, p in tqdm(_sample_paths(dfs, n, seed), desc="quality"):
        try:
            with Image.open(root / p) as im:
                px = _thumb(im, 512)
        except OSError:
            continue
        rows.append({"source": src, "sharpness": float(cv2.Laplacian(px, cv2.CV_64F).var()),
                     "mean": float(px.mean()), "dark_frac": float((px <= 5).mean()), "bright_frac": float((px >= 250).mean())})
    res = pd.DataFrame(rows)
    if res.empty:
        return res, {}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for src, part in res.groupby("source"):
        axes[0].hist(np.log10(part["sharpness"] + 1), bins=40, histtype="step", linewidth=1.5, label=src)
        axes[1].hist(part["mean"], bins=40, histtype="step", linewidth=1.5, label=src)
    axes[0].set_xlabel("log10(Laplacian variance)  (higher = sharper)")
    axes[0].set_title("Sharpness")
    axes[1].set_xlabel("Mean brightness (8-bit)")
    axes[1].set_title("Exposure")
    for ax in axes:
        ax.legend(fontsize=8)
    eda.save(fig, out / "18_sharpness_exposure.png")
    train_like = res[res["source"].isin(["tbx11k", "nih"])]
    base = train_like if len(train_like) else res
    thresholds = {"min_sharpness_p1": round(float(base["sharpness"].quantile(0.01)), 1),
                  "mean_brightness_p1": round(float(base["mean"].quantile(0.01)), 1),
                  "mean_brightness_p99": round(float(base["mean"].quantile(0.99)), 1)}
    return res, thresholds


# 19. External readings that disagree with labels
def reading_mismatches(dfs: dict, root: Path, local: Path) -> pd.DataFrame:
    parts = [df[df["notes"].str.contains("reading_says_normal|reading_not_normal", regex=True)]
             for name, df in dfs.items() if name in ("shenzhen", "montgomery")]
    if not parts:
        return pd.DataFrame()
    m = pd.concat(parts)[["image_id", TB_COL, "original_subtypes", "age", "sex", "notes", "raw_path"]]
    items = []
    for r in m.itertuples():
        with Image.open(root / r.raw_path) as im:
            items.append((_thumb(im), f"{r.image_id.split(':', 1)[1]}\nlabel TB={r[2]} | {str(r.original_subtypes)[:30]}"))
    _grid(items, local / "reading_mismatches.png", title="External images whose reading disagrees with the label")
    return m.drop(columns="raw_path").rename(columns={TB_COL: "label_tb", "original_subtypes": "reading"})
