"""Image preprocessing: 8-bit gray, trim flat borders, pad to square, resize, per-image stats.

Usage: python -m ml.preprocessing.preprocess [--datasets ...] [--workers 4] [--overwrite]
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from ml.datasets.common import to_8bit_gray
from ml.datasets.manifest import finalize

VERSION = "v2"
SOURCES = ["tbx11k", "shenzhen", "montgomery", "nih"]


@dataclass(frozen=True)
class Config:
    size: int = 320
    flat_std: float = 4.0       # rows/cols with std <= this are "flat"
    min_content: float = 0.5    # never trim away more than half the image area
    min_trim: float = 0.02      # ignore trims thinner than this fraction of the side
    intermediate: int = 512     # downsample larger images to this first (resolution harmonization)
    pad_value: int = 0


CONFIG = Config()
TRANSFORM_COLS = ["orig_w", "orig_h", "crop_x0", "crop_y0", "crop_x1", "crop_y1",
                  "pad_left", "pad_top", "square_side", "scale"]
STAT_COLS = ["px_mean", "px_std", "px_p1", "px_p99", "px_sharpness", "px_black_frac", "px_white_frac"]


def load_gray(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(to_8bit_gray(im), dtype=np.uint8)


def flat_trim_box(arr: np.ndarray, cfg: Config = CONFIG) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1) of the non-flat content; the full image if trimming is unsafe."""
    h, w = arr.shape
    rows = np.where(arr.std(axis=1) > cfg.flat_std)[0]
    cols = np.where(arr.std(axis=0) > cfg.flat_std)[0]
    if not len(rows) or not len(cols):
        return 0, 0, w, h
    x0, x1, y0, y1 = int(cols[0]), int(cols[-1]) + 1, int(rows[0]), int(rows[-1]) + 1
    if (x1 - x0) * (y1 - y0) < cfg.min_content * w * h:
        return 0, 0, w, h
    min_x, min_y = cfg.min_trim * w, cfg.min_trim * h
    x0 = x0 if x0 >= min_x else 0
    y0 = y0 if y0 >= min_y else 0
    x1 = x1 if w - x1 >= min_x else w
    y1 = y1 if h - y1 >= min_y else h
    return x0, y0, x1, y1


def preprocess_array(arr: np.ndarray, cfg: Config = CONFIG) -> tuple[np.ndarray, dict]:
    h, w = arr.shape
    x0, y0, x1, y1 = flat_trim_box(arr, cfg)
    crop = arr[y0:y1, x0:x1]
    ch, cw = crop.shape
    side = max(ch, cw)
    pad_top, pad_left = (side - ch) // 2, (side - cw) // 2
    square = np.full((side, side), cfg.pad_value, dtype=np.uint8)
    square[pad_top:pad_top + ch, pad_left:pad_left + cw] = crop
    img = Image.fromarray(square)
    if side > cfg.intermediate > cfg.size:
        img = img.resize((cfg.intermediate, cfg.intermediate), Image.LANCZOS, reducing_gap=3.0)
    out = np.asarray(img.resize((cfg.size, cfg.size), Image.LANCZOS, reducing_gap=3.0))
    transform = {"orig_w": w, "orig_h": h, "crop_x0": x0, "crop_y0": y0, "crop_x1": x1, "crop_y1": y1,
                 "pad_left": pad_left, "pad_top": pad_top, "square_side": side, "scale": cfg.size / side}
    return out, transform


def image_stats(px: np.ndarray) -> dict:
    p1, p99 = np.percentile(px, (1, 99))
    return {"px_mean": float(px.mean()), "px_std": float(px.std()), "px_p1": float(p1), "px_p99": float(p99),
            "px_sharpness": float(cv2.Laplacian(px, cv2.CV_64F).var()),
            "px_black_frac": float((px <= 5).mean()), "px_white_frac": float((px >= 250).mean())}


def map_box(x: float, y: float, w: float, h: float, t: dict) -> tuple[float, float, float, float]:
    """Original-image box -> processed-image box."""
    s = t["scale"]
    return ((x - t["crop_x0"] + t["pad_left"]) * s, (y - t["crop_y0"] + t["pad_top"]) * s, w * s, h * s)


def process_one(job: tuple[str, str, str, dict]) -> dict:
    image_id, src, dst, cfg_dict = job
    cfg = Config(**cfg_dict)
    try:
        out, transform = preprocess_array(load_gray(Path(src)), cfg)
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(out).save(dst, optimize=False)
        return {"image_id": image_id, "proc_status": "ok", "proc_reason": "", **transform, **image_stats(out)}
    except (OSError, ValueError) as exc:
        return {"image_id": image_id, "proc_status": "failed", "proc_reason": f"{type(exc).__name__}: {exc}"[:200]}


def processed_rel_path(row, cfg: Config = CONFIG) -> str:
    raw = Path(row.raw_path)
    name = raw.name if row.source in ("nih", "shenzhen", "montgomery") else f"{raw.parent.name}/{raw.name}"
    return f"processed/images_{cfg.size}/{row.source}/{name}"


def load_manifest(root: Path, source: str) -> pd.DataFrame:
    raw = pd.read_csv(root / "processed" / "manifests" / f"{source}_manifest.csv", dtype=str,
                      keep_default_na=False).replace("", pd.NA)
    return finalize(raw.to_dict("records"))


def run_source(root: Path, source: str, workers: int = 4, overwrite: bool = False, limit: int | None = None,
               cfg: Config = CONFIG, chunk: int = 2000) -> pd.DataFrame:
    """Process one dataset; resumable through a partial results file."""
    mdir = root / "processed" / "manifests"
    final = mdir / f"{source}_processed.csv"
    partial = mdir / f"{source}_processed.{VERSION}.partial.csv"
    if final.exists() and not overwrite and limit is None:
        existing = pd.read_csv(final, dtype={"patient_id": str})
        if (existing.get("preprocess_version") == VERSION).all():
            return existing
    if overwrite and partial.exists():
        partial.unlink()

    df = load_manifest(root, source)
    if limit is not None:
        df = df.groupby("split", group_keys=False).apply(lambda p: p.head(max(1, limit // df["split"].nunique())))
    df = df.assign(processed_path=[processed_rel_path(r, cfg) for r in df.itertuples()])

    done = pd.read_csv(partial) if partial.exists() else pd.DataFrame(columns=["image_id"])
    todo = df[~df["image_id"].isin(set(done["image_id"]))]
    jobs = [(r.image_id, str(root / r.raw_path), str(root / r.processed_path), asdict(cfg)) for r in todo.itertuples()]

    bar = tqdm(total=len(df), initial=len(df) - len(jobs), desc=f"preprocess {source}")
    pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        for start in range(0, len(jobs), chunk):
            batch = jobs[start:start + chunk]
            results = list(pool.map(process_one, batch, chunksize=32)) if pool else [process_one(j) for j in batch]
            pd.DataFrame(results).to_csv(partial, mode="a", header=not partial.exists(), index=False)
            bar.update(len(batch))
    finally:
        bar.close()
        if pool:
            pool.shutdown()

    results = pd.read_csv(partial).drop_duplicates("image_id", keep="last")
    out = df.merge(results, on="image_id", how="left")
    out["proc_status"] = out["proc_status"].fillna("missing")
    out["preprocess_version"] = VERSION
    out["preprocess_size"] = cfg.size
    if limit is None:
        out.to_csv(final, index=False)
        partial.unlink(missing_ok=True)
    return out


def transform_bboxes(root: Path, nih_processed: pd.DataFrame, cfg: Config = CONFIG) -> pd.DataFrame:
    src = root / "processed" / "manifests" / "nih_bboxes.csv"
    if not src.exists():
        return pd.DataFrame()
    boxes = pd.read_csv(src)
    t = nih_processed.set_index("image_id")
    rows = []
    for b in boxes.itertuples(index=False):
        if b.image_id not in t.index or t.at[b.image_id, "proc_status"] != "ok":
            continue
        x, y, w, h = map_box(b.x, b.y, b.w, b.h, t.loc[b.image_id, TRANSFORM_COLS].to_dict())
        rows.append({**b._asdict(), f"x_{cfg.size}": x, f"y_{cfg.size}": y, f"w_{cfg.size}": w, f"h_{cfg.size}": h})
    out = pd.DataFrame(rows)
    out.to_csv(root / "processed" / "manifests" / f"nih_bboxes_{cfg.size}.csv", index=False)
    return out


def summarize(processed: dict[str, pd.DataFrame]) -> dict:
    out = {"version": VERSION, "config": asdict(CONFIG), "datasets": {}}
    for name, df in processed.items():
        ok = df[df["proc_status"] == "ok"]
        out["datasets"][name] = {
            "rows": int(len(df)), "ok": int(len(ok)),
            "failed": df.loc[df["proc_status"] != "ok", ["image_id", "proc_status", "proc_reason"]].to_dict("records"),
            "trimmed": int(((ok["crop_x1"] - ok["crop_x0"]) * (ok["crop_y1"] - ok["crop_y0"])
                            < ok["orig_w"] * ok["orig_h"]).sum()),
            "padded": int(((ok["pad_left"] > 0) | (ok["pad_top"] > 0)).sum()),
            "stats_median": ok[STAT_COLS].median().round(3).to_dict(),
        }
    return out


def main() -> None:
    from ml.config import data_root

    parser = argparse.ArgumentParser(description="Preprocess all images to 320 px.")
    parser.add_argument("--datasets", nargs="+", choices=SOURCES, default=SOURCES)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = data_root()
    processed = {s: run_source(root, s, args.workers, args.overwrite) for s in args.datasets}
    if "nih" in processed:
        transform_bboxes(root, processed["nih"])
    print(json.dumps(summarize(processed), indent=2, default=str))


if __name__ == "__main__":
    main()
