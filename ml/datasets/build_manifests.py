"""Build all dataset manifests.

Usage: python -m ml.datasets.build_manifests [--datasets tbx11k shenzhen montgomery nih]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ml.datasets.common import hash_files, lsh_pairs, resolve_duplicates
from ml.datasets.manifest import validate_manifest
from ml.datasets.nih import build_nih
from ml.datasets.nlm_tb import build_nlm_set
from ml.datasets.targets import ALL_FINDINGS, label_column
from ml.datasets.tbx11k import build_tbx11k

DATASETS = ["tbx11k", "shenzhen", "montgomery", "nih"]
TB_SOURCES = {"tbx11k", "shenzhen", "montgomery"}


def summarize(df: pd.DataFrame) -> dict:
    out = {"total": int(len(df)), "splits": {}}
    for split, part in df.groupby("split"):
        positives = {}
        for f in ALL_FINDINGS:
            col = part[label_column(f)]
            if col.notna().any():
                positives[f] = {"positive": int((col == 1).sum()), "known": int(col.notna().sum())}
        out["splits"][split] = {
            "images": int(len(part)),
            "patients": int(part.loc[part["patient_id"] != "", "patient_id"].nunique()),
            "positives": positives,
            "sex": part["sex"].replace("", "unknown").value_counts().to_dict(),
            "view": part["view"].replace("", "unknown").value_counts().to_dict(),
            "age_median": None if part["age"].isna().all() else float(part["age"].median()),
            "uncertain": int(part["uncertain"].sum()),
        }
    notes = df["notes"].str.split(";").explode()
    out["notes"] = notes[notes != ""].str.split(":").str[0].value_counts().to_dict()
    return out


def nih_hash_table(data_root: Path, nih: pd.DataFrame, compute: bool, workers: int) -> pd.DataFrame | None:
    cache = data_root / "processed" / "manifests" / "nih_hashes.csv"
    if cache.exists():
        cached = pd.read_csv(cache, dtype=str, keep_default_na=False)
        if set(nih["image_id"]) <= set(cached["image_id"]):
            return cached
    if not compute:
        return None
    results = hash_files([str(data_root / p) for p in nih["raw_path"]], workers, desc="hashing NIH")
    table = pd.DataFrame({"image_id": nih["image_id"].to_numpy(),
                          "sha256": [r[0] for r in results], "phash": [r[1] for r in results]})
    table.to_csv(cache, index=False)
    return table


def dedupe_nih(nih: pd.DataFrame, hashes: pd.DataFrame, drop_distance: int) -> tuple[pd.DataFrame, list[dict], pd.DataFrame]:
    nih = nih.drop(columns=["sha256", "phash"]).merge(hashes[["image_id", "sha256", "phash"]], on="image_id", how="left")
    nih[["sha256", "phash"]] = nih[["sha256", "phash"]].fillna("")
    hashed = nih.index[nih["phash"] != ""].tolist()
    found, _ = lsh_pairs(nih.loc[hashed, "phash"].tolist(), max_distance=min(drop_distance, 7))
    pairs = [(hashed[a], hashed[c], d) for a, c, d in found]
    return resolve_duplicates(nih, drop_distance, drop_distance, pairs=pairs)


def build_all(data_root: Path, datasets: list[str], seed: int = 42, tbx_val: float = 0.15,
              nih_val: float = 0.10, drop_distance: int = 6, report_distance: int = 12,
              hash_nih: bool = False, workers: int = 4) -> dict:
    out_dir = data_root / "processed" / "manifests"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames, skipped, summary = {}, [], {"seed": seed, "drop_distance": drop_distance,
                                        "report_distance": report_distance, "datasets": {}}

    if "tbx11k" in datasets:
        frames["tbx11k"], s = build_tbx11k(data_root, tbx_val, seed)
        skipped += s
    for name in ("shenzhen", "montgomery"):
        if name in datasets:
            frames[name], s = build_nlm_set(data_root, name)
            skipped += s

    # Keep the external test set free of duplicates from training data.
    tb_names = [n for n in frames if n in TB_SOURCES]
    if tb_names:
        combined = pd.concat([frames[n] for n in tb_names], ignore_index=True)
        kept, removals, dup_report = resolve_duplicates(combined, drop_distance, report_distance)
        skipped += removals
        dup_report.to_csv(out_dir / "tb_duplicates.csv", index=False)
        summary["tb_duplicate_pairs_reported"] = int(len(dup_report))
        summary["tb_duplicates_removed"] = len(removals)
        for n in tb_names:
            frames[n] = kept[kept["source"] == n].reset_index(drop=True)

    if "nih" in datasets:
        frames["nih"], s, bboxes = build_nih(data_root, nih_val, seed)
        skipped += s
        hashes = nih_hash_table(data_root, frames["nih"], hash_nih, workers)
        if hashes is None:
            summary["nih_duplicates"] = "skipped: no nih_hashes.csv (run EDA section 23 or use --hash-nih)"
        else:
            frames["nih"], removals, nih_dups = dedupe_nih(frames["nih"], hashes, drop_distance)
            skipped += removals
            nih_dups.to_csv(out_dir / "nih_duplicates.csv", index=False)
            summary["nih_duplicates"] = {"pairs": int(len(nih_dups)), "removed": len(removals),
                                         "removed_cross_split": sum(r["reason"] == "duplicate_cross_split" for r in removals)}
        bboxes = bboxes[bboxes["image_id"].isin(frames["nih"]["image_id"])].reset_index(drop=True)
        bboxes.to_csv(out_dir / "nih_bboxes.csv", index=False)
        summary["nih_bboxes"] = {"boxes": int(len(bboxes)), "by_split": bboxes["split"].value_counts().to_dict(),
                                 "label_missing_for_box": int((~bboxes["label_present"]).sum())}

    for name, df in frames.items():
        validate_manifest(df)
        df.to_csv(out_dir / f"{name}_manifest.csv", index=False)
        summary["datasets"][name] = summarize(df)

    skipped_df = pd.DataFrame(skipped, columns=["image_id", "source", "reason"])
    skipped_df.to_csv(out_dir / "skipped.csv", index=False)
    summary["skipped"] = skipped_df.groupby(["source", "reason"]).size().rename("n").reset_index().to_dict("records")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def main() -> None:
    from ml.config import data_root

    parser = argparse.ArgumentParser(description="Build PNEUMYST AI dataset manifests.")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tbx-val", type=float, default=0.15, help="Fraction of TBX11K official train used as val")
    parser.add_argument("--nih-val", type=float, default=0.10, help="Fraction of NIH train_val PATIENTS used as val")
    parser.add_argument("--drop-distance", type=int, default=6,
                        help="Max perceptual-hash distance (of 256 bits) treated as the same image across splits")
    parser.add_argument("--report-distance", type=int, default=12,
                        help="Max distance listed in tb_duplicates.csv for manual review")
    parser.add_argument("--hash-nih", action="store_true", help="Hash all NIH images if nih_hashes.csv is missing")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    summary = build_all(data_root(), args.datasets, args.seed, args.tbx_val, args.nih_val,
                        args.drop_distance, args.report_distance, args.hash_nih, args.workers)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
