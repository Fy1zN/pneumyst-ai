"""Helpers shared by all dataset builders."""
from __future__ import annotations

import hashlib
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

# Lower rank wins when a duplicate crosses splits.
SPLIT_RANK = {"external_test": 0, "test": 1, "val": 2, "train": 3}


def read_list(path: Path) -> list[str]:
    lines = [ln.strip().replace("\\", "/") for ln in path.read_text().splitlines()]
    return [ln for ln in lines if ln]


def to_8bit_gray(im: Image.Image) -> Image.Image:
    """Any PNG mode to 8-bit grayscale; 16-bit is min-max scaled, not clipped."""
    if im.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
        arr = np.asarray(im, dtype=np.float32)
        lo, hi = float(arr.min()), float(arr.max())
        if hi <= lo:
            return Image.fromarray(np.zeros(arr.shape, dtype=np.uint8))
        return Image.fromarray(((arr - lo) / (hi - lo) * 255.0).round().astype(np.uint8))
    return im.convert("L")


def inspect_image(path: Path, min_side: int = 64) -> tuple[dict | None, str | None]:
    """Return image facts, or (None, reason) if unusable."""
    if not path.exists():
        return None, "missing_file"
    try:
        raw = path.read_bytes()
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            im.load()
            width, height, mode = im.width, im.height, im.mode
            phash = str(imagehash.phash(to_8bit_gray(im), hash_size=16))
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        return None, f"corrupt_image: {type(exc).__name__}"
    if min(width, height) < min_side:
        return None, f"too_small: {width}x{height}"
    return {"width": width, "height": height, "mode": mode,
            "sha256": hashlib.sha256(raw).hexdigest(), "phash": phash}, None


def near_duplicate_pairs(hashes: list[str], max_distance: int, chunk: int = 128) -> list[tuple[int, int, int]]:
    """All index pairs (i < j) whose perceptual hashes differ by <= max_distance bits."""
    if not hashes:
        return []
    packed = np.stack([np.frombuffer(bytes.fromhex(h), dtype=np.uint8) for h in hashes])
    pairs: list[tuple[int, int, int]] = []
    for start in range(0, len(packed), chunk):
        block = packed[start:start + chunk]
        dist = _POPCOUNT[block[:, None, :] ^ packed[None, :, :]].sum(axis=2, dtype=np.int32)
        ii, jj = np.nonzero(dist <= max_distance)
        keep = jj > ii + start
        for i, j in zip(ii[keep], jj[keep]):
            pairs.append((int(i + start), int(j), int(dist[i, j])))
    return pairs


def resolve_duplicates(df: pd.DataFrame, drop_distance: int, report_distance: int
                       ) -> tuple[pd.DataFrame, list[dict], pd.DataFrame]:
    """Drop the lower-ranked copy of duplicates that cross splits. Returns (kept, removals, report)."""
    df = df.reset_index(drop=True)
    hashed = df.index[df["phash"] != ""].tolist()
    pairs = near_duplicate_pairs(df.loc[hashed, "phash"].tolist(), report_distance)

    report, to_drop = [], set()
    for a_pos, b_pos, dist in pairs:
        i, j = hashed[a_pos], hashed[b_pos]
        a, b = df.loc[i], df.loc[j]
        exact = a["sha256"] == b["sha256"] and a["sha256"] != ""
        crosses = a["split"] != b["split"]
        action = "report_only"
        if crosses and (exact or dist <= drop_distance):
            loser = i if SPLIT_RANK[a["split"]] > SPLIT_RANK[b["split"]] else j
            to_drop.add(loser)
            action = f"removed {df.loc[loser, 'image_id']} from {df.loc[loser, 'split']}"
        report.append({"image_a": a["image_id"], "split_a": a["split"],
                       "image_b": b["image_id"], "split_b": b["split"],
                       "phash_distance": dist, "exact_match": exact, "action": action})

    removals = [{"image_id": df.loc[k, "image_id"], "source": df.loc[k, "source"],
                 "reason": "duplicate_of_image_in_more_important_split"} for k in sorted(to_drop)]
    kept = df.drop(index=sorted(to_drop)).reset_index(drop=True)
    report_df = pd.DataFrame(report, columns=["image_a", "split_a", "image_b", "split_b",
                                              "phash_distance", "exact_match", "action"])
    return kept, removals, report_df
