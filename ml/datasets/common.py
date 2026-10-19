"""Helpers shared by all dataset builders."""
from __future__ import annotations

import hashlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

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


def _close_pairs(sub: np.ndarray, max_distance: int, chunk: int = 128):
    for start in range(0, len(sub), chunk):
        block = sub[start:start + chunk]
        dist = _POPCOUNT[block[:, None, :] ^ sub[None, :, :]].sum(axis=2, dtype=np.int32)
        ii, jj = np.nonzero(dist <= max_distance)
        keep = jj > ii + start
        for a, b in zip(ii[keep], jj[keep]):
            yield int(a + start), int(b), int(dist[a, b])


def lsh_pairs(hashes: list[str], max_distance: int = 6, bands: int = 8, max_bucket: int = 60_000) -> tuple[list, int]:
    """Near-duplicate pairs via banding. Exhaustive for max_distance < bands."""
    if not hashes:
        return [], 0
    packed = np.stack([np.frombuffer(bytes.fromhex(h), dtype=np.uint8) for h in hashes])
    width = packed.shape[1] // bands
    found, skipped = {}, 0
    for b in range(bands):
        buckets: dict[bytes, list[int]] = {}
        for i, k in enumerate(packed[:, b * width:(b + 1) * width]):
            buckets.setdefault(k.tobytes(), []).append(i)
        for members in buckets.values():
            if len(members) < 2:
                continue
            if len(members) > max_bucket:
                skipped += 1
                continue
            m = np.array(members)
            for a, c, d in _close_pairs(packed[m], max_distance):
                found[(int(m[a]), int(m[c]))] = d
    return [(i, j, d) for (i, j), d in found.items()], skipped


def hash_image_file(path: str) -> tuple[str, str]:
    """(sha256, 256-bit perceptual hash) of an image file, or empty strings if unreadable."""
    try:
        raw = Path(path).read_bytes()
        with Image.open(path) as im:
            return hashlib.sha256(raw).hexdigest(), str(imagehash.phash(to_8bit_gray(im), hash_size=16))
    except (OSError, ValueError):
        return "", ""


def hash_files(paths: list[str], workers: int = 4, desc: str = "hashing") -> list[tuple[str, str]]:
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            return list(tqdm(ex.map(hash_image_file, paths, chunksize=64), total=len(paths), desc=desc))
    return [hash_image_file(p) for p in tqdm(paths, desc=desc)]


def resolve_duplicates(df: pd.DataFrame, drop_distance: int, report_distance: int,
                       pairs: list[tuple[int, int, int]] | None = None
                       ) -> tuple[pd.DataFrame, list[dict], pd.DataFrame]:
    """Keep one copy of every duplicate (exact or phash <= drop_distance).

    Across splits the more important split wins; within a split the first row wins.
    Returns (kept, removals, report).
    """
    df = df.reset_index(drop=True)
    if pairs is None:
        hashed = df.index[df["phash"] != ""].tolist()
        pairs = [(hashed[a], hashed[b], d) for a, b, d in
                 near_duplicate_pairs(df.loc[hashed, "phash"].tolist(), report_distance)]

    report, to_drop = [], set()
    for i, j, dist in sorted(pairs):
        a, b = df.loc[i], df.loc[j]
        exact = a["sha256"] == b["sha256"] and a["sha256"] != ""
        crosses = a["split"] != b["split"]
        action = "report_only"
        if (exact or dist <= drop_distance) and i not in to_drop and j not in to_drop:
            if crosses:
                loser = i if SPLIT_RANK[a["split"]] > SPLIT_RANK[b["split"]] else j
            else:
                loser = j
            to_drop.add(loser)
            action = f"removed {df.loc[loser, 'image_id']} from {df.loc[loser, 'split']}"
        report.append({"image_a": a["image_id"], "split_a": a["split"],
                       "image_b": b["image_id"], "split_b": b["split"],
                       "phash_distance": dist, "exact_match": exact, "action": action})

    removals = [{"image_id": df.loc[k, "image_id"], "source": df.loc[k, "source"],
                 "reason": "duplicate_cross_split" if any(
                     r["action"].startswith(f"removed {df.loc[k, 'image_id']}") and r["split_a"] != r["split_b"]
                     for r in report) else "duplicate_same_split"} for k in sorted(to_drop)]
    kept = df.drop(index=sorted(to_drop)).reset_index(drop=True)
    report_df = pd.DataFrame(report, columns=["image_a", "split_a", "image_b", "split_b",
                                              "phash_distance", "exact_match", "action"])
    return kept, removals, report_df
