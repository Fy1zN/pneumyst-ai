"""Shenzhen and Montgomery manifest builder (external test only)."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from ml.datasets.common import inspect_image
from ml.datasets.manifest import empty_labels, finalize
from ml.datasets.targets import TB, label_column

SETS = {"shenzhen": "shenzhen", "montgomery": "montgomery"}
FILE_RE = re.compile(r"^(?P<stem>[A-Za-z]+_\d{4})_(?P<label>[01])\.png$")

_SEX_FIELD = re.compile(r"patient'?s\s+sex\s*:\s*([MF])", re.I)
_SEX_WORD = re.compile(r"\b(female|male)\b", re.I)
_AGE_FIELD = re.compile(r"patient'?s\s+age\s*:\s*0*(\d+)\s*([YMD])?", re.I)
_AGE_YEARS = re.compile(r"(\d+)\s*(?:yrs?|years?|yr|y)\b", re.I)
_AGE_MONTHS = re.compile(r"(\d+)\s*(?:months?|mons?|mths?|m)\b", re.I)
_AGE_DAYS = re.compile(r"(\d+)\s*(?:days?|d)\b", re.I)


def parse_reading(text: str) -> tuple[str, float | None, str]:
    """Return (sex, age_in_years, finding_text) from a clinical reading."""
    sex = ""
    m = _SEX_FIELD.search(text)
    if m:
        sex = m.group(1).upper()
    else:
        m = _SEX_WORD.search(text)
        if m:
            sex = "F" if m.group(1).lower() == "female" else "M"

    age = None
    m = _AGE_FIELD.search(text)
    if m:
        value, unit = int(m.group(1)), (m.group(2) or "Y").upper()
        age = value if unit == "Y" else value / 12 if unit == "M" else value / 365
    else:
        for pattern, divisor in ((_AGE_YEARS, 1), (_AGE_MONTHS, 12), (_AGE_DAYS, 365)):
            m = pattern.search(text)
            if m:
                age = int(m.group(1)) / divisor
                break

    finding_lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line or _SEX_FIELD.search(line) or _AGE_FIELD.search(line):
            continue
        if _SEX_WORD.search(line) and (_AGE_YEARS.search(line) or _AGE_MONTHS.search(line) or _AGE_DAYS.search(line)):
            continue
        finding_lines.append(line)
    return sex, (round(age, 2) if age is not None else None), " / ".join(finding_lines)


def build_nlm_set(data_root: Path, name: str) -> tuple[pd.DataFrame, list[dict]]:
    if name not in SETS:
        raise ValueError(f"Unknown NLM set: {name}")
    root = data_root / "raw" / SETS[name]
    png_dir, reading_dir = root / "CXR_png", root / "ClinicalReadings"
    if not png_dir.exists():
        raise FileNotFoundError(f"{png_dir} not found")

    rows, skipped = [], []
    for path in tqdm(sorted(png_dir.glob("*.png")), desc=name):
        image_id = f"{name}:{path.name}"
        m = FILE_RE.match(path.name)
        if not m:
            skipped.append({"image_id": image_id, "source": name, "reason": "unexpected_file_name"})
            continue
        info, reason = inspect_image(path)
        if info is None:
            skipped.append({"image_id": image_id, "source": name, "reason": reason})
            continue

        tb = int(m.group("label"))
        notes = []
        reading_path = reading_dir / f"{path.stem}.txt"
        if reading_path.exists():
            reading = reading_path.read_text(errors="replace")
            sex, age, finding = parse_reading(reading)
            says_normal = finding.strip().lower() == "normal"
            if tb == 1 and says_normal:
                notes.append("reading_says_normal_but_label_tb")
            if tb == 0 and finding and not says_normal:
                notes.append("label_normal_but_reading_not_normal")
        else:
            sex, age, finding = "", None, ""
            notes.append("no_reading")
        if age is not None and age < 18:
            notes.append("pediatric")
        if age is not None and not 0 <= age <= 110:
            notes.append(f"implausible_age:{age}")
            age = None

        labels = empty_labels()
        labels[label_column(TB)] = tb
        rows.append({
            "image_id": image_id, "source": name,
            "raw_path": path.relative_to(data_root).as_posix(),
            "age": age, "sex": sex,
            "original_label": "tb" if tb else "normal", "original_subtypes": finding,
            "uncertain": False, "annotation_type": "image_label",
            "label_source": "clinical_reading", "official_split": "",
            "split": "external_test", "notes": ";".join(notes),
            **info, **labels,
        })
    return finalize(rows), skipped
