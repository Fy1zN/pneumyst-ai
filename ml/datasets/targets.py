"""Findings, display groups and label columns."""
from __future__ import annotations

TB = "Tuberculosis"

NIH_FINDINGS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Hernia",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pleural_Thickening",
    "Pneumonia",
    "Pneumothorax",
]
NIH_NO_FINDING = "No Finding"

ALL_FINDINGS = [TB] + NIH_FINDINGS

DISPLAY_NAMES = {f: f.replace("_", " ").capitalize() for f in ALL_FINDINGS}
DISPLAY_NAMES[TB] = "Tuberculosis"

GROUPS = {
    "Infection": [TB, "Pneumonia", "Consolidation", "Infiltration"],
    "Pleura": ["Effusion", "Pneumothorax", "Pleural_Thickening"],
    "Lung tissue": ["Atelectasis", "Edema", "Emphysema", "Fibrosis", "Nodule", "Mass"],
    "Heart and other": ["Cardiomegaly", "Hernia"],
}


def label_column(finding: str) -> str:
    """Manifest column holding 0 / 1 / empty (unknown) for a finding."""
    return "label_" + finding.lower()


LABEL_COLUMNS = [label_column(f) for f in ALL_FINDINGS]

assert sorted(sum(GROUPS.values(), [])) == sorted(ALL_FINDINGS), "every finding must be in exactly one group"
