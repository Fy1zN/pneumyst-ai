"""Rule-based checks on an LLM-written summary. Each returned issue is fed back to the model on a retry."""
from __future__ import annotations

import re

MIN_WORDS, MAX_WORDS = 25, 180
BANNED = {
    r"\byou have\b": "speaks as if the person has a condition",
    r"\bdiagnos\w*": "uses diagnosis language",
    r"\bconfirm\w*": "claims a finding is confirmed",
    r"\brul\w* out\b": "claims a condition is ruled out",
    r"\b(definitely|certainly|certain)\b": "claims certainty",
    r"\b(treat\w*|prescri\w*|medication\w*|antibiotic\w*|surgery|dose\w*)\b": "gives treatment advice",
    r"\b(normal|healthy|clear)\s+(chest|lungs?|x-?ray|scan)\b": "calls the image normal",
}
# Digits inside words (densenet121, v2) are names, not numbers.
NUMBER = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?!\w)")


def allowed_numbers(facts: dict) -> set[str]:
    values = {str(len(facts[k])) for k in ("likely", "possible", "unlikely", "not_banded", "unavailable")}
    values.add(str(len(facts["likely"]) + len(facts["possible"])))
    values |= {"15", "14", "320"}
    for band in ("likely", "possible", "not_banded"):
        for f in facts[band]:
            values |= {NUMBER.search(v).group() for k, v in f.items() if k != "finding"}
    return values


def sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def check_summary(text: str, facts: dict) -> list[str]:
    issues = []
    lower = text.lower()
    words = len(text.split())
    if not MIN_WORDS <= words <= MAX_WORDS:
        issues.append(f"length is {words} words; write between {MIN_WORDS} and {MAX_WORDS}")
    for band in ("likely", "possible"):
        for f in facts[band]:
            if f["finding"].lower() not in lower:
                issues.append(f"{f['finding']} ({band}) is not mentioned")
    for name in facts["unlikely"]:
        for s in sentences(text):
            if re.search(rf"\b{re.escape(name.lower())}\b", s.lower()) and "unlikely" not in s.lower():
                issues.append(f"{name} was scored unlikely but is mentioned without saying so")
                break
    unknown = sorted(set(NUMBER.findall(text)) - allowed_numbers(facts))
    if unknown:
        issues.append(f"numbers not in the facts: {', '.join(unknown)}; only use scores exactly as given")
    for pattern, reason in BANNED.items():
        match = re.search(pattern, lower)
        if match:
            issues.append(f"'{match.group()}' {reason}")
    if (facts["unavailable"] or facts["problems"]) and not re.search(r"unavailable|failed|could not", lower):
        issues.append("does not say which parts of the screening were unavailable or failed")
    return issues