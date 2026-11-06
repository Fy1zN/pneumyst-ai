"""Fixed rules over the case context: the WHO four-symptom screen and cautions about known model blind spots.

The rules read only what the user entered. They are shown next to the model scores and never change them.
"""
from __future__ import annotations

RULES_VERSION = 1
WHO_SYMPTOMS = {"cough": "current cough", "fever": "fever", "night_sweats": "night sweats",
                "weight_loss": "weight loss"}
OTHER_SYMPTOMS = {"breathlessness": "breathlessness", "chest_pain": "chest pain", "coughing_blood": "coughing up blood"}
# (code, field, value that triggers it, text); worded from what the training data and EDA showed.
CAUTIONS = [
    ("CHILD", "age_band", "0-17",
     "The models were developed mainly on adult chest X-rays; results for children are less reliable."),
    ("LATERAL_VIEW", "view", "lateral",
     "The models were trained on frontal (PA and AP) X-rays only; scores for a side view are not meaningful."),
    ("AP_VIEW", "view", "AP",
     "Some scores partly reflect the AP view itself rather than the lungs (for example Edema); read them with care."),
    ("LYING_POSITION", "position", "lying",
     "In images taken lying down the heart looks larger and fluid spreads out, which can affect the Cardiomegaly "
     "and Effusion scores."),
    ("PHONE_PHOTO", "image_source", "phone photo of film",
     "Photos of film can lose detail and add glare or distortion; scores may be less reliable."),
    ("PREVIOUS_TB", "previous_tb", "yes",
     "Healed scars from earlier TB can raise the Tuberculosis score even without active disease."),
]


def who_screen(case: dict) -> dict | None:
    """Positive if any WHO symptom is present, negative only if all four were answered no, otherwise incomplete."""
    answers = {field: case.get(field) for field in WHO_SYMPTOMS}
    if all(value is None for value in answers.values()):
        return None
    positive = [WHO_SYMPTOMS[f] for f, value in answers.items() if value == "yes"]
    not_asked = [WHO_SYMPTOMS[f] for f, value in answers.items() if value is None]
    result = "positive" if positive else ("incomplete" if not_asked else "negative")
    return {"result": result, "positive_symptoms": positive, "not_asked": not_asked}


def cautions(case: dict) -> list[dict]:
    return [{"code": code, "text": text} for code, field, value, text in CAUTIONS if case.get(field) == value]


def case_notes(case: dict | None) -> dict | None:
    if not case:
        return None
    return {"who_screen": who_screen(case), "cautions": cautions(case)}