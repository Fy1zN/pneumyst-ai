"""Fixed facts the assistant may quote: what each finding is and how the models were built and tested.

Written from the README's evaluation results, so answers rest on measured numbers rather than the LLM's memory.
"""
from __future__ import annotations

FINDINGS = {
    "Tuberculosis": ("A bacterial lung infection; on X-rays often patchy shadows or cavities in the upper lungs.",
                     "Separate TB model trained on TBX11K. Test AUROC 0.891 on Shenzhen but 0.744 on the external "
                     "Montgomery set, where its scores run low and only 38% of TB cases reach possible or likely."),
    "Pneumonia": ("Infection filling the air spaces of the lung.",
                  "Test AUROC 0.722, one of the weaker findings; NIH pneumonia labels are especially noisy."),
    "Consolidation": ("Lung tissue filled with fluid or cells instead of air, often from infection.",
                      "Test AUROC 0.743."),
    "Infiltration": ("A loose term for hazy opacity in the lung tissue.",
                     "Test AUROC 0.696, the weakest finding; the label itself is vague."),
    "Effusion": ("Fluid collecting between the lung and the chest wall.", "Test AUROC 0.826."),
    "Pneumothorax": ("Air between the lung and the chest wall, which can collapse part of the lung.",
                     "Test AUROC 0.863."),
    "Pleural thickening": ("Thickened lining around the lung, often from old inflammation.", "Test AUROC 0.781."),
    "Atelectasis": ("Part of the lung not fully inflated.", "Test AUROC 0.772."),
    "Edema": ("Fluid in the lung tissue, often linked to heart problems.",
              "Test AUROC 0.850 overall but 0.779 on AP images; part of the score reflects the AP view itself."),
    "Emphysema": ("Damaged, over-inflated air sacs, usually from smoking.", "Test AUROC 0.926, the strongest finding."),
    "Fibrosis": ("Scarring of the lung tissue.", "Test AUROC 0.825."),
    "Nodule": ("A small round spot in the lung, usually under 3 cm.", "Test AUROC 0.762."),
    "Mass": ("A larger spot in the lung, over 3 cm.", "Test AUROC 0.825."),
    "Cardiomegaly": ("An enlarged heart outline.",
                     "Test AUROC 0.894; images taken lying down make the heart look larger."),
    "Hernia": ("Part of the stomach pushing up through the diaphragm.",
               "Test AUROC 0.901, but only 15 validation cases, so no reliable band: the score is shown, not "
               "interpreted."),
}

MODEL_CARD = {
    "purpose": "Research screening of chest X-rays for 15 findings. Not a medical device and not a diagnosis.",
    "models": {
        "nih_densenet121_v1": "DenseNet-121 for 14 findings, trained on NIH ChestX-ray14; official test mean AUROC "
                              "0.813.",
        "tb_densenet121_v2": "DenseNet-121 for tuberculosis, trained on TBX11K; test AUROC 0.891 on Shenzhen and "
                             "0.744 on Montgomery.",
    },
    "bands": "Each calibrated score is compared with a possible and a likely threshold fixed on validation data. "
             "Rare findings can be likely at low percentages because their thresholds are low.",
    "heatmaps": "Occlusion sensitivity: regions are hidden one at a time and the drop in score is shown. Red marks "
                "what influenced the model most, not the outline of a lesion. Shown only for possible and likely "
                "findings.",
    "known_limits": [
        "Trained mostly on adult, frontal (PA and AP) X-rays.",
        "NIH labels were mined from radiology reports by text processing and contain errors.",
        "Some scores partly reflect the AP or PA view rather than the lungs.",
        "TB results are weaker on images from unfamiliar hospitals.",
    ],
}