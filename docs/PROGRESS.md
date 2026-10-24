# PNEUMYST AI · Progress and handoff notes

Working notes for continuing the project in a new session. The [README](../README.md) is the public description and holds the full results; this file records **where things are, how we work, and why decisions were made**.

_Last updated: 24 Oct 2026 · `main` at the README results merge (PR #3)._

---

## 1. Status

| Stage | State | Where |
|---|---|---|
| Data download, manifests, dedupe, leak checks | ✅ done | `ml/datasets/`, `DATA_ROOT/processed/manifests/` |
| EDA (23 sections) + shortcut probes | ✅ done | `notebooks/eda.ipynb`, `ml/eda/`, `docs/figures/eda/` |
| Preprocessing to 320 px (121,141 images, 0 failures) | ✅ done | `ml/preprocessing/`, `notebooks/preprocessing.ipynb`, `DATA_ROOT/processed/images_320/` |
| Training pipeline (34 tests) | ✅ done | `ml/training/`, `ml/tests/` |
| NIH 14-finding model v1 | ✅ trained, calibrated, tested | `notebooks/train.ipynb`, `ml/configs/nih_v1_calibration.json` |
| TB model v2 (replaces v1) | ✅ trained, calibrated, tested | `notebooks/train.ipynb`, `ml/configs/tb_v2_calibration.json` |
| README results section | ✅ merged | `README.md` → "Model results" |
| **Occlusion Sensitivity heatmaps** | ⏳ **next** | — |
| Backend (FastAPI, Clerk, MySQL, R2), reports, frontend | ⬜ not started | — |

## 2. Environment

- **Laptop:** Windows, **RTX 3050 Laptop GPU, 4 GB**, runs hot (~87 °C under load: keep it plugged in, on a hard surface, sleep set to Never during training).
- **Python:** `.venv` (3.12.8), PyTorch **2.11.0+cu128**. PyTorch is installed separately from `requirements.txt`.
- **Data:** `DATA_ROOT` in `.env` → `C:\Users\krish.LAPTOP-3EPNQNP7\Downloads\pneumyst-data` (`raw/`, `processed/manifests/`, `processed/images_320/`).
- **Checkpoints (git-ignored, back them up!):** `ml/checkpoints/<run>/best.pt` (selected epoch) and `last.pt` (resume).
  - `tb_densenet121_v1` (superseded), `tb_densenet121_v2` (**current TB**), `nih_densenet121_v1` (**current NIH**)
  - Each run folder also has `history.json`, `val_predictions.csv`, `test_predictions.csv` (and `dev_predictions.csv` for TB v2).
- **Tests:** `python -m pytest ml/tests -q` → 34 passed.

## 3. Current models

| | NIH v1 | TB v2 |
|---|---|---|
| Architecture | DenseNet-121, ImageNet weights, 320 px | same |
| Training | 10 epochs (best **7**), batch 16 × accum 2, padding aug 30% | 12 epochs (best **8**), batch 16 × accum 2, padding 50%, label smoothing 0.1, wider aug (`TB_V2_AUG` in notebook) |
| Model selection | NIH val mean AUROC | **Shenzhen-dev** AUROC (TBX11K val saturates at 1.000) |
| Calibration | Platt per finding on NIH val | Platt on Shenzhen-dev |
| Bands | possible = 90% sensitivity, likely = 90% specificity | same |
| **Test result** | official NIH test mean AUROC **0.813** | Shenzhen-test **0.891**, Montgomery **0.744**, TBX11K test 0.996 |

Key weaknesses (all in README → Limitations): TB catches only 38% of Montgomery TB at Shenzhen-tuned thresholds; Edema/Consolidation partly use the AP view; Infiltration 0.696 (0.599 on PA); Hernia `insufficient_data` (15 val positives); more false reds on the sicker NIH test set.

## 4. Decisions and why

| Decision | Why |
|---|---|
| **No `channels_last`** anywhere | On this GPU it made DenseNet backward ~10× slower (8 vs 82 img/s). Measured with `tr.benchmark_speed` + a channels_last/AMP grid. |
| AMP on, `memory_efficient=False`, batch 16 | Fastest measured setting (84 img/s, 2.4 GB). |
| Padding augmentation TB 50% / NIH 30% | External TB sets are 97–100% padded vs 0.2% of TBX11K; NIH trimming correlates mildly with AP view (AP 17% vs PA 8% trimmed). |
| No horizontal flips | Anatomy and L/R markers are not symmetric. |
| No CLAHE / per-image normalisation in preprocessing | EDA probes showed it does not remove hospital fingerprints; random CLAHE is used only as TB v2 augmentation. |
| TB trained only on TBX11K | Avoids learning the hospital instead of the disease (datasets are 99–100% separable by image statistics). |
| Shenzhen split 50/50 dev/test, frozen in `ml/configs/tb_shenzhen_split.csv` | Needed an external set for model selection; **Montgomery stays untouched** as the clean test. TB v1 had already seen all of Shenzhen, so Montgomery is the only fully clean TB number. |
| Platt scaling (not temperature only) | `pos_weight` inflated NIH scores 3–20×; Platt fixes both the slope and the shift. |
| Three bands instead of one threshold | A single 90%-sensitivity threshold flagged 24–74% of all images per finding. |
| Hernia shown as `insufficient_data` | Only 15 validation positives; kept for v1 even though test looked good (changing it after seeing test would be cheating). |
| Pre-registered TB rule | v2 replaces v1 if Montgomery AUROC higher, ECE lower, false reds not higher → it did. |
| No TB v3 for now | Would need chest-X-ray-pretrained weights (check their training data first) or multi-hospital training; left as future work. |

## 5. How we work

- **Code is shared in chat and pasted by Krish** (no direct pushes of code). Full files are preferred over diffs. Code is tested in a sandbox first.
- **README/doc updates** may be pushed to branch `claude/busy-goldberg-hc9ps6` and merged through a PR.
- **Explain first, then code** when asked; keep explanations simple.
- **Freeze before testing:** decide everything on validation, commit configs, then run each test set **once**. No retuning after seeing test results.
- After editing any `ml/` file, **restart the notebook kernel** and rerun the setup cell (Python caches imports). Never "Run All" on `train.ipynb`; the training cells are commented out after use.
- X-ray images are **kept** in notebook outputs on purpose (visual validation). Aggregate charts go to `docs/figures/`.
- The README "reliability tiers" idea was replaced by the bands; don't reintroduce tiers unless asked.

## 6. Next: Occlusion Sensitivity heatmaps

Plan (not started):
1. `ml/explain/occlusion.py`: slide a patch (e.g. 32 px, stride 16, baseline = blurred image or mean) over the 320 px image, batch the occluded copies on the GPU, heatmap = drop in **calibrated** score per region; upsample to 320 px.
2. Works for both models (14 NIH outputs in one pass, TB single output).
3. Checks from the README evaluation plan: repeatability, sensitivity to patch / stride / baseline, score drop when top regions are occluded, **overlap with the 984 NIH test boxes** (`nih_bboxes_320.csv`), latency and peak memory on the 3050.
4. Questions to answer: does Edema rely on the "PORTABLE" text or AP features? Where does TB v2 look on missed Montgomery cases?
5. Then: inference utilities (load both models + configs, one image → 15 calibrated scores, bands, heatmaps), which the backend will call.
