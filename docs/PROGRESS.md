# PNEUMYST AI · Progress and handoff notes

Working notes for continuing the project in a new session. The [README](../README.md) is the public description and holds the full results; this file records **where things are, how we work, and why decisions were made**.

_Last updated: 25 Oct 2026 · `main` at the inference utility commit. **The v1 ML work is complete.**_

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
| Occlusion Sensitivity heatmaps + display rules | ✅ done, checked, frozen | `ml/explain/`, `notebooks/explain.ipynb`, `docs/figures/explain/` |
| Inference utility (both models → 15 scores, bands, heatmaps) | ✅ done, parity-checked | `ml/inference/screener.py`, `notebooks/inference.ipynb` |
| **Backend** (FastAPI, Clerk, MySQL, R2) | ⏳ **next** | — |
| Orchestrator + PDF report + audit JSON, Next.js dashboard, deploy | ⬜ not started | — |

## 2. Environment

- **Laptop:** Windows, **RTX 3050 Laptop GPU, 4 GB**, runs hot (~87 °C under load: keep it plugged in, on a hard surface, sleep set to Never during training).
- **Python:** `.venv` (3.12.8), PyTorch **2.11.0+cu128**. PyTorch is installed separately from `requirements.txt`.
- **Data:** `DATA_ROOT` in `.env` → `C:\Users\krish.LAPTOP-3EPNQNP7\Downloads\pneumyst-data` (`raw/`, `processed/manifests/`, `processed/images_320/`).
- **Checkpoints (git-ignored, back them up!):** `ml/checkpoints/<run>/best.pt` (selected epoch) and `last.pt` (resume).
  - `tb_densenet121_v1` (superseded), `tb_densenet121_v2` (**current TB**), `nih_densenet121_v1` (**current NIH**)
  - Each run folder also has `history.json`, `val_predictions.csv`, `test_predictions.csv` (and `dev_predictions.csv` for TB v2).
- **Tests:** `python -m pytest ml/tests -q` → 56 passed.
- **Heatmap cache:** `DATA_ROOT/processed/explain/cache/` (one `.npz` per image and setting). The first box-overlap run took about an hour; reruns read the cache.

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
| Occlusion: 32 px patch, stride 16, **blurred** baseline (σ 12), calibrated score drops | Black squares look like training padding; calibrated drops are comparable across findings. ≈1.45 s and 0.44 GB per image. |
| Heatmap display rules (frozen) | Map only for possible/likely findings; hidden if no region removes ≥ 5% of the score; smoothed σ 8; colour scale = max(peak, 25% of score); outline at half the peak. Low-score maps were noise painted red, and the hottest spot moves with patch size about half the time, so regions are shown instead of points. |
| No TB v3 for now | Would need chest-X-ray-pretrained weights (check their training data first) or multi-hospital training; left as future work. |

## 5. How we work

- **Code is shared in chat and pasted by Krish** (no direct pushes of code). Full files are preferred over diffs. Code is tested in a sandbox first.
- **README/doc updates** may be pushed to branch `claude/busy-goldberg-hc9ps6` and merged through a PR.
- **Explain first, then code** when asked; keep explanations simple.
- **Freeze before testing:** decide everything on validation, commit configs, then run each test set **once**. No retuning after seeing test results.
- After editing any `ml/` file, **restart the notebook kernel** and rerun the setup cell (Python caches imports). Never "Run All" on `train.ipynb`; the training cells are commented out after use.
- X-ray images are **kept** in notebook outputs on purpose (visual validation). Aggregate charts go to `docs/figures/`.
- The README "reliability tiers" idea was replaced by the bands; don't reintroduce tiers unless asked.

## 6. Occlusion Sensitivity: results

`ml/explain/occlusion.py` (`occlusion_map`, checks, `display_map`, `overlay`), checked in `notebooks/explain.ipynb`. Full write-up in the README → Explainability.

- **Box overlap (984 NIH test boxes):** hottest spot inside the box 43% vs 8% chance (Cardiomegaly 86%, Pneumothorax 18%, Atelectasis 23%).
- **Deletion:** hiding the hottest 10% drops the score 0.189 vs 0.049 for random; beats random in 99% of 120 images.
- **Stability:** median Spearman 0.74–0.77 across patch 24/48 and mean fill; same hotspot region only ~50%.
- **Edema AP shortcut:** not found (border heat AP 0.15 vs PA 0.19, even spread would be 0.42).
- **TB v2 Montgomery:** caught cases → upper lobes; missed cases mostly look in the right area but score too low.
- **Display rules:** with them, missed-TB noise maps are hidden, 96% of boxed findings still get a map, and the outline touches the box in 82%.

Notebook tip: after a restart, only the setup, gallery (`boxes = pd.read_csv(...)`), Montgomery (`mg = load_processed(...)`) and display-rules cells are needed to redo the preview. **Skip the box-overlap cell (`rows = []`)** unless the cache is gone.

## 7. Inference utility: results

`ml/inference/screener.py`: `Screener.load()` once, `screen(path | bytes | array, explain=True)` per image, `result.to_dict()` for JSON, `result.overlay(finding)` for an RGB heatmap image.

- **Parity** (`notebooks/inference.ipynb`, 250 raw test images): preprocessing identical 100%, max score difference 0.0007, same band 99.95%, NIH matches the saved `test_predictions.csv`.
- **Speed (RTX 3050):** 111 ms per image without heatmaps, about 2 s with them (occlusion runs only for models with a possible/likely finding).
- **Statuses:** `INVALID_IMAGE` (+ reason), `MODEL_UNAVAILABLE`, `INFERENCE_FAILED`, `EXPLANATION_OOM`, `EXPLANATION_DEGENERATE`, `INVALID_HEATMAP`, `EXPLANATION_FAILED`; overall `success` / `partial` / `failed`.
- **Windows note:** never put large byte strings in `pytest.mark.parametrize` without `ids=`; the test ID goes into an environment variable with a 32,767-character limit.
- **For the UI later:** rare findings become "possible" at very low scores (Cardiomegaly from 0.013), so lead with the band word rather than the percentage. Example seen: a missed Montgomery TB case where the Cardiomegaly and Pneumonia maps lit up the right upper lung.

## 8. Next: backend

Plan first (explain, then code), as usual. Planned in the README: FastAPI + Clerk JWT + MySQL (SQLAlchemy + Alembic) + Cloudflare R2, with the screener loaded once at startup. The backend machine needs `ml/checkpoints/tb_densenet121_v2/best.pt` and `ml/checkpoints/nih_densenet121_v1/best.pt`.
