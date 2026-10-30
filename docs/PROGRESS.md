# PNEUMYST AI · Progress and handoff notes

Working notes for continuing the project in a new session. The [README](../README.md) is the public description and holds the full results; this file records **where things are, how we work, and why decisions were made**.

_Last updated: 31 Oct 2026 · `main` at backend step 4. **The v1 ML work is complete; the backend is being built in steps.**_

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
| Backend step 1: FastAPI `/health`, `/models`, `/analyze` (14 API tests) | ✅ done, run live | `backend/app/`, `backend/tests/` |
| Backend step 2: MySQL + Alembic + history (9 tests) | ✅ done, run live on MySQL | `backend/app/db/`, `backend/app/repositories/`, `backend/app/api/analyses.py`, `backend/migrations/` |
| Backend step 3: Clerk authentication (19 tests) | ✅ done, run live (401 checks) | `backend/app/auth/`, `backend/app/api/deps.py` |
| Backend step 4: image storage + signed links (16 tests) | ✅ done, run live | `backend/app/storage/`, `backend/app/api/files.py` |
| **Backend step 5: PDF report + audit JSON + LangGraph report workflow** | ⏳ **next** (needs an OpenAI or Gemini key) | — |
| Step 6: LangGraph assistant `/ask` + hardening | ⬜ not started | — |
| Next.js dashboard, deploy | ⬜ not started | — |

## 2. Environment

- **Laptop:** Windows, **RTX 3050 Laptop GPU, 4 GB**, runs hot (~87 °C under load: keep it plugged in, on a hard surface, sleep set to Never during training).
- **Python:** `.venv` (3.12.8), PyTorch **2.11.0+cu128**. PyTorch is installed separately from `requirements.txt`.
- **Data:** `DATA_ROOT` in `.env` → `C:\Users\krish.LAPTOP-3EPNQNP7\Downloads\pneumyst-data` (`raw/`, `processed/manifests/`, `processed/images_320/`).
- **Checkpoints (git-ignored, back them up!):** `ml/checkpoints/<run>/best.pt` (selected epoch) and `last.pt` (resume).
  - `tb_densenet121_v1` (superseded), `tb_densenet121_v2` (**current TB**), `nih_densenet121_v1` (**current NIH**)
  - Each run folder also has `history.json`, `val_predictions.csv`, `test_predictions.csv` (and `dev_predictions.csv` for TB v2).
- **Tests:** `python -m pytest ml/tests backend/tests -q` → 114 passed (56 ML + 58 backend). A `StarletteDeprecationWarning` about `httpx` is harmless.
- **MySQL 8.0.46** (Windows service `MySQL80`, port 3306, firewall port not opened) + Workbench. Database `pneumyst` (utf8mb4) and user `pneumyst@localhost` with rights on that database only. `.env` holds `DATABASE_URL` (password of letters and digits) and `DEV_USER_ID=local-dev`.
- **Clerk:** development instance `endless-adder-4170`. `.env` holds `CLERK_SECRET_KEY`, `CLERK_ISSUER`, `CLERK_JWKS_URL`, `CLERK_AUTHORIZED_PARTIES`. `AUTH_MODE=dev` for daily local work (no token needed), `AUTH_MODE=clerk` to test auth; never `dev` when deployed. The publishable key goes in `frontend/.env.local` later.
- **Image storage:** `.env` holds `FILE_SIGNING_KEY` (random, 64 characters); images go to `DATA_ROOT\storage\analyses\<id>\`. The server refuses to start with storage on and no key.
- **Migrations:** `alembic -c backend/alembic.ini upgrade head` (reads `DATABASE_URL` from `.env`). Migration `0001` created `users`, `analyses`, `audit_events` on the local MySQL.
- **Run the API:** `uvicorn backend.app.main:app` from the repo root, then `http://localhost:8000/docs`. Restart uvicorn after code changes (no auto-reload).
- **Git habit:** after merging a PR on GitHub, `git pull` (or `git pull --rebase` if you already have local commits) before pushing.
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
- **Code style:** professional code: no emojis in code or code output, comments only where they explain *why*. VS Code "Insert Final Newline" should be on.
- Never show passwords or API keys in screenshots or chat; they live only in `.env`.

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

## 8. Backend

### Decisions
| Decision | Why |
|---|---|
| **Screening is a fixed pipeline**, not run by an agent | The steps never change; an LLM choosing them could only make results less predictable. Every stage reports its own status. |
| **Assistant agent on top of saved results** (step 5–6) | Tool-calling LLM (OpenAI or Gemini via `LLM_PROVIDER`) for the report summary and `POST /analyses/{id}/ask`. Read-only, user-scoped, allow-listed tools (`get_analysis`, `list_history`, `compare_analyses`, `get_finding_info`, `get_model_card`, `draft_report`), bounded tool calls, timeout, every call in `audit_events`, output checks, template fallback. **LangGraph (decided 31 Oct):** step 5 report workflow as a small graph (gather facts → LLM draft → checks → retry once or template fallback); step 6 assistant as a tool-calling graph with `recursion_limit` as the step bound. Chosen because the project targets GenAI roles and both parts are real workflows, not single calls. |
| **Store only the 320 px model input, never the original upload** | It is what the models saw, so heatmaps line up exactly; PNGs re-encoded from arrays carry no metadata or file name; small files. The upload's SHA-256 still identifies repeats. |
| Signed links instead of the Clerk token for images | `<img>` tags cannot send a bearer token. Links are HMAC-signed (key + expiry), last 5 minutes, and are only handed out by endpoints that check Clerk. |
| **No RAG** | Decided against for now; the LLM only works from the analysis results and our own finding descriptions. |
| Upload format by file signature, PNG/JPEG only | File names and content types can lie; no DICOM means no patient metadata arrives. |
| One screening at a time (`threading.Lock`, 60 s queue timeout → `503 SERVER_BUSY`) | 4 GB GPU. A running GPU job cannot be cancelled safely, so the timeout covers the queue wait only. |
| Removed `agents/`, `ml_models/`, `utils/` packages | The pipeline lives in `services/`, models load through `Screener`; the agent will get `backend/app/agent/`. |

### Step 1 results (run live)
- `/health` ok on CUDA; `/models` lists checkpoint SHA-256 for both models.
- Internet TB JPEG → TB 0.94 likely (confirmed TB). Montgomery `MCUCXR_0352_1` → TB 0.036 (a known missed case).
- Same image twice → identical scores. First request NIH 1.2 s (cuDNN warm-up), later 35 ms; total about 2.9 s with heatmaps, about 0.1 s with `explain=false`. **Hardening TODO:** warm-up pass at startup.

### Step 2 results (run live)
- Analyze → `persistence: saved`; MySQL shows 1 `analyses` row and 5 `audit_events` (input, both models, NIH heatmaps, TB heatmaps `skipped` with NULL latency).
- `GET /analyses` (paged, newest first, `flagged` findings), `GET /analyses/{id}`, `DELETE /analyses/{id}` (also removes audit events). Other users get 404. DB failure → results still returned with `STORAGE_FAILED`.
- Gotcha: `Access denied` from Alembic meant the `.env` password did not match MySQL (placeholder text left in).

### Step 3 results (run live)
- Clerk session tokens verified locally with the instance JWKS (RS256, cached keys): issuer, expiry, not-before, issued-at, subject and authorized party (`azp`) checked. The token's `sub` is the user id for every query.
- `/analyze`, `/analyses*` and `/models` need a token; `/health` stays public. Swagger has an **Authorize** button (bearer token).
- Live: no token → `401 "sign in required"`; fake token → `401 "invalid or expired session"` (server logs the reason). JWKS unreachable → `503 AUTH_UNAVAILABLE`. The app refuses to start in `clerk` mode without issuer and JWKS settings.
- A real signed-in token (200) is checked once the frontend exists: sign in, then `await window.Clerk.session.getToken()` in the browser console.
- The Clerk secret key was exposed in chat once and rotated. Secrets live only in `.env`.

### Step 4 results (run live)
- Analyze (NIH possible/likely findings, TB unlikely) → `stages.images: success`, 12 heatmap links for exactly the 12 `shown` findings, none for TB (`below_possible`), Edema or Hernia. Links expire 5 minutes after creation.
- Cardiomegaly overlay opened in the browser from the signed link: red on both heart borders, region outlined. `GET /analyses/{id}` gives fresh links; `DELETE` removed the folder in `DATA_ROOT\storage`.
- Image saving took about 0.7 s for 13 PNGs (`optimize=True`); `total` covers screening only. Possible hardening: drop `optimize` or add a full request time.
- Failure rules: storage error → scores kept, `errors.images = STORAGE_FAILED`; database error → stored images removed again; no storage configured → `STORAGE_NOT_CONFIGURED`.

### Plan
1. ✅ API skeleton
2. ✅ **MySQL + Alembic:** tables `users`, `analyses` (findings/stages/errors/timings/preprocess/models as JSON; `heatmap_keys`, `report`, `report_source` filled later), `audit_events`. `/analyze` saves (`persistence: saved/not_saved`, `STORAGE_FAILED` never loses results); `GET /analyses` (paged), `GET /analyses/{id}`, `DELETE /analyses/{id}`; other users' analyses return 404. Tests on in-memory SQLite.
3. ✅ **Clerk JWT auth:** the token's user id scopes every query; `AUTH_MODE=dev` keeps `DEV_USER_ID` for local work.
4. ✅ **Image storage:** 320 px image + shown overlays on local disk, HMAC-signed 5-minute links, delete cleanup. R2 comes at deploy.
5. PDF report + audit JSON + LangGraph report workflow (draft, checks, template fallback).
6. LangGraph assistant `ask` endpoint + hardening (startup warm-up, limits, full test list).

The backend machine needs `ml/checkpoints/tb_densenet121_v2/best.pt` and `ml/checkpoints/nih_densenet121_v1/best.pt`. CPU deployment makes heatmaps slow (30–60 s); decide at deploy time (GPU host, background heatmaps, or coarser grid).
