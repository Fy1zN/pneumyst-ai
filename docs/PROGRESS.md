# PNEUMYST AI · Progress and handoff notes

Working notes for continuing the project in a new session. The [README](../README.md) is the public description and holds the full results; this file records **where things are, how we work, and why decisions were made**.

_Last updated: 12 Nov 2026 · `main` at frontend step F4b. **The v1 ML work and the backend are complete; the frontend is being built in steps (F1 to F4 done).**_

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
| Backend step 5a: LangGraph report workflow + guardrails (26 tests) | ✅ done, run live with Gemini | `backend/app/agent/`, `GET /analyses/{id}/report`, migration `0002` |
| Backend step 5b: PDF report + audit JSON + case context (16 tests) | ✅ done, run live | `backend/app/reporting/`, migration `0003` |
| Backend step 5c: clinical context + WHO symptom screen + cautions (17 tests) | ✅ done, run live | `backend/app/services/case_notes.py`, `backend/app/reporting/` |
| Backend step 6a: LangGraph assistant agent `/ask` (8 tests) | ✅ done, run live with Gemini | `backend/app/agent/assistant.py`, `tools.py`, `knowledge.py` |
| Backend step 6b: hardening (9 tests) | ✅ done, run live | `backend/app/core/limits.py`, `backend/app/services/reports.py`, `backend/app/main.py` |
| Design (Figma: screens, phone, PDF, error states) | ✅ done, exported | `docs/design/` |
| Frontend F1: Next.js setup, theme, glass nav, footer, typed API client | ✅ done, run live against the backend | `frontend/` |
| Frontend F2: Clerk sign-in, protected pages, account menu, signed API calls | ✅ done, run live (email, Google and GitHub sign-in) | `frontend/src/proxy.ts`, `frontend/src/app/sign-in/`, `frontend/src/lib/use-api.ts` |
| Frontend F3: New scan page (case form, upload, progress card, every error state) | ✅ done, browser-tested against a test backend | `frontend/src/components/scan/`, `frontend/src/lib/case-fields.ts`, `frontend/src/lib/scan-errors.ts` |
| Frontend F4a: results page (scan list, X-ray viewer, findings by band, Report tab, downloads) | ✅ done, run live on a real scan | `frontend/src/components/results/`, `frontend/src/lib/format.ts` |
| Frontend F4b: Ask and Details tabs, delete scan, phone viewer | ✅ done, run live (Ask with Gemini, delete) | `frontend/src/components/results/ask-tab.tsx`, `details-tab.tsx` |
| **Frontend: chest X-ray check in the browser, then F5** | ⏳ **next** | `frontend/src/lib/`, `frontend/src/components/scan/` |
| Frontend F5, deploy | ⬜ not started | — |

## 2. Environment

- **Laptop:** Windows, **RTX 3050 Laptop GPU, 4 GB**, runs hot (~87 °C under load: keep it plugged in, on a hard surface, sleep set to Never during training).
- **Python:** `.venv` (3.12.8), PyTorch **2.11.0+cu128**. PyTorch is installed separately from `requirements.txt`.
- **Data:** `DATA_ROOT` in `.env` → `C:\Users\krish.LAPTOP-3EPNQNP7\Downloads\pneumyst-data` (`raw/`, `processed/manifests/`, `processed/images_320/`).
- **Checkpoints (git-ignored, back them up!):** `ml/checkpoints/<run>/best.pt` (selected epoch) and `last.pt` (resume).
  - `tb_densenet121_v1` (superseded), `tb_densenet121_v2` (**current TB**), `nih_densenet121_v1` (**current NIH**)
  - Each run folder also has `history.json`, `val_predictions.csv`, `test_predictions.csv` (and `dev_predictions.csv` for TB v2).
- **Tests:** `python -m pytest ml/tests backend/tests -q` → 186 passed (56 ML + 130 backend). Tests read PDFs with `pypdf`. A `StarletteDeprecationWarning` about `httpx` is harmless.
- **MySQL 8.0.46** (Windows service `MySQL80`, port 3306, firewall port not opened) + Workbench. Database `pneumyst` (utf8mb4) and user `pneumyst@localhost` with rights on that database only. `.env` holds `DATABASE_URL` (password of letters and digits) and `DEV_USER_ID=local-dev`.
- **Clerk:** development instance `endless-adder-4170`. `.env` holds `CLERK_SECRET_KEY`, `CLERK_ISSUER`, `CLERK_JWKS_URL`, `CLERK_AUTHORIZED_PARTIES`. `AUTH_MODE=dev` for daily local work (no token needed), `AUTH_MODE=clerk` to test auth; never `dev` when deployed. The publishable key goes in `frontend/.env.local` later.
- **LLM:** Gemini key from AI Studio (personal account, free tier; never the company OpenAI key). `.env`: `LLM_PROVIDER=gemini`, `LLM_MODEL=gemini-3.5-flash`, `LLM_FALLBACK_MODEL=gemini-3.5-flash-lite`, `LLM_THINKING_LEVEL=low`, `LLM_TIMEOUT_S=30`. Packages: `langgraph`, `langchain-google-genai`, `langchain-openai`. The key was visible in a screenshot once and rotated.
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
| **Assistant agent on top of saved results** (step 5–6) | Tool-calling LLM (OpenAI or Gemini via `LLM_PROVIDER`) for the report summary and `POST /analyses/{id}/ask`. Read-only, user-scoped, allow-listed tools (`get_analysis`, `list_history`, `compare_analyses`, `get_finding_info`, `get_model_card`; `draft_report` dropped because 5a writes reports), bounded tool calls, timeout, every call in `audit_events`, output checks, template fallback. **LangGraph (decided 31 Oct):** step 5 report workflow as a small graph (gather facts → LLM draft → checks → retry once or template fallback); step 6 assistant as a tool-calling graph with `recursion_limit` as the step bound. Chosen because the project targets GenAI roles and both parts are real workflows, not single calls. |
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

### Step 5a results (run live)
- `gemini-3.8-flash` first: one `504 DEADLINE_EXCEEDED` (default thinking, 20 s timeout) and one `503` "high demand"; both fell back to the template as designed. Led to: fallback model, `LLM_THINKING_LEVEL=low`, and not saving templates after a temporary LLM failure.
- `gemini-3.5-flash`: `source: llm`, all likely and possible findings named with exact scores, "the model scored X as likely" wording. Second request served instantly from MySQL.
- The provider reported `gemini-3.6-flash` as the serving model for a `gemini-3.5-flash` request; the report stores what the provider reports, not what was asked for.
- **Latency finding:** one successful call took ~58 s although `LLM_TIMEOUT_S=30`; the Gemini timeout is a deadline hint, not a hard stop. **Hardening TODO:** a total deadline enforced by our code, and writing the report in the background right after `/analyze`.
- Possible later: store rejected drafts and their issues for LLM evaluation (how often each rule fails).

### Step 5b results (run live)
- PDF opened from Swagger: real 320 px image and three heatmaps (Emphysema likely; Infiltration, Nodule possible), case context box, summary labelled "Written by gemini-3.5-flash-lite" (the fallback model answered because the main one was busy), findings by band, models, limitations, clinician review. Two A4 pages.
- Case context: case number pattern blocks names with spaces; age band, sex and reason come from fixed lists; bad values return `422 INVALID_REQUEST`. Validation errors now use the same error shape as every other failure.
- Seen in the PDF: stage order is shuffled because MySQL reorders JSON keys (cosmetic, fix in 5c). "Emphysema likely at 3.9%" is correct (likely threshold 2.4%) but reads oddly; the UI should lead with the band word.
- The test upload used age band 0-17 at random. The models were trained mostly on adult X-rays, so 5c adds a caution for 0-17 that never changes the scores.

### Decisions (5b)
| Decision | Why |
|---|---|
| **Case context is de-identified and from fixed lists** (case number, age band, sex, reason) | Full patient details would be personal health data (DPDP Act, HIPAA, GDPR). The clinic keeps the link from case number to patient; the PDF has a handwritten patient line. |
| **Symptoms and case context are shown side by side, never combined with scores** | The models are image-only and no dataset has symptoms, so any combined score would be untested. The clinician connects them. |
| PDFs and audit records are rebuilt on every request | Nothing to store, delete or go stale; the database and stored images are the single source. |

### Step 5c results (run live)
- Version 0.7.0 on port 8001 (an old 0.6.0 server still held 8000; check the version at the top of `/docs`).
- PDF: clinical context table, one caution (treated for TB before) and correctly none for PA, standing, digital, 18-29; WHO screen POSITIVE (all four) next to TB LIKELY (77%); summary still about the X-ray only; stages in pipeline order; two pages.
- Bug found and fixed before commit: a 30-character case number ran into the next column; table values now wrap.
- 5b audit record checked live: case context, all findings with thresholds, report trail (`gemini-3.6-flash`, 6.1 s with `LLM_THINKING_LEVEL=low`, down from 58 s), no storage keys or links.

### Decisions (5c)
| Decision | Why |
|---|---|
| **Image context drives cautions from our own findings** (lateral, AP, lying, phone photo, child, previous TB) | The models' known blind spots (frontal-only training, the AP/PA shortcut from EDA, adult data) should be visible to the clinician when they apply. |
| **Not asked is different from no** | A report must never claim a symptom is absent when nobody asked; the WHO screen is incomplete in that case. |
| Not collected: HIV status, pregnancy, exact age, dates, location, ethnicity, free text | Too sensitive or identifying, and nothing in the app would use them. |

### Step 6a results (run live)
- Version 0.8.0. "Why is tuberculosis marked likely?" on an analysis where TB was unlikely: Gemini called `get_analysis` and corrected the premise with the real numbers (2.7%, thresholds 33% and 69%). Grounded answer, not a made-up reason.
- Trap "Do I have TB and which antibiotic should I take?": refused. Log: `assistant answer rejected: 'diagnose' uses diagnosis language; 'medications' gives treatment advice`, most likely Gemini's own polite decline replaced by the fixed refusal (the checks block these words everywhere; accepted trade-off).
- Two indentation slips while pasting (`ANSWER_MAX_WORDS` and `add_events` ended up inside the previous function after its `return`); both caught before commit. Watch VS Code's auto-indent after a `return`.
- Ports: old uvicorn processes kept holding 8000 and 8001; `Get-Process python | Stop-Process -Force` clears them. Always Ctrl+C before starting a new server.

### Decisions (6a)
| Decision | Why |
|---|---|
| The assistant never sees case context or symptoms | Keeps the rule that case context never reaches the LLM, and stops the agent mixing symptoms with scores. |
| Questions and answers are not stored | Questions are free text and may contain personal details; only tools used and the outcome are audited. Stateless, one question at a time. |
| No template answer when the LLM fails (503 instead) | A made-up fallback answer would be worse than none. |

### Step 6b results (run live)
- Version 0.9.0. `/health`: status ok, cuda, both models, database ok, storage and LLM configured, provider gemini, auth mode dev.
- Background report: analyze saved at 13:56:16.300, the Gemini call started 47 ms later on its own and returned at 13:56:21.6; opening the report later was instant with `source: llm`. Background log lines show `[-]` as request id because they belong to no request.
- `X-Request-ID` on every response and in the log line of that request.
- Rate limits are in memory (one server); multiple server processes at deploy would need Redis.

### Plan
1. ✅ API skeleton
2. ✅ **MySQL + Alembic:** tables `users`, `analyses` (findings/stages/errors/timings/preprocess/models as JSON; `heatmap_keys`, `report`, `report_source` filled later), `audit_events`. `/analyze` saves (`persistence: saved/not_saved`, `STORAGE_FAILED` never loses results); `GET /analyses` (paged), `GET /analyses/{id}`, `DELETE /analyses/{id}`; other users' analyses return 404. Tests on in-memory SQLite.
3. ✅ **Clerk JWT auth:** the token's user id scopes every query; `AUTH_MODE=dev` keeps `DEV_USER_ID` for local work.
4. ✅ **Image storage:** 320 px image + shown overlays on local disk, HMAC-signed 5-minute links, delete cleanup. R2 comes at deploy.
5. ✅ (5a) LangGraph report workflow: draft, checks, one retry, template and fallback model, stored once. ✅ (5b) PDF report, audit JSON, case context. ✅ (5c) clinical context, WHO four-symptom screen and cautions, shown side by side and never combined with the scores.
6. ✅ (6a) LangGraph assistant `ask` endpoint. ✅ (6b) hardening: warm-up, background reports, deadlines, rate limits, health details, request IDs, test coverage table.

**Frontend (see section 9).** Next.js with Clerk sign-in, upload with case context, results with heatmap viewer and cautions, history, report and PDF download, assistant chat (with a "don't type personal details" note). Downloads fetch with the token and open as a blob; images use the signed links. Then deploy (R2 storage, GPU or background heatmaps, Redis if several server processes).

The backend machine needs `ml/checkpoints/tb_densenet121_v2/best.pt` and `ml/checkpoints/nih_densenet121_v1/best.pt`. CPU deployment makes heatmaps slow (30–60 s); decide at deploy time (GPU host, background heatmaps, or coarser grid).

## 9. Frontend

The design lives in Figma and is exported to [`docs/design/`](design/README.md) (screens, phone layouts, PDF pages, error states, colour and font tokens). The frontend is built from those pictures in steps.

### Decisions
| Decision | Why |
|---|---|
| Next.js 16 (App Router, TypeScript), React 19, Tailwind CSS 4 | Server and client components in one app, typed end to end, styling straight from the design tokens |
| Design tokens as Tailwind theme colours (`brand-green`, `band-likely`, ...) | One place to change a colour; the page code reads like the design |
| Fonts through `next/font` (Inter, Playfair Display italic, JetBrains Mono, Montserrat) | Served from our own site, so visitors' browsers never contact Google |
| API types generated from the backend's `/openapi.json` (`npm run api:types`, file committed) | A field the backend does not send is a compile error, and the site builds without the backend running |
| One `apiFetch` helper turning every error into `ApiError` with the backend's `error_code` and `Retry-After` | Pages pick the user-facing message per code (see `docs/design/boards/error-states.png`) |
| Brand green, mint and white only; red and amber only for findings and warnings | A result must never read as "all clear" because of a green colour |
| Glass nav: translucent green bar, sage glass pill for the current page, hover glass | Matches the approved design; real CSS instead of Figma effects |
| Clerk's prebuilt `<SignIn>` / `<SignUp>` styled through `appearance` and `localization`, not custom forms | Clerk already handles passwords, email codes, social sign-in, bot protection and lockouts; the card is close to the design, not pixel-identical |
| `src/proxy.ts` (Next.js 16's name for middleware) with `clerkMiddleware`; `/scan`, `/history`, `/results` private | Signed-out visitors go to sign-in and come back to the page they asked for |
| `useApi()` adds a fresh Clerk session token to every backend call | One place for auth headers; Clerk refreshes the token |
| Parts that read the URL or the signed-in user sit inside `<Suspense>` (nav, Clerk card, account check) | Cache Components (on by default in Next.js 16) refuses request-time data outside Suspense; each gets a same-size placeholder |
| Case form choices typed against the generated API types; "Not asked" is sent as nothing | A choice the backend does not accept fails to compile, and "not asked" never turns into "no" for the WHO screen |
| Browser checks before upload (PNG/JPEG, 20 MB) on top of the backend's checks | Obvious mistakes cost no request; the backend stays the authority |
| Progress steps advance on typical timings and jump to done when the single response arrives | The backend answers once; real step-by-step progress would need streaming from the backend |
| Every backend error mapped to the error-board wording (`scan-errors.ts`); the form keeps its answers | The user always sees what happened and what to do next, never a raw code |
| Glass panels at 55% white with a 6px blur over the lungs | Matches the Figma design; the dropdowns keep solid white boxes so they stay readable |

### Step F1 results (run live)
- `npm run typecheck`, `npm run lint` and `npm run build` clean (routes `/`, `/scan`, `/history`, `/icon.svg`).
- Home page against the real backend: badge shows `Backend: ok · cuda · v0.9.0 · database ok · llm configured`; with the backend stopped it shows "Backend not reachable" instead of crashing.
- Nav pill moves between Home, New scan and History; the phone menu opens and closes itself after navigating; footer GitHub and LinkedIn links open the profiles.
- `api:types` needs the backend running first; run it again whenever a backend response changes.
- The committed `package-lock.json` was written by npm 10.9.2 on Windows and is rejected by `npm ci` on npm 10.9.4 (a hoisting difference for `fdir`). `npm install` works; revisit before adding CI or deploying.

### Step F2 results (run live)
- Register and sign-in with email code, Google and GitHub; New scan and History redirect signed-out visitors to sign-in and back; avatar menu and sign out work.
- Backend switched to `AUTH_MODE=clerk`; the home page check shows `Signed in · token accepted · 11 scans saved` (earlier dev-mode scans were saved under the same Clerk user id through `DEV_USER_ID`).
- `typecheck`, `lint` and `build` clean; sign-in and register pages are partly prerendered (◐), the proxy is active.
- On a fresh clone, `npm run typecheck` fails on `LayoutProps` until `npm run dev` or `npm run build` has generated the route types once; switch the script to `next typegen && tsc --noEmit` before adding CI.
- Deploy note: GRADIA and PNEUMYST currently share one Clerk development app; give PNEUMYST its own production Clerk app at deploy.

### Step F3 results
- Browser-tested in the sandbox against a test backend that returns every outcome: a name-like case number is blocked with a hint; a PDF and a 21 MB file are stopped before upload; Cough duration appears only after Cough is "Yes"; a blank image shows "This image looks blank." in the upload box with the form kept; 429 shows the amber banner with the minutes from `Retry-After`; 503 shows "Screening could not run" with the reference; 401 shows the session card; not saved shows the flagged findings; success shows the progress card and opens `/results/{id}`; heatmaps off sends `explain=false` and drops that step; only the answered fields are sent, with the Bearer token.
- `typecheck`, `lint` and `build` clean on the pushed code; `/results/[id]` is a temporary page (case number, time, flagged findings, saved details) until F4.
- Run live: a real scan from the New scan page was saved and opened on `/results/{id}`.

### Step F4a results (run live)
- Real scans on the user's machine: X-ray and heatmaps load from the signed links, every heatmap chip switches the image, Overlay and the opacity slider work, the Gemini summary appears after "Writing the summary", the caution box shows, the PDF and audit record download with the backend's file names.
- Browser-tested in the sandbox against a test backend (31 checks): scan list of 8 with Load more and the case-number filter; partial result banner naming the unfinished steps; template summary; report error with Try again; download error message; no heatmaps (turned off) and no stored images; a failed model's 14 findings listed as unavailable; expired image links refetch the analysis once, not in a loop; unknown id shows "Scan not found"; no horizontal scroll at 390 px.
- The side panels scroll with the page (sticky panels were tried and removed: they felt detached under the floating nav).
- The PDF still has the pre-design layout until B1; the audit record is JSON until B2.
- `typecheck`, `lint` and `build` clean on the pushed code.

### Step F4b results (run live)
- No backend change: `DELETE /analyses/{id}` already existed (owner-only, removes the row, audit events and stored images; covered by `test_analyses.py` and `test_storage.py`) and CORS already allowed DELETE.
- Run live: Ask answers from Gemini with the tools it checked; an off-topic question ("which AI is replying") is refused as designed; Details shows case context, WHO screen, models, timings; Delete removes the scan and returns to New scan.
- Browser-tested in the sandbox against a test backend (40 checks): question length 3 to 500, Enter to send, suggestion chips, refused answers, busy then Try again, 429 with minutes, JSON body and Bearer token sent; chat kept when switching tabs; Details rows, failed stages and timings; delete dialog (Esc, Keep it, error keeps the dialog open); phone X-ray / Heatmap switch with no horizontal scroll at 390 px; both pictures side by side from 640 px.
- Fixed: stage names in the partial-result banner now read the backend's real keys (`tb_densenet121_v2`, `explain:nih_densenet121_v1`), which F4a had mapped from short names that never occur.
- Style decisions from the user: question bubbles in mint with a light green border; the delete dialog and button use the brand green, not red.
- Found while testing: a screenshot of the app was screened and got 8 "possible" findings, because the models score any image. Next step adds a browser-side chest X-ray check (greyscale, white share, dark share, flat areas) before upload; the API itself still accepts any image, so this is a convenience check, not a guarantee.

### Plan
1. ✅ **F1:** Next.js setup, theme and fonts, glass nav and phone menu, footer, typed API client, backend health check.
2. ✅ **F2:** Clerk sign-in and register styled like the design; the API client sends the session token; `AUTH_MODE=clerk` locally.
3. ✅ **F3:** New scan: case form from fixed lists, upload box, screening progress card, every error state.
4. ✅ **F4a:** Results: scan list, X-ray viewer (side by side and overlay), findings by band, Report tab, PDF and audit downloads.
5. ✅ **F4b:** Ask and Details tabs, phone layout of the results page, Delete this scan.
6. ⏳ **X-ray check:** reject screenshots, documents and colour photos in the browser before upload.
7. ⬜ **F5:** Landing and History pages, phone layouts, final checks; then deploy.
