<div align="center">

# 🫁 PNEUMYST AI

### Decoding the unseen.

**A single-agent, explainable chest X-ray screening research platform**

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-DenseNet--121-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-backend-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-frontend-000000?logo=nextdotjs&logoColor=white)](https://nextjs.org/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-UI-06B6D4?logo=tailwindcss&logoColor=white)](https://tailwindcss.com/)
[![Clerk](https://img.shields.io/badge/Clerk-auth-6C47FF?logo=clerk&logoColor=white)](https://clerk.com/)
[![MySQL](https://img.shields.io/badge/MySQL-8-4479A1?logo=mysql&logoColor=white)](https://www.mysql.com/)
[![Cloudflare R2](https://img.shields.io/badge/Cloudflare_R2-storage-F38020?logo=cloudflare&logoColor=white)](https://developers.cloudflare.com/r2/)
[![Explainability](https://img.shields.io/badge/XAI-Occlusion_Sensitivity-8A2BE2)](#-explainability-occlusion-sensitivity)
[![Status](https://img.shields.io/badge/status-in_development-orange)](#-implementation-ledger)
[![Use](https://img.shields.io/badge/use-research_only-red)](#%EF%B8%8F-medical-disclaimer)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

[Overview](#-overview) •
[Architecture](#-system-architecture) •
[Workflow](#-analysis-workflow) •
[Data & privacy](#-accounts-data-and-privacy) •
[Reports](#-screening-report-and-audit-trail) •
[Datasets](#-datasets) •
[Getting started](#-getting-started) •
[Evaluation](#-evaluation-protocol) •
[Roadmap](#-roadmap) •
[Limitations](#-limitations)

</div>

---

> [!WARNING]
> **PNEUMYST AI is a research prototype, not a medical device.** It is not cleared or approved for clinical use, it does not diagnose disease, and its outputs must be reviewed by a qualified clinician. A high score is not a diagnosis, and a heatmap is not proof of a finding.

---

## 📌 Overview

PNEUMYST AI analyses a chest X-ray and returns **screening scores** for five radiographic targets, a **visual explanation** of which image regions influenced each score, explicit **uncertainty and failure flags**, and a **structured report** for clinician review. Signed-in users keep a private **history** of their analyses.

It is built around three engineering principles:

1. **A valid prediction stays available when its explanation fails.**
2. **An unavailable model never becomes a fabricated negative result.**
3. **An attractive heatmap is never mistaken for proof of a medical finding.**

### Screening targets

| # | Target | Model | Training data | Status |
|:-:|---|---|---|:-:|
| 1 | Tuberculosis (TB) | DenseNet-121 (binary) | TBX11K | 🟡 Planned |
| 2 | Pneumonia | DenseNet-121 (multi-label) | NIH ChestX-ray14 | 🟡 Planned |
| 3 | Pneumothorax | DenseNet-121 (multi-label) | NIH ChestX-ray14 | 🟡 Planned |
| 4 | Pleural effusion | DenseNet-121 (multi-label) | NIH ChestX-ray14 | 🟡 Planned |
| 5 | Lung nodule / mass | DenseNet-121 (multi-label) | NIH ChestX-ray14 | 🟡 Planned |

> A nodule or mass score does **not** indicate lung cancer. Each target is a screening signal only.

### Key features

- 🧠 **Trained computer-vision models** do all image analysis; the LLM never invents predictions.
- 🔍 **Occlusion Sensitivity** heatmaps with validation, caching and adaptive GPU batching.
- 🤖 **One orchestrator agent** that sequences six typed, allow-listed tools.
- 🛡️ **Fault tolerance by design**: independent statuses, bounded retries, deadlines, safe error codes.
- 🔐 **Secure accounts** with Clerk; every analysis is private to the user who created it.
- 🗄️ **Persistent history** in MySQL, with heatmaps in a private Cloudflare R2 bucket.
- 📄 **Downloadable Screening Analysis Report (PDF)** and a machine-readable **audit record (JSON)** for every analysis.
- 📊 **Honest evaluation** on held-out and external datasets from different hospitals.
- 🖥️ **Clinical-style dashboard** in Next.js that shows uncertainty instead of hiding it.

---

## 🏗 System architecture

```mermaid
flowchart TB
    subgraph FE["Frontend · Next.js + Tailwind"]
        LG["Sign in / sign up<br/>Clerk components"]
        U["Upload and preview"]
        R["Results dashboard<br/>scores · heatmap · report"]
        H["History"]
    end

    CL["Clerk<br/>authentication"]

    subgraph API["Backend · FastAPI + Pydantic"]
        AU["Auth dependency<br/>verify Clerk JWT"]
        EP["/analyze · /analyses · /models · /health"]
        RP["Repositories<br/>queries scoped to user_id"]
    end

    subgraph AG["Layer C · Orchestrator agent"]
        O["Single agent<br/>allow-listed tools only"]
    end

    subgraph CV["Layer B · Computer vision services"]
        TB["TB classifier<br/>DenseNet-121"]
        ML["Multi-label classifier<br/>DenseNet-121"]
        OS["Occlusion Sensitivity<br/>engine"]
        LOC["Localization<br/>optional"]
    end

    subgraph PRE["Layer A · Input and preprocessing"]
        V["Validation · decoding<br/>resize · normalise"]
    end

    subgraph DATA["Persistence"]
        DB[("MySQL 8<br/>users · analyses")]
        R2[("Cloudflare R2<br/>private heatmaps")]
        C[("Explanation cache")]
    end

    LLM["LLM provider<br/>optional"]

    LG <--> CL
    U -- "Bearer JWT" --> AU --> EP --> O
    AU -. "JWKS" .-> CL
    O --> V
    O --> TB & ML
    O --> OS
    O --> LOC
    OS <--> C
    O -. "sanitised outputs only" .-> LLM
    EP --> RP --> DB
    EP --> R2
    EP --> R
    H --> EP
```

| Layer | Responsibility | Key rule |
|---|---|---|
| **A · Input** | Validate, decode, preprocess, record metadata | Reject invalid images before any model runs |
| **B · Vision** | Classification, explanation, optional localization | Models produce every score; nothing is inferred from text |
| **C · Agent** | Sequence tools, gather outputs, compile flags, write report | Cannot change scores, thresholds or evidence |
| **D · API/UI** | Authenticate, serve, persist and display results | Every request is tied to a verified user; missing models show as unavailable |

---

## 🔄 Analysis workflow

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant UI as Next.js UI
    participant Clerk
    participant API as FastAPI
    participant Agent as Orchestrator
    participant CV as Vision services
    participant XAI as Occlusion engine
    participant DB as MySQL
    participant R2 as Cloudflare R2

    User->>UI: Sign in
    UI->>Clerk: Authenticate
    Clerk-->>UI: Session + JWT
    User->>UI: Upload chest X-ray
    UI->>API: POST /analyze (Bearer JWT)
    API->>API: Verify JWT → user_id
    API->>Agent: Start analysis
    Agent->>CV: validate_image
    alt Invalid image
        CV-->>Agent: INVALID_IMAGE
        Agent-->>API: Rejected with safe error code
    else Valid image
        Agent->>CV: classify_chest_xray
        CV-->>Agent: Per-target scores + model versions
        Agent->>CV: localize_findings
        CV-->>Agent: Supported outputs or "unsupported"
        Agent->>XAI: generate_occlusion_explanation
        alt Explanation succeeds
            XAI-->>Agent: Validated heatmap
        else Timeout / OOM / degenerate map
            XAI-->>Agent: Status + error code, no heatmap
        end
        Agent->>Agent: validate_predictions (fixed thresholds)
        Agent->>Agent: generate_report (LLM or deterministic)
        Agent-->>API: Typed result
        API->>R2: Upload heatmap (private)
        API->>DB: Save analysis row for user_id
    end
    API-->>UI: JSON response + short-lived heatmap URL
    UI-->>User: Scores · heatmap · flags · report
```

Partial success is a feature: if one model or the explanation fails, every other valid result is still returned and saved with its own status.

---

## 🔍 Explainability: Occlusion Sensitivity

Occlusion Sensitivity estimates how much each image region matters to a score by hiding patches of the image and measuring how the score changes.

```mermaid
flowchart LR
    A["Preprocessed<br/>X-ray"] --> B["Original<br/>target score"]
    A --> C["Grid of patches<br/>size · stride"]
    C --> D["Mask each patch<br/>with baseline"]
    D --> E["Batched forward<br/>passes"]
    E --> F["Score drop<br/>per patch"]
    B --> F
    F --> G["Map back to<br/>image grid"]
    G --> H{"Validate<br/>shape · NaN · degenerate"}
    H -- valid --> I["Overlay + legend"]
    H -- invalid --> J["No heatmap<br/>status + error code"]
```

| Aspect | Design |
|---|---|
| Grid | Configurable patch size and stride (starting point: 16 × 16 over the model input) |
| Speed | Bounded batches, `torch.inference_mode()`, GPU when available, coarse grid on CPU |
| OOM handling | Halve batch size → bounded retries → coarse-grid fallback (recorded) → optional CPU |
| Deadlines | Separate explanation deadline, cancellation checks between batches |
| Concurrency | Limited explanation jobs so simultaneous uploads cannot exhaust VRAM |
| Cache key | Image hash + model version + target + preprocessing + baseline + patch/stride + algorithm version |
| Validation | Rejects wrong shapes, NaN/Inf, constant maps and stale artifacts |

> A heatmap shows what influenced the **model**, not where disease **is**. Masking can introduce artificial patterns, and results depend on patch size, stride and baseline.

---

## 🛡 Failure handling

Every stage reports its own status, so one failure never hides or fakes another result.

| Stage | Possible statuses |
|---|---|
| Input validation | `success` · `failed` |
| Classification (per target) | `success` · `unavailable` · `timeout` · `failed` |
| Localization | `success` · `unavailable` · `not_requested` |
| Explainability | `success` · `unavailable` · `timeout` · `failed` · `not_requested` |
| Report | `success` (LLM) · `success` (deterministic fallback) · `failed` |
| Persistence | `saved` · `not_saved` (result still returned to the user) |

| Error code | Meaning |
|---|---|
| `INVALID_IMAGE` | File could not be decoded or failed validation |
| `MODEL_UNAVAILABLE` | Checkpoint missing or not loaded; never reported as negative |
| `INFERENCE_TIMEOUT` / `INFERENCE_FAILED` | Model call exceeded its deadline or errored |
| `EXPLANATION_TIMEOUT` / `EXPLANATION_OOM` | Heatmap stopped by deadline or GPU memory limits |
| `INVALID_HEATMAP` / `EXPLANATION_DEGENERATE` | Map failed validation or carried no information |
| `UNAUTHENTICATED` / `FORBIDDEN` | Missing or invalid token, or access to another user's analysis |
| `STORAGE_FAILED` | Heatmap upload or database write failed; analysis result is still returned |

---

## 🔐 Accounts, data and privacy

| Component | Service | Stores |
|---|---|---|
| Authentication | **Clerk** | User identity and sessions; passwords never touch our backend |
| Database | **MySQL 8** | Users and analysis records: scores, statuses, flags, report, model versions |
| File storage | **Cloudflare R2** (private bucket) | Heatmap overlays; original X-rays only if the user opts in |

### Database tables

| Table | Key columns |
|---|---|
| `users` | `id` (Clerk user ID, `VARCHAR(64)`), `email`, `consent_accepted_at`, `created_at` |
| `analyses` | `id` (`CHAR(36)` UUID), `user_id`, `created_at`, `model_versions` (JSON), `scores` (JSON), `statuses` (JSON), `flags` (JSON), `report` (TEXT), `report_source` (`llm` / `deterministic`), `case_context` (JSON, nullable, user-entered), `heatmap_key`, `image_key` (nullable), `image_hash` |
| `audit_events` | `id`, `analysis_id`, `step`, `tool`, `status`, `error_code`, `latency_ms`, `config` (JSON), `fallback_used`, `created_at` |

All tables use `utf8mb4`, timestamps are stored in UTC, and schema changes go through Alembic migrations.

### Privacy rules

- **Results are saved by default; X-rays are not.** The original image is stored only if the user explicitly chooses to save it.
- **Users only see their own data.** Every query is filtered by the verified `user_id`, and tests check that one user cannot read another user's analysis.
- **Private storage only.** Heatmaps are served through presigned URLs that expire within minutes.
- **Consent at sign-up.** Users confirm they will upload only de-identified or public research images.
- **PNG/JPEG only.** DICOM is not accepted, so no patient metadata arrives in file headers.
- **Real deletion.** Deleting an analysis removes its database row and its stored files.
- **Nothing identifiable goes to the LLM.** Only sanitised scores and statuses are sent, and only when the LLM is enabled.

---

## 📄 Screening report and audit trail

Every analysis can be exported in two forms. Neither contains patient-identifying information, and **no patient details are ever generated by AI**.

```mermaid
flowchart LR
    A["Stored analysis<br/>MySQL + R2"] --> B["Screening Analysis Report<br/>PDF · for people"]
    A --> C["Audit record<br/>JSON · for reproducibility"]
    D["Optional case context<br/>entered by user"] --> B
    E["Clinician review<br/>signed by a human"] -.-> B
```

### Screening Analysis Report (PDF)

| Section | Contents |
|---|---|
| Header | "Research Screening Analysis — not a diagnostic report", analysis ID, date/time (UTC) |
| Case context *(optional)* | Age band, sex, reason for X-ray, entered by the user and labelled **"provided by user, not used by the models"** |
| Image | Thumbnail, heatmap overlay with legend, dimensions, format, image hash |
| Results | Per-target score, validated threshold, above/below threshold, status |
| Explanation | Target explained, method, grid settings, any fallback used |
| Flags | Low-confidence results, unavailable models, quality warnings |
| Summary | Report text generated only from the results above (LLM or deterministic template) |
| Limitations | Research-use statement and known limitations |
| Clinician review | Blank fields for reviewer name, decision, signature and date |

**Never included:** name, patient ID, date of birth, exact address, phone number or any field filled in by AI.

### Audit record (JSON)

Analysis ID, user ID, timestamps, image hash, model names, versions and checkpoint hashes, preprocessing configuration, thresholds used, every tool call with status, latency and error code, explanation settings and fallbacks, and whether the summary came from the LLM or the deterministic template. Any result in the PDF can be traced back to the exact computation that produced it.

---

## 🗂 Datasets

All datasets are public research datasets. **No images are stored in this repository.** Download each from its official source and accept its terms.

| Dataset | Role | Size | Access | Source |
|---|---|---|---|---|
| **TBX11K** | Train/validate TB model | 11,200 X-rays with TB boxes | Public download | [GitHub](https://github.com/yun-liu/tuberculosis) |
| **NIH ChestX-ray14** | Train/test multi-label model | 112,120 X-rays, 30,805 patients | Public, no registration | [NIH Box](https://nihcc.app.box.com/v/ChestXray-NIHCC) |
| **Shenzhen** | External TB test | 662 X-rays | Public | [NLM](https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/) |
| **Montgomery County** | External TB test | 138 X-rays | Public | [NLM](https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/) |
| **CheXpert** (validation set) | External test for multi-label model | Radiologist-labelled validation set | Free registration | [Stanford AIMI](https://aimi.stanford.edu/datasets/chexpert-chest-x-rays) |
| **VinDr-CXR** | Optional future external test | — | Credentialed PhysioNet access | [PhysioNet](https://physionet.org/content/vindr-cxr/) |

### Data engineering rules

- Every image keeps its **source, image ID, patient ID, original label, canonical label, annotation type, split, uncertainty flag and label source**.
- **Missing labels are never treated as negative.**
- **Patient-level splits** wherever patient IDs exist; official splits are preferred.
- **Perceptual-hash deduplication** across datasets to prevent leakage.
- TB is trained **only within TBX11K** to avoid learning the hospital instead of the disease.
- NIH labels are NLP-mined from reports and are noisy; this is reported, not hidden.
- `nodule_mass` is defined explicitly as `Nodule OR Mass`.
- TBX11K test labels are held by the authors for an online challenge, so the local test split is carved from the official validation set and documented.

---

## 🧰 Tech stack

| Area | Tools |
|---|---|
| **Machine learning** | Python 3.12 · PyTorch · Torchvision · scikit-learn · NumPy · Pandas · OpenCV · Pillow · Albumentations |
| **Explainability** | Occlusion Sensitivity (NumPy + PyTorch) · Matplotlib overlays |
| **Backend** | FastAPI · Uvicorn · Pydantic · OpenAPI/Swagger |
| **Authentication** | Clerk (Next.js components · JWT verification via JWKS in FastAPI) |
| **Database** | MySQL 8 · SQLAlchemy 2.0 · PyMySQL · Alembic migrations |
| **Object storage** | Cloudflare R2 (S3-compatible) · boto3 · presigned URLs |
| **Reports** | ReportLab (PDF) · JSON audit records |
| **Frontend** | Next.js · React · Tailwind CSS · Recharts · `@clerk/nextjs` |
| **Optional** | Ultralytics YOLOv8 (TB localization) · U-Net (pneumothorax segmentation) |
| **Tooling** | Git · GitHub · Antigravity · VS Code |
| **Deployment** | Vercel (frontend) · Hugging Face Spaces / Render (backend, CPU) · hosted MySQL (e.g. Aiven) · Cloudflare R2 |

**Reference hardware:** Windows, NVIDIA RTX 3050 (4 GB VRAM). Training and inference are configured for limited GPU memory with mixed precision and adaptive batch sizes.

---

## 📁 Project structure

```text
pneumyst-ai/
├── frontend/                 # Next.js dashboard (created with create-next-app)
│   ├── app/                  # pages: sign-in, sign-up, upload, results, history
│   ├── components/           # viewer, score cards, status chips, report
│   ├── lib/                  # API client (attaches Clerk token)
│   └── types/                # TypeScript mirrors of backend schemas
├── backend/
│   ├── app/
│   │   ├── api/              # FastAPI routes: analyze, analyses, models, health
│   │   ├── agents/           # single orchestrator + tool registry
│   │   ├── auth/             # Clerk JWT verification → current user
│   │   ├── core/             # config (.env), logging, limits
│   │   ├── db/               # MySQL engine, session, SQLAlchemy table models
│   │   ├── repositories/     # database queries, always scoped to user_id
│   │   ├── storage/          # Cloudflare R2 client, presigned URLs
│   │   ├── reporting/        # PDF report builder + JSON audit export
│   │   ├── schemas/          # Pydantic request/response models
│   │   ├── services/         # inference, explanation, report
│   │   ├── ml_models/        # ML model registry and checkpoint loaders
│   │   └── utils/
│   ├── migrations/           # Alembic migrations
│   ├── tests/                # unit, integration, API, auth and failure-path tests
│   └── requirements.txt
├── ml/
│   ├── configs/              # versioned training + threshold configs
│   ├── datasets/             # dataset loaders + canonical label mapping
│   ├── preprocessing/        # validation, resizing, manifests, dedupe
│   ├── training/             # training loops with safe checkpointing
│   ├── evaluation/           # metrics, calibration, external tests
│   ├── inference/            # standalone inference utilities
│   └── checkpoints/          # local only, git-ignored
├── docs/                     # architecture, model cards, evaluation, limitations
├── .env.example
├── .gitignore
└── README.md
```

---

## 🚀 Getting started

> Setup steps are added as each stage lands. Commands marked *planned* describe the intended interface.

### Prerequisites

- Python 3.12
- NVIDIA GPU with a recent driver (optional but recommended for training)
- Node.js 18+ (for the frontend)
- MySQL 8 (local install for development)
- Free accounts on [Clerk](https://clerk.com/) and [Cloudflare](https://dash.cloudflare.com/) (for R2)
- About 60 GB of free disk space for raw datasets (about 10 GB after preprocessing)

### 1. Clone and create the environment

```powershell
git clone https://github.com/Fy1zN/pneumyst-ai.git
cd pneumyst-ai
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # Windows PowerShell
# source .venv/bin/activate           # macOS / Linux
```

Install PyTorch with the CUDA build matching your driver from [pytorch.org](https://pytorch.org/get-started/locally/), then:

```powershell
pip install -r backend/requirements.txt
```

### 2. Set up MySQL

```sql
CREATE DATABASE pneumyst CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER 'pneumyst'@'localhost' IDENTIFIED BY 'change-me';
GRANT ALL PRIVILEGES ON pneumyst.* TO 'pneumyst'@'localhost';
```

### 3. Configure

```powershell
Copy-Item .env.example .env
```

Fill in `.env`:

- `DATA_ROOT`: a folder **outside** the repository.
- `DATABASE_URL`: your MySQL connection string.
- `CLERK_SECRET_KEY`, `CLERK_JWKS_URL`, `CLERK_ISSUER`: from your Clerk application.
- `R2_*`: your R2 bucket name, endpoint and API token.

The frontend uses its own `frontend/.env.local` with `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY` and the API URL.

### 4. Create the database tables *(planned)*

```powershell
cd backend
alembic upgrade head
```

### 5. Download and preprocess data *(planned)*

```powershell
python -m ml.preprocessing.build_manifests --dataset tbx11k
python -m ml.preprocessing.build_manifests --dataset nih
```

### 6. Train *(planned)*

```powershell
python -m ml.training.train --config ml/configs/tb_densenet121.yaml
python -m ml.training.train --config ml/configs/nih_multilabel_densenet121.yaml
```

### 7. Run the backend *(planned)*

```powershell
uvicorn backend.app.main:app --reload
```

API docs: `http://localhost:8000/docs`

### 8. Run the frontend *(planned)*

```powershell
cd frontend
npm install
npm run dev
```

App: `http://localhost:3000`

---

## 🔌 API

All endpoints except `/health` require a valid Clerk session token: `Authorization: Bearer <token>`.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service health and model load status |
| `GET` | `/models` | Registered models, versions and availability per target |
| `POST` | `/analyze` | Upload an X-ray; returns scores, explanation status, flags and report, and saves the result |
| `GET` | `/analyses` | The signed-in user's analysis history |
| `GET` | `/analyses/{id}` | One past analysis with a short-lived heatmap URL |
| `DELETE` | `/analyses/{id}` | Delete an analysis and its stored files |
| `GET` | `/analyses/{id}/report.pdf` | Download the Screening Analysis Report |
| `GET` | `/analyses/{id}/audit.json` | Download the audit record |

Internal file paths and storage keys are never exposed to the frontend.

---

## 📏 Evaluation protocol

| Task | Metrics |
|---|---|
| Classification | AUROC · AUPRC · sensitivity · specificity · precision · F1 · confusion matrix · calibration · bootstrap CIs |
| External generalisation | Same metrics on Shenzhen, Montgomery and CheXpert validation |
| Explainability | Repeatability · sensitivity to baseline and grid · score drop when top regions are occluded · overlap with TBX11K boxes · latency · peak memory · failure rate |
| Localization *(optional)* | mAP@0.5 · mAP@0.5:0.95 · Dice · IoU |

Thresholds are chosen on validation data only and stored in version-controlled config. Test sets are never used for model selection.

### Results

> **No results yet.** This table is filled only with metrics from completed, reproducible runs.

| Model | Test set | AUROC | AUPRC | Sensitivity | Specificity |
|---|---|:-:|:-:|:-:|:-:|
| TB · DenseNet-121 | TBX11K (local test) | — | — | — | — |
| TB · DenseNet-121 | Shenzhen + Montgomery | — | — | — | — |
| Multi-label · DenseNet-121 | NIH official test | — | — | — | — |
| Multi-label · DenseNet-121 | CheXpert validation | — | — | — | — |

---

## 🗺 Roadmap

```mermaid
gantt
    title PNEUMYST AI build plan
    dateFormat YYYY-MM-DD
    axisFormat %b %d
    section Data and models
    Setup and downloads          :a1, 2026-10-18, 1d
    Dataset engineering          :a2, 2026-10-19, 1d
    TB model                     :a3, 2026-10-20, 2d
    Gate 1 TB verified           :milestone, 2026-10-21, 0d
    Occlusion Sensitivity        :a4, 2026-10-22, 2d
    NIH multi-label model        :a5, 2026-10-23, 2d
    section Backend
    FastAPI + Clerk + MySQL + R2 :b1, 2026-10-25, 1d
    Gate 2 backend works         :milestone, 2026-10-25, 0d
    Agent + report + audit       :b2, 2026-10-26, 1d
    section Frontend and release
    Next.js + login + history    :c1, 2026-10-27, 2d
    Tests and docs               :c2, 2026-10-29, 1d
    Deploy                       :c3, 2026-10-30, 1d
    v1.0                         :milestone, 2026-10-31, 0d
```

---

## ✅ Implementation ledger

| Feature | Status |
|---|:-:|
| Repository skeleton | ✅ Implemented |
| Environment and GPU check | ⏳ In progress |
| Dataset manifests and canonical labels | ⬜ Not started |
| TB classifier | ⬜ Not started |
| NIH multi-label classifier | ⬜ Not started |
| Occlusion Sensitivity engine | ⬜ Not started |
| Explanation cache and fallbacks | ⬜ Not started |
| Orchestrator agent and tools | ⬜ Not started |
| FastAPI endpoints | ⬜ Not started |
| Clerk authentication | ⬜ Not started |
| MySQL schema and migrations | ⬜ Not started |
| Cloudflare R2 storage | ⬜ Not started |
| Screening report (PDF) + audit record (JSON) | ⬜ Not started |
| Next.js dashboard and history | ⬜ Not started |
| Test suite (18 required cases + auth isolation + report checks) | ⬜ Not started |
| Deployment | ⬜ Not started |
| Localization (optional) | ⬜ Not started |

Statuses move through **Not started → In progress → Implemented → Tested → Verified in running app**. A feature is never marked complete just because code exists.

---

## ⚠️ Limitations

- Trained on public research data; performance on other hospitals, devices and populations is unknown until externally validated.
- NIH ChestX-ray14 labels were extracted from reports by NLP and contain errors, especially for pneumonia.
- Models may learn shortcuts (for example, chest drains for pneumothorax); heatmaps are used to check for this but cannot rule it out.
- Occlusion Sensitivity is a model-attribution aid, not a causal explanation and not clinical evidence.
- No prospective or clinical validation has been performed.
- The account system is designed for a research prototype and has not been audited for clinical data protection standards.

## ⚕️ Medical disclaimer

PNEUMYST AI is for **research and education only**. It is not intended to diagnose, treat, cure or prevent any disease, and must not be used for clinical decision-making. Always consult a qualified healthcare professional.

---

## 🙏 Acknowledgements and citations

- **NIH ChestX-ray14** — data provided by the NIH Clinical Center. Download site: <https://nihcc.app.box.com/v/ChestXray-NIHCC>.
  Wang X. et al., *ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases*, CVPR 2017.
- **TBX11K** — Liu Y. et al., *Rethinking Computer-aided Tuberculosis Diagnosis*, CVPR 2020.
- **Shenzhen and Montgomery** — Jaeger S. et al., *Two public chest X-ray datasets for computer-aided screening of pulmonary diseases*, Quant Imaging Med Surg, 2014. Provided by the U.S. National Library of Medicine.
- **CheXpert** — Irvin J. et al., *CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison*, AAAI 2019. Data provided by the Stanford Center for Artificial Intelligence in Medicine and Imaging (AIMI); see the AIMI dataset page for the required acknowledgement wording.

## 📄 License

Code in this repository is released under the [MIT License](LICENSE). Datasets are **not** included and remain under their own licenses and terms of use.

---

<div align="center">

Built with care for honest, explainable medical AI research.

</div>