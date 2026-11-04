# VETIVER AI Platform

**Geotechnical Site Suitability Predictor & Literature Research Assistant**  
Jomo Kenyatta University of Agriculture and Technology (JKUAT)

> **Project Lead & Technical Lead:** Brian  
> **Team:** 6 members — Technical & Systems Division + Data & Sanitization Division  
> **Stack:** CatBoost · LlamaIndex · Qwen 2.5 3B · Ollama · Streamlit  
> **GitHub:** https://github.com/brayo-kowino/VETIVER-ML-MODEL

---

## What This System Does

The VETIVER AI Platform is a dual-engine decision support tool for *Chrysopogon zizanioides* (Vetiver grass) bio-engineering projects:

| Engine | Technology | What it does |
|:---|:---|:---|
| **Predictive Engine** | CatBoost + TreeSHAP | Classifies geotechnical site suitability (`SUITABLE` / `UNSUITABLE`) from soil, climate, watering, salinity, and application inputs. Outputs a confidence % and an explainable SHAP attribution chart per prediction. |
| **Literature Assistant** | Qwen 2.5 3B + LlamaIndex + Ollama | Answers research questions from a curated library of peer-reviewed Vetiver papers using a Retrieval-Augmented Generation (RAG) pipeline. No hallucinations — answers are grounded in the indexed literature. |

---

## Platform Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        Streamlit Web UI                          │
│                                                                  │
│   Tab 1: Geotechnical Predictor      Tab 2: Literature Chat      │
│   ┌─────────────────────────┐        ┌─────────────────────────┐ │
│   │ Site Parameters (5)     │        │ Natural Language Query  │ │
│   │ Soil / Climate /        │        │          ↓              │ │
│   │ Watering / Salinity /   │        │   LlamaIndex Vector     │ │
│   │ Application             │        │   Search (Top-K = 3)    │ │
│   │          ↓              │        │          ↓              │ │
│   │   CatBoost Classifier   │        │  Qwen 2.5 3B via Ollama │ │
│   │   vetiver_model.cbm     │        │  nomic-embed-text       │ │
│   │          ↓              │        │  storage/ (vector DB)   │ │
│   │  SUITABLE / UNSUITABLE  │        │          ↓              │ │
│   │  Confidence %           │        │  Grounded Answer        │ │
│   │  TreeSHAP Bar Chart     │        │  (from real papers)     │ │
│   └─────────────────────────┘        └─────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

---

## Model Performance (Phase 1 — First Data Batch)

Trained on **91 field success records + 16 engineered failure boundaries** using 5-Fold Stratified Cross-Validation:

| Metric | Score |
|:---|:---|
| Mean Accuracy | **89.70%** ± 4.62% |
| Mean Precision | **92.56%** ± 2.54% |
| Mean Recall | **95.56%** ± 4.16% |
| Mean F1-Score | **93.99%** ± 2.76% |

---

## Repository Structure

```
VETIVER/
│
├── app.py                           ← Main Streamlit application (run this)
├── config.py                        ← All configuration (models, paths, RAG settings)
├── train_model.py                   ← CatBoost training + 5-fold CV + SHAP generation
├── rag_engine.py                    ← LlamaIndex query engine + citation extractor
├── shap_utils.py                    ← TreeSHAP local attribution utility
├── requirements.txt                 ← All Python dependencies (pinned versions)
│
├── vetiver_model.cbm                ← Trained CatBoost model (commit-tracked)
│
├── data/
│   ├── vetiver_growth_data.csv      ← Training dataset (91 success records)
│   └── shap_summary.png             ← Global SHAP feature importance plot
│
├── knowledge_base/                  ← PDF / CSV papers for the RAG assistant
│   └── vetiver_growth_data.csv      ← Current RAG document source
│
├── storage/                         ← LlamaIndex vector index (auto-generated, git-ignored)
│
├── .streamlit/config.toml           ← Streamlit theme config
├── Dockerfile                       ← Docker image definition
├── docker-compose.yml               ← Docker Compose for JKUAT workstation deployment
├── .env.example                     ← Environment variable template (copy to .env)
│
├── TEAM_CHARTER.md                  ← Team roles, responsibilities, and Git workflow
├── data_collection_guide.html       ← Failure data collection schema (open in browser)
└── 01_baseline_catboost.ipynb       ← Exploratory ML notebook
```

---

## Setup: Getting the Project Running on Your Machine

### Prerequisites — Install these first

| Tool | Why needed | Download |
|:---|:---|:---|
| **Git** | Clone the repo and contribute code | https://git-scm.com |
| **Python 3.11 or 3.12** | Run the application | https://www.python.org/downloads/ |
| **Ollama** | Run Qwen 2.5 3B locally (Literature Assistant engine) | https://ollama.com |

> **Note on Python version:** The repo was developed on Python 3.14. Python 3.11 or 3.12 are recommended for team members as they have the best library compatibility with CatBoost and LlamaIndex.

---

### Step-by-Step Setup

#### Step 1 — Clone the repository

Open a terminal (PowerShell on Windows, Terminal on Mac/Linux) and run:

```bash
git clone https://github.com/brayo-kowino/VETIVER-ML-MODEL.git
cd VETIVER-ML-MODEL
```

---

#### Step 2 — Create and activate a virtual environment

This keeps all project packages isolated from your system Python.

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Mac / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` at the start of your terminal prompt once activated.

---

#### Step 3 — Install Python dependencies

```bash
pip install -r requirements.txt
```

This installs: `catboost`, `streamlit`, `pandas`, `llama-index`, `python-dotenv`, `matplotlib`, and all other dependencies at the exact pinned versions.

---

#### Step 4 — Set up environment variables

```bash
# Windows
copy .env.example .env

# Mac / Linux
cp .env.example .env
```

Open `.env` in any text editor. The defaults work fine for local development — only change them if your Ollama is on a different port or machine.

---

#### Step 5 — Set up Ollama and pull the AI models

Ollama runs the Literature Assistant engine locally on your machine.

**1. Start Ollama** (run this once; it stays running in the background):
```bash
ollama serve
```

**2. Pull the two required models** (one-time download, needs internet):
```bash
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
```

> `qwen2.5:3b` is about **2 GB**. `nomic-embed-text` is about **274 MB**.  
> Run these on a good internet connection (university WiFi or home broadband). Once downloaded, everything runs offline.

---

#### Step 6 — Add the dataset files

The dataset files (`.csv`) are **not stored in Git** (they are in `.gitignore`).  
Brian will share them via the team Google Drive. Place them as follows:

```
data/vetiver_growth_data.csv          ← Training data
knowledge_base/vetiver_growth_data.csv ← Same file, copy it here too for the RAG assistant
```

---

#### Step 7 — Train the model (or use the pre-trained one)

The trained model file `vetiver_model.cbm` is already in the repo — you can skip this step and go straight to running the app.

If you want to retrain from scratch (e.g. after adding new data):
```bash
python train_model.py
```

This runs 5-fold cross-validation, prints accuracy/F1 metrics, saves the model, and generates the SHAP summary plot.

---

#### Step 8 — Run the application

Make sure Ollama is running (`ollama serve` in a separate terminal), then:

```bash
streamlit run app.py
```

The app opens automatically at **http://localhost:8501** in your browser.

---

## Contributing — Git Workflow for the Team

> Read `TEAM_CHARTER.md` for the full governance rules. This is the short version.

### The Golden Rule
**Never push directly to `main`.** All changes go through a feature branch and a Pull Request reviewed by Brian.

### Daily workflow

```bash
# 1. Always start by pulling the latest code
git pull origin main

# 2. Create your branch (use your name + what you're doing)
git checkout -b data/alice-failure-records       # Data team example
git checkout -b feat/bob-whatsapp-webhook        # Technical team example

# 3. Make your changes, then stage and commit
git add .
git commit -m "data: add 15 failure records from Kenya and Ethiopia"

# 4. Push your branch to GitHub
git push origin data/alice-failure-records

# 5. Open a Pull Request on GitHub → assign Brian for review
```

### Branch naming convention

| Team | Prefix | Example |
|:---|:---|:---|
| Technical — new feature | `feat/` | `feat/whatsapp-api-webhook` |
| Technical — bug fix | `fix/` | `fix/shap-chart-layout` |
| Data team — adding records | `data/` | `data/kenya-failure-records` |
| Data team — cleaning | `clean/` | `clean/salinity-unit-normalisation` |
| Documentation | `docs/` | `docs/update-readme` |

---

## Running on JKUAT Workstations (Production)

For the full on-premise deployment at JKUAT using Docker:

```bash
# 1. Make sure Ollama is running on the host machine
ollama serve

# 2. Start the full stack with Docker Compose
docker compose up --build -d

# 3. Access the platform
# Open http://localhost:8501 in a browser on the same machine
# Or access from the network: http://<workstation-ip>:8501
```

To stop:
```bash
docker compose down
```

---

## Configuration Reference

All settings are in [`config.py`](config.py) and can be overridden via `.env`:

| Setting | Default | What it controls |
|:---|:---|:---|
| `OLLAMA_LLM_MODEL` | `qwen2.5:3b` | The language model for the Literature Assistant |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | The embedding model for vector search |
| `OLLAMA_NUM_THREADS` | `6` | CPU threads allocated to Ollama inference |
| `OLLAMA_NUM_PREDICT` | `-1` | Max tokens generated (-1 = no limit) |
| `OLLAMA_KEEP_ALIVE` | `24h` | How long the model stays loaded in RAM |
| `LLM_REQUEST_TIMEOUT` | `300` | Seconds before a Literature query times out |
| `RAG_SIMILARITY_TOP_K` | `3` | How many paper chunks are retrieved per query |
| `RAG_CHUNK_SIZE` | `512` | Token size of each indexed document chunk |

---

## Troubleshooting

| Problem | Fix |
|:---|:---|
| `ModuleNotFoundError` on startup | Run `pip install -r requirements.txt` with venv activated |
| `Literature Assistant: Offline` in sidebar | Run `ollama serve` in a separate terminal, then refresh the browser |
| `CatBoost model not found` error | Run `python train_model.py` to generate `vetiver_model.cbm` |
| `Knowledge base index not initialized` | Make sure `knowledge_base/` has at least one `.csv` or `.pdf` file |
| PowerShell `Activate.ps1` blocked | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once |
| Ollama model download is slow | Use a wired connection or university fibre. The 2 GB download is one-time only. |

---

## License

Academic research use — JKUAT. Licensing terms to be defined before public release.
