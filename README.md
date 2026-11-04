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

> **This project uses Visual Studio Code (VS Code) as the standard IDE.** All terminal commands are run inside the VS Code integrated terminal. Follow every step in order.

---

### Step 1 — Install Required Software

Download and install all four tools before anything else:

| Tool | Purpose | Download |
|:---|:---|:---|
| **Git** | Clone the repo and contribute code | https://git-scm.com |
| **Python 3.11 or 3.12** | Run the application | https://www.python.org/downloads |
| **Ollama** | Runs the Qwen 2.5 3B model locally | https://ollama.com |
| **VS Code** | The project IDE — editor, terminal, Git, everything | https://code.visualstudio.com |

> Python 3.11 or 3.12 is recommended. The project was built on 3.14 but library compatibility is best on 3.11/3.12.

---

### Step 2 — Install VS Code Extensions

Open VS Code → click the **Extensions** icon on the left sidebar (or press `Ctrl+Shift+X`) → search and install:

| Extension | Publisher |
|:---|:---|
| `Python` | Microsoft |
| `Pylance` | Microsoft |
| `Jupyter` | Microsoft |
| `GitLens` | GitKraken |

---

### Step 3 — Clone the Repository into VS Code

1. Open VS Code
2. Press `Ctrl+Shift+P` to open the Command Palette
3. Type `Git: Clone` and press Enter
4. Paste the repository URL:
   ```
   https://github.com/brayo-kowino/VETIVER-ML-MODEL.git
   ```
5. Choose a folder on your computer to save it (e.g. your Desktop or Documents)
6. When VS Code asks *"Would you like to open the cloned repository?"* — click **Open**

---

### Step 4 — Open the Integrated Terminal

Inside VS Code, press **`Ctrl + Backtick`** (the key above Tab) to open the integrated terminal.  
All remaining commands are run here — do not use an external terminal.

---

### Step 5 — Create the Virtual Environment

In the VS Code terminal, run:

**Windows:**
```powershell
python -m venv venv
```

**Mac / Linux:**
```bash
python3 -m venv venv
```

---

### Step 6 — Select the venv as Your Python Interpreter

1. Press `Ctrl+Shift+P`
2. Type `Python: Select Interpreter` and press Enter
3. Select the option that shows `venv` in its path — it will look like:
   - Windows: `.\venv\Scripts\python.exe`
   - Mac/Linux: `./venv/bin/python`

> If `venv` does not appear in the list, click **+ Enter interpreter path** and navigate to it manually.

After selecting, VS Code will automatically activate the venv in every new terminal you open in this project.

---

### Step 7 — Install Project Dependencies

In the VS Code terminal:

```bash
pip install -r requirements.txt
```

This installs `catboost`, `streamlit`, `pandas`, `llama-index`, `python-dotenv`, `matplotlib`, and all other dependencies at the exact pinned versions. Wait for it to complete fully.

---

### Step 8 — Set Up Environment Variables

In the VS Code terminal:

```powershell
# Windows
copy .env.example .env
```
```bash
# Mac / Linux
cp .env.example .env
```

The defaults in `.env` work fine for local development. No changes needed unless Ollama is on a different port.

---

### Step 9 — Pull the Ollama AI Models (one-time, needs internet)

In the VS Code terminal, run these two commands:

```bash
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
```

> `qwen2.5:3b` is approximately **2 GB**. `nomic-embed-text` is approximately **274 MB**.  
> Run this on a good internet connection. Once downloaded, everything runs fully offline.

---

### Step 10 — Run the Application

**1. Start Ollama** — open a second VS Code terminal (click the `+` icon in the terminal panel) and run:
```bash
ollama serve
```
Leave this terminal running in the background.

**2. Start the app** — in your first terminal, run:
```bash
streamlit run app.py
```

The app opens automatically in your browser at **http://localhost:8501**.

---

## Retraining the Model (after new data is added)

The trained model `vetiver_model.cbm` is already in the repo — you can run the app without retraining.

When Brian commits new data and you have pulled it, retrain with:

```bash
python train_model.py
```

This runs 5-fold cross-validation, prints accuracy/F1 metrics, saves the updated model, and regenerates the SHAP summary plot.

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
