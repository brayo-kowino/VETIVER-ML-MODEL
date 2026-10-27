# 🌿 Vetiver — Geotechnical Suitability & RAG Chat

A hybrid ML + RAG application for **Vetiver grass** site-suitability analysis.  
It combines a **CatBoost classifier** (tabular prediction) with a **Qwen 2.5 3B LLM** (knowledge-base Q&A via Ollama) inside a single **Streamlit** web interface.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Streamlit  UI                      │
│                                                     │
│  ┌──────────────────┐   ┌────────────────────────┐  │
│  │  Tab 1: Analytics │   │  Tab 2: Vetiver Chat  │  │
│  │   (CatBoost)      │   │   (RAG / LlamaIndex)  │  │
│  └────────┬─────────┘   └──────────┬─────────────┘  │
│           │                        │                 │
│   vetiver_model.cbm        knowledge_base/*.csv      │
│                                    │                 │
│                          ┌─────────▼──────────┐      │
│                          │   Ollama (local)   │      │
│                          │  qwen2.5:3b (LLM)  │      │
│                          │  nomic-embed-text   │      │
│                          └────────────────────┘      │
└─────────────────────────────────────────────────────┘
```

## Key Components

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Tabular ML** | CatBoost | Predict site suitability from soil, climate & salinity features |
| **RAG Pipeline** | LlamaIndex + Ollama | Answer natural-language questions against engineering literature |
| **LLM** | Qwen 2.5 3B (via Ollama) | Generate answers grounded in retrieved context |
| **Embeddings** | nomic-embed-text (via Ollama) | Vector-embed knowledge base documents for retrieval |
| **UI** | Streamlit | Two-tab web app (analytics + chat) |

## Dataset

91 rows of real-world Vetiver grass growth observations sourced from peer-reviewed papers.  
Features: `Location`, `Soil type/structure`, `Soil density`, `Climate`, `Watering regime`, `Salinity`, `Duration since planting`, `Root length`, `Application`, `Reference`.

The same CSV lives in **two** places (by design):

| Path | Used by |
|------|---------|
| `data/vetiver_growth_data.csv` | Notebook / model training |
| `knowledge_base/vetiver_growth_data.csv` | LlamaIndex RAG ingestion |

---

## Prerequisites

| Requirement | Verified |
|-------------|----------|
| **Python 3.14+** | ✅ |
| **Ollama** installed & running | ✅ |
| `ollama pull qwen2.5:3b` | ✅ |
| `ollama pull nomic-embed-text` | ✅ |

## Quick Start

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd VETIVER

# 2. Create & activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Pull the Ollama models (if not already done)
ollama pull qwen2.5:3b
ollama pull nomic-embed-text

# 5. (Optional) Copy and edit the env file
copy .env.example .env       # Windows
# cp .env.example .env       # macOS / Linux

# 6. Run the Streamlit app
streamlit run app.py
```

The app opens at **http://localhost:8501**.

---

## Project Structure

```
VETIVER/
├── data/
│   ├── vetiver_growth_data.csv      # Training dataset (91 rows)
│   └── shap_summary.png             # Global SHAP feature importance plot
├── knowledge_base/
│   └── vetiver_growth_data.csv      # RAG document source
├── storage/                         # Persisted LlamaIndex vector store (git-ignored)
├── catboost_info/                   # CatBoost training logs (git-ignored)
├── 01_baseline_catboost.ipynb       # Exploratory model notebook
├── train_model.py                   # Automated 5-Fold CV training & SHAP generation
├── shap_utils.py                    # Real-time TreeSHAP local attribution utility
├── rag_engine.py                    # Modular LlamaIndex query engine & citation formatter
├── app.py                           # Streamlit application (main entry point)
├── config.py                        # Centralized project configuration
├── vetiver_model.cbm                # Trained CatBoost model artifact
├── requirements.txt                 # Python dependencies (pinned)
├── .env.example                     # Environment variable template
├── .gitignore                       # Git ignore rules
└── README.md                        # This file
```

## Configuration

All tuneable parameters are in [`config.py`](config.py), backed by env vars defined in [`.env.example`](.env.example):

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_LLM_MODEL` | `qwen2.5:3b` | Ollama LLM model name |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Ollama embedding model |
| `LLM_REQUEST_TIMEOUT` | `300` | LLM request timeout (seconds) |
| `RAG_SIMILARITY_TOP_K` | `3` | Top-k retrieved chunks |
| `RAG_CHUNK_SIZE` | `512` | Token chunk size |
| `RAG_CHUNK_OVERLAP` | `64` | Token chunk overlap |

## Re-training the Model

To execute the 5-fold stratified cross-validation and recompute SHAP attributions:

```bash
python train_model.py
```

**Cross-Validation Results (5 Folds):**
- **Mean Accuracy:** 89.70% (± 4.62%)
- **Mean Precision:** 92.56% (± 2.54%)
- **Mean Recall:** 95.56% (± 4.16%)
- **Mean F1-Score:** 93.99% (± 2.76%)

---

## License

*To be defined.*
