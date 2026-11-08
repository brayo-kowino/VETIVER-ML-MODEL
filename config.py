"""
Centralized configuration for the Vetiver project.
Reads from environment variables with sensible defaults.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
KNOWLEDGE_BASE_DIR = os.path.join(os.path.dirname(__file__), "knowledge_base")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "vetiver_model.cbm")
STORAGE_DIR = os.path.join(os.path.dirname(__file__), "storage")

# ── Ollama / LLM ──────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "qwen2.5:3b")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
LLM_REQUEST_TIMEOUT = float(os.getenv("LLM_REQUEST_TIMEOUT", "300"))
OLLAMA_NUM_THREADS = int(os.getenv("OLLAMA_NUM_THREADS", "6"))  # Utilize 6 CPU threads
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "768"))  # Balanced token generation window
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "24h")  # Keep model resident in RAM

# ── RAG ────────────────────────────────────────────────
RAG_SIMILARITY_TOP_K = int(os.getenv("RAG_SIMILARITY_TOP_K", "4"))
RAG_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "512"))
RAG_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "64"))
RAG_SYSTEM_PROMPT = os.getenv(
    "RAG_SYSTEM_PROMPT",
    (
        "You are an expert geotechnical and soil bio-engineering consultant specializing in Vetiver grass "
        "(Chrysopogon zizanioides) for slope stabilization, erosion control, and infrastructure revetment.\n\n"
        "Core Directives:\n"
        "1. Deliver direct, concise, and scientifically rigorous answers tailored specifically to what the user asks.\n"
        "2. Strictly obey the user's requested response format (e.g. simple bullet points, brief summary, numbered steps).\n"
        "3. Avoid unnecessary walls of text, unsolicited generic introductions, or rambling background dissertations.\n"
        "4. When geotechnical mechanics or failure thresholds are relevant, reference specific parameters (c', phi', Wu-Waldron cr, Eurocode 7) concisely.\n"
        "5. Speak directly with engineering authority; never use robotic disclaimers like 'according to the dataset'."
    ),
)

# ── CatBoost Features & Physics Constraints ──────────────
CANONICAL_DATASET_PATH = os.path.join(DATA_DIR, "vetiver_canonical_ml_dataset.csv")

# Full feature list passed to model
MODEL_FEATURES = [
    "soil_mechanical_class",
    "climate_rainfall_zone",
    "establishment_moisture",
    "hydrological_stress",
    "est_salinity_ec_ds_m",
    "geotechnical_hazard",
]

# Categorical columns
CAT_FEATURES = [
    "soil_mechanical_class",
    "climate_rainfall_zone",
    "establishment_moisture",
    "hydrological_stress",
    "geotechnical_hazard",
]

# Numerical columns
NUM_FEATURES = [
    "est_salinity_ec_ds_m",
]

# Physics-informed monotonicity: Higher salinity can only decrease suitability or stay flat
MONOTONE_CONSTRAINTS = {
    "est_salinity_ec_ds_m": -1,
}

# Salinity tier to numeric ECe mapping
SALINITY_TIER_TO_ECE = {
    "Non-Saline (< 2 dS/m)": 0.5,
    "Slight to Moderate (2–8 dS/m)": 4.0,
    "High Saline (8–16 dS/m)": 10.0,
    "Extreme Hypersaline (> 16 dS/m)": 19.5,
}

# Compatibility aliases
CANONICAL_FEATURES = MODEL_FEATURES
CATEGORICAL_FEATURES = MODEL_FEATURES
