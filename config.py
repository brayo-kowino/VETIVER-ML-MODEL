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
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "-1"))  # -1 = no token limit; model generates until natural completion
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "24h")  # Keep model resident in RAM

# ── RAG ────────────────────────────────────────────────
RAG_SIMILARITY_TOP_K = int(os.getenv("RAG_SIMILARITY_TOP_K", "3"))
RAG_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "512"))
RAG_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "64"))
RAG_SYSTEM_PROMPT = os.getenv(
    "RAG_SYSTEM_PROMPT",
    (
        "You are an expert geotechnical and bio-engineering specialist in Vetiver grass "
        "(Chrysopogon zizanioides) for slope stabilization, erosion control, and soil bio-engineering.\n\n"
        "Guidelines:\n"
        "1. Provide direct, authoritative, and scientifically rigorous explanations.\n"
        "2. Naturally incorporate specific soil mechanics, root tensile dynamics, site trials, and regional observations into your answers.\n"
        "3. Never say 'according to the dataset', 'based on the provided context', or similar robotic disclaimers. Speak directly as an expert in the field.\n"
        "4. Structure your insights clearly with practical engineering advice."
    ),
)

# ── CatBoost Features ─────────────────────────────────
CATEGORICAL_FEATURES = [
    "Soil type/structure",
    "climate",
    "watering regime",
    "Application",
    "salinity",
]
