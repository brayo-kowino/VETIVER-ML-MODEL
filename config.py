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

# ── RAG ────────────────────────────────────────────────
RAG_SIMILARITY_TOP_K = int(os.getenv("RAG_SIMILARITY_TOP_K", "3"))
RAG_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "512"))
RAG_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "64"))
RAG_SYSTEM_PROMPT = os.getenv(
    "RAG_SYSTEM_PROMPT",
    (
        "You are an expert geotechnical and bio-engineering assistant specializing in Vetiver grass "
        "(Chrysopogon zizanioides) for slope stabilization, erosion control, and soil conservation.\n\n"
        "Instructions:\n"
        "1. Strictly ground your answer in the provided context and empirical data from the knowledge base.\n"
        "2. When discussing site conditions, cite specific locations, soil types, root depths, and literature references from the data.\n"
        "3. If the context does not contain adequate evidence to answer the question, honestly state: "
        "'I do not have sufficient empirical data in the knowledge base to answer this question reliably.'\n"
        "4. Be concise, professional, and scientifically rigorous."
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
