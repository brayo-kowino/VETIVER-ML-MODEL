import os
import json
import logging
import httpx
import streamlit as st
from catboost import CatBoostClassifier

import config
from rag_engine import build_query_engine

logger = logging.getLogger("vetiver")


@st.cache_data(ttl=30, show_spinner=False)
def check_ollama_health() -> bool:
    """Check if the local Ollama is reachable."""
    try:
        resp = httpx.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=4)
        resp.raise_for_status()
        return True
    except Exception as exc:
        logger.warning("Ollama health-check failed: %s", exc)
        return False


@st.cache_resource
def load_catboost(mtime: float = 0.0):
    """Load the trained CatBoost classifier from disk, invalidating cache on model file change."""
    try:
        model = CatBoostClassifier()
        model.load_model(config.MODEL_PATH)
        logger.info("CatBoost model loaded from %s (mtime: %s)", config.MODEL_PATH, mtime)
        return model
    except FileNotFoundError:
        logger.error("Model file not found at %s", config.MODEL_PATH)
        return None
    except Exception as exc:
        logger.error("Failed to load CatBoost model: %s", exc)
        return None


@st.cache_data
def load_regional_hotspots(mtime: float = 0.0):
    """Load regional hotspot GeoJSON/JSON registry from disk."""
    hotspot_path = os.path.join(config.DATA_DIR, "regional_hotspots.json")
    if os.path.exists(hotspot_path):
        with open(hotspot_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


@st.cache_resource
def initialize_rag(kb_mtime: float = 0.0, ollama_online: bool = True):
    """Build or reload the LlamaIndex query engine and vector store index."""
    if not ollama_online:
        logger.error("Ollama offline — skipping RAG init.")
        return None, None
    query_engine, index = build_query_engine(
        persist_dir=config.STORAGE_DIR,
        knowledge_base_dir=config.KNOWLEDGE_BASE_DIR,
        similarity_top_k=config.RAG_SIMILARITY_TOP_K,
    )
    return query_engine, index
