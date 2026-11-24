import os
import json
import logging
import httpx
import streamlit as st
from catboost import CatBoostClassifier

import config

logger = logging.getLogger("vetiver")


class ChatUnavailable(RuntimeError):
    """An actionable error the chat can show to its user."""


def check_ollama_models() -> None:
    """Check that the configured server exposes both models the research chat needs."""
    def model_tag(name: str) -> str:
        name = name.removeprefix("registry.ollama.ai/").removeprefix("library/")
        return name if ":" in name.rsplit("/", 1)[-1] else f"{name}:latest"

    try:
        response = httpx.get(f"{config.OLLAMA_BASE_URL.rstrip('/')}/api/tags", timeout=4)
        response.raise_for_status()
        available_models = response.json()["models"]
        if not isinstance(available_models, list):
            raise ValueError("Invalid Ollama model list")
        available = {model_tag(model.get("name") or model["model"]) for model in available_models}
    except httpx.RequestError as exc:
        raise ChatUnavailable(
            "Vetiver AI couldn't reach Ollama. Start the Ollama server and try again."
        ) from exc
    except (httpx.HTTPStatusError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise ChatUnavailable(
            "Vetiver AI couldn't read the Ollama model list. Check the configured Ollama URL and try again."
        ) from exc

    required = dict.fromkeys((config.OLLAMA_LLM_MODEL, config.OLLAMA_EMBED_MODEL))
    missing = [f"`{name}`" for name in required if model_tag(name) not in available]
    if missing:
        raise ChatUnavailable(f"Ollama is missing {', '.join(missing)}. Install the listed models and try again.")


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


@st.cache_resource(show_spinner=False)
def initialize_rag(kb_mtime: float = 0.0):
    """Build or reload the LlamaIndex query engine and vector store index."""
    from rag_engine import build_query_engine

    query_engine, index = build_query_engine(
        persist_dir=config.STORAGE_DIR,
        knowledge_base_dir=config.KNOWLEDGE_BASE_DIR,
        similarity_top_k=config.RAG_SIMILARITY_TOP_K,
    )
    if index is None:
        raise ChatUnavailable(
            "The Vetiver knowledge base couldn't load. Check the knowledge base files and server logs, then try again."
        )
    return query_engine, index
