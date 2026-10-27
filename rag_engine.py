"""
RAG Engine module for Vetiver Chat.
Handles document loading, vector indexing with persistence,
query engine creation with system prompt, and source attribution.
"""
import os
import logging
from typing import Optional, Tuple, List, Dict, Any

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

import config

logger = logging.getLogger("vetiver.rag")


def configure_settings() -> None:
    """Configure global LlamaIndex settings with Ollama LLM and Embeddings."""
    Settings.llm = Ollama(
        model=config.OLLAMA_LLM_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        request_timeout=config.LLM_REQUEST_TIMEOUT,
        system_prompt=config.RAG_SYSTEM_PROMPT,
    )
    Settings.embed_model = OllamaEmbedding(
        model_name=config.OLLAMA_EMBED_MODEL,
        base_url=config.OLLAMA_BASE_URL,
    )
    Settings.node_parser = SentenceSplitter(
        chunk_size=config.RAG_CHUNK_SIZE,
        chunk_overlap=config.RAG_CHUNK_OVERLAP,
    )


def build_query_engine(
    persist_dir: str = config.STORAGE_DIR,
    knowledge_base_dir: str = config.KNOWLEDGE_BASE_DIR,
    similarity_top_k: int = config.RAG_SIMILARITY_TOP_K,
    force_rebuild: bool = False,
) -> Tuple[Optional[Any], Optional[VectorStoreIndex]]:
    """
    Load or build a persistent VectorStoreIndex and return (query_engine, index).
    If persist_dir contains an index and force_rebuild is False, loads from disk.
    Otherwise reads knowledge_base_dir, embeds, and saves to persist_dir.
    """
    configure_settings()

    index: Optional[VectorStoreIndex] = None
    docstore_path = os.path.join(persist_dir, "docstore.json")

    try:
        if os.path.exists(docstore_path) and not force_rebuild:
            logger.info("Loading persisted index from %s", persist_dir)
            storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
            index = load_index_from_storage(storage_context)
            logger.info("Persisted index successfully loaded.")
        else:
            if not os.path.isdir(knowledge_base_dir):
                logger.warning("Knowledge base directory not found: %s", knowledge_base_dir)
                return None, None

            logger.info("Building new index from directory: %s", knowledge_base_dir)
            documents = SimpleDirectoryReader(knowledge_base_dir).load_data()
            if not documents:
                logger.warning("No documents found in %s", knowledge_base_dir)
                return None, None

            index = VectorStoreIndex.from_documents(documents)
            os.makedirs(persist_dir, exist_ok=True)
            index.storage_context.persist(persist_dir=persist_dir)
            logger.info("Index created and saved to %s", persist_dir)

        if index is None:
            return None, None

        query_engine = index.as_query_engine(
            streaming=True,
            similarity_top_k=similarity_top_k,
        )
        return query_engine, index

    except Exception as exc:
        logger.error("Failed to build RAG query engine: %s", exc, exc_info=True)
        return None, None


def extract_source_metadata(response) -> List[Dict[str, Any]]:
    """
    Extract citation and score information from LlamaIndex response source_nodes.
    """
    sources = []
    if not hasattr(response, "source_nodes") or not response.source_nodes:
        return sources

    for idx, node_with_score in enumerate(response.source_nodes, 1):
        node = node_with_score.node
        meta = getattr(node, "metadata", {}) or {}
        file_name = meta.get("file_name", "knowledge_base document")
        score = getattr(node_with_score, "score", None)
        text = node.get_text() if hasattr(node, "get_text") else getattr(node, "text", "")
        
        sources.append({
            "index": idx,
            "file_name": file_name,
            "score": round(score, 3) if score is not None else None,
            "snippet": text.strip()[:240].replace("\n", " "),
        })

    return sources


def format_sources_markdown(sources: List[Dict[str, Any]]) -> str:
    """Format extracted sources into readable markdown for Streamlit expander."""
    if not sources:
        return ""

    lines = ["**Retrieved Literature Evidence:**\n"]
    for s in sources:
        score_str = f" *(relevance score: {s['score']:.3f})*" if s['score'] is not None else ""
        lines.append(f"- **Source {s['index']} ({s['file_name']}){score_str}:**")
        lines.append(f"  > \"{s['snippet']}...\"\n")

    return "\n".join(lines)
