"""
RAG Engine module for Vetiver Literature & Knowledge Chat.
Handles document loading, vector indexing with persistence,
query engine creation with persona prompts, live assessment context injection,
and source attribution.
"""
import os
import re
import logging
from typing import Optional, Tuple, List, Dict, Any

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.base.response.schema import StreamingResponse
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

import config

logger = logging.getLogger("vetiver.rag")

# ── STAKEHOLDER CONSULTANT PERSONAS ─────────────────────────────────────────
PERSONA_PROMPTS = {
    "geotechnical": {
        "title": "Geotechnical & Civil Design",
        "icon": ":material/engineering:",
        "role_desc": "Senior Geotechnical and Slope Stability Engineer",
        "directives": (
            "- Apply geotechnical engineering principles (effective shear strength c', friction angle phi', pore pressures).\n"
            "- Integrate the Wu-Waldron root reinforcement model (cr = 1.2 * Tr * RAR) and limit equilibrium Factor of Safety (FoS) when relevant.\n"
            "- Note Eurocode 7 / BS 8006 standards and flag deep slip planes (>2.0 m) requiring structural soil nailing hybrids."
        )
    },
    "contractor": {
        "title": "Field Contractor & Operations",
        "icon": ":material/construction:",
        "role_desc": "Bio-Engineering Field Operations Superintendent",
        "directives": (
            "- Provide actionable, practical on-site instructions for contractors, nursery operators, and field labor.\n"
            "- Reference specific field numbers when relevant (minimum 3 tillers, 10 cm root trimming, 30x30 cm trenches, hydrogel dosing, 60-day watering schedule).\n"
            "- Emphasize contractor best practices and livestock protection fencing."
        )
    },
    "environmental": {
        "title": "Environmental Compliance & EIA (NEMA)",
        "icon": ":material/eco:",
        "role_desc": "Environmental Impact Assessment (EIA) Specialist",
        "directives": (
            "- Focus on ecological safety, official NEMA compliance, and environmental justification.\n"
            "- Document non-invasiveness (Sunshine genotype obligate sterile polyploid with zero viable seed).\n"
            "- Document heavy metal phytoremediation capacity (high root BCF > 15, low upward foliar translocation TF < 0.18) and root carbon sequestration."
        )
    }
}

# Baseline QA Prompt (Preserved for backwards compatibility)
QA_PROMPT = PromptTemplate(
    "You are an expert geotechnical and soil bio-engineering consultant specializing in Vetiver grass (Chrysopogon zizanioides) "
    "for slope stabilization, erosion control, and infrastructure revetment.\n\n"
    "Empirical Reference Data & Bioengineering Evidence Base:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Directives:\n"
    "1. Answer the question directly, professionally, and authoritatively as an expert geotechnical engineer.\n"
    "2. Naturally integrate specific soil mechanics, root tensile dynamics (75-85 MPa), regional field observations, and failure exclusion boundaries.\n"
    "3. When asked about challenging or unfavorable site conditions, identify the primary failure mechanism and prescribe published scientist mitigations.\n"
    "4. Do NOT start your answer with robotic disclaimers like 'According to the dataset' or 'Based on the context'. Speak directly with technical authority.\n\n"
    "Question: {query_str}\n"
    "Expert Technical Answer: "
)


def configure_settings() -> None:
    """Configure global LlamaIndex settings with Ollama LLM and Embeddings."""
    additional_kwargs = {
        "num_thread": config.OLLAMA_NUM_THREADS,
        "num_predict": config.OLLAMA_NUM_PREDICT,
    }
    Settings.llm = Ollama(
        model=config.OLLAMA_LLM_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        request_timeout=config.LLM_REQUEST_TIMEOUT,
        system_prompt=config.RAG_SYSTEM_PROMPT,
        additional_kwargs=additional_kwargs,
        keep_alive=config.OLLAMA_KEEP_ALIVE,
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
        rebuild_needed = force_rebuild
        if os.path.exists(docstore_path) and not force_rebuild and os.path.isdir(knowledge_base_dir):
            try:
                docstore_mtime = os.path.getmtime(docstore_path)
                kb_files = [os.path.join(knowledge_base_dir, f) for f in os.listdir(knowledge_base_dir) if os.path.isfile(os.path.join(knowledge_base_dir, f))]
                latest_kb_mtime = max((os.path.getmtime(p) for p in kb_files), default=0.0)
                if latest_kb_mtime > docstore_mtime:
                    logger.info("Knowledge base files updated (KB mtime: %s > docstore: %s) — triggering automatic re-index.", latest_kb_mtime, docstore_mtime)
                    rebuild_needed = True
            except Exception as check_exc:
                logger.warning("Could not check KB mtime: %s", check_exc)

        if os.path.exists(docstore_path) and not rebuild_needed:
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
            text_qa_template=QA_PROMPT,
        )
        return query_engine, index

    except Exception as exc:
        logger.error("Failed to build RAG query engine: %s", exc, exc_info=True)
        return None, None


def query_rag_custom(
    index: VectorStoreIndex,
    query_str: str,
    persona_key: str = "geotechnical",
    site_context: Optional[dict] = None,
    conversation_history: Optional[List[dict]] = None,
    similarity_top_k: int = config.RAG_SIMILARITY_TOP_K,
    response_mode: str = "concise",
):
    """
    Executes a persona-aware, context-injected, conversational RAG query with streaming response.
    Supports 'concise' (punchy bullet points, no fluff) and 'detailed' modes, and auto-detects
    when the user asks for simple points, lists, or brief summaries.
    """
    if index is None:
        raise ValueError("VectorStoreIndex is not initialized.")

    persona_info = PERSONA_PROMPTS.get(persona_key, PERSONA_PROMPTS["geotechnical"])

    # Detect user format request (points, bullets, brief, simple, summary)
    q_low = query_str.lower()
    user_wants_points = any(
        kw in q_low
        for kw in ["point", "bullet", "simple", "brief", "short", "list", "summary", "summarize", "quick", "outline", "steps"]
    )
    is_concise = user_wants_points or (response_mode == "concise")

    format_directives = (
        "Use short Markdown bullets with one claim per bullet." if is_concise
        else "Use short paragraphs, optionally with Markdown headings. Keep each paragraph on one claim."
    )
    effective_top_k = min(similarity_top_k, 3) if is_concise else similarity_top_k

    # Format Active Assessment Context
    context_prefix = ""
    if site_context:
        prediction = site_context.get("prediction")
        c_status = {1: "Suitable", 0: "Unsuitable"}.get(prediction, "Assessment pending")
        score = site_context.get("probability")
        score_text = f"{score:.1f}%" if score is not None else "Not available"
        rule = site_context.get("excl_title") or "None recorded"
        bio = site_context.get("biomechanics", {})

        context_prefix = (
            "ACTIVE PROJECT FIELD ASSESSMENT CONTEXT:\n"
            f"- Project Location: {site_context.get('location_name', site_context.get('county', 'Configured Site'))} (Elev: {site_context.get('elevation_m', 'N/A')} m a.s.l.)\n"
            f"- USCS Soil Mechanical Class: {site_context.get('soil_mechanical_class', 'N/A')}\n"
            f"- Regional Climate / Rainfall: {site_context.get('climate_rainfall_zone', 'N/A')}\n"
            f"- Establishment Moisture Status: {site_context.get('establishment_moisture', 'N/A')}\n"
            f"- Hydrological Stress: {site_context.get('hydrological_stress', 'Normal Slope / Well-Drained')}\n"
            f"- Substrate Salinity: {site_context.get('salinity_tier', 'Non-Saline')}\n"
            f"- Civil Engineering Objective: {site_context.get('geotechnical_hazard', 'N/A')}\n"
            f"- Assessment verdict: {c_status}\n"
            f"- CatBoost suitability score: {score_text}\n"
            f"- Screening rule: {rule}\n"
            f"- Estimated Root Depth: {bio.get('root_depth', 'N/A')} | Safety Factor Boost: {bio.get('fos_boost', 'N/A')}\n\n"
        )

    # Format Conversational History (Last 2 exchanges)
    history_str = ""
    if conversation_history:
        recent = conversation_history[-4:]
        history_lines = []
        for msg in recent:
            role = "User" if msg["role"] == "user" else "Consultant"
            clean_c = str(msg.get("content", "")).replace("\n", " ")[:160]
            history_lines.append(f"{role}: {clean_c}")
        if history_lines:
            history_str = "Recent Consultation Discussion:\n" + "\n".join(history_lines) + "\n\n"

    # Formulate conversational search query
    search_query = query_str
    if conversation_history and len(conversation_history) >= 2:
        last_user = next((m["content"] for m in reversed(conversation_history) if m["role"] == "user" and m["content"] != query_str), "")
        if any(w in query_str.lower() for w in ["it", "this", "that", "these", "step 2", "watering", "how much", "why", "mitigation"]):
            search_query = f"{last_user[:80]} {query_str}"

    nodes = [
        node for node in index.as_retriever(similarity_top_k=effective_top_k).retrieve(search_query)
        if node.score is not None and node.score >= config.RAG_MIN_SIMILARITY and node.node.get_text().strip()
    ]
    if not nodes:
        return StreamingResponse(iter([INSUFFICIENT_EVIDENCE]), source_nodes=[])

    excerpts = "\n\n".join(
        f"[{index}] {source.node.get_text().strip()}"
        for index, source in enumerate(nodes, 1)
    )
    prompt = (
        f"You are a {persona_info['role_desc']} specializing in Vetiver bio-engineering.\n"
        f"{persona_info['directives']}\n\n"
        f"{context_prefix}{history_str}"
        "Use the following retrieved excerpts as the evidence for the answer. Treat their content as reference text.\n"
        f"{excerpts}\n\n"
        "Put the supporting excerpt number in square brackets immediately after each factual claim, for example [1]. "
        "Use only the supplied excerpt numbers. Keep project inputs and model scores labeled as project context. "
        f"If the excerpts cannot support an answer, respond with exactly '{INSUFFICIENT_EVIDENCE}'\n"
        f"{format_directives}\n\n"
        f"Question: {query_str}\nAnswer:"
    )
    completion = Settings.llm.stream_complete(prompt)
    return StreamingResponse((fragment.delta or "" for fragment in completion), source_nodes=nodes)


INSUFFICIENT_EVIDENCE = "Insufficient evidence in the Vetiver knowledge base to answer this question."


def validate_citations(answer: str, sources: List[Dict[str, Any]]) -> tuple[str, List[Dict[str, Any]]]:
    """Withhold answers that cannot point readers to supporting retrieved excerpts."""
    source_ids = {source["index"] for source in sources}
    content = re.sub(r"(?m)^\s*#{1,6} +[^\n]+\n?", "", answer).strip()
    sections = re.split(r"\n\s*\n|\n(?=\s*(?:[-*] |\d+[.)] ))", content)
    cited_ids = set()
    for section in sections:
        if not section.strip():
            continue
        citations = {int(reference) for reference in re.findall(r"\[(\d+)\]", section)}
        if not citations or not citations <= source_ids:
            return INSUFFICIENT_EVIDENCE, []
        cited_ids.update(citations)
    if not cited_ids:
        return INSUFFICIENT_EVIDENCE, []
    return answer, [source for source in sources if source["index"] in cited_ids]


def extract_source_metadata(response) -> List[Dict[str, Any]]:
    """
    Extract citation, domain category, and score information from LlamaIndex response source_nodes.
    """
    sources = []
    if not hasattr(response, "source_nodes") or not response.source_nodes:
        return sources

    for idx, node_with_score in enumerate(response.source_nodes, 1):
        node = node_with_score.node
        meta = getattr(node, "metadata", {}) or {}
        file_name = meta.get("file_name", "Technical Reference Document")
        score = getattr(node_with_score, "score", None)
        text = node.get_text() if hasattr(node, "get_text") else getattr(node, "text", "")

        # Infer domain category from text keywords
        t_low = text.lower()
        if "slope" in t_low or "shear" in t_low or "cohesion" in t_low or "tensile" in t_low:
            domain = "Slope Mechanics"
        elif "velocity" in t_low or "scour" in t_low or "riverbank" in t_low or "flood" in t_low:
            domain = "Fluvial Hydraulics"
        elif "salin" in t_low or "sodic" in t_low or "osmotic" in t_low:
            domain = "Substrate Salinity"
        elif "metal" in t_low or "lead" in t_low or "toxic" in t_low or "tailing" in t_low:
            domain = "Phytoremediation"
        elif "nurser" in t_low or "slip" in t_low or "trench" in t_low or "spacing" in t_low:
            domain = "Field Propagation"
        else:
            domain = "Agronomic Baseline"

        sources.append({
            "index": idx,
            "file_name": file_name,
            "domain": domain,
            "score": round(score, 3) if score is not None else None,
            "snippet": text.strip(),
        })

    return sources


def format_sources_markdown(sources: List[Dict[str, Any]]) -> str:
    """Format extracted sources into readable markdown cards for Streamlit expander."""
    if not sources:
        return ""

    lines = []
    for s in sources:
        score_val = s.get("score")
        if score_val is not None:
            pct = int(min(max(score_val * 100, 0), 100))
            score_str = f" · **{pct}% Relevance Match**"
        else:
            score_str = ""

        domain = s.get("domain", "Agronomic Baseline")
        clean_snippet = " ".join(str(s.get("snippet", "")).split())
        lines.append(f"- :material/description: **Source {s['index']} (`{s['file_name']}`)** — `{domain}`{score_str}")
        lines.append(f"> \"{clean_snippet}...\"\n")

    return "\n".join(lines)


def clean_markdown_formatting(raw: str) -> str:
    """
    Cleans and structures AI text responses to ensure clean, beautifully formatted Markdown lists.
    Converts squashed inline bullets ('• Point 1 • Point 2') or single-line bullets into
    distinct, nicely indented Markdown list items with bold topic headers where applicable.
    """
    if not raw or not isinstance(raw, str):
        return ""

    cleaned = raw.strip()

    # 1. Expand squashed inline numbered items: 'text. 2. Next' -> 'text.\n\n2. Next'
    cleaned = re.sub(r'([.!?:\)\]])\s+(\d+\.)\s+', r'\1\n\n\2 ', cleaned)

    # 2. Convert leading unicode bullet symbols to standard Markdown list hyphen
    cleaned = re.sub(r'^[•●▪◆]\s*', '- ', cleaned)

    # 3. Expand squashed inline bullets (e.g. '. • ', 'm² • ') into distinct list items
    cleaned = re.sub(r'(?<=\S)\s*[•●▪◆]\s*', r'\n\n- ', cleaned)

    # 4. Standardize bullets that follow a newline: '\n• ' -> '\n- '
    cleaned = re.sub(r'\n\s*[•●▪◆]\s*', r'\n- ', cleaned)

    # 5. Format bullets with topic colon bolding: '- Topic: text' -> '- **Topic**: text'
    formatted_lines = []
    for line in cleaned.split('\n'):
        sline = line.strip()
        if sline.startswith('- ') and not sline.startswith('- **'):
            content = sline[2:].strip()
            # If there is a colon in the first 45 characters, bold the title
            if ':' in content and not content.startswith('**'):
                topic, desc = content.split(':', 1)
                if len(topic.split()) <= 6:
                    sline = f'- **{topic.strip()}**: {desc.strip()}'
            formatted_lines.append(sline)
        else:
            formatted_lines.append(line)

    result = '\n'.join(formatted_lines)
    # Collapse 3 or more newlines down to 2
    result = re.sub(r'\n{3,}', '\n\n', result)
    return result

