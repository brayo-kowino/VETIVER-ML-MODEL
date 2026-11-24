import re
from pathlib import Path

import config
from services.model_service import ChatUnavailable, check_ollama_models, initialize_rag


def conversational_reply(question: str) -> str | None:
    """Recognize complete social messages so a greeting cannot swallow a question."""
    text = " ".join(re.findall(r"[^\W_]+(?:['’][^\W_]+)?", question.casefold())).replace("’", "'")
    greeting = (
        r"(?:hi|hello|hey|hiya|greetings|good (?:morning|afternoon|evening|day))"
        r"(?: there| vetiver(?: ai)?)?"
    )
    check_in = r"(?:how (?:are you(?: doing)?|is it going)|how's it going)"
    if re.fullmatch(rf"{greeting}(?: {check_in})?|{check_in}", text):
        return "Hi! How can I help you today?"
    if re.fullmatch(r"(?:thanks|thank you)(?: so much| very much| vetiver(?: ai)?)?", text):
        return "You're welcome!"
    if text in {"who are you", "what can you do", "how can you help", "help"}:
        return (
            "I'm Vetiver AI. I can help you explore Vetiver research, slope stabilization, "
            "erosion control, and the conditions at your project site."
        )
    return None


def answer_research_question(
    question: str,
    *,
    response_format: str,
    site_context: dict | None,
    history: list[dict],
) -> tuple[str, list[dict]]:
    check_ollama_models()

    from ollama import ResponseError

    from rag_engine import (
        clean_markdown_formatting,
        extract_source_metadata,
        query_rag_custom,
        validate_citations,
    )

    knowledge_base = Path(config.KNOWLEDGE_BASE_DIR)
    kb_mtime = max(
        (path.stat().st_mtime for path in knowledge_base.glob("*") if path.is_file()),
        default=0.0,
    )
    _, index = initialize_rag(kb_mtime)
    try:
        response = query_rag_custom(
            index=index,
            query_str=question,
            persona_key="general",
            site_context=site_context,
            conversation_history=history,
            response_mode=response_format,
        )
        answer = clean_markdown_formatting("".join(response.response_gen))
    except ResponseError as exc:
        if exc.status_code != 404:
            raise
        raise ChatUnavailable(
            "Ollama couldn't find the requested model. Check the configured chat and embedding models."
        ) from exc
    return validate_citations(answer, extract_source_metadata(response))
