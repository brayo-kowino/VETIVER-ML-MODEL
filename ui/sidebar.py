import os
from datetime import datetime
import streamlit as st

import config
from services.report_service import export_chat_markdown


def render_sidebar(cb_model, ollama_online: bool, query_engine) -> None:
    """Render friendly sidebar readiness, guide, and collapsed developer telemetry."""
    with st.sidebar:
        st.subheader(":material/eco: Vetiver AI Platform")
        st.caption("Geotechnical Site Assessment · 3D GIS Studio · AI Consultation ")
        st.divider()

        st.markdown("##### System Readiness")
        if cb_model:
            ml_card = ("ready", "check_circle", "ML Model", "Ready")
        else:
            ml_card = ("offline", "error", "ML Model", "Offline")

        if ollama_online and query_engine:
            ai_card = ("ready", "smart_toy", "Vetiver AI", "Ready")
        elif ollama_online:
            ai_card = ("indexing", "sync", "Vetiver AI", "Indexing")
        else:
            ai_card = ("offline", "power_off", "Vetiver AI", "Offline")

        def readiness_card(card):
            tone, icon, title, status = card
            colors = {
                "ready": ("#DCFCE7", "#166534", "#15803D"),
                "indexing": ("#FEF3C7", "#92400E", "#D97706"),
                "offline": ("#FEE2E2", "#991B1B", "#DC2626"),
            }
            background, text_color, icon_color = colors[tone]
            return (
                f'<div class="readiness-card" style="background:{background};">'
                f'<span class="readiness-icon" style="color:{icon_color};">'
                f'<span class="material-symbols-rounded" style="font-size:18px;">{icon}</span>'
                '</span>'
                f'<span class="readiness-copy" style="color:{text_color};">'
                f'<span class="readiness-title" style="white-space:nowrap;">{title}</span>'
                f'<strong class="readiness-status" style="color:{text_color};">{status}</strong>'
                '</span>'
                '</div>'
            )

        readiness_col1, readiness_col2 = st.columns(2)
        with readiness_col1:
            st.markdown(
                readiness_card(ml_card),
                unsafe_allow_html=True,
            )
        with readiness_col2:
            st.markdown(
                readiness_card(ai_card),
                unsafe_allow_html=True,
            )

        st.divider()

        st.markdown("##### How to Use")
        st.caption(
            "1. **Select Site:** Enter GPS coords or pick a corridor in Tab 1 or Tab 2.\n\n"
            "2. **Assess Stability:** Review slope safety factor (FoS), root anchoring & remediations.\n\n"
            "3. **Design & Export:** Generate contractor BoQ, download reports, or chat with AI."
        )

        st.divider()

        # Chat and Session Actions
        if st.session_state.get("messages"):
            chat_md = export_chat_markdown(st.session_state.messages)
            st.download_button(
                label="Download Chat History (.md)",
                data=chat_md,
                file_name=f"vetiver_chat_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                icon=":material/download:",
                width="stretch",
            )
            if st.button("Clear Chat Conversation", icon=":material/delete_sweep:", width="stretch"):
                st.session_state.messages = []
                st.rerun()

        # Advanced Developer Specs (Cleanly Collapsed)
        with st.expander("🛠️ Advanced Engine Specs", expanded=False):
            st.caption(
                "• **Evidence Base:** 184 Empirical Trials (124 Success + 60 Failure Cases)\n"
                "• **Physics Boundaries:** 10 Deterministic Biomechanical Limit Rules\n"
                "• **Classifier:** CatBoost (80.5% Acc, 97.6% Recall, 87.1% F1)\n"
                "• **Explainability:** Native TreeSHAP Feature Attributions\n"
                "• **Knowledge Engine:** LlamaIndex Vector RAG + Qwen 2.5 3B"
            )
            docstore_file = os.path.join(config.STORAGE_DIR, "docstore.json")
            storage_status = "Persisted Index Available" if os.path.exists(docstore_file) else "Index Not Created"
            st.caption(f"• **Index Status:** {storage_status}")
            st.caption(f"• **CPU Threads:** {config.OLLAMA_NUM_THREADS}")
            st.caption(f"• **Top-K Retrieval:** {config.RAG_SIMILARITY_TOP_K}")
            st.caption(f"• **Chunk Size / Overlap:** {config.RAG_CHUNK_SIZE} / {config.RAG_CHUNK_OVERLAP}")
