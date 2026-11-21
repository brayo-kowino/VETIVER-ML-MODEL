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
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            if cb_model:
                st.success("ML Model\n**Ready**", icon=":material/check_circle:")
            else:
                st.error("ML Model\n**Offline**", icon=":material/error:")
        with col_s2:
            if ollama_online and query_engine:
                st.success("Vetiver AI\n**Ready**", icon=":material/smart_toy:")
            elif ollama_online:
                st.warning("Vetiver AI\n**Indexing**", icon=":material/sync:")
            else:
                st.error("Vetiver AI\n**Offline**", icon=":material/power_off:")

        st.divider()
        st.markdown("##### How to Use")
        st.caption(
            "1. **Select Site:** Enter GPS coords or pick a corridor in Tab 1 or Tab 2.\n\n"
            "2. **Assess Stability:** Review slope safety factor (FoS), root anchoring & remediations.\n\n"
            "3. **Design & Export:** Generate contractor BoQ, download reports, or chat with AI."
        )

        st.divider()
        # Session Actions
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
        with st.expander("Advanced Engine Specs", expanded=False):
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
