import os
import streamlit as st
import config


def render_sidebar(cb_model) -> None:
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

        ai_card = ("idle", "smart_toy", "Vetiver AI", "On demand")

        def readiness_card(card):
            tone, icon, title, status = card
            colors = {
                "ready": ("#DCFCE7", "#166534", "#15803D"),
                "idle": ("#F1F5F9", "#334155", "#475569"),
                "offline": ("#FEE2E2", "#991B1B", "#DC2626"),
            }
            background, text_color, icon_color = colors[tone]
            if icon == "check_circle":
                icon_svg = (
                    '<svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">'
                    '<circle cx="12" cy="12" r="9" fill="none" stroke="currentColor" stroke-width="2"/>'
                    '<path d="M7.5 12.2l3 3 6-6" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>'
                    '</svg>'
                )
            elif icon == "error":
                icon_svg = (
                    '<svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">'
                    '<circle cx="12" cy="12" r="9" fill="none" stroke="currentColor" stroke-width="2"/>'
                    '<path d="M12 7v6M12 17h.01" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>'
                    '</svg>'
                )
            else:
                icon_svg = (
                    '<svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">'
                    '<rect x="4" y="6" width="16" height="12" rx="3" fill="none" stroke="currentColor" stroke-width="2"/>'
                    '<path d="M9 10h6M9 14h4" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>'
                    '</svg>'
                )
            return (
                f'<div class="readiness-card" style="background:{background};">'
                f'<span class="readiness-icon" style="color:{icon_color};">'
                f'{icon_svg}'
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
        # Advanced Developer Specs (Cleanly Collapsed)
        with st.expander("Advanced Engine Specs", expanded=False):
            st.caption(
                "• **Evidence Base:** 184 Empirical Trials (124 Success + 60 Failure Cases)\n"
                "• **Physics Boundaries:** 10 Deterministic Biomechanical Limit Rules\n"
                "• **Classifier:** CatBoost with grouped validation by source reference\n"
                "• **Explainability:** Native TreeSHAP Feature Attributions\n"
                "• **Knowledge Engine:** LlamaIndex Vector RAG + Qwen 2.5 3B"
            )

            docstore_file = os.path.join(config.STORAGE_DIR, "docstore.json")
            storage_status = "Persisted Index Available" if os.path.exists(docstore_file) else "Index Not Created"
            st.caption(f"• **Index Status:** {storage_status}")
            st.caption(f"• **CPU Threads:** {config.OLLAMA_NUM_THREADS}")
            st.caption(f"• **Top-K Retrieval:** {config.RAG_SIMILARITY_TOP_K}")
            st.caption(f"• **Chunk Size / Overlap:** {config.RAG_CHUNK_SIZE} / {config.RAG_CHUNK_OVERLAP}")
