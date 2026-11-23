import logging
from html import escape
from datetime import datetime

import streamlit as st

import config
from services.report_service import export_chat_markdown
from services.assessment_service import get_active_site_context

logger = logging.getLogger("vetiver")


def render_excerpts(sources: list[dict]) -> None:
    for source in sources:
        with st.expander(f"[{source['index']}] {source['file_name']}"):
            st.text(source["snippet"])


def render_tab_chat(rag_index, ollama_online: bool = True) -> None:
    st.subheader("Vetiver", anchor=False)
    st.caption("Ask questions grounded in field trials, geotechnical standards, and your active site conditions.")

    if not ollama_online:
        st.markdown("""
        <div style="min-height: 360px; display: flex; align-items: center; justify-content: center;">
            <div style="width: min(520px, 90%); text-align: center; border: 1px solid #FECACA; background: #FEF2F2; border-radius: 14px; padding: 38px 28px;">
                <div style="font-family: 'Material Symbols Rounded'; font-size: 56px; line-height: 1; color: #DC2626;">block</div>
                <div style="font-size: 1.25rem; font-weight: 800; color: #991B1B; margin-top: 14px;">Vetiver AI unavailable</div>
                <div style="font-size: 0.92rem; color: #7F1D1D; line-height: 1.55; margin-top: 8px;">
                    The cloud deployment does not include a local Ollama runtime or the Qwen 2.5 3B model.
                    The literature assistant and RAG chat require both to be running.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    from rag_engine import (
        PERSONA_PROMPTS,
        clean_markdown_formatting,
        extract_source_metadata,
        query_rag_custom,
        validate_citations,
    )

    # Auto-sanitize and clean legacy messages in session state
    if "messages" in st.session_state and st.session_state.messages:
        cleaned_messages = []
        for m in st.session_state.messages:
            content_str = m.get("content", "")
            if isinstance(content_str, str):
                if "<div class=" in content_str or "citation-card" in content_str:
                    content_str = content_str.split("<div class=")[0].strip()
                if m.get("role") == "assistant":
                    content_str = clean_markdown_formatting(content_str)
                m["content"] = content_str
            cleaned_messages.append(m)
        st.session_state.messages = cleaned_messages

    # 1. Consultant Persona Switcher, Response Style, Export & Reset
    persona_col, style_col, export_col, clear_col = st.columns([2.4, 1.8, 1.2, 1.0])
    with persona_col:
        selected_persona = st.selectbox(
            "Consultant Persona",
            options=["geotechnical", "contractor", "environmental"],
            format_func=lambda x: f"{PERSONA_PROMPTS[x]['title']}",
            key="selected_consultant_persona"
        )
    with style_col:
        response_style = st.selectbox(
            "Response Format",
            options=["concise", "detailed"],
            format_func=lambda x: "Concise Bullet Points" if x == "concise" else "Comprehensive Report",
            key="rag_response_style"
        )
    with export_col:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if st.session_state.get("messages"):
            chat_md = export_chat_markdown(st.session_state.messages)
            st.download_button(
                label="Export",
                data=chat_md,
                file_name=f"vetiver_consultation_memo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                icon=":material/download:",
                width="stretch",
                help="Export consultation history with citations as a Markdown engineering memo."
            )
        else:
            st.button("Export", disabled=True, icon=":material/download:", width="stretch", help="No messages to export yet")
    with clear_col:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if st.button("Reset Chat", icon=":material/restart_alt:", width="stretch", help="Clear conversation history"):
            st.session_state.messages = []
            st.rerun()

    # 2. Active Field Assessment Context Bridge
    state = st.session_state
    active_site = None
    bridge_enabled = False

    if state.active_site["kind"] != "manual" or state.current_assessment:
        active_site = get_active_site_context(state.active_site, state.site_inputs, state.current_assessment)
        prediction = active_site["prediction"]
        is_suit = prediction == 1
        badge_cls = "badge-suitable" if is_suit else "badge-risk"
        badge_text = {1: "SUITABLE", 0: "UNSUITABLE"}.get(prediction, "ASSESSMENT PENDING")
        score = active_site["probability"]
        score_text = f" · {score:.1f}% CatBoost suitability score" if score is not None else ""
        border_col = "#16A34A" if is_suit else "#DC2626"

        st.markdown(f"""
        <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-left: 3px solid {border_col}; border-radius: 8px; padding: 10px 14px; margin-bottom: 8px; font-size: 0.84rem; color: #334155; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 6px;">
            <div>
                <b>Active site</b> · {escape(active_site['location_name'])} ({active_site.get('elevation_m', 'N/A')} m a.s.l.) ·
                Soil: <b>{active_site.get('soil_mechanical_class', 'N/A')}</b> · 
                Rainfall: <b>{active_site.get('climate_rainfall_zone', 'N/A')}</b>
            </div>
            <span class="studio-badge {badge_cls}">{badge_text}{score_text}</span>
        </div>
        """, unsafe_allow_html=True)

        bridge_enabled = st.checkbox(
            "Include Active Site Parameters in LLM Context",
            value=True,
            help="Automatically passes the site's elevation, soil class, and failure limits to the consultant."
        )

    # 3. Suggested Action Prompts (Collapsible once conversation starts)
    has_history = len(st.session_state.messages) > 0
    site_name_short = active_site.get('location_name', 'Site') if active_site else "Active Site"
    if len(site_name_short) > 22:
        site_name_short = site_name_short[:20] + "..."
    btn1_label = f"Audit Assessment ({site_name_short})" if active_site else "General Slope Stability Design"

    selected_prompt = None
    with st.expander(":material/flash_on: Suggested Engineering Consultation Prompts", expanded=not has_history):
        prompt_r1 = st.columns(3)
        p1 = prompt_r1[0].button(btn1_label, icon=":material/psychology:", width="stretch")
        p2 = prompt_r1[1].button("Contractor Field Checklist", icon=":material/assignment:", width="stretch")
        p3 = prompt_r1[2].button("NEMA EIA Non-Invasiveness", icon=":material/verified_user:", width="stretch")

        prompt_r2 = st.columns(3)
        p4 = prompt_r2[0].button("Wu-Waldron Root Shear (cr)", icon=":material/architecture:", width="stretch")
        p5 = prompt_r2[1].button("Collapsible Pumice Hydro-Piping", icon=":material/warning:", width="stretch")
        p6 = prompt_r2[2].button("Fluvial Scour & Flooding Limits", icon=":material/water_drop:", width="stretch")

        if p1:
            if active_site:
                selected_prompt = (
                    f"Perform a comprehensive bio-engineering geotechnical synthesis for the active project site ({active_site.get('location_name', 'Site')}). "
                    f"Evaluate its soil mechanics ({active_site.get('soil_mechanical_class')}), elevation ({active_site.get('elevation_m')} m), "
                    f"and assessment status ({badge_text}). Discuss supporting literature and any field verification needed."
                )
            else:
                selected_prompt = "What is the standard geotechnical bio-engineering design methodology for stabilizing cut slopes with Vetiver grass rows under Eurocode 7 standards?"
        elif p2:
            selected_prompt = (
                "Generate an exhaustive contractor field operations checklist for Vetiver slope revetment: "
                "specify nursery slip quality criteria (minimum tillers, root and leaf trimming lengths), contour trench excavation dimensions, "
                "hydrogel and organic compost dosing rates, and the mandatory 60-day irrigation and livestock fencing schedule."
            )
        elif p3:
            selected_prompt = (
                "Provide an official technical statement for NEMA Environmental Impact Assessment (EIA) submission certifying the ecological safety "
                "and obligate sterility of Chrysopogon zizanioides (Sunshine genotype). Detail why it cannot become an invasive weed, "
                "and document its root bioconcentration factor (BCF) and translocation factor (TF) for heavy metals."
            )
        elif p4:
            selected_prompt = (
                "Explain the Wu-Waldron (1977) root reinforcement equation cr = 1.2 * Tr * RAR and how root tensile strength (75–85 MPa) "
                "translates into effective cohesion increases (c' + cr). How does this affect limit equilibrium Factor of Safety on slopes greater than 35 degrees?"
            )
        elif p5:
            selected_prompt = (
                "Why do Vetiver bioengineering rows experience severe hydro-piping and subsurface collapse in volcanic pumice soils "
                "(such as Nakuru, Menengai, or the Great Rift Valley), and what structural soil nailing or geotextile hybrids are required to prevent washouts?"
            )
        elif p6:
            selected_prompt = (
                "What are the maximum permissible hydraulic shear stress (tau_c) and flow velocity thresholds (m/s) that mature Vetiver hedgerows can resist "
                "in riverbank and roadside culvert revetments? What are the critical limits for continuous flood submergence?"
            )

    # 4. Conversation History & Chat Flow
    if rag_index is None:
        st.warning("Knowledge base index is not initialized. Ensure documents exist in `knowledge_base/`.", icon=":material/warning:")
    else:
        # Chat container ensures ALL messages stay ABOVE the input bar
        chat_container = st.container()

        # Chat input is placed at the bottom
        user_query = st.chat_input("Ask about slope shear cohesion (c'), slip depths, nursery slips, or NEMA EIA compliance...")
        active_query = user_query or selected_prompt

        with chat_container:
            # 1. Render all past messages
            for msg in st.session_state.messages:
                is_user = msg["role"] == "user"
                if is_user:
                    with st.chat_message("user", avatar=":material/person:"):
                        st.markdown(msg["content"])
                else:
                    msg_persona = msg.get("persona", selected_persona)
                    p_info = PERSONA_PROMPTS.get(msg_persona, PERSONA_PROMPTS["geotechnical"])
                    p_icon = p_info.get("icon", ":material/psychology:")

                    with st.chat_message("assistant", avatar=p_icon):
                        st.caption(f":material/verified: **{p_info['title']}** · *{p_info['role_desc']}*")
                        if msg.get("site_name"):
                            st.caption(f"Site context · {msg['site_name']}")
                        st.markdown(clean_markdown_formatting(msg["content"]))
                        render_excerpts(msg.get("sources", []))

            # 2. Render newly active query inside container (ABOVE the chat input)
            if active_query:
                site_ctx_for_call = active_site if bridge_enabled else None
                site_key = (
                    (state.active_site["id"], tuple(state.site_inputs.items()), site_ctx_for_call["probability"])
                    if site_ctx_for_call else None
                )
                st.session_state.messages.append({"role": "user", "content": active_query, "site_key": site_key})
                with st.chat_message("user", avatar=":material/person:"):
                    st.markdown(active_query)

                p_info = PERSONA_PROMPTS.get(selected_persona, PERSONA_PROMPTS["geotechnical"])
                p_icon = p_info.get("icon", ":material/psychology:")

                with st.chat_message("assistant", avatar=p_icon):
                    st.caption(f":material/verified: **{p_info['title']}** · *{p_info['role_desc']}*")
                    response_placeholder = st.empty()

                    try:
                        with st.spinner("Analyzing empirical literature & site constraints..."):
                            response = query_rag_custom(
                                index=rag_index,
                                query_str=active_query,
                                persona_key=selected_persona,
                                site_context=site_ctx_for_call,
                                conversation_history=[m for m in st.session_state.messages if m.get("site_key") == site_key],
                                similarity_top_k=config.RAG_SIMILARITY_TOP_K,
                                response_mode=response_style,
                            )
                            full_response = clean_markdown_formatting("".join(response.response_gen))
                            full_response, sources = validate_citations(full_response, extract_source_metadata(response))
                        response_placeholder.markdown(full_response)
                        render_excerpts(sources)

                        # Save to state
                        st.session_state.messages.append({
                            "role": "assistant",
                            "persona": selected_persona,
                            "content": full_response,
                            "sources": sources,
                            "site_key": site_key,
                            "site_name": site_ctx_for_call["location_name"] if site_ctx_for_call else None,
                        })

                    except Exception as exc:
                        logger.error("RAG query failed: %s", exc, exc_info=True)
                        err_msg = "An error occurred while retrieving literature context. Please check Ollama and try again."
                        response_placeholder.error(err_msg, icon=":material/error:")
                        st.session_state.messages.append({"role": "assistant", "persona": selected_persona, "content": err_msg})
