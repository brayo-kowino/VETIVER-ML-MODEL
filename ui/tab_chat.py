import logging
from datetime import datetime

import streamlit as st

from services.assessment_service import get_active_site_context
from services.chat_service import answer_research_question, conversational_reply
from services.model_service import ChatUnavailable
from services.report_service import export_chat_markdown

logger = logging.getLogger("vetiver")


def render_message(message: dict) -> None:
    role = message["role"]
    with st.chat_message(role, avatar=":material/eco:" if role == "assistant" else None):
        if role == "user":
            st.markdown(message["content"])
        else:
            render_assistant_content(message)


def render_assistant_content(message: dict) -> None:
    # Older sessions embedded source cards in the Markdown.
    st.markdown(message["content"].split("<div class=", 1)[0].strip())
    if message.get("site_name"):
        st.caption(f"Site context · {message['site_name']}")
    sources = message.get("sources", [])
    if sources:
        with st.expander(f"Sources ({len(sources)})", icon=":material/article:"):
            for source in sources:
                st.markdown(f"**[{source['index']}]** `{source['file_name']}`")
                st.text(source["snippet"])


def render_welcome(site_context: dict | None) -> None:
    with st.container(key="vetiver_chat_welcome", gap=None):
        st.markdown(":material/eco:", text_alignment="center")
        st.markdown("## How can I help?", text_alignment="center")
        st.markdown(
            "Ask about Vetiver, slope stabilization, or your site.",
            text_alignment="center",
        )
        prompts = [
            "What slope risks should I check at my site?" if site_context else "How does Vetiver stabilize slopes?",
            "When does Vetiver need structural support?",
        ]
        with st.container(
            key="vetiver_chat_starters", horizontal=True, horizontal_alignment="center", gap="small"
        ):
            for prompt in prompts:
                st.button(
                    prompt,
                    icon=":material/arrow_outward:",
                    on_click=select_prompt,
                    args=(prompt,),
                )


def select_prompt(prompt: str) -> None:
    st.session_state.vetiver_chat_suggestion = prompt


def clear_conversation() -> None:
    st.session_state.messages = []
    st.session_state.pop("vetiver_chat_suggestion", None)


def render_tab_chat(tab) -> None:
    state = st.session_state
    user_input = st.chat_input("Message Vetiver AI", key="vetiver_chat_input")
    suggestion = state.pop("vetiver_chat_suggestion", "")
    question = (user_input or suggestion).strip()

    active_site = None
    if state.active_site["kind"] != "manual" or state.current_assessment:
        active_site = get_active_site_context(state.active_site, state.site_inputs, state.current_assessment)

    with tab, st.container(key="vetiver_chat", gap=None):
        with st.container(
            key="vetiver_chat_header",
            horizontal=True,
            horizontal_alignment="distribute",
            vertical_alignment="center",
            gap="small",
        ):
            title = st.container(key="vetiver_chat_title", gap="xsmall", width="content")
            with st.container(horizontal=True, vertical_alignment="center", gap="xsmall", width="content"):
                with st.popover("Settings", icon=":material/tune:", type="tertiary"):
                    response_format = st.selectbox(
                        "Response length",
                        ["concise", "detailed"],
                        format_func=str.capitalize,
                        key="rag_response_style",
                    )
                    include_site = active_site is not None and st.toggle(
                        "Use active site", key="chat_include_site"
                    )
                    export_slot = st.empty()
                if state.messages or question:
                    st.button("New chat", icon=":material/add:", on_click=clear_conversation)

        site_context = active_site if include_site else None
        with title:
            st.subheader(":material/eco: Vetiver AI", anchor=False)
            if site_context:
                st.caption(f"Active site · {site_context['location_name']}")

        # Keep an empty slot in the same position to clear the welcome before a slow reply.
        welcome = st.empty()
        if not state.messages and not question:
            with welcome.container():
                render_welcome(site_context)
            return

        with st.container(key="vetiver_chat_history", height=480, border=False, gap="small"):
            for message in state.messages:
                render_message(message)

            if question:
                reply = conversational_reply(question)
                site = site_context if reply is None else None
                site_key = (
                    (state.active_site["id"], tuple(state.site_inputs.items()), site["probability"])
                    if site else None
                )
                history = [
                    message for message in state.messages
                    if message.get("site_key") == site_key and not message.get("conversational")
                ]
                user_message = {
                    "role": "user",
                    "content": question,
                    "site_key": site_key,
                    "conversational": reply is not None,
                }
                render_message(user_message)

                sources = []
                with st.chat_message("assistant", avatar=":material/eco:"):
                    if reply is None:
                        with st.spinner("Thinking…"):
                            try:
                                reply, sources = answer_research_question(
                                    question,
                                    response_format=response_format,
                                    site_context=site,
                                    history=history,
                                )
                            except ChatUnavailable as exc:
                                logger.warning("Vetiver chat unavailable: %s", exc, exc_info=True)
                                reply = str(exc)
                            except Exception:
                                logger.exception("Vetiver research question failed")
                                reply = "Vetiver AI couldn't complete that request. Please try again."

                    assistant_message = {
                        "role": "assistant",
                        "content": reply,
                        "sources": sources,
                        "site_key": site_key,
                        "site_name": site["location_name"] if site else None,
                        "conversational": user_message["conversational"],
                    }
                    state.messages.extend((user_message, assistant_message))
                    render_assistant_content(assistant_message)

        export_slot.download_button(
            "Download conversation",
            data=export_chat_markdown(state.messages),
            file_name=f"vetiver_chat_{datetime.now():%Y%m%d_%H%M%S}.md",
            mime="text/markdown",
            icon=":material/download:",
            on_click="ignore",
            width="stretch",
        )
