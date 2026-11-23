import os
import logging
import streamlit as st

import config
from services.model_service import (
    check_ollama_health,
    load_catboost,
    initialize_rag,
)
from ui.styles import load_css
from ui.state import init_session_state
from ui.sidebar import render_sidebar
from ui.tab_assessment import render_tab_assessment
from ui.tab_corridor import render_tab_corridor
from ui.tab_chat import render_tab_chat

# ── Logging Configuration ──────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("vetiver")
logging.getLogger("httpx").setLevel(logging.WARNING)

# ── 1. Page Configuration ──────────────────────────────
st.set_page_config(
    page_title="Vetiver AI Platform",
    page_icon=":material/eco:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 2. Load Modular Design System & Styles ─────────────
load_css("static/styles.css")

model_mtime = os.path.getmtime(config.MODEL_PATH) if os.path.exists(config.MODEL_PATH) else 0.0
hotspot_file = os.path.join(config.DATA_DIR, "regional_hotspots.json")
hotspot_mtime = os.path.getmtime(hotspot_file) if os.path.exists(hotspot_file) else 0.0
cb_model = load_catboost(model_mtime)
init_session_state()

tab1, tab2, tab3 = st.tabs([
    ":material/analytics: Site Assessment",
    ":material/map: 3D GIS Studio",
    ":material/smart_toy: Vetiver AI",
], key="active_workspace_tab", on_change="rerun")

ollama_online = None
query_engine = rag_index = None
if tab3.open:
    ollama_online = check_ollama_health()
    if ollama_online:
        kb_files = [
            os.path.join(config.KNOWLEDGE_BASE_DIR, name)
            for name in os.listdir(config.KNOWLEDGE_BASE_DIR)
        ] if os.path.isdir(config.KNOWLEDGE_BASE_DIR) else []
        kb_mtime = max((os.path.getmtime(p) for p in kb_files if os.path.isfile(p)), default=0.0)
        query_engine, rag_index = initialize_rag(kb_mtime)

render_sidebar(cb_model, ollama_online, query_engine)

if tab1.open:
    with tab1:
        render_tab_assessment(cb_model=cb_model, hotspot_mtime=hotspot_mtime)
elif tab2.open:
    with tab2:
        render_tab_corridor(cb_model=cb_model, hotspot_mtime=hotspot_mtime)
elif tab3.open:
    with tab3:
        render_tab_chat(rag_index=rag_index, ollama_online=ollama_online)
