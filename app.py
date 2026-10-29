import os
import json
import logging
from datetime import datetime

import streamlit as st
import pandas as pd
import httpx
from catboost import CatBoostClassifier

import config
from rag_engine import (
    build_query_engine,
    extract_source_metadata,
    format_sources_markdown,
)
from shap_utils import explain_prediction

# ── Logging ────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("vetiver")

# --- 1. PAGE SETUP & REAL ICONS ---
st.set_page_config(
    page_title="Vetiver AI Platform",
    page_icon=":material/eco:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS: Real icons, User-Right / Bot-Left chat alignment, fixed chat bar
st.markdown("""
    <style>
    /* Tabs polish */
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 48px; border-radius: 4px 4px 0px 0px; font-weight: 500; }
    
    /* Metrics scorecards */
    .stMetric {
        background-color: #F8FAF9;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 14px;
    }

    /* Executive Verdict Banners */
    .verdict-banner-suitable {
        background: linear-gradient(135deg, #F0FDF4 0%, #DCFCE7 100%);
        border: 1.5px solid #86EFAC;
        border-radius: 12px;
        padding: 22px 26px;
        margin-bottom: 20px;
        box-shadow: 0 4px 14px rgba(22, 163, 74, 0.08);
    }
    .verdict-banner-unsuitable {
        background: linear-gradient(135deg, #FEF2F2 0%, #FEE2E2 100%);
        border: 1.5px solid #FECACA;
        border-radius: 12px;
        padding: 22px 26px;
        margin-bottom: 20px;
        box-shadow: 0 4px 14px rgba(220, 38, 38, 0.08);
    }
    .kpi-row {
        display: flex;
        gap: 12px;
        margin-top: 14px;
        flex-wrap: wrap;
    }
    .kpi-chip {
        flex: 1;
        min-width: 160px;
        background: rgba(255, 255, 255, 0.85);
        border: 1px solid rgba(0, 0, 0, 0.08);
        border-radius: 8px;
        padding: 10px 14px;
        text-align: center;
    }
    .kpi-chip-title {
        font-size: 0.72rem;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.04em;
        color: #475569;
    }
    .kpi-chip-val {
        font-size: 1.15rem;
        font-weight: 800;
        margin-top: 2px;
    }

    /* Chat message row container: transparent, no outer box or double border */
    div[data-testid="stChatMessage"] {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0.4rem 0 !important;
        margin-bottom: 0.4rem !important;
        width: 100% !important;
    }

    /* USER CHAT: Avatar and bubble on the RIGHT */
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]),
    div[data-testid="stChatMessage"]:has([aria-label="Chat message from user"]),
    div[aria-label="Chat message from user"] {
        flex-direction: row-reverse !important;
        display: flex !important;
        justify-content: flex-start !important;
        gap: 0.75rem !important;
    }

    /* User Bubble: Single clean border & background ONLY on the content */
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) div[data-testid="stChatMessageContent"],
    div[data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) div[data-testid="stChatMessageContent"] {
        background-color: #E8F5E9 !important; /* Soft botanical mint */
        border: 1px solid #C8E6C9 !important;
        border-radius: 16px 4px 16px 16px !important;
        padding: 0.75rem 1.15rem !important;
        max-width: 78% !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04) !important;
        color: #1B5E20 !important;
    }

    /* BOT / ASSISTANT CHAT: Avatar and bubble on the LEFT */
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]),
    div[data-testid="stChatMessage"]:has([aria-label="Chat message from assistant"]),
    div[aria-label="Chat message from assistant"] {
        flex-direction: row !important;
        display: flex !important;
        justify-content: flex-start !important;
        gap: 0.75rem !important;
    }

    /* Assistant Bubble: Single clean border & background ONLY on the content */
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) div[data-testid="stChatMessageContent"],
    div[data-testid="stChatMessage"]:has([aria-label="Chat message from assistant"]) div[data-testid="stChatMessageContent"] {
        background-color: #F8FAFB !important; /* Neutral slate surface */
        border: 1px solid #E2E8F0 !important;
        border-radius: 4px 16px 16px 16px !important;
        padding: 0.75rem 1.15rem !important;
        max-width: 85% !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03) !important;
        color: #1E293B !important;
    }

    /* Eliminate internal child borders completely */
    div[data-testid="stChatMessageContent"] div[data-testid="stMarkdownContainer"],
    div[data-testid="stChatMessageContent"] p {
        border: none !important;
        background: transparent !important;
        box-shadow: none !important;
    }

    /* Chat input styling — inside tab only */
    div[data-testid="stChatInput"] {
        border-radius: 24px !important;
        border: 1.5px solid #CBD5E1 !important;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.06) !important;
        background-color: #FFFFFF !important;
    }
    div[data-testid="stChatInput"]:focus-within {
        border-color: #2E7D32 !important;
        box-shadow: 0 4px 20px rgba(46, 125, 50, 0.16) !important;
    }
    </style>
""", unsafe_allow_html=True)


# ── 2. OLLAMA HEALTH CHECK ─────────────────────────────
@st.cache_data(ttl=30, show_spinner=False)
def check_ollama_health() -> bool:
    """Check if the local Ollama daemon is reachable."""
    try:
        resp = httpx.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=4)
        resp.raise_for_status()
        return True
    except Exception as exc:
        logger.warning("Ollama health-check failed: %s", exc)
        return False


ollama_online = check_ollama_health()


# --- 3. LOAD PREDICTIVE MODEL (CATBOOST) ---
@st.cache_resource
def load_catboost():
    """Load the trained CatBoost classifier from disk."""
    try:
        model = CatBoostClassifier()
        model.load_model(config.MODEL_PATH)
        logger.info("CatBoost model loaded from %s", config.MODEL_PATH)
        return model
    except FileNotFoundError:
        logger.error("Model file not found at %s", config.MODEL_PATH)
        return None
    except Exception as exc:
        logger.error("Failed to load CatBoost model: %s", exc)
        return None


# --- 4. LOAD RAG PIPELINE ---
@st.cache_resource
def initialize_rag():
    """Build or reload the LlamaIndex query engine."""
    if not ollama_online:
        logger.error("Ollama offline — skipping RAG init.")
        return None
    query_engine, _ = build_query_engine(
        persist_dir=config.STORAGE_DIR,
        knowledge_base_dir=config.KNOWLEDGE_BASE_DIR,
        similarity_top_k=config.RAG_SIMILARITY_TOP_K,
    )
    return query_engine


# Initialize engines
cb_model = load_catboost()
query_engine = initialize_rag()

# --- 5. SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_analytics_report" not in st.session_state:
    st.session_state.last_analytics_report = None
if "current_assessment" not in st.session_state:
    st.session_state.current_assessment = None
if "preset_soil" not in st.session_state:
    st.session_state.preset_soil = "tropical hillside soil"
if "preset_climate" not in st.session_state:
    st.session_state.preset_climate = "tropical monsoon"
if "preset_watering" not in st.session_state:
    st.session_state.preset_watering = "rainfed"
if "preset_salinity" not in st.session_state:
    st.session_state.preset_salinity = "Low to Non-saline"
if "preset_application" not in st.session_state:
    st.session_state.preset_application = "Slope stabilization through root reinforcement"


def run_assessment(inputs_dict: dict) -> dict:
    """Executes CatBoost prediction, probability, SHAP, and report generation."""
    if not cb_model:
        return None
    input_data = pd.DataFrame({k: [v] for k, v in inputs_dict.items()})
    pred = int(cb_model.predict(input_data)[0])
    prob = float(cb_model.predict_proba(input_data)[0][1] * 100)
    explanation = explain_prediction(cb_model, input_data)
    exp_df = explanation["explanation_df"]
    report_md = generate_suitability_report(inputs_dict, pred, prob, exp_df)
    return {
        "inputs": inputs_dict,
        "prediction": pred,
        "probability": prob,
        "explanation": explanation,
        "exp_df": exp_df,
        "report_md": report_md,
    }


# --- REPORT GENERATION HELPERS ---
def generate_suitability_report(
    inputs: dict,
    prediction: int,
    probability: float,
    exp_df: pd.DataFrame,
) -> str:
    """Generate a structured Markdown engineering report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    verdict = "SUITABLE" if prediction == 1 else "UNSUITABLE"
    conf = f"{probability:.1f}%" if prediction == 1 else f"{100 - probability:.1f}%"

    lines = [
        "# VETIVER GEOTECHNICAL SITE SUITABILITY REPORT",
        f"**Generated:** {timestamp}",
        "**Assessment Standard:** Empirical Agronomic & Geotechnical Boundary Baselines",
        "\n---",
        "## 1. Executive Verdict",
        f"- **System Verdict:** **{verdict}**",
        f"- **Confidence Level:** {conf}",
        f"- **Risk Level:** {'Low Geotechnical Risk' if prediction == 1 else 'High Failure Risk'}",
        "\n## 2. Configured Site Parameters",
        f"- **Soil Type / Structure:** `{inputs['Soil type/structure']}`",
        f"- **Climate Regime:** `{inputs['climate']}`",
        f"- **Watering Regime:** `{inputs['watering regime']}`",
        f"- **Salinity Level:** `{inputs['salinity']}`",
        f"- **Intended Application:** `{inputs['Application']}`",
        "\n## 3. TreeSHAP Feature Attributions",
        "The table below indicates the mathematical contribution of each site condition to the final verdict:\n",
        "| Site Parameter | Input Value | SHAP Impact (Log-Odds) | Directional Effect |",
        "| :--- | :--- | :--- | :--- |",
    ]
    for _, row in exp_df.iterrows():
        lines.append(f"| {row['Feature']} | {row['Input Value']} | {row['SHAP Impact']:+.4f} | {row['Effect']} |")

    lines.extend([
        "\n## 4. Engineering Recommendations",
        "- **Establishment Protocol:** Ensure initial moisture during the first 30–60 days before mature root network anchoring."
        if prediction == 1 else
        "- **Mitigation Protocol:** Extreme soil compaction, solid bedrock refusal, or hyper-salinity detected. Pre-treatment or mechanical retaining recommended.",
        "- **Monitoring:** Inspect root development and contour hedgerow density at 6 and 12-month intervals.",
        "\n---",
        "*Report compiled by Vetiver AI Decision Support Platform.*"
    ])
    return "\n".join(lines)


def export_chat_markdown(messages: list) -> str:
    """Format full chat history with citations into an exportable Markdown file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# Vetiver Literature & Knowledge Chat Session",
        f"**Exported:** {timestamp}\n",
        "---",
    ]
    for idx, msg in enumerate(messages, 1):
        role_label = "User" if msg["role"] == "user" else "Vetiver AI"
        lines.append(f"\n### {idx}. {role_label}")
        lines.append(msg["content"])
        if msg.get("sources"):
            lines.append(f"\n> **Citations ({len(msg['sources'])} matched sources):**")
            for s in msg["sources"]:
                score_str = f" [Score: {s['score']:.3f}]" if s.get("score") is not None else ""
                lines.append(f"> - `{s['file_name']}`{score_str}: \"{s['snippet']}...\"")
    return "\n".join(lines)


# --- 6. SIDEBAR BRANDING & REAL ICONS ---
with st.sidebar:
    st.subheader(":material/eco: Vetiver AI Platform")
    st.caption("Bio-Engineering Decision Support & RAG Knowledge Assistant")
    st.divider()

    st.markdown("### System Status")
    if cb_model:
        st.success("Tabular Model: Online (CatBoost)", icon=":material/check_circle:")
    else:
        st.error("Tabular Model: Offline", icon=":material/error:")

    if not ollama_online:
        st.error("Ollama Daemon: Offline (port 11434)", icon=":material/power_off:")
    elif query_engine:
        st.success(f"LLM Engine: Online ({config.OLLAMA_LLM_MODEL})", icon=":material/smart_toy:")
        st.success(f"Embeddings: Online ({config.OLLAMA_EMBED_MODEL})", icon=":material/hub:")
    else:
        st.warning("Language Model: Offline (Check knowledge base)", icon=":material/warning:")

    st.divider()

    # System Diagnostics Expander
    with st.expander("System Diagnostics & Storage", icon=":material/tune:"):
        docstore_file = os.path.join(config.STORAGE_DIR, "docstore.json")
        storage_status = "Persisted Index Available" if os.path.exists(docstore_file) else "Index Not Created"
        st.caption(f"• **Index Status:** {storage_status}")
        st.caption(f"• **CPU Threads:** {config.OLLAMA_NUM_THREADS} allocated threads")
        st.caption(f"• **Max Generation:** {config.OLLAMA_NUM_PREDICT} tokens")
        st.caption(f"• **RAM Residency:** {config.OLLAMA_KEEP_ALIVE}")
        st.caption(f"• **Top-K Retrieval:** {config.RAG_SIMILARITY_TOP_K}")
        st.caption(f"• **Chunk Size / Overlap:** {config.RAG_CHUNK_SIZE} / {config.RAG_CHUNK_OVERLAP}")
        st.caption(f"• **Timeout:** {config.LLM_REQUEST_TIMEOUT}s")
        st.caption(f"• **Model Features:** {len(config.CATEGORICAL_FEATURES)} categorical variables")

    st.divider()
    st.markdown("### Model Specifications")
    st.caption(
        "• **Algorithm:** Gradient Boosted Trees (CatBoost)\n"
        "• **Validation:** 5-Fold Stratified Cross-Validation (89.7% Mean Acc)\n"
        "• **Explainability:** Native TreeSHAP Attributions\n"
        "• **Knowledge Engine:** LlamaIndex Vector RAG + Qwen 2.5 3B"
    )

    if st.session_state.messages:
        chat_md = export_chat_markdown(st.session_state.messages)
        st.download_button(
            label="Export Chat Session (.md)",
            data=chat_md,
            file_name=f"vetiver_chat_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
            icon=":material/download:",
            use_container_width=True,
        )

    if st.button("Clear Chat History", icon=":material/delete_sweep:", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# --- 7. HYBRID UI (TABS WITH MATERIAL ICONS) ---
tab1, tab2 = st.tabs([
    ":material/analytics: Geotechnical Suitability & SHAP",
    ":material/forum: Literature & Knowledge Chat"
])


# =======================================================
# TAB 1: CATBOOST ANALYTICS & SHAP EXPLANATION
# =======================================================
with tab1:
    st.subheader("Geotechnical Site Suitability Assessment", anchor=False)
    st.caption("CatBoost Classifier · 5-Fold Stratified CV · TreeSHAP Attribution")

    # --- CUSTOM SITE PARAMETER INPUTS ---
    with st.container(border=True):
        st.markdown("##### :material/tune: Site Geotechnical Parameters")
        col1, col2 = st.columns(2)
        with col1:
            soil_options = [
                "tropical hillside soil",
                "sandy clay",
                "clay soil",
                "loose sandy soil",
                "steep hill slope soil",
                "various soils",
                "gold mine tailings",
                "solid bedrock",
                "permafrost frozen soil",
            ]
            soil_idx = soil_options.index(st.session_state.preset_soil) if st.session_state.preset_soil in soil_options else 0
            soil_type = st.selectbox(
                "Soil Type / Structure",
                soil_options,
                index=soil_idx,
                help="Geotechnical soil classification and mechanical texture of the site slope.",
            )

            climate_options = [
                "tropical monsoon",
                "tropical",
                "tropical to subtropical",
                "tropical humid",
                "subtropical highland",
                "semi-arid",
                "desert arid",
                "arctic freezing",
            ]
            climate_idx = climate_options.index(st.session_state.preset_climate) if st.session_state.preset_climate in climate_options else 0
            climate = st.selectbox(
                "Climate Regime",
                climate_options,
                index=climate_idx,
                help="Regional climatic zone determining precipitation and temperature ranges.",
            )

        with col2:
            watering_options = [
                "rainfed",
                "natural rainfall",
                "seasonal rainfall",
                "high intensity rainfall",
                "extreme rainfall",
                "plants were watered as necessary",
                "no water",
                "flooded completely",
            ]
            watering_idx = watering_options.index(st.session_state.preset_watering) if st.session_state.preset_watering in watering_options else 0
            watering = st.selectbox(
                "Watering Regime",
                watering_options,
                index=watering_idx,
                help="Hydrological moisture supply and drainage characteristics of the slope.",
            )

            salinity_options = [
                "Low to Non-saline",
                "NR",
                "saline tolerant",
                "moderate to high tolerance",
                "high",
                "extreme",
            ]
            salinity_idx = salinity_options.index(st.session_state.preset_salinity) if st.session_state.preset_salinity in salinity_options else 0
            salinity = st.selectbox(
                "Salinity Level",
                salinity_options,
                index=salinity_idx,
                help="Soil salinity / electrical conductivity (EC). Extreme salinity (>12 dS/m) causes biological stress.",
            )

        app_options = [
            "Slope stabilization through root reinforcement",
            "erosion control and slope stabilization",
            "erosion control and runoff reduction",
            "soil erosion control using contour hedgerows",
            "reduction of runoff and soil erosion",
            "rehabilitation of gold mine tailings",
            "water treatment",
            "land restoration",
        ]
        app_idx = app_options.index(st.session_state.preset_application) if st.session_state.preset_application in app_options else 0
        application = st.selectbox(
            "Intended Bio-Engineering Application",
            app_options,
            index=app_idx,
            help="Target geotechnical civil engineering objective.",
        )

        btn_c1, btn_c2 = st.columns([3, 1])
        with btn_c1:
            run_clicked = st.button("Run Geotechnical Analytics", icon=":material/play_arrow:", type="primary", use_container_width=True)
            if run_clicked:
                current_inputs = {
                    "Soil type/structure": soil_type,
                    "climate": climate,
                    "watering regime": watering,
                    "Application": application,
                    "salinity": salinity,
                }
                st.session_state.preset_soil = soil_type
                st.session_state.preset_climate = climate
                st.session_state.preset_watering = watering
                st.session_state.preset_salinity = salinity
                st.session_state.preset_application = application
                st.session_state.current_assessment = run_assessment(current_inputs)
                st.rerun()

        with btn_c2:
            if st.button("Reset", icon=":material/refresh:", use_container_width=True):
                st.session_state.current_assessment = None
                st.rerun()

    # --- RENDER ASSESSMENT RESULTS (PERSISTENT ACROSS RERUNS) ---
    if st.session_state.current_assessment is not None:
        assessment = st.session_state.current_assessment
        pred = assessment["prediction"]
        prob = assessment["probability"]
        exp_df = assessment["exp_df"]
        report_md = assessment["report_md"]

        st.divider()

        # 1. Executive Banner
        if pred == 1:
            st.markdown(f"""
                <div class="verdict-banner-suitable">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 16px;">
                        <div>
                            <span style="background: #16A34A; color: white; font-size: 0.75rem; font-weight: 700; padding: 4px 12px; border-radius: 100px; text-transform: uppercase; letter-spacing: 0.05em;">Positive Geotechnical Match</span>
                            <h2 style="color: #14532D; margin: 10px 0 6px 0; font-size: 1.55rem; font-weight: 800;">VERDICT: SUITABLE FOR VETIVER BIO-ENGINEERING</h2>
                            <p style="color: #166534; font-size: 0.95rem; margin: 0; line-height: 1.45;">Site parameters align with empirical stabilization baselines. Conditions support deep root anchoring without critical chemical or moisture limits.</p>
                        </div>
                        <div style="text-align: right; min-width: 140px;">
                            <div style="font-size: 2.2rem; font-weight: 900; color: #14532D; line-height: 1;">{prob:.1f}%</div>
                            <div style="font-size: 0.76rem; font-weight: 700; color: #166534; text-transform: uppercase; margin-top: 4px;">Model Confidence</div>
                        </div>
                    </div>
                    <div class="kpi-row">
                        <div class="kpi-chip">
                            <div class="kpi-chip-title">Root Depth Potential</div>
                            <div class="kpi-chip-val" style="color: #166534;">2.5 – 3.5 m</div>
                        </div>
                        <div class="kpi-chip">
                            <div class="kpi-chip-title">Mean Tensile Strength</div>
                            <div class="kpi-chip-val" style="color: #166534;">75 – 85 MPa</div>
                        </div>
                        <div class="kpi-chip">
                            <div class="kpi-chip-title">Safety Factor Boost</div>
                            <div class="kpi-chip-val" style="color: #166534;">+35% to +45%</div>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="verdict-banner-unsuitable">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 16px;">
                        <div>
                            <span style="background: #DC2626; color: white; font-size: 0.75rem; font-weight: 700; padding: 4px 12px; border-radius: 100px; text-transform: uppercase; letter-spacing: 0.05em;">Critical Failure Warning</span>
                            <h2 style="color: #7F1D1D; margin: 10px 0 6px 0; font-size: 1.55rem; font-weight: 800;">VERDICT: UNFAVORABLE — HIGH FAILURE RISK</h2>
                            <p style="color: #991B1B; font-size: 0.95rem; margin: 0; line-height: 1.45;">Site conditions violate physiological or geotechnical boundary limits (e.g. bedrock refusal, extreme salinity, or severe water deficit). Biological establishment is not recommended without engineering remediation.</p>
                        </div>
                        <div style="text-align: right; min-width: 140px;">
                            <div style="font-size: 2.2rem; font-weight: 900; color: #7F1D1D; line-height: 1;">{100 - prob:.1f}%</div>
                            <div style="font-size: 0.76rem; font-weight: 700; color: #991B1B; text-transform: uppercase; margin-top: 4px;">Failure Confidence</div>
                        </div>
                    </div>
                    <div class="kpi-row">
                        <div class="kpi-chip">
                            <div class="kpi-chip-title">Root Penetration</div>
                            <div class="kpi-chip-val" style="color: #991B1B;">Impeded (&lt; 0.3 m)</div>
                        </div>
                        <div class="kpi-chip">
                            <div class="kpi-chip-title">Geotechnical Hazard</div>
                            <div class="kpi-chip-val" style="color: #991B1B;">High Risk of Slip</div>
                        </div>
                        <div class="kpi-chip">
                            <div class="kpi-chip-title">Recommended Action</div>
                            <div class="kpi-chip-val" style="color: #991B1B;">Soil Amendment / Shoring</div>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        # 2. Metric Scorecards
        m1, m2, m3 = st.columns(3)
        if pred == 1:
            m1.metric(label="System Verdict", value="SUITABLE", delta="Positive Match")
            m2.metric(label="Suitability Confidence", value=f"{prob:.1f}%")
            m3.metric(label="Risk Assessment", value="Low Geotechnical Risk", delta="Safe")
        else:
            m1.metric(label="System Verdict", value="UNSUITABLE", delta="- Negative Match", delta_color="inverse")
            m2.metric(label="Confidence Level", value=f"{100 - prob:.1f}%")
            m3.metric(label="Risk Assessment", value="High Failure Risk", delta="- Unfavorable", delta_color="inverse")

        # 3. Decision Explainability (SHAP)
        st.subheader("Decision Explainability (TreeSHAP Feature Attribution)")
        st.caption("Quantifies the exact mathematical contribution (log-odds impact) of each site condition toward the verdict:")

        chart_data = exp_df.set_index("Feature")["SHAP Impact"]
        st.bar_chart(chart_data)

        st.dataframe(
            exp_df,
            column_config={
                "Feature": "Site Parameter",
                "Input Value": "Configured Value",
                "SHAP Impact": st.column_config.NumberColumn("SHAP Value (Log-Odds Impact)", format="%.4f"),
                "Effect": "Directional Influence",
            },
            hide_index=True,
            use_container_width=True,
        )

        # 4. Report Preview & Download
        with st.expander("Preview Full Geotechnical Engineering Report", icon=":material/description:"):
            st.markdown(report_md)

        st.download_button(
            label="Download Full Geotechnical Report (.md)",
            data=report_md,
            file_name=f"vetiver_site_assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
            icon=":material/download:",
            use_container_width=True,
        )

        shap_plot_file = os.path.join(config.DATA_DIR, "shap_summary.png")
        if os.path.exists(shap_plot_file):
            with st.expander("View Global Model Feature Importance (SHAP Summary)", icon=":material/bar_chart:"):
                st.image(shap_plot_file, caption="Global Mean |SHAP| values computed across all empirical and failure boundaries.")


# =======================================================
# TAB 2: RAG LITERATURE CHAT WITH CITATIONS
# =======================================================
with tab2:
    st.header("Vetiver Research & Literature Assistant")
    st.markdown("Ask natural language questions about Vetiver grass performance, root reinforcement, and site applications.")

    # Quick prompt suggestions with real icons
    st.markdown("**Suggested Technical Prompts:**")
    prompt_cols = st.columns(3)
    p1 = prompt_cols[0].button("Soil types supporting Vetiver", icon=":material/terrain:", use_container_width=True)
    p2 = prompt_cols[1].button("Maximum recorded root depth", icon=":material/straighten:", use_container_width=True)
    p3 = prompt_cols[2].button("Surviving floods & extreme rain", icon=":material/water_drop:", use_container_width=True)

    selected_prompt = None
    if p1:
        selected_prompt = "What soil types and climates are documented to support Vetiver grass establishment best?"
    elif p2:
        selected_prompt = "What is the maximum root length or depth recorded for mature Vetiver, and in which locations?"
    elif p3:
        selected_prompt = "How does Vetiver grass perform under extreme rainfall, flooding, or steep slope conditions?"

    if not ollama_online:
        st.error("Ollama daemon is not running. Please start Ollama (`ollama serve`) and refresh this page.", icon=":material/power_off:")
    elif query_engine is None:
        st.warning("Knowledge base index is not initialized. Ensure documents exist in `knowledge_base/`.", icon=":material/warning:")
    else:
        # Display conversation history (User Right, Bot Left)
        for msg in st.session_state.messages:
            is_user = msg["role"] == "user"
            avatar_icon = ":material/person:" if is_user else ":material/psychology:"

            with st.chat_message(msg["role"], avatar=avatar_icon):
                st.markdown(msg["content"])
                if msg.get("sources"):
                    with st.expander(f"Verified Citations ({len(msg['sources'])} sources matched)", icon=":material/library_books:"):
                        st.markdown(format_sources_markdown(msg["sources"]))

        # Handle user input — chat input lives inside tab2 only
        user_query = st.chat_input("Ask about Vetiver applications, root depth, soil shear strength, or site trials...")
        active_query = user_query or selected_prompt

        if active_query:
            # 1. Append and display user message on the RIGHT
            st.session_state.messages.append({"role": "user", "content": active_query})
            with st.chat_message("user", avatar=":material/person:"):
                st.markdown(active_query)

            # 2. Query RAG engine with a clean SPINNER, display on the LEFT
            with st.chat_message("assistant", avatar=":material/psychology:"):
                response_placeholder = st.empty()

                try:
                    # Circular animated spinner during retrieval & initial LLM computation
                    with st.spinner("Thinking..."):
                        streaming_response = query_engine.query(active_query)
                        gen = streaming_response.response_gen
                        first_chunk = next(gen, "")

                    # Real-time word-by-word streaming
                    full_response = first_chunk
                    response_placeholder.markdown(full_response + "▌")
                    for chunk in gen:
                        full_response += chunk
                        response_placeholder.markdown(full_response + "▌")

                    response_placeholder.markdown(full_response)

                    # Extract source citations
                    sources = extract_source_metadata(streaming_response)
                    if sources:
                        with st.expander(f"Verified Citations ({len(sources)} sources matched)", icon=":material/library_books:"):
                            st.markdown(format_sources_markdown(sources))

                    # Save to state
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_response,
                        "sources": sources,
                    })

                except Exception as exc:
                    logger.error("RAG query failed: %s", exc, exc_info=True)
                    err_msg = "An error occurred while retrieving literature context. Please check Ollama and try again."
                    response_placeholder.error(err_msg, icon=":material/error:")
                    st.session_state.messages.append({"role": "assistant", "content": err_msg})
