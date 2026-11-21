import os
import json
import math
import html
import logging
from datetime import datetime

import streamlit as st
import pandas as pd
import httpx
import pydeck as pdk
from catboost import CatBoostClassifier

import config
from rag_engine import (
    build_query_engine,
    extract_source_metadata,
    format_sources_markdown,
    query_rag_custom,
    clean_markdown_formatting,
    PERSONA_PROMPTS,
)
from shap_utils import explain_prediction
from geospatial_resolver import resolve_coordinate_geotechnics
from geotechnical_engine import (
    calculate_slope_stability,
    generate_slope_cross_section_fig,
    generate_fos_sensitivity_fig,
    calculate_field_boq,
    generate_temporal_maturation_fig,
)
from corridor_engine import (
    BUILTIN_CORRIDORS,
    analyze_corridor_waypoints,
    generate_corridor_profile_fig,
    export_corridor_geojson,
)

# ── Logging ────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("vetiver")
logging.getLogger("httpx").setLevel(logging.WARNING)

# --- 1. PAGE SETUP & REAL ICONS ---
st.set_page_config(
    page_title="Vetiver AI Platform",
    page_icon=":material/eco:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS: Modern Studio-Grade Design System, Real Material Icons, Split-Screen Layouts
st.markdown("""
    <style>
    /* Google Fonts & Root Design Tokens */
    :root {
        --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        --bg-surface: #F8FAFC;
        --bg-card: #FFFFFF;
        --border-subtle: #E2E8F0;
        --border-strong: #CBD5E1;
        --brand-emerald: #16A34A;
        --brand-emerald-dark: #14532D;
        --brand-emerald-light: #DCFCE7;
    }

    /* Global block padding optimization */
    .block-container {
        padding-top: 1.8rem !important;
        padding-bottom: 2.2rem !important;
        max-width: 98% !important;
    }

    /* Tabs styling — Modern clean line with Material Icons */
    .stTabs [data-baseweb="tab-list"] {
        gap: 14px !important;
        border-bottom: 2px solid #E2E8F0 !important;
    }
    .stTabs [data-baseweb="tab"] {
        height: 46px !important;
        border-radius: 8px 8px 0px 0px !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
        color: #64748B !important;
        padding: 0 16px !important;
    }
    .stTabs [aria-selected="true"] {
        color: #16A34A !important;
        border-bottom: 3px solid #16A34A !important;
        background: transparent !important;
    }

    /* Studio Card Surface */
    .studio-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        margin-bottom: 14px;
    }

    /* Studio Inspector Docked Card */
    .studio-inspector {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.04);
    }

    /* Executive Verdict Banners */
    .verdict-banner-suitable {
        background: linear-gradient(135deg, #F0FDF4 0%, #DCFCE7 100%);
        border: 1.5px solid #86EFAC;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 16px;
        box-shadow: 0 3px 12px rgba(22, 163, 74, 0.08);
    }
    .verdict-banner-unsuitable {
        background: linear-gradient(135deg, #FEF2F2 0%, #FEE2E2 100%);
        border: 1.5px solid #FECACA;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 16px;
        box-shadow: 0 3px 12px rgba(220, 38, 38, 0.08);
    }

    /* KPI metric chips */
    .kpi-row {
        display: flex;
        gap: 10px;
        margin-top: 14px;
        flex-wrap: wrap;
    }
    .kpi-chip {
        flex: 1;
        min-width: 130px;
        background: rgba(255, 255, 255, 0.92);
        border: 1px solid rgba(0, 0, 0, 0.08);
        border-radius: 8px;
        padding: 10px 12px;
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
        font-size: 1.12rem;
        font-weight: 800;
        margin-top: 2px;
    }

    /* Studio Badges */
    .studio-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }
    .badge-suitable {
        background: #DCFCE7;
        color: #15803D;
        border: 1px solid #BBF7D0;
    }
    .badge-unsuitable {
        background: #FEE2E2;
        color: #B91C1C;
        border: 1px solid #FECACA;
    }
    .badge-amber {
        background: #FEF3C7;
        color: #B45309;
        border: 1px solid #FDE68A;
    }
    .badge-blue {
        background: #E0F2FE;
        color: #0369A1;
        border: 1px solid #BAE6FD;
    }

    /* Form inputs and buttons polish */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        border-radius: 8px !important;
        border-color: #CBD5E1 !important;
    }
    div.stButton > button {
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        transition: all 0.15s ease-in-out !important;
    }
    div.stButton > button[kind="primary"] {
        background: #16A34A !important;
        border: 1px solid #15803D !important;
        box-shadow: 0 1px 3px rgba(22, 163, 74, 0.22) !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background: #15803D !important;
        box-shadow: 0 3px 8px rgba(22, 163, 74, 0.32) !important;
    }

    /* =======================================================
       MODERN STUDIO CHAT INTERFACE DESIGN SYSTEM
       ======================================================= */
    
    /* Modern Streamlit Chat Message Styling */
    /* Outer container: completely transparent with NO border to eliminate double boundaries */
    div[data-testid="stChatMessage"] {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
        margin: 0 0 16px 0 !important;
        width: 100% !important;
    }

    /* USER MESSAGE: Pushed to the RIGHT (row-reverse) */
    div[aria-label="Chat message from user"],
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]),
    div[data-testid="stChatMessage"]:has([aria-label*="user"]) {
        display: flex !important;
        flex-direction: row-reverse !important;
        justify-content: flex-start !important;
        align-items: flex-start !important;
        gap: 10px !important;
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        margin-bottom: 14px !important;
        width: 100% !important;
    }
    /* Single clean right-aligned user bubble */
    div[aria-label="Chat message from user"] div[data-testid="stChatMessageContent"],
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) div[data-testid="stChatMessageContent"],
    div[data-testid="stChatMessage"]:has([aria-label*="user"]) div[data-testid="stChatMessageContent"] {
        background-color: #EFF6FF !important;
        border: 1.5px solid #BFDBFE !important;
        border-radius: 18px 4px 18px 18px !important;
        padding: 10px 16px !important;
        max-width: 72% !important;
        box-shadow: 0 1px 4px rgba(37, 99, 235, 0.08) !important;
    }
    div[aria-label="Chat message from user"] div[data-testid="stChatMessageContent"] *,
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) div[data-testid="stChatMessageContent"] * {
        color: #1E3A8A !important;
        font-size: 0.94rem !important;
        line-height: 1.55 !important;
    }
    div[aria-label="Chat message from user"] div[data-testid*="Avatar"],
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) div[data-testid*="Avatar"],
    div[data-testid="stChatMessage"]:has([aria-label*="user"]) div[data-testid*="Avatar"] {
        background-color: #DBEAFE !important;
        color: #1D4ED8 !important;
        border: 1px solid #BFDBFE !important;
        border-radius: 50% !important;
        flex-shrink: 0 !important;
    }

    /* ASSISTANT MESSAGE: Pushed to the LEFT (row) */
    div[aria-label="Chat message from assistant"],
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]),
    div[data-testid="stChatMessage"]:has([aria-label*="assistant"]) {
        display: flex !important;
        flex-direction: row !important;
        justify-content: flex-start !important;
        align-items: flex-start !important;
        gap: 12px !important;
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        margin-bottom: 18px !important;
        width: 100% !important;
    }
    /* Single clean left-aligned assistant card */
    div[aria-label="Chat message from assistant"] div[data-testid="stChatMessageContent"],
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) div[data-testid="stChatMessageContent"],
    div[data-testid="stChatMessage"]:has([aria-label*="assistant"]) div[data-testid="stChatMessageContent"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-left: 4px solid #16A34A !important;
        border-radius: 4px 18px 18px 18px !important;
        padding: 14px 18px !important;
        max-width: 85% !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04) !important;
    }
    div[aria-label="Chat message from assistant"] div[data-testid="stChatMessageContent"] *,
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) div[data-testid="stChatMessageContent"] * {
        color: #1E293B !important;
        font-size: 0.94rem !important;
        line-height: 1.65 !important;
    }

    /* Markdown List & Bullet Point Typography inside chat messages */
    div[data-testid="stChatMessageContent"] ul,
    div[data-testid="stChatMessageContent"] ol {
        margin: 8px 0 10px 0 !important;
        padding-left: 22px !important;
    }
    div[data-testid="stChatMessageContent"] li {
        margin-bottom: 8px !important;
        line-height: 1.6 !important;
        color: #1E293B !important;
    }
    div[data-testid="stChatMessageContent"] li strong {
        color: #0F172A !important;
        font-weight: 700 !important;
    }
    div[data-testid="stChatMessageContent"] p {
        margin-bottom: 8px !important;
        line-height: 1.65 !important;
    }
    div[aria-label="Chat message from assistant"] div[data-testid*="Avatar"],
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) div[data-testid*="Avatar"],
    div[data-testid="stChatMessage"]:has([aria-label*="assistant"]) div[data-testid*="Avatar"] {
        background-color: #DCFCE7 !important;
        color: #15803D !important;
        border: 1px solid #86EFAC !important;
        border-radius: 50% !important;
        flex-shrink: 0 !important;
    }

    /* Floating Chat Input with Smooth Focus Ring */
    div[data-testid="stChatInput"] {
        border-radius: 28px !important;
        border: 1.5px solid #CBD5E1 !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.06) !important;
        background-color: #FFFFFF !important;
        padding: 4px 10px !important;
        transition: all 0.2s ease-in-out !important;
    }
    div[data-testid="stChatInput"]:focus-within {
        border-color: #16A34A !important;
        box-shadow: 0 4px 24px rgba(22, 163, 74, 0.18) !important;
    }
    div[data-testid="stChatInput"] textarea {
        font-size: 0.92rem !important;
        color: #1E293B !important;
        line-height: 1.5 !important;
    }
    div[data-testid="stChatInput"] button {
        border-radius: 9999px !important;
        background-color: #16A34A !important;
        color: #FFFFFF !important;
        transition: transform 0.15s ease !important;
    }
    div[data-testid="stChatInput"] button:hover {
        background-color: #15803D !important;
        transform: scale(1.06) !important;
    }
    </style>
""", unsafe_allow_html=True)


# ── 2. OLLAMA HEALTH CHECK ─────────────────────────────
@st.cache_data(ttl=30, show_spinner=False)
def check_ollama_health() -> bool:
    """Check if the local Ollama is reachable."""
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
def load_catboost(mtime: float = 0.0):
    """Load the trained CatBoost classifier from disk, invalidating cache on model file change."""
    try:
        model = CatBoostClassifier()
        model.load_model(config.MODEL_PATH)
        logger.info("CatBoost model loaded from %s (mtime: %s)", config.MODEL_PATH, mtime)
        return model
    except FileNotFoundError:
        logger.error("Model file not found at %s", config.MODEL_PATH)
        return None
    except Exception as exc:
        logger.error("Failed to load CatBoost model: %s", exc)
        return None


hotspot_file = os.path.join(config.DATA_DIR, "regional_hotspots.json")
hotspot_mtime = os.path.getmtime(hotspot_file) if os.path.exists(hotspot_file) else 0.0

@st.cache_data
def load_regional_hotspots(mtime: float = 0.0):
    hotspot_path = os.path.join(config.DATA_DIR, "regional_hotspots.json")
    if os.path.exists(hotspot_path):
        with open(hotspot_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


# --- 4. LOAD RAG PIPELINE ---
@st.cache_resource
def initialize_rag(kb_mtime: float = 0.0):
    """Build or reload the LlamaIndex query engine and vector store index."""
    if not ollama_online:
        logger.error("Ollama offline — skipping RAG init.")
        return None, None
    query_engine, index = build_query_engine(
        persist_dir=config.STORAGE_DIR,
        knowledge_base_dir=config.KNOWLEDGE_BASE_DIR,
        similarity_top_k=config.RAG_SIMILARITY_TOP_K,
    )
    return query_engine, index


# Initialize engines with automatic model & knowledge base mtime cache invalidation
model_mtime = os.path.getmtime(config.MODEL_PATH) if os.path.exists(config.MODEL_PATH) else 0.0
kb_files = [os.path.join(config.KNOWLEDGE_BASE_DIR, f) for f in os.listdir(config.KNOWLEDGE_BASE_DIR)] if os.path.isdir(config.KNOWLEDGE_BASE_DIR) else []
kb_mtime = max((os.path.getmtime(p) for p in kb_files if os.path.isfile(p)), default=0.0)

cb_model = load_catboost(model_mtime)
query_engine, rag_index = initialize_rag(kb_mtime)

# --- 5. SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_analytics_report" not in st.session_state:
    st.session_state.last_analytics_report = None
if "current_assessment" not in st.session_state:
    st.session_state.current_assessment = None
if "preset_soil" not in st.session_state:
    st.session_state.preset_soil = "Sandy Clay / Loam"
if "preset_climate" not in st.session_state:
    st.session_state.preset_climate = "Sub-Humid / Bimodal (700–1400 mm/yr)"
if "preset_moisture" not in st.session_state:
    st.session_state.preset_moisture = "Adequate (Irrigated / In-Season Rain)"
if "preset_hydro" not in st.session_state:
    st.session_state.preset_hydro = "Normal Slope / Well-Drained"
if "preset_salinity" not in st.session_state:
    st.session_state.preset_salinity = "Non-Saline (< 2 dS/m)"
if "preset_hazard" not in st.session_state:
    st.session_state.preset_hazard = "Shallow Slope / Embankment Slip (0.3–1.5 m)"
if "selected_hotspot_id" not in st.session_state:
    st.session_state.selected_hotspot_id = "ke_voi"
if "custom_coord_site" not in st.session_state:
    st.session_state.custom_coord_site = None
if "custom_assessment_result" not in st.session_state:
    st.session_state.custom_assessment_result = None
if "coord_lat" not in st.session_state:
    st.session_state.coord_lat = -1.2921
if "coord_lon" not in st.session_state:
    st.session_state.coord_lon = 36.8219
if "pan_to_custom" not in st.session_state:
    st.session_state.pan_to_custom = False


def check_hard_exclusion_boundaries(inputs_dict: dict) -> tuple:
    """
    Check if inputs violate non-negotiable physical / geotechnical exclusion boundaries
    from Sheet 2 (10 AI Boundary Parameters) and the 60 Empirical Failure Cases.
    """
    soil = str(inputs_dict.get("soil_mechanical_class", inputs_dict.get("Soil type/structure", ""))).lower()
    climate = str(inputs_dict.get("climate_rainfall_zone", inputs_dict.get("climate", ""))).lower()
    moisture = str(inputs_dict.get("establishment_moisture", inputs_dict.get("watering regime", ""))).lower()
    hydro = str(inputs_dict.get("hydrological_stress", "")).lower()
    salinity = str(inputs_dict.get("salinity_tier", inputs_dict.get("salinity", ""))).lower()
    hazard = str(inputs_dict.get("geotechnical_hazard", inputs_dict.get("Application", ""))).lower()

    if "bedrock" in soil or "hardpan" in soil:
        return True, "Mechanical Root Refusal on Solid Bedrock / Hardpan", (
            "Continuous solid bedrock or indurated petroferric hardpan (<0.2 m depth) physically blocks vertical "
            "geotropic root penetration, providing zero mechanical tensile anchoring. Bioengineering cannot supersede "
            "solid rock mechanics without mechanical rock drilling, pocket-augering, or structural soil nails (Liu et al. 2007; EIAR 2021)."
        )
    if "permafrost" in climate or "freezing" in climate or "permafrost" in soil:
        return True, "Thermal Crown Death / Ground Frost Freeze", (
            "Prolonged ground frost deeper than 10–15 cm kills vetiver crowns. Vetiver cannot survive continental permafrost or frozen ground (Xia et al. 2003)."
        )
    if "alpine" in climate or ">2,200 m" in climate:
        return True, "High-Altitude Cold Vegetative Growth Arrest", (
            "Sustained ambient temperatures below 10–12°C at elevations >2,200 m inhibit C4 photosynthetic enzymes and arrest root elongation. "
            "Substitute or companion-plant with cold-hardy montane tussock grasses (NEMA Uganda 2020; Rodriguez & Diaz 2005)."
        )
    if "extreme hypersaline" in salinity or "> 16" in salinity or float(inputs_dict.get("est_salinity_ec_ds_m", 0.0)) > 16.0:
        return True, "Hyper-Saline Osmotic Collapse & Ionic Toxicity", (
            "Soil electrical conductivity exceeds physiological threshold (>16 dS/m / SAR > 25), causing fatal reverse osmotic desiccation (Truong 1994; Mumo et al. 2022)."
        )
    if "submergence" in hydro or "inundation" in hydro or "flooded" in hydro:
        return True, "Anoxic Crown Rot & Tissue Suffocation", (
            "Continuous total submergence exceeding 21–30 days drowns leaf crowns. Leaf foliage must remain aerated (Le & Truong 2004; LVBC 2021)."
        )
    if "flash" in hydro or "high-velocity" in hydro or ">2.5" in hydro:
        return True, "Juvenile Hydraulic Shear Washout", (
            "Hydraulic shear velocities >2.5 m/s scour unanchored juvenile slips before 4–6 months root maturity. "
            "Deploy pinned biodegradable coir/jute netting and rock check-dams (Truong et al. 2002; KARI 2018)."
        )
    if "severe drought" in moisture or "no water" in moisture or "zero water" in moisture:
        return True, "Establishment Desiccation Mortality (Zero Moisture / Drought Spell)", (
            "Transplanted bare-root vetiver slips suffer 85–100% desiccation mortality when planted under zero watering or severe drought conditions, "
            "particularly on steep cut-slopes in arid or semi-arid zones (Hengchaovanich 1999; KALRO Katumani 2019; Al-Otaibi 2014). "
            "Fresh slips lack an established root column to tap deep groundwater. "
            "Mitigation Protocol: (1) Mandatory establishment irrigation during initial 30–60 days; "
            "(2) Incorporate cross-linked polyacrylamide hydrogel (5–10 g/m) into furrow backfill; "
            "(3) Substitute bare-root slips with containerized polybag nursery seedlings with pre-formed root plugs."
        )
    if "deep rotational" in hazard or "slip plane > 2" in hazard:
        return True, "Deep Rotational Shear Plane Exceedance (> 2.0 m)", (
            "Rotational shear failure surface is deeper than 2.0 m, completely bypassing the biological root reinforcement envelope. "
            "Vetiver cannot stabilize deep slope mass movements as a standalone solution (Hengchaovanich 1998; KeRRA 2022). "
            "Mitigation Protocol: Structural steel soil nails (L >= 4–6 m), driven micropiles, or anchored gabions required."
        )

    return False, "", ""


def calculate_biomechanical_metrics(inputs: dict) -> dict:
    """
    Computes dynamic geotechnical and biomechanical metrics based on:
    - USCS Soil Mechanical Class penetrability & bulk density
    - Wu-Waldron (1977) root reinforcement model (cr = 1.2 * Tr * RAR)
    - Infinite Slope Limit Equilibrium Factor of Safety Boost (delta_FoS)
    """
    soil_l = str(inputs.get("soil_mechanical_class", "")).lower()
    haz_l = str(inputs.get("geotechnical_hazard", "")).lower()
    moisture_l = str(inputs.get("establishment_moisture", "")).lower()
    ec = float(inputs.get("est_salinity_ec_ds_m", 0.5))

    # 1. Soil Mechanical Parameters & Base Root Elongation
    if "tailing" in soil_l or "spoil" in soil_l:
        base_d = 1.6
        rar = 0.00014
        gamma = 19.5
    elif "black cotton" in soil_l or "heavy" in soil_l or "expansive" in soil_l:
        base_d = 2.0
        rar = 0.00017
        gamma = 19.0
    elif "laterite" in soil_l or "saprolite" in soil_l:
        base_d = 2.5
        rar = 0.00022
        gamma = 18.5
    elif "coarse sand" in soil_l or "pumice" in soil_l:
        base_d = 3.3
        rar = 0.00020
        gamma = 16.5
    elif "hardpan" in soil_l or "bedrock" in soil_l:
        base_d = 0.15
        rar = 0.00002
        gamma = 22.0
    else:  # Sandy Clay / Loam
        base_d = 2.8
        rar = 0.00026
        gamma = 18.0

    # Moisture & Salinity Stress Multipliers
    if "drought" in moisture_l or "zero" in moisture_l:
        base_d *= 0.75
        rar *= 0.80
    if ec > 8.0:
        base_d *= 0.85
        rar *= 0.85

    d_min = round(base_d * 0.90, 1)
    d_max = round(base_d * 1.12, 1)

    # 2. Wu-Waldron Root Reinforcement Model (Tr = 75–85 MPa)
    tr_kpa = 75000.0  # 75 MPa
    cr_surface = 1.2 * tr_kpa * rar  # Additional root shear cohesion in kPa

    # 3. Target Slip Depth & Slope Kinematics
    if "surficial" in haz_l or "< 0.3" in haz_l:
        z = 0.25
        beta_deg = 30.0
    elif "riverbank" in haz_l:
        z = 0.65
        beta_deg = 32.0
    elif "shallow" in haz_l or "0.3" in haz_l:
        z = 0.90
        beta_deg = 35.0
    elif "tailing" in haz_l:
        z = 1.00
        beta_deg = 30.0
    elif "deep" in haz_l or "> 2" in haz_l:
        z = 2.80
        beta_deg = 35.0
    else:
        z = 0.85
        beta_deg = 33.0

    beta = math.radians(beta_deg)
    tau_driving = gamma * z * math.sin(beta) * math.cos(beta)

    # Effective root envelope (roots must cross shear plane to provide tensile anchoring)
    effective_root_envelope = base_d * 0.78
    if z < effective_root_envelope and tau_driving > 0.1:
        depth_decay = math.exp(-1.2 * (z / base_d))
        cr_at_z = cr_surface * depth_decay
        raw_boost = min((cr_at_z / tau_driving) * 100 * 0.22, 52.0)
        boost_min = max(int(round(raw_boost * 0.88)), 10)
        boost_max = min(int(round(raw_boost * 1.15)), 60)
        if boost_min >= boost_max:
            boost_max = boost_min + 8
        fos_str = f"+{boost_min}% to +{boost_max}%"
        cr_disp = f"cr ≈ +{cr_at_z:.1f} kPa"
    else:
        fos_str = "0% (Slip Exceeds Roots)"
        cr_disp = "cr ≈ 0 kPa"

    # 4. Domain-Specific Technical Specifications (Toxins, Hydraulics, Sediment, Mechanics)
    if "tailing" in haz_l or "spoil" in haz_l:
        domain_type = "phytoremediation"
        kpi1_label = "Heavy Metal Sequestration"
        kpi1_val = "85% – 95% (BCF > 15)"
        kpi2_label = "Target Toxins Buffer"
        kpi2_val = "Pb, As, Cd, Cu, Zn, CN⁻"
        kpi3_label = "Acid Drainage Tolerance"
        kpi3_val = "pH ≥ 3.5 (Al³⁺ Sat 70%)"
        summary_desc = (
            "Substrate phytostabilization capacity: Vetiver isolates toxic heavy metals (Pb, As, Cd, Cu, Zn) within the root matrix "
            "(Bioconcentration Factor BCF > 15, Translocation Factor TF < 0.18), preventing upward foliar translocation and stopping toxic leachate plume percolation."
        )
    elif "riverbank" in haz_l:
        domain_type = "hydrodynamic"
        kpi1_label = "Max Permissible Velocity"
        kpi1_val = "3.5 – 4.5 m/s"
        kpi2_label = "Tractive Shear Resistance"
        kpi2_val = "τcrit ≥ 280 N/m²"
        kpi3_label = "Manning's Roughness (n)"
        kpi3_val = "0.040 – 0.075"
        summary_desc = (
            "Hydrodynamic fluvial protection: Interlocking root net armors bank toes against subaqueous scour up to 4.5 m/s, "
            "dissipating boundary shear stress and suppressing rotational riverbank collapse."
        )
    elif "surficial" in haz_l or "< 0.3" in haz_l:
        domain_type = "erosion_control"
        kpi1_label = "Soil Loss Reduction"
        kpi1_val = "> 92% (< 3.0 t/ha/yr)"
        kpi2_label = "Runoff Attenuation"
        kpi2_val = "–50% to –65% Volume"
        kpi3_label = "Sediment Trap Efficiency"
        kpi3_val = "90% – 98% Retained"
        summary_desc = (
            "Hydrological runoff & sediment mitigation: Dense vegetative hedgerows form natural living terraces, "
            "reducing surface erosion from >50 t/ha/yr to <3 t/ha/yr while filtering 95%+ of overland sediment wash."
        )
    else:  # Shallow slope or general
        domain_type = "slope_mechanics"
        kpi1_label = "Root Depth Potential"
        kpi1_val = f"{d_min:.1f} – {d_max:.1f} m"
        kpi2_label = "Wu-Waldron Added Cohesion"
        kpi2_val = cr_disp
        kpi3_label = "Safety Factor Boost"
        kpi3_val = fos_str
        summary_desc = (
            f"Geotechnical slope stabilization: Dense root reinforcement envelope down to {d_max:.1f} m penetrates across "
            f"the shallow failure plane (z = {z:.2f} m), generating {cr_disp} added tensile shear strength and boosting slope FoS by {fos_str}."
        )

    return {
        "root_depth": f"{d_min:.1f} – {d_max:.1f} m",
        "tensile_strength": "75 – 85 MPa",
        "root_cohesion": cr_disp,
        "fos_boost": fos_str,
        "slip_depth_m": z,
        "domain_type": domain_type,
        "kpi1_label": kpi1_label,
        "kpi1_val": kpi1_val,
        "kpi2_label": kpi2_label,
        "kpi2_val": kpi2_val,
        "kpi3_label": kpi3_label,
        "kpi3_val": kpi3_val,
        "summary_desc": summary_desc,
    }


def run_assessment(inputs_dict: dict) -> dict:
    """Executes CatBoost prediction, probability, SHAP, and report generation with domain guardrails."""
    if not cb_model:
        return None

    # Ensure est_salinity_ec_ds_m is populated
    if "est_salinity_ec_ds_m" not in inputs_dict:
        tier = inputs_dict.get("salinity_tier", inputs_dict.get("salinity", "Non-Saline (< 2 dS/m)"))
        inputs_dict["est_salinity_ec_ds_m"] = config.SALINITY_TIER_TO_ECE.get(tier, 0.5)

    # Format DataFrame matching exact MODEL_FEATURES for CatBoost
    model_row = {}
    for feat in config.MODEL_FEATURES:
        if feat in inputs_dict:
            model_row[feat] = [inputs_dict[feat]]
        else:
            model_row[feat] = [0.5] if feat in config.NUM_FEATURES else ["Unknown"]

    input_data = pd.DataFrame(model_row)
    for num_col in config.NUM_FEATURES:
        if num_col in input_data.columns:
            input_data[num_col] = pd.to_numeric(input_data[num_col], errors="coerce").fillna(0.5)
    for cat_col in config.CAT_FEATURES:
        if cat_col in input_data.columns:
            input_data[cat_col] = input_data[cat_col].astype(str)

    pred = int(cb_model.predict(input_data)[0])
    prob = float(cb_model.predict_proba(input_data)[0][1] * 100)

    # Geotechnical Domain Exclusion Guardrail Override
    is_exclusion, excl_title, excl_details = check_hard_exclusion_boundaries(inputs_dict)
    if is_exclusion:
        pred = 0
        prob = min(prob, 8.5)  # Enforce high failure confidence

    # Compute dynamic biomechanical geotechnical metrics
    biomechanics = calculate_biomechanical_metrics(inputs_dict)

    explanation = explain_prediction(cb_model, input_data)
    exp_df = explanation["explanation_df"]
    report_md = generate_suitability_report(inputs_dict, pred, prob, exp_df, is_exclusion, excl_title, excl_details, biomechanics)
    return {
        "inputs": inputs_dict,
        "prediction": pred,
        "probability": prob,
        "explanation": explanation,
        "exp_df": exp_df,
        "report_md": report_md,
        "is_exclusion": is_exclusion,
        "excl_title": excl_title,
        "excl_details": excl_details,
        "biomechanics": biomechanics,
    }


def get_active_site_context() -> Optional[Dict[str, Any]]:
    """Extracts normalized active assessment context from session state for LLM grounding."""
    assessment = st.session_state.get("current_assessment")
    custom_site = st.session_state.get("custom_coord_site")

    if not assessment and not custom_site:
        return None

    inputs = assessment.get("inputs", {}) if assessment else {}

    loc_name = inputs.get("location_name") or inputs.get("county")
    if not loc_name and custom_site:
        loc_name = custom_site.get("name") or custom_site.get("county")
    if not loc_name:
        loc_name = "Configured Engineering Site"

    elev = inputs.get("elevation_m")
    if elev is None and custom_site:
        elev = custom_site.get("elevation_m")
    elev_str = f"{elev:.0f}" if isinstance(elev, (int, float)) else str(elev or "N/A")

    soil = inputs.get("soil_mechanical_class") or (custom_site.get("soil_class") if custom_site else "N/A")
    climate = inputs.get("climate_rainfall_zone") or (custom_site.get("rainfall_zone") if custom_site else "N/A")
    moisture = inputs.get("establishment_moisture") or (custom_site.get("moisture_def") if custom_site else "N/A")
    hydro = inputs.get("hydrological_stress", "Normal Slope / Well-Drained")
    salinity = inputs.get("salinity_tier") or (custom_site.get("salinity") if custom_site else "Non-Saline")
    hazard = inputs.get("geotechnical_hazard") or (custom_site.get("civil_hazard") if custom_site else "N/A")

    pred = assessment.get("prediction") if assessment else None
    prob = assessment.get("probability", 0.0) if assessment else 0.0
    excl_title = assessment.get("excl_title", "None — Within safe biophysical envelope") if assessment else "N/A"
    biomechanics = assessment.get("biomechanics", {}) if assessment else {}

    return {
        "location_name": loc_name,
        "county": inputs.get("county", loc_name),
        "elevation_m": elev_str,
        "soil_mechanical_class": soil,
        "climate_rainfall_zone": climate,
        "establishment_moisture": moisture,
        "hydrological_stress": hydro,
        "salinity_tier": salinity,
        "geotechnical_hazard": hazard,
        "prediction": pred,
        "probability": prob,
        "excl_title": excl_title,
        "biomechanics": biomechanics,
    }


# --- REPORT GENERATION HELPERS ---
def generate_suitability_report(
    inputs: dict,
    prediction: int,
    probability: float,
    exp_df: pd.DataFrame,
    is_exclusion: bool = False,
    excl_title: str = "",
    excl_details: str = "",
    biomechanics: dict = None,
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
    ]
    if prediction == 1:
        bio = biomechanics if biomechanics else calculate_biomechanical_metrics(inputs)
        lines.extend([
            f"- **Estimated Vertical Root Depth:** `{bio['root_depth']}`",
            f"- **Mean Root Tensile Strength:** `{bio['tensile_strength']}`",
            f"- **Wu-Waldron Additional Shear Cohesion:** `{bio['root_cohesion']}`",
            f"- **Factor of Safety (ΔFoS) Boost:** `{bio['fos_boost']}` (at design slip depth $z = {bio['slip_depth_m']:.2f}\\text{{ m}}$)",
            f"- **Domain Technical Specification:** {bio.get('summary_desc', 'N/A')}",
            f"- **Technical Metric 1 ({bio.get('kpi1_label', 'Metric 1')}):** `{bio.get('kpi1_val', 'N/A')}`",
            f"- **Technical Metric 2 ({bio.get('kpi2_label', 'Metric 2')}):** `{bio.get('kpi2_val', 'N/A')}`",
            f"- **Technical Metric 3 ({bio.get('kpi3_label', 'Metric 3')}):** `{bio.get('kpi3_val', 'N/A')}`",
        ])
    elif is_exclusion:
        lines.extend([
            f"- **Diagnosed Critical Failure Limit:** `{excl_title}`",
            f"- **Biomechanical Failure Mechanism:** {excl_details}",
        ])

    lines.extend([
        "\n## 2. Configured Site Parameters",
        f"- **Soil Mechanical Class:** `{inputs.get('soil_mechanical_class', inputs.get('Soil type/structure', 'N/A'))}`",
        f"- **Precipitation Zone:** `{inputs.get('climate_rainfall_zone', inputs.get('climate', 'N/A'))}`",
        f"- **Establishment Moisture:** `{inputs.get('establishment_moisture', inputs.get('watering regime', 'N/A'))}`",
        f"- **Hydrological Stress:** `{inputs.get('hydrological_stress', 'Normal')}`",
        f"- **Salinity Tier:** `{inputs.get('salinity_tier', inputs.get('salinity', 'N/A'))}`",
        f"- **Geotechnical Hazard Mode:** `{inputs.get('geotechnical_hazard', inputs.get('Application', 'N/A'))}`",
        f"- **Civil Project Scope:** `{inputs.get('geotechnical_hazard_display', inputs.get('Application', inputs.get('geotechnical_hazard', 'N/A')))}`",
        "\n## 3. TreeSHAP Feature Attributions",
        "The table below indicates the mathematical contribution of each site condition to the final verdict:\n",
        "| Site Parameter | Input Value | SHAP Impact (Log-Odds) | Directional Effect |",
        "| :--- | :--- | :--- | :--- |",
    ])
    for _, row in exp_df.iterrows():
        lines.append(f"| {row['Feature']} | {row['Input Value']} | {row['SHAP Impact']:+.4f} | {row['Effect']} |")

    rec_lines = [
        "\n## 4. Engineering Recommendations & Field Protocols",
    ]
    if prediction == 1:
        rec_lines.extend([
            "### 4.1 Optimal Establishment Protocol",
            "- **Initial Moisture:** Provide adequate irrigation or in-season rain during the first 30–60 days until the taproot column anchors past 1.0 m.",
            "- **Contour Spacing:** Spacing between slips in a row should be 10–15 cm to form a tight, closed sediment-trapping hedge within 6 months.",
            "\n### 4.2 Universal Field Stewardship Standards",
            "- **Solar Exposure:** Strictly obligate C4 heliophyte requiring >= 6 hours direct daily sunlight (zero canopy shading).",
            "- **Livestock Exclusion:** Enforce 12-month perimeter fencing to prevent browsing of tender juvenile tillers.",
            "- **Termite Protection:** Dip roots in neem oil solution or Metarhizium anisopliae biopesticide prior to planting in vulnerable soils.",
            "- **Certified Quality:** Use 3-tiller slips with trimmed roots (10 cm) and pruned leaves (15–20 cm).",
        ])
    else:
        remed = get_two_tier_remediations({
            "is_exclusion": is_exclusion,
            "excl_title": excl_title,
            "inputs": inputs,
        })
        rec_lines.extend([
            f"### 4.1 Tier 1: {remed['primary_title']}",
            f"- **Biomechanical Failure Mechanism:** {remed['primary_mech']}",
            "- **Prescribed Action Protocols:**",
        ])
        for proto_text, _ in remed['protocols']:
            clean_proto = proto_text.replace("**", "")
            rec_lines.append(f"  * {clean_proto}")

        rec_lines.extend([
            "\n### 4.2 Tier 2: Universal Field Stewardship & Biological Prerequisites",
            "- **Solar Exposure:** Strictly obligate C4 heliophyte requiring >= 6 hours direct daily sunlight (zero canopy shading).",
            "- **Livestock Exclusion:** Enforce 12-month perimeter fencing to prevent browsing of tender juvenile tillers.",
            "- **Termite Protection:** Dip roots in neem oil solution or Metarhizium anisopliae biopesticide prior to planting in vulnerable soils.",
            "- **Certified Quality:** Use 3-tiller slips with trimmed roots (10 cm) and pruned leaves (15–20 cm).",
        ])

    rec_lines.extend([
        "\n### 4.3 Long-Term Monitoring & Maintenance",
        "- Inspect root development, contour hedge closure, and localized rill formation at 3, 6, and 12-month intervals.",
        "\n---",
        "*Report compiled by Vetiver AI Decision Support Platform.*"
    ])
    lines.extend(rec_lines)
    return "\n".join(lines)


def get_two_tier_remediations(assessment: dict) -> dict:
    """
    Constructs a two-tier remediation strategy:
    1. Targeted Primary Engineering Prescription (Site-Specific Deep Dive)
    2. Universal Field Stewardship & Biological Prerequisites (Shading, Grazing, Termites, Quality)
    """
    is_exclusion = assessment.get("is_exclusion", False)
    excl_title = assessment.get("excl_title", "")
    title_lower = excl_title.lower()
    inputs = assessment.get("inputs", {})
    hydro = str(inputs.get("hydrological_stress", "")).lower()
    soil = str(inputs.get("soil_mechanical_class", "")).lower()
    salinity = str(inputs.get("salinity_tier", "")).lower()
    hazard = str(inputs.get("geotechnical_hazard", "")).lower()
    moisture = str(inputs.get("establishment_moisture", "")).lower()

    # 1. Deep Rotational Slip Plane (>2.0 m)
    if "rotational" in title_lower or "slip plane" in title_lower or "deep" in title_lower or "rotational" in hazard:
        primary_title = "Deep Structural Nail Pinning (>2.0 m Slip Depth)"
        primary_mech = (
            "Rotational shear failure planes deeper than 2.0 m completely bypass the biological root reinforcement envelope (2.0–3.0 m). "
            "Vetiver cannot arrest deep mass movements as a standalone solution (Hengchaovanich 1998; KeRRA 2022)."
        )
        protocols = [
            ("**Pattern Soil Nailing / Micropiles:** Install structural steel soil nails (L = 4–6 m, diameter 25–32 mm) grouted into stable bedrock or deep competent shear horizons.", ":material/carpenter:"),
            ("**Anchored Gabion Retaining Toe:** Construct anchored gabion toe buttresses to counteract deep rotational overturning moments at the base of the slope.", ":material/view_in_ar:"),
            ("**Living Surficial Bio-Armor:** Deploy Vetiver hedgerows across the slope face strictly as a bio-veneer to eliminate surficial erosion, reduce stormwater infiltration, and prevent face sloughing between nails.", ":material/grass:"),
        ]
    # 2. Prolonged Submergence / Inundation
    elif "submergence" in title_lower or "inundation" in title_lower or "anoxic" in title_lower or "crown rot" in title_lower or "submergence" in hydro or "inundation" in hydro:
        primary_title = "Hydrodynamic Submergence Mitigation (>21 Days Inundation)"
        primary_mech = (
            "Vetiver roots can thrive under continuous water immersion indefinitely, but the vegetative leaf crown requires atmospheric oxygen. "
            "Continuous crown submergence exceeding 21–30 days induces anaerobic tissue suffocation and irreversible crown rot (Le & Truong 2004; LVBC 2021)."
        )
        protocols = [
            ("**Raised Contour Berms / Bund Planting:** Construct 30–50 cm high compacted earth bunds along planting contours so crowns and the upper 15 cm of leaf canopy remain elevated above peak stagnant flood levels.", ":material/terrain:"),
            ("**Drawdown Window Scheduling:** Mobilize field transplanting strictly at the start of seasonal water recession (drawdown) to allow roots 60–90 days of anchoring before prolonged inundation.", ":material/calendar_month:"),
            ("**Surface Relief Swales & Cutoffs:** Excavate shallow drainage swales (slope gradient S >= 0.5%) or aggregate French drains to evacuate standing backwater within 7–10 days post-storm.", ":material/water_drop:"),
        ]
    # 6. Severe Drought / Zero Irrigation
    elif "drought" in title_lower or "desiccation" in title_lower or "zero water" in moisture or "severe drought" in moisture:
        primary_title = "Establishment Hydration & Hydrogel Buffering"
        primary_mech = (
            "Transplanting bare-root slips under zero irrigation or severe moisture deficit causes 85–100% transplant mortality "
            "before roots can establish contact with subsoil capillary moisture (Hengchaovanich 1999; KALRO 2019)."
        )
        protocols = [
            ("**Mandatory 60-Day Establishment Watering:** Apply 2–3 L of water per slip every 2–3 days during the initial 30 days, tapering to weekly watering for the subsequent 30 days until root depth reaches 1.0 m.", ":material/water_drop:"),
            ("**Polyacrylamide Hydrogel Polymers:** Incorporate cross-linked agricultural hydrogel (5–10 g/linear meter) into trench backfill to retain moisture directly around juvenile root crowns.", ":material/bubble_chart:"),
            ("**Polybag Pre-Rooted Nursery Slips:** Deploy containerized polybag nursery plants with pre-formed root cores instead of vulnerable bare-root slips.", ":material/yard:"),
        ]
    # 7. Ground Frost / High Altitude Cold
    elif "frost" in title_lower or "freeze" in title_lower or "altitude" in title_lower or "alpine" in title_lower:
        primary_title = "Cold-Hardy Montane Stratification (<10°C / >2,200 m)"
        primary_mech = (
            "Prolonged ground frost deeper than 10–15 cm or persistent temperatures below 10–12°C halts C4 photosynthetic enzymes, "
            "causing subterranean crown freeze and vegetative death (Xia et al. 2003; NEMA Uganda 2020)."
        )
        protocols = [
            ("**Solar Microclimate Aspect Selection:** Confine plantings strictly to warm, sun-facing southern or western slope aspects, and apply 10 cm organic straw/woodchip mulch over crowns.", ":material/wb_sunny:"),
            ("**Montane Companion Species:** Interplant or companion-plant with indigenous cold-hardy montane tussock grasses (e.g. Festuca, Pennisetum clandestinum).", ":material/forest:"),
            ("**Spring Planting Windows:** Restrict planting exclusively to early spring/rainy seasons to maximize warm vegetative growth before winter frost onset.", ":material/event:"),
        ]
    # 8. Statistical Unfavorable
    else:
        primary_title = "Site Soil Conditioning & Vigor Building"
        primary_mech = (
            "Combined site soil, climate, and terrain parameters fall outside empirical stabilization baselines, indicating a high risk of poor vigor or slope instability."
        )
        protocols = [
            ("**Compost & Soil Conditioning:** Incorporate organic compost and well-rotted manure (2–4 kg/m) into planting furrows to build structural aggregates and moisture capacity.", ":material/recycling:"),
            ("**Establishment Irrigation:** Ensure regular moisture during the initial 60 days to accelerate vertical root penetration.", ":material/water_drop:"),
            ("**Nursery Slip Hardening:** Use vigorous 3-tiller slips pre-conditioned in nursery polybags to guarantee root viability.", ":material/potted_plant:"),
        ]

    stewardship = [
        {
            "icon": ":material/wb_sunny:",
            "title": "Strict Solar Exposure (Obligate Heliophyte)",
            "body": "Vetiver is an obligate C4 sun grass requiring **>= 6 hours of direct sunlight daily**. Never plant under dense tree canopies, under bridges/overpasses, or in permanent structural shade. Light deficits cause severe etiolation, chlorosis, and 90–100% mortality within 4 months.",
        },
        {
            "icon": ":material/fence:",
            "title": "12-Month Livestock Grazing Exclusion",
            "body": "Tender juvenile slips (0–12 months) are palatable to cattle, goats, and wildlife. Enforce **mandatory 12-month perimeter live fencing** (e.g. *Euphorbia tirucalli*, thorny *Acacia*, or wire mesh). Once mature, coarse siliceous leaves become unpalatable.",
        },
        {
            "icon": ":material/pest_control:",
            "title": "Termite & Subterranean Pest Defense",
            "body": "In semi-arid savannahs or dry cut-slopes, subterranean termites can sever tender young root shoots. Dip bare-root slips in an organic neem oil solution or entomopathogenic biopesticide (*Metarhizium anisopliae*) prior to planting.",
        },
        {
            "icon": ":material/verified:",
            "title": "Certified Slip Quality Standards",
            "body": "Plant certified slips with a minimum of **3 healthy tillers**, roots trimmed to **10 cm**, and leaf foliage pruned back to **15–20 cm** to minimize transplant transpiration stress and stimulate vigorous basal tillering.",
        },
    ]

    return {
        "primary_title": primary_title,
        "primary_mech": primary_mech,
        "protocols": protocols,
        "stewardship": stewardship,
    }


def get_failure_banner_details(assessment: dict) -> dict:
    """
    Returns specific badge, title, diagnostic subtitle, and tailored KPI metrics
    for the exact diagnosed failure mode rather than generic boilerplate.
    """
    is_exclusion = assessment.get("is_exclusion", False)
    excl_title = assessment.get("excl_title", "")
    title_lower = excl_title.lower()

    if "hydraulic" in title_lower or "washout" in title_lower or "velocity" in title_lower:
        return {
            "badge": "Hydrodynamic Limit Exceeded",
            "title": "VERDICT: UNFAVORABLE — JUVENILE HYDRAULIC WASHOUT",
            "subtitle": "Runoff shear velocities exceed >2.5 m/s. Unanchored juvenile slips (first 0–90 days) will be hydraulically scoured and washed downstream before anchoring taproots can develop.",
            "kpi1_title": "Tractive Shear",
            "kpi1_val": "> 2.5 m/s (Critical)",
            "kpi2_title": "Failure Mechanism",
            "kpi2_val": "Hydraulic Slip Scour",
            "kpi3_title": "Required Precursor",
            "kpi3_val": "Pinned Coir Netting / Baffles",
        }

    if "bedrock" in title_lower or "hardpan" in title_lower or "refusal" in title_lower:
        return {
            "badge": "Mechanical Root Refusal",
            "title": "VERDICT: UNFAVORABLE — SOLID BEDROCK / HARDPAN REFUSAL",
            "subtitle": "Continuous bedrock or indurated petroferric hardpan (<0.2 m depth) physically blocks vertical taproot penetration, providing zero mechanical tensile anchoring across the slip plane.",
            "kpi1_title": "Root Penetration",
            "kpi1_val": "Refusal (< 0.2 m)",
            "kpi2_title": "Failure Mechanism",
            "kpi2_val": "Planar Boundary Slide",
            "kpi3_title": "Required Precursor",
            "kpi3_val": "Tractor Ripping / Soil Nails",
        }

    if "saline" in title_lower or "osmotic" in title_lower or "salinity" in title_lower:
        return {
            "badge": "Osmotic Collapse",
            "title": "VERDICT: UNFAVORABLE — HYPER-SALINE OSMOTIC TOXICITY",
            "subtitle": "Substrate electrical conductivity exceeds physiological threshold (>16 dS/m), inducing fatal reverse osmotic desiccation and rapid leaf and crown necrosis.",
            "kpi1_title": "Substrate Salinity",
            "kpi1_val": "> 16 dS/m (Lethal)",
            "kpi2_title": "Failure Mechanism",
            "kpi2_val": "Osmotic Water Extraction",
            "kpi3_title": "Required Precursor",
            "kpi3_val": "Gypsum Dosing (5–10 t/ha)",
        }

    if "rotational" in title_lower or "slip plane" in title_lower or "deep" in title_lower:
        return {
            "badge": "Structural Envelope Exceeded",
            "title": "VERDICT: UNFAVORABLE — SHEAR PLANE EXCEEDS ROOT ENVELOPE",
            "subtitle": "Rotational slip surface is deeper than 2.0 m, completely bypassing the 2.0–3.0 m biological root reinforcement zone. Vetiver cannot stabilize deep mass movements alone.",
            "kpi1_title": "Slip Plane Depth",
            "kpi1_val": "> 2.0 m (Bypasses Roots)",
            "kpi2_title": "Failure Mechanism",
            "kpi2_val": "Deep Mass Movement",
            "kpi3_title": "Required Precursor",
            "kpi3_val": "Steel Soil Nails (L >= 4–6 m)",
        }

    if "submergence" in title_lower or "crown rot" in title_lower or "anoxic" in title_lower or "suffocation" in title_lower:
        return {
            "badge": "Anoxic Crown Suffocation",
            "title": "VERDICT: UNFAVORABLE — CONTINUOUS SUBMERGENCE ROT",
            "subtitle": "Total continuous inundation exceeding 21–30 days drowns leaf crowns, causing anaerobic tissue asphyxiation and crown rot. Leaf foliage must remain aerated.",
            "kpi1_title": "Inundation Duration",
            "kpi1_val": "> 21 Days (Exceeded)",
            "kpi2_title": "Failure Mechanism",
            "kpi2_val": "Anaerobic Crown Rot",
            "kpi3_title": "Required Precursor",
            "kpi3_val": "Surface Drainage Swales",
        }

    if "drought" in title_lower or "desiccation" in title_lower or "moisture" in title_lower:
        return {
            "badge": "Moisture Deficit Mortality",
            "title": "VERDICT: UNFAVORABLE — ESTABLISHMENT WATER DEFICIT",
            "subtitle": "Planting bare-root slips under zero irrigation or severe moisture deficit causes 85–100% transplant mortality before taproots can establish contact with subsoil water reserves.",
            "kpi1_title": "Establishment Moisture",
            "kpi1_val": "Zero / Severe Deficit",
            "kpi2_title": "Failure Mechanism",
            "kpi2_val": "Juvenile Crown Desiccation",
            "kpi3_title": "Required Precursor",
            "kpi3_val": "60-Day Watering / Hydrogel",
        }

    if "frost" in title_lower or "freeze" in title_lower or "altitude" in title_lower or "cold" in title_lower:
        return {
            "badge": "Thermal Growth Arrest",
            "title": "VERDICT: UNFAVORABLE — THERMAL ARREST / FROST KILL",
            "subtitle": "Ground frost deeper than 10–15 cm or persistent high-altitude temperatures below 10–12°C halts C4 photosynthesis and causes subterranean crown mortality.",
            "kpi1_title": "Thermal Regime",
            "kpi1_val": "Freezing / High Cold",
            "kpi2_title": "Failure Mechanism",
            "kpi2_val": "Enzymatic Growth Arrest",
            "kpi3_title": "Required Precursor",
            "kpi3_val": "Cold-Hardy Montane Grasses",
        }

    # Statistical Unsuitable (CatBoost model probability < 50%)
    return {
        "badge": "Marginal Empirical Alignment",
        "title": "VERDICT: UNFAVORABLE — HIGH FAILURE RISK",
        "subtitle": "Combined site geotechnical, climatic, and hydrological stresses fall outside proven empirical stabilization baselines, indicating a high risk of poor vigor or slope failure.",
        "kpi1_title": "Root Anchoring",
        "kpi1_val": "Compromised Potential",
        "kpi2_title": "Geotechnical Risk",
        "kpi2_val": "Elevated Failure Profile",
        "kpi3_title": "Recommended Action",
        "kpi3_val": "Substrate & Water Remediation",
    }


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
        st.error("Ollama: Offline (port 11434)", icon=":material/power_off:")
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
    st.markdown("### Model & Evidence Specifications")
    st.caption(
        "• **Evidence Base:** Strictly 184 Empirical Trials (124 Success + 60 Scientific Failure Cases)\n"
        "• **Physics Boundaries:** 10 Deterministic Biomechanical Limit Rules (Sheet 2)\n"
        "• **Feature Pipeline:** 6 Canonical Geotechnical Variables (USCS Classification)\n"
        "• **Algorithm:** Gradient Boosted Trees (CatBoost)\n"
        "• **Validation:** 5-Fold Stratified CV (80.5% Accuracy | 97.6% Recall | 87.1% F1-Score)\n"
        "• **Explainability:** Native TreeSHAP Feature Attributions\n"
        "• **Knowledge Engine:** LlamaIndex Vector RAG + Qwen 2.5 3B (60-Case Peer-Reviewed Field Register)"
    )

    if st.session_state.messages:
        chat_md = export_chat_markdown(st.session_state.messages)
        st.download_button(
            label="Export Chat Session (.md)",
            data=chat_md,
            file_name=f"vetiver_chat_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
            icon=":material/download:",
            width="stretch",
        )

    if st.button("Clear Chat History", icon=":material/delete_sweep:", width="stretch"):
        st.session_state.messages = []
        st.rerun()


hazard_display_map = {
    "Shallow Slope / Embankment Slip (0.3–1.5 m)": "Shallow Slope / Embankment Slip (0.3–1.5 m) — Road Cuts, Railway Batters, Terraces",
    "Surficial Sheet & Rill Erosion (< 0.3 m)": "Surficial Sheet & Rill Erosion (< 0.3 m) — Agricultural Contours, Gully Flanks, Topsoil Wash",
    "Riverbank / Hydrodynamic Toe Scour": "Riverbank / Hydrodynamic Toe Scour — River Channels, Canal Lining, Drainage Dikes",
    "Mine Tailings & Spoil Containment": "Mine Tailings & Spoil Containment — Tailings Dams, Landfills, Industrial Spoil",
    "Deep Rotational Slide (> 2.0 m)": "Deep Rotational Slide (> 2.0 m) — Deep Mass Landslides (Structural Nails Required)",
}

# --- 7. HYBRID UI (TABS WITH MATERIAL ICONS) ---
tab1, tab2, tab3 = st.tabs([
    ":material/analytics: Geotechnical Analytics",
    ":material/map: 3D GIS Map",
    ":material/forum: VetiverChat Assistant",
])


# =======================================================
# TAB 1: CATBOOST ANALYTICS & SHAP EXPLANATION
# =======================================================
with tab1:
    st.subheader("Geotechnical Site Suitability Assessment", anchor=False)

    # --- SITE INGESTION & REGIONAL BASELINE CENTER ---
    with st.container(border=True):
        st.markdown("##### :material/place: Fast Site Ingestion & Regional Baseline Presets")
        t1_ingest_tab1, t1_ingest_tab2 = st.tabs([
            ":material/travel_explore: Ingest from GPS Coordinates (Lat, Lon)",
            ":material/account_balance: Pre-Computed Regional Hotspots",
        ])

        with t1_ingest_tab1:
            st.caption("Enter coordinates or click a quick corridor shortcut to automatically resolve soil class, rainfall regime, elevation, and salinity into the parameter controls below:")
            t1_gc1, t1_gc2, t1_gc3 = st.columns([1.5, 1.5, 1.8])
            with t1_gc1:
                t1_in_lat = st.number_input(
                    "Latitude (°N)",
                    min_value=-90.0,
                    max_value=90.0,
                    value=float(st.session_state.coord_lat),
                    format="%.5f",
                    step=0.01,
                    key="t1_input_geo_lat"
                )
            with t1_gc2:
                t1_in_lon = st.number_input(
                    "Longitude (°E)",
                    min_value=-180.0,
                    max_value=180.0,
                    value=float(st.session_state.coord_lon),
                    format="%.5f",
                    step=0.01,
                    key="t1_input_geo_lon"
                )
            with t1_gc3:
                st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
                t1_resolve_btn = st.button(
                    "Ingest Baseline Parameters",
                    type="primary",
                    width="stretch",
                    icon=":material/download:",
                    key="btn_t1_resolve_custom_gps"
                )

            # Quick corridor pills
            st.caption("Quick Corridor Shortcuts:")
            t1_p1, t1_p2, t1_p3, t1_p4, t1_p5, t1_p6 = st.columns(6)
            if t1_p1.button("Nairobi", width="stretch", key="t1_btn_nrb"):
                st.session_state.coord_lat = -1.2921
                st.session_state.coord_lon = 36.8219
                c_site = resolve_coordinate_geotechnics(-1.2921, 36.8219)
                st.session_state.custom_coord_site = c_site
                st.session_state.preset_soil = c_site["soil_mechanical_class"]
                st.session_state.preset_climate = c_site["climate_rainfall_zone"]
                st.session_state.preset_moisture = c_site["establishment_moisture"]
                st.session_state.preset_hydro = c_site["hydrological_stress"]
                st.session_state.preset_salinity = c_site["salinity_tier"]
                st.session_state.preset_hazard = c_site["default_hazard"]
                st.session_state.pan_to_custom = True
                st.rerun()
            if t1_p2.button("Samburu", width="stretch", key="t1_btn_smb"):
                st.session_state.coord_lat = 1.0000
                st.session_state.coord_lon = 36.8219
                c_site = resolve_coordinate_geotechnics(1.0000, 36.8219)
                st.session_state.custom_coord_site = c_site
                st.session_state.preset_soil = c_site["soil_mechanical_class"]
                st.session_state.preset_climate = c_site["climate_rainfall_zone"]
                st.session_state.preset_moisture = c_site["establishment_moisture"]
                st.session_state.preset_hydro = c_site["hydrological_stress"]
                st.session_state.preset_salinity = c_site["salinity_tier"]
                st.session_state.preset_hazard = c_site["default_hazard"]
                st.session_state.pan_to_custom = True
                st.rerun()
            if t1_p3.button("Eldoret", width="stretch", key="t1_btn_eld"):
                st.session_state.coord_lat = 0.5143
                st.session_state.coord_lon = 35.2698
                c_site = resolve_coordinate_geotechnics(0.5143, 35.2698)
                st.session_state.custom_coord_site = c_site
                st.session_state.preset_soil = c_site["soil_mechanical_class"]
                st.session_state.preset_climate = c_site["climate_rainfall_zone"]
                st.session_state.preset_moisture = c_site["establishment_moisture"]
                st.session_state.preset_hydro = c_site["hydrological_stress"]
                st.session_state.preset_salinity = c_site["salinity_tier"]
                st.session_state.preset_hazard = c_site["default_hazard"]
                st.session_state.pan_to_custom = True
                st.rerun()
            if t1_p4.button("Naivasha", width="stretch", key="t1_btn_nvs"):
                st.session_state.coord_lat = -0.7172
                st.session_state.coord_lon = 36.4312
                c_site = resolve_coordinate_geotechnics(-0.7172, 36.4312)
                st.session_state.custom_coord_site = c_site
                st.session_state.preset_soil = c_site["soil_mechanical_class"]
                st.session_state.preset_climate = c_site["climate_rainfall_zone"]
                st.session_state.preset_moisture = c_site["establishment_moisture"]
                st.session_state.preset_hydro = c_site["hydrological_stress"]
                st.session_state.preset_salinity = c_site["salinity_tier"]
                st.session_state.preset_hazard = c_site["default_hazard"]
                st.session_state.pan_to_custom = True
                st.rerun()
            if t1_p5.button("Embu", width="stretch", key="t1_btn_emb"):
                st.session_state.coord_lat = -0.5342
                st.session_state.coord_lon = 37.4544
                c_site = resolve_coordinate_geotechnics(-0.5342, 37.4544)
                st.session_state.custom_coord_site = c_site
                st.session_state.preset_soil = c_site["soil_mechanical_class"]
                st.session_state.preset_climate = c_site["climate_rainfall_zone"]
                st.session_state.preset_moisture = c_site["establishment_moisture"]
                st.session_state.preset_hydro = c_site["hydrological_stress"]
                st.session_state.preset_salinity = c_site["salinity_tier"]
                st.session_state.preset_hazard = c_site["default_hazard"]
                st.session_state.pan_to_custom = True
                st.rerun()
            if t1_p6.button("Garissa", width="stretch", key="t1_btn_grs"):
                st.session_state.coord_lat = -0.4532
                st.session_state.coord_lon = 39.6461
                c_site = resolve_coordinate_geotechnics(-0.4532, 39.6461)
                st.session_state.custom_coord_site = c_site
                st.session_state.preset_soil = c_site["soil_mechanical_class"]
                st.session_state.preset_climate = c_site["climate_rainfall_zone"]
                st.session_state.preset_moisture = c_site["establishment_moisture"]
                st.session_state.preset_hydro = c_site["hydrological_stress"]
                st.session_state.preset_salinity = c_site["salinity_tier"]
                st.session_state.preset_hazard = c_site["default_hazard"]
                st.session_state.pan_to_custom = True
                st.rerun()

            if t1_resolve_btn:
                with st.spinner(f"Ingesting baseline parameters for ({t1_in_lat:.4f}°N, {t1_in_lon:.4f}°E)..."):
                    st.session_state.coord_lat = t1_in_lat
                    st.session_state.coord_lon = t1_in_lon
                    c_site = resolve_coordinate_geotechnics(t1_in_lat, t1_in_lon)
                    st.session_state.custom_coord_site = c_site
                    st.session_state.preset_soil = c_site["soil_mechanical_class"]
                    st.session_state.preset_climate = c_site["climate_rainfall_zone"]
                    st.session_state.preset_moisture = c_site["establishment_moisture"]
                    st.session_state.preset_hydro = c_site["hydrological_stress"]
                    st.session_state.preset_salinity = c_site["salinity_tier"]
                    st.session_state.preset_hazard = c_site["default_hazard"]
                    st.session_state.pan_to_custom = True
                    st.rerun()

            if st.session_state.custom_coord_site is not None:
                c_site = st.session_state.custom_coord_site
                st.success(
                    f"**Ingested {c_site['county']}, {c_site['country']}** (Elev: `{c_site['elevation_m']} m a.s.l.`) via **{c_site['data_sources']}**.\n\n"
                    f"Parameters populated in the input controls below. Feel free to customize establishment watering or engineering hazard before running analytics.",
                    icon=":material/check_circle:"
                )

        with t1_ingest_tab2:
            all_hotspots = load_regional_hotspots(hotspot_mtime)
            if all_hotspots:
                hs_quick_c1, hs_quick_c2 = st.columns([3, 1])
                with hs_quick_c1:
                    hs_quick_labels = ["-- Select a Pre-Computed Regional Hotspot (Optional) --"] + [
                        f"{h['county_region']}: {h['name']}" for h in all_hotspots
                    ]
                    sel_quick = st.selectbox(
                        "Quick Load Regional Hotspot Preset",
                        hs_quick_labels,
                        index=0,
                        label_visibility="collapsed",
                        help="Automatically load real pre-computed soil, climate, and hazard parameters for this geographic site."
                    )
                with hs_quick_c2:
                    if st.button("Apply Hotspot", icon=":material/download:", width="stretch", key="btn_apply_hotspot_preset"):
                        if sel_quick != hs_quick_labels[0]:
                            matched_h = next(h for h in all_hotspots if f"{h['county_region']}: {h['name']}" == sel_quick)
                            st.session_state.preset_soil = matched_h["soil_mechanical_class"]
                            st.session_state.preset_climate = matched_h["climate_rainfall_zone"]
                            st.session_state.preset_moisture = matched_h["establishment_moisture"]
                            st.session_state.preset_hydro = matched_h["hydrological_stress"]
                            st.session_state.preset_salinity = matched_h["salinity_tier"]
                            st.session_state.preset_hazard = matched_h["default_hazard"]
                            st.session_state.selected_hotspot_id = matched_h["id"]
                            auto_inputs = {
                                "soil_mechanical_class": matched_h["soil_mechanical_class"],
                                "climate_rainfall_zone": matched_h["climate_rainfall_zone"],
                                "establishment_moisture": matched_h["establishment_moisture"],
                                "hydrological_stress": matched_h["hydrological_stress"],
                                "salinity_tier": matched_h["salinity_tier"],
                                "est_salinity_ec_ds_m": config.SALINITY_TIER_TO_ECE.get(matched_h["salinity_tier"], 0.5),
                                "geotechnical_hazard": matched_h["default_hazard"],
                                "geotechnical_hazard_display": hazard_display_map.get(matched_h["default_hazard"], matched_h["default_hazard"]),
                            }
                            st.session_state.current_assessment = run_assessment(auto_inputs)
                            st.rerun()

    # --- CUSTOM SITE PARAMETER INPUTS ---
    with st.container(border=True):
        st.markdown("##### :material/tune: Site Geotechnical Parameters")
        col1, col2 = st.columns(2)
        with col1:
            soil_options = [
                "Sandy Clay / Loam",
                "Cohesive Clay / Nitisol / Laterite",
                "Coarse Sand / Loose Pumice",
                "Indurated Hardpan / Bedrock",
                "Mine Tailings / Industrial Spoil",
            ]
            soil_idx = soil_options.index(st.session_state.preset_soil) if st.session_state.preset_soil in soil_options else 0
            soil_type = st.selectbox(
                "Soil Mechanical Class (USCS)",
                soil_options,
                index=soil_idx,
                help="Geotechnical soil classification governing cohesion, shear strength, and root penetration rate.",
            )

            climate_options = [
                "Sub-Humid / Bimodal (700–1400 mm/yr)",
                "Humid / Tropical Monsoon (>1400 mm/yr)",
                "Semi-Arid (300–700 mm/yr)",
                "Hyper-Arid / Desert (<300 mm/yr)",
                "Alpine High-Altitude (>2,200 m)",
                "Cold Freezing / Permafrost",
            ]
            climate_idx = climate_options.index(st.session_state.preset_climate) if st.session_state.preset_climate in climate_options else 0
            climate_zone = st.selectbox(
                "Precipitation & Thermal Regime",
                climate_options,
                index=climate_idx,
                help="Regional rainfall envelope and thermal boundary.",
            )

            hazard_options = [
                "Shallow Slope / Embankment Slip (0.3–1.5 m)",
                "Surficial Sheet & Rill Erosion (< 0.3 m)",
                "Deep Rotational Slide (> 2.0 m)",
                "Riverbank / Hydrodynamic Toe Scour",
                "Mine Tailings & Spoil Containment",
            ]
            hazard_display_map = {
                "Shallow Slope / Embankment Slip (0.3–1.5 m)": "Shallow Slope / Embankment Slip (0.3–1.5 m) — Road Cuts, Railway Batters, Terraces",
                "Surficial Sheet & Rill Erosion (< 0.3 m)": "Surficial Sheet & Rill Erosion (< 0.3 m) — Agricultural Contours, Gully Flanks, Topsoil Wash",
                "Riverbank / Hydrodynamic Toe Scour": "Riverbank / Hydrodynamic Toe Scour — River Channels, Canal Lining, Drainage Dikes",
                "Mine Tailings & Spoil Containment": "Mine Tailings & Spoil Containment — Tailings Dams, Landfills, Industrial Spoil",
                "Deep Rotational Slide (> 2.0 m)": "Deep Rotational Slide (> 2.0 m) — Deep Mass Landslides (Structural Nails Required)",
            }
            hazard_idx = hazard_options.index(st.session_state.preset_hazard) if st.session_state.preset_hazard in hazard_options else 0
            hazard = st.selectbox(
                "Geotechnical Engineering Objective & Slip Depth",
                hazard_options,
                index=hazard_idx,
                format_func=lambda x: hazard_display_map.get(x, x),
                help="Target civil bio-engineering objective and estimated slip surface depth.",
            )

        with col2:
            moisture_options = [
                "Adequate (Irrigated / In-Season Rain)",
                "Severe Drought / Zero Watering",
            ]
            moisture_idx = moisture_options.index(st.session_state.preset_moisture) if st.session_state.preset_moisture in moisture_options else 0
            moisture = st.selectbox(
                "Establishment Moisture (First 30–60 Days)",
                moisture_options,
                index=moisture_idx,
                help="Crucial initial moisture status before deep root network maturation.",
            )

            hydro_options = [
                "Normal Slope / Well-Drained",
                "High-Velocity Hydraulic Runoff (>2.5 m/s)",
                "Prolonged Inundation (>21 days)",
            ]
            hydro_idx = hydro_options.index(st.session_state.preset_hydro) if st.session_state.preset_hydro in hydro_options else 0
            hydro_stress = st.selectbox(
                "Hydrological Extreme Stress",
                hydro_options,
                index=hydro_idx,
                help="Hydrodynamic tractive shear or continuous crown submergence hazards.",
            )

            salinity_options = [
                "Non-Saline (< 2 dS/m)",
                "Slight to Moderate (2–8 dS/m)",
                "High Saline (8–16 dS/m)",
                "Extreme Hypersaline (> 16 dS/m)",
            ]
            salinity_idx = salinity_options.index(st.session_state.preset_salinity) if st.session_state.preset_salinity in salinity_options else 0
            salinity = st.selectbox(
                "Substrate Salinity (ECe / Osmotic Pressure)",
                salinity_options,
                index=salinity_idx,
                help="Soil electrical conductivity (ECe). Extreme salinity (>16 dS/m) induces fatal osmotic water extraction.",
            )

        btn_c1, btn_c2 = st.columns([3, 1])
        with btn_c1:
            run_clicked = st.button("Run Geotechnical Analytics", icon=":material/play_arrow:", type="primary", width="stretch")
            if run_clicked:
                current_inputs = {
                    "soil_mechanical_class": soil_type,
                    "climate_rainfall_zone": climate_zone,
                    "establishment_moisture": moisture,
                    "hydrological_stress": hydro_stress,
                    "salinity_tier": salinity,
                    "est_salinity_ec_ds_m": config.SALINITY_TIER_TO_ECE.get(salinity, 0.5),
                    "geotechnical_hazard": hazard,
                    "geotechnical_hazard_display": hazard_display_map.get(hazard, hazard),
                }
                st.session_state.preset_soil = soil_type
                st.session_state.preset_climate = climate_zone
                st.session_state.preset_moisture = moisture
                st.session_state.preset_hydro = hydro_stress
                st.session_state.preset_salinity = salinity
                st.session_state.preset_hazard = hazard
                st.session_state.current_assessment = run_assessment(current_inputs)
                st.rerun()

        with btn_c2:
            if st.button("Reset", icon=":material/refresh:", width="stretch"):
                st.session_state.current_assessment = None
                st.rerun()

    # --- BIOMECHANICAL BOUNDARY PRESCREENER (EXPANDABLE) ---
    with st.expander(":material/speed: Biomechanical Limit State Prescreener (Quantitative Failure Envelope)", icon=":material/speed:"):
        st.markdown(
            "Quickly test numerical site parameters against the **10 scientifically documented critical failure boundaries** "
            "(*Hengchaovanich 1998, Truong 1994, Perez 2012, KeRRA 2022, MININFRA 2020, KARI 2018*):"
        )
        ps_c1, ps_c2, ps_c3 = st.columns(3)
        with ps_c1:
            test_slip_depth = st.number_input("Est. Shear Slip Depth (m)", min_value=0.2, max_value=8.0, value=1.5, step=0.2, help="Depth of critical slip surface. Vetiver biological anchor envelope is 1.5–2.0 m.")
            test_bulk_density = st.number_input("Soil Bulk Density (g/cm³)", min_value=0.8, max_value=2.3, value=1.45, step=0.05, help="Soil compaction. >1.75 g/cm³ causes severe root mechanical refusal.")
        with ps_c2:
            test_salinity = st.number_input("Salinity EC (dS/m)", min_value=0.0, max_value=35.0, value=2.0, step=1.0, help="Electrical conductivity. >16–20 dS/m causes fatal osmotic desiccation.")
            test_shade = st.slider("Overhead Tree Canopy Shading (%)", min_value=0, max_value=100, value=20, step=5, help="Solar irradiance. Vetiver is an obligate C4 grass requiring ≥40% full sunlight.")
        with ps_c3:
            test_submergence = st.number_input("Continuous Crown Submergence (Days)", min_value=0, max_value=60, value=5, step=1, help="Continuous drowning of leaf crown. >21–30 days induces anoxic crown rotting.")
            test_elevation = st.number_input("Site Elevation (m a.s.l.)", min_value=0, max_value=3500, value=1400, step=100, help="High-altitude tropical mountain slopes. >2,200 m causes low-temperature growth arrest.")

        # Prescreener boundary checks
        boundary_warnings = []
        if test_slip_depth > 2.0:
            boundary_warnings.append(
                f":material/warning: **Deep Slip Surface Failure Risk (Slip Depth {test_slip_depth:.1f} m > 2.0 m):** "
                "Shear plane exceeds the effective 1.5–2.0 m root anchorage curtain. "
                "**Mitigation (Hengchaovanich 1998, KeRRA 2022):** Deep structural failures require passive driven soil nails (L ≥ 4 m) "
                "or rock anchors, using Vetiver strictly as a surficial erosion veneer."
            )
        if test_bulk_density > 1.75:
            boundary_warnings.append(
                f":material/warning: **Mechanical Root Refusal (Bulk Density {test_bulk_density:.2f} g/cm³ > 1.75 g/cm³):** "
                "Extreme subsoil compaction physically halts downward geotropic root elongation (<0.4 m). "
                "**Mitigation (Tran et al. 2003, EIAR 2021):** Deep tractor ripping or manual augering to ≥60 cm depth along planting furrows is mandatory."
            )
        if test_salinity > 16.0:
            boundary_warnings.append(
                f":material/warning: **Hyper-Saline Osmotic Collapse (EC {test_salinity:.1f} dS/m > 16 dS/m):** "
                "Substrate salinity exceeds physiological osmotic regulation. "
                "**Mitigation (Truong 1994, Mumo et al. 2022):** Pre-treat sodic/saline substrates with agricultural gypsum (CaSO₄·2H₂O) at 5–12 t/ha and pre-harden polybag slips in saline nursery beds."
            )
        if test_shade > 60:
            boundary_warnings.append(
                f":material/warning: **Canopy Shading & Etiolation (Canopy Cover {test_shade}% > 60%):** "
                "Incident sunlight drops <40%. As an obligate C4 heliophyte, vetiver will suffer tillering arrest and biomass die-off. "
                "**Mitigation (Utomo et al. 2000, TaCRI 2017):** Enforce scheduled bi-annual lateral canopy lopping and orient rows East-West."
            )
        if test_submergence > 21:
            boundary_warnings.append(
                f":material/warning: **Anoxic Rot Under Prolonged Inundation ({test_submergence} Days > 21 Days):** "
                "Leaf crowns drowned continuously under stagnant water undergo tissue decay. "
                "**Mitigation (Le & Truong 2004, LVBC 2021):** Restrict planting to elevated river levees/berms where upper foliage remains aerated."
            )
        if test_elevation > 2200:
            boundary_warnings.append(
                f":material/warning: **High-Altitude Cold Growth Arrest (Elevation {test_elevation} m > 2,200 m):** "
                "Sustained ambient temperatures < 12°C inhibit photosynthetic C4 enzyme synthesis. "
                "**Mitigation (NEMA Uganda 2020, Rodriguez & Diaz 2005):** Combine or substitute with cold-adapted native montane tussock grasses (*Pennisetum clandestinum*, *Festuca* spp.)."
            )

        if boundary_warnings:
            st.error(f"**Critical Exclusion Boundaries Detected ({len(boundary_warnings)} Warnings):**", icon=":material/warning:")
            for w in boundary_warnings:
                st.markdown(w)
        else:
            st.success("**All Biophysical Parameters Within Safe Vetiver Operating Envelope** (Slip ≤ 2.0 m, BD ≤ 1.75 g/cm³, EC ≤ 16 dS/m, Shade ≤ 60%, Submergence ≤ 21 d, Elevation ≤ 2,200 m).", icon=":material/verified:")

    # --- RENDER ASSESSMENT RESULTS (PERSISTENT ACROSS RERUNS) ---
    if st.session_state.current_assessment is not None:
        assessment = st.session_state.current_assessment
        pred = assessment["prediction"]
        prob = assessment["probability"]
        exp_df = assessment["exp_df"]
        report_md = assessment["report_md"]

        st.divider()

        # 0. Domain Exclusion Override Alert
        if assessment.get("is_exclusion"):
            st.error(
                f"**CRITICAL BIOMECHANICAL BOUNDARY EXCLUSION: {assessment['excl_title']}**\n\n"
                f"{assessment['excl_details']}",
                icon=":material/block:",
            )

        # 1. Executive Banner
        if pred == 1:
            kpi_bio = assessment.get("biomechanics") or calculate_biomechanical_metrics(assessment.get("inputs", {}))
            st.markdown(f"""
                <div class="verdict-banner-suitable">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 16px;">
                        <div>
                            <span style="background: #16A34A; color: white; font-size: 0.75rem; font-weight: 700; padding: 4px 12px; border-radius: 100px; text-transform: uppercase; letter-spacing: 0.05em;">Positive Geotechnical Match</span>
                            <h2 style="color: #14532D; margin: 10px 0 6px 0; font-size: 1.55rem; font-weight: 800;">VERDICT: SUITABLE FOR VETIVER BIO-ENGINEERING</h2>
                            <p style="color: #166534; font-size: 0.95rem; margin: 0; line-height: 1.45;">{kpi_bio.get('summary_desc', 'Site parameters align with empirical stabilization baselines. Conditions support deep root anchoring without critical chemical or moisture limits.')}</p>
                        </div>
                        <div style="text-align: right; min-width: 140px;">
                            <div style="font-size: 2.2rem; font-weight: 900; color: #14532D; line-height: 1;">{prob:.1f}%</div>
                            <div style="font-size: 0.76rem; font-weight: 700; color: #166534; text-transform: uppercase; margin-top: 4px;">Model Confidence</div>
                        </div>
                    </div>
                    <div class="kpi-row">
                        <div class="kpi-chip">
                            <div class="kpi-chip-title">{kpi_bio.get('kpi1_label', 'Root Depth Potential')}</div>
                            <div class="kpi-chip-val" style="color: #166534;">{kpi_bio.get('kpi1_val', kpi_bio['root_depth'])}</div>
                        </div>
                        <div class="kpi-chip">
                            <div class="kpi-chip-title">{kpi_bio.get('kpi2_label', 'Mean Tensile Strength')}</div>
                            <div class="kpi-chip-val" style="color: #166534;">{kpi_bio.get('kpi2_val', kpi_bio['tensile_strength'])}</div>
                        </div>
                        <div class="kpi-chip">
                            <div class="kpi-chip-title">{kpi_bio.get('kpi3_label', 'Safety Factor Boost')}</div>
                            <div class="kpi-chip-val" style="color: #166534;">{kpi_bio.get('kpi3_val', kpi_bio['fos_boost'])}</div>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            diag = get_failure_banner_details(assessment)
            st.markdown(f"""
                <div class="verdict-banner-unsuitable">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 16px;">
                        <div>
                            <span style="background: #DC2626; color: white; font-size: 0.75rem; font-weight: 700; padding: 4px 12px; border-radius: 100px; text-transform: uppercase; letter-spacing: 0.05em;">{diag['badge']}</span>
                            <h2 style="color: #7F1D1D; margin: 10px 0 6px 0; font-size: 1.55rem; font-weight: 800;">{diag['title']}</h2>
                            <p style="color: #991B1B; font-size: 0.95rem; margin: 0; line-height: 1.45;">{diag['subtitle']}</p>
                        </div>
                        <div style="text-align: right; min-width: 140px;">
                            <div style="font-size: 2.2rem; font-weight: 900; color: #7F1D1D; line-height: 1;">{100 - prob:.1f}%</div>
                            <div style="font-size: 0.76rem; font-weight: 700; color: #991B1B; text-transform: uppercase; margin-top: 4px;">Failure Confidence</div>
                        </div>
                    </div>
                    <div class="kpi-row">
                        <div class="kpi-chip">
                            <div class="kpi-chip-title">{diag['kpi1_title']}</div>
                            <div class="kpi-chip-val" style="color: #991B1B;">{diag['kpi1_val']}</div>
                        </div>
                        <div class="kpi-chip">
                            <div class="kpi-chip-title">{diag['kpi2_title']}</div>
                            <div class="kpi-chip-val" style="color: #991B1B;">{diag['kpi2_val']}</div>
                        </div>
                        <div class="kpi-chip">
                            <div class="kpi-chip-title">{diag['kpi3_title']}</div>
                            <div class="kpi-chip-val" style="color: #991B1B;">{diag['kpi3_val']}</div>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        # 2. Scientist Mitigations (if unsuitable)
        if pred == 0:
            with st.expander(":material/medical_services: Scientist-Recommended Countermeasures & Remediations", expanded=True):
                remed = get_two_tier_remediations(assessment)

                # Tier 1: Targeted Primary Engineering Prescription
                st.markdown(f"#### :material/build: Tier 1: Targeted Engineering Prescription ({remed['primary_title']})")
                st.info(f"**Biomechanical Limit Mechanism:** {remed['primary_mech']}", icon=":material/info:")

                p_cols = st.columns(len(remed['protocols']))
                for col, (proto_text, proto_icon) in zip(p_cols, remed['protocols']):
                    with col:
                        st.markdown(f"""
                        <div style="background: #FFF5F5; border: 1px solid #FECACA; border-top: 3px solid #DC2626; border-radius: 8px; padding: 14px; height: 100%;">
                            <p style="font-size: 0.88rem; line-height: 1.45; margin: 0; color: #7F1D1D;">{proto_text}</p>
                        </div>
                        """, unsafe_allow_html=True)

                st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)

                # Tier 2: Universal Field Stewardship & Biological Prerequisites
                st.markdown("#### :material/shield: Tier 2: Universal Field Stewardship & Environmental Prerequisites")
                st.caption("Essential operational field standards required on every bioengineering installation to prevent preventable vegetative failure:")

                s_c1, s_c2 = st.columns(2)
                for idx, item in enumerate(remed['stewardship']):
                    target_col = s_c1 if idx % 2 == 0 else s_c2
                    with target_col:
                        st.markdown(f"""
                        <div style="background: #F8FAF9; border: 1px solid #E2E8F0; border-left: 3px solid #16A34A; border-radius: 8px; padding: 12px 16px; margin-bottom: 12px;">
                            <div style="font-weight: 700; font-size: 0.92rem; color: #0F172A; margin-bottom: 4px;">{item['icon']} {item['title']}</div>
                            <div style="font-size: 0.86rem; color: #475569; line-height: 1.4;">{item['body']}</div>
                        </div>
                        """, unsafe_allow_html=True)

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
            width="stretch",
        )

        # 4. Field Engineering Design & Biomechanical Suite
        st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)
        st.subheader("Field Engineering Design & Biomechanical Suite", anchor=False)
        st.caption("Interactive 2D Slope Cross-Section · Contractor Bill of Quantities (BoQ) · 12-Month Temporal Maturation Curve")

        ds_tab1, ds_tab2, ds_tab3 = st.tabs([
            ":material/architecture: 2D Slope Stability Cross-Section",
            ":material/calculate: Field Design & BoQ Material Estimator",
            ":material/timeline: Temporal Maturation & Vulnerability Window"
        ])

        # Extract active geotechnical parameters from assessment
        cur_inputs = assessment.get("inputs", {})
        cur_soil = cur_inputs.get("soil_mechanical_class", "Sandy Clay / Loam")
        bio_info = assessment.get("biomechanics", {})
        try:
            r_str = bio_info.get("root_depth", "2.8").split("–")[0].replace("m", "").strip()
            cur_root_depth = float(r_str)
        except Exception:
            cur_root_depth = 2.8

        try:
            cr_str = bio_info.get("root_cohesion", "16.5").replace("cr ≈", "").replace("cr ≈ +", "").replace("+", "").replace("kPa", "").strip()
            cur_cr = float(cr_str)
            if cur_cr <= 0.0:
                cur_cr = 16.5
        except Exception:
            cur_cr = 16.5

        with ds_tab1:
            st.caption("Interactive limit equilibrium slope analyzer based on the Wu-Waldron (1977) tensile root reinforcement model:")
            cs_col1, cs_col2, cs_col3, cs_col4 = st.columns(4)
            with cs_col1:
                ui_slope_h = st.slider("Slope Vertical Height (m)", min_value=2.0, max_value=20.0, value=6.0, step=0.5, key="ui_cs_height")
            with cs_col2:
                ui_slope_angle = st.slider("Slope Angle β (°)", min_value=20.0, max_value=60.0, value=35.0, step=1.0, help="Slope gradient (35° ≈ 1:1.4 road cut)", key="ui_cs_angle")
            with cs_col3:
                ui_wt_ratio = st.slider("Groundwater Table Level", min_value=0.0, max_value=0.8, value=0.25, step=0.05, help="Ratio of porewater table height relative to slope", key="ui_cs_wt")
            with cs_col4:
                ui_slip_z = st.slider("Target Slip Surface Depth (m)", min_value=0.4, max_value=3.5, value=1.0, step=0.1, help="Depth of critical shear failure plane", key="ui_cs_z")

            # Render cross section figure
            cs_fig, cs_stab = generate_slope_cross_section_fig(
                height_m=ui_slope_h,
                slope_angle_deg=ui_slope_angle,
                soil_class=cur_soil,
                water_table_ratio=ui_wt_ratio,
                root_depth_m=cur_root_depth,
                root_cohesion_kpa=cur_cr,
                design_slip_depth_m=ui_slip_z,
            )
            st.plotly_chart(cs_fig, width="stretch")

            # Scorecard row
            cs_m1, cs_m2, cs_m3, cs_m4 = st.columns(4)
            with cs_m1:
                st.markdown(f'''
                <div class="kpi-chip">
                    <div class="kpi-chip-title">Unreinforced FoS</div>
                    <div class="kpi-chip-val" style="color: {'#DC2626' if cs_stab['fos_unreinforced'] < 1.30 else '#16A34A'};">{cs_stab['fos_unreinforced']:.2f}</div>
                </div>
                ''', unsafe_allow_html=True)
            with cs_m2:
                st.markdown(f'''
                <div class="kpi-chip">
                    <div class="kpi-chip-title">Bio-Reinforced FoS</div>
                    <div class="kpi-chip-val" style="color: {'#16A34A' if cs_stab['fos_reinforced'] >= 1.30 else '#D97706'};">{cs_stab['fos_reinforced']:.2f}</div>
                </div>
                ''', unsafe_allow_html=True)
            with cs_m3:
                st.markdown(f'''
                <div class="kpi-chip">
                    <div class="kpi-chip-title">Safety Factor Boost</div>
                    <div class="kpi-chip-val" style="color: #16A34A;">+{cs_stab['fos_boost_pct']:.1f}%</div>
                </div>
                ''', unsafe_allow_html=True)
            with cs_m4:
                st.markdown(f'''
                <div class="kpi-chip">
                    <div class="kpi-chip-title">Stability Standard</div>
                    <div class="kpi-chip-val" style="color: {'#16A34A' if cs_stab['is_stable_reinforced'] else '#DC2626'};">{'CERTIFIED (FoS ≥ 1.30)' if cs_stab['is_stable_reinforced'] else 'SUB-STANDARD'}</div>
                </div>
                ''', unsafe_allow_html=True)

            # --- DYNAMIC LIMIT EQUILIBRIUM SENSITIVITY SUITE ---
            st.markdown("<div style='margin-top: 16px;'></div>", unsafe_allow_html=True)
            with st.expander(":material/ssid_chart: Dynamic Limit Equilibrium Sensitivity Curves (FoS vs β & FoS vs ru)", expanded=True):
                sens_c1, sens_c2 = st.columns([2.5, 1.5])
                with sens_c1:
                    sens_mode = st.radio(
                        "Evaluate Sensitivity Against",
                        ["Slope Inclination Angle (β)", "Groundwater / Porewater Pressure Ratio (ru)"],
                        horizontal=True,
                        help="Select whether to evaluate Factor of Safety response across steepening slope angles or rising porewater tables.",
                        key="ui_sens_mode_radio"
                    )
                with sens_c2:
                    st.caption(f":material/info: **Active Baseline:** `{cur_soil}` · **Root Depth:** `{cur_root_depth:.1f} m` · **Root Cohesion:** `+{cur_cr:.1f} kPa`")

                sens_var = "water_table" if "ru" in sens_mode else "angle"
                sens_fig, sens_stats = generate_fos_sensitivity_fig(
                    height_m=ui_slope_h,
                    slope_angle_deg=ui_slope_angle,
                    soil_class=cur_soil,
                    water_table_ratio=ui_wt_ratio,
                    root_depth_m=cur_root_depth,
                    root_cohesion_kpa=cur_cr,
                    design_slip_depth_m=ui_slip_z,
                    sensitivity_variable=sens_var,
                )
                st.plotly_chart(sens_fig, width="stretch")

        with ds_tab2:
            st.caption("Contractor-grade planting geometry and Bill of Quantities (BoQ) material takeoff based on site dimensions:")
            boq_c1, boq_c2, boq_c3 = st.columns(3)
            with boq_c1:
                ui_boq_w = st.number_input("Slope Crest Width / Length along road (m)", min_value=5.0, max_value=500.0, value=30.0, step=5.0, key="ui_boq_w")
            with boq_c2:
                ui_slip_sp = st.selectbox("In-Row Slip Spacing", [0.10, 0.12, 0.15], index=1, format_func=lambda x: f"{int(x*100)} cm ({'Dense Barrier' if x==0.10 else 'Standard Civil Contour'})", key="ui_boq_spacing")
            with boq_c3:
                ui_slip_rate = st.number_input("Nursery Slip Unit Rate (KES)", min_value=3.0, max_value=50.0, value=12.0, step=1.0, key="ui_boq_rate")

            boq_res = calculate_field_boq(
                height_m=ui_slope_h,
                width_m=ui_boq_w,
                slope_angle_deg=ui_slope_angle,
                slip_spacing_m=ui_slip_sp,
                slip_unit_rate_kes=ui_slip_rate,
            )

            # BoQ KPI Chips
            bk1, bk2, bk3, bk4 = st.columns(4)
            with bk1:
                st.markdown(f'''
                <div class="kpi-chip">
                    <div class="kpi-chip-title">Total Slips Needed</div>
                    <div class="kpi-chip-val" style="color: #16A34A;">{boq_res['total_slips']:,} Slips</div>
                </div>
                ''', unsafe_allow_html=True)
            with bk2:
                st.markdown(f'''
                <div class="kpi-chip">
                    <div class="kpi-chip-title">Contour Hedgerows</div>
                    <div class="kpi-chip-val" style="color: #0F172A;">{boq_res['num_contour_rows']} Rows ({boq_res['total_linear_meters']:.0f}m)</div>
                </div>
                ''', unsafe_allow_html=True)
            with bk3:
                st.markdown(f'''
                <div class="kpi-chip">
                    <div class="kpi-chip-title">Total Budget (KES)</div>
                    <div class="kpi-chip-val" style="color: #0369A1;">KES {boq_res['total_cost_kes']:,}</div>
                </div>
                ''', unsafe_allow_html=True)
            with bk4:
                st.markdown(f'''
                <div class="kpi-chip">
                    <div class="kpi-chip-title">Total Budget (USD)</div>
                    <div class="kpi-chip-val" style="color: #0369A1;">${boq_res['total_cost_usd']:,.2f}</div>
                </div>
                ''', unsafe_allow_html=True)

            # BoQ Schedule Table
            df_boq = pd.DataFrame(boq_res["boq_items"])
            st.dataframe(
                df_boq,
                column_config={
                    "item_no": "Item No.",
                    "description": "Item Description & Specification",
                    "unit": "Unit",
                    "quantity": st.column_config.NumberColumn("Quantity", format="%.1f"),
                    "rate_kes": st.column_config.NumberColumn("Rate (KES)", format="%.1f"),
                    "total_kes": st.column_config.NumberColumn("Total Cost (KES)", format="%,d"),
                },
                hide_index=True,
                width="stretch",
            )

            boq_csv = df_boq.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Bill of Quantities (.csv)",
                data=boq_csv,
                file_name=f"vetiver_boq_schedule_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                icon=":material/download:",
                width="stretch",
            )

        with ds_tab3:
            st.caption("12-month biomechanical root penetration, added cohesion growth, and slope safety factor progression:")
            mat_fig = generate_temporal_maturation_fig(
                mature_root_depth_m=cur_root_depth,
                mature_cr_kpa=cur_cr,
                fos_base=cs_stab["fos_unreinforced"],
                fos_mature=cs_stab["fos_reinforced"],
            )
            st.plotly_chart(mat_fig, width="stretch")
            st.info(
                "**Field Agronomic Timing Protocol:** Align field transplanting with the onset of the regional rainy season "
                "(e.g., March–April Long Rains or October–November Short Rains in Kenya). During the initial 60-day Vulnerability Window, "
                "install temporary coir/jute erosion netting to prevent scouring of juvenile root plugs prior to deep tensile anchor establishment.",
                icon=":material/calendar_month:"
            )

        # 5. Report Preview & Download
        with st.expander("Preview Full Geotechnical Engineering Report", icon=":material/description:"):
            st.markdown(report_md)

        st.download_button(
            label="Download Full Geotechnical Report (.md)",
            data=report_md,
            file_name=f"vetiver_site_assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
            icon=":material/download:",
            width="stretch",
        )

        shap_plot_file = os.path.join(config.DATA_DIR, "shap_summary.png")
        if os.path.exists(shap_plot_file):
            with st.expander("View Global Model Feature Importance (SHAP Summary)", icon=":material/bar_chart:"):
                st.image(shap_plot_file, caption="Global Mean |SHAP| values computed across all empirical and failure boundaries.")


# =======================================================
# TAB 2: REGIONAL HOTSPOTS & 3D GIS MAP
# =======================================================
with tab2:
    st.subheader("Regional Hotspots & 3D GIS Map", anchor=False)

    hotspots = load_regional_hotspots(hotspot_mtime)
    if not hotspots:
        st.warning("No regional hotspot database found at data/regional_hotspots.json.", icon=":material/warning:")
    else:
        # Top Compact Bar: Geographic Scope, Basemap Style, 3D Elevation
        top_c1, top_c2, top_c3 = st.columns([1.8, 1.3, 0.9])
        with top_c1:
            scope_mode = st.radio(
                "Geographic Scope Filter",
                ["Kenya Corridors & Counties (Detailed)", "East Africa Regional Sites", "All Global Benchmark Sites"],
                horizontal=True,
                help="Switch between Kenya national corridors, regional East Africa, or global benchmark sites."
            )

        if "Kenya" in scope_mode:
            filtered_hotspots = [h for h in hotspots if h.get("country") == "Kenya"]
            default_lat, default_lon, default_zoom, default_pitch = 0.2, 37.8, 6.5, 30
        elif "East Africa" in scope_mode:
            filtered_hotspots = [h for h in hotspots if h.get("country") in ["Kenya", "Ethiopia", "Zambia", "Zimbabwe"]]
            default_lat, default_lon, default_zoom, default_pitch = -2.0, 34.5, 4.8, 25
        else:
            filtered_hotspots = hotspots
            default_lat, default_lon, default_zoom, default_pitch = 5.0, 25.0, 2.2, 20

        with top_c2:
            map_style_choice = st.selectbox(
                "Cartographic Basemap",
                ["Positron (Light Boundaries & Roads)", "Voyager (Topography & Contours)", "Dark Matter (High-Contrast GIS)"],
                index=0,
                help="Select your preferred cartographic tile style."
            )
        with top_c3:
            st.markdown("<div style='margin-top: 26px;'></div>", unsafe_allow_html=True)
            show_3d_elevation = st.checkbox("3D Elevation", value=False, help="Render slender 3D bio-engineering pillars indicating site altitude.")

        # Map style URL mapping
        style_url_map = {
            "Positron (Light Boundaries & Roads)": "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
            "Voyager (Topography & Contours)": "https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json",
            "Dark Matter (High-Contrast GIS)": "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
        }
        selected_style_url = style_url_map.get(map_style_choice, "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json")

        # Validate that session state contains a valid hotspot id within the active filter
        valid_ids = [h["id"] for h in filtered_hotspots]
        if "selected_hotspot_id" not in st.session_state or st.session_state.selected_hotspot_id not in valid_ids:
            st.session_state.selected_hotspot_id = valid_ids[0]

        cur_idx = next((i for i, h in enumerate(filtered_hotspots) if h["id"] == st.session_state.selected_hotspot_id), 0)
        selected_hs = filtered_hotspots[cur_idx]

        # ── LINEAR INFRASTRUCTURE CORRIDOR BATCH PROFILER ──
        with st.expander(":material/alt_route: Linear Infrastructure Corridor Batch Profiler (Highway Alignments & Cuttings)", expanded=False):
            st.markdown(
                "Evaluate linear civil transport infrastructure alignments across chainages. Automatically batches "
                "coordinate resolution, USCS soil transitions, elevation profiles, and geotechnical suitability for road cuts and railway batters."
            )
            cor_c1, cor_c2, cor_c3 = st.columns([2.0, 1.2, 1.0])
            with cor_c1:
                cor_choice = st.selectbox(
                    "Select Highway Alignment Corridor",
                    list(BUILTIN_CORRIDORS.keys()),
                    format_func=lambda k: BUILTIN_CORRIDORS[k]["name"],
                    key="sel_infra_corridor"
                )
            with cor_c2:
                uploaded_corridor_csv = st.file_uploader("Or Upload Custom Corridor (.csv)", type=["csv"], help="CSV with columns: lat, lon, chainage_km (optional), station (optional)", key="upload_corridor_csv")
            with cor_c3:
                st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
                run_corridor_btn = st.button("Run Batch Profiler", type="primary", icon=":material/play_arrow:", width="stretch", key="btn_run_corridor")

            if run_corridor_btn or "df_corridor_cache" in st.session_state:
                if run_corridor_btn:
                    with st.spinner("Batch evaluating geotechnical parameters along corridor chainages..."):
                        if uploaded_corridor_csv is not None:
                            try:
                                in_df = pd.read_csv(uploaded_corridor_csv)
                                custom_wps = []
                                for idx, r in in_df.iterrows():
                                    custom_wps.append({
                                        "chainage_km": r.get("chainage_km", idx * 5),
                                        "station": r.get("station", f"Station {idx+1}"),
                                        "lat": float(r["lat"] if "lat" in r else r["latitude"]),
                                        "lon": float(r["lon"] if "lon" in r else r["longitude"]),
                                        "hazard": r.get("hazard", "Shallow Slope / Embankment Slip (0.3–1.5 m)")
                                    })
                                res_df = analyze_corridor_waypoints(custom_wps, run_assessment)
                                st.session_state.df_corridor_cache = res_df
                                st.session_state.corridor_cache_name = "Custom Uploaded Corridor"
                            except Exception as e:
                                st.error(f"Error parsing uploaded CSV: {e}", icon=":material/error:")
                        else:
                            active_cor = BUILTIN_CORRIDORS[cor_choice]
                            res_df = analyze_corridor_waypoints(active_cor["waypoints"], run_assessment)
                            st.session_state.df_corridor_cache = res_df
                            st.session_state.corridor_cache_name = active_cor["name"]

                if "df_corridor_cache" in st.session_state:
                    c_df = st.session_state.df_corridor_cache
                    c_name = st.session_state.get("corridor_cache_name", "Highway Alignment")

                    c_fig = generate_corridor_profile_fig(c_df, c_name)
                    st.plotly_chart(c_fig, width="stretch")

                    st.markdown("##### :material/table_chart: Corridor Geotechnical Chainage Schedule")
                    st.dataframe(
                        c_df[["Chainage (km)", "Station Name", "Elevation (m)", "USCS Soil Class", "Verdict", "Model Confidence", "Category", "FoS Boost", "Prescribed Engineering Protocol"]],
                        hide_index=True,
                        width="stretch"
                    )

                    cor_btn_c1, cor_btn_c2 = st.columns(2)
                    with cor_btn_c1:
                        cor_csv = c_df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="Download Corridor Schedule (.csv)",
                            data=cor_csv,
                            file_name=f"corridor_geotechnical_profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            icon=":material/download:",
                            width="stretch",
                        )
                    with cor_btn_c2:
                        cor_geojson = export_corridor_geojson(c_df, c_name)
                        st.download_button(
                            label="Download GIS Route (.geojson)",
                            data=cor_geojson,
                            file_name=f"corridor_gis_route_{datetime.now().strftime('%Y%m%d_%H%M%S')}.geojson",
                            mime="application/geo+json",
                            icon=":material/map:",
                            width="stretch",
                            help="Export standard GeoJSON vector alignment and waypoint parameters for QGIS / ArcGIS."
                        )

        # ── 62 / 38 SPLIT-SCREEN STUDIO LAYOUT ──
        col_map, col_inspector = st.columns([1.58, 1.0], gap="medium")

        with col_map:
            # Inline GPS Geocoder & Quick Corridor Bar
            with st.container(border=True):
                g_c1, g_c2, g_c3 = st.columns([1.2, 1.2, 1.6])
                with g_c1:
                    in_lat = st.number_input(
                        "Latitude (°N)",
                        min_value=-90.0,
                        max_value=90.0,
                        value=float(st.session_state.coord_lat),
                        format="%.5f",
                        step=0.01,
                        key="t2_in_lat"
                    )
                with g_c2:
                    in_lon = st.number_input(
                        "Longitude (°E)",
                        min_value=-180.0,
                        max_value=180.0,
                        value=float(st.session_state.coord_lon),
                        format="%.5f",
                        step=0.01,
                        key="t2_in_lon"
                    )
                with g_c3:
                    st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
                    resolve_btn = st.button(
                        "Locate & Ingest Site",
                        type="primary",
                        width="stretch",
                        icon=":material/travel_explore:",
                        key="btn_t2_locate"
                    )

                # Corridor shortcuts row
                cp1, cp2, cp3, cp4, cp5, cp6 = st.columns(6)
                if cp1.button("Nairobi", width="stretch", key="t2_pill_nrb"):
                    st.session_state.coord_lat = -1.2921
                    st.session_state.coord_lon = 36.8219
                    c_site = resolve_coordinate_geotechnics(-1.2921, 36.8219)
                    st.session_state.custom_coord_site = c_site
                    st.session_state.preset_soil = c_site["soil_mechanical_class"]
                    st.session_state.preset_climate = c_site["climate_rainfall_zone"]
                    st.session_state.preset_moisture = c_site["establishment_moisture"]
                    st.session_state.preset_hydro = c_site["hydrological_stress"]
                    st.session_state.preset_salinity = c_site["salinity_tier"]
                    st.session_state.preset_hazard = c_site["default_hazard"]
                    st.session_state.custom_assessment_result = None
                    st.session_state.pan_to_custom = True
                    st.rerun()
                if cp2.button("Samburu", width="stretch", key="t2_pill_smb"):
                    st.session_state.coord_lat = 1.0000
                    st.session_state.coord_lon = 36.8219
                    c_site = resolve_coordinate_geotechnics(1.0000, 36.8219)
                    st.session_state.custom_coord_site = c_site
                    st.session_state.preset_soil = c_site["soil_mechanical_class"]
                    st.session_state.preset_climate = c_site["climate_rainfall_zone"]
                    st.session_state.preset_moisture = c_site["establishment_moisture"]
                    st.session_state.preset_hydro = c_site["hydrological_stress"]
                    st.session_state.preset_salinity = c_site["salinity_tier"]
                    st.session_state.preset_hazard = c_site["default_hazard"]
                    st.session_state.custom_assessment_result = None
                    st.session_state.pan_to_custom = True
                    st.rerun()
                if cp3.button("Eldoret", width="stretch", key="t2_pill_eld"):
                    st.session_state.coord_lat = 0.5143
                    st.session_state.coord_lon = 35.2698
                    c_site = resolve_coordinate_geotechnics(0.5143, 35.2698)
                    st.session_state.custom_coord_site = c_site
                    st.session_state.preset_soil = c_site["soil_mechanical_class"]
                    st.session_state.preset_climate = c_site["climate_rainfall_zone"]
                    st.session_state.preset_moisture = c_site["establishment_moisture"]
                    st.session_state.preset_hydro = c_site["hydrological_stress"]
                    st.session_state.preset_salinity = c_site["salinity_tier"]
                    st.session_state.preset_hazard = c_site["default_hazard"]
                    st.session_state.custom_assessment_result = None
                    st.session_state.pan_to_custom = True
                    st.rerun()
                if cp4.button("Naivasha", width="stretch", key="t2_pill_nvs"):
                    st.session_state.coord_lat = -0.7172
                    st.session_state.coord_lon = 36.4312
                    c_site = resolve_coordinate_geotechnics(-0.7172, 36.4312)
                    st.session_state.custom_coord_site = c_site
                    st.session_state.preset_soil = c_site["soil_mechanical_class"]
                    st.session_state.preset_climate = c_site["climate_rainfall_zone"]
                    st.session_state.preset_moisture = c_site["establishment_moisture"]
                    st.session_state.preset_hydro = c_site["hydrological_stress"]
                    st.session_state.preset_salinity = c_site["salinity_tier"]
                    st.session_state.preset_hazard = c_site["default_hazard"]
                    st.session_state.custom_assessment_result = None
                    st.session_state.pan_to_custom = True
                    st.rerun()
                if cp5.button("Embu", width="stretch", key="t2_pill_emb"):
                    st.session_state.coord_lat = -0.5342
                    st.session_state.coord_lon = 37.4544
                    c_site = resolve_coordinate_geotechnics(-0.5342, 37.4544)
                    st.session_state.custom_coord_site = c_site
                    st.session_state.preset_soil = c_site["soil_mechanical_class"]
                    st.session_state.preset_climate = c_site["climate_rainfall_zone"]
                    st.session_state.preset_moisture = c_site["establishment_moisture"]
                    st.session_state.preset_hydro = c_site["hydrological_stress"]
                    st.session_state.preset_salinity = c_site["salinity_tier"]
                    st.session_state.preset_hazard = c_site["default_hazard"]
                    st.session_state.custom_assessment_result = None
                    st.session_state.pan_to_custom = True
                    st.rerun()
                if cp6.button("Garissa", width="stretch", key="t2_pill_grs"):
                    st.session_state.coord_lat = -0.4532
                    st.session_state.coord_lon = 39.6461
                    c_site = resolve_coordinate_geotechnics(-0.4532, 39.6461)
                    st.session_state.custom_coord_site = c_site
                    st.session_state.preset_soil = c_site["soil_mechanical_class"]
                    st.session_state.preset_climate = c_site["climate_rainfall_zone"]
                    st.session_state.preset_moisture = c_site["establishment_moisture"]
                    st.session_state.preset_hydro = c_site["hydrological_stress"]
                    st.session_state.preset_salinity = c_site["salinity_tier"]
                    st.session_state.preset_hazard = c_site["default_hazard"]
                    st.session_state.custom_assessment_result = None
                    st.session_state.pan_to_custom = True
                    st.rerun()

                if resolve_btn:
                    with st.spinner(f"Querying environmental providers for ({in_lat:.4f}°N, {in_lon:.4f}°E)..."):
                        st.session_state.coord_lat = in_lat
                        st.session_state.coord_lon = in_lon
                        c_site = resolve_coordinate_geotechnics(in_lat, in_lon)
                        st.session_state.custom_coord_site = c_site
                        st.session_state.preset_soil = c_site["soil_mechanical_class"]
                        st.session_state.preset_climate = c_site["climate_rainfall_zone"]
                        st.session_state.preset_moisture = c_site["establishment_moisture"]
                        st.session_state.preset_hydro = c_site["hydrological_stress"]
                        st.session_state.preset_salinity = c_site["salinity_tier"]
                        st.session_state.preset_hazard = c_site["default_hazard"]
                        st.session_state.custom_assessment_result = None
                        st.session_state.pan_to_custom = True
                        st.rerun()

            # Camera parameters
            if st.session_state.get("pan_to_custom") and st.session_state.custom_coord_site:
                map_lat = st.session_state.custom_coord_site["latitude"]
                map_lon = st.session_state.custom_coord_site["longitude"]
                map_zoom = 9.8
                map_pitch = 40
            else:
                map_lat = selected_hs["latitude"]
                map_lon = selected_hs["longitude"]
                map_zoom = 8.2 if "Kenya" in scope_mode else 6.8
                map_pitch = 38

            view_state = pdk.ViewState(
                latitude=map_lat,
                longitude=map_lon,
                zoom=map_zoom,
                pitch=map_pitch,
                bearing=0,
            )

            df_map = pd.DataFrame(filtered_hotspots)
            df_active = pd.DataFrame([selected_hs])

            # Layer 1: Base Hotspot Pins (Interactive Scatterplot)
            scatter_layer = pdk.Layer(
                "ScatterplotLayer",
                id="hotspot-scatter",
                data=df_map,
                get_position=["longitude", "latitude"],
                get_color="color",
                pickable=True,
                auto_highlight=True,
                highlight_color=[250, 204, 21, 220],
                radius_min_pixels=10,
                radius_max_pixels=24,
                stroked=True,
                filled=True,
                line_width_min_pixels=2.5,
                get_line_color=[255, 255, 255, 255],
            )

            # Layer 2: Active Focused Beacon Ring (Electric Cyan Halo)
            halo_layer = pdk.Layer(
                "ScatterplotLayer",
                id="hotspot-active-halo",
                data=df_active,
                get_position=["longitude", "latitude"],
                get_color=[0, 0, 0, 0],
                stroked=True,
                filled=False,
                radius_min_pixels=18,
                radius_max_pixels=34,
                line_width_min_pixels=3.5,
                get_line_color=[56, 189, 248, 255],
                pickable=False,
            )

            # Layer 3: Clean Pill Badges with Site Short Names
            text_layer = pdk.Layer(
                "TextLayer",
                id="hotspot-labels",
                data=df_map,
                get_position=["longitude", "latitude"],
                get_text="short_label",
                get_size=12,
                get_color=[255, 255, 255, 255],
                get_text_anchor="'start'",
                get_alignment_baseline="'center'",
                pixel_offset=[14, -1],
                background=True,
                get_background_color=[15, 23, 42, 225],
                background_padding=[4, 7, 4, 7],
                pickable=True,
            )

            map_layers = [scatter_layer, halo_layer, text_layer]

            # Optional Layer 4: Custom Queried GPS Target Beacon Layer
            if st.session_state.custom_coord_site is not None:
                c_site = st.session_state.custom_coord_site
                df_custom = pd.DataFrame([{
                    "latitude": c_site["latitude"],
                    "longitude": c_site["longitude"],
                    "name": f"Target: {c_site['county']}",
                    "county_region": c_site["county"],
                    "country": c_site["country"],
                    "elevation_m": c_site["elevation_m"],
                    "status": "Queried GPS Coordinate",
                    "soil_mechanical_class": c_site["soil_mechanical_class"],
                    "climate_rainfall_zone": c_site["climate_rainfall_zone"],
                    "kpi1": f"Salinity: {c_site['salinity_tier']}",
                    "kpi2": f"Hazard: {c_site['default_hazard']}",
                    "citation": c_site["data_sources"],
                }])

                custom_scatter = pdk.Layer(
                    "ScatterplotLayer",
                    id="custom-target-scatter",
                    data=df_custom,
                    get_position=["longitude", "latitude"],
                    get_color=[245, 158, 11, 240],  # Glowing Amber
                    pickable=True,
                    auto_highlight=True,
                    highlight_color=[251, 191, 36, 255],
                    radius_min_pixels=13,
                    radius_max_pixels=28,
                    stroked=True,
                    filled=True,
                    line_width_min_pixels=3,
                    get_line_color=[255, 255, 255, 255],
                )
                custom_halo = pdk.Layer(
                    "ScatterplotLayer",
                    id="custom-target-halo",
                    data=df_custom,
                    get_position=["longitude", "latitude"],
                    get_color=[0, 0, 0, 0],
                    stroked=True,
                    filled=False,
                    radius_min_pixels=24,
                    radius_max_pixels=46,
                    line_width_min_pixels=3.5,
                    get_line_color=[245, 158, 11, 220],
                    pickable=False,
                )
                custom_text = pdk.Layer(
                    "TextLayer",
                    id="custom-target-labels",
                    data=df_custom,
                    get_position=["longitude", "latitude"],
                    get_text="name",
                    get_size=13,
                    get_color=[255, 255, 255, 255],
                    get_text_anchor="'start'",
                    get_alignment_baseline="'center'",
                    pixel_offset=[16, -1],
                    background=True,
                    get_background_color=[180, 83, 9, 235],
                    background_padding=[4, 8, 4, 8],
                    pickable=True,
                )
                map_layers.extend([custom_scatter, custom_halo, custom_text])

            # Optional Layer 5: 3D Elevation Columns
            if show_3d_elevation:
                column_layer = pdk.Layer(
                    "ColumnLayer",
                    id="hotspot-pillars",
                    data=df_map,
                    get_position=["longitude", "latitude"],
                    get_elevation="elevation_m",
                    elevation_scale=3.5,
                    radius=1200,
                    get_fill_color="color",
                    pickable=False,
                )
                map_layers.insert(0, column_layer)

            deck = pdk.Deck(
                layers=map_layers,
                initial_view_state=view_state,
                map_style=selected_style_url,
                tooltip={
                    "html": """
                        <div style="font-family: -apple-system, BlinkMacSystemFont, sans-serif; font-size: 13px; color: #F8FAFC; background: #0F172A; padding: 12px; border-radius: 8px; border: 1px solid #334155; box-shadow: 0 8px 24px rgba(0,0,0,0.4); max-width: 320px;">
                            <div style="font-weight: 800; font-size: 14px; color: #38BDF8; margin-bottom: 4px;">{name}</div>
                            <div style="color: #94A3B8; margin-bottom: 6px; font-size: 12px;">{county_region}, {country} &bull; Elev: {elevation_m} m a.s.l.</div>
                            <div style="margin-bottom: 4px;"><b>Geotechnical Status:</b> <span style="font-weight: 700; color: #4ADE80;">{status}</span></div>
                            <div style="margin-bottom: 4px;"><b>USCS Soil Class:</b> {soil_mechanical_class}</div>
                            <div style="margin-bottom: 4px;"><b>Precipitation Zone:</b> {climate_rainfall_zone}</div>
                            <div style="margin-bottom: 4px;"><b>Key Metric:</b> {kpi1} | {kpi2}</div>
                            <div style="font-size: 11px; color: #38BDF8; margin-top: 6px; border-top: 1px solid #334155; padding-top: 4px;"><i>Tip: Click this pin to inspect engineering parameters.</i></div>
                        </div>
                    """,
                    "style": {"zIndex": "1000"}
                }
            )

            custom_key_part = f"cust_{st.session_state.custom_coord_site['latitude']}_{st.session_state.custom_coord_site['longitude']}" if st.session_state.custom_coord_site else "none"
            deck_chart_key = f"pdk_gis_{scope_mode}_{selected_hs['id']}_{custom_key_part}_{map_style_choice}"
            deck_event = st.pydeck_chart(
                deck,
                width="stretch",
                height=650,
                on_select="rerun",
                selection_mode="single-object",
                key=deck_chart_key,
            )

            # Handle Map Pin Click Selection Event
            if deck_event and hasattr(deck_event, "selection") and deck_event.selection:
                clicked_id = None
                clicked_objs = deck_event.selection.get("objects", {}).get("hotspot-scatter", [])
                if not clicked_objs:
                    clicked_objs = deck_event.selection.get("objects", {}).get("hotspot-labels", [])
                if clicked_objs and isinstance(clicked_objs, list) and len(clicked_objs) > 0:
                    first_item = clicked_objs[0]
                    if isinstance(first_item, dict):
                        clicked_id = first_item.get("id")

                if not clicked_id:
                    clicked_indices = deck_event.selection.get("indices", {}).get("hotspot-scatter", [])
                    if not clicked_indices:
                        clicked_indices = deck_event.selection.get("indices", {}).get("hotspot-labels", [])
                    if clicked_indices and isinstance(clicked_indices, list) and len(clicked_indices) > 0:
                        idx = clicked_indices[0]
                        if 0 <= idx < len(filtered_hotspots):
                            clicked_id = filtered_hotspots[idx]["id"]

                if clicked_id and clicked_id in valid_ids and clicked_id != st.session_state.selected_hotspot_id:
                    st.session_state.selected_hotspot_id = clicked_id
                    st.session_state.custom_coord_site = None
                    st.session_state.pan_to_custom = False
                    st.rerun()

            st.caption(":material/info: **3D GIS Guide:** Left-Click any pin to inspect. Right-Click + Drag (or Ctrl + Drag) to adjust 3D pitch/bearing. Scroll to zoom.")

        # ── INSPECTOR & GEOTECHNICAL WORKBENCH (RIGHT ~38%) ──
        with col_inspector:
            with st.container(border=True):
                if st.session_state.custom_coord_site is not None:
                    c_res = st.session_state.custom_coord_site
                    st.markdown(f"#### :material/place: {c_res['county']}, {c_res['country']}")
                    st.markdown(f"""
                    <div style="display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 10px;">
                        <span class="studio-badge badge-blue">Elevation: {c_res['elevation_m']} m</span>
                        <span class="studio-badge badge-amber">GPS: {c_res['latitude']:.4f}°N, {c_res['longitude']:.4f}°E</span>
                        <span class="studio-badge badge-suitable">Source: {c_res['data_sources']}</span>
                    </div>
                    """, unsafe_allow_html=True)

                    if c_res["is_known_hotspot"]:
                        matched_h = c_res["matched_hotspot"]
                        st.success(
                            f"**Known Regional Benchmark ({c_res['distance_to_nearest_km']} km away):** {matched_h['name']}",
                            icon=":material/verified:"
                        )
                        st.caption(f"**Field Summary:** {matched_h['summary_desc']}")
                    else:
                        st.info(
                            f"**Uncomputed Field Site** — Ingested via {c_res['data_sources']}. Nearest pre-computed trial is {c_res['distance_to_nearest_km']} km away.",
                            icon=":material/public:"
                        )

                    init_soil = c_res.get("soil_mechanical_class", st.session_state.preset_soil)
                    init_clim = c_res.get("climate_rainfall_zone", st.session_state.preset_climate)
                    init_moist = c_res.get("establishment_moisture", st.session_state.preset_moisture)
                    init_hydro = c_res.get("hydrological_stress", st.session_state.preset_hydro)
                    init_sal = c_res.get("salinity_tier", st.session_state.preset_salinity)
                    init_haz = c_res.get("default_hazard", st.session_state.preset_hazard)

                else:
                    hs_display_options = [f"{h['county_region']}: {h['name']}" for h in filtered_hotspots]
                    sel_hs_name = st.selectbox(
                        "Active Hotspot Pin",
                        hs_display_options,
                        index=cur_idx,
                        key="t2_hs_dropdown",
                        help="Select a regional hotspot to inspect."
                    )
                    chosen_h = next((h for h in filtered_hotspots if f"{h['county_region']}: {h['name']}" == sel_hs_name), filtered_hotspots[cur_idx])
                    if chosen_h["id"] != st.session_state.selected_hotspot_id:
                        st.session_state.selected_hotspot_id = chosen_h["id"]
                        selected_hs = chosen_h
                        st.rerun()

                    st.markdown(f"#### :material/place: {selected_hs['name']}")
                    stat_badge = "badge-suitable" if selected_hs['status'] == "Suitable" else "badge-unsuitable"
                    st.markdown(f"""
                    <div style="display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 10px;">
                        <span class="studio-badge {stat_badge}">Status: {selected_hs['status']}</span>
                        <span class="studio-badge badge-blue">Elevation: {selected_hs['elevation_m']} m</span>
                        <span class="studio-badge badge-amber">GPS: {selected_hs['latitude']:.4f}°N, {selected_hs['longitude']:.4f}°E</span>
                    </div>
                    """, unsafe_allow_html=True)
                    st.caption(f"**Empirical Summary:** {selected_hs['summary_desc']}")
                    st.caption(f"**Citation:** *{selected_hs['citation']}*")

                    init_soil = selected_hs.get("soil_mechanical_class", st.session_state.preset_soil)
                    init_clim = selected_hs.get("climate_rainfall_zone", st.session_state.preset_climate)
                    init_moist = selected_hs.get("establishment_moisture", st.session_state.preset_moisture)
                    init_hydro = selected_hs.get("hydrological_stress", st.session_state.preset_hydro)
                    init_sal = selected_hs.get("salinity_tier", st.session_state.preset_salinity)
                    init_haz = selected_hs.get("default_hazard", st.session_state.preset_hazard)

                st.markdown("##### :material/tune: Geotechnical Parameters")
                w_col1, w_col2 = st.columns(2)
                with w_col1:
                    soil_options = [
                        "Sandy Clay / Loam",
                        "Cohesive Clay / Nitisol / Laterite",
                        "Coarse Sand / Loose Pumice",
                        "Indurated Hardpan / Bedrock",
                        "Mine Tailings / Industrial Spoil",
                    ]
                    s_idx = soil_options.index(init_soil) if init_soil in soil_options else 0
                    t2_soil = st.selectbox("USCS Soil Class", soil_options, index=s_idx, key="t2_w_soil")

                    climate_options = [
                        "Sub-Humid / Bimodal (700–1400 mm/yr)",
                        "Humid / Tropical Monsoon (>1400 mm/yr)",
                        "Semi-Arid (300–700 mm/yr)",
                        "Hyper-Arid / Desert (<300 mm/yr)",
                        "Alpine High-Altitude (>2,200 m)",
                        "Cold Freezing / Permafrost",
                    ]
                    c_idx = climate_options.index(init_clim) if init_clim in climate_options else 0
                    t2_climate = st.selectbox("Precipitation Envelope", climate_options, index=c_idx, key="t2_w_clim")

                    hazard_options = [
                        "Shallow Slope / Embankment Slip (0.3–1.5 m)",
                        "Surficial Sheet & Rill Erosion (< 0.3 m)",
                        "Deep Rotational Slide (> 2.0 m)",
                        "Riverbank / Hydrodynamic Toe Scour",
                        "Mine Tailings & Spoil Containment",
                    ]
                    h_idx = hazard_options.index(init_haz) if init_haz in hazard_options else 0
                    t2_hazard = st.selectbox("Civil Hazard Objective", hazard_options, index=h_idx, key="t2_w_haz")

                with w_col2:
                    moisture_options = [
                        "Adequate (Irrigated / In-Season Rain)",
                        "Severe Drought / Zero Watering",
                    ]
                    m_idx = moisture_options.index(init_moist) if init_moist in moisture_options else 0
                    t2_moisture = st.selectbox("Establishment Moisture", moisture_options, index=m_idx, key="t2_w_moist")

                    hydro_options = [
                        "Normal Slope / Well-Drained",
                        "High-Velocity Hydraulic Runoff (>2.5 m/s)",
                        "Prolonged Inundation (>21 days)",
                    ]
                    hy_idx = hydro_options.index(init_hydro) if init_hydro in hydro_options else 0
                    t2_hydro = st.selectbox("Hydrological Stress", hydro_options, index=hy_idx, key="t2_w_hydro")

                    salinity_options = [
                        "Non-Saline (< 2 dS/m)",
                        "Slight to Moderate (2–8 dS/m)",
                        "High Saline (8–16 dS/m)",
                        "Extreme Hypersaline (> 16 dS/m)",
                    ]
                    sal_idx = salinity_options.index(init_sal) if init_sal in salinity_options else 0
                    t2_salinity = st.selectbox("Substrate Salinity", salinity_options, index=sal_idx, key="t2_w_sal")

                if st.session_state.custom_coord_site and st.session_state.custom_coord_site.get("notes"):
                    st.caption(f":material/description: **Geotechnical Notes:** {st.session_state.custom_coord_site['notes']}")

                # Action buttons
                ab_c1, ab_c2 = st.columns([1.5, 1.5])
                with ab_c1:
                    run_t2_eval = st.button("Run Assessment", type="primary", width="stretch", icon=":material/play_arrow:", key="btn_t2_run_eval")
                with ab_c2:
                    send_t2_tab1 = st.button("Sync to Tab 1", width="stretch", icon=":material/open_in_new:", key="btn_t2_sync_tab1")

                if st.session_state.custom_coord_site is not None:
                    if st.button("Clear Custom Query", width="stretch", icon=":material/close:", key="btn_t2_clear"):
                        st.session_state.custom_coord_site = None
                        st.session_state.custom_assessment_result = None
                        st.session_state.pan_to_custom = False
                        st.rerun()

                if run_t2_eval or send_t2_tab1:
                    eval_inputs = {
                        "soil_mechanical_class": t2_soil,
                        "climate_rainfall_zone": t2_climate,
                        "establishment_moisture": t2_moisture,
                        "hydrological_stress": t2_hydro,
                        "salinity_tier": t2_salinity,
                        "est_salinity_ec_ds_m": config.SALINITY_TIER_TO_ECE.get(t2_salinity, 0.5),
                        "geotechnical_hazard": t2_hazard,
                        "geotechnical_hazard_display": hazard_display_map.get(t2_hazard, t2_hazard),
                    }
                    st.session_state.preset_soil = t2_soil
                    st.session_state.preset_climate = t2_climate
                    st.session_state.preset_moisture = t2_moisture
                    st.session_state.preset_hydro = t2_hydro
                    st.session_state.preset_salinity = t2_salinity
                    st.session_state.preset_hazard = t2_hazard
                    st.session_state.current_assessment = run_assessment(eval_inputs)
                    st.session_state.custom_assessment_result = st.session_state.current_assessment
                    if send_t2_tab1:
                        st.success("Parameters synchronized with Tab 1! Switch to Tab 1 to view SHAP explanation and export report.", icon=":material/check_circle:")
                    st.rerun()

                # Assessment Display Dock
                active_res = st.session_state.custom_assessment_result or (st.session_state.current_assessment if st.session_state.current_assessment else None)
                if active_res:
                    res_pred = active_res["prediction"]
                    res_prob = active_res["probability"]
                    bio = active_res.get("biomechanics") or calculate_biomechanical_metrics(active_res.get("inputs", {}))

                    if res_pred == 1:
                        st.markdown(f"""
                        <div class="verdict-banner-suitable" style="padding: 14px 16px; margin-top: 12px; margin-bottom: 0;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <span class="studio-badge badge-suitable">SUITABLE</span>
                                <span style="font-size: 1.35rem; font-weight: 800; color: #14532D;">{res_prob:.1f}% Confidence</span>
                            </div>
                            <div style="font-weight: 700; color: #14532D; font-size: 0.95rem; margin-top: 6px;">Positive Geotechnical Alignment</div>
                            <p style="font-size: 0.82rem; color: #166534; margin: 4px 0 0 0; line-height: 1.35;">{bio.get('summary_desc', '')}</p>
                            <div class="kpi-row" style="margin-top: 8px;">
                                <div class="kpi-chip" style="min-width: 90px; padding: 6px 8px;">
                                    <div class="kpi-chip-title" style="font-size: 0.65rem;">Root Depth</div>
                                    <div class="kpi-chip-val" style="font-size: 0.92rem; color: #166534;">{bio.get('kpi1_val', bio['root_depth'])}</div>
                                </div>
                                <div class="kpi-chip" style="min-width: 90px; padding: 6px 8px;">
                                    <div class="kpi-chip-title" style="font-size: 0.65rem;">Added Cohesion</div>
                                    <div class="kpi-chip-val" style="font-size: 0.92rem; color: #166534;">{bio.get('kpi2_val', bio['root_cohesion'])}</div>
                                </div>
                                <div class="kpi-chip" style="min-width: 90px; padding: 6px 8px;">
                                    <div class="kpi-chip-title" style="font-size: 0.65rem;">FoS Boost</div>
                                    <div class="kpi-chip-val" style="font-size: 0.92rem; color: #166534;">{bio.get('kpi3_val', bio['fos_boost'])}</div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        diag = get_failure_banner_details(active_res)
                        st.markdown(f"""
                        <div class="verdict-banner-unsuitable" style="padding: 14px 16px; margin-top: 12px; margin-bottom: 0;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <span class="studio-badge badge-unsuitable">{diag['badge']}</span>
                                <span style="font-size: 1.35rem; font-weight: 800; color: #7F1D1D;">{100 - res_prob:.1f}% Failure Conf.</span>
                            </div>
                            <div style="font-weight: 700; color: #7F1D1D; font-size: 0.92rem; margin-top: 6px;">{diag['title']}</div>
                            <p style="font-size: 0.82rem; color: #991B1B; margin: 4px 0 0 0; line-height: 1.35;">{diag['subtitle']}</p>
                            <div class="kpi-row" style="margin-top: 8px;">
                                <div class="kpi-chip" style="min-width: 90px; padding: 6px 8px;">
                                    <div class="kpi-chip-title" style="font-size: 0.65rem;">{diag['kpi1_title']}</div>
                                    <div class="kpi-chip-val" style="font-size: 0.92rem; color: #991B1B;">{diag['kpi1_val']}</div>
                                </div>
                                <div class="kpi-chip" style="min-width: 90px; padding: 6px 8px;">
                                    <div class="kpi-chip-title" style="font-size: 0.65rem;">{diag['kpi2_title']}</div>
                                    <div class="kpi-chip-val" style="font-size: 0.92rem; color: #991B1B;">{diag['kpi2_val']}</div>
                                </div>
                                <div class="kpi-chip" style="min-width: 90px; padding: 6px 8px;">
                                    <div class="kpi-chip-title" style="font-size: 0.65rem;">{diag['kpi3_title']}</div>
                                    <div class="kpi-chip-val" style="font-size: 0.92rem; color: #991B1B;">{diag['kpi3_val']}</div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    st.caption(":material/info: Results synchronized with Tab 1. Switch to Tab 1 to view SHAP attribution and download the report.")


# =======================================================
# TAB 3: RAG LITERATURE CHAT WITH CITATIONS
# =======================================================
with tab3:
    st.markdown("### :material/psychology: Vetiver Technical Consultation")

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
    active_site = get_active_site_context()
    bridge_enabled = False

    if active_site and (active_site.get("prediction") is not None or st.session_state.get("custom_coord_site")):
        is_suit = active_site.get("prediction") == 1
        badge_cls = "badge-suitable" if is_suit else "badge-risk"
        badge_text = "SUITABLE" if is_suit else ("AT RISK / UNFAVORABLE" if active_site.get("prediction") == 0 else "QUERY PRE-COMPUTED")
        conf_val = active_site.get("probability", 0.0) if is_suit else (100.0 - active_site.get("probability", 0.0))
        conf_str = f"({conf_val:.1f}% confidence)" if active_site.get("prediction") is not None else ""
        border_col = "#16A34A" if is_suit else "#DC2626"

        st.markdown(f"""
        <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-left: 3px solid {border_col}; border-radius: 8px; padding: 10px 14px; margin-bottom: 8px; font-size: 0.84rem; color: #334155; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 6px;">
            <div>
                <b>Tethered Site:</b> {active_site.get('location_name', 'Site')} ({active_site.get('elevation_m', 'N/A')} m a.s.l.) · 
                Soil: <b>{active_site.get('soil_mechanical_class', 'N/A')}</b> · 
                Rainfall: <b>{active_site.get('climate_rainfall_zone', 'N/A')}</b>
            </div>
            <span class="studio-badge {badge_cls}">{badge_text} {conf_str}</span>
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
                    f"and ML verdict ({active_site.get('excl_title')}). Detail required field preparation, root depth development timelines, and Factor of Safety (FoS) stabilization measures."
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
    if not ollama_online:
        st.error("Ollama is not running. Please start Ollama (`ollama serve`) and refresh this page.", icon=":material/power_off:")
    elif query_engine is None and rag_index is None:
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
                        st.markdown(clean_markdown_formatting(msg["content"]))
                        if msg.get("sources"):
                            with st.expander(f":material/library_books: Verified Evidence ({len(msg['sources'])} sources matched)", expanded=False):
                                st.markdown(format_sources_markdown(msg["sources"]))

            # 2. Render newly active query inside container (ABOVE the chat input)
            if active_query:
                st.session_state.messages.append({"role": "user", "content": active_query})
                with st.chat_message("user", avatar=":material/person:"):
                    st.markdown(active_query)

                p_info = PERSONA_PROMPTS.get(selected_persona, PERSONA_PROMPTS["geotechnical"])
                p_icon = p_info.get("icon", ":material/psychology:")

                with st.chat_message("assistant", avatar=p_icon):
                    st.caption(f":material/verified: **{p_info['title']}** · *{p_info['role_desc']}*")
                    response_placeholder = st.empty()

                    try:
                        site_ctx_for_call = active_site if bridge_enabled else None
                        with st.spinner("Analyzing empirical literature & site constraints..."):
                            if rag_index is not None:
                                streaming_response = query_rag_custom(
                                    index=rag_index,
                                    query_str=active_query,
                                    persona_key=selected_persona,
                                    site_context=site_ctx_for_call,
                                    conversation_history=st.session_state.messages,
                                    similarity_top_k=config.RAG_SIMILARITY_TOP_K,
                                    response_mode=response_style,
                                )
                            else:
                                streaming_response = query_engine.query(active_query)

                            gen = streaming_response.response_gen
                            first_chunk = next(gen, "")

                        # Real-time word-by-word streaming
                        full_response = first_chunk
                        response_placeholder.markdown(clean_markdown_formatting(full_response) + "▌")
                        for chunk in gen:
                            full_response += chunk
                            response_placeholder.markdown(clean_markdown_formatting(full_response) + "▌")

                        full_response = clean_markdown_formatting(full_response)
                        response_placeholder.markdown(full_response)

                        # Extract source citations
                        sources = extract_source_metadata(streaming_response)
                        if sources:
                            with st.expander(f":material/library_books: Verified Evidence ({len(sources)} sources matched)", expanded=False):
                                st.markdown(format_sources_markdown(sources))

                        # Save to state
                        st.session_state.messages.append({
                            "role": "assistant",
                            "persona": selected_persona,
                            "content": full_response,
                            "sources": sources,
                        })

                    except Exception as exc:
                        logger.error("RAG query failed: %s", exc, exc_info=True)
                        err_msg = "An error occurred while retrieving literature context. Please check Ollama and try again."
                        response_placeholder.error(err_msg, icon=":material/error:")
                        st.session_state.messages.append({"role": "assistant", "persona": selected_persona, "content": err_msg})
