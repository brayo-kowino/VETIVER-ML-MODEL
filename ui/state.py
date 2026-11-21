import streamlit as st


def init_session_state() -> None:
    """Initialize all persistent session variables if not already present."""
    defaults = {
        "messages": [],
        "last_analytics_report": None,
        "current_assessment": None,
        "preset_soil": "Sandy Clay / Loam",
        "preset_climate": "Sub-Humid / Bimodal (700–1400 mm/yr)",
        "preset_moisture": "Adequate (Irrigated / In-Season Rain)",
        "preset_hydro": "Normal Slope / Well-Drained",
        "preset_salinity": "Non-Saline (< 2 dS/m)",
        "preset_hazard": "Shallow Slope / Embankment Slip (0.3–1.5 m)",
        "selected_hotspot_id": "ke_voi",
        "custom_coord_site": None,
        "custom_assessment_result": None,
        "coord_lat": -1.2921,
        "coord_lon": 36.8219,
        "pan_to_custom": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val
