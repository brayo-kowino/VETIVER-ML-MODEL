import streamlit as st

import config
from services.biomechanics import HAZARD_DISPLAY_MAP


PARAMETER_OPTIONS = {
    "soil_mechanical_class": ("Soil class", (
        "Sandy Clay / Loam", "Cohesive Clay / Nitisol / Laterite", "Coarse Sand / Loose Pumice",
        "Indurated Hardpan / Bedrock", "Mine Tailings / Industrial Spoil",
    )),
    "climate_rainfall_zone": ("Precipitation regime", (
        "Sub-Humid / Bimodal (700–1400 mm/yr)", "Humid / Tropical Monsoon (>1400 mm/yr)",
        "Semi-Arid (300–700 mm/yr)", "Hyper-Arid / Desert (<300 mm/yr)",
        "Alpine High-Altitude (>2,200 m)", "Cold Freezing / Permafrost",
    )),
    "geotechnical_hazard": ("Geotechnical objective", tuple(HAZARD_DISPLAY_MAP)),
    "establishment_moisture": ("Establishment moisture", (
        "Adequate (Irrigated / In-Season Rain)", "Severe Drought / Zero Watering",
    )),
    "hydrological_stress": ("Hydrological stress", (
        "Normal Slope / Well-Drained", "High-Velocity Hydraulic Runoff (>2.5 m/s)",
        "Prolonged Inundation (>21 days)",
    )),
    "salinity_tier": ("Substrate salinity", tuple(config.SALINITY_TIER_TO_ECE)),
}


def manual_site() -> dict:
    return {"id": "manual", "name": "Manually configured site", "kind": "manual"}


def benchmark_site(hotspot: dict) -> dict:
    return {**hotspot, "kind": "benchmark"}


def coordinate_site(resolved: dict) -> dict:
    return {
        **resolved,
        "id": f"gps:{resolved['latitude']:.5f},{resolved['longitude']:.5f}",
        "name": f"{resolved['county']}, {resolved['country']}",
        "kind": "coordinates",
    }


def baseline_inputs(site: dict) -> dict:
    baseline = {}
    for field, (_, options) in PARAMETER_OPTIONS.items():
        source_field = "default_hazard" if field == "geotechnical_hazard" else field
        value = site.get(source_field)
        baseline[field] = value if value in options else options[0]
    return baseline


def init_session_state() -> None:
    defaults = {
        "messages": [],
        "active_site": manual_site(),
        "site_inputs": baseline_inputs(manual_site()),
        "field_measured_inputs": [],
        "current_assessment": None,
        "site_revision": 0,
        "coord_lat": -1.2921,
        "coord_lon": 36.8219,
        "location_search_results": [],
        "site_shortlist": {},
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    assessment = st.session_state.current_assessment
    if assessment and assessment.get("site", {}).get("id") != st.session_state.active_site["id"]:
        st.session_state.current_assessment = None


def activate_site(site: dict, *, focus_map: bool = False) -> None:
    state = st.session_state
    if site != state.active_site:
        state.active_site = site
        state.site_inputs = baseline_inputs(site)
        state.field_measured_inputs = []
        state.current_assessment = None
        state.site_revision += 1
    if "latitude" in site:
        state.coord_lat = site["latitude"]
        state.coord_lon = site["longitude"]
        if focus_map:
            state.gis_camera_focus = (site["latitude"], site["longitude"])


def input_provenance(site: dict, inputs: dict, measured=()) -> dict:
    sources = {}
    for field, value in inputs.items():
        source_field = "default_hazard" if field == "geotechnical_hazard" else field
        if field in measured:
            sources[field] = "Field-measured (user confirmed)"
        elif source_field not in site:
            sources[field] = (
                "Application default (review before use)"
                if value == PARAMETER_OPTIONS[field][1][0] else "Manual input"
            )
        elif site[source_field] != value:
            sources[field] = "Manual input (overrides baseline)"
        elif site["kind"] == "benchmark":
            sources[field] = "Published benchmark baseline"
        else:
            sources[field] = site.get("input_sources", {}).get(source_field, "Regional default (inferred)")
    if "elevation_m" in site:
        sources["elevation_m"] = (
            "Published benchmark baseline" if site["kind"] == "benchmark"
            else site.get("input_sources", {}).get("elevation_m", "Regional default (inferred)")
        )
    return sources
