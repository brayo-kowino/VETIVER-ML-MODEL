import os
import json
from datetime import datetime

import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium

import config
from services.biomechanics import (
    HAZARD_DISPLAY_MAP as hazard_display_map,
    calculate_biomechanical_metrics,
)
from services.remediation import get_failure_banner_details, get_two_tier_remediations
from services.assessment_service import run_assessment
from services.model_service import load_regional_hotspots
from geospatial_resolver import resolve_coordinate_geotechnics
from geotechnical_engine import calculate_slope_stability, calculate_field_boq
def render_tab_corridor(cb_model=None, hotspot_mtime: float = 0.0) -> None:
    st.subheader("Regional Hotspots & 3D GIS Map", anchor=False)

    hotspots = load_regional_hotspots(hotspot_mtime)
    if not hotspots:
        st.warning("No regional hotspot database found at data/regional_hotspots.json.", icon=":material/warning:")
    else:
        # Top Compact Bar: Geographic Scope, Basemap Style, 3D Elevation
        top_c1, top_c2 = st.columns([1.8, 1.3])
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

        # Validate that session state contains a valid hotspot id within the active filter
        valid_ids = [h["id"] for h in filtered_hotspots]
        if "selected_hotspot_id" not in st.session_state or st.session_state.selected_hotspot_id not in valid_ids:
            st.session_state.selected_hotspot_id = valid_ids[0]

        cur_idx = next((i for i, h in enumerate(filtered_hotspots) if h["id"] == st.session_state.selected_hotspot_id), 0)
        selected_hs = filtered_hotspots[cur_idx]

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

            if st.session_state.get("pan_to_custom") and st.session_state.custom_coord_site:
                map_lat = st.session_state.custom_coord_site["latitude"]
                map_lon = st.session_state.custom_coord_site["longitude"]
                map_zoom = 9
            else:
                map_lat = selected_hs["latitude"]
                map_lon = selected_hs["longitude"]
                map_zoom = 7 if "Kenya" in scope_mode else 5

            tile_name = {
                "Positron (Light Boundaries & Roads)": "CartoDB positron",
                "Voyager (Topography & Contours)": "OpenStreetMap",
                "Dark Matter (High-Contrast GIS)": "CartoDB dark_matter",
            }.get(map_style_choice, "CartoDB positron")

            folium_map = folium.Map(
                location=[map_lat, map_lon],
                zoom_start=map_zoom,
                tiles=tile_name,
                control_scale=True,
            )

            for hotspot in filtered_hotspots:
                is_selected = hotspot["id"] == st.session_state.selected_hotspot_id
                folium.CircleMarker(
                    location=[hotspot["latitude"], hotspot["longitude"]],
                    radius=8 if is_selected else 6,
                    color="#14532D" if is_selected else "#2563EB",
                    fill=True,
                    fill_color="#22C55E" if is_selected else "#60A5FA",
                    fill_opacity=0.95,
                    weight=3 if is_selected else 2,
                    tooltip=f"{hotspot['name']} — {hotspot['county_region']}",
                    popup=folium.Popup(
                        f"<b>{hotspot['name']}</b><br>"
                        f"{hotspot['county_region']}, {hotspot['country']}<br>"
                        f"Elevation: {hotspot['elevation_m']} m",
                        max_width=260,
                    ),
                ).add_to(folium_map)

            if st.session_state.custom_coord_site is not None:
                c_site = st.session_state.custom_coord_site
                folium.Marker(
                    location=[c_site["latitude"], c_site["longitude"]],
                    tooltip=f"Selected site: {c_site['county']}",
                    icon=folium.Icon(color="orange", icon="crosshairs", prefix="fa"),
                ).add_to(folium_map)

            map_event = st_folium(
                folium_map,
                height=650,
                use_container_width=True,
                returned_objects=["last_clicked", "last_object_clicked"],
                key="folium_gis_click_map",
            )

            click_location = map_event.get("last_clicked") or map_event.get("last_object_clicked")
            if click_location:
                click_lat = float(click_location["lat"])
                click_lon = float(click_location["lng"])
                click_key = f"{click_lat:.5f},{click_lon:.5f}"

                if click_key != st.session_state.processed_map_selection:
                    matched_hotspot = None
                    for hotspot in filtered_hotspots:
                        if abs(hotspot["latitude"] - click_lat) < 0.0005 and abs(hotspot["longitude"] - click_lon) < 0.0005:
                            matched_hotspot = hotspot
                            break

                    st.session_state.processed_map_selection = click_key
                    st.session_state.coord_lat = click_lat
                    st.session_state.coord_lon = click_lon

                    if matched_hotspot is not None:
                        st.session_state.selected_hotspot_id = matched_hotspot["id"]
                        st.session_state.custom_coord_site = None
                    else:
                        c_site = resolve_coordinate_geotechnics(click_lat, click_lon)
                        st.session_state.custom_coord_site = c_site
                        st.session_state.preset_soil = c_site["soil_mechanical_class"]
                        st.session_state.preset_climate = c_site["climate_rainfall_zone"]
                        st.session_state.preset_moisture = c_site["establishment_moisture"]
                        st.session_state.preset_hydro = c_site["hydrological_stress"]
                        st.session_state.preset_salinity = c_site["salinity_tier"]
                        st.session_state.preset_hazard = c_site["default_hazard"]

                    st.session_state.pan_to_custom = matched_hotspot is None
                    st.session_state.current_assessment = None
                    st.session_state.custom_assessment_result = None
                    st.rerun()

            st.caption(":material/info: **GIS Guide:** Left-click anywhere to ingest that coordinate. Click a colored pin to inspect that hotspot.")

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
                        st.session_state.current_assessment = None
                        st.session_state.custom_assessment_result = None
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
                ab_c1, ab_c2 = st.columns([2, 1])
                with ab_c1:
                    run_t2_eval = st.button("Re-evaluate Parameters", type="primary", width="stretch", icon=":material/refresh:", key="btn_t2_run_eval")
                with ab_c2:
                    clear_clicked = False
                    if st.session_state.custom_coord_site is not None:
                        clear_clicked = st.button("Clear", width="stretch", icon=":material/close:", key="btn_t2_clear")
                        if clear_clicked:
                            st.session_state.custom_coord_site = None
                            st.session_state.custom_assessment_result = None
                            st.session_state.pan_to_custom = False
                            st.rerun()

                if run_t2_eval or st.session_state.current_assessment is None:
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
                    st.session_state.current_assessment = run_assessment(eval_inputs, cb_model=cb_model)
                    st.session_state.custom_assessment_result = st.session_state.current_assessment
                    if run_t2_eval:
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

                    # 1-Click direct report download right here on the map!
                    if active_res.get("report_md"):
                        st.download_button(
                            label="Download Full Geotechnical Report (.md)",
                            data=active_res["report_md"],
                            file_name=f"vetiver_site_assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                            mime="text/markdown",
                            icon=":material/download:",
                            width="stretch",
                            key="btn_t2_download_report",
                        )

                    # AI Decision Attribution (SHAP) directly on the map
                    if active_res.get("exp_df") is not None:
                        with st.expander("📊 AI Decision Attribution (TreeSHAP)", expanded=False):
                            chart_data = active_res["exp_df"].set_index("Feature")["SHAP Impact"]
                            st.bar_chart(chart_data)
                            st.dataframe(
                                active_res["exp_df"],
                                column_config={
                                    "Feature": "Parameter",
                                    "Input Value": "Configured",
                                    "SHAP Impact": st.column_config.NumberColumn("Log-Odds Impact", format="%.4f"),
                                    "Effect": "Direction",
                                },
                                hide_index=True,
                                width="stretch",
                            )

                    # Recommended Remediations directly on the map if unsuitable
                    if res_pred == 0:
                        with st.expander("🛠️ Recommended Countermeasures & Remediations", expanded=False):
                            remed = get_two_tier_remediations(active_res)
                            st.markdown(f"**Targeted Action:** {remed['primary_title']}")
                            st.caption(remed['primary_mech'])
                            for proto_text, _ in remed['protocols']:
                                st.markdown(f"- {proto_text}")

                    st.caption("✅ Analysis complete. Parameters & verdict are automatically synchronized across the platform.")


