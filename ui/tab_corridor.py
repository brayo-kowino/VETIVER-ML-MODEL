import os
import json
from datetime import datetime

import streamlit as st
import pandas as pd
import pydeck as pdk

import config
from services.biomechanics import (
    HAZARD_DISPLAY_MAP as hazard_display_map,
    calculate_biomechanical_metrics,
)
from services.remediation import get_failure_banner_details, get_two_tier_remediations
from services.assessment_service import run_assessment
from services.model_service import load_regional_hotspots
from geospatial_resolver import resolve_coordinate_geotechnics, search_location
from geotechnical_engine import calculate_slope_stability, calculate_field_boq


def sync_coordinate_inputs(latitude: float, longitude: float) -> None:
    """Synchronize shared coordinate state without mutating existing widgets."""
    st.session_state.coord_lat = latitude
    st.session_state.coord_lon = longitude


def render_tab_corridor(cb_model=None, hotspot_mtime: float = 0.0) -> None:
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
                index=1,
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

        # ── 62 / 38 SPLIT-SCREEN STUDIO LAYOUT ──
        col_map, col_inspector = st.columns([1.58, 1.0], gap="medium")

        with col_map:
            # Inline GPS Geocoder & Quick Corridor Bar
            with st.container(border=True):
                search_text, search_button = st.columns([3.2, 1.0])
                with search_text:
                    place_query = st.text_input(
                        "Search place name",
                        placeholder="Nairobi, Moi International Airport, Voi...",
                        key="t2_place_search",
                    )
                with search_button:
                    st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
                    place_search_btn = st.button(
                        "Search",
                        width="stretch",
                        icon=":material/search:",
                        key="btn_t2_place_search",
                    )

                if place_search_btn:
                    if place_query.strip():
                        st.session_state.location_search_results = search_location(place_query)
                        if not st.session_state.location_search_results:
                            st.warning("No location matches were found. Try a town, county, or landmark name.", icon=":material/search_off:")
                    else:
                        st.session_state.location_search_results = []
                        st.warning("Enter a place name to search.", icon=":material/info:")

                if st.session_state.location_search_results:
                    result_labels = [
                        result["display_name"]
                        for result in st.session_state.location_search_results
                    ]
                    selected_result_label = st.selectbox(
                        "Search result",
                        result_labels,
                        key="t2_place_search_result",
                    )
                    selected_result = next(
                        result
                        for result in st.session_state.location_search_results
                        if result["display_name"] == selected_result_label
                    )
                    if st.button("Use selected location", type="primary", width="stretch", icon=":material/my_location:", key="btn_t2_use_search_result"):
                        search_lat = selected_result["latitude"]
                        search_lon = selected_result["longitude"]
                        sync_coordinate_inputs(search_lat, search_lon)
                        c_site = resolve_coordinate_geotechnics(search_lat, search_lon)
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

                g_c1, g_c2, g_c3 = st.columns([1.2, 1.2, 1.6])
                with g_c1:
                    in_lat = st.number_input(
                        "Latitude (°N)",
                        min_value=-90.0,
                        max_value=90.0,
                        value=float(st.session_state.coord_lat),
                        format="%.5f",
                        step=0.01,
                        key="t2_in_lat",
                    )
                with g_c2:
                    in_lon = st.number_input(
                        "Longitude (°E)",
                        min_value=-180.0,
                        max_value=180.0,
                        value=float(st.session_state.coord_lon),
                        format="%.5f",
                        step=0.01,
                        key="t2_in_lon",
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
                    sync_coordinate_inputs(-1.2921, 36.8219)
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
                    sync_coordinate_inputs(1.0000, 36.8219)
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
                    sync_coordinate_inputs(0.5143, 35.2698)
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
                    sync_coordinate_inputs(-0.7172, 36.4312)
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
                    sync_coordinate_inputs(-0.5342, 37.4544)
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
                    sync_coordinate_inputs(-0.4532, 39.6461)
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
                        sync_coordinate_inputs(in_lat, in_lon)
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

                selection_key = f"{deck_chart_key}:{clicked_id}" if clicked_id else None
                already_processed = selection_key == st.session_state.processed_map_selection

                if (
                    clicked_id
                    and clicked_id in valid_ids
                    and clicked_id != st.session_state.selected_hotspot_id
                    and not already_processed
                ):
                    st.session_state.processed_map_selection = selection_key
                    st.session_state.selected_hotspot_id = clicked_id
                    clicked_hotspot = next(h for h in filtered_hotspots if h["id"] == clicked_id)
                    sync_coordinate_inputs(clicked_hotspot["latitude"], clicked_hotspot["longitude"])
                    st.session_state.custom_coord_site = None
                    st.session_state.pan_to_custom = False
                    st.session_state.current_assessment = None
                    st.session_state.custom_assessment_result = None
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
                        sync_coordinate_inputs(chosen_h["latitude"], chosen_h["longitude"])
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


