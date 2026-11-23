from datetime import datetime

import streamlit as st
import pandas as pd
import pydeck as pdk

from services.biomechanics import calculate_biomechanical_metrics
from services.remediation import get_failure_banner_details, get_two_tier_remediations
from services.model_service import load_regional_hotspots
from geospatial_resolver import resolve_coordinate_geotechnics, search_location
from ui.assessment_inputs import render_assessment_inputs
from ui.state import PARAMETER_OPTIONS, activate_site, benchmark_site, coordinate_site, manual_site


def render_shortlist() -> None:
    state = st.session_state
    shortlist = state.site_shortlist
    if not shortlist:
        return

    def remove_site(site_id: str) -> None:
        state.site_shortlist = {key: value for key, value in shortlist.items() if key != site_id}
        state.gis_compare_sites = [key for key in state.gis_compare_sites if key != site_id]
    with st.expander("Compare shortlisted sites", expanded=True):
        st.caption("Assessments are saved for this session.")
        site_ids = st.multiselect(
            "Sites to compare",
            list(shortlist),
            default=list(shortlist)[:2],
            max_selections=2,
            format_func=lambda site_id: shortlist[site_id]["site"]["name"],
            key="gis_compare_sites",
        )
        if not site_ids:
            return

        columns = st.columns(len(site_ids))
        for column, site_id in zip(columns, site_ids):
            result = shortlist[site_id]
            with column:
                st.subheader(result["site"]["name"], anchor=False)
                verdict = "Suitable" if result["prediction"] else "Unsuitable"
                st.write(f"**Verdict** · {verdict}")
                st.write(f"**CatBoost suitability score** · {result['probability']:.1f}%")
                if result["is_exclusion"]:
                    st.write(f"**Screening rule** · {result['excl_title']}")
                else:
                    effects = result["exp_df"].sort_values("SHAP Impact")
                    if not effects.empty and effects.iloc[0]["SHAP Impact"] < 0:
                        st.write(f"**Largest negative model contribution** · {effects.iloc[0]['Feature']}")
                st.table([
                    {
                        "Input": label,
                        "Value": result["inputs"][field],
                        "Source": result["input_sources"][field],
                    }
                    for field, (label, _) in PARAMETER_OPTIONS.items()
                ])
                st.button("Remove from shortlist", key=f"gis_remove_{site_id}", on_click=remove_site, args=(site_id,))


@st.fragment
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
                key="gis_scope",
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

        county_col, status_col = st.columns(2)
        counties = sorted({h["county_region"] for h in filtered_hotspots})
        statuses = sorted({h["status"] for h in filtered_hotspots})
        counties_selected = county_col.multiselect("County or region", counties, key=f"gis_counties_{scope_mode}")
        statuses_selected = status_col.multiselect("Benchmark status", statuses, key=f"gis_statuses_{scope_mode}")
        filtered_hotspots = [
            h for h in filtered_hotspots
            if (not counties_selected or h["county_region"] in counties_selected)
            and (not statuses_selected or h["status"] in statuses_selected)
        ]
        if not filtered_hotspots:
            st.info("No benchmark pins match these filters.")

        state = st.session_state
        site = state.active_site
        custom_site = site if site["kind"] == "coordinates" else None
        selected_hs = site if site["kind"] == "benchmark" else None
        hotspots_by_id = {h["id"]: h for h in filtered_hotspots}
        hotspot_dropdown_key = "t2_hs_dropdown"
        selected_id = site["id"] if site["id"] in hotspots_by_id else None
        if state.get(hotspot_dropdown_key) != selected_id:
            state[hotspot_dropdown_key] = selected_id

        def inspect_hotspot(hotspot_id, *, focus_map=False):
            hotspot = hotspots_by_id.get(hotspot_id)
            if hotspot is not None:
                activate_site(benchmark_site(hotspot), focus_map=focus_map)

        def select_dropdown_hotspot():
            inspect_hotspot(state[hotspot_dropdown_key], focus_map=True)
            # Reset the chart's own pick state when jumping from the dropdown.
            state.hotspot_map_revision = state.get("hotspot_map_revision", 0) + 1

        if state.get("gis_previous_scope", scope_mode) != scope_mode:
            state.gis_camera_focus = None
        state.gis_previous_scope = scope_mode

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
                        c_site = resolve_coordinate_geotechnics(search_lat, search_lon)
                        activate_site(coordinate_site(c_site), focus_map=True)
                        st.rerun()

                g_c1, g_c2, g_c3 = st.columns([1.2, 1.2, 1.6])
                coordinate_key = f"{st.session_state.coord_lat}_{st.session_state.coord_lon}"
                with g_c1:
                    in_lat = st.number_input(
                        "Latitude (°N)",
                        min_value=-90.0,
                        max_value=90.0,
                        value=float(st.session_state.coord_lat),
                        format="%.5f",
                        step=0.01,
                        key=f"t2_in_lat_{coordinate_key}",
                    )
                with g_c2:
                    in_lon = st.number_input(
                        "Longitude (°E)",
                        min_value=-180.0,
                        max_value=180.0,
                        value=float(st.session_state.coord_lon),
                        format="%.5f",
                        step=0.01,
                        key=f"t2_in_lon_{coordinate_key}",
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
                shortcuts = [
                    ("Nairobi", -1.2921, 36.8219), ("Samburu", 1.0000, 36.8219),
                    ("Eldoret", 0.5143, 35.2698), ("Naivasha", -0.7172, 36.4312),
                    ("Embu", -0.5342, 37.4544), ("Garissa", -0.4532, 39.6461),
                ]
                for column, (name, latitude, longitude) in zip(st.columns(6), shortcuts):
                    if column.button(name, width="stretch", key=f"t2_pill_{name.lower()}"):
                        resolved = resolve_coordinate_geotechnics(latitude, longitude)
                        activate_site(coordinate_site(resolved), focus_map=True)
                        st.rerun()

                if resolve_btn:
                    with st.spinner(f"Querying environmental providers for ({in_lat:.4f}°N, {in_lon:.4f}°E)..."):
                        c_site = resolve_coordinate_geotechnics(in_lat, in_lon)
                        activate_site(coordinate_site(c_site), focus_map=True)
                        st.rerun()

            # Pin inspection leaves the initial view unchanged so deck.gl keeps the current camera.
            focus = state.get("gis_camera_focus")
            if focus:
                map_lat, map_lon = focus
                map_zoom, map_pitch = 9.8, 40
            else:
                map_lat, map_lon = default_lat, default_lon
                map_zoom, map_pitch = default_zoom, default_pitch

            view_state = pdk.ViewState(
                latitude=map_lat,
                longitude=map_lon,
                zoom=map_zoom,
                pitch=map_pitch,
                bearing=0,
            )

            df_map = pd.DataFrame(filtered_hotspots)
            df_active = pd.DataFrame([selected_hs] if selected_id else [])

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
            if custom_site is not None:
                c_site = custom_site
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
                            <div style="margin-bottom: 4px;"><b>Site status:</b> <span style="font-weight: 700;">{status}</span></div>
                            <div style="margin-bottom: 4px;"><b>USCS Soil Class:</b> {soil_mechanical_class}</div>
                            <div style="margin-bottom: 4px;"><b>Precipitation Zone:</b> {climate_rainfall_zone}</div>
                            <div style="margin-bottom: 4px;"><b>Key Metric:</b> {kpi1} | {kpi2}</div>
                            <div style="font-size: 11px; color: #38BDF8; margin-top: 6px; border-top: 1px solid #334155; padding-top: 4px;"><i>Left-click this pin to inspect engineering parameters.</i></div>
                        </div>
                    """,
                    "style": {"zIndex": "1000"}
                }
            )

            revision = st.session_state.get("hotspot_map_revision", 0)
            deck_chart_key = f"pdk_gis_hotspot_selector_{scope_mode}_{revision}_{tuple(hotspots_by_id)}"

            def select_map_hotspot():
                selection = st.session_state[deck_chart_key].selection
                for layer_id in ("hotspot-scatter", "hotspot-labels"):
                    objects = selection.get("objects", {}).get(layer_id, [])
                    hotspot_id = objects[0].get("id") if objects and isinstance(objects[0], dict) else None
                    if isinstance(hotspot_id, str):
                        inspect_hotspot(hotspot_id)
                        return

                    indices = selection.get("indices", {}).get(layer_id, [])
                    if indices and isinstance(indices[0], int) and 0 <= indices[0] < len(filtered_hotspots):
                        inspect_hotspot(filtered_hotspots[indices[0]]["id"])
                        return

            with st.container(key="gis_hotspot_map", gap=None):
                st.pydeck_chart(
                    deck,
                    width="stretch",
                    height=650,
                    on_select=select_map_hotspot,
                    selection_mode="single-object",
                    key=deck_chart_key,
                )
                st.html(
                    """
                    <script>
                    (() => {
                        const map = document.currentScript.closest('.st-key-gis_hotspot_map');
                        if (!map || map.dataset.contextMenuDisabled) return;

                        map.addEventListener('contextmenu', (event) => event.preventDefault(), { capture: true });
                        map.dataset.contextMenuDisabled = 'true';
                    })();
                    </script>
                    """,
                    unsafe_allow_javascript=True,
                )

            st.caption("Click a pin or label to inspect. Ctrl + drag to tilt or rotate. Scroll to zoom.")
            legend = {h["status"]: h["color"] for h in hotspots}
            markers = [
                f'<span style="color:rgb({",".join(map(str, color[:3]))})">●</span> {status}'
                for status, color in legend.items()
            ]
            st.markdown(" · ".join([*markers, "Cyan ring · active benchmark", "Amber · GPS site"]), unsafe_allow_html=True)

        # ── INSPECTOR & GEOTECHNICAL WORKBENCH (RIGHT ~38%) ──
        with col_inspector:
            with st.container(border=True):
                hotspot_labels = {h["id"]: f"{h['county_region']} · {h['name']}" for h in filtered_hotspots}
                st.selectbox(
                    "Jump to benchmark",
                    list(hotspot_labels),
                    index=None,
                    format_func=hotspot_labels.__getitem__,
                    key=hotspot_dropdown_key,
                    on_change=select_dropdown_hotspot,
                    disabled=not hotspot_labels,
                )
                if custom_site is not None:
                    c_res = custom_site
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

                elif selected_hs is not None:
                    st.markdown(f"#### :material/place: {selected_hs['name']}")
                    stat_badge = "badge-suitable" if selected_hs['status'] == "Suitable" else "badge-unsuitable"
                    st.markdown(f"""
                    <div style="display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 10px;">
                        <span class="studio-badge {stat_badge}">Benchmark status · {selected_hs['status']}</span>
                        <span class="studio-badge badge-blue">Elevation: {selected_hs['elevation_m']} m</span>
                        <span class="studio-badge badge-amber">GPS: {selected_hs['latitude']:.4f}°N, {selected_hs['longitude']:.4f}°E</span>
                    </div>
                    """, unsafe_allow_html=True)
                    st.caption(f"**Empirical Summary:** {selected_hs['summary_desc']}")
                    st.caption(f"**Citation:** *{selected_hs['citation']}*")

                else:
                    st.info("Select a benchmark pin or locate a GPS site to assess it.")

            if site["kind"] != "manual":
                with st.container(border=True):
                    st.markdown("##### :material/tune: Geotechnical parameters")
                    if custom_site and custom_site.get("notes"):
                        st.caption(custom_site["notes"])
                    if custom_site and st.button("Clear GPS site", key="btn_t2_clear"):
                        activate_site(manual_site())
                        st.rerun()
                    render_assessment_inputs("t2", cb_model)

                active_res = state.current_assessment
                if active_res:
                    res_pred = active_res["prediction"]
                    res_prob = active_res["probability"]
                    if active_res["is_exclusion"]:
                        st.warning(f"Screening rule override · {active_res['excl_title']}")
                    bio = active_res.get("biomechanics") or calculate_biomechanical_metrics(active_res.get("inputs", {}))

                    if res_pred == 1:
                        st.markdown(f"""
                        <div class="verdict-banner-suitable" style="padding: 14px 16px; margin-top: 12px; margin-bottom: 0;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <span class="studio-badge badge-suitable">SUITABLE</span>
                                <span style="font-size: 1.35rem; font-weight: 800; color: #14532D;">{res_prob:.1f}% CatBoost suitability score</span>
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
                                <span style="font-size: 1.35rem; font-weight: 800; color: #7F1D1D;">{res_prob:.1f}% CatBoost suitability score</span>
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

                    if st.button("Save assessment to shortlist", key="gis_save_site", icon=":material/bookmark:"):
                        state.site_shortlist = {**state.site_shortlist, site["id"]: active_res}
                        st.success("Assessment saved to the shortlist.")

        render_shortlist()


