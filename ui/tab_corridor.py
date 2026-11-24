from datetime import datetime
from html import escape

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
    with st.expander("Compare shortlisted sites", icon=":material/compare_arrows:"):
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


def locate_site(latitude: float, longitude: float) -> None:
    with st.spinner("Looking up site conditions…"):
        resolved = resolve_coordinate_geotechnics(latitude, longitude)
        activate_site(coordinate_site(resolved), focus_map=True)
    st.session_state.gis_close_finder = True
    st.rerun()


def render_location_picker() -> None:
    state = st.session_state
    search, action = st.columns([2.6, 1], vertical_alignment="bottom")
    query = search.text_input("Place name", placeholder="Town or landmark", key="t2_place_search")
    if action.button("Search", width="stretch", key="btn_t2_place_search"):
        state.location_search_results = search_location(query.strip()) if query.strip() else []
        if not state.location_search_results:
            message = "No places matched. Try a town, county, or landmark." if query.strip() else "Enter a place name."
            st.warning(message)

    if state.location_search_results:
        results = state.location_search_results
        selected = st.selectbox(
            "Search result", [result["display_name"] for result in results], key="t2_place_search_result"
        )
        if st.button("Use selected location", type="primary", width="stretch", key="btn_t2_use_search_result"):
            result = next(result for result in results if result["display_name"] == selected)
            locate_site(result["latitude"], result["longitude"])

    st.divider()
    coordinate_key = f"{state.coord_lat}_{state.coord_lon}"
    latitude_col, longitude_col = st.columns(2)
    latitude = latitude_col.number_input(
        "Latitude", -90.0, 90.0, value=float(state.coord_lat), format="%.5f", step=0.01,
        key=f"t2_in_lat_{coordinate_key}",
    )
    longitude = longitude_col.number_input(
        "Longitude", -180.0, 180.0, value=float(state.coord_lon), format="%.5f", step=0.01,
        key=f"t2_in_lon_{coordinate_key}",
    )
    if st.button("Locate coordinates", type="primary", width="stretch", key="btn_t2_locate"):
        locate_site(latitude, longitude)

    st.caption("Quick locations")
    shortcuts = [
        ("Nairobi", -1.2921, 36.8219), ("Samburu", 1.0000, 36.8219),
        ("Eldoret", 0.5143, 35.2698), ("Naivasha", -0.7172, 36.4312),
        ("Embu", -0.5342, 37.4544), ("Garissa", -0.4532, 39.6461),
    ]
    with st.container(horizontal=True, gap="xsmall"):
        for name, lat, lon in shortcuts:
            if st.button(name, key=f"t2_pill_{name.lower()}"):
                locate_site(lat, lon)


def render_hotspot_map(
    filtered_hotspots: list[dict],
    active_hotspot: dict | None,
    custom_site: dict | None,
    scope_mode: str,
    default_camera: tuple[float, float, float, float],
    map_style: str,
    show_3d_elevation: bool,
    inspect_hotspot,
) -> None:
    default_lat, default_lon, default_zoom, default_pitch = default_camera
    # Pin inspection leaves the initial view unchanged so deck.gl keeps the current camera.
    focus = st.session_state.get("gis_camera_focus")
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
    df_active = pd.DataFrame([active_hotspot] if active_hotspot else [])

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
        map_style=map_style,
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
    deck_chart_key = f"pdk_gis_hotspot_selector_{scope_mode}_{revision}_{tuple(h['id'] for h in filtered_hotspots)}"

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
            height=420,
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


def render_site_details(site: dict) -> None:
    state = st.session_state
    custom_site = site["kind"] == "coordinates"
    with st.container(key="gis_site_card", border=True, gap="small"):
        if site["kind"] == "manual":
            st.subheader("Active site", anchor=False)
            st.caption("Select a map pin, choose a benchmark, or use Find site to see its details here.")
            return

        with st.container(horizontal=True, horizontal_alignment="distribute", vertical_alignment="center"):
            st.subheader(site["name"], anchor=False)
            if custom_site and st.button("Clear GPS site", key="btn_t2_clear"):
                activate_site(manual_site())
                state.gis_camera_focus = None
                st.rerun()

        badges = [("blue", f"Elevation · {site['elevation_m']} m")]
        badges.append(("amber", f"GPS · {site['latitude']:.4f}°, {site['longitude']:.4f}°"))
        if custom_site:
            badges.append(("blue", site["data_sources"]))
        else:
            status = site["status"]
            tone = "suitable" if status == "Suitable" else "unsuitable"
            badges.insert(0, (tone, f"Benchmark · {status}"))
        st.html(
            '<div class="gis-site-badges">' + "".join(
                f'<span class="studio-badge badge-{tone}">{escape(str(label))}</span>' for tone, label in badges
            ) + '</div>'
        )

        with st.expander("Site evidence", icon=":material/article:", expanded=True):
            if not custom_site:
                st.markdown(site["summary_desc"])
                st.caption(site["citation"])
            elif site["is_known_hotspot"]:
                match = site["matched_hotspot"]
                st.markdown(f"**Nearby benchmark · {match['name']}** ({site['distance_to_nearest_km']} km away)")
                st.caption(match["summary_desc"])
            else:
                st.caption(f"Nearest documented benchmark is {site['distance_to_nearest_km']} km away.")
            if site.get("notes"):
                st.caption(site["notes"])


def render_site_assessment(site: dict, cb_model) -> None:
    with st.container(key="gis_inputs_card", border=True):
        st.markdown("#### Assessment inputs")
        render_assessment_inputs("t2", cb_model, column_count=3, show_site=False)

    result = st.session_state.current_assessment
    if result is not None:
        render_assessment_result(result, site["id"])


def render_assessment_result(result: dict, site_id: str) -> None:
    state = st.session_state
    suitable = result["prediction"] == 1
    if suitable:
        bio = result.get("biomechanics") or calculate_biomechanical_metrics(result.get("inputs", {}))
        badge = "Suitable"
        title = "Positive geotechnical alignment"
        summary = bio.get("summary_desc", "")
        metrics = [
            ("Root depth", bio.get("kpi1_val", bio["root_depth"])),
            ("Added cohesion", bio.get("kpi2_val", bio["root_cohesion"])),
            ("FoS boost", bio.get("kpi3_val", bio["fos_boost"])),
        ]
    else:
        diagnosis = get_failure_banner_details(result)
        badge, title, summary = diagnosis["badge"], diagnosis["title"], diagnosis["subtitle"]
        metrics = [(diagnosis[f"kpi{i}_title"], diagnosis[f"kpi{i}_val"]) for i in range(1, 4)]

    tone = "suitable" if suitable else "unsuitable"
    metric_html = "".join(
        f'<div><dt>{escape(str(label))}</dt><dd>{escape(str(value))}</dd></div>'
        for label, value in metrics
    )
    with st.container(key="gis_result_card", border=True):
        st.markdown("#### Assessment result")
        if result["is_exclusion"]:
            st.warning(f"Screening rule override · {result['excl_title']}")
        st.html(f"""
            <div class="gis-verdict gis-verdict-{tone}">
                <div class="gis-verdict-heading">
                    <div>
                        <span class="studio-badge badge-{tone}">{escape(badge)}</span>
                        <h5>{escape(title)}</h5>
                        <p>{escape(summary)}</p>
                    </div>
                    <div class="gis-verdict-score">
                        <strong>{result['probability']:.1f}%</strong>
                        <span>CatBoost suitability score</span>
                    </div>
                </div>
                <dl class="gis-verdict-metrics">{metric_html}</dl>
            </div>
        """)
        with st.container(horizontal=True, gap="small"):
            if result.get("report_md"):
                st.download_button(
                    "Download geotechnical report",
                    data=result["report_md"],
                    file_name=f"vetiver_site_assessment_{datetime.now():%Y%m%d_%H%M%S}.md",
                    mime="text/markdown",
                    icon=":material/download:",
                    key="btn_t2_download_report",
                )
            if st.button("Save to shortlist", key="gis_save_site", icon=":material/bookmark:"):
                state.site_shortlist = {**state.site_shortlist, site_id: result}
                st.toast("Assessment saved to the shortlist.", icon=":material/bookmark:")

        if result.get("exp_df") is not None:
            with st.expander("Model contributions (TreeSHAP)", icon=":material/bar_chart:"):
                st.bar_chart(result["exp_df"].set_index("Feature")["SHAP Impact"])
                st.dataframe(
                    result["exp_df"],
                    column_config={
                        "Feature": "Parameter",
                        "Input Value": "Configured",
                        "SHAP Impact": st.column_config.NumberColumn("Log-odds impact", format="%.4f"),
                        "Effect": "Direction",
                    },
                    hide_index=True,
                    width="stretch",
                )

        if not suitable:
            with st.expander("Recommended countermeasures", icon=":material/handyman:"):
                remed = get_two_tier_remediations(result)
                st.markdown(f"**Targeted action · {remed['primary_title']}**")
                st.caption(remed["primary_mech"])
                for protocol, _ in remed["protocols"]:
                    st.markdown(f"- {protocol}")


@st.fragment
def render_tab_corridor(cb_model=None, hotspot_mtime: float = 0.0) -> None:
    hotspots = load_regional_hotspots(hotspot_mtime)
    state = st.session_state

    with st.container(key="vetiver_gis", gap="small"):
        if not hotspots:
            st.subheader("3D GIS Studio", anchor=False)
            st.warning("The regional benchmark data could not be loaded.")
            return

        if state.pop("gis_close_finder", False):
            state.gis_find_site = False
        with st.container(
            key="gis_header", horizontal=True, horizontal_alignment="distribute", vertical_alignment="center",
        ):
            st.subheader("3D GIS Studio", anchor=False, width="content")
            with st.container(horizontal=True, width="content", gap="xsmall"):
                with st.popover("Find site", icon=":material/search:", key="gis_find_site", on_change="rerun"):
                    render_location_picker()
                settings = st.popover("Map settings", icon=":material/tune:")

        with st.container(key="gis_map_toolbar", border=True):
            region, benchmark = st.columns([1, 2.2])
            scope_labels = {
                "Kenya Corridors & Counties (Detailed)": "Kenya",
                "East Africa Regional Sites": "East Africa",
                "All Global Benchmark Sites": "Global",
            }
            scope_mode = region.selectbox("Region", list(scope_labels), format_func=scope_labels.__getitem__, key="gis_scope")
            if scope_labels[scope_mode] == "Kenya":
                scoped_hotspots = [hotspot for hotspot in hotspots if hotspot.get("country") == "Kenya"]
                default_camera = (0.2, 37.8, 6.5, 30)
            elif scope_labels[scope_mode] == "East Africa":
                scoped_hotspots = [
                    hotspot for hotspot in hotspots if hotspot.get("country") in {"Kenya", "Ethiopia", "Zambia", "Zimbabwe"}
                ]
                default_camera = (-2.0, 34.5, 4.8, 25)
            else:
                scoped_hotspots = hotspots
                default_camera = (5.0, 25.0, 2.2, 20)

            with settings:
                style_urls = {
                    "Voyager": "https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json",
                    "Positron": "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
                    "Dark Matter": "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
                }
                basemap = st.selectbox("Basemap", list(style_urls), key="gis_basemap")
                show_3d_elevation = st.toggle("Show elevation pillars", key="gis_show_elevation")
                counties = st.multiselect(
                    "County or region", sorted({h["county_region"] for h in scoped_hotspots}), key=f"gis_counties_{scope_mode}"
                )
                statuses = st.multiselect(
                    "Benchmark status", sorted({h["status"] for h in scoped_hotspots}), key=f"gis_statuses_{scope_mode}"
                )

            filtered = [
                hotspot for hotspot in scoped_hotspots
                if (not counties or hotspot["county_region"] in counties)
                and (not statuses or hotspot["status"] in statuses)
            ]
            site = state.active_site
            custom_site = site if site["kind"] == "coordinates" else None
            hotspots_by_id = {h["id"]: h for h in filtered}
            active_hotspot = hotspots_by_id.get(site["id"])
            dropdown_key = "t2_hs_dropdown"
            selected_id = active_hotspot["id"] if active_hotspot else None
            if state.get(dropdown_key) != selected_id:
                state[dropdown_key] = selected_id

            def inspect_hotspot(hotspot_id: str | None, *, focus_map: bool = False) -> None:
                hotspot = hotspots_by_id.get(hotspot_id)
                if hotspot is not None:
                    activate_site(benchmark_site(hotspot), focus_map=focus_map)

            def jump_to_hotspot() -> None:
                inspect_hotspot(state[dropdown_key], focus_map=True)
                state.hotspot_map_revision = state.get("hotspot_map_revision", 0) + 1

            labels = {h["id"]: f"{h['county_region']} · {h['name']}" for h in filtered}
            benchmark.selectbox(
                "Jump to benchmark", list(labels), index=None, format_func=labels.__getitem__,
                key=dropdown_key, on_change=jump_to_hotspot, disabled=not labels,
                placeholder="Choose a benchmark or select a map pin",
            )

        if state.get("gis_previous_scope", scope_mode) != scope_mode:
            state.gis_camera_focus = None
        state.gis_previous_scope = scope_mode

        with st.container(key="gis_explorer", gap=None):
            map_column, site_column = st.columns([2, 1], gap="small")

        with map_column, st.container(key="gis_map_card", border=True, gap=None):
            render_hotspot_map(
                filtered, active_hotspot, custom_site, scope_mode, default_camera,
                style_urls[basemap], show_3d_elevation, inspect_hotspot,
            )

            with st.container(
                key="gis_map_footer", horizontal=True, horizontal_alignment="distribute",
                vertical_alignment="center", gap="xsmall",
            ):
                count = len(filtered)
                if count:
                    st.caption(
                        f"{count} benchmark{'s' if count != 1 else ''} · Click a pin to inspect it. Ctrl + drag to tilt or rotate.",
                        width="content",
                    )
                else:
                    st.caption("No benchmarks match these filters. Adjust them in Map settings.", width="content")
                legend = {h["status"]: h["color"] for h in hotspots}
                markers = [
                    f'<span><i style="--pin-color:rgb({",".join(map(str, color[:3]))})"></i>{escape(status)}</span>'
                    for status, color in legend.items()
                ]
                markers.extend([
                    '<span><i class="gis-legend-ring"></i>Active benchmark</span>',
                    '<span><i style="--pin-color:#f59e0b"></i>GPS site</span>',
                ])
                st.html(f'<div class="gis-legend">{"".join(markers)}</div>', width="content")

        with site_column:
            render_site_details(site)

        if site["kind"] != "manual":
            render_site_assessment(site, cb_model)

        render_shortlist()
