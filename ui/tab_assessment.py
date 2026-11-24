import os
from html import escape
from datetime import datetime

import streamlit as st
import pandas as pd

import config
from services.biomechanics import calculate_biomechanical_metrics
from services.remediation import (
    get_two_tier_remediations,
    get_failure_banner_details,
)
from ui.assessment_inputs import render_assessment_inputs
from ui.state import activate_site, benchmark_site, coordinate_site, manual_site
from services.model_service import load_regional_hotspots
from geospatial_resolver import resolve_coordinate_geotechnics
from geotechnical_engine import (
    generate_slope_cross_section_fig,
    generate_fos_sensitivity_fig,
    calculate_field_boq,
    generate_temporal_maturation_fig,
)


def render_project_location(hotspot_mtime: float) -> None:
    state = st.session_state
    site = state.active_site
    st.subheader(":material/place: Project location", anchor=False)

    sources = {
        "coordinates": "GPS coordinates",
        "benchmark": "Benchmark site",
        "shortcuts": "Quick locations",
        "manual": "Manual entry",
    }
    preferred = site["kind"] if site["kind"] != "manual" else "coordinates"
    source = st.selectbox(
        "Location source", list(sources), index=list(sources).index(preferred),
        format_func=sources.__getitem__, key="t1_location_source",
    )

    if source == "coordinates":
        latitude_col, longitude_col = st.columns(2, gap="xsmall")
        latitude = latitude_col.number_input(
            "Latitude", min_value=-90.0, max_value=90.0, value=float(state.coord_lat),
            format="%.5f", step=0.01, key=f"t1_input_geo_lat_{state.site_revision}",
        )
        longitude = longitude_col.number_input(
            "Longitude", min_value=-180.0, max_value=180.0, value=float(state.coord_lon),
            format="%.5f", step=0.01, key=f"t1_input_geo_lon_{state.site_revision}",
        )
        if st.button("Locate site", icon=":material/my_location:", key="btn_t1_resolve_custom_gps"):
            load_location(latitude, longitude)
    elif source == "benchmark":
        hotspots = load_regional_hotspots(hotspot_mtime)
        by_id = {item["id"]: item for item in hotspots}
        if by_id:
            labels = {item["id"]: f"{item['county_region']} · {item['name']}" for item in hotspots}
            options = list(labels)
            selected = st.selectbox(
                "Benchmark location", options, index=options.index(site["id"]) if site["id"] in labels else None,
                format_func=labels.__getitem__, key="t1_benchmark_location", placeholder="Choose a benchmark",
            )
            if st.button("Use benchmark", icon=":material/add_location:", key="btn_apply_hotspot_preset", disabled=selected is None):
                activate_site(benchmark_site(by_id[selected]), focus_map=True)
                st.rerun()
        else:
            st.info("No benchmark locations are available.", icon=":material/info:")
    elif source == "shortcuts":
        shortcuts = [
            ("Nairobi", -1.2921, 36.8219), ("Samburu", 1.0000, 36.8219),
            ("Eldoret", 0.5143, 35.2698), ("Naivasha", -0.7172, 36.4312),
            ("Embu", -0.5342, 37.4544), ("Garissa", -0.4532, 39.6461),
        ]
        with st.container(horizontal=True, gap="xsmall"):
            for name, latitude, longitude in shortcuts:
                if st.button(name, key=f"t1_btn_{name.lower()}"):
                    load_location(latitude, longitude)
    elif site["kind"] != "manual":
        if st.button("Start a manual assessment", icon=":material/edit_note:", key="t1_switch_manual"):
            activate_site(manual_site())
            st.rerun()

    with st.container(key="assessment_active_site", gap="xsmall"):
        st.markdown(f":material/location_on: **{site['name']}**")
        if site["kind"] == "coordinates":
            st.caption(f"Elevation · {site['elevation_m']} m · {site['data_sources']}")
        elif site["kind"] == "benchmark":
            st.caption(f"Benchmark · {site['county_region']} · {site['elevation_m']} m")
        else:
            st.caption("A GPS location or benchmark can prefill the site conditions.")


def load_location(latitude: float, longitude: float) -> None:
    with st.spinner("Looking up site conditions…"):
        activate_site(coordinate_site(resolve_coordinate_geotechnics(latitude, longitude)), focus_map=True)
    st.rerun()


def render_suitability_result(assessment: dict) -> None:
    with st.container(key="assessment_result", border=True):
        with st.container(horizontal=True, horizontal_alignment="distribute", vertical_alignment="center"):
            st.subheader(":material/analytics: Assessment result", anchor=False, width="content")
            st.download_button(
                "Download report", data=assessment["report_md"],
                file_name=f"vetiver_site_assessment_{datetime.now():%Y%m%d_%H%M%S}.md",
                mime="text/markdown", icon=":material/download:", key="t1_btn_download_report_top",
                on_click="ignore",
            )
        if assessment.get("is_exclusion"):
            st.error(
                f"**Screening rule · {assessment['excl_title']}**\n\n{assessment['excl_details']}",
                icon=":material/block:",
            )

        suitable = assessment["prediction"] == 1
        tone = "suitable" if suitable else "unsuitable"
        if suitable:
            bio = assessment.get("biomechanics") or calculate_biomechanical_metrics(assessment.get("inputs", {}))
            badge = "Suitable"
            title = "Positive geotechnical alignment"
            summary = bio.get("summary_desc", "")
            metrics = [
                (bio.get("kpi1_label", "Root depth potential"), bio.get("kpi1_val", bio["root_depth"])),
                (bio.get("kpi2_label", "Mean tensile strength"), bio.get("kpi2_val", bio["tensile_strength"])),
                (bio.get("kpi3_label", "Safety factor boost"), bio.get("kpi3_val", bio["fos_boost"])),
            ]
        else:
            diagnosis = get_failure_banner_details(assessment)
            badge, title, summary = diagnosis["badge"], diagnosis["title"], diagnosis["subtitle"]
            metrics = [(diagnosis[f"kpi{i}_title"], diagnosis[f"kpi{i}_val"]) for i in range(1, 4)]

        metric_html = "".join(
            f'<div><dt>{escape(str(label))}</dt><dd>{escape(str(value))}</dd></div>'
            for label, value in metrics
        )
        st.html(f"""
            <div class="gis-verdict gis-verdict-{tone}">
                <div class="gis-verdict-heading">
                    <div>
                        <span class="studio-badge badge-{tone}">{escape(badge)}</span>
                        <h5>{escape(title)}</h5>
                        <p>{escape(summary)}</p>
                    </div>
                    <div class="gis-verdict-score">
                        <strong>{assessment['probability']:.1f}%</strong>
                        <span>CatBoost suitability score</span>
                    </div>
                </div>
                <dl class="gis-verdict-metrics">{metric_html}</dl>
            </div>
        """)

        if not suitable:
            render_countermeasures(assessment)

        exp_df = assessment.get("exp_df")
        if exp_df is not None:
            with st.expander("Model contributions (TreeSHAP)", icon=":material/bar_chart:"):
                st.caption("Log-odds contribution of each site condition to the CatBoost score.")
                st.bar_chart(exp_df.set_index("Feature")["SHAP Impact"])
                st.dataframe(
                    exp_df,
                    column_config={
                        "Feature": "Site parameter",
                        "Input Value": "Configured value",
                        "SHAP Impact": st.column_config.NumberColumn("Log-odds impact", format="%.4f"),
                        "Effect": "Direction",
                    },
                    hide_index=True,
                    width="stretch",
                )

        with st.expander("Preview geotechnical report", icon=":material/description:"):
            st.markdown(assessment["report_md"])

        shap_plot_file = os.path.join(config.DATA_DIR, "shap_summary.png")
        if os.path.exists(shap_plot_file):
            with st.expander("Global model feature importance", icon=":material/insights:"):
                st.image(shap_plot_file, caption="Global mean |SHAP| across the training observations.")


def render_countermeasures(assessment: dict) -> None:
    with st.expander("Recommended countermeasures", icon=":material/handyman:", expanded=True):
        remed = get_two_tier_remediations(assessment)
        st.markdown(f"**{remed['primary_title']}**")
        st.caption(remed["primary_mech"])
        for protocol, icon in remed["protocols"]:
            st.markdown(f"{icon} {protocol}")

        st.markdown("##### Field stewardship")
        stewardship_columns = st.columns(2, gap="small")
        for index, item in enumerate(remed["stewardship"]):
            with stewardship_columns[index % 2], st.container(border=True):
                st.markdown(f"**{item['title']}**")
                st.markdown(item["body"])


def render_boundary_checks() -> None:
    with st.expander("Biomechanical limits", icon=":material/fact_check:"):
        st.markdown(
            "Test site conditions against the **10 scientifically documented critical failure boundaries** "
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
                f"**Deep Slip Surface Failure Risk (Slip Depth {test_slip_depth:.1f} m > 2.0 m):** "
                "Shear plane exceeds the effective 1.5–2.0 m root anchorage curtain. Structural nails or micropiles required."
            )
        if test_bulk_density > 1.75:
            boundary_warnings.append(
                f"**Mechanical Root Refusal Risk (Bulk Density {test_bulk_density:.2f} g/cm³ > 1.75 g/cm³):** "
                "Compaction exceeds penetration threshold. Subsoil ripping required."
            )
        if test_salinity > 16.0:
            boundary_warnings.append(
                f"**Fatal Osmotic Collapse Risk (EC {test_salinity:.1f} dS/m > 16.0 dS/m):** "
                "Exceeds physiological salinity limit. Soil gypsum leaching required."
            )
        if test_shade > 60:
            boundary_warnings.append(
                f"**Photosynthetic Growth Arrest Risk (Canopy Shading {test_shade}% > 60%):** "
                "Obligate C4 heliophyte requires ≥40% direct sunlight."
            )
        if test_submergence > 21:
            boundary_warnings.append(
                f"**Anoxic Crown Rot Risk (Crown Inundation {test_submergence} days > 21 days):** "
                "Exceeds physiological crown submergence limit."
            )
        if test_elevation > 2200:
            boundary_warnings.append(
                f"**Cold Vegetative Arrest Risk (Elevation {test_elevation} m > 2,200 m a.s.l.):** "
                "Low temperatures inhibit root elongation. Montane companion grasses advised."
            )

        if boundary_warnings:
            for w in boundary_warnings:
                st.warning(w, icon=":material/warning:")
        else:
            st.success("**All Biophysical Parameters Within Safe Vetiver Operating Envelope** (Slip ≤ 2.0 m, BD ≤ 1.75 g/cm³, EC ≤ 16 dS/m, Shade ≤ 60%, Submergence ≤ 21 d, Elevation ≤ 2,200 m).", icon=":material/verified:")



def render_engineering_tools(assessment: dict) -> None:
    with st.expander("Engineering tools", icon=":material/architecture:"):
        with st.expander("Slope geometry", icon=":material/straighten:"):
            cs_col1, cs_col2, cs_col3, cs_col4 = st.columns(4)
            with cs_col1:
                ui_slope_h = st.slider("Slope Height (m)", min_value=2.0, max_value=20.0, value=6.0, step=0.5, key="ui_cs_height")
            with cs_col2:
                ui_slope_angle = st.slider("Slope Angle β (°)", min_value=20.0, max_value=60.0, value=35.0, step=1.0, help="Slope gradient (35° ≈ 1:1.4 road cut)", key="ui_cs_angle")
            with cs_col3:
                ui_wt_ratio = st.slider("Groundwater Table", min_value=0.0, max_value=0.8, value=0.25, step=0.05, help="Porewater table height ratio relative to slope height", key="ui_cs_wt")
            with cs_col4:
                ui_slip_z = st.slider("Target Slip Depth (m)", min_value=0.4, max_value=3.5, value=1.0, step=0.1, help="Depth of critical shear failure plane", key="ui_cs_z")

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

        # Render 2D Cross Section Figure & Calculations
        cs_fig, cs_stab = generate_slope_cross_section_fig(
            height_m=ui_slope_h,
            slope_angle_deg=ui_slope_angle,
            soil_class=cur_soil,
            water_table_ratio=ui_wt_ratio,
            root_depth_m=cur_root_depth,
            root_cohesion_kpa=cur_cr,
            design_slip_depth_m=ui_slip_z,
        )

        views = {
            "cross_section": ":material/terrain: Cross-section",
            "materials": ":material/list_alt: Bill of quantities",
            "growth": ":material/timeline: Growth timeline",
        }
        design_view = st.segmented_control(
            "Engineering view", list(views), default="cross_section", required=True,
            format_func=views.__getitem__, key="t1_engineering_view", label_visibility="collapsed",
        )

        if design_view == "cross_section":
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

            # Dynamic Sensitivity Curves (Collapsible)
            with st.expander("Sensitivity curves", icon=":material/show_chart:"):
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

                sens_var = "slope_angle" if "Angle" in sens_mode else "water_table"
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

        elif design_view == "materials":
            st.caption("Contractor-grade planting geometry and material budget estimate:")
            boq_c1, boq_c2, boq_c3 = st.columns(3)
            with boq_c1:
                ui_boq_w = st.number_input("Slope Crest Width along road (m)", min_value=5.0, max_value=500.0, value=30.0, step=5.0, key="ui_boq_w")
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

        else:  # Temporal Maturation Timeline
            st.caption("12-month root penetration, added cohesion growth, and slope safety factor progression:")
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


def render_tab_assessment(cb_model=None, hotspot_mtime: float = 0.0) -> None:
    with st.container(key="vetiver_assessment", gap="small"):
        with st.container(key="assessment_header", gap="xsmall"):
            st.subheader("Site Assessment", anchor=False)
            st.caption("Set the project location and review site conditions before running the assessment.")

        with st.container(key="assessment_setup", gap=None):
            location_col, conditions_col = st.columns([1, 1.55], gap="small", border=True)

        with location_col:
            render_project_location(hotspot_mtime)
        with conditions_col:
            st.subheader(":material/tune: Site conditions", anchor=False)
            render_assessment_inputs("t1", cb_model, show_site=False)

        assessment = st.session_state.current_assessment
        if assessment is not None:
            render_suitability_result(assessment)
        render_boundary_checks()
        if assessment is not None:
            render_engineering_tools(assessment)
