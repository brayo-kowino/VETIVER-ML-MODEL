import os
import json
import html
from datetime import datetime

import streamlit as st
import pandas as pd

import config
from services.biomechanics import (
    HAZARD_DISPLAY_MAP as hazard_display_map,
    calculate_biomechanical_metrics,
)
from services.remediation import (
    get_two_tier_remediations,
    get_failure_banner_details,
)
from services.report_service import generate_suitability_report
from services.assessment_service import run_assessment
from services.model_service import load_regional_hotspots
from geospatial_resolver import resolve_coordinate_geotechnics
from geotechnical_engine import (
    calculate_slope_stability,
    generate_slope_cross_section_fig,
    generate_fos_sensitivity_fig,
    calculate_field_boq,
    generate_temporal_maturation_fig,
)
from corridor_engine import export_corridor_geojson


def render_tab_assessment(cb_model=None, hotspot_mtime: float = 0.0) -> None:
    st.subheader("Site Suitability Assessment", anchor=False)
   
    # ── STEP 1: SELECT OR INGEST LOCATION ──────────────────────
    with st.container(border=True):
        st.markdown("##### :material/place: Step 1: Select or Ingest Project Site")
        
        ingest_mode = st.radio(
            "Location Ingestion Mode",
            [
                "📍 GPS Coordinates (Auto-Detect)",
                "Regional Benchmark Sites",
                "Quick Corridor Shortcuts",
            ],
            horizontal=True,
            label_visibility="collapsed",
            key="t1_ingest_mode_radio"
        )

        if ingest_mode == "📍 GPS Coordinates (Auto-Detect)":
            st.caption("Enter coordinates to automatically fetch soil class, rainfall regime, elevation, and salinity baselines:")
            t1_gc1, t1_gc2, t1_gc3 = st.columns([1.5, 1.5, 1.8])
            with t1_gc1:
                t1_in_lat = st.number_input(
                    "Latitude (°N)",
                    min_value=-90.0,
                    max_value=90.0,
                    value=float(st.session_state.coord_lat),
                    format="%.5f",
                    step=0.01,
                )
            with t1_gc2:
                t1_in_lon = st.number_input(
                    "Longitude (°E)",
                    min_value=-180.0,
                    max_value=180.0,
                    value=float(st.session_state.coord_lon),
                    format="%.5f",
                    step=0.01,
                )
            with t1_gc3:
                st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
                t1_resolve_btn = st.button(
                    "Auto-Detect Baseline Parameters",
                    type="primary",
                    width="stretch",
                    icon=":material/download:",
                    key="btn_t1_resolve_custom_gps"
                )

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
                    f"**Detected: {c_site['county']}, {c_site['country']}** (Elev: `{c_site['elevation_m']} m a.s.l.`) via **{c_site['data_sources']}**.\n\n"
                    f"Baseline parameters loaded below. You can customize moisture regime or engineering hazard before running analysis.",
                    icon=":material/check_circle:"
                )

        elif ingest_mode == "Regional Benchmark Sites":
            all_hotspots = load_regional_hotspots(hotspot_mtime)
            if all_hotspots:
                st.caption("Select a pre-calculated East Africa civil corridor or benchmark location:")
                hs_quick_c1, hs_quick_c2 = st.columns([3, 1])
                with hs_quick_c1:
                    hs_quick_labels = ["-- Choose a Benchmark Location --"] + [
                        f"{h['county_region']}: {h['name']}" for h in all_hotspots
                    ]
                    sel_quick = st.selectbox(
                        "Benchmark Location",
                        hs_quick_labels,
                        index=0,
                        label_visibility="collapsed",
                    )
                with hs_quick_c2:
                    if st.button("Apply Location", icon=":material/download:", width="stretch", key="btn_apply_hotspot_preset"):
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
                            st.session_state.current_assessment = run_assessment(auto_inputs, cb_model=cb_model)
                            st.rerun()

        else:  # Quick Corridor Shortcuts
            st.caption("Click any major infrastructure corridor for 1-click parameter configuration:")
            t1_p1, t1_p2, t1_p3, t1_p4, t1_p5, t1_p6 = st.columns(6)
            shortcuts = [
                (t1_p1, "Nairobi", -1.2921, 36.8219),
                (t1_p2, "Samburu", 1.0000, 36.8219),
                (t1_p3, "Eldoret", 0.5143, 35.2698),
                (t1_p4, "Naivasha", -0.7172, 36.4312),
                (t1_p5, "Embu", -0.5342, 37.4544),
                (t1_p6, "Garissa", -0.4532, 39.6461),
            ]
            for btn_col, name, lat, lon in shortcuts:
                if btn_col.button(name, width="stretch", key=f"t1_btn_{name.lower()}"):
                    st.session_state.coord_lat = lat
                    st.session_state.coord_lon = lon
                    c_site = resolve_coordinate_geotechnics(lat, lon)
                    st.session_state.custom_coord_site = c_site
                    st.session_state.preset_soil = c_site["soil_mechanical_class"]
                    st.session_state.preset_climate = c_site["climate_rainfall_zone"]
                    st.session_state.preset_moisture = c_site["establishment_moisture"]
                    st.session_state.preset_hydro = c_site["hydrological_stress"]
                    st.session_state.preset_salinity = c_site["salinity_tier"]
                    st.session_state.preset_hazard = c_site["default_hazard"]
                    st.session_state.pan_to_custom = True
                    st.rerun()

    # ── STEP 2: REVIEW & REFINE SITE PARAMETERS ────────────────
    with st.container(border=True):
        st.markdown("##### :material/tune: Step 2: Review & Customize Site Parameters")
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
                help="Governs soil shear strength, bulk density, and vertical root elongation rate.",
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
                help="Annual rainfall and temperature regime.",
            )

            hazard_options = [
                "Shallow Slope / Embankment Slip (0.3–1.5 m)",
                "Surficial Sheet & Rill Erosion (< 0.3 m)",
                "Deep Rotational Slide (> 2.0 m)",
                "Riverbank / Hydrodynamic Toe Scour",
                "Mine Tailings & Spoil Containment",
            ]
            hazard_idx = hazard_options.index(st.session_state.preset_hazard) if st.session_state.preset_hazard in hazard_options else 0
            hazard = st.selectbox(
                "Geotechnical Engineering Objective & Slip Depth",
                hazard_options,
                index=hazard_idx,
                format_func=lambda x: hazard_display_map.get(x, x),
                help="Target bio-engineering objective and design failure depth.",
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
                help="Initial watering during transplant establishment before deep taproots mature.",
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
                help="Fluvial tractive shear or continuous crown submergence hazards.",
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
                help="Soil electrical conductivity (ECe). Extreme salinity (>16 dS/m) induces osmotic desiccation.",
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
                st.session_state.current_assessment = run_assessment(current_inputs, cb_model=cb_model)
                st.rerun()

        with btn_c2:
            if st.button("Reset", icon=":material/refresh:", width="stretch"):
                st.session_state.current_assessment = None
                st.rerun()

    # ── BIOMECHANICAL BOUNDARY PRESCREENER (EXPANDABLE) ────────
    with st.expander("🔍 Scientific Failure Boundaries & Biomechanical Limits", expanded=False):
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
                f":material/warning: **Deep Slip Surface Failure Risk (Slip Depth {test_slip_depth:.1f} m > 2.0 m):** "
                "Shear plane exceeds the effective 1.5–2.0 m root anchorage curtain. Structural nails or micropiles required."
            )
        if test_bulk_density > 1.75:
            boundary_warnings.append(
                f":material/warning: **Mechanical Root Refusal Risk (Bulk Density {test_bulk_density:.2f} g/cm³ > 1.75 g/cm³):** "
                "Compaction exceeds penetration threshold. Subsoil ripping required."
            )
        if test_salinity > 16.0:
            boundary_warnings.append(
                f":material/warning: **Fatal Osmotic Collapse Risk (EC {test_salinity:.1f} dS/m > 16.0 dS/m):** "
                "Exceeds physiological salinity limit. Soil gypsum leaching required."
            )
        if test_shade > 60:
            boundary_warnings.append(
                f":material/warning: **Photosynthetic Growth Arrest Risk (Canopy Shading {test_shade}% > 60%):** "
                "Obligate C4 heliophyte requires ≥40% direct sunlight."
            )
        if test_submergence > 21:
            boundary_warnings.append(
                f":material/warning: **Anoxic Crown Rot Risk (Crown Inundation {test_submergence} days > 21 days):** "
                "Exceeds physiological crown submergence limit."
            )
        if test_elevation > 2200:
            boundary_warnings.append(
                f":material/warning: **Cold Vegetative Arrest Risk (Elevation {test_elevation} m > 2,200 m a.s.l.):** "
                "Low temperatures inhibit root elongation. Montane companion grasses advised."
            )

        if boundary_warnings:
            for w in boundary_warnings:
                st.warning(w)
        else:
            st.success("**All Biophysical Parameters Within Safe Vetiver Operating Envelope** (Slip ≤ 2.0 m, BD ≤ 1.75 g/cm³, EC ≤ 16 dS/m, Shade ≤ 60%, Submergence ≤ 21 d, Elevation ≤ 2,200 m).", icon=":material/verified:")

    # ── STEP 3: RENDER ASSESSMENT RESULTS ──────────────────────
    if st.session_state.current_assessment is not None:
        assessment = st.session_state.current_assessment
        pred = assessment["prediction"]
        prob = assessment["probability"]
        exp_df = assessment["exp_df"]
        report_md = assessment["report_md"]

        st.divider()
        st.markdown("##### :material/analytics: Step 3: Slope Stability Verdict & Biological Reinforcement")

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
                            <span style="background: #16A34A; color: white; font-size: 0.75rem; font-weight: 700; padding: 4px 12px; border-radius: 100px; text-transform: uppercase; letter-spacing: 0.05em;">Positive Geotechnical Alignment</span>
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

        # 1-Click Direct Report Download
        st.download_button(
            label="Download Full Geotechnical Suitability Report (.md)",
            data=report_md,
            file_name=f"vetiver_site_assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
            icon=":material/download:",
            width="stretch",
            key="t1_btn_download_report_top",
        )

        # 2. Scientist Mitigations (if unsuitable)
        if pred == 0:
            with st.expander("🛠️ Scientist-Recommended Countermeasures & Remediations", expanded=True):
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

        # 3. Decision Explainability (SHAP) - Cleanly Collapsed
        with st.expander("📊 AI Decision Attribution (Why did the model reach this verdict?)", expanded=False):
            st.caption("Mathematical contribution (log-odds impact) of each site condition toward the verdict:")
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

        # ── STEP 4: FIELD ENGINEERING DESIGN & BOQ SUITE ───────────
        st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)
        st.markdown("##### :material/architecture: Step 4: Field Engineering & Material Estimator")
        
        # Shared Slope Geometry Control Bar
        with st.container(border=True):
            st.caption("Configure physical slope dimensions for 2D stability simulation, BOQ budgeting, and growth timelines:")
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

        design_view = st.radio(
            "Engineering View Selection",
            [
                "📐 2D Slope Stability Cross-Section",
                "📋 Contractor Bill of Quantities (BoQ)",
                "⏳ 12-Month Growth & Maturation Timeline",
            ],
            horizontal=True,
            label_visibility="collapsed",
            key="t1_design_view_radio"
        )

        if design_view == "📐 2D Slope Stability Cross-Section":
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
            st.markdown("<div style='margin-top: 16px;'></div>", unsafe_allow_html=True)
            with st.expander("📈 Dynamic Sensitivity Curves (FoS vs Slope Angle & Groundwater Table)", expanded=False):
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

        elif design_view == "📋 Contractor Bill of Quantities (BoQ)":
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

        # 5. Full Report Preview & SHAP Summary (Collapsible)
        with st.expander("📄 Preview Full Geotechnical Engineering Report", icon=":material/description:"):
            st.markdown(report_md)

        shap_plot_file = os.path.join(config.DATA_DIR, "shap_summary.png")
        if os.path.exists(shap_plot_file):
            with st.expander("📊 View Global Model Feature Importance (SHAP Summary)", icon=":material/bar_chart:"):
                st.image(shap_plot_file, caption="Global Mean |SHAP| values computed across all empirical and failure boundaries.")
