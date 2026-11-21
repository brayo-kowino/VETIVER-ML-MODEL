import math

HAZARD_DISPLAY_MAP = {
    "Shallow Slope / Embankment Slip (0.3–1.5 m)": "Shallow Slope / Embankment Slip (0.3–1.5 m) — Road Cuts, Railway Batters, Terraces",
    "Surficial Sheet & Rill Erosion (< 0.3 m)": "Surficial Sheet & Rill Erosion (< 0.3 m) — Agricultural Contours, Gully Flanks, Topsoil Wash",
    "Riverbank / Hydrodynamic Toe Scour": "Riverbank / Hydrodynamic Toe Scour — River Channels, Canal Lining, Drainage Dikes",
    "Mine Tailings & Spoil Containment": "Mine Tailings & Spoil Containment — Tailings Dams, Landfills, Industrial Spoil",
    "Deep Rotational Slide (> 2.0 m)": "Deep Rotational Slide (> 2.0 m) — Deep Mass Landslides (Structural Nails Required)",
}


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
