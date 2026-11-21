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
    # 3. Severe Drought / Zero Irrigation
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
    # 4. Ground Frost / High Altitude Cold
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
    # 5. Statistical Unfavorable
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
            "icon": "",
            "title": "Strict Solar Exposure (Obligate Heliophyte)",
            "body": "Vetiver is an obligate C4 sun grass requiring **>= 6 hours of direct sunlight daily**. Never plant under dense tree canopies, under bridges/overpasses, or in permanent structural shade. Light deficits cause severe etiolation, chlorosis, and 90–100% mortality within 4 months.",
        },
        {
            "icon": "",
            "title": "12-Month Livestock Grazing Exclusion",
            "body": "Tender juvenile slips (0–12 months) are palatable to cattle, goats, and wildlife. Enforce **mandatory 12-month perimeter live fencing** (e.g. *Euphorbia tirucalli*, thorny *Acacia*, or wire mesh). Once mature, coarse siliceous leaves become unpalatable.",
        },
        {
            "icon": "",
            "title": "Termite & Subterranean Pest Defense",
            "body": "In semi-arid savannahs or dry cut-slopes, subterranean termites can sever tender young root shoots. Dip bare-root slips in an organic neem oil solution or entomopathogenic biopesticide (*Metarhizium anisopliae*) prior to planting.",
        },
        {
            "icon": "",
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
