"""
Generate rich, authoritative markdown knowledge base documents from:
1. data/vetiver_failure_data.csv (60 cases with Failure Mode, Cause Details, Mitigations, Citations)
2. data/vetiver_boundary_parameters.csv (10 AI Boundary Limits & Feature Rules)
"""
import os
import pandas as pd

KB_DIR = os.path.join(os.path.dirname(__file__), "knowledge_base")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(KB_DIR, exist_ok=True)

df_fail = pd.read_csv(os.path.join(DATA_DIR, "vetiver_failure_data.csv"))
df_bounds = pd.read_csv(os.path.join(DATA_DIR, "vetiver_boundary_parameters.csv"))

# 1. Failure & Mitigation Field Manual
guide_lines = [
    "# VETIVER GRASS (CHRYSOPOGON ZIZANIOIDES) BIOENGINEERING FAILURE REGISTER & MITIGATION MANUAL",
    "",
    "## Overview & Engineering Intent",
    "While Vetiver grass (*Chrysopogon zizanioides*) is exceptionally resilient and widely deployed for slope stabilization, "
    "erosion control, and tailings remediation, bioengineering failures occur when physical, geotechnical, hydrological, or physiological "
    "exclusion boundaries are breached. This document compiles 60 scientifically documented failure cases across the globe and East Africa, "
    "pairing each failure mode with published peer-reviewed engineer and scientist countermeasures.",
    "",
    "---",
    "",
    "## Biomechanical & Geotechnical Limit States Summary",
    "1. **Slip Plane Depth Boundary:** Vetiver root reinforcement creates a dense tensile anchor envelope primarily down to 1.5–2.0 m (rarely 3.0 m in deep porous soils). Deep-seated rotational or translational slip surfaces deeper than 2.0 m exceed the biological shear envelope and require structural precursors (soil nails, micropiles, rock anchors).",
    "2. **Hydraulic Shear Velocity Boundary:** Juvenile hedgerows (<6 months) fail by hydrodynamic uprooting at flow velocities >2.0–2.5 m/s. Once mature (root crown tillering >6 months), hedgerows resist tractive shear stresses up to 300–400 N/m² and flows up to 3.5–5.0 m/s.",
    "3. **Chemical / Salinity Boundary:** The physiological threshold for Vetiver survival is electrical conductivity EC ≤ 16–20 dS/m and SAR ≤ 25. Hypersaline substrates require gypsum pre-treatment (5–12 t/ha) and gradual saline nursery hardening.",
    "4. **Light & Heliophyte Boundary:** Vetiver is an obligate C4 heliophyte requiring at least 40–50% full sunlight. Overstory canopy shading >60% causes etiolation, tillering cessation, and eventual biomass collapse.",
    "5. **Anoxic Submergence Limit:** Foliage leaf crowns must maintain contact with atmospheric oxygen. Continuous total submergence of crowns exceeding 21–30 days induces anoxic crown rotting.",
    "6. **Thermal & Elevation Boundary:** Tropical Vetiver cessation of active growth occurs when sustained mean temperatures drop <10–12°C or elevations exceed 2,200 m in East African highlands.",
    "",
    "---",
    "",
    "## Detailed Case Register (60 Empirical Failure Cases & Countermeasures)",
    "",
]

for idx, r in df_fail.iterrows():
    cid = idx + 1
    guide_lines.append(f"### Case {cid}: {r['Location(country/region)']} – {r['Failure Mode']}")
    guide_lines.append(f"- **Application:** {r['Application']}")
    guide_lines.append(f"- **Substrate & Soil Type:** {r['Soil type/structure']} (Density: {r['Soil density']}, Salinity: {r['salinity']})")
    guide_lines.append(f"- **Climate & Watering:** {r['climate']} | {r['watering regime']}")
    guide_lines.append(f"- **Duration & Root Length:** {r['Duration since planting ']} | Root depth: {r['Root length(cm)']}")
    guide_lines.append(f"- **Citation / Reference:** {r['Reference']}")
    guide_lines.append(f"- **Failure Mechanism Details:** {r['Failure Cause Details']}")
    guide_lines.append(f"- **Published Scientist Mitigation:** {r['Mitigation']}")
    guide_lines.append(f"- **Geotechnical Exclusion Boundary:** `{r['Exclusion Boundary']}`")
    guide_lines.append("")

guide_path = os.path.join(KB_DIR, "vetiver_failure_and_mitigation_guide.md")
with open(guide_path, "w", encoding="utf-8") as f:
    f.write("\n".join(guide_lines))
print(f"Generated {guide_path} ({len(df_fail)} cases)")

# 2. Boundary Parameters Document
param_lines = [
    "# VETIVER GRASS BIOMECHANICAL & PHYSIOLOGICAL BOUNDARY CONDITIONS FOR AI BIOENGINEERING",
    "",
    "This reference defines the strict quantitative failure thresholds, primary mechanisms, empirical citations, and rule representations for machine learning and geotechnical decision support.",
    "",
    "| Stress Factor / Regional Domain | Critical Threshold for Failure | Primary Failure Mechanism | Empirical Citations | AI Feature Representation |",
    "| :--- | :--- | :--- | :--- | :--- |",
]

for _, r in df_bounds.iterrows():
    param_lines.append(
        f"| **{r['STRESS FACTOR / REGIONAL DOMAIN']}** | {r['CRITICAL THRESHOLD FOR FAILURE']} | {r['PRIMARY FAILURE MECHANISM']} | {r['EMPIRICAL EAST AFRICAN & GLOBAL CITATION']} | `{r['AI FEATURE REPRESENTATION']}` |"
    )

param_lines.extend([
    "",
    "---",
    "",
    "## Engineering Decision Logic & Prescreening Guidelines",
    "",
    "### 1. Geotechnical Slope Failure (Slip Surface Depth > 2.0 m)",
    "- **Condition:** If rotational or translational shear failure planes are estimated at depths > 2.0 m.",
    "- **Decision:** Do NOT rely on Vetiver as a standalone structural retaining system. Vetiver provides surficial veneer protection (0–2 m). Combine with driven soil nails (L ≥ 4 m) and sub-horizontal PVC weepholes.",
    "",
    "### 2. High Soil Compaction (Bulk Density > 1.75 g/cm³)",
    "- **Condition:** Soil bulk density exceeds 1.75 g/cm³ in over-compacted clays or unweathered saprolites.",
    "- **Decision:** Mechanical ripping or augering along planting furrows to ≥60 cm is mandatory to shatter hardpans before planting slips.",
    "",
    "### 3. Hypersaline / Sodic Environments (EC > 16.0 dS/m or SAR > 25)",
    "- **Condition:** Electrical conductivity > 16 dS/m (e.g. mine tailings, coastal brine pools, trona pans).",
    "- **Decision:** Apply agricultural gypsum (CaSO4·2H2O) at 5–12 t/ha to displace exchangeable sodium, flush with fresh water, and harden polybag slips prior to field transplanting.",
    "",
    "### 4. Continuous Shading (Canopy Cover > 60%)",
    "- **Condition:** Agroforestry or dense tree plantations where photosynthetically active radiation (PAR) drops < 40%.",
    "- **Decision:** Enforce scheduled canopy pruning and orient hedgerows East-West.",
    "",
    "### 5. Continuous Flooding / Leaf Crown Submergence (> 21 Days)",
    "- **Condition:** River floodways or backflow swamps where leaf crowns are drowned beneath water for > 3 weeks.",
    "- **Decision:** Construct elevated earthen levees or plant above mean seasonal flood crest so upper foliage remains aerated.",
    "",
    "### 6. Cold / High Tropical Elevation (> 2,200 m or Mean Temp < 10°C)",
    "- **Condition:** Highland zones with continuous temperatures < 12°C (e.g. Mt. Elgon, Aberdares, Lake Bunyonyi).",
    "- **Decision:** Combine or replace Vetiver with cold-adapted native montane tussock grasses (e.g., *Pennisetum clandestinum*, *Festuca* spp.).",
    "",
    "### 7. Hyper-acidic / High Aluminum Soils (pH < 3.5 or Al Saturation > 70%)",
    "- **Condition:** Acid sulfate coastal soils or pyritic mining tailings.",
    "- **Decision:** Incorporate agricultural lime (CaCO3) at 5–10 t/ha to raise pH > 4.5 and add rock phosphate (Minjingu PR at 200 kg/ha) or diammonium phosphate (DAP) to precipitate toxic free Al3+.",
    "",
    "### 8. Collapsible Pumice / Hydro-piping Risk",
    "- **Condition:** Porous volcanic pumice (e.g. Menengai / Rift Valley) with bulk density < 1.15 g/cm³ subject to flash wetting.",
    "- **Decision:** Slopes require dynamic compaction or cohesive clay veneers prior to planting to prevent subterranean piping tunnels from bypassing roots.",
    "",
    "### 9. Flash Flood Discharge Velocity (> 2.5 m/s in Juvenile Slips < 6 Months)",
    "- **Condition:** Ephemeral gulley floors or active storm drain ditches.",
    "- **Decision:** Secure channel floors with upstream dry-stone check dams or pinned biodegradable jute/coir geotextiles until crowns achieve 6 months basal maturity.",
    "",
    "### 10. Shallow Petroferric Hardpans / Bedrock (< 0.2 m Depth)",
    "- **Condition:** Indurated laterite petroferric pans or solid shallow bedrock mantles < 20 cm.",
    "- **Decision:** Mechanical chiseling/pocket-augering to ≥50 cm is mandatory; otherwise divert hedgerows to colluvial soil pockets.",
])

param_path = os.path.join(KB_DIR, "vetiver_engineering_boundary_thresholds.md")
with open(param_path, "w", encoding="utf-8") as f:
    f.write("\n".join(param_lines))
print(f"Generated {param_path}")
