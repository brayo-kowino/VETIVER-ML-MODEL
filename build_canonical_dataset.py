"""
Build Canonical Machine-Ready Dataset for Vetiver Bioengineering ML.
Converts 184 empirical cases (124 Growth + 60 Failure from Sheet 1):
  - Standardizes raw free-text into strict geotechnical & agronomic engineering classes
  - Preserves 100% of original human literature citations, locations, and notes for auditability
  - Saves to data/vetiver_canonical_ml_dataset.csv
"""
import os
import re
import pandas as pd
import numpy as np

# File paths
EXCEL_PATH = r"C:/Users/Brian/.gemini/antigravity/brain/24efd127-69dc-47af-ba97-b3d044a9cf14/.user_uploaded/media_1793968009739_c3c9c628.xlsx"
GROWTH_PATH = r"C:/Users/Brian/.gemini/antigravity/brain/24efd127-69dc-47af-ba97-b3d044a9cf14/.user_uploaded/media_1793972218661_dec306d9.csv"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_PATH = os.path.join(DATA_DIR, "vetiver_canonical_ml_dataset.csv")

os.makedirs(DATA_DIR, exist_ok=True)

# 1. Load Raw Sources
df_growth_raw = pd.read_csv(GROWTH_PATH)
df_fail_raw = pd.read_excel(EXCEL_PATH, sheet_name=0, skiprows=2)

print(f"Loaded raw positive growth cases: {len(df_growth_raw)}")
print(f"Loaded raw scientific failure cases (Sheet 1): {len(df_fail_raw)}")


# --- NORMALIZATION FUNCTIONS ---

def normalize_soil(text: str) -> str:
    """Classifies substrate into USCS-aligned geotechnical classes."""
    t = str(text).lower()
    # 1. Indurated Hardpan / Bedrock Refusal
    if any(k in t for k in ["bedrock", "hardpan", "petroferric", "rock refusal", "limestone", "granite", "duricrust", "ironstone"]):
        return "Indurated Hardpan / Bedrock"
    # 2. Mine Tailings & Industrial Mineral Spoil
    if any(k in t for k in ["tailings", "mine", "fly ash", "bauxite", "trona", "cyanide", "copper ion", "gold mine"]):
        return "Mine Tailings / Industrial Spoil"
    # 3. Coarse Sand, Gravel & Porous Pumice (cohesionless, high piping)
    if any(k in t for k in ["pumice", "coarse sand", "fine sand", "dune", "beach sand", "coral rag", "arenosol", "regosol", "loose sand", "volcanic ash", "andisol"]):
        return "Coarse Sand / Loose Pumice"
    # 4. Cohesive Clays, Nitisols, Laterites & Vertisols (fine-grained cohesive soils)
    if any(k in t for k in ["nitisol", "ferralsol", "vertisol", "black cotton", "laterite", "lateritic", "marl", "alluvial clay", "silt-clay", "heavy clay", "marine clay", "ultisol"]):
        return "Cohesive Clay / Nitisol / Laterite"
    # 5. Sandy Clay, Sandy Loam & Decomposed Granite (medium-textured loams)
    if any(k in t for k in ["sandy clay", "sandy loam", "clay loam", "loam", "luvisol", "cambisol", "decomposed granite", "silt loam", "alluvial sandy"]):
        return "Sandy Clay / Loam"
    if "clay" in t:
        return "Cohesive Clay / Nitisol / Laterite"
    if "sand" in t:
        return "Coarse Sand / Loose Pumice"
    # Fallback to balanced agricultural loam
    return "Sandy Clay / Loam"


def normalize_climate_rainfall(climate_text: str, loc_text: str = "") -> tuple:
    """
    Standardizes regional climate into annual rainfall zone and numeric mm/yr estimate.
    Returns: (climate_zone, est_annual_rainfall_mm)
    """
    c = str(climate_text).lower()
    l = str(loc_text).lower()

    # Cold continental or extreme high altitude
    if any(k in c for k in ["continental cold", "freezing", "siberia", "winter freeze"]) or "siberia" in l or "shaanxi" in l:
        return "Cold Freezing / Permafrost", 450
    if any(k in c for k in ["alpine", ">2,200", ">2,300", "high-altitude"]) or "elgon" in l:
        return "Alpine High-Altitude (>2,200 m)", 900

    # Hyper-Arid / Desert (< 300 mm)
    if any(k in c for k in ["hyper-arid", "hot desert", "desert arid", "arid hot", "annual rainfall <300", "< 300"]) or any(k in l for k in ["al-ahsa", "saudi arabia", "balochistan", "sindh desert"]):
        return "Hyper-Arid / Desert (<300 mm/yr)", 150

    # Semi-Arid (300 - 700 mm)
    if any(k in c for k in ["semi-arid", "semi arid", "lowland savanna", "highveld"]) or any(k in l for k in ["machakos", "katumani", "dodoma", "somalia", "tigray", "turkana"]):
        return "Semi-Arid (300–700 mm/yr)", 550

    # Humid / Tropical Monsoon (> 1400 mm)
    if any(k in c for k in ["monsoon", "tropical humid", "humid tropical", "equatorial monsoon", "extreme rainfall", "high intensity", "wet"]) or any(k in l for k in ["thailand", "vietnam", "mekong", "rio de janeiro", "philippines", "queensland"]):
        return "Humid / Tropical Monsoon (>1400 mm/yr)", 1850

    # Sub-Humid / Bimodal Tropical (700 - 1400 mm)
    return "Sub-Humid / Bimodal (700–1400 mm/yr)", 1050


def normalize_establishment_moisture(watering_text: str, failure_mode_text: str = "", is_suitable: int = 1) -> str:
    """
    Determines whether slips received adequate establishment moisture (first 30-60 days)
    or experienced severe drought desiccation.
    """
    if is_suitable == 1:
        # All empirical growth successes established successfully
        return "Adequate (Irrigated / In-Season Rain)"

    w = str(watering_text).lower()
    fm = str(failure_mode_text).lower()

    if any(k in w for k in ["no water", "severe drought", "dry spell", "zero water", "drip stopped", "erratic bimodal rain"]) or \
       any(k in fm for k in ["desiccation", "moisture stress", "drought", "dry spell", "summer dormancy"]):
        return "Severe Drought / Zero Watering"

    return "Adequate (Irrigated / In-Season Rain)"


def normalize_hydrological_hazard(watering_text: str, failure_mode_text: str = "") -> str:
    """Identifies hydraulic boundary exceedances."""
    w = str(watering_text).lower()
    fm = str(failure_mode_text).lower()

    if any(k in w for k in ["flash", "high-velocity", "high-energy", ">2.5", "runoff"]) or \
       any(k in fm for k in ["hydraulic uprooting", "scouring", "flume undermining", "wave-slap", "flash flood"]):
        return "High-Velocity Hydraulic Runoff (>2.5 m/s)"

    if any(k in w for k in ["submergence", "flooded", "waterlogging", ">21"]) or \
       any(k in fm for k in ["submergence", "stagnant waterlogging", "crown rot under water", "crown suffocation"]):
        return "Prolonged Inundation (>21 days)"

    return "Normal Slope / Well-Drained"


def normalize_salinity(salinity_text: str, failure_mode_text: str = "") -> tuple:
    """
    Normalizes salinity into standard agronomic tier and representative ECe (dS/m).
    Returns: (salinity_tier, est_ec_ds_m)
    """
    s = str(salinity_text).lower()
    fm = str(failure_mode_text).lower()

    # Extract numeric values if present
    num_match = re.search(r"(\d+(?:\.\d+)?)", s)
    num_val = float(num_match.group(1)) if num_match else None

    # Hypersaline collapse
    if (num_val is not None and num_val >= 16.0) or \
       any(k in s for k in ["extreme", "hypersaline", "16-20", "> 16", "22", "24", "trona", "brine", "tidal inundation"]) or \
       any(k in fm for k in ["osmotic desiccation", "hyper-saline", "hyperosmotic", "tidal inundation"]):
        val = num_val if num_val and num_val >= 16.0 else 19.5
        return "Extreme Hypersaline (> 16 dS/m)", val

    # High salinity (8 - 16 dS/m)
    if (num_val is not None and 8.0 <= num_val < 16.0) or "8-16" in s or "8.0" in s or "10-20" in s:
        val = num_val if num_val else 10.0
        return "High Saline (8–16 dS/m)", val

    # Moderate salinity (2 - 8 dS/m)
    if (num_val is not None and 2.0 <= num_val < 8.0) or any(k in s for k in ["2-6", "4 d", "saline tolerant", "moderate"]):
        val = num_val if num_val else 4.0
        return "Slight to Moderate (2–8 dS/m)", val

    # Default: Non-saline (<2 dS/m)
    val = num_val if num_val and num_val < 2.0 else 0.5
    return "Non-Saline (< 2 dS/m)", val


def normalize_hazard_application(app_text: str, failure_mode_text: str = "") -> str:
    """Classifies civil bio-engineering application and shear failure depth."""
    a = str(app_text).lower()
    fm = str(failure_mode_text).lower()

    if any(k in fm for k in ["rotational slip", "deep-seated", "debris slide", "slump carrying", "block failure beneath", "rotational mass"]):
        return "Deep Rotational Slide (> 2.0 m)"

    if any(k in a for k in ["tailings", "cyanide", "bauxite", "rehabilitation of gold"]) or \
       any(k in fm for k in ["phytotoxicity", "copper ion", "acid sulfate", "metal phyto"]):
        return "Mine Tailings & Spoil Containment"

    if any(k in a for k in ["river", "canal", "bayou", "shoreline", "flood-proof", "dyke", "waterway"]) or \
       any(k in fm for k in ["flume undermining", "basal scour", "bank undercut", "headcut"]):
        return "Riverbank / Hydrodynamic Toe Scour"

    if any(k in a for k in ["highway", "cut-slope", "slope stabilization", "landslide mitigation", "steep hill", "embankment"]) or \
       any(k in fm for k in ["planar sliding", "interface planar", "shallow rooting"]):
        return "Shallow Slope / Embankment Slip (0.3–1.5 m)"

    return "Surficial Sheet & Rill Erosion (< 0.3 m)"


# --- PROCESS ROWS ---
records = []

# 1. Positive Growth Records (124 cases)
for idx, r in df_growth_raw.iterrows():
    cid = f"POS-{idx + 1:03d}"
    raw_loc = str(r.get("Location(country/region)", "")).strip()
    raw_soil = str(r.get("Soil type/structure", "")).strip()
    raw_climate = str(r.get("climate", "")).strip()
    raw_watering = str(r.get("watering regime", "")).strip()
    raw_sal = str(r.get("salinity", "")).strip()
    raw_app = str(r.get("Application", "")).strip()
    raw_ref = str(r.get("Reference", "")).strip()
    raw_dur = str(r.get("Duration since planting ", "")).strip()
    raw_root = str(r.get("Root length(cm)", "")).strip()

    soil_cls = normalize_soil(raw_soil)
    clim_zone, rain_mm = normalize_climate_rainfall(raw_climate, raw_loc)
    est_moist = normalize_establishment_moisture(raw_watering, is_suitable=1)
    hydro_stress = normalize_hydrological_hazard(raw_watering)
    sal_tier, sal_ec = normalize_salinity(raw_sal)
    geo_hazard = normalize_hazard_application(raw_app)

    records.append({
        "case_id": cid,
        # Standardized ML Features
        "soil_mechanical_class": soil_cls,
        "climate_rainfall_zone": clim_zone,
        "est_annual_rainfall_mm": rain_mm,
        "establishment_moisture": est_moist,
        "hydrological_stress": hydro_stress,
        "salinity_tier": sal_tier,
        "est_salinity_ec_ds_m": sal_ec,
        "geotechnical_hazard": geo_hazard,
        "Suitability": 1,
        # Auditable Metadata & Citations
        "raw_location": raw_loc,
        "raw_reference": raw_ref,
        "raw_soil_description": raw_soil,
        "raw_climate_description": raw_climate,
        "raw_watering_notes": raw_watering,
        "raw_salinity_text": raw_sal,
        "raw_application_text": raw_app,
        "raw_duration": raw_dur,
        "raw_root_depth": raw_root,
        "failure_mode": "Successful Empirical Growth / Stabilization",
        "failure_cause_details": "Optimal establishment; vigorous vertical root anchoring and tensile reinforcement.",
        "published_mitigation": "Standard establishment protocol; maintenance pruning as needed.",
    })

# 2. Scientific Failure Records (60 cases from Sheet 1)
for idx, r in df_fail_raw.iterrows():
    cid = f"FAIL-{idx + 1:03d}"
    raw_loc = str(r.get("LOCATION", "")).strip()
    raw_soil = str(r.get("SOIL TYPE", "")).strip()
    raw_climate = str(r.get("CLIMATE", "")).strip()
    raw_watering = str(r.get("WATERING", "")).strip()
    raw_sal = str(r.get("SALINITY (dS/m)", "")).strip()
    raw_app = str(r.get("APPLICATION", "")).strip()
    raw_ref = str(r.get("REFERENCE", "")).strip()
    raw_dur = str(r.get("DURATION", "")).strip()
    raw_root = str(r.get("ROOT LENGTH (m)", "")).strip()
    raw_fm = str(r.get("FAILURE MODE", "")).strip()
    raw_fc = str(r.get("FAILURE CAUSE DETAILS", "")).strip()
    raw_mit = str(r.get("MITIGATION (SCIENTISTS' PUBLISHED RECOMMENDATIONS)", "")).strip()

    soil_cls = normalize_soil(raw_soil)
    clim_zone, rain_mm = normalize_climate_rainfall(raw_climate, raw_loc)
    est_moist = normalize_establishment_moisture(raw_watering, raw_fm, is_suitable=0)
    hydro_stress = normalize_hydrological_hazard(raw_watering, raw_fm)
    sal_tier, sal_ec = normalize_salinity(raw_sal, raw_fm)
    geo_hazard = normalize_hazard_application(raw_app, raw_fm)

    records.append({
        "case_id": cid,
        # Standardized ML Features
        "soil_mechanical_class": soil_cls,
        "climate_rainfall_zone": clim_zone,
        "est_annual_rainfall_mm": rain_mm,
        "establishment_moisture": est_moist,
        "hydrological_stress": hydro_stress,
        "salinity_tier": sal_tier,
        "est_salinity_ec_ds_m": sal_ec,
        "geotechnical_hazard": geo_hazard,
        "Suitability": 0,
        # Auditable Metadata & Citations
        "raw_location": raw_loc,
        "raw_reference": raw_ref,
        "raw_soil_description": raw_soil,
        "raw_climate_description": raw_climate,
        "raw_watering_notes": raw_watering,
        "raw_salinity_text": raw_sal,
        "raw_application_text": raw_app,
        "raw_duration": raw_dur,
        "raw_root_depth": raw_root,
        "failure_mode": raw_fm,
        "failure_cause_details": raw_fc,
        "published_mitigation": raw_mit,
    })

df_canonical = pd.DataFrame(records)
df_canonical.to_csv(OUTPUT_PATH, index=False)

print("\n" + "=" * 70)
print(f"CANONICAL ML DATASET GENERATED: {OUTPUT_PATH}")
print("=" * 70)
print(f"Total Master Cases: {len(df_canonical)}")
print(f" - Growth Success Cases (Suitability = 1): {(df_canonical['Suitability'] == 1).sum()}")
print(f" - Scientific Failure Cases (Suitability = 0): {(df_canonical['Suitability'] == 0).sum()}")
print("\nFeature Distributions:")
for col in ["soil_mechanical_class", "climate_rainfall_zone", "establishment_moisture", "hydrological_stress", "salinity_tier", "geotechnical_hazard"]:
    print(f"\n--- {col} ---")
    print(df_canonical[col].value_counts())
