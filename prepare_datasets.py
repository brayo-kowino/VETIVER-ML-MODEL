"""
Script to prepare and synchronize Vetiver empirical datasets:
1. Loads 124-case growth data -> data/vetiver_growth_data.csv & knowledge_base/vetiver_growth_data.csv
2. Processes strictly 60-case scientific failure dataset (Sheet 1) -> data/vetiver_failure_data.csv
3. Processes 10-rule AI boundary parameters (Sheet 2) -> data/vetiver_boundary_parameters.csv
4. Produces master empirical dataset (184 cases: 124 success + 60 failure) -> data/vetiver_combined_dataset.csv
5. Generates canonical machine-ready dataset -> data/vetiver_canonical_ml_dataset.csv
"""
import os
import sys
import pandas as pd

# Paths
EXCEL_PATH = r"C:/Users/Brian/.gemini/antigravity/brain/24efd127-69dc-47af-ba97-b3d044a9cf14/.user_uploaded/media_1793968009739_c3c9c628.xlsx"
NEW_GROWTH_CSV = r"C:/Users/Brian/.gemini/antigravity/brain/24efd127-69dc-47af-ba97-b3d044a9cf14/.user_uploaded/media_1793972218661_dec306d9.csv"

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
KB_DIR = os.path.join(os.path.dirname(__file__), "knowledge_base")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(KB_DIR, exist_ok=True)

# 1. Growth Data (124 empirical success cases)
print("\n[1/4] Processing empirical growth data...")
df_growth = pd.read_csv(NEW_GROWTH_CSV)
for col in df_growth.select_dtypes(include=["object", "string"]).columns:
    df_growth[col] = df_growth[col].astype(str).str.strip()

df_growth["Soil type/structure"] = df_growth["Soil type/structure"].str.replace("\n", " ").str.strip()

growth_csv_path = os.path.join(DATA_DIR, "vetiver_growth_data.csv")
kb_growth_csv_path = os.path.join(KB_DIR, "vetiver_growth_data.csv")
df_growth.to_csv(growth_csv_path, index=False)
df_growth.to_csv(kb_growth_csv_path, index=False)
print(f"      Saved {len(df_growth)} rows to {growth_csv_path} and {kb_growth_csv_path}")

# 2. Failure Data (Strictly 60 empirical cases from Sheet 1)
print("\n[2/4] Processing scientific failure register (Sheet 1)...")
df_fail_raw = pd.read_excel(EXCEL_PATH, sheet_name=0, skiprows=2)
for col in df_fail_raw.select_dtypes(include=["object", "string"]).columns:
    df_fail_raw[col] = df_fail_raw[col].astype(str).str.strip()

df_fail = pd.DataFrame()
df_fail["Location(country/region)"] = df_fail_raw["LOCATION"]
df_fail["Soil type/structure"] = df_fail_raw["SOIL TYPE"]
df_fail["Soil density"] = df_fail_raw["SOIL DENSITY (g/cm\u00b3)"].apply(
    lambda x: f"{x} g/cm\u00b3" if pd.notnull(x) and str(x) != "nan" else "NR"
)
df_fail["climate"] = df_fail_raw["CLIMATE"]
df_fail["watering regime"] = df_fail_raw["WATERING"]
df_fail["salinity"] = df_fail_raw["SALINITY (dS/m)"].apply(
    lambda x: f"{x} dS/m" if pd.notnull(x) and str(x) != "nan" else "NR"
)
df_fail["Duration since planting "] = df_fail_raw["DURATION"]
df_fail["Root length(cm)"] = df_fail_raw["ROOT LENGTH (m)"].apply(
    lambda x: f"{round(float(x) * 100)} cm" if pd.notnull(x) and str(x) != "nan" else "NR"
)
df_fail["Application"] = df_fail_raw["APPLICATION"]
df_fail["Reference"] = df_fail_raw["REFERENCE"]
df_fail["Suitability"] = 0

# Diagnostics
df_fail["Failure Mode"] = df_fail_raw["FAILURE MODE"]
df_fail["Failure Cause Details"] = df_fail_raw["FAILURE CAUSE DETAILS"]
df_fail["Mitigation"] = df_fail_raw["MITIGATION (SCIENTISTS' PUBLISHED RECOMMENDATIONS)"]
df_fail["Exclusion Boundary"] = df_fail_raw["SUITABILITY (EXCLUSION BOUNDARY)"]

fail_csv_path = os.path.join(DATA_DIR, "vetiver_failure_data.csv")
df_fail.to_csv(fail_csv_path, index=False)
print(f"      Saved strictly {len(df_fail)} failure cases to {fail_csv_path}")

# 3. Boundary Parameters (Sheet 2 - 10 Rules)
print("\n[3/4] Processing AI Boundary Parameters (Sheet 2)...")
df_bounds = pd.read_excel(EXCEL_PATH, sheet_name=1, skiprows=1)
for col in df_bounds.select_dtypes(include=["object", "string"]).columns:
    df_bounds[col] = df_bounds[col].astype(str).str.strip()

bounds_csv_path = os.path.join(DATA_DIR, "vetiver_boundary_parameters.csv")
df_bounds.to_csv(bounds_csv_path, index=False)
print(f"      Saved {len(df_bounds)} boundary conditions to {bounds_csv_path}")

# 4. Master Combined ML Dataset (Strictly 184 cases)
print("\n[4/4] Building combined empirical ML dataset (184 cases)...")
standard_cols = [
    "Location(country/region)",
    "Soil type/structure",
    "Soil density",
    "climate",
    "watering regime",
    "salinity",
    "Duration since planting ",
    "Root length(cm)",
    "Application",
    "Reference",
    "Suitability",
]

df_growth_ml = df_growth.copy()
df_growth_ml["Suitability"] = 1

df_combined = pd.concat([df_growth_ml[standard_cols], df_fail[standard_cols]], ignore_index=True)
comb_csv_path = os.path.join(DATA_DIR, "vetiver_combined_dataset.csv")
df_combined.to_csv(comb_csv_path, index=False)
print(f"      Saved {len(df_combined)} rows to {comb_csv_path}")
print(f"      Suitable (1): {(df_combined['Suitability'] == 1).sum()} | Unsuitable (0): {(df_combined['Suitability'] == 0).sum()}")

# 5. Build Canonical ML Dataset
print("\n[5/5] Invoking build_canonical_dataset.py...")
import build_canonical_dataset

print("\nAll datasets synchronized successfully!")
