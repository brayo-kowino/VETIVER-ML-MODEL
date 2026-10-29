"""
Vetiver CatBoost Classifier Training Script.
Includes realistic geotechnical failure boundary cases,
5-fold stratified cross-validation, SHAP evaluation, and model serialization.
"""
import os
import sys

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
import shap

import config


def generate_failure_cases() -> pd.DataFrame:
    """
    Synthesize realistic negative/failure cases based on known agronomic and geotechnical
    failure boundaries for Vetiver grass (Chrysopogon zizanioides).
    """
    failures = [
        # 1-5: Extreme Cold / Freezing / Permafrost (Vetiver crown dies in frozen soil)
        {
            "Location(country/region)": "Siberia, Russia",
            "Soil type/structure": "permafrost frozen soil",
            "Soil density": "NR",
            "climate": "arctic freezing",
            "watering regime": "no water",
            "salinity": "NR",
            "Duration since planting ": "NR",
            "Root length(cm)": "0",
            "Application": "slope stabilization",
            "Reference": "Agronomic failure baseline",
            "Suitability": 0,
        },
        {
            "Location(country/region)": "Northern Canada",
            "Soil type/structure": "tundra peaty soil",
            "Soil density": "NR",
            "climate": "arctic freezing",
            "watering regime": "natural rainfall",
            "salinity": "NR",
            "Duration since planting ": "NR",
            "Root length(cm)": "0",
            "Application": "erosion control and slope stabilization",
            "Reference": "Agronomic failure baseline",
            "Suitability": 0,
        },
        {
            "Location(country/region)": "High Alpine, Switzerland",
            "Soil type/structure": "rocky mountain scree",
            "Soil density": "NR",
            "climate": "arctic freezing",
            "watering regime": "rainfed",
            "salinity": "NR",
            "Duration since planting ": "NR",
            "Root length(cm)": "0",
            "Application": "slope stabilization",
            "Reference": "Alpine frost failure",
            "Suitability": 0,
        },
        {
            "Location(country/region)": "Patagonia, Argentina",
            "Soil type/structure": "clay soil",
            "Soil density": "NR",
            "climate": "arctic freezing",
            "watering regime": "extreme rainfall",
            "salinity": "NR",
            "Duration since planting ": "NR",
            "Root length(cm)": "0",
            "Application": "erosion control",
            "Reference": "Cold-frost baseline",
            "Suitability": 0,
        },

        # 6-10: Solid Bedrock / Impenetrable Hardpan (Roots cannot penetrate)
        {
            "Location(country/region)": "Granite Quarry, USA",
            "Soil type/structure": "solid bedrock",
            "Soil density": "high",
            "climate": "tropical",
            "watering regime": "rainfed",
            "salinity": "NR",
            "Duration since planting ": "NR",
            "Root length(cm)": "0",
            "Application": "slope stabilization",
            "Reference": "Impenetrable strata failure",
            "Suitability": 0,
        },
        {
            "Location(country/region)": "Basalt Formation, India",
            "Soil type/structure": "solid bedrock",
            "Soil density": "high",
            "climate": "tropical monsoon",
            "watering regime": "natural rainfall",
            "salinity": "NR",
            "Duration since planting ": "NR",
            "Root length(cm)": "0",
            "Application": "erosion control and slope stabilization",
            "Reference": "Geotechnical refusal",
            "Suitability": 0,
        },
        {
            "Location(country/region)": "Limestone Escarpment, UK",
            "Soil type/structure": "solid bedrock",
            "Soil density": "high",
            "climate": "semi-arid",
            "watering regime": "no water",
            "salinity": "NR",
            "Duration since planting ": "NR",
            "Root length(cm)": "0",
            "Application": "slope stabilization",
            "Reference": "Bedrock boundary condition",
            "Suitability": 0,
        },

        # 11-15: Severe Arid Deserts without Irrigation (Desiccation prior to establishment)
        {
            "Location(country/region)": "Rub al Khali, Saudi Arabia",
            "Soil type/structure": "loose sandy soil",
            "Soil density": "NR",
            "climate": "desert arid",
            "watering regime": "no water",
            "salinity": "NR",
            "Duration since planting ": "NR",
            "Root length(cm)": "0",
            "Application": "erosion control",
            "Reference": "Severe moisture deficit",
            "Suitability": 0,
        },
        {
            "Location(country/region)": "Sahara, Egypt",
            "Soil type/structure": "loose sandy soil",
            "Soil density": "NR",
            "climate": "desert arid",
            "watering regime": "no water",
            "salinity": "high",
            "Duration since planting ": "NR",
            "Root length(cm)": "0",
            "Application": "reduction of runoff and soil erosion",
            "Reference": "Hyper-arid desiccation",
            "Suitability": 0,
        },
        {
            "Location(country/region)": "Atacama Desert, Chile",
            "Soil type/structure": "various soils",
            "Soil density": "NR",
            "climate": "desert arid",
            "watering regime": "no water",
            "salinity": "extreme",
            "Duration since planting ": "NR",
            "Root length(cm)": "0",
            "Application": "erosion control",
            "Reference": "Zero precipitation baseline",
            "Suitability": 0,
        },

        # 16-20: Prolonged Waterlogging in Anaerobic Heavy Clays
        {
            "Location(country/region)": "Stagnant Marsh, Bangladesh",
            "Soil type/structure": "clay",
            "Soil density": "NR",
            "climate": "tropical",
            "watering regime": "flooded completely",
            "salinity": "NR",
            "Duration since planting ": "NR",
            "Root length(cm)": "0",
            "Application": "slope stabilization",
            "Reference": "Anaerobic root rot",
            "Suitability": 0,
        },
        {
            "Location(country/region)": "Swamp Basin, Nigeria",
            "Soil type/structure": "clay soil",
            "Soil density": "NR",
            "climate": "tropical humid",
            "watering regime": "flooded completely",
            "salinity": "NR",
            "Duration since planting ": "NR",
            "Root length(cm)": "0",
            "Application": "soil erosion control using contour hedgerows",
            "Reference": "Submersion stress",
            "Suitability": 0,
        },

        # 21-25: Extreme Salinity & Toxic Tailings with Moisture Deprivation
        {
            "Location(country/region)": "Salt Flat, Bolivia",
            "Soil type/structure": "various soils",
            "Soil density": "NR",
            "climate": "semi-arid",
            "watering regime": "no water",
            "salinity": "extreme",
            "Duration since planting ": "NR",
            "Root length(cm)": "0",
            "Application": "erosion control and slope stabilization",
            "Reference": "Hyper-salinity threshold exceedance",
            "Suitability": 0,
        },
        {
            "Location(country/region)": "Toxic Leachate Tailings, Australia",
            "Soil type/structure": "gold mine tailings",
            "Soil density": "NR",
            "climate": "desert arid",
            "watering regime": "no water",
            "salinity": "extreme",
            "Duration since planting ": "NR",
            "Root length(cm)": "0",
            "Application": "rehabilitation of gold mine tailings",
            "Reference": "Unamended tailings failure",
            "Suitability": 0,
        },
        {
            "Location(country/region)": "Coastal Sodic Flat, Pakistan",
            "Soil type/structure": "clay",
            "Soil density": "NR",
            "climate": "tropical monsoon",
            "watering regime": "flooded completely",
            "salinity": "extreme",
            "Duration since planting ": "NR",
            "Root length(cm)": "0",
            "Application": "slope stabilization",
            "Reference": "Saline waterlog failure",
            "Suitability": 0,
        },
        {
            "Location(country/region)": "Incompatible Shaded Substrate",
            "Soil type/structure": "solid bedrock",
            "Soil density": "NR",
            "climate": "tropical humid",
            "watering regime": "extreme rainfall",
            "salinity": "high",
            "Duration since planting ": "NR",
            "Root length(cm)": "0",
            "Application": "erosion control and slope stabilization",
            "Reference": "Compound geotechnical failure",
            "Suitability": 0,
        },
    ]
    return pd.DataFrame(failures)


def run_training():
    print("=" * 65)
    print("VETIVER CATBOOST CLASSIFIER - TRAINING & VALIDATION")
    print("=" * 65)

    # 1. Load empirical dataset
    csv_path = os.path.join(config.DATA_DIR, "vetiver_growth_data.csv")
    print(f"\n[1/5] Loading empirical dataset from: {csv_path}")
    raw_df = pd.read_csv(csv_path)
    raw_df.fillna("NR", inplace=True)
    raw_df["Suitability"] = 1
    print(f"      Loaded {len(raw_df)} empirical success cases.")

    # 2. Add realistic failure boundary data
    print("\n[2/5] Injecting agronomic failure boundary cases...")
    failure_df = generate_failure_cases()
    print(f"      Injected {len(failure_df)} realistic failure conditions.")

    df_all = pd.concat([raw_df, failure_df], ignore_index=True)
    print(f"      Total combined dataset: {len(df_all)} rows.")
    print(f"      Class balance -> Suitable: {(df_all['Suitability'] == 1).sum()} | Unsuitable: {(df_all['Suitability'] == 0).sum()}")

    # 3. Features & Target
    cat_features = config.CATEGORICAL_FEATURES
    X = df_all[cat_features]
    y = df_all["Suitability"]

    # 4. Stratified 5-Fold Cross-Validation
    print("\n[3/5] Running Stratified 5-Fold Cross-Validation...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_accuracies = []
    fold_f1s = []
    fold_recalls = []
    fold_precisions = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        fold_model = CatBoostClassifier(
            iterations=120,
            learning_rate=0.08,
            depth=5,
            cat_features=cat_features,
            verbose=0,
            random_seed=42 + fold,
        )
        fold_model.fit(X_train, y_train)
        y_pred = fold_model.predict(X_val)

        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, zero_division=0)
        rec = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)

        fold_accuracies.append(acc)
        fold_f1s.append(f1)
        fold_recalls.append(rec)
        fold_precisions.append(prec)

        print(f"      Fold {fold} -> Accuracy: {acc*100:.1f}% | Precision: {prec*100:.1f}% | Recall: {rec*100:.1f}% | F1: {f1*100:.1f}%")

    print("\n      Cross-Validation Summary (5 Folds):")
    print(f"      • Mean Accuracy  : {np.mean(fold_accuracies)*100:.2f}% (± {np.std(fold_accuracies)*100:.2f}%)")
    print(f"      • Mean Precision : {np.mean(fold_precisions)*100:.2f}% (± {np.std(fold_precisions)*100:.2f}%)")
    print(f"      • Mean Recall    : {np.mean(fold_recalls)*100:.2f}% (± {np.std(fold_recalls)*100:.2f}%)")
    print(f"      • Mean F1-Score  : {np.mean(fold_f1s)*100:.2f}% (± {np.std(fold_f1s)*100:.2f}%)")

    # 5. Train Final Model on All Data
    print("\n[4/5] Training final model on all data and calculating SHAP values...")
    final_model = CatBoostClassifier(
        iterations=150,
        learning_rate=0.08,
        depth=5,
        cat_features=cat_features,
        verbose=0,
        random_seed=42,
    )
    final_model.fit(X, y)

    # 6. SHAP Summary Plot
    pool = Pool(X, cat_features=cat_features)
    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(pool)

    os.makedirs(config.DATA_DIR, exist_ok=True)
    shap_plot_path = os.path.join(config.DATA_DIR, "shap_summary.png")

    plt.figure(figsize=(9, 5))
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.title("CatBoost SHAP Global Feature Importance (Site Suitability)", fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig(shap_plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"      SHAP summary plot saved to: {shap_plot_path}")

    # 7. Save model
    print(f"\n[5/5] Saving production model to: {config.MODEL_PATH}")
    final_model.save_model(config.MODEL_PATH)
    print("      Model saved successfully.")
    print("=" * 65)
    print("Training and validation completed successfully!\n")


if __name__ == "__main__":
    run_training()
