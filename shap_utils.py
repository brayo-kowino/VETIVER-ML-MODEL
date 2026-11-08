"""
SHAP Explainability utilities for CatBoost predictions.
Extracts feature attributions using CatBoost's native TreeSHAP engine.
"""
from typing import Dict, Any, List
import pandas as pd
from catboost import CatBoostClassifier, Pool

import config


def explain_prediction(
    model: CatBoostClassifier,
    input_data: pd.DataFrame,
    feature_names: List[str] = None,
) -> Dict[str, Any]:
    """
    Computes SHAP feature importance for a single observation.
    Returns:
        dict with:
            - 'feature_names': list of features
            - 'shap_values': list of SHAP impact values for each feature
            - 'base_value': model expected value (prior log-odds)
            - 'contributions': list of dicts with feature name, value, impact, and direction
            - 'explanation_df': pd.DataFrame ready for display or plotting
    """
    if feature_names is None:
        feature_names = config.MODEL_FEATURES

    # Ensure input_data matches feature order
    row_df = input_data[feature_names].copy()
    
    # Ensure numeric columns are proper float type
    for num_col in config.NUM_FEATURES:
        if num_col in row_df.columns:
            row_df[num_col] = pd.to_numeric(row_df[num_col], errors="coerce").fillna(0.5)

    # Use explicitly defined categorical features
    cat_cols = [c for c in config.CAT_FEATURES if c in row_df.columns]
    pool = Pool(row_df, cat_features=cat_cols)

    # CatBoost ShapValues returns (1, n_features + 1), where last column is base_value
    raw_shap = model.get_feature_importance(pool, type="ShapValues")
    feature_shap = raw_shap[0, :-1].tolist()
    base_value = float(raw_shap[0, -1])

    # Human-friendly labels for UI tables
    friendly_names = {
        "soil_mechanical_class": "Soil Mechanical Class",
        "climate_rainfall_zone": "Precipitation & Climate Zone",
        "establishment_moisture": "Establishment Moisture (0–60 d)",
        "hydrological_stress": "Hydrological Extreme Stress",
        "est_salinity_ec_ds_m": "Substrate Salinity (ECe)",
        "geotechnical_hazard": "Geotechnical Objective & Depth",
    }

    # Reverse lookup for salinity display
    ece_to_tier = {
        0.5: "Non-Saline (< 2 dS/m)",
        4.0: "Slight to Moderate (2–8 dS/m)",
        10.0: "High Saline (8–16 dS/m)",
        19.5: "Extreme Hypersaline (> 16 dS/m)",
    }

    contributions = []
    for feat_name, shap_val in zip(feature_names, feature_shap):
        raw_val = row_df.iloc[0][feat_name]
        if feat_name == "est_salinity_ec_ds_m":
            display_val = ece_to_tier.get(float(raw_val), f"{raw_val} dS/m")
        else:
            display_val = str(raw_val)

        impact_label = "Favors Suitable" if shap_val > 0.01 else ("Favors Unsuitable" if shap_val < -0.01 else "Neutral")
        contributions.append({
            "Feature": friendly_names.get(feat_name, feat_name),
            "Input Value": display_val,
            "SHAP Impact": round(float(shap_val), 4),
            "Effect": impact_label,
        })

    # Sort by absolute impact
    contributions.sort(key=lambda x: abs(x["SHAP Impact"]), reverse=True)
    explanation_df = pd.DataFrame(contributions)

    return {
        "feature_names": feature_names,
        "shap_values": feature_shap,
        "base_value": base_value,
        "contributions": contributions,
        "explanation_df": explanation_df,
    }
