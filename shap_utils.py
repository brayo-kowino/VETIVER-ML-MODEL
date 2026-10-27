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
        feature_names = config.CATEGORICAL_FEATURES

    # Ensure input_data matches feature order
    row_df = input_data[feature_names].copy()
    pool = Pool(row_df, cat_features=feature_names)

    # CatBoost ShapValues returns (1, n_features + 1), where last column is base_value
    raw_shap = model.get_feature_importance(pool, type="ShapValues")
    feature_shap = raw_shap[0, :-1].tolist()
    base_value = float(raw_shap[0, -1])

    contributions = []
    for feat_name, shap_val in zip(feature_names, feature_shap):
        input_val = str(row_df.iloc[0][feat_name])
        impact_label = "Favors Suitable" if shap_val > 0 else ("Favors Unsuitable" if shap_val < 0 else "Neutral")
        contributions.append({
            "Feature": feat_name,
            "Input Value": input_val,
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
