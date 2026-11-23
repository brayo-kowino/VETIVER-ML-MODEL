from typing import Optional
import pandas as pd

import config
from shap_utils import explain_prediction
from services.biomechanics import check_hard_exclusion_boundaries, calculate_biomechanical_metrics
from services.report_service import generate_suitability_report


def run_assessment(inputs_dict: dict, cb_model=None, *, site=None, input_sources=None) -> Optional[dict]:
    """Executes CatBoost prediction, probability, SHAP, and report generation with domain guardrails."""
    if cb_model is None:
        from services.model_service import load_catboost
        cb_model = load_catboost()

    if cb_model is None:
        return None

    inputs_dict = dict(inputs_dict)
    site = dict(site or {})
    input_sources = dict(input_sources or {})

    # Ensure est_salinity_ec_ds_m is populated
    if "est_salinity_ec_ds_m" not in inputs_dict:
        tier = inputs_dict.get("salinity_tier", inputs_dict.get("salinity", "Non-Saline (< 2 dS/m)"))
        inputs_dict["est_salinity_ec_ds_m"] = config.SALINITY_TIER_TO_ECE.get(tier, 0.5)

    # Format DataFrame matching exact MODEL_FEATURES for CatBoost
    model_row = {}
    for feat in config.MODEL_FEATURES:
        if feat in inputs_dict:
            model_row[feat] = [inputs_dict[feat]]
        else:
            model_row[feat] = [0.5] if feat in config.NUM_FEATURES else ["Unknown"]

    input_data = pd.DataFrame(model_row)
    for num_col in config.NUM_FEATURES:
        if num_col in input_data.columns:
            input_data[num_col] = pd.to_numeric(input_data[num_col], errors="coerce").fillna(0.5)
    for cat_col in config.CAT_FEATURES:
        if cat_col in input_data.columns:
            input_data[cat_col] = input_data[cat_col].astype(str)

    pred = int(cb_model.predict(input_data)[0])
    prob = float(cb_model.predict_proba(input_data)[0][1] * 100)

    # Geotechnical Domain Exclusion Guardrail Override
    is_exclusion, excl_title, excl_details = check_hard_exclusion_boundaries(inputs_dict)
    if is_exclusion:
        pred = 0

    # Compute dynamic biomechanical geotechnical metrics
    biomechanics = calculate_biomechanical_metrics(inputs_dict)

    explanation = explain_prediction(cb_model, input_data)
    exp_df = explanation["explanation_df"]
    report_md = generate_suitability_report(
        inputs_dict, pred, prob, exp_df, is_exclusion, excl_title, excl_details, biomechanics,
        site=site, input_sources=input_sources,
    )
    return {
        "inputs": inputs_dict,
        "site": site,
        "input_sources": input_sources,
        "prediction": pred,
        "probability": prob,
        "explanation": explanation,
        "exp_df": exp_df,
        "report_md": report_md,
        "is_exclusion": is_exclusion,
        "excl_title": excl_title,
        "excl_details": excl_details,
        "biomechanics": biomechanics,
    }


def get_active_site_context(site: dict, inputs: dict, assessment: Optional[dict] = None) -> dict:
    """Use the saved assessment only when it matches the current site and inputs."""
    if assessment and (
        assessment.get("site", {}).get("id") != site["id"]
        or any(assessment["inputs"].get(field) != value for field, value in inputs.items())
    ):
        assessment = None
    result = assessment or {}
    return {
        **inputs,
        "location_name": site["name"],
        "elevation_m": site.get("elevation_m", "N/A"),
        "latitude": site.get("latitude"),
        "longitude": site.get("longitude"),
        "prediction": result.get("prediction"),
        "probability": result.get("probability"),
        "excl_title": result.get("excl_title") if result.get("is_exclusion") else None,
        "biomechanics": result.get("biomechanics", {}),
    }
