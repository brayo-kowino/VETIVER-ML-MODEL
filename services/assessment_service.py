from typing import Optional, Dict, Any
import pandas as pd

import config
from shap_utils import explain_prediction
from services.biomechanics import check_hard_exclusion_boundaries, calculate_biomechanical_metrics
from services.report_service import generate_suitability_report


def run_assessment(inputs_dict: dict, cb_model=None) -> Optional[dict]:
    """Executes CatBoost prediction, probability, SHAP, and report generation with domain guardrails."""
    if cb_model is None:
        from services.model_service import load_catboost
        cb_model = load_catboost()

    if cb_model is None:
        return None

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
        prob = min(prob, 8.5)  # Enforce high failure confidence

    # Compute dynamic biomechanical geotechnical metrics
    biomechanics = calculate_biomechanical_metrics(inputs_dict)

    explanation = explain_prediction(cb_model, input_data)
    exp_df = explanation["explanation_df"]
    report_md = generate_suitability_report(
        inputs_dict, pred, prob, exp_df, is_exclusion, excl_title, excl_details, biomechanics
    )
    return {
        "inputs": inputs_dict,
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


def get_active_site_context(
    assessment: Optional[Dict[str, Any]] = None,
    custom_site: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Extracts normalized active assessment context for LLM grounding."""
    if assessment is None or custom_site is None:
        try:
            import streamlit as st
            if assessment is None:
                assessment = st.session_state.get("current_assessment")
            if custom_site is None:
                custom_site = st.session_state.get("custom_coord_site")
        except Exception:
            pass

    if not assessment and not custom_site:
        return None

    inputs = assessment.get("inputs", {}) if assessment else {}

    loc_name = inputs.get("location_name") or inputs.get("county")
    if not loc_name and custom_site:
        loc_name = custom_site.get("name") or custom_site.get("county")
    if not loc_name:
        loc_name = "Configured Engineering Site"

    elev = inputs.get("elevation_m")
    if elev is None and custom_site:
        elev = custom_site.get("elevation_m")
    elev_str = f"{elev:.0f}" if isinstance(elev, (int, float)) else str(elev or "N/A")

    soil = inputs.get("soil_mechanical_class") or (custom_site.get("soil_class") if custom_site else "N/A")
    climate = inputs.get("climate_rainfall_zone") or (custom_site.get("rainfall_zone") if custom_site else "N/A")
    moisture = inputs.get("establishment_moisture") or (custom_site.get("moisture_def") if custom_site else "N/A")
    hydro = inputs.get("hydrological_stress", "Normal Slope / Well-Drained")
    salinity = inputs.get("salinity_tier") or (custom_site.get("salinity") if custom_site else "Non-Saline")
    hazard = inputs.get("geotechnical_hazard") or (custom_site.get("civil_hazard") if custom_site else "N/A")

    pred = assessment.get("prediction") if assessment else None
    prob = assessment.get("probability", 0.0) if assessment else 0.0
    excl_title = assessment.get("excl_title", "None — Within safe biophysical envelope") if assessment else "N/A"
    biomechanics = assessment.get("biomechanics", {}) if assessment else {}

    return {
        "location_name": loc_name,
        "county": inputs.get("county", loc_name),
        "elevation_m": elev_str,
        "soil_mechanical_class": soil,
        "climate_rainfall_zone": climate,
        "establishment_moisture": moisture,
        "hydrological_stress": hydro,
        "salinity_tier": salinity,
        "geotechnical_hazard": hazard,
        "prediction": pred,
        "probability": prob,
        "excl_title": excl_title,
        "biomechanics": biomechanics,
    }
