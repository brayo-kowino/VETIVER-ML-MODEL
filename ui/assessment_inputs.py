import streamlit as st

import config
from services.assessment_service import run_assessment
from services.biomechanics import HAZARD_DISPLAY_MAP
from ui.state import PARAMETER_OPTIONS, input_provenance


def render_assessment_inputs(prefix: str, cb_model=None) -> None:
    state = st.session_state
    site = state.active_site
    widget_prefix = f"{prefix}_{state.site_revision}"
    measured_key = f"{widget_prefix}_measured"
    st.caption(f"Active site · {site['name']}")

    def update_input(field: str, widget_key: str) -> None:
        state.site_inputs = {**state.site_inputs, field: state[widget_key]}
        state.field_measured_inputs = [key for key in state.field_measured_inputs if key != field]
        state[measured_key] = state.field_measured_inputs
        state.current_assessment = None

    def update_measured() -> None:
        state.field_measured_inputs = state[measured_key]
        state.current_assessment = None

    columns = st.columns(2)
    for index, (field, (label, options)) in enumerate(PARAMETER_OPTIONS.items()):
        key = f"{widget_prefix}_{field}"
        columns[index // 3].selectbox(
            label,
            options,
            index=options.index(state.site_inputs[field]),
            format_func=HAZARD_DISPLAY_MAP.get if field == "geotechnical_hazard" else str,
            key=key,
            on_change=update_input,
            args=(field, key),
        )

    sources = input_provenance(site, state.site_inputs, state.field_measured_inputs)
    with st.expander("Input sources"):
        st.multiselect(
            "Inputs confirmed by field measurement",
            [field for field in PARAMETER_OPTIONS if field != "geotechnical_hazard"],
            default=state.field_measured_inputs if measured_key not in state else None,
            format_func=lambda field: PARAMETER_OPTIONS[field][0],
            key=measured_key,
            on_change=update_measured,
        )
        st.table([
            {"Input": PARAMETER_OPTIONS[field][0] if field in PARAMETER_OPTIONS else "Elevation", "Source": source}
            for field, source in sources.items()
        ])
        if site.get("citation"):
            st.caption(site["citation"])

    if state.current_assessment is None:
        st.caption("Run an assessment to see the verdict for these inputs.")
    label = "Reassess this site" if site["kind"] != "manual" or state.current_assessment else "Run geotechnical assessment"
    assess_col, clear_col = st.columns([2, 1])
    if assess_col.button(label, type="primary", icon=":material/play_arrow:", key=f"{prefix}_assess"):
        inputs = dict(state.site_inputs)
        inputs["est_salinity_ec_ds_m"] = config.SALINITY_TIER_TO_ECE[inputs["salinity_tier"]]
        inputs["geotechnical_hazard_display"] = HAZARD_DISPLAY_MAP[inputs["geotechnical_hazard"]]
        sources["est_salinity_ec_ds_m"] = "Estimated from the selected salinity tier"
        state.current_assessment = run_assessment(inputs, cb_model=cb_model, site=site, input_sources=sources)
        if state.current_assessment is None:
            st.warning("The suitability model is unavailable.")
    if state.current_assessment is not None:
        clear_col.button("Clear verdict", key=f"{prefix}_clear_result", on_click=state.update, args=({"current_assessment": None},))
