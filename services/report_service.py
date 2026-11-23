from datetime import datetime
import pandas as pd

from services.biomechanics import calculate_biomechanical_metrics
from services.remediation import get_two_tier_remediations


def generate_suitability_report(
    inputs: dict,
    prediction: int,
    probability: float,
    exp_df: pd.DataFrame,
    is_exclusion: bool = False,
    excl_title: str = "",
    excl_details: str = "",
    biomechanics: dict = None,
    *,
    site: dict | None = None,
    input_sources: dict | None = None,
) -> str:
    """Generate a structured Markdown engineering report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    verdict = "SUITABLE" if prediction == 1 else "UNSUITABLE"
    site = site or {}
    input_sources = input_sources or {}

    lines = [
        "# VETIVER GEOTECHNICAL SITE SUITABILITY REPORT",
        f"**Generated** · {timestamp}",
        f"**Site** · {site.get('name', 'Manually configured site')}",
    ]
    if "latitude" in site:
        lines.append(f"**Coordinates** · {site['latitude']:.5f}, {site['longitude']:.5f}")
    if site.get("citation"):
        lines.append(f"**Benchmark reference** · {site['citation']}")
    lines.extend([
        "\n---",
        "## 1. Executive Verdict",
        f"- **System verdict** · **{verdict}**",
        f"- **CatBoost suitability score** · {probability:.1f}%",
        f"- **Verdict basis** · {'Screening rule override' if is_exclusion else 'CatBoost classification'}",
    ])
    if prediction == 1:
        bio = biomechanics if biomechanics else calculate_biomechanical_metrics(inputs)
        lines.extend([
            f"- **Estimated Vertical Root Depth:** `{bio['root_depth']}`",
            f"- **Mean Root Tensile Strength:** `{bio['tensile_strength']}`",
            f"- **Wu-Waldron Additional Shear Cohesion:** `{bio['root_cohesion']}`",
            f"- **Factor of Safety (ΔFoS) Boost:** `{bio['fos_boost']}` (at design slip depth $z = {bio['slip_depth_m']:.2f}\\text{{ m}}$)",
            f"- **Domain Technical Specification:** {bio.get('summary_desc', 'N/A')}",
            f"- **Technical Metric 1 ({bio.get('kpi1_label', 'Metric 1')}):** `{bio.get('kpi1_val', 'N/A')}`",
            f"- **Technical Metric 2 ({bio.get('kpi2_label', 'Metric 2')}):** `{bio.get('kpi2_val', 'N/A')}`",
            f"- **Technical Metric 3 ({bio.get('kpi3_label', 'Metric 3')}):** `{bio.get('kpi3_val', 'N/A')}`",
        ])
    elif is_exclusion:
        lines.extend([
            f"- **Diagnosed Critical Failure Limit:** `{excl_title}`",
            f"- **Biomechanical Failure Mechanism:** {excl_details}",
        ])

    lines.extend([
        "\n## 2. Configured Site Parameters",
        "",
        "| Parameter | Value | Source |",
        "| --- | --- | --- |",
    ])
    parameters = {
        "soil_mechanical_class": "Soil mechanical class",
        "climate_rainfall_zone": "Precipitation regime",
        "establishment_moisture": "Establishment moisture",
        "hydrological_stress": "Hydrological stress",
        "salinity_tier": "Salinity tier",
        "est_salinity_ec_ds_m": "Model ECe (dS/m)",
        "geotechnical_hazard": "Geotechnical objective",
        "elevation_m": "Elevation (m)",
    }
    for field, label in parameters.items():
        value = inputs.get(field, site.get(field))
        if value is not None:
            lines.append(f"| {label} | {value} | {input_sources.get(field, 'Unspecified')} |")
    lines.extend([
        "\n## 3. TreeSHAP Feature Attributions",
        "Each value describes the contribution of an input to the CatBoost score.\n",
        "| Site Parameter | Input Value | SHAP Impact (Log-Odds) | Directional Effect |",
        "| :--- | :--- | :--- | :--- |",
    ])
    for _, row in exp_df.iterrows():
        lines.append(f"| {row['Feature']} | {row['Input Value']} | {row['SHAP Impact']:+.4f} | {row['Effect']} |")

    rec_lines = [
        "\n## 4. Engineering Recommendations & Field Protocols",
    ]
    if prediction == 1:
        rec_lines.extend([
            "### 4.1 Optimal Establishment Protocol",
            "- **Initial Moisture:** Provide adequate irrigation or in-season rain during the first 30–60 days until the taproot column anchors past 1.0 m.",
            "- **Contour Spacing:** Spacing between slips in a row should be 10–15 cm to form a tight, closed sediment-trapping hedge within 6 months.",
            "\n### 4.2 Universal Field Stewardship Standards",
            "- **Solar Exposure:** Strictly obligate C4 heliophyte requiring >= 6 hours direct daily sunlight (zero canopy shading).",
            "- **Livestock Exclusion:** Enforce 12-month perimeter fencing to prevent browsing of tender juvenile tillers.",
            "- **Termite Protection:** Dip roots in neem oil solution or Metarhizium anisopliae biopesticide prior to planting in vulnerable soils.",
            "- **Certified Quality:** Use 3-tiller slips with trimmed roots (10 cm) and pruned leaves (15–20 cm).",
        ])
    else:
        remed = get_two_tier_remediations({
            "is_exclusion": is_exclusion,
            "excl_title": excl_title,
            "inputs": inputs,
        })
        rec_lines.extend([
            f"### 4.1 Tier 1: {remed['primary_title']}",
            f"- **Biomechanical Failure Mechanism:** {remed['primary_mech']}",
            "- **Prescribed Action Protocols:**",
        ])
        for proto_text, _ in remed['protocols']:
            clean_proto = proto_text.replace("**", "")
            rec_lines.append(f"  * {clean_proto}")

        rec_lines.extend([
            "\n### 4.2 Tier 2: Universal Field Stewardship & Biological Prerequisites",
            "- **Solar Exposure:** Strictly obligate C4 heliophyte requiring >= 6 hours direct daily sunlight (zero canopy shading).",
            "- **Livestock Exclusion:** Enforce 12-month perimeter fencing to prevent browsing of tender juvenile tillers.",
            "- **Termite Protection:** Dip roots in neem oil solution or Metarhizium anisopliae biopesticide prior to planting in vulnerable soils.",
            "- **Certified Quality:** Use 3-tiller slips with trimmed roots (10 cm) and pruned leaves (15–20 cm).",
        ])

    rec_lines.extend([
        "\n### 4.3 Long-Term Monitoring & Maintenance",
        "- Inspect root development, contour hedge closure, and localized rill formation at 3, 6, and 12-month intervals.",
        "\n---",
        "*Report compiled by Vetiver AI Decision Support Platform.*"
    ])
    lines.extend(rec_lines)
    return "\n".join(lines)


def export_chat_markdown(messages: list) -> str:
    """Format full chat history with citations into an exportable Markdown file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# Vetiver Literature & Knowledge Chat Session",
        f"**Exported:** {timestamp}\n",
        "---",
    ]
    for idx, msg in enumerate(messages, 1):
        role_label = "User" if msg["role"] == "user" else "Vetiver AI"
        lines.append(f"\n### {idx}. {role_label}")
        if msg.get("site_name"):
            lines.append(f"Site context · {msg['site_name']}\n")
        lines.append(msg["content"])
        if msg.get("sources"):
            lines.append("\n**Source excerpts**")
            for source in msg["sources"]:
                lines.append(f"\n**[{source['index']}] {source['file_name']}**\n")
                lines.extend(f"> {line}" for line in source["snippet"].splitlines())
    return "\n".join(lines)
