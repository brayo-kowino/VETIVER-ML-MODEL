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
) -> str:
    """Generate a structured Markdown engineering report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    verdict = "SUITABLE" if prediction == 1 else "UNSUITABLE"
    conf = f"{probability:.1f}%" if prediction == 1 else f"{100 - probability:.1f}%"

    lines = [
        "# VETIVER GEOTECHNICAL SITE SUITABILITY REPORT",
        f"**Generated:** {timestamp}",
        "**Assessment Standard:** Empirical Agronomic & Geotechnical Boundary Baselines",
        "\n---",
        "## 1. Executive Verdict",
        f"- **System Verdict:** **{verdict}**",
        f"- **Confidence Level:** {conf}",
        f"- **Risk Level:** {'Low Geotechnical Risk' if prediction == 1 else 'High Failure Risk'}",
    ]
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
        f"- **Soil Mechanical Class:** `{inputs.get('soil_mechanical_class', inputs.get('Soil type/structure', 'N/A'))}`",
        f"- **Precipitation Zone:** `{inputs.get('climate_rainfall_zone', inputs.get('climate', 'N/A'))}`",
        f"- **Establishment Moisture:** `{inputs.get('establishment_moisture', inputs.get('watering regime', 'N/A'))}`",
        f"- **Hydrological Stress:** `{inputs.get('hydrological_stress', 'Normal')}`",
        f"- **Salinity Tier:** `{inputs.get('salinity_tier', inputs.get('salinity', 'N/A'))}`",
        f"- **Geotechnical Hazard Mode:** `{inputs.get('geotechnical_hazard', inputs.get('Application', 'N/A'))}`",
        f"- **Civil Project Scope:** `{inputs.get('geotechnical_hazard_display', inputs.get('Application', inputs.get('geotechnical_hazard', 'N/A')))}`",
        "\n## 3. TreeSHAP Feature Attributions",
        "The table below indicates the mathematical contribution of each site condition to the final verdict:\n",
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
        lines.append(msg["content"])
        if msg.get("sources"):
            lines.append(f"\n> **Citations ({len(msg['sources'])} matched sources):**")
            for s in msg["sources"]:
                score_str = f" [Score: {s['score']:.3f}]" if s.get("score") is not None else ""
                lines.append(f"> - `{s['file_name']}`{score_str}: \"{s['snippet']}...\"")
    return "\n".join(lines)
