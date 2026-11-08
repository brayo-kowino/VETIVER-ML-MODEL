"""
corridor_engine.py
Infrastructure Corridor Batch Profiler for Linear Civil Works (Highways, Railways, Pipelines).
Provides pre-computed transport corridors across East Africa and custom CSV batch analysis.
"""

import math
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from geospatial_resolver import resolve_coordinate_geotechnics

# ── PRE-COMPUTED REGIONAL INFRASTRUCTURE CORRIDORS ─────────────────────────
BUILTIN_CORRIDORS = {
    "a109_mombasa_highway": {
        "name": "A109 Highway: Nairobi – Mombasa Alignment (325 km)",
        "description": "Critical trans-African logistics corridor traversing Kapiti black cotton plains, Makindu semi-arid loams, and Voi metamorphic highway rock-cuts.",
        "waypoints": [
            {"chainage_km": 0, "station": "Athi River / Kapiti Cuttings", "lat": -1.4532, "lon": 36.9821, "hazard": "Shallow Slope / Embankment Slip (0.3–1.5 m)"},
            {"chainage_km": 42, "station": "Machakos Junction Batters", "lat": -1.5841, "lon": 37.1120, "hazard": "Surficial Sheet & Rill Erosion (< 0.3 m)"},
            {"chainage_km": 78, "station": "Konza / Ulu Hill Cuttings", "lat": -1.7421, "lon": 37.1845, "hazard": "Shallow Slope / Embankment Slip (0.3–1.5 m)"},
            {"chainage_km": 118, "station": "Sultan Hamud Drainage Basin", "lat": -2.0234, "lon": 37.3712, "hazard": "Riverbank / Hydrodynamic Toe Scour"},
            {"chainage_km": 158, "station": "Emali Railway / Road Batters", "lat": -2.1541, "lon": 37.5218, "hazard": "Shallow Slope / Embankment Slip (0.3–1.5 m)"},
            {"chainage_km": 208, "station": "Kibwezi Basaltic Zone", "lat": -2.4215, "lon": 37.9712, "hazard": "Surficial Sheet & Rill Erosion (< 0.3 m)"},
            {"chainage_km": 265, "station": "Mtito Andei River Approaches", "lat": -2.6841, "lon": 38.1724, "hazard": "Riverbank / Hydrodynamic Toe Scour"},
            {"chainage_km": 295, "station": "Tsavo River Bridge Batters", "lat": -2.9912, "lon": 38.4612, "hazard": "Riverbank / Hydrodynamic Toe Scour"},
            {"chainage_km": 325, "station": "Voi Deep Highway Batters", "lat": -3.3941, "lon": 38.5671, "hazard": "Shallow Slope / Embankment Slip (0.3–1.5 m)"},
        ]
    },
    "rift_valley_escarpment": {
        "name": "Rift Valley Highway: Mai Mahiu – Naivasha – Nakuru (125 km)",
        "description": "High-risk volcanic escarpment corridor characterized by collapsible Menengai/Longonot pumice soils, severe hydro-piping, and gully canyons.",
        "waypoints": [
            {"chainage_km": 0, "station": "Limuru Escarpment Crest (A104)", "lat": -1.1215, "lon": 36.6521, "hazard": "Shallow Slope / Embankment Slip (0.3–1.5 m)"},
            {"chainage_km": 18, "station": "Mai Mahiu Fault Scarp Cuttings", "lat": -1.0124, "lon": 36.5821, "hazard": "Shallow Slope / Embankment Slip (0.3–1.5 m)"},
            {"chainage_km": 36, "station": "Mt. Longonot Flank Gullies", "lat": -0.9124, "lon": 36.4512, "hazard": "Surficial Sheet & Rill Erosion (< 0.3 m)"},
            {"chainage_km": 54, "station": "Naivasha Lake Basin Cuts", "lat": -0.7172, "lon": 36.4312, "hazard": "Surficial Sheet & Rill Erosion (< 0.3 m)"},
            {"chainage_km": 75, "station": "Kariandusi Diatomite Cuttings", "lat": -0.5821, "lon": 36.3512, "hazard": "Shallow Slope / Embankment Slip (0.3–1.5 m)"},
            {"chainage_km": 96, "station": "Gilgil River Escarpment Flanks", "lat": -0.4912, "lon": 36.3124, "hazard": "Riverbank / Hydrodynamic Toe Scour"},
            {"chainage_km": 125, "station": "Nakuru Menengai Pumice Slopes", "lat": -0.2812, "lon": 36.0712, "hazard": "Shallow Slope / Embankment Slip (0.3–1.5 m)"},
        ]
    },
    "mau_summit_kisumu": {
        "name": "Western Highway: Mau Summit – Kericho – Kisumu (135 km)",
        "description": "High-altitude equatorial corridor passing from alpine cold zones (>2,200 m) through humid Kericho tea red nitisols down to Lake Victoria clay basins.",
        "waypoints": [
            {"chainage_km": 0, "station": "Mau Summit High-Altitude Pass", "lat": -0.1621, "lon": 35.6812, "hazard": "Shallow Slope / Embankment Slip (0.3–1.5 m)"},
            {"chainage_km": 25, "station": "Kedowa Heavy Rainfall Cuts", "lat": -0.2215, "lon": 35.5124, "hazard": "Shallow Slope / Embankment Slip (0.3–1.5 m)"},
            {"chainage_km": 55, "station": "Kericho Central Tea Slopes", "lat": -0.3689, "lon": 35.2863, "hazard": "Surficial Sheet & Rill Erosion (< 0.3 m)"},
            {"chainage_km": 82, "station": "Kapsaos High-Gradient Batters", "lat": -0.3124, "lon": 35.1512, "hazard": "Shallow Slope / Embankment Slip (0.3–1.5 m)"},
            {"chainage_km": 108, "station": "Awasi Escarpment Drainage Toe", "lat": -0.1421, "lon": 35.0812, "hazard": "Riverbank / Hydrodynamic Toe Scour"},
            {"chainage_km": 135, "station": "Kisumu Lakeside Alluvial Plain", "lat": -0.0917, "lon": 34.7680, "hazard": "Riverbank / Hydrodynamic Toe Scour"},
        ]
    }
}


def analyze_corridor_waypoints(waypoints: list, run_assessment_fn) -> pd.DataFrame:
    """
    Batches coordinate resolution and CatBoost assessment across linear infrastructure waypoints.
    """
    rows = []
    for wp in waypoints:
        lat = wp["lat"]
        lon = wp["lon"]
        ch = wp.get("chainage_km", 0)
        station = wp.get("station", f"km {ch:03d}")
        haz = wp.get("hazard", "Shallow Slope / Embankment Slip (0.3–1.5 m)")

        # Resolve environmental baseline
        site_data = resolve_coordinate_geotechnics(lat, lon)

        # Build model inputs with project nursery establishment (Adequate moisture)
        inputs = {
            "soil_mechanical_class": site_data["soil_mechanical_class"],
            "climate_rainfall_zone": site_data["climate_rainfall_zone"],
            "establishment_moisture": site_data["establishment_moisture"],
            "hydrological_stress": site_data["hydrological_stress"],
            "salinity_tier": site_data["salinity_tier"],
            "est_salinity_ec_ds_m": 0.5,
            "geotechnical_hazard": haz,
        }

        # Execute assessment
        assessment = run_assessment_fn(inputs)
        pred = assessment["prediction"]
        prob = assessment["probability"]
        is_excl = assessment["is_exclusion"]
        bio = assessment.get("biomechanics", {})

        if pred == 1:
            category = "Suitable (Pure Bio-Armor)"
            cat_color = "#16A34A"
            recom = f"Contour Vetiver hedges at VI = 1.2 m. Expected {bio.get('fos_boost', '+35% to +50%')} FoS boost."
        elif is_excl and "pumice" in str(site_data["soil_mechanical_class"]).lower():
            category = "Caution (Hydro-Piping Hazard)"
            cat_color = "#D97706"
            recom = "Collapsible volcanic pumice: Pinned geofabric + subsurface aggregate cutoff drains mandatory."
        elif is_excl and "altitude" in str(assessment.get("excl_title", "")).lower():
            category = "Restricted (High-Altitude Cold)"
            cat_color = "#2563EB"
            recom = "High altitude cold (>2,200 m): Combine with cold-hardy montane grasses (Festuca)."
        else:
            category = "Hybrid (Structural Nailing Required)"
            cat_color = "#DC2626"
            recom = f"High shear hazard: Structural steel soil nails + Vetiver surficial bio-armor veneer."

        rows.append({
            "Chainage (km)": ch,
            "Station Name": station,
            "Latitude": lat,
            "Longitude": lon,
            "Elevation (m)": site_data["elevation_m"],
            "USCS Soil Class": site_data["soil_mechanical_class"],
            "Precipitation Zone": site_data["climate_rainfall_zone"],
            "Civil Hazard": haz,
            "Verdict": "SUITABLE" if pred == 1 else "AT RISK",
            "Model Confidence": f"{prob:.1f}%" if pred == 1 else f"{100-prob:.1f}%",
            "Category": category,
            "Category Color": cat_color,
            "FoS Boost": bio.get("fos_boost", "N/A"),
            "Prescribed Engineering Protocol": recom,
        })

    return pd.DataFrame(rows)


def generate_corridor_profile_fig(df_corridor: pd.DataFrame, corridor_name: str) -> go.Figure:
    """
    Renders an interactive Dual-Subplot Infrastructure Alignment Profile (Plotly).
    Top subplot: Elevation profile with colored categorization pins.
    Bottom subplot: Geotechnical Confidence & Safety Factor Boost.
    """
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=(
            "<b>Corridor Topographic Elevation Profile & Geotechnical Zonation</b>",
            "<b>Suitability Confidence & Biomechanical Reinforcement Profile</b>"
        ),
        row_heights=[0.58, 0.42]
    )

    # 1. Elevation Profile
    fig.add_trace(
        go.Scatter(
            x=df_corridor["Chainage (km)"],
            y=df_corridor["Elevation (m)"],
            mode="lines",
            name="Alignment Elevation (m a.s.l.)",
            line=dict(color="#64748B", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(241, 245, 249, 0.8)",
            hovertemplate="Chainage: km %{x}<br>Elevation: %{y} m<extra></extra>"
        ),
        row=1, col=1
    )

    # Add colored marker pins on elevation profile
    color_map = {
        "Suitable (Pure Bio-Armor)": "#16A34A",
        "Caution (Hydro-Piping Hazard)": "#D97706",
        "Restricted (High-Altitude Cold)": "#2563EB",
        "Hybrid (Structural Nailing Required)": "#DC2626"
    }

    for cat, color in color_map.items():
        sub_df = df_corridor[df_corridor["Category"] == cat]
        if not sub_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=sub_df["Chainage (km)"],
                    y=sub_df["Elevation (m)"],
                    mode="markers+text",
                    name=cat,
                    text=[f"km {c}" for c in sub_df["Chainage (km)"]],
                    textposition="top center",
                    textfont=dict(size=10, color=color),
                    marker=dict(
                        size=13,
                        color=color,
                        line=dict(color="#FFFFFF", width=2.0)
                    ),
                    hovertemplate="<b>%{customdata[0]}</b> (km %{x})<br>Elevation: %{y} m<br>Soil: %{customdata[1]}<br>Protocol: %{customdata[2]}<extra></extra>",
                    customdata=sub_df[["Station Name", "USCS Soil Class", "Prescribed Engineering Protocol"]].values
                ),
                row=1, col=1
            )

    # 2. Confidence / Safety Factor Boost Subplot
    # Convert Confidence string to numeric
    conf_nums = df_corridor["Model Confidence"].str.rstrip("%").astype(float)
    fig.add_trace(
        go.Bar(
            x=df_corridor["Chainage (km)"],
            y=conf_nums,
            name="Geotechnical Model Confidence (%)",
            marker=dict(
                color=df_corridor["Category Color"],
                line=dict(color="#334155", width=1)
            ),
            hovertemplate="Chainage: km %{x}<br>Confidence: %{y}%<br>Status: %{customdata}<extra></extra>",
            customdata=df_corridor["Verdict"]
        ),
        row=2, col=1
    )

    fig.add_hline(
        y=80.0,
        line_dash="dash",
        line_color="#16A34A",
        line_width=1.5,
        annotation_text="High Confidence Threshold (80%)",
        annotation_position="bottom right",
        annotation_font=dict(size=9, color="#16A34A"),
        row=2, col=1
    )

    fig.update_layout(
        title=dict(
            text=f"<b>Linear Infrastructure Geotechnical Zonation: {corridor_name}</b>",
            font=dict(size=15, color="#0F172A")
        ),
        xaxis=dict(title="Chainage Distance (km)", showgrid=True, gridcolor="#E2E8F0"),
        xaxis2=dict(title="Chainage Distance (km)", showgrid=True, gridcolor="#E2E8F0"),
        yaxis=dict(title="Elevation (m a.s.l.)", showgrid=True, gridcolor="#E2E8F0"),
        yaxis2=dict(title="Suitability Confidence (%)", showgrid=True, gridcolor="#E2E8F0", range=[0, 105]),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#CBD5E1",
            borderwidth=1
        ),
        margin=dict(l=50, r=40, t=70, b=80),
        height=580,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
    )

    return fig
