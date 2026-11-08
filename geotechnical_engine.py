"""
geotechnical_engine.py
Biomechanical Slope Stability Limit Equilibrium Engine,
Interactive 2D Slope Cross-Section Visualizer (Plotly),
Field Design & Bill of Quantities (BoQ) Material Takeoff Calculator,
and Dynamic Temporal Maturation Curve Engine.
"""

import math
import numpy as np
import plotly.graph_objects as go


def get_soil_shear_parameters(soil_class: str) -> tuple:
    """
    Returns (c_prime_kpa, phi_deg, unit_weight_gamma_kn_m3)
    based on USCS soil class.
    """
    s_lower = str(soil_class).lower()
    if "cohesive" in s_lower or "clay" in s_lower or "nitisol" in s_lower or "laterite" in s_lower:
        return 14.0, 27.0, 18.5
    elif "coarse sand" in s_lower or "pumice" in s_lower:
        return 2.0, 32.0, 16.5
    elif "mine tailing" in s_lower or "spoil" in s_lower:
        return 4.0, 29.0, 19.5
    elif "hardpan" in s_lower or "bedrock" in s_lower:
        return 35.0, 38.0, 22.0
    else:  # Sandy Clay / Loam
        return 10.0, 28.0, 18.0


def calculate_slope_stability(
    height_m: float = 6.0,
    slope_angle_deg: float = 35.0,
    soil_class: str = "Sandy Clay / Loam",
    water_table_ratio: float = 0.25,
    root_depth_m: float = 2.8,
    root_cohesion_kpa: float = 16.5,
    design_slip_depth_m: float = 1.0,
) -> dict:
    """
    Computes Limit Equilibrium Factor of Safety (FoS) for unreinforced vs bio-reinforced slope.
    Uses infinite slope limit equilibrium with pore-pressure ratio and Wu-Waldron root reinforcement.
    """
    c_prime, phi_deg, gamma = get_soil_shear_parameters(soil_class)
    beta = math.radians(slope_angle_deg)
    phi = math.radians(phi_deg)

    z = max(float(design_slip_depth_m), 0.3)
    gamma_w = 9.81
    u = water_table_ratio * gamma_w * z * (math.cos(beta) ** 2)

    # Driving shear stress at slip depth z
    tau_driving = gamma * z * math.sin(beta) * math.cos(beta)
    if tau_driving < 0.1:
        tau_driving = 0.1

    # Normal effective stress
    sigma_n = gamma * z * (math.cos(beta) ** 2)
    sigma_n_eff = max(sigma_n - u, 0.5)

    # Unreinforced resisting shear strength
    tau_resisting_base = c_prime + (sigma_n_eff * math.tan(phi))
    fos_base = round(tau_resisting_base / tau_driving, 2)

    # Vetiver root reinforcement:
    # If slip plane z is within effective root curtain (85% of root depth)
    effective_root_envelope = root_depth_m * 0.85
    if z <= effective_root_envelope:
        depth_decay = math.exp(-1.1 * (z / max(root_depth_m, 1.0)))
        effective_cr = root_cohesion_kpa * depth_decay
    else:
        effective_cr = 0.0

    tau_resisting_reinforced = (c_prime + effective_cr) + (sigma_n_eff * math.tan(phi))
    fos_reinforced = round(tau_resisting_reinforced / tau_driving, 2)
    fos_boost_pct = round(((fos_reinforced - fos_base) / max(fos_base, 0.1)) * 100, 1)

    return {
        "c_prime": c_prime,
        "phi_deg": phi_deg,
        "gamma": gamma,
        "slip_depth_m": z,
        "tau_driving_kpa": round(tau_driving, 2),
        "tau_resisting_base_kpa": round(tau_resisting_base, 2),
        "tau_resisting_reinforced_kpa": round(tau_resisting_reinforced, 2),
        "effective_cr_kpa": round(effective_cr, 2),
        "fos_unreinforced": fos_base,
        "fos_reinforced": fos_reinforced,
        "fos_boost_pct": fos_boost_pct,
        "is_stable_base": fos_base >= 1.30,
        "is_stable_reinforced": fos_reinforced >= 1.30,
    }


def generate_slope_cross_section_fig(
    height_m: float = 6.0,
    slope_angle_deg: float = 35.0,
    soil_class: str = "Sandy Clay / Loam",
    water_table_ratio: float = 0.25,
    root_depth_m: float = 2.8,
    root_cohesion_kpa: float = 16.5,
    design_slip_depth_m: float = 1.0,
) -> go.Figure:
    """
    Renders an interactive, publication-quality 2D Geotechnical Cross-Section Diagram (Plotly).
    Shows ground profile, unreinforced failure envelope, vetiver root anchors, and water table.
    """
    stability = calculate_slope_stability(
        height_m=height_m,
        slope_angle_deg=slope_angle_deg,
        soil_class=soil_class,
        water_table_ratio=water_table_ratio,
        root_depth_m=root_depth_m,
        root_cohesion_kpa=root_cohesion_kpa,
        design_slip_depth_m=design_slip_depth_m,
    )

    beta_rad = math.radians(slope_angle_deg)
    slope_run = height_m / math.tan(beta_rad)
    toe_run = 3.0
    crest_run = 4.0
    total_x = toe_run + slope_run + crest_run

    # 1. Slope Surface Polygon
    x_toe = toe_run
    x_crest = toe_run + slope_run

    surface_x = [0.0, x_toe, x_crest, total_x]
    surface_y = [0.0, 0.0, height_m, height_m]

    fig = go.Figure()

    # Soil body fill
    poly_x = [0.0, x_toe, x_crest, total_x, total_x, 0.0, 0.0]
    poly_y = [0.0, 0.0, height_m, height_m, -2.5, -2.5, 0.0]
    fig.add_trace(go.Scatter(
        x=poly_x,
        y=poly_y,
        fill="toself",
        fillcolor="rgba(241, 245, 249, 0.9)",
        line=dict(color="#64748B", width=2.5),
        name="Soil Substrate",
        hoverinfo="skip"
    ))

    # Slope Face Surface Outline
    fig.add_trace(go.Scatter(
        x=surface_x,
        y=surface_y,
        mode="lines",
        line=dict(color="#0F172A", width=3.5),
        name="Slope Surface Profile",
        hoverinfo="skip"
    ))

    # 2. Critical Shear Slip Plane (Dashed Red Line at depth z)
    z = stability["slip_depth_m"]
    # Offset along the normal: normal is (-sin beta, cos beta) downwards
    slip_x = [x_toe, x_crest + 0.5]
    slip_y = [-z * math.cos(beta_rad), height_m - z * math.cos(beta_rad)]

    # Arc-style slip curve
    slip_arc_x = np.linspace(x_toe + 0.2, x_crest + 0.8, 30)
    # Circular depression curve
    sag = z * 1.15
    slip_arc_y = np.linspace(0.0, height_m, 30) - (sag * (1 - ((slip_arc_x - (x_toe + slope_run/2)) / (slope_run/1.5))**2))
    # Clip slip_arc_y to be below surface
    surf_interp = np.interp(slip_arc_x, surface_x, surface_y)
    slip_arc_y = np.minimum(slip_arc_y, surf_interp - z * 0.85)

    fig.add_trace(go.Scatter(
        x=slip_arc_x,
        y=slip_arc_y,
        mode="lines",
        line=dict(color="#EF4444", width=2.5, dash="dash"),
        name=f"Critical Slip Plane (z = {z:.1f} m)",
        hovertemplate=f"Critical Shear Plane<br>Design Depth: {z:.2f} m<br>Unreinforced FoS: {stability['fos_unreinforced']}<extra></extra>"
    ))

    # 3. Water Table Line (Cyan Dashed)
    if water_table_ratio > 0.05:
        wt_y_toe = -0.5
        wt_y_crest = height_m * (1.0 - water_table_ratio * 0.8)
        fig.add_trace(go.Scatter(
            x=[0.0, total_x],
            y=[wt_y_toe, wt_y_crest],
            mode="lines",
            line=dict(color="#0284C7", width=1.8, dash="dot"),
            name="Groundwater Table (Phreatic Surface)",
            hovertemplate="Phreatic Water Line<extra></extra>"
        ))

    # 4. Vetiver Hedgerows & Root Reinforcement Curtain
    # Spacing along slope face
    face_length = math.sqrt(height_m**2 + slope_run**2)
    vi = 1.2 if slope_angle_deg >= 35 else 1.6  # Vertical interval
    num_rows = max(int(height_m / vi), 2)
    s_step = 1.0 / (num_rows + 1)

    for i in range(1, num_rows + 1):
        frac = i * s_step
        # Surface coordinate of this row
        rx = x_toe + frac * slope_run
        ry = frac * height_m

        # Above ground leaf crown (0.6 m tall)
        crown_x = [rx, rx + 0.15, rx - 0.15, rx]
        crown_y = [ry, ry + 0.6, ry + 0.6, ry]
        fig.add_trace(go.Scatter(
            x=crown_x,
            y=crown_y,
            fill="toself",
            fillcolor="#16A34A",
            line=dict(color="#15803D", width=1.5),
            name="Vetiver Crown" if i == 1 else None,
            showlegend=(i == 1),
            hoverinfo="skip"
        ))

        # Subsurface Root Column (vertical depth root_depth_m)
        root_bot_x = rx
        root_bot_y = ry - root_depth_m

        # Main taproot spine
        fig.add_trace(go.Scatter(
            x=[rx, root_bot_x],
            y=[ry, root_bot_y],
            mode="lines",
            line=dict(color="#15803D", width=3.5),
            name="Vetiver Root Curtain" if i == 1 else None,
            showlegend=(i == 1),
            hovertemplate=f"Row {i} Vetiver Root Column<br>Depth: {root_depth_m:.1f} m<br>Added Cohesion: +{stability['effective_cr_kpa']} kPa<extra></extra>"
        ))

        # Root branching fibers
        for branch_depth in [0.35, 0.70, 1.05, 1.45, 1.9, 2.3]:
            if branch_depth < root_depth_m:
                by = ry - branch_depth
                spread = 0.22 * (1.0 - (branch_depth / root_depth_m))
                fig.add_trace(go.Scatter(
                    x=[rx - spread, rx + spread],
                    y=[by - 0.08, by + 0.08],
                    mode="lines",
                    line=dict(color="#22C55E", width=1.5),
                    hoverinfo="skip",
                    showlegend=False
                ))

    # FoS Result Annotations
    base_color = "#DC2626" if stability["fos_unreinforced"] < 1.30 else "#16A34A"
    reinf_color = "#16A34A" if stability["fos_reinforced"] >= 1.30 else "#D97706"

    fig.add_annotation(
        x=x_crest + 0.5,
        y=height_m + 0.8,
        text=f"<b>Unreinforced FoS: {stability['fos_unreinforced']}</b><br><span style='color:{base_color}'>{'FAILING / AT RISK' if stability['fos_unreinforced'] < 1.30 else 'STABLE'}</span>",
        showarrow=True,
        arrowhead=2,
        arrowcolor="#EF4444",
        ax=40,
        ay=-35,
        bgcolor="#FEF2F2",
        bordercolor="#FECACA",
        borderwidth=1.5,
        font=dict(size=12, color="#991B1B"),
    )

    fig.add_annotation(
        x=x_toe + slope_run * 0.5,
        y=height_m * 0.5 - root_depth_m * 0.6,
        text=f"<b>Vetiver Reinforced FoS: {stability['fos_reinforced']}</b> ({stability['fos_boost_pct']:+}% Boost)<br><span style='color:{reinf_color}'>{'STABLE & CERTIFIED' if stability['fos_reinforced'] >= 1.30 else 'MARGINAL'}</span>",
        showarrow=True,
        arrowhead=2,
        arrowcolor="#16A34A",
        ax=-50,
        ay=40,
        bgcolor="#F0FDF4",
        bordercolor="#BBF7D0",
        borderwidth=1.5,
        font=dict(size=12, color="#14532D"),
    )

    fig.update_layout(
        title=dict(
            text=f"<b>2D Limit Equilibrium Slope Cross-Section (H = {height_m}m, β = {slope_angle_deg}°, USCS: {soil_class})</b>",
            font=dict(size=15, color="#0F172A")
        ),
        xaxis=dict(
            title="Horizontal Distance (m)",
            zeroline=False,
            showgrid=True,
            gridcolor="#E2E8F0",
            scaleanchor="y",
            scaleratio=1,
        ),
        yaxis=dict(
            title="Elevation (m a.s.l.)",
            zeroline=False,
            showgrid=True,
            gridcolor="#E2E8F0",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.28,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#CBD5E1",
            borderwidth=1
        ),
        margin=dict(l=40, r=40, t=60, b=70),
        height=480,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
    )

    return fig, stability


def calculate_field_boq(
    height_m: float = 6.0,
    width_m: float = 30.0,
    slope_angle_deg: float = 35.0,
    slip_spacing_m: float = 0.12,
    slip_unit_rate_kes: float = 12.0,
    kes_to_usd: float = 130.0,
) -> dict:
    """
    Computes contractor-grade Field Design & Bill of Quantities (BoQ) Takeoff.
    """
    beta_rad = math.radians(slope_angle_deg)
    slope_face_length = height_m / math.sin(beta_rad)
    slope_surface_area_m2 = slope_face_length * width_m

    # Recommended Vertical Interval (VI)
    if slope_angle_deg >= 45.0:
        recommended_vi = 1.0
    elif slope_angle_deg >= 32.0:
        recommended_vi = 1.3
    else:
        recommended_vi = 1.8

    # Recommended Horizontal Interval (HI) along slope
    recommended_hi = recommended_vi / math.sin(beta_rad)
    num_contour_rows = max(int(height_m / recommended_vi), 1)

    total_linear_meters = num_contour_rows * width_m

    # Number of slips needed (including 10% safety contingency)
    raw_slips = total_linear_meters / slip_spacing_m
    contingency_pct = 10.0
    total_slips = int(math.ceil(raw_slips * (1.0 + contingency_pct / 100.0)))

    # Material Takeoffs:
    # 1. Trench Excavation (0.3m wide x 0.3m deep)
    trench_vol_m3 = round(total_linear_meters * 0.3 * 0.3, 2)

    # 2. Well-Rotted Organic Compost (3.0 kg per linear meter)
    compost_kg = round(total_linear_meters * 3.0, 1)

    # 3. Cross-linked agricultural hydrogel polymer (8.0 g per linear meter)
    hydrogel_kg = round((total_linear_meters * 8.0) / 1000.0, 2)

    # 4. 60-Day Establishment Water (35 L total per slip)
    water_liters = int(total_slips * 35)

    # Cost Estimation:
    cost_slips_kes = total_slips * slip_unit_rate_kes
    cost_compost_kes = compost_kg * 15.0  # ~15 KES/kg
    cost_hydrogel_kes = hydrogel_kg * 1200.0  # ~1200 KES/kg
    cost_labor_planting_kes = total_linear_meters * 45.0  # Trenching, placement, backfill
    cost_watering_kes = (water_liters / 1000.0) * 450.0  # Bowser delivery & application

    total_kes = cost_slips_kes + cost_compost_kes + cost_hydrogel_kes + cost_labor_planting_kes + cost_watering_kes
    total_usd = total_kes / max(kes_to_usd, 1.0)

    boq_items = [
        {
            "item_no": "1.0",
            "description": f"Certified 3-Tiller Vetiver Nursery Slips (trimmed roots 10cm, pruned foliage 20cm, incl. {contingency_pct:.0f}% contingency)",
            "unit": "Slips",
            "quantity": total_slips,
            "rate_kes": slip_unit_rate_kes,
            "total_kes": cost_slips_kes,
        },
        {
            "item_no": "2.0",
            "description": "Planting Trench Excavation & Re-compacted Furrow Backfilling (0.30 m width × 0.30 m depth on surveyed contour)",
            "unit": "m³",
            "quantity": trench_vol_m3,
            "rate_kes": round(cost_labor_planting_kes / max(trench_vol_m3, 0.1), 1),
            "total_kes": cost_labor_planting_kes,
        },
        {
            "item_no": "3.0",
            "description": "Well-rotted Organic Compost / Soil Conditioner incorporated into trench root-zone (3.0 kg/linear meter)",
            "unit": "kg",
            "quantity": compost_kg,
            "rate_kes": 15.0,
            "total_kes": cost_compost_kes,
        },
        {
            "item_no": "4.0",
            "description": "Cross-linked Agricultural Polyacrylamide Hydrogel Polymers (moisture-buffering root coat, 8.0 g/linear meter)",
            "unit": "kg",
            "quantity": hydrogel_kg,
            "rate_kes": 1200.0,
            "total_kes": cost_hydrogel_kes,
        },
        {
            "item_no": "5.0",
            "description": "60-Day Dedicated Establishment Irrigation Regimen (initial 30-day bi-daily watering + subsequent weekly hardening)",
            "unit": "m³",
            "quantity": round(water_liters / 1000.0, 1),
            "rate_kes": 450.0,
            "total_kes": cost_watering_kes,
        },
    ]

    return {
        "slope_face_length_m": round(slope_face_length, 2),
        "slope_surface_area_m2": round(slope_surface_area_m2, 1),
        "recommended_vi_m": recommended_vi,
        "recommended_hi_m": round(recommended_hi, 2),
        "num_contour_rows": num_contour_rows,
        "total_linear_meters": round(total_linear_meters, 1),
        "total_slips": total_slips,
        "trench_vol_m3": trench_vol_m3,
        "compost_kg": compost_kg,
        "hydrogel_kg": hydrogel_kg,
        "water_liters": water_liters,
        "total_cost_kes": round(total_kes),
        "total_cost_usd": round(total_usd, 2),
        "cost_per_m2_kes": round(total_kes / max(slope_surface_area_m2, 1.0), 1),
        "boq_items": boq_items,
    }


def generate_temporal_maturation_fig(
    mature_root_depth_m: float = 2.8,
    mature_cr_kpa: float = 16.5,
    fos_base: float = 0.95,
    fos_mature: float = 1.45,
) -> go.Figure:
    """
    Renders an interactive 12-month Temporal Maturation & Vulnerability Window Curve (Plotly).
    Plots root depth, added shear cohesion, and slope Safety Factor progression over time.
    """
    months = np.linspace(0, 12, 49)

    # Sigmoidal / Gompertz bio-growth curve
    # Month 0: 0.15m (nursery slip core)
    # Month 2: 0.50m
    # Month 6: 1.80m
    # Month 12: mature_root_depth_m
    k_growth = 0.38
    root_curve = mature_root_depth_m * (1.0 - np.exp(-k_growth * months)) + 0.12 * np.exp(-months)

    # Added cohesion growth follows root matrix density
    cr_curve = mature_cr_kpa * ((months / 12.0) ** 1.35)

    # Factor of Safety evolution
    fos_curve = fos_base + (fos_mature - fos_base) * (cr_curve / max(mature_cr_kpa, 0.1))

    fig = go.Figure()

    # Vulnerability Shading (0 to 2 months)
    fig.add_vrect(
        x0=0,
        x1=2.0,
        fillcolor="rgba(239, 68, 68, 0.12)",
        layer="below",
        line_width=0,
        annotation_text="CRITICAL VULNERABILITY WINDOW (Day 0–60)<br>Coir Netting / Surface Straw Mulch Mandatory",
        annotation_position="top left",
        annotation_font=dict(size=10, color="#B91C1C", weight="bold")
    )

    # Intermediate Stabilization Shading (2 to 6 months)
    fig.add_vrect(
        x0=2.0,
        x1=6.0,
        fillcolor="rgba(245, 158, 11, 0.08)",
        layer="below",
        line_width=0,
        annotation_text="RAPID CANOPY TILLERING & CLOSURE (Month 2–6)",
        annotation_position="top left",
        annotation_font=dict(size=10, color="#B45309")
    )

    # Mature Armor Shading (6 to 12 months)
    fig.add_vrect(
        x0=6.0,
        x1=12.0,
        fillcolor="rgba(22, 163, 74, 0.08)",
        layer="below",
        line_width=0,
        annotation_text="FULL BIO-TENSILE CAGE MATURATION (Month 6–12)",
        annotation_position="top left",
        annotation_font=dict(size=10, color="#15803D")
    )

    # Trace 1: Safety Factor (Left Axis)
    fig.add_trace(go.Scatter(
        x=months,
        y=fos_curve,
        mode="lines",
        name="Factor of Safety (FoS)",
        line=dict(color="#16A34A", width=3.5),
        hovertemplate="Month %{x:.1f}: FoS = %{y:.2f}<extra></extra>"
    ))

    # Regulatory Stability Threshold (FoS = 1.30)
    fig.add_hline(
        y=1.30,
        line_dash="dash",
        line_color="#475569",
        line_width=1.5,
        annotation_text="Regulatory Stability Benchmark (FoS = 1.30)",
        annotation_position="bottom right",
        annotation_font=dict(size=10, color="#475569")
    )

    # Trace 2: Root Depth (Right Axis)
    fig.add_trace(go.Scatter(
        x=months,
        y=root_curve,
        mode="lines",
        name="Root Depth (m)",
        yaxis="y2",
        line=dict(color="#0284C7", width=2.5, dash="dot"),
        hovertemplate="Month %{x:.1f}: Root Depth = %{y:.2f} m<extra></extra>"
    ))

    # Trace 3: Added Shear Cohesion (Right Axis)
    fig.add_trace(go.Scatter(
        x=months,
        y=cr_curve,
        mode="lines",
        name="Added Cohesion cr (kPa)",
        yaxis="y3",
        line=dict(color="#8B5CF6", width=2.0, dash="dash"),
        hovertemplate="Month %{x:.1f}: cr = +%{y:.1f} kPa<extra></extra>"
    ))

    fig.update_layout(
        title=dict(
            text="<b>12-Month Temporal Root Maturation & Geotechnical Strength Growth Curve</b>",
            font=dict(size=15, color="#0F172A")
        ),
        xaxis=dict(
            title="Elapsed Time Post-Transplanting (Months)",
            range=[0, 12.2],
            showgrid=True,
            gridcolor="#E2E8F0",
        ),
        yaxis=dict(
            title="Slope Factor of Safety (FoS)",
            showgrid=True,
            gridcolor="#E2E8F0",
            range=[min(fos_base * 0.85, 0.7), max(fos_mature * 1.15, 1.6)],
        ),
        yaxis2=dict(
            title="Root Penetration Depth (m)",
            overlaying="y",
            side="right",
            range=[0, max(mature_root_depth_m * 1.25, 3.5)],
            showgrid=False,
        ),
        yaxis3=dict(
            title="Added Cohesion (kPa)",
            overlaying="y",
            side="right",
            position=0.95,
            range=[0, max(mature_cr_kpa * 1.3, 25.0)],
            showgrid=False,
            visible=False,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.28,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#CBD5E1",
            borderwidth=1
        ),
        margin=dict(l=50, r=60, t=60, b=70),
        height=450,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
    )

    return fig
