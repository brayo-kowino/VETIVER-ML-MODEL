# VETIVER GRASS BIOMECHANICAL & PHYSIOLOGICAL BOUNDARY CONDITIONS FOR AI BIOENGINEERING

This reference defines the strict quantitative failure thresholds, primary mechanisms, empirical citations, and rule representations for machine learning and geotechnical decision support.

| Stress Factor / Regional Domain | Critical Threshold for Failure | Primary Failure Mechanism | Empirical Citations | AI Feature Representation |
| :--- | :--- | :--- | :--- | :--- |
| **Slope Slip Surface Depth** | > 1.5 – 2.0 m below surface | Deep rotational / translational shear slip below root mat | Hengchaovanich (1998); KeRRA (2022); MININFRA (2020) | `slip_depth_m > 2.0` |
| **Soil Bulk Density** | > 1.75 g/cm³ (High compaction) | Mechanical impedance; root penetration arrested <0.4 m | Tran et al. (2003); EIAR (2021) | `bulk_density_g_cm3 > 1.75` |
| **Salinity (ECe) & Sodicity** | > 16.0 – 20.0 dS/m (SAR > 25) | Osmotic desiccation, severe cellular chlorosis & mortality | Truong (1994); Mumo et al. (2022); KeNHA (2019) | `salinity_ec_ds_m > 16.0 | sar > 25` |
| **Light / Solar Irradiance** | < 40% full sunlight (>60% shading) | Obligate C4 heliophyte etiolation and biomass collapse | Utomo et al. (2000); TaCRI (2017) | `canopy_shade_pct > 60` |
| **Continuous Submergence** | > 21 – 30 consecutive days fully drowned | Anoxic crown rot and tissue decay | Le & Truong (2004); LVBC (2021) | `submergence_days > 21` |
| **Minimum Temperature / Elevation** | Sustained < 10–12°C avg or Elevation > 2,200 m | Crown cellular frost kill / cold vegetative dormancy | Xia et al. (2003); NEMA Uganda (2020) | `elevation_m > 2200 | temp_c < 10` |
| **Extreme Soil Acidity / P-Fixation** | pH < 3.5 or Al saturation > 70% | Aluminum phytotoxicity and P-immobilization stunting | van Wyk et al. (2006); ISABU (2018) | `soil_ph < 3.5 | al_sat_pct > 70` |
| **Hydro-piping / Pumice Macro-pores** | Pumice bulk density < 1.15 g/cm³, high piping | Subterranean flume tunnel collapse bypassing root net | GDC Engineering Report (2021) | `pumice_content == 1 & piping_risk == 1` |
| **Hydraulic Shear (Juvenile)** | Flow velocity > 2.5–3.2 m/s before 6 months | Hydraulic uprooting and stream tractive shear stripping | Truong et al. (2002); KARI (2018) | `flow_velocity_ms > 2.5 & age_mo < 6` |
| **Shallow Bedrock / Petroferric Hardpan** | Solid rock / laterite hardpan < 0.2 m depth | Physical root refusal, zero mechanical tensile anchor | Liu et al. (2007); EIAR Field Report (2021) | `hardpan_depth_m < 0.2` |

---

## Engineering Decision Logic & Prescreening Guidelines

### 1. Geotechnical Slope Failure (Slip Surface Depth > 2.0 m)
- **Condition:** If rotational or translational shear failure planes are estimated at depths > 2.0 m.
- **Decision:** Do NOT rely on Vetiver as a standalone structural retaining system. Vetiver provides surficial veneer protection (0–2 m). Combine with driven soil nails (L ≥ 4 m) and sub-horizontal PVC weepholes.

### 2. High Soil Compaction (Bulk Density > 1.75 g/cm³)
- **Condition:** Soil bulk density exceeds 1.75 g/cm³ in over-compacted clays or unweathered saprolites.
- **Decision:** Mechanical ripping or augering along planting furrows to ≥60 cm is mandatory to shatter hardpans before planting slips.

### 3. Hypersaline / Sodic Environments (EC > 16.0 dS/m or SAR > 25)
- **Condition:** Electrical conductivity > 16 dS/m (e.g. mine tailings, coastal brine pools, trona pans).
- **Decision:** Apply agricultural gypsum (CaSO4·2H2O) at 5–12 t/ha to displace exchangeable sodium, flush with fresh water, and harden polybag slips prior to field transplanting.

### 4. Continuous Shading (Canopy Cover > 60%)
- **Condition:** Agroforestry or dense tree plantations where photosynthetically active radiation (PAR) drops < 40%.
- **Decision:** Enforce scheduled canopy pruning and orient hedgerows East-West.

### 5. Continuous Flooding / Leaf Crown Submergence (> 21 Days)
- **Condition:** River floodways or backflow swamps where leaf crowns are drowned beneath water for > 3 weeks.
- **Decision:** Construct elevated earthen levees or plant above mean seasonal flood crest so upper foliage remains aerated.

### 6. Cold / High Tropical Elevation (> 2,200 m or Mean Temp < 10°C)
- **Condition:** Highland zones with continuous temperatures < 12°C (e.g. Mt. Elgon, Aberdares, Lake Bunyonyi).
- **Decision:** Combine or replace Vetiver with cold-adapted native montane tussock grasses (e.g., *Pennisetum clandestinum*, *Festuca* spp.).

### 7. Hyper-acidic / High Aluminum Soils (pH < 3.5 or Al Saturation > 70%)
- **Condition:** Acid sulfate coastal soils or pyritic mining tailings.
- **Decision:** Incorporate agricultural lime (CaCO3) at 5–10 t/ha to raise pH > 4.5 and add rock phosphate (Minjingu PR at 200 kg/ha) or diammonium phosphate (DAP) to precipitate toxic free Al3+.

### 8. Collapsible Pumice / Hydro-piping Risk
- **Condition:** Porous volcanic pumice (e.g. Menengai / Rift Valley) with bulk density < 1.15 g/cm³ subject to flash wetting.
- **Decision:** Slopes require dynamic compaction or cohesive clay veneers prior to planting to prevent subterranean piping tunnels from bypassing roots.

### 9. Flash Flood Discharge Velocity (> 2.5 m/s in Juvenile Slips < 6 Months)
- **Condition:** Ephemeral gulley floors or active storm drain ditches.
- **Decision:** Secure channel floors with upstream dry-stone check dams or pinned biodegradable jute/coir geotextiles until crowns achieve 6 months basal maturity.

### 10. Shallow Petroferric Hardpans / Bedrock (< 0.2 m Depth)
- **Condition:** Indurated laterite petroferric pans or solid shallow bedrock mantles < 20 cm.
- **Decision:** Mechanical chiseling/pocket-augering to ≥50 cm is mandatory; otherwise divert hedgerows to colluvial soil pockets.