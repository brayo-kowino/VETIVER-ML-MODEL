"""
Geospatial Coordinate Resolver for Vetiver Bio-Engineering Engine.
Retrieves or infers USCS soil mechanical classification, climate/precipitation regime,
elevation, salinity, and hazards for arbitrary GPS coordinates (Lat, Lon).

Supports:
1. Live Open-Meteo & ISRIC SoilGrids REST APIs with timeout guardrails.
2. High-precision Offline Kenya & East Africa GIS Matrix for instant offline resolution.
3. Proximity matching against pre-computed regional hotspots (Haversine formula).
"""

import json
import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple
import httpx
import config

logger = logging.getLogger(__name__)


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance between two GPS coordinates in kilometers."""
    r = 6371.0  # Earth radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2.0) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2.0) ** 2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return r * c


def find_nearest_known_hotspot(lat: float, lon: float, max_distance_km: float = 35.0) -> Tuple[Optional[Dict[str, Any]], float]:
    """Check if given coordinates match or are adjacent to an existing pre-computed hotspot."""
    hotspot_path = os.path.join(config.DATA_DIR, "regional_hotspots.json")
    if not os.path.exists(hotspot_path):
        return None, 99999.0

    try:
        with open(hotspot_path, "r", encoding="utf-8") as f:
            hotspots = json.load(f)
    except Exception as exc:
        logger.error("Error reading regional hotspots: %s", exc)
        return None, 99999.0

    nearest_h = None
    min_dist = 99999.0
    for h in hotspots:
        h_lat = h.get("latitude")
        h_lon = h.get("longitude")
        if h_lat is not None and h_lon is not None:
            dist = haversine_distance_km(lat, lon, h_lat, h_lon)
            if dist < min_dist:
                min_dist = dist
                nearest_h = h

    if nearest_h and min_dist <= max_distance_km:
        return nearest_h, min_dist
    return nearest_h, min_dist


def fetch_live_elevation_and_climate(lat: float, lon: float) -> Optional[Dict[str, Any]]:
    """Query Open-Meteo elevation and climate APIs with a 2.5s timeout."""
    try:
        elev_url = f"https://api.open-meteo.com/v1/elevation?latitude={lat}&longitude={lon}"
        with httpx.Client(timeout=2.5) as client:
            resp_e = client.get(elev_url)
            elevation = None
            if resp_e.status_code == 200:
                e_data = resp_e.json()
                elevation = e_data.get("elevation", [None])[0]

            clim_url = (
                f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}"
                f"&start_date=2023-01-01&end_date=2023-12-31&daily=precipitation_sum,temperature_2m_mean"
            )
            resp_c = client.get(clim_url)
            annual_rain_mm = None
            mean_temp_c = None
            if resp_c.status_code == 200:
                c_data = resp_c.json()
                daily_p = c_data.get("daily", {}).get("precipitation_sum", [])
                daily_t = c_data.get("daily", {}).get("temperature_2m_mean", [])
                if daily_p:
                    annual_rain_mm = sum(p for p in daily_p if p is not None)
                if daily_t:
                    valid_t = [t for t in daily_t if t is not None]
                    if valid_t:
                        mean_temp_c = sum(valid_t) / len(valid_t)

            return {
                "elevation_m": elevation,
                "annual_rain_mm": annual_rain_mm,
                "mean_temp_c": mean_temp_c,
            }
    except Exception as exc:
        logger.debug("Live Open-Meteo query skipped or timed out: %s", exc)
        return None


def fetch_live_soilgrids(lat: float, lon: float) -> Optional[Dict[str, Any]]:
    """Query ISRIC SoilGrids REST API v2.0 for clay, sand, silt, and pH with a 2.5s timeout."""
    try:
        url = (
            f"https://rest.isric.org/soilgrids/v2.0/properties/query?"
            f"lat={lat}&lon={lon}&property=clay&property=sand&property=silt&property=phh2o&depth=0-5cm&depth=15-30cm"
        )
        with httpx.Client(timeout=2.5) as client:
            resp = client.get(url)
            if resp.status_code == 200:
                data = resp.json()
                layers = {lyr["name"]: lyr for lyr in data.get("properties", {}).get("layers", [])}
                clay_g_kg = None
                sand_g_kg = None
                ph_x10 = None
                if "clay" in layers and layers["clay"]["depths"]:
                    clay_g_kg = layers["clay"]["depths"][0]["values"].get("mean")
                if "sand" in layers and layers["sand"]["depths"]:
                    sand_g_kg = layers["sand"]["depths"][0]["values"].get("mean")
                if "phh2o" in layers and layers["phh2o"]["depths"]:
                    ph_x10 = layers["phh2o"]["depths"][0]["values"].get("mean")

                clay_pct = (clay_g_kg / 10.0) if clay_g_kg is not None else None
                sand_pct = (sand_g_kg / 10.0) if sand_g_kg is not None else None
                ph = (ph_x10 / 10.0) if ph_x10 is not None else None

                return {
                    "clay_pct": clay_pct,
                    "sand_pct": sand_pct,
                    "ph": ph,
                }
    except Exception as exc:
        logger.debug("Live SoilGrids query skipped or timed out: %s", exc)
        return None


def fetch_live_reverse_geocode(lat: float, lon: float) -> Optional[Dict[str, str]]:
    """Query OpenStreetMap Nominatim for county/state and country with a 2.5s timeout."""
    try:
        url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json"
        headers = {"User-Agent": "VetiverBioEngDecisionSupport/1.0"}
        with httpx.Client(timeout=2.5) as client:
            resp = client.get(url, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                addr = data.get("address", {})
                county = addr.get("county") or addr.get("state_district") or addr.get("state") or "Unknown County"
                state = addr.get("state") or addr.get("region") or county
                country = addr.get("country") or "Unknown"
                return {"county": county, "region": state, "country": country}
    except Exception as exc:
        logger.debug("Live Nominatim reverse geocode skipped: %s", exc)
        return None


def search_location(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Forward-geocode a place name through Open-Meteo with Nominatim fallback."""
    query = str(query).strip()
    if not query:
        return []

    try:
        headers = {"User-Agent": "VetiverBioEngDecisionSupport/1.0"}
        with httpx.Client(timeout=4.0) as client:
            open_meteo_url = "https://geocoding-api.open-meteo.com/v1/search"
            open_meteo_params = {
                "name": query,
                "count": limit,
                "language": "en",
                "format": "json",
            }
            resp = client.get(open_meteo_url, params=open_meteo_params, headers=headers)
            if resp.status_code == 200:
                results = []
                for item in resp.json().get("results", []):
                    try:
                        name_parts = [
                            item.get("name"),
                            item.get("admin1"),
                            item.get("country"),
                        ]
                        display_name = ", ".join(part for part in name_parts if part)
                        results.append({
                            "display_name": display_name,
                            "latitude": float(item["latitude"]),
                            "longitude": float(item["longitude"]),
                            "type": item.get("feature_code"),
                        })
                    except (KeyError, TypeError, ValueError):
                        continue
                if results:
                    return results

            nominatim_url = "https://nominatim.openstreetmap.org/search"
            nominatim_params = {
                "q": query,
                "format": "jsonv2",
                "addressdetails": 1,
                "limit": limit,
            }
            resp = client.get(nominatim_url, params=nominatim_params, headers=headers)
            if resp.status_code != 200:
                return []

            results = []
            for item in resp.json():
                try:
                    results.append({
                        "display_name": item.get("display_name", query),
                        "latitude": float(item["lat"]),
                        "longitude": float(item["lon"]),
                        "type": item.get("type"),
                    })
                except (KeyError, TypeError, ValueError):
                    continue
            return results
    except Exception as exc:
        logger.debug("Nominatim location search skipped: %s", exc)
        return []


def resolve_offline_gis_matrix(lat: float, lon: float) -> Dict[str, Any]:
    """
    Deterministic Offline Geospatial Knowledge Matrix for Kenya & East Africa.
    Derives USCS soil classification, precipitation envelope, elevation, salinity,
    and geotechnical hazards from latitude/longitude bounding domains.
    """
    # 1. Check if coordinate falls within Kenya bounding box (approx -4.7 to +5.0 lat, 33.9 to 41.9 lon)
    is_kenya = (-4.7 <= lat <= 5.0) and (33.9 <= lon <= 41.9)
    country = "Kenya" if is_kenya else ("Ethiopia" if (3.0 <= lat <= 15.0 and 33.0 <= lon <= 48.0) else "East Africa / Global")

    # Domain 1: Lake Magadi & Southern Rift Sodic/Trona Basin
    if -2.2 <= lat <= -1.6 and 36.0 <= lon <= 36.4:
        return {
            "county": "Kajiado County",
            "region": "Southern Rift Valley (Lake Magadi Basin)",
            "country": "Kenya",
            "elevation_m": 610,
            "soil_mechanical_class": "Sandy Clay / Loam",
            "climate_rainfall_zone": "Semi-Arid (300–700 mm/yr)",
            "establishment_moisture": "Severe Drought / Zero Watering",
            "hydrological_stress": "Normal Slope / Well-Drained",
            "salinity_tier": "Extreme Hypersaline (> 16 dS/m)",
            "est_salinity_ec_ds_m": 22.0,
            "default_hazard": "Surficial Sheet & Rill Erosion (< 0.3 m)",
            "notes": "Hyper-saline trona alkaline crust zone. Reverse osmotic desiccation boundary.",
        }

    # Domain 1b: Nairobi Metropolitan Core & Upper Uplands
    if -1.40 <= lat <= -1.15 and 36.65 <= lon <= 37.10:
        return {
            "county": "Nairobi County",
            "region": "Nairobi Metropolitan & Central Uplands",
            "country": "Kenya",
            "elevation_m": 1680,
            "soil_mechanical_class": "Cohesive Clay / Nitisol / Laterite",
            "climate_rainfall_zone": "Sub-Humid / Bimodal (700–1400 mm/yr)",
            "establishment_moisture": "Adequate (Irrigated / In-Season Rain)",
            "hydrological_stress": "Normal Slope / Well-Drained",
            "salinity_tier": "Non-Saline (< 2 dS/m)",
            "est_salinity_ec_ds_m": 0.4,
            "default_hazard": "Shallow Slope / Embankment Slip (0.3–1.5 m)",
            "notes": "Nairobi volcanic red clay (Nitisol) over trachytic tuffs. Highly suited for road cuts & slope stabilization.",
        }

    # Domain 1c: Athi River, Kitengela & Kapiti Plains (Expansive Vertisol / Black Cotton Basin)
    if -1.65 <= lat <= -1.35 and 36.85 <= lon <= 37.35:
        return {
            "county": "Machakos / Kajiado County",
            "region": "Kapiti Plains & Athi River Industrial Corridor",
            "country": "Kenya",
            "elevation_m": 1520,
            "soil_mechanical_class": "Sandy Clay / Loam",
            "climate_rainfall_zone": "Semi-Arid (300–700 mm/yr)",
            "establishment_moisture": "Adequate (Irrigated / In-Season Rain)",
            "hydrological_stress": "Normal Slope / Well-Drained",
            "salinity_tier": "Non-Saline (< 2 dS/m)",
            "est_salinity_ec_ds_m": 0.8,
            "default_hazard": "Surficial Sheet & Rill Erosion (< 0.3 m)",
            "notes": "Expansive smectite clay (Black Cotton Soil) prone to seasonal shrink-swell cracks and road shoulder slumping.",
        }

    # Domain 2: Rift Valley Pumice Axis (Menengai, Nakuru, Naivasha, Longonot, Gilgil)
    if -0.9 <= lat <= 0.4 and 35.8 <= lon <= 36.5:
        return {
            "county": "Nakuru County",
            "region": "Central Rift Valley Pumice Corridor",
            "country": "Kenya",
            "elevation_m": 1890,
            "soil_mechanical_class": "Coarse Sand / Loose Pumice",
            "climate_rainfall_zone": "Sub-Humid / Bimodal (700–1400 mm/yr)",
            "establishment_moisture": "Adequate (Irrigated / In-Season Rain)",
            "hydrological_stress": "Normal Slope / Well-Drained",
            "salinity_tier": "Non-Saline (< 2 dS/m)",
            "est_salinity_ec_ds_m": 0.6,
            "default_hazard": "Shallow Slope / Embankment Slip (0.3–1.5 m)",
            "notes": "Volcanic pumice lapilli & porous tuff. High subterranean piping vulnerability.",
        }

    # Domain 2b: Western Mau Escarpment & Kericho Tea Belt
    if -0.6 <= lat <= 0.0 and 35.1 <= lon <= 35.75:
        return {
            "county": "Kericho / Bomet County",
            "region": "Western Mau Escarpment & Tea Highlands",
            "country": "Kenya",
            "elevation_m": 2050,
            "soil_mechanical_class": "Cohesive Clay / Nitisol / Laterite",
            "climate_rainfall_zone": "Humid / Tropical Monsoon (>1400 mm/yr)",
            "establishment_moisture": "Adequate (Irrigated / In-Season Rain)",
            "hydrological_stress": "Normal Slope / Well-Drained",
            "salinity_tier": "Non-Saline (< 2 dS/m)",
            "est_salinity_ec_ds_m": 0.3,
            "default_hazard": "Shallow Slope / Embankment Slip (0.3–1.5 m)",
            "notes": "Steep humid lateritic cut-slopes with annual rainfall >1,600 mm. Exceptional root anchoring cohesion.",
        }

    # Domain 3b: North Rift Plateau & Highland Grain Belt (Eldoret, Kitale, Uasin Gishu, Trans-Nzoia)
    if 0.2 <= lat <= 1.4 and 34.8 <= lon <= 35.6:
        return {
            "county": "Uasin Gishu / Trans Nzoia County",
            "region": "North Rift Highland Plateau (Eldoret / Cherang'any)",
            "country": "Kenya",
            "elevation_m": 2080,
            "soil_mechanical_class": "Sandy Clay / Loam",
            "climate_rainfall_zone": "Sub-Humid / Bimodal (700–1400 mm/yr)",
            "establishment_moisture": "Adequate (Irrigated / In-Season Rain)",
            "hydrological_stress": "Normal Slope / Well-Drained",
            "salinity_tier": "Non-Saline (< 2 dS/m)",
            "est_salinity_ec_ds_m": 0.4,
            "default_hazard": "Surficial Sheet & Rill Erosion (< 0.3 m)",
            "notes": "Fertile highland plateau soils. Moderate slope erosion on agricultural contour catchments.",
        }

    # Domain 3: Central Highlands & Aberdares Montane Slopes (Murang'a, Nyeri, Kiambu, Kirinyaga, Embu)
    if -1.4 <= lat <= -0.1 and 36.5 <= lon <= 37.6:
        # High altitude cold arrest boundary
        if 36.5 <= lon <= 36.8 and -0.7 <= lat <= -0.2:
            return {
                "county": "Nyandarua County",
                "region": "Aberdare Montane Forest Crest",
                "country": "Kenya",
                "elevation_m": 2550,
                "soil_mechanical_class": "Cohesive Clay / Nitisol / Laterite",
                "climate_rainfall_zone": "Alpine High-Altitude (>2,200 m)",
                "establishment_moisture": "Adequate (Irrigated / In-Season Rain)",
                "hydrological_stress": "Normal Slope / Well-Drained",
                "salinity_tier": "Non-Saline (< 2 dS/m)",
                "est_salinity_ec_ds_m": 0.3,
                "default_hazard": "Shallow Slope / Embankment Slip (0.3–1.5 m)",
                "notes": "Highland montane zone > 2,200 m a.s.l. Photosynthetic C4 cold arrest risk.",
            }
        return {
            "county": "Murang'a / Nyeri / Kiambu County",
            "region": "Central Highlands (Mount Kenya Southern Batters)",
            "country": "Kenya",
            "elevation_m": 1650,
            "soil_mechanical_class": "Cohesive Clay / Nitisol / Laterite",
            "climate_rainfall_zone": "Humid / Tropical Monsoon (>1400 mm/yr)",
            "establishment_moisture": "Adequate (Irrigated / In-Season Rain)",
            "hydrological_stress": "Normal Slope / Well-Drained",
            "salinity_tier": "Non-Saline (< 2 dS/m)",
            "est_salinity_ec_ds_m": 0.4,
            "default_hazard": "Shallow Slope / Embankment Slip (0.3–1.5 m)",
            "notes": "Deep lateritic red clay (Nitisol). Outstanding root anchoring matrix.",
        }

    # Domain 4: Lake Victoria Riparian Basin (Kisumu, Siaya, Homa Bay, Migori, Nyando)
    if -1.1 <= lat <= 0.3 and 34.0 <= lon <= 35.1:
        return {
            "county": "Kisumu County",
            "region": "Lake Victoria Winam Gulf Floodplain",
            "country": "Kenya",
            "elevation_m": 1145,
            "soil_mechanical_class": "Sandy Clay / Loam",
            "climate_rainfall_zone": "Humid / Tropical Monsoon (>1400 mm/yr)",
            "establishment_moisture": "Adequate (Irrigated / In-Season Rain)",
            "hydrological_stress": "High-Velocity Hydraulic Runoff (>2.5 m/s)",
            "salinity_tier": "Non-Saline (< 2 dS/m)",
            "est_salinity_ec_ds_m": 0.5,
            "default_hazard": "Riverbank / Hydrodynamic Toe Scour",
            "notes": "Fluvial deposition zone. High tractive shear and seasonal flood pulses.",
        }

    # Domain 5: Coastal Marine Belt (Mombasa, Kilifi, Kwale, Malindi)
    if -4.7 <= lat <= -2.0 and 39.2 <= lon <= 41.0:
        return {
            "county": "Mombasa / Kilifi County",
            "region": "Indian Ocean Coastal Littoral Strip",
            "country": "Kenya",
            "elevation_m": 25,
            "soil_mechanical_class": "Coarse Sand / Loose Pumice",
            "climate_rainfall_zone": "Humid / Tropical Monsoon (>1400 mm/yr)",
            "establishment_moisture": "Adequate (Irrigated / In-Season Rain)",
            "hydrological_stress": "Normal Slope / Well-Drained",
            "salinity_tier": "Slight to Moderate (2–8 dS/m)",
            "est_salinity_ec_ds_m": 3.8,
            "default_hazard": "Surficial Sheet & Rill Erosion (< 0.3 m)",
            "notes": "Coastal marine sand dunes and coral rag. Moderate sea spray salinity.",
        }

    # Domain 6: Semi-Arid Eastern Drylands (Machakos, Kitui, Makueni, Taita Taveta, Voi)
    if -3.9 <= lat <= -0.8 and 37.0 <= lon <= 38.9:
        return {
            "county": "Machakos / Taita Taveta County",
            "region": "Eastern Agro-Pastoral Plateau",
            "country": "Kenya",
            "elevation_m": 1100,
            "soil_mechanical_class": "Sandy Clay / Loam",
            "climate_rainfall_zone": "Semi-Arid (300–700 mm/yr)",
            "establishment_moisture": "Adequate (Irrigated / In-Season Rain)",
            "hydrological_stress": "Normal Slope / Well-Drained",
            "salinity_tier": "Non-Saline (< 2 dS/m)",
            "est_salinity_ec_ds_m": 0.8,
            "default_hazard": "Surficial Sheet & Rill Erosion (< 0.3 m)",
            "notes": "Basement complex soils subject to intense seasonal sheet wash.",
        }

    # Domain 6b: Laikipia Plateau & Samburu Central Uplands (Maralal, Rumuruti, Nanyuki)
    if 0.1 <= lat <= 1.6 and 36.4 <= lon <= 37.5:
        return {
            "county": "Samburu / Laikipia County",
            "region": "Central-North Agro-Pastoral Plateau (Maralal / Laikipia)",
            "country": "Kenya",
            "elevation_m": 1850,
            "soil_mechanical_class": "Sandy Clay / Loam",
            "climate_rainfall_zone": "Sub-Humid / Bimodal (700–1400 mm/yr)",
            "establishment_moisture": "Adequate (Irrigated / In-Season Rain)",
            "hydrological_stress": "Normal Slope / Well-Drained",
            "salinity_tier": "Non-Saline (< 2 dS/m)",
            "est_salinity_ec_ds_m": 0.5,
            "default_hazard": "Shallow Slope / Embankment Slip (0.3–1.5 m)",
            "notes": "Highland plateau plateau soils over metamorphic basement. Suitable for slope stabilization with seasonal rain.",
        }

    # Domain 6c: Eastern Lowlands & Garissa Basin (Garissa, Tana River, Kitui East)
    if -2.2 <= lat <= 1.6 and 38.6 <= lon <= 41.5:
        return {
            "county": "Garissa / Tana River County",
            "region": "Garissa Arid Lowlands & Tana River Valley",
            "country": "Kenya",
            "elevation_m": 220,
            "soil_mechanical_class": "Coarse Sand / Loose Pumice",
            "climate_rainfall_zone": "Hyper-Arid / Desert (<300 mm/yr)",
            "establishment_moisture": "Severe Drought / Zero Watering",
            "hydrological_stress": "Normal Slope / Well-Drained",
            "salinity_tier": "Slight to Moderate (2–8 dS/m)",
            "est_salinity_ec_ds_m": 3.5,
            "default_hazard": "Surficial Sheet & Rill Erosion (< 0.3 m)",
            "notes": "Arid alluvial sands & seasonal gullying along northern pastoral road cuts. 60-day establishment irrigation mandatory.",
        }

    # Domain 7: Northern Deserts & Lowland Arid Basins (Turkana, Marsabit Chalbi, Wajir Lowlands)
    if 1.6 < lat <= 4.8 and 35.0 <= lon <= 41.5:
        return {
            "county": "Turkana / Marsabit County",
            "region": "Northern Arid Lowlands (Chalbi / Kerio)",
            "country": "Kenya",
            "elevation_m": 480,
            "soil_mechanical_class": "Coarse Sand / Loose Pumice",
            "climate_rainfall_zone": "Hyper-Arid / Desert (<300 mm/yr)",
            "establishment_moisture": "Severe Drought / Zero Watering",
            "hydrological_stress": "Normal Slope / Well-Drained",
            "salinity_tier": "High Saline (8–16 dS/m)",
            "est_salinity_ec_ds_m": 9.5,
            "default_hazard": "Surficial Sheet & Rill Erosion (< 0.3 m)",
            "notes": "Hyper-arid moisture deficit. Auxiliary irrigation required for first 60 days.",
        }

    # Domain 8: Ethiopian Highlands & Horn of Africa
    if 5.0 <= lat <= 14.0 and 35.0 <= lon <= 41.0:
        return {
            "county": "Oromia / Amhara Region",
            "region": "Ethiopian Volcanic Trapp Highlands",
            "country": "Ethiopia",
            "elevation_m": 1850,
            "soil_mechanical_class": "Cohesive Clay / Nitisol / Laterite",
            "climate_rainfall_zone": "Humid / Tropical Monsoon (>1400 mm/yr)",
            "establishment_moisture": "Adequate (Irrigated / In-Season Rain)",
            "hydrological_stress": "Normal Slope / Well-Drained",
            "salinity_tier": "Non-Saline (< 2 dS/m)",
            "est_salinity_ec_ds_m": 0.4,
            "default_hazard": "Shallow Slope / Embankment Slip (0.3–1.5 m)",
            "notes": "Deep volcanic vertisol/nitisol slopes with intense agricultural gully erosion.",
        }

    # Generic Global Tropical / Subtropical Fallback
    abs_lat = abs(lat)
    if abs_lat <= 23.5:
        clim = "Humid / Tropical Monsoon (>1400 mm/yr)" if abs_lat < 10 else "Sub-Humid / Bimodal (700–1400 mm/yr)"
    elif abs_lat <= 35.0:
        clim = "Semi-Arid (300–700 mm/yr)"
    else:
        clim = "Cold Freezing / Permafrost"

    return {
        "county": f"Regional Site ({lat:.2f}°, {lon:.2f}°)",
        "region": "Global Civil Engineering Sector",
        "country": country,
        "elevation_m": 950,
        "soil_mechanical_class": "Sandy Clay / Loam",
        "climate_rainfall_zone": clim,
        "establishment_moisture": "Adequate (Irrigated / In-Season Rain)",
        "hydrological_stress": "Normal Slope / Well-Drained",
        "salinity_tier": "Non-Saline (< 2 dS/m)",
        "est_salinity_ec_ds_m": 0.5,
        "default_hazard": "Shallow Slope / Embankment Slip (0.3–1.5 m)",
        "notes": "Inferred from global latitudinal geoclimatic envelopes.",
    }


try:
    import streamlit as st
    _cache_dec = st.cache_data(ttl=3600, show_spinner=False)
except Exception:
    def _cache_dec(f):
        return f


@_cache_dec
def resolve_coordinate_geotechnics(lat: float, lon: float) -> Dict[str, Any]:
    """
    Main pipeline to resolve arbitrary GPS coordinates:
    1. Check proximity to known pre-computed hotspots (within 25 km).
    2. Attempt live queries (Open-Meteo, SoilGrids, OSM) with graceful timeout.
    3. Fallback to deterministic Kenya/East Africa Geospatial Knowledge Matrix.
    4. Format ready-to-run dictionary for CatBoost and biomechanical models.
    """
    # 1. Proximity to known benchmark hotspots
    nearest_hotspot, dist_km = find_nearest_known_hotspot(lat, lon, max_distance_km=25.0)
    is_known = nearest_hotspot is not None and dist_km <= 25.0

    # 2. Offline GIS Baseline
    base = resolve_offline_gis_matrix(lat, lon)

    # 3. Live Environmental Providers (Best effort)
    live_clim = fetch_live_elevation_and_climate(lat, lon)
    live_soil = fetch_live_soilgrids(lat, lon)
    live_geo = fetch_live_reverse_geocode(lat, lon)

    data_sources = []

    # Integrate Reverse Geocoding
    if live_geo and live_geo.get("county") != "Unknown County":
        base["county"] = live_geo["county"]
        base["region"] = live_geo.get("region", base["region"])
        base["country"] = live_geo.get("country", base["country"])
        data_sources.append("OSM Reverse Geocode")
    else:
        data_sources.append("Kenya GIS Regional Atlas")

    # Integrate Elevation & Climate
    if live_clim:
        if live_clim.get("elevation_m") is not None:
            base["elevation_m"] = int(round(live_clim["elevation_m"]))
            data_sources.append("Open-Meteo Elevation")
            if base["elevation_m"] > 2200:
                base["climate_rainfall_zone"] = "Alpine High-Altitude (>2,200 m)"

        rain_mm = live_clim.get("annual_rain_mm")
        if rain_mm is not None:
            data_sources.append("Open-Meteo Precipitation (2023 Archive)")
            if base["elevation_m"] <= 2200:
                if rain_mm > 1400:
                    base["climate_rainfall_zone"] = "Humid / Tropical Monsoon (>1400 mm/yr)"
                elif rain_mm >= 700:
                    base["climate_rainfall_zone"] = "Sub-Humid / Bimodal (700–1400 mm/yr)"
                elif rain_mm >= 300:
                    base["climate_rainfall_zone"] = "Semi-Arid (300–700 mm/yr)"
                else:
                    base["climate_rainfall_zone"] = "Hyper-Arid / Desert (<300 mm/yr)"
                    base["establishment_moisture"] = "Severe Drought / Zero Watering"

    # Integrate SoilGrids
    if live_soil:
        clay = live_soil.get("clay_pct")
        sand = live_soil.get("sand_pct")
        ph = live_soil.get("ph")
        if clay is not None and sand is not None:
            data_sources.append(f"ISRIC SoilGrids v2 (Clay: {clay:.1f}%, Sand: {sand:.1f}%)")
            if sand >= 65.0:
                base["soil_mechanical_class"] = "Coarse Sand / Loose Pumice"
            elif clay >= 38.0:
                base["soil_mechanical_class"] = "Cohesive Clay / Nitisol / Laterite"
            else:
                base["soil_mechanical_class"] = "Sandy Clay / Loam"
        if ph is not None and ph < 4.0:
            base["notes"] += f" Acidic substrate detected (pH {ph:.1f})."

    resolved_result = {
        "latitude": round(lat, 5),
        "longitude": round(lon, 5),
        "county": base["county"],
        "region": base["region"],
        "country": base["country"],
        "elevation_m": base["elevation_m"],
        "soil_mechanical_class": base["soil_mechanical_class"],
        "climate_rainfall_zone": base["climate_rainfall_zone"],
        "establishment_moisture": base["establishment_moisture"],
        "hydrological_stress": base["hydrological_stress"],
        "salinity_tier": base["salinity_tier"],
        "est_salinity_ec_ds_m": base["est_salinity_ec_ds_m"],
        "default_hazard": base["default_hazard"],
        "notes": base["notes"],
        "data_sources": " · ".join(data_sources),
        "is_known_hotspot": is_known,
        "matched_hotspot": nearest_hotspot if is_known else None,
        "distance_to_nearest_km": round(dist_km, 1),
    }

    return resolved_result
