"""
Core Energy Computation Engine.
Handles machine learning inference, 24-hour diurnal profile simulation,
solar PV net-metering, carbon footprint accounting, and efficiency star ratings.
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from config import APPLIANCE_POWER_KW, CARBON_FACTORS

@st.cache_resource(show_spinner=False)
def load_energy_model(model_path: str):
    """Loads the trained Random Forest Regressor."""
    if not os.path.exists(model_path):
        return None
    try:
        return joblib.load(model_path)
    except Exception:
        return None

def simulate_solar_curve(solar_kw: float) -> np.ndarray:
    """
    Generates a 24-hour solar generation profile (kWh per hour)
    using sun angle trigonometric distribution between 6:00 and 18:00.
    """
    solar_hourly = np.zeros(24)
    if solar_kw <= 0:
        return solar_hourly

    # Solar generation hours: 06:00 to 18:00 (peak around 12:00-13:00)
    for h in range(6, 18):
        # Sine model accounting for solar insolation angle and derating factor (~0.85)
        angle = (h - 6) * np.pi / 12
        solar_hourly[h] = solar_kw * 0.85 * (np.sin(angle) ** 1.6)

    return np.maximum(0.0, solar_hourly)

def compute_equipment_load(building_type_idx: int, appliances: dict, ambient_temp: float) -> tuple:
    """
    Computes active equipment and appliance consumption based on building type,
    ambient temperature (cooling demand), and typical operating hours.
    Returns (breakdown_dict, total_daily_kwh, hourly_distribution_array).
    """
    breakdown = {}
    hourly_distribution = np.zeros(24)

    if building_type_idx == 0:  # House
        ac_hours = 8.0 if ambient_temp > 28 else (4.0 if ambient_temp > 22 else 1.0)
        fan_kwh = appliances.get("fan", 0) * APPLIANCE_POWER_KW["residential"]["fan"] * 10.0
        ac_kwh = appliances.get("ac", 0) * APPLIANCE_POWER_KW["residential"]["ac"] * ac_hours
        tv_kwh = appliances.get("tv", 0) * APPLIANCE_POWER_KW["residential"]["tv"] * 4.0
        fridge_kwh = appliances.get("fridge", 0) * APPLIANCE_POWER_KW["residential"]["fridge"] * 24.0
        light_kwh = appliances.get("light", 0) * APPLIANCE_POWER_KW["residential"]["light"] * 6.0
        wm_kwh = appliances.get("washing_machine", 0) * APPLIANCE_POWER_KW["residential"]["washing_machine"] * 1.0

        breakdown = {
            "Air Conditioning": ac_kwh,
            "Fans": fan_kwh,
            "Refrigeration": fridge_kwh,
            "Lighting": light_kwh,
            "TV & Appliances": tv_kwh + wm_kwh
        }
        total_kwh = sum(breakdown.values())
        for h in range(24):
            # Diurnal residential pattern: morning boost (6-9) and evening peak (18-23)
            weight = 0.03 if h < 6 else (0.05 if h < 18 else 0.075)
            hourly_distribution[h] = total_kwh * weight

    elif building_type_idx == 1:  # Office
        hvac_hours = 9.0 if ambient_temp > 24 else 5.0
        pc_kwh = appliances.get("workstations", 0) * APPLIANCE_POWER_KW["office"]["workstation"] * 8.5
        ac_kwh = appliances.get("ac_units", 0) * APPLIANCE_POWER_KW["office"]["ac_unit"] * hvac_hours
        light_kwh = appliances.get("office_lights", 0) * APPLIANCE_POWER_KW["office"]["office_light"] * 9.0
        server_kwh = appliances.get("server_racks", 0) * APPLIANCE_POWER_KW["office"]["server_rack"] * 24.0
        pantry_kwh = appliances.get("pantry", 0) * APPLIANCE_POWER_KW["office"]["pantry"] * 3.0

        breakdown = {
            "HVAC / Cooling": ac_kwh,
            "IT & Workstations": pc_kwh,
            "Office Lighting": light_kwh,
            "Continuous Servers": server_kwh,
            "Pantry & Others": pantry_kwh
        }
        total_kwh = sum(breakdown.values())
        for h in range(24):
            weight = 0.08 if 9 <= h <= 18 else 0.012
            hourly_distribution[h] = total_kwh * weight

    else:  # Mall
        chiller_hours = 12.0 if ambient_temp > 25 else 8.0
        chiller_kwh = appliances.get("chillers", 0) * APPLIANCE_POWER_KW["mall"]["chiller"] * chiller_hours
        light_kwh = appliances.get("retail_lights", 0) * APPLIANCE_POWER_KW["mall"]["retail_light"] * 12.0
        esc_kwh = appliances.get("escalators", 0) * APPLIANCE_POWER_KW["mall"]["escalator"] * 12.0
        food_kwh = appliances.get("food_court", 0) * APPLIANCE_POWER_KW["mall"]["food_court"] * 14.0

        breakdown = {
            "Central Chillers": chiller_kwh,
            "Retail Lighting": light_kwh,
            "Escalators & Elevators": esc_kwh,
            "Food Court Freezers": food_kwh
        }
        total_kwh = sum(breakdown.values())
        for h in range(24):
            weight = 0.075 if 10 <= h <= 22 else 0.01
            hourly_distribution[h] = total_kwh * weight

    return breakdown, total_kwh, hourly_distribution

def run_energy_simulation(
    model,
    temp: float,
    humidity: float,
    day_idx: int,
    building_type_idx: int,
    occupancy: int,
    area: float,
    appliances: dict,
    solar_kw: float,
    tariff_flat: float,
    use_tou: bool,
    carbon_region: str
) -> dict:
    """
    Executes the comprehensive 24-hour building energy audit simulation.
    """
    hours = np.arange(24)

    # 1. Structural base model prediction
    if model is not None:
        batch_inputs = pd.DataFrame([{
            'temperature': temp + (3.0 * np.sin((h - 9) * np.pi / 12) if 8 <= h <= 18 else -2.0),
            'humidity': humidity,
            'hour': h,
            'day': day_idx,
            'building_type': building_type_idx,
            'occupancy': occupancy,
            'area': area
        } for h in hours])
        structural_kwh = np.maximum(0.1, model.predict(batch_inputs))
    else:
        structural_kwh = np.full(24, max(0.5, area * 0.001))

    # 2. Active equipment loads
    breakdown, eq_daily_kwh, eq_hourly = compute_equipment_load(building_type_idx, appliances, temp)

    # 3. Total gross electrical demand (kW / kWh per hour)
    gross_hourly = structural_kwh + eq_hourly
    daily_gross_kwh = float(np.sum(gross_hourly))

    # 4. Solar PV generation and Net Metering
    solar_hourly = simulate_solar_curve(solar_kw)
    daily_solar_kwh = float(np.sum(solar_hourly))

    # Net grid import and export
    grid_import_hourly = np.maximum(0.0, gross_hourly - solar_hourly)
    solar_export_hourly = np.maximum(0.0, solar_hourly - gross_hourly)
    daily_net_import_kwh = float(np.sum(grid_import_hourly))
    daily_export_kwh = float(np.sum(solar_export_hourly))

    # 5. Billing & Financials (Flat vs. Time-of-Use)
    # Standard Time-of-Use schedule:
    # Peak (18:00 - 22:00): +30% surcharge
    # Off-Peak (23:00 - 06:00): -20% discount
    # Normal: base tariff
    daily_cost = 0.0
    for h in range(24):
        rate = tariff_flat
        if use_tou:
            if 18 <= h <= 22:
                rate = tariff_flat * 1.30
            elif h >= 23 or h <= 6:
                rate = tariff_flat * 0.80
        daily_cost += (grid_import_hourly[h] * rate) - (solar_export_hourly[h] * (tariff_flat * 0.65))

    daily_cost = max(0.0, daily_cost)
    monthly_cost = daily_cost * 30.0
    monthly_gross_kwh = daily_gross_kwh * 30.0
    monthly_net_import_kwh = daily_net_import_kwh * 30.0
    monthly_solar_kwh = daily_solar_kwh * 30.0

    # Without solar baseline cost for ROI comparison
    baseline_daily_cost = daily_gross_kwh * tariff_flat
    monthly_savings = max(0.0, (baseline_daily_cost * 30.0) - monthly_cost)

    # 6. Energy Use Intensity (EUI) & Star Rating
    # EUI in kWh / sq ft / year
    annual_kwh = daily_gross_kwh * 365.0
    eui = annual_kwh / max(1.0, area)

    # Benchmarking star rating (Based on BEE / Energy Star norms)
    if building_type_idx == 0:   # Residential
        if eui <= 4.0: stars, rating_text = 5, "5-Star (Exceptional Efficiency)"
        elif eui <= 6.5: stars, rating_text = 4, "4-Star (High Efficiency)"
        elif eui <= 9.0: stars, rating_text = 3, "3-Star (Standard Average)"
        elif eui <= 12.0: stars, rating_text = 2, "2-Star (Below Average)"
        else: stars, rating_text = 1, "1-Star (High Energy Consumer)"
    elif building_type_idx == 1: # Office
        if eui <= 18.0: stars, rating_text = 5, "5-Star (Green Building Standard)"
        elif eui <= 28.0: stars, rating_text = 4, "4-Star (Well Managed)"
        elif eui <= 40.0: stars, rating_text = 3, "3-Star (Typical Commercial)"
        elif eui <= 55.0: stars, rating_text = 2, "2-Star (High Basal Load)"
        else: stars, rating_text = 1, "1-Star (Energy Inefficient)"
    else:                        # Mall / Retail
        if eui <= 45.0: stars, rating_text = 5, "5-Star (Top Tier Facility)"
        elif eui <= 70.0: stars, rating_text = 4, "4-Star (Optimized Chillers)"
        elif eui <= 95.0: stars, rating_text = 3, "3-Star (Standard Retail)"
        elif eui <= 125.0: stars, rating_text = 2, "2-Star (Cooling Losses)"
        else: stars, rating_text = 1, "1-Star (Severe HVAC Inefficiency)"

    # 7. Carbon Accounting
    carbon_intensity = CARBON_FACTORS.get(carbon_region, 0.45)
    annual_co2_kg = annual_kwh * carbon_intensity
    annual_co2_tons = annual_co2_kg / 1000.0
    # EPA metric: 1 mature tree sequesters ~21.77 kg CO2/year
    trees_offset = int(annual_co2_kg / 21.77)

    # 8. Actionable recommendations
    recommendations = []
    if temp > 28:
        cooling_save_kwh = daily_gross_kwh * 0.12 * 30.0
        recommendations.append({
            "title": "🌡 Calibrate Thermostats to 24°C - 26°C",
            "desc": f"Bumping thermostat setpoint up by 2°C cuts thermal load by 12-18%, saving approx. {cooling_save_kwh:.0f} kWh (${cooling_save_kwh * tariff_flat:,.0f}) every month.",
            "impact": "High"
        })

    if solar_kw <= 0:
        recommendations.append({
            "title": "☀️ Rooftop Solar Feasibility",
            "desc": f"Installing a {min(15.0, round(area * 0.003, 1))} kW solar PV system could eliminate up to {min(85, int((15 * 4.5 * 30 / monthly_gross_kwh) * 100))}% of your grid electricity import.",
            "impact": "High"
        })
    else:
        recommendations.append({
            "title": "⚡ Maximize Solar Self-Consumption",
            "desc": "Run dishwashers, washing machines, and thermal pre-cooling between 11:00 AM and 3:00 PM to capture 100% of peak solar generation behind the meter.",
            "impact": "Medium"
        })

    if use_tou:
        recommendations.append({
            "title": "⏰ Shift Flexible Loads to Night Valley",
            "desc": "Off-peak tariff applies between 11 PM and 6 AM. Scheduling EV charging or laundry during these hours saves 20% on tariff rates.",
            "impact": "Medium"
        })

    recommendations.append({
        "title": "🔌 Standby Power Mitigation",
        "desc": "Smart power strips reduce vampire draw from entertainment setups and IT peripherals, cutting baseline idle power by 6-9%.",
        "impact": "Low"
    })

    # Hourly Profile DataFrame
    hourly_df = pd.DataFrame({
        "Hour": [f"{h:02d}:00" for h in hours],
        "Gross Demand (kW)": np.round(gross_hourly, 2),
        "Solar Generation (kW)": np.round(solar_hourly, 2),
        "Net Grid Import (kW)": np.round(grid_import_hourly, 2),
        "Surplus Solar Export (kW)": np.round(solar_export_hourly, 2)
    })

    return {
        "daily_gross_kwh": daily_gross_kwh,
        "monthly_gross_kwh": monthly_gross_kwh,
        "daily_solar_kwh": daily_solar_kwh,
        "monthly_solar_kwh": monthly_solar_kwh,
        "daily_net_import_kwh": daily_net_import_kwh,
        "monthly_net_import_kwh": monthly_net_import_kwh,
        "daily_cost": daily_cost,
        "monthly_cost": monthly_cost,
        "monthly_savings": monthly_savings,
        "eui": eui,
        "stars": stars,
        "rating_text": rating_text,
        "annual_co2_tons": annual_co2_tons,
        "trees_offset": trees_offset,
        "equipment_breakdown": breakdown,
        "recommendations": recommendations,
        "hourly_df": hourly_df
    }
