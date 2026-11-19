"""
Application Configuration and Energy Constants.
Defines appliance power ratings, quick presets, regional grid carbon factors,
and building efficiency benchmarks.
"""

# Default open meteorology API endpoint
DEFAULT_CITY = "Delhi"

# Appliance power consumption ratings in Kilowatts (kW)
APPLIANCE_POWER_KW = {
    "residential": {
        "fan": 0.075,            # Standard BLDC / induction ceiling fan (75W)
        "ac": 1.500,             # 1.5 Ton 3-star inverter split AC (1500W)
        "tv": 0.100,             # 43-55" LED TV (100W)
        "fridge": 0.150,         # Double door 260L refrigerator (avg running ~150W)
        "light": 0.015,          # 15W LED bulb / batten
        "washing_machine": 0.500 # Front/Top load washing cycle (500W avg)
    },
    "office": {
        "workstation": 0.150,    # PC/laptop + dual monitor setup (150W)
        "ac_unit": 2.500,        # 2.0 - 3.0 Ton commercial cassette AC (2500W)
        "office_light": 0.040,   # 40W commercial LED troffer panel
        "server_rack": 0.800,    # Small-to-mid enterprise rack (800W continuous)
        "pantry": 1.200          # Coffee machine, microwave, water dispenser
    },
    "mall": {
        "chiller": 45.000,       # 45 kW per chiller unit (commercial central HVAC)
        "retail_light": 0.050,   # Track & showroom display fixtures (50W)
        "escalator": 12.000,     # Commercial escalator / elevator continuous draw (12 kW)
        "food_court": 3.500      # Heavy commercial deep freezers & prep units (3.5 kW)
    }
}

# Regional grid carbon emission intensities (kg CO2 per kWh)
CARBON_FACTORS = {
    "India (CEA 2024 Grid Factor)": 0.82,
    "United States (EPA eGRID Avg)": 0.39,
    "European Union (EEA Average)": 0.23,
    "Global Average Benchmark": 0.45
}

# Quick starter presets for one-click setup
PRESETS = {
    "Custom Configuration": None,
    "2BHK Apartment (1,000 sq ft, 3 people)": {
        "building_type": "House",
        "area": 1000.0,
        "occupancy": 3,
        "solar_kw": 2.0,
        "appliances": {"fan": 3, "ac": 1, "tv": 1, "fridge": 1, "light": 6, "washing_machine": 1}
    },
    "3BHK Villa (2,200 sq ft, 5 people)": {
        "building_type": "House",
        "area": 2200.0,
        "occupancy": 5,
        "solar_kw": 5.0,
        "appliances": {"fan": 5, "ac": 3, "tv": 2, "fridge": 1, "light": 14, "washing_machine": 1}
    },
    "Tech Company Office (6,000 sq ft, 45 staff)": {
        "building_type": "Office",
        "area": 6000.0,
        "occupancy": 45,
        "solar_kw": 10.0,
        "appliances": {"workstations": 45, "ac_units": 6, "office_lights": 50, "server_racks": 2, "pantry": 3}
    },
    "Community Shopping Mall (45,000 sq ft)": {
        "building_type": "Mall",
        "area": 45000.0,
        "occupancy": 2500,
        "solar_kw": 50.0,
        "appliances": {"chillers": 4, "retail_lights": 750, "escalators": 6, "food_court": 8}
    }
}
