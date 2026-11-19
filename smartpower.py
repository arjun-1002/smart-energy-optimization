import os
import requests
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import joblib

# Set random seed for reproducibility
np.random.seed(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

# ================= DATASET GENERATION =================
# Building types: 0 = House, 1 = Office, 2 = Mall
samples_per_type = 4000
records = []

# --- 1. House (0) ---
for _ in range(samples_per_type):
    temp = np.random.uniform(15, 45)
    humidity = np.random.uniform(20, 90)
    hour = np.random.randint(0, 24)
    day = np.random.randint(0, 7)
    btype = 0
    occ = np.random.randint(1, 9)
    area = np.random.uniform(500, 3500)

    # Base standby load (refrigerator, router, standby electronics)
    base = 0.15 + (area * 0.00005)

    # Cooling load when ambient temperature exceeds comfortable 24°C
    cooling = max(0.0, (temp - 24) * 0.04) * (occ / 3.0)

    # Diurnal activity schedule: morning peak (6-9), evening peak (18-23)
    if day >= 5: # Weekend: elevated daytime usage
        activity = 1.3 if (8 <= hour <= 23) else 0.5
    else:        # Weekday: morning rush and evening family time
        activity = 1.5 if (18 <= hour <= 23) else (1.2 if (6 <= hour <= 9) else 0.6)

    hourly_kwh = (base + cooling + (occ * 0.05)) * activity + np.random.normal(0, 0.02)
    hourly_kwh = max(0.1, float(hourly_kwh))
    records.append([temp, humidity, hour, day, btype, occ, area, hourly_kwh])

# --- 2. Office (1) ---
for _ in range(samples_per_type):
    temp = np.random.uniform(15, 45)
    humidity = np.random.uniform(20, 90)
    hour = np.random.randint(0, 24)
    day = np.random.randint(0, 7)
    btype = 1
    area = np.random.uniform(1500, 30000)
    max_occ = int(area / 100)

    is_workday = day < 5
    is_workhours = 8 <= hour <= 19

    if is_workday and is_workhours:
        occ = np.random.randint(int(max_occ * 0.4), max_occ + 1)
        base = area * 0.0008
        hvac = max(0.0, (temp - 22) * 0.0006 * area)
        plug_load = occ * 0.15  # PCs, monitors, servers
        lighting = area * 0.001
        hourly_kwh = base + hvac + plug_load + lighting
    elif is_workday and not is_workhours:
        occ = np.random.randint(0, max(2, int(max_occ * 0.05)))
        hourly_kwh = (area * 0.0004) + (occ * 0.08) + max(0.0, (temp - 26) * 0.0002 * area)
    else:  # Weekend
        occ = np.random.randint(0, max(2, int(max_occ * 0.03)))
        hourly_kwh = (area * 0.0003) + (occ * 0.08)

    hourly_kwh += np.random.normal(0, max(0.1, hourly_kwh * 0.03))
    hourly_kwh = max(1.0, float(hourly_kwh))
    records.append([temp, humidity, hour, day, btype, occ, area, hourly_kwh])

# --- 3. Mall (2) ---
for _ in range(samples_per_type):
    temp = np.random.uniform(15, 45)
    humidity = np.random.uniform(20, 90)
    hour = np.random.randint(0, 24)
    day = np.random.randint(0, 7)
    btype = 2
    area = np.random.uniform(15000, 150000)
    base_occ = int(area / 30)

    is_open = 10 <= hour <= 22
    is_weekend = day >= 5

    if is_open:
        occ = int(np.random.uniform(0.5, 1.0) * base_occ * (1.3 if is_weekend else 0.8))
        hvac = max(0.0, (temp - 22) * 0.0015 * area)
        lighting = area * 0.002
        chillers_escalators = area * 0.001 + (occ * 0.01)
        base = area * 0.002
        hourly_kwh = base + chillers_escalators + hvac + lighting
    else:
        occ = np.random.randint(5, 50)  # cleaning / security staff
        hourly_kwh = (area * 0.0005) + (occ * 0.05)

    hourly_kwh += np.random.normal(0, max(0.5, hourly_kwh * 0.03))
    hourly_kwh = max(5.0, float(hourly_kwh))
    records.append([temp, humidity, hour, day, btype, occ, area, hourly_kwh])

feature_cols = ['temperature', 'humidity', 'hour', 'day', 'building_type', 'occupancy', 'area']
df = pd.DataFrame(records, columns=feature_cols + ['energy'])

X = df[feature_cols]
y = df['energy']

print("Training Random Forest Regressor on realistic building energy simulation...")
best_model = RandomForestRegressor(
    n_estimators=60,
    max_depth=15,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
best_model.fit(X, y)

score = r2_score(y, best_model.predict(X))
print(f"Model trained successfully. R² Score: {score:.4f}")

# Save the trained model with compression (keeping file well below GitHub 100MB limit)
joblib.dump(best_model, MODEL_PATH, compress=3)
model_size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
print(f"Model saved to {MODEL_PATH} ({model_size_mb:.2f} MB) ✅")

# ================= WEATHER API & INFERENCE DEMO =================
API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
CITY = "Delhi"

def get_weather(city=CITY, api_key=API_KEY):
    if not api_key:
        return 30.0, 50.0, "Clear (Demo Mode)"
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={requests.utils.quote(city)}&appid={api_key}&units=metric"
        response = requests.get(url, timeout=4)
        data = response.json()
        if response.status_code == 200 and 'main' in data:
            temp = data['main']['temp']
            humidity = data['main']['humidity']
            condition = data['weather'][0]['main'] if data.get('weather') else "Clear"
            return temp, humidity, condition
        else:
            print(f"API Warning ({response.status_code}): {data.get('message', 'Failed to fetch')}")
            return 30.0, 50.0, "Clear (Fallback)"
    except Exception as e:
        print(f"Network error fetching weather: {e}")
        return 30.0, 50.0, "Clear (Fallback)"

def get_time():
    now = datetime.now()
    return now.hour, now.weekday()

def suggest(temp, hour, building_type=0):
    suggestions = []
    if temp > 28:
        suggestions.append("Set AC thermostat to 24-26°C to save up to 18% on cooling.")
    if 12 <= hour <= 17:
        suggestions.append("Avoid running heavy, non-essential loads during afternoon peak hours.")
    elif 18 <= hour <= 22:
        suggestions.append("Evening peak grid demand: turn off idle appliances and use task lighting.")
    if building_type == 1:
        suggestions.append("Ensure sleep timers are enabled on workstations and office display panels.")
    elif building_type == 2:
        suggestions.append("Optimize chiller staging and schedule escalator slow-down during non-peak footfall.")
    return suggestions

# Test real-time prediction
temp, humidity, condition = get_weather(CITY)
hour, day = get_time()

test_input = pd.DataFrame([{
    'temperature': temp,
    'humidity': humidity,
    'hour': hour,
    'day': day,
    'building_type': 0, # House
    'occupancy': 4,
    'area': 1500
}])

hourly_pred = best_model.predict(test_input)[0]

print("\n--- REAL-TIME PREDICTION SAMPLE ---")
print(f"City: {CITY} | Weather: {condition} | Temp: {temp}°C | Humidity: {humidity}%")
print(f"Current Time: Hour {hour}, Day {day}")
print(f"Predicted Current Hourly Load (House 1500 sq ft, 4 people): {hourly_pred:.2f} kWh")

print("\nSmart Suggestions:")
for s in suggest(temp, hour, 0):
    print(f" - {s}")
