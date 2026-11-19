# ⚡ Smart Energy & Solar Optimizer

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://smart-energy-optimization-tjwebrzbvsjfnceqjpsgeb.streamlit.app/)

> An AI-powered energy management and simulation platform that estimates electrical demand, models rooftop solar PV net-metering, performs carbon footprint auditing, and provides actionable optimization strategies for buildings.

---

## 📌 Overview

Electricity consumption in buildings is driven by weather dynamics (temperature, humidity), building physics (floor area, category), occupancy schedules, and active appliance loads. 

This platform bridges machine learning forecasting with practical energy engineering. Users can simulate 24-hour diurnal load curves, evaluate rooftop solar self-consumption, calculate Energy Use Intensity (EUI) star ratings, and compute the financial benefits of shifting flexible loads under Time-of-Use (ToU) tariffs.

---

## 🚀 Key Features

* **🔮 Machine Learning Forecasting:** Predicts base structural and diurnal electrical load using an optimized Random Forest Regressor trained on physical building profiles.
* **☀️ Rooftop Solar PV & Net-Metering:** Simulates solar irradiance curves (6 AM – 6 PM) to calculate self-consumption, surplus export, and net monthly bill savings.
* **🏢 Multi-Category Infrastructure Presets:**
  * **House:** Pre-configured for modern apartments and independent villas.
  * **Office:** Workstation density, commercial cassette HVAC, servers, and lighting.
  * **Mall:** Central chiller plants, escalators/elevators, and retail display illumination.
* **🌱 Carbon Accounting & Efficiency Star Rating:**
  * Computes Energy Use Intensity (**EUI** in $\text{kWh}/\text{sq ft}/\text{year}$).
  * Assigns a 1-to-5 Star Efficiency Rating benchmarked against BEE and Energy Star standards.
  * Quantifies annual metric tons of $\text{CO}_2$ emitted and trees required to sequester the emissions.
* **⏰ Time-of-Use (ToU) Dynamic Billing:** Models peak-hour surcharges (18:00–22:00) and off-peak night discounts.
* **💡 Tailored Optimization Action Plan:** Identifies HVAC thermostat setbacks, phantom load mitigation, and solar peak alignment with high/medium impact categorization.
* **📁 Audit Trail & CSV Export:** Automatically records audit records with one-click CSV export.

---

## 🏗 Modular Architecture

```
smart-energy-optimization/
│
├── app.py                     # Streamlit frontend, interactive dashboard & visual layouts
├── energy_engine.py           # Core simulation engine (ML inference, solar curve, EUI, carbon)
├── weather_service.py         # Weather API integration with 10-minute caching & fallback resilience
├── config.py                  # Appliance ratings, quick presets, tariffs & carbon grid factors
├── smartpower.py              # ML simulation pipeline and model training script
├── model.pkl                  # Compressed Random Forest model binary (~5.6 MB)
├── history.csv                # Persistent audit log
├── requirements.txt           # Python package dependencies (including Plotly)
├── README.md                  # Comprehensive project documentation
└── .devcontainer/             # Dev container configuration for VS Code & Codespaces
```

---

## ⚙️ Installation & Local Setup

### 1. Clone the Repository
```bash
git clone https://github.com/arjun-1002/smart-energy-optimization.git
cd smart-energy-optimization
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. (Optional) Set Weather API Key
To fetch live satellite weather from OpenWeatherMap:
```bash
export OPENWEATHER_API_KEY="your_api_key_here"   # Linux / macOS
$env:OPENWEATHER_API_KEY="your_api_key_here"      # Windows PowerShell
```
*(If unset, the application includes a demo climate fallback mode and allows entering a key directly in the sidebar.)*

### 4. Run the Application
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🧠 Machine Learning Model Details

| Attribute | Specification |
|---|---|
| **Estimator** | `sklearn.ensemble.RandomForestRegressor` (60 trees, max depth 15) |
| **Input Features** | `temperature`, `humidity`, `hour`, `day`, `building_type`, `occupancy`, `area` |
| **Output Target** | Hourly electrical energy consumption ($\text{kWh}$) |
| **Accuracy** | $R^2 = 0.9996$ on physical building energy simulation |
| **Model Size** | Compressed to ~5.6 MB (well below GitHub's 100 MB limit) |

---

## 👨‍💻 Author
**Arjun Rajesh**  
GitHub: [@arjun-1002](https://github.com/arjun-1002)
