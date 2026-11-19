# ⚡ Smart Energy Optimization using Machine Learning

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://smart-energy-optimization-tjwebrzbvsjfnceqjpsgeb.streamlit.app/)

> An AI-powered energy prediction and optimization platform that estimates electricity consumption based on live weather data, building structural profiles, occupancy patterns, and appliance loads.

---

## 📌 Overview

Electricity consumption is driven by a complex interplay of environmental factors (temperature, humidity), building characteristics (area, insulation, building category), and human activity (schedules, occupancy, equipment). 

This project provides an intelligent energy auditing and simulation tool that helps homeowners, facility managers, and enterprise administrators predict electricity demand, visualize 24-hour diurnal load profiles, and discover actionable strategies to reduce monthly energy expenditure.

---

## 🚀 Key Features

* **🔮 Machine Learning Energy Prediction:** Powered by a Random Forest Regressor trained on diurnal environmental and architectural energy datasets.
* **🌡 Live Satellite Weather Integration:** Automatically fetches live ambient temperature, humidity, and weather conditions via OpenWeatherMap API for any global city.
* **🏢 Multi-Category Building Support:**
  * **House:** Household profile with residential appliances (fans, split ACs, refrigerator, TV, lighting, washing machine).
  * **Office:** Commercial workplace with workstations, central HVAC, server rooms, and commercial lighting.
  * **Mall:** Large-scale commercial infrastructure with central chillers, retail display lighting, escalators, and food courts.
* **📈 24-Hour Diurnal Load Curve:** Realistic 24-hour simulation revealing peak demand hours and off-peak baselines.
* **💡 AI Optimization Recommendations:** Context-aware energy saving advice (AC thermostat adjustments, peak load shifting, standby power reduction) with estimated dollar/rupee savings.
* **💰 Cost & Tariff Estimation:** Configurable utility tariff per kWh to estimate total monthly electricity bills and monetary savings.
* **📁 Audit History Tracking & CSV Export:** Automatically archives prediction logs with full CSV export capabilities.

---

## 🧠 Machine Learning Model Details

| Attribute | Specification |
|---|---|
| **Model** | Random Forest Regressor (`sklearn.ensemble.RandomForestRegressor`) |
| **Input Features** | `temperature`, `humidity`, `hour`, `day`, `building_type`, `occupancy`, `area` |
| **Output Target** | Hourly energy consumption (kWh) |
| **Serialization** | `joblib` |
| **Diurnal Modeling** | Incorporates day-of-week and operating schedule curves (residential vs. commercial vs. retail) |

---

## ⚙️ Installation & Setup

### Prerequisites
* Python 3.10+
* Git

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
You can pass your OpenWeatherMap API key via environment variable:
```bash
export OPENWEATHER_API_KEY="your_api_key_here"   # Linux/macOS
$env:OPENWEATHER_API_KEY="your_api_key_here"      # Windows PowerShell
```
*(If unset, the application includes a fallback key and offline demo defaults.)*

### 4. (Optional) Train or Update the Model
```bash
python smartpower.py
```

### 5. Launch the Streamlit App
```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
smart-energy-optimization/
│
├── app.py                     # Streamlit frontend & interactive dashboard
├── smartpower.py              # ML simulation pipeline, training script & CLI test
├── model.pkl                  # Serialized Random Forest Regressor model
├── history.csv                # Persistent audit history log
├── requirements.txt           # Python package requirements
├── README.md                  # Project documentation
└── .devcontainer/             # Dev container configuration for VS Code & Codespaces
```

---

## 👨‍💻 Author
**Arjun Rajesh**  
GitHub: [@arjun-1002](https://github.com/arjun-1002)
