import os
import urllib.parse
from datetime import datetime
import streamlit as st
import numpy as np
import pandas as pd
import requests
import joblib
import plotly.graph_objects as go

# ================= CONFIG & PAGE SETUP =================
st.set_page_config(
    page_title="Smart Energy AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
HISTORY_PATH = os.path.join(BASE_DIR, "history.csv")

# ================= CUSTOM STYLES =================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: #ffffff;
}
.metric-card {
    background: rgba(255, 255, 255, 0.07);
    border: 1px solid rgba(255, 255, 255, 0.12);
    padding: 16px 20px;
    border-radius: 12px;
    margin-bottom: 12px;
    box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.25);
}
.recommendation-card {
    background: rgba(0, 255, 163, 0.08);
    border-left: 4px solid #00FFA3;
    padding: 14px 18px;
    border-radius: 8px;
    margin-bottom: 10px;
}
.warning-card {
    background: rgba(255, 193, 7, 0.1);
    border-left: 4px solid #FFC107;
    padding: 14px 18px;
    border-radius: 8px;
    margin-bottom: 10px;
}
.stButton>button {
    background: linear-gradient(135deg, #00FFA3, #00C9A7) !important;
    color: #0b1a1f !important;
    font-weight: 700 !important;
    border-radius: 8px !important;
    border: none !important;
    padding: 0.6rem 1.2rem !important;
    width: 100% !important;
    transition: all 0.2s ease-in-out !important;
}
.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0, 255, 163, 0.4);
}
.history-table {
    width: 100%;
    border-collapse: collapse;
    background: rgba(255, 255, 255, 0.04);
    border-radius: 8px;
    overflow: hidden;
    margin: 15px 0;
}
.history-table th {
    background: rgba(0, 255, 163, 0.15);
    color: #00FFA3;
    padding: 10px 14px;
    text-align: left;
    font-size: 0.9rem;
}
.history-table td {
    padding: 9px 14px;
    border-top: 1px solid rgba(255, 255, 255, 0.08);
    color: #e0e0e0;
    font-size: 0.88rem;
}
.history-table tr:hover {
    background: rgba(255, 255, 255, 0.07);
}
</style>
""", unsafe_allow_html=True)

# ================= MODEL LOADER =================
@st.cache_resource(show_spinner=False)
def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"Failed to load model file: {e}")
        return None

model = load_trained_model()

# ================= WEATHER API HELPER =================
def fetch_weather(city_name, api_key):
    clean_city = city_name.strip()
    if not clean_city:
        clean_city = "Delhi"
    
    if not api_key:
        return 28.0, 50.0, "Clear (Demo)", False, "No OpenWeatherMap API key configured. Using standard climate data (enter key in sidebar or set OPENWEATHER_API_KEY)."

    encoded_city = urllib.parse.quote(clean_city)
    url = f"https://api.openweathermap.org/data/2.5/weather?q={encoded_city}&appid={api_key}&units=metric"
    
    try:
        response = requests.get(url, timeout=3.5)
        if response.status_code == 200:
            data = response.json()
            temp = float(data['main']['temp'])
            humidity = float(data['main']['humidity'])
            condition = data['weather'][0]['main'] if data.get('weather') else "Clear"
            return temp, humidity, condition, True, None
        elif response.status_code == 404:
            return 28.0, 50.0, "Clear (Default)", False, f"City '{clean_city}' not found. Using default climate data."
        else:
            return 28.0, 50.0, "Clear (Default)", False, f"Weather API returned status code {response.status_code}."
    except requests.exceptions.Timeout:
        return 28.0, 50.0, "Clear (Default)", False, "Weather API request timed out. Using default climate data."
    except Exception as e:
        return 28.0, 50.0, "Clear (Default)", False, f"Network error fetching weather: {e}"

# ================= HEADER =================
st.markdown("<h1 style='text-align:center; color:#00FFA3; margin-bottom:0;'>⚡ Smart Energy Optimization</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#B0BEC5; margin-bottom:20px;'>AI-Powered Energy Consumption Prediction, Simulation & Smart Efficiency</p>", unsafe_allow_html=True)

now = datetime.now()
hour_now = now.hour
day_now = now.weekday()

col_time1, col_time2 = st.columns([1, 1])
with col_time1:
    st.markdown(f"""
    <div class="metric-card" style="text-align:center;">
        📅 <b>Date:</b> {now.strftime("%A, %d %B %Y")}
    </div>
    """, unsafe_allow_html=True)
with col_time2:
    st.markdown(f"""
    <div class="metric-card" style="text-align:center;">
        ⏰ <b>Local Time:</b> {now.strftime("%I:%M:%S %p")} (Hour {hour_now:02d}:00)
    </div>
    """, unsafe_allow_html=True)

# ================= SIDEBAR INPUTS =================
st.sidebar.markdown("## ⚙️ Configuration")

# API Key handling (Load from environment or let user enter via sidebar)
default_api_key = os.getenv("OPENWEATHER_API_KEY", "")
with st.sidebar.expander("🔑 Weather API Settings"):
    api_key_input = st.text_input("OpenWeatherMap Key", value=default_api_key, type="password", placeholder="Paste your API key here")
    active_api_key = api_key_input.strip()

city = st.sidebar.text_input("📍 Location / City", value="Delhi")

st.sidebar.markdown("---")
st.sidebar.markdown("## 🏢 Building Details")

building_types = ["House", "Office", "Mall"]
building_type_name = st.sidebar.selectbox("Building Type", building_types)
building_type_idx = building_types.index(building_type_name)

# Building-specific inputs
if building_type_idx == 0:  # House
    area = st.sidebar.number_input("Floor Area (sq ft)", min_value=100.0, max_value=20000.0, value=1500.0, step=100.0)
    occupancy = st.sidebar.number_input("Occupants (People)", min_value=1, max_value=25, value=4, step=1)
    
    st.sidebar.markdown("### 🔌 Household Appliances")
    fan = st.sidebar.number_input("Ceiling Fans (75W)", min_value=0, max_value=20, value=3)
    ac = st.sidebar.number_input("Air Conditioners (1.5 kW)", min_value=0, max_value=10, value=1)
    tv = st.sidebar.number_input("TVs (100W)", min_value=0, max_value=10, value=1)
    fridge = st.sidebar.number_input("Refrigerators (150W)", min_value=0, max_value=5, value=1)
    light = st.sidebar.number_input("LED Lights (15W)", min_value=0, max_value=100, value=8)
    washing_machine = st.sidebar.number_input("Washing Machine (500W)", min_value=0, max_value=5, value=1)
    
elif building_type_idx == 1:  # Office
    area = st.sidebar.number_input("Office Floor Area (sq ft)", min_value=500.0, max_value=100000.0, value=5000.0, step=500.0)
    occupancy = st.sidebar.number_input("Employees / Staff", min_value=1, max_value=1000, value=35, step=5)
    
    st.sidebar.markdown("### 💻 Office Equipment")
    workstations = st.sidebar.number_input("Workstations / PCs (150W)", min_value=0, max_value=1000, value=35)
    ac_units = st.sidebar.number_input("Split/Central ACs (2.5 kW)", min_value=0, max_value=50, value=4)
    office_lights = st.sidebar.number_input("Commercial LED Fixtures (40W)", min_value=0, max_value=500, value=40)
    server_racks = st.sidebar.number_input("Server Racks (800W)", min_value=0, max_value=20, value=1)
    pantry_appliances = st.sidebar.number_input("Pantry Equipment (1.2 kW)", min_value=0, max_value=20, value=2)

else:  # Mall
    area = st.sidebar.number_input("Mall Floor Area (sq ft)", min_value=5000.0, max_value=500000.0, value=50000.0, step=5000.0)
    default_mall_occ = 4000 if day_now >= 5 else 2000
    occupancy = st.sidebar.number_input("Average Daily Footfall / Occupancy", min_value=100, max_value=50000, value=default_mall_occ, step=200)
    
    st.sidebar.markdown("### 🏬 Commercial Facilities")
    chillers = st.sidebar.number_input("Central Chillers (45 kW per unit)", min_value=1, max_value=50, value=4)
    retail_lights = st.sidebar.number_input("Retail Display Lights (50W fixtures)", min_value=10, max_value=10000, value=800)
    escalators = st.sidebar.number_input("Escalators & Elevators (12 kW)", min_value=1, max_value=50, value=6)
    food_court = st.sidebar.number_input("Food Court Refrigeration (3.5 kW)", min_value=0, max_value=50, value=6)

st.sidebar.markdown("---")
tariff = st.sidebar.number_input("Electricity Tariff ($ or ₹ per Unit)", min_value=0.5, max_value=100.0, value=8.0, step=0.5)

# ================= CALCULATION & INFERENCE ENGINE =================
def run_simulation(temp, humidity, btype_idx, occ, b_area):
    hours = np.arange(24)
    
    # Model 24-hour baseline predictions
    if model is not None:
        batch_inputs = pd.DataFrame([{
            'temperature': temp + (3.0 * np.sin((h - 9) * np.pi / 12) if 8 <= h <= 18 else -2.0),
            'humidity': humidity,
            'hour': h,
            'day': day_now,
            'building_type': btype_idx,
            'occupancy': occ,
            'area': b_area
        } for h in hours])
        
        sim_preds = model.predict(batch_inputs)
        sim_preds = np.maximum(0.1, sim_preds)
    else:
        sim_preds = np.full(24, 1.0)
    
    # Calculate Equipment Specific Energy (Daily kWh)
    equipment_breakdown = {}
    
    if btype_idx == 0:  # House
        ac_hours = 8.0 if temp > 28 else (4.0 if temp > 22 else 1.0)
        fan_kwh = fan * 0.075 * 10.0
        ac_kwh = ac * 1.5 * ac_hours
        tv_kwh = tv * 0.100 * 4.0
        fridge_kwh = fridge * 0.150 * 24.0
        light_kwh = light * 0.015 * 6.0
        wm_kwh = washing_machine * 0.500 * 1.0
        
        equipment_breakdown = {
            "Cooling (AC)": ac_kwh,
            "Fans": fan_kwh,
            "Refrigeration": fridge_kwh,
            "Lighting": light_kwh,
            "Entertainment & Wash": tv_kwh + wm_kwh
        }
        total_equipment_kwh = sum(equipment_breakdown.values())
        equipment_hourly = np.zeros(24)
        for h in range(24):
            frac = 0.03 if h < 6 else (0.05 if h < 18 else 0.07)
            equipment_hourly[h] = total_equipment_kwh * frac
            
    elif btype_idx == 1:  # Office
        hvac_hours = 9.0 if temp > 24 else 5.0
        pc_kwh = workstations * 0.150 * 8.5
        ac_kwh = ac_units * 2.5 * hvac_hours
        light_kwh = office_lights * 0.040 * 9.0
        server_kwh = server_racks * 0.800 * 24.0
        pantry_kwh = pantry_appliances * 1.200 * 3.0
        
        equipment_breakdown = {
            "HVAC / Air Conditioning": ac_kwh,
            "Workstations / IT": pc_kwh,
            "Lighting": light_kwh,
            "Servers & Network": server_kwh,
            "Pantry & Others": pantry_kwh
        }
        total_equipment_kwh = sum(equipment_breakdown.values())
        equipment_hourly = np.zeros(24)
        for h in range(24):
            frac = 0.08 if 9 <= h <= 18 else 0.01
            equipment_hourly[h] = total_equipment_kwh * frac
            
    else:  # Mall
        chiller_hours = 12.0 if temp > 25 else 8.0
        chiller_kwh = chillers * 45.0 * chiller_hours
        light_kwh = retail_lights * 0.050 * 12.0
        esc_kwh = escalators * 12.0 * 12.0
        food_kwh = food_court * 3.5 * 14.0
        
        equipment_breakdown = {
            "Central Chillers": chiller_kwh,
            "Retail Lighting": light_kwh,
            "Escalators & Lifts": esc_kwh,
            "Food Court & Storage": food_kwh
        }
        total_equipment_kwh = sum(equipment_breakdown.values())
        equipment_hourly = np.zeros(24)
        for h in range(24):
            frac = 0.075 if 10 <= h <= 22 else 0.01
            equipment_hourly[h] = total_equipment_kwh * frac

    # Total profile combines structural simulation + equipment profile
    total_hourly = sim_preds + equipment_hourly
    daily_total = float(np.sum(total_hourly))
    current_hourly = float(total_hourly[hour_now])
    monthly_total = daily_total * 30.0
    monthly_cost = monthly_total * tariff

    # Smart Recommendations Generator
    recommendations = []
    if temp > 28:
        cooling_save_pct = 6.0 * min(4, int(temp - 24))
        save_units = daily_total * (cooling_save_pct / 100.0) * 30.0
        recommendations.append({
            "title": "🌡 Optimize AC Setpoint to 24°C - 26°C",
            "desc": f"Raising your thermostat by just 2°C can reduce cooling load by {cooling_save_pct:.0f}%, saving approx. {save_units:.1f} units (${save_units * tariff:.0f} / ₹{save_units * tariff:.0f}) per month.",
            "type": "tip"
        })
    
    if btype_idx == 0:
        recommendations.append({
            "title": "⏰ Shift Heavy Loads Off-Peak",
            "desc": "Run washing machines and water pumps during off-peak hours (before 11 AM or after 9 PM) to relieve peak-hour stress and take advantage of time-of-day tariffs.",
            "type": "tip"
        })
        recommendations.append({
            "title": "🔌 Eliminate Phantom / Standby Power",
            "desc": "Televisions, microwave displays, and chargers consume standby electricity 24/7. Using smart power strips can reduce your household base load by 5-8%.",
            "type": "tip"
        })
    elif btype_idx == 1:
        recommendations.append({
            "title": "🖥 Automated IT Power Management",
            "desc": "Ensure automated sleep modes are enforced on workstations. Turning off 30+ idle screens and PCs overnight saves up to 45 kWh daily.",
            "type": "tip"
        })
        recommendations.append({
            "title": "💡 Daylight Harvesting & Motion Sensors",
            "desc": "Dim perimeter lighting near windows during peak daylight hours to reduce office lighting energy by up to 25%.",
            "type": "tip"
        })
    else:
        recommendations.append({
            "title": "🏬 Intelligent Escalator & Chiller Modulation",
            "desc": "Implement VFDs (Variable Frequency Drives) on escalators and chillers during morning hours (10 AM - 1 PM) when footfall is under 40% capacity.",
            "type": "tip"
        })
        recommendations.append({
            "title": "❄️ Pre-cooling Thermal Storage",
            "desc": "Pre-cool commercial common areas 1 hour before peak afternoon heat (12 PM) to avoid high peak-demand tariff penalties.",
            "type": "tip"
        })

    # Prepare DataFrame for 24h curve
    hourly_df = pd.DataFrame({
        "Hour": [f"{h:02d}:00" for h in hours],
        "Predicted Demand (kW / kWh)": total_hourly,
        "Structural Base (kWh)": sim_preds,
        "Active Equipment (kWh)": equipment_hourly
    })

    return {
        "daily_total": daily_total,
        "current_hourly": current_hourly,
        "monthly_total": monthly_total,
        "monthly_cost": monthly_cost,
        "equipment_breakdown": equipment_breakdown,
        "recommendations": recommendations,
        "hourly_df": hourly_df
    }

# ================= BUTTON & STATE MANAGEMENT =================
predict_clicked = st.sidebar.button("🚀 Calculate & Optimize Energy", use_container_width=True)

if predict_clicked or "prediction_data" not in st.session_state:
    with st.spinner("Fetching meteorological data and running ML simulation..."):
        temp_val, hum_val, cond_val, is_live, err_msg = fetch_weather(city, active_api_key)
        
        results = run_simulation(temp_val, hum_val, building_type_idx, occupancy, area)
        
        st.session_state["prediction_data"] = {
            "city": city,
            "temp": temp_val,
            "humidity": hum_val,
            "condition": cond_val,
            "is_live": is_live,
            "err_msg": err_msg,
            "building_type": building_type_name,
            "occupancy": occupancy,
            "area": area,
            "results": results,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # Safe logging to history.csv
        try:
            new_log = pd.DataFrame([{
                "Timestamp": st.session_state["prediction_data"]["timestamp"],
                "City": city,
                "Building_Type": building_type_name,
                "Area_sqft": area,
                "Occupancy": occupancy,
                "Daily_Units": round(results["daily_total"], 2),
                "Monthly_Units": round(results["monthly_total"], 2),
                "Temp_C": round(temp_val, 1)
            }])
            if os.path.exists(HISTORY_PATH):
                existing_hist = pd.read_csv(HISTORY_PATH)
                combined = pd.concat([existing_hist, new_log], ignore_index=True)
            else:
                combined = new_log
            combined.to_csv(HISTORY_PATH, index=False)
        except Exception as e:
            st.sidebar.error(f"Could not write to history file: {e}")

# ================= DASHBOARD DISPLAY =================
data = st.session_state["prediction_data"]
res = data["results"]

# Weather Status Card
weather_color = "#00FFA3" if data["is_live"] else "#FFC107"
weather_status_text = "Live Satellite Weather" if data["is_live"] else "Standard Climate Mode"

st.markdown(f"""
<div class="metric-card" style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap;">
    <div>
        <span style="font-size:1.2rem; font-weight:bold;">📍 {data['city']}</span> | 
        <span style="color:{weather_color}; font-weight:600;">● {weather_status_text}</span>: {data['condition']}
    </div>
    <div style="font-size:1.1rem;">
        🌡 <b>{data['temp']:.1f}°C</b> &nbsp;&nbsp;|&nbsp;&nbsp; 💧 <b>{data['humidity']:.0f}% Humidity</b>
    </div>
</div>
""", unsafe_allow_html=True)

if data["err_msg"]:
    st.info(data["err_msg"])

# Top Metrics Row
m1, m2, m3, m4 = st.columns(4)
with m1:
    m1.metric("⚡ Daily Energy", f"{res['daily_total']:.1f} kWh", help="Total predicted daily electricity consumption in units (kWh)")
with m2:
    m2.metric("⏱ Current Demand", f"{res['current_hourly']:.2f} kW", help="Estimated electricity load for the current hour")
with m3:
    m3.metric("📅 Monthly Forecast", f"{res['monthly_total']:,.0f} Units", help="Projected 30-day energy consumption")
with m4:
    m4.metric("💰 Est. Monthly Cost", f"{res['monthly_cost']:,.0f}", help=f"Estimated bill based on tariff of {tariff} per unit")

# Tabs for Organized Analysis
tab1, tab2, tab3, tab4 = st.tabs(["📈 24-Hour Load Profile", "🔌 Load Breakdown", "💡 Smart Recommendations", "📁 Audit History"])

with tab1:
    st.markdown("### 24-Hour Projected Electricity Demand")
    st.caption("Hourly profile generated by the Machine Learning Model based on ambient temperature, diurnal cycle, and operating hours.")
    
    hourly_series = res["hourly_df"]
    
    # Plotly 24-hour interactive curve (Zero pyarrow dependency)
    fig_24h = go.Figure()
    fig_24h.add_trace(go.Scatter(
        x=hourly_series["Hour"],
        y=hourly_series["Predicted Demand (kW / kWh)"],
        mode="lines+markers",
        name="Total Demand (kW)",
        line=dict(color="#00FFA3", width=3.5),
        marker=dict(size=6, color="#00FFA3"),
        hovertemplate="Time: %{x}<br>Demand: %{y:.2f} kW<extra></extra>"
    ))
    fig_24h.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.25)",
        font=dict(color="#ffffff"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.08)", title="Time of Day"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.08)", title="Power Demand (kW)"),
        margin=dict(l=20, r=20, t=25, b=20),
        height=320
    )
    st.plotly_chart(fig_24h, use_container_width=True)
    
    peak_hour_idx = int(np.argmax(hourly_series["Predicted Demand (kW / kWh)"]))
    peak_val = hourly_series["Predicted Demand (kW / kWh)"][peak_hour_idx]
    st.markdown(f"**⚡ Peak Consumption Hour:** `{peak_hour_idx:02d}:00` (Demand: `{peak_val:.2f} kW`)")

with tab2:
    st.markdown("### Equipment & Infrastructure Breakdown")
    eq_data = res["equipment_breakdown"]
    
    col_breakdown1, col_breakdown2 = st.columns([1, 1])
    with col_breakdown1:
        # Plotly bar chart for equipment
        categories = list(eq_data.keys())
        values = list(eq_data.values())
        
        fig_eq = go.Figure(go.Bar(
            x=categories,
            y=values,
            marker_color="#00C9A7",
            hovertemplate="%{x}: %{y:.1f} kWh<extra></extra>"
        ))
        fig_eq.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0.25)",
            font=dict(color="#ffffff"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.08)", title="Daily Consumption (kWh)"),
            margin=dict(l=20, r=20, t=25, b=20),
            height=300
        )
        st.plotly_chart(fig_eq, use_container_width=True)
    
    with col_breakdown2:
        st.markdown("#### Category Distribution")
        tot_eq = sum(eq_data.values())
        if tot_eq > 0:
            for cat, val in eq_data.items():
                pct = (val / tot_eq) * 100
                st.write(f"• **{cat}:** `{val:.1f} kWh` ({pct:.1f}%)")
        else:
            st.write("No active equipment loaded.")

with tab3:
    st.markdown("### 💡 AI Energy Optimization Recommendations")
    st.caption("Custom-tailored actions to minimize waste and reduce monthly utility expenses.")
    
    for rec in res["recommendations"]:
        card_class = "recommendation-card" if rec["type"] == "tip" else "warning-card"
        st.markdown(f"""
        <div class="{card_class}">
            <h4 style="margin:0 0 6px 0; color:#00FFA3;">{rec['title']}</h4>
            <p style="margin:0; color:#E0E0E0; font-size:0.95rem;">{rec['desc']}</p>
        </div>
        """, unsafe_allow_html=True)
        
    est_saving = res['monthly_cost'] * 0.15
    st.success(f"🎯 **Potential Savings:** Implementing these measures can reduce your utility bills by approx. **15% to 20%** (~**${est_saving:,.0f} / ₹{est_saving:,.0f}** saved monthly)!")

with tab4:
    st.markdown("### 📁 Prediction & Audit History")
    if os.path.exists(HISTORY_PATH):
        try:
            hist_df = pd.read_csv(HISTORY_PATH)
            if not hist_df.empty:
                # Render clean HTML table without pyarrow dependency
                display_df = hist_df.tail(10).iloc[::-1]
                table_html = "<table class='history-table'><thead><tr>"
                for col in display_df.columns:
                    table_html += f"<th>{col}</th>"
                table_html += "</tr></thead><tbody>"
                for _, row in display_df.iterrows():
                    table_html += "<tr>"
                    for val in row:
                        table_html += f"<td>{val}</td>"
                    table_html += "</tr>"
                table_html += "</tbody></table>"
                
                st.markdown(table_html, unsafe_allow_html=True)
                
                # Plotly History trend
                if len(hist_df) > 1:
                    st.markdown("#### Historical Energy Trend (Recent Runs)")
                    fig_hist = go.Figure()
                    fig_hist.add_trace(go.Scatter(
                        y=hist_df["Daily_Units"].tail(20),
                        mode="lines+markers",
                        line=dict(color="#00FFA3", width=2.5),
                        marker=dict(size=5, color="#00FFA3"),
                        name="Daily Units"
                    ))
                    fig_hist.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0.25)",
                        font=dict(color="#ffffff"),
                        xaxis=dict(gridcolor="rgba(255,255,255,0.08)", title="Run Index"),
                        yaxis=dict(gridcolor="rgba(255,255,255,0.08)", title="Daily Units (kWh)"),
                        margin=dict(l=20, r=20, t=25, b=20),
                        height=260
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                # Download CSV button
                csv_data = hist_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download History CSV",
                    data=csv_data,
                    file_name="energy_prediction_history.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.info("No records logged in history yet.")
        except Exception as e:
            st.error(f"Error reading history log: {e}")
    else:
        st.info("History file will be generated on your first prediction.")

# ================= FOOTER =================
st.markdown("---")
st.markdown("<p style='text-align:center; color:#9E9E9E;'>⚡ Smart Energy AI | Save Energy • Save Future</p>", unsafe_allow_html=True)
