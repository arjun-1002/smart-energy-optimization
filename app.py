"""
Smart Energy Optimization & Net-Zero Simulation Dashboard.
A human-crafted, production-ready Streamlit application for intelligent
building energy prediction, solar PV net-metering, carbon auditing, and cost optimization.
"""

import os
from datetime import datetime
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

import config
from weather_service import fetch_weather_cached
from energy_engine import load_energy_model, run_energy_simulation

# ================= PAGE SETUP =================
st.set_page_config(
    page_title="Smart Energy AI - Building & Solar Optimizer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
HISTORY_PATH = os.path.join(BASE_DIR, "history.csv")

# ================= CUSTOM CSS =================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0b171b, #12252d, #1b3540);
    color: #f0f4f8;
}
.kpi-card {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 12px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
}
.rating-pill {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 0.95rem;
    margin-top: 5px;
}
.rec-card-high {
    background: rgba(0, 255, 163, 0.07);
    border-left: 4px solid #00FFA3;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 12px;
}
.rec-card-med {
    background: rgba(255, 193, 7, 0.08);
    border-left: 4px solid #FFC107;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 12px;
}
.history-table {
    width: 100%;
    border-collapse: collapse;
    background: rgba(255, 255, 255, 0.03);
    border-radius: 8px;
    overflow: hidden;
    margin: 15px 0;
}
.history-table th {
    background: rgba(0, 255, 163, 0.12);
    color: #00FFA3;
    padding: 10px 14px;
    text-align: left;
    font-size: 0.88rem;
    font-weight: 600;
}
.history-table td {
    padding: 9px 14px;
    border-top: 1px solid rgba(255, 255, 255, 0.06);
    color: #e2e8f0;
    font-size: 0.86rem;
}
.history-table tr:hover {
    background: rgba(255, 255, 255, 0.06);
}
.stButton>button {
    background: linear-gradient(135deg, #00FFA3, #00C9A7) !important;
    color: #08161b !important;
    font-weight: 700 !important;
    border-radius: 8px !important;
    border: none !important;
    padding: 0.65rem 1.2rem !important;
    width: 100% !important;
    transition: all 0.2s ease-in-out !important;
}
.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0, 255, 163, 0.35);
}
</style>
""", unsafe_allow_html=True)

# Load machine learning model
model = load_energy_model(MODEL_PATH)

# ================= HEADER & TIME =================
st.markdown("<h1 style='text-align:center; color:#00FFA3; margin-bottom:2px;'>⚡ Smart Energy & Solar Optimizer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#94a3b8; font-size:1.05rem; margin-bottom:20px;'>Intelligent Building Energy Forecasting, Rooftop Solar Net-Metering & Carbon Auditing</p>", unsafe_allow_html=True)

now = datetime.now()
hour_now = now.hour
day_now = now.weekday()

col_t1, col_t2 = st.columns(2)
with col_t1:
    st.markdown(f"""
    <div class="kpi-card" style="text-align:center;">
        📅 <b>Date:</b> {now.strftime("%A, %d %B %Y")}
    </div>
    """, unsafe_allow_html=True)
with col_t2:
    st.markdown(f"""
    <div class="kpi-card" style="text-align:center;">
        ⏰ <b>Current Time:</b> {now.strftime("%I:%M:%S %p")} &nbsp;|&nbsp; <b>Active Hour:</b> {hour_now:02d}:00
    </div>
    """, unsafe_allow_html=True)

# ================= SIDEBAR: PRESETS & CONFIG =================
st.sidebar.markdown("## 🏢 Quick Setup & Presets")

preset_choice = st.sidebar.selectbox("Load Configuration Preset", list(config.PRESETS.keys()))
active_preset = config.PRESETS[preset_choice]

# Weather API settings
default_api_key = os.getenv("OPENWEATHER_API_KEY", "")
with st.sidebar.expander("🔑 Weather API Configuration", expanded=False):
    api_key_input = st.text_input("OpenWeatherMap API Key", value=default_api_key, type="password", placeholder="Enter your key or leave blank for demo")
    active_api_key = api_key_input.strip()

city = st.sidebar.text_input("📍 Location / City", value=config.DEFAULT_CITY)

st.sidebar.markdown("---")
st.sidebar.markdown("## 🏗 Building Specifications")

building_options = ["House", "Office", "Mall"]
def_btype_idx = 0
if active_preset:
    def_btype_idx = building_options.index(active_preset["building_type"])

building_type = st.sidebar.selectbox("Building Category", building_options, index=def_btype_idx)
btype_idx = building_options.index(building_type)

# Floor area & occupancy
if btype_idx == 0:  # House
    def_area = active_preset["area"] if active_preset else 1500.0
    def_occ = active_preset["occupancy"] if active_preset else 4
    area = st.sidebar.number_input("Floor Area (sq ft)", 200.0, 20000.0, float(def_area), 100.0)
    occupancy = st.sidebar.number_input("Residents / Occupants", 1, 25, int(def_occ), 1)

    st.sidebar.markdown("### 🔌 Residential Appliances")
    app_vals = active_preset["appliances"] if active_preset else {"fan": 3, "ac": 1, "tv": 1, "fridge": 1, "light": 8, "washing_machine": 1}
    fan = st.sidebar.number_input("Ceiling Fans (75W)", 0, 20, app_vals.get("fan", 3))
    ac = st.sidebar.number_input("Inverter ACs (1.5 kW)", 0, 10, app_vals.get("ac", 1))
    tv = st.sidebar.number_input("Televisions (100W)", 0, 10, app_vals.get("tv", 1))
    fridge = st.sidebar.number_input("Refrigerators (150W)", 0, 5, app_vals.get("fridge", 1))
    light = st.sidebar.number_input("LED Bulbs (15W)", 0, 100, app_vals.get("light", 8))
    washing_machine = st.sidebar.number_input("Washing Machine (500W)", 0, 5, app_vals.get("washing_machine", 1))

    appliances_dict = {"fan": fan, "ac": ac, "tv": tv, "fridge": fridge, "light": light, "washing_machine": washing_machine}

elif btype_idx == 1:  # Office
    def_area = active_preset["area"] if active_preset else 5000.0
    def_occ = active_preset["occupancy"] if active_preset else 35
    area = st.sidebar.number_input("Office Floor Area (sq ft)", 500.0, 100000.0, float(def_area), 500.0)
    occupancy = st.sidebar.number_input("Staff / Employees", 1, 1000, int(def_occ), 5)

    st.sidebar.markdown("### 💻 Office Equipment")
    app_vals = active_preset["appliances"] if active_preset else {"workstations": 35, "ac_units": 4, "office_lights": 40, "server_racks": 1, "pantry": 2}
    workstations = st.sidebar.number_input("Workstations & PCs (150W)", 0, 1000, app_vals.get("workstations", 35))
    ac_units = st.sidebar.number_input("Commercial Cassette ACs (2.5 kW)", 0, 50, app_vals.get("ac_units", 4))
    office_lights = st.sidebar.number_input("Commercial LED Panels (40W)", 0, 500, app_vals.get("office_lights", 40))
    server_racks = st.sidebar.number_input("Server Racks (800W)", 0, 20, app_vals.get("server_racks", 1))
    pantry = st.sidebar.number_input("Pantry Appliances (1.2 kW)", 0, 20, app_vals.get("pantry", 2))

    appliances_dict = {"workstations": workstations, "ac_units": ac_units, "office_lights": office_lights, "server_racks": server_racks, "pantry": pantry}

else:  # Mall
    def_area = active_preset["area"] if active_preset else 45000.0
    def_occ = active_preset["occupancy"] if active_preset else (4000 if day_now >= 5 else 2000)
    area = st.sidebar.number_input("Mall Area (sq ft)", 5000.0, 500000.0, float(def_area), 5000.0)
    occupancy = st.sidebar.number_input("Daily Footfall / Occupancy", 100, 50000, int(def_occ), 100)

    st.sidebar.markdown("### 🏬 Commercial Facilities")
    app_vals = active_preset["appliances"] if active_preset else {"chillers": 4, "retail_lights": 800, "escalators": 6, "food_court": 6}
    chillers = st.sidebar.number_input("Central Chillers (45 kW per unit)", 1, 50, app_vals.get("chillers", 4))
    retail_lights = st.sidebar.number_input("Retail Display Lights (50W)", 10, 10000, app_vals.get("retail_lights", 800))
    escalators = st.sidebar.number_input("Escalators & Elevators (12 kW)", 1, 50, app_vals.get("escalators", 6))
    food_court = st.sidebar.number_input("Food Court Refrigeration (3.5 kW)", 0, 50, app_vals.get("food_court", 6))

    appliances_dict = {"chillers": chillers, "retail_lights": retail_lights, "escalators": escalators, "food_court": food_court}

# ================= SIDEBAR: SOLAR & TARIFF =================
st.sidebar.markdown("---")
st.sidebar.markdown("## ☀️ Rooftop Solar & Tariffs")

def_solar = active_preset.get("solar_kw", 0.0) if active_preset else 0.0
solar_kw = st.sidebar.number_input("Rooftop Solar PV Capacity (kW)", 0.0, 500.0, float(def_solar), 1.0, help="Installed solar capacity. Generates power between 06:00 and 18:00.")

tariff_flat = st.sidebar.number_input("Standard Tariff ($ or ₹ per Unit)", 1.0, 100.0, 8.0, 0.5)
use_tou = st.sidebar.checkbox("Enable Time-of-Use (ToU) Dynamic Rates", value=True, help="Applies +30% surcharge during evening peak hours (18:00-22:00) and -20% discount off-peak.")

carbon_region = st.sidebar.selectbox("Grid Carbon Factor", list(config.CARBON_FACTORS.keys()))

# ================= RUN SIMULATION & SESSION STATE =================
calculate_btn = st.sidebar.button("🚀 Run Energy Audit & Optimization", use_container_width=True)

if calculate_btn or "sim_data" not in st.session_state:
    with st.spinner("Analyzing building physics, weather data, and solar production..."):
        temp_val, hum_val, cond_val, is_live, err_msg = fetch_weather_cached(city, active_api_key)

        res = run_energy_simulation(
            model=model,
            temp=temp_val,
            humidity=hum_val,
            day_idx=day_now,
            building_type_idx=btype_idx,
            occupancy=occupancy,
            area=area,
            appliances=appliances_dict,
            solar_kw=solar_kw,
            tariff_flat=tariff_flat,
            use_tou=use_tou,
            carbon_region=carbon_region
        )

        st.session_state["sim_data"] = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "city": city,
            "temp": temp_val,
            "humidity": hum_val,
            "condition": cond_val,
            "is_live": is_live,
            "err_msg": err_msg,
            "building_type": building_type,
            "area": area,
            "occupancy": occupancy,
            "solar_kw": solar_kw,
            "res": res
        }

        # Safe audit logging
        try:
            log_entry = pd.DataFrame([{
                "Timestamp": st.session_state["sim_data"]["timestamp"],
                "City": city,
                "Building_Type": building_type,
                "Area_sqft": area,
                "Solar_kW": solar_kw,
                "Gross_Daily_kWh": round(res["daily_gross_kwh"], 2),
                "Solar_Daily_kWh": round(res["daily_solar_kwh"], 2),
                "Net_Monthly_Cost": round(res["monthly_cost"], 2),
                "Star_Rating": f"{res['stars']}-Star"
            }])
            if os.path.exists(HISTORY_PATH):
                hist_data = pd.concat([pd.read_csv(HISTORY_PATH), log_entry], ignore_index=True)
            else:
                hist_data = log_entry
            hist_data.to_csv(HISTORY_PATH, index=False)
        except Exception:
            pass

data = st.session_state["sim_data"]
res = data["res"]

# ================= STATUS BAR =================
weather_badge = "#00FFA3" if data["is_live"] else "#FFC107"
mode_name = "Live Weather Station" if data["is_live"] else "Standard Climate Mode"

st.markdown(f"""
<div class="kpi-card" style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap;">
    <div>
        <span style="font-size:1.15rem; font-weight:700;">📍 {data['city']}</span> &nbsp;|&nbsp;
        <span style="color:{weather_badge}; font-weight:600;">● {mode_name}</span>: {data['condition']}
    </div>
    <div style="font-size:1.05rem;">
        🌡 <b>{data['temp']:.1f}°C</b> &nbsp;&nbsp;|&nbsp;&nbsp; 💧 <b>{data['humidity']:.0f}% Humidity</b>
    </div>
</div>
""", unsafe_allow_html=True)

if data["err_msg"]:
    st.info(data["err_msg"])

# ================= TOP KPI METRICS =================
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("⚡ Gross Consumption", f"{res['daily_gross_kwh']:.1f} kWh/day", f"{res['monthly_gross_kwh']:,.0f} Units/mo")
with c2:
    solar_delta = f"-{res['daily_solar_kwh']:.1f} kWh/day" if solar_kw > 0 else "No Solar PV"
    st.metric("☀️ Solar Production", f"{res['daily_solar_kwh']:.1f} kWh/day", solar_delta, delta_color="normal")
with c3:
    st.metric("🔌 Net Grid Import", f"{res['daily_net_import_kwh']:.1f} kWh/day", f"{res['monthly_net_import_kwh']:,.0f} Units/mo")
with c4:
    savings_sub = f"Save ${res['monthly_savings']:,.0f}/mo" if res['monthly_savings'] > 0 else "Standard Rate"
    st.metric("💰 Est. Monthly Bill", f"${res['monthly_cost']:,.0f}", savings_sub, delta_color="normal")

# ================= MAIN TABS =================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 24h Load & Solar Profile",
    "☀️ Solar PV & Net-Metering",
    "🔌 Equipment Breakdown",
    "🌱 Carbon & Star Rating (EUI)",
    "💡 Smart Optimization & History"
])

with tab1:
    st.markdown("### 24-Hour Diurnal Electricity Demand vs. Solar Generation")
    st.caption("Hourly load profile generated by the Machine Learning Model combined with rooftop solar bell curve.")

    h_df = res["hourly_df"]
    fig_24h = go.Figure()

    # Gross demand line
    fig_24h.add_trace(go.Scatter(
        x=h_df["Hour"],
        y=h_df["Gross Demand (kW)"],
        mode="lines+markers",
        name="Gross Power Demand (kW)",
        line=dict(color="#FF7043", width=3),
        marker=dict(size=5),
        hovertemplate="Time: %{x}<br>Demand: %{y:.2f} kW<extra></extra>"
    ))

    # Solar generation area
    if solar_kw > 0:
        fig_24h.add_trace(go.Scatter(
            x=h_df["Hour"],
            y=h_df["Solar Generation (kW)"],
            mode="lines",
            name="Rooftop Solar PV (kW)",
            line=dict(color="#00FFA3", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(0, 255, 163, 0.15)",
            hovertemplate="Time: %{x}<br>Solar: %{y:.2f} kW<extra></extra>"
        ))

    # Net grid import line
    fig_24h.add_trace(go.Scatter(
        x=h_df["Hour"],
        y=h_df["Net Grid Import (kW)"],
        mode="lines",
        name="Net Grid Draw (kW)",
        line=dict(color="#42A5F5", width=2, dash="dot"),
        hovertemplate="Time: %{x}<br>Net Import: %{y:.2f} kW<extra></extra>"
    ))

    fig_24h.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.25)",
        font=dict(color="#f1f5f9"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.07)", title="Hour of the Day"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.07)", title="Electrical Load (kW)"),
        margin=dict(l=20, r=20, t=30, b=20),
        height=350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_24h, use_container_width=True)

    peak_hour = int(np.argmax(h_df["Gross Demand (kW)"]))
    st.markdown(f"**⚡ Peak Consumption Hour:** `{peak_hour:02d}:00` (Demand: `{h_df['Gross Demand (kW)'][peak_hour]:.2f} kW`) &nbsp;|&nbsp; **☀️ Peak Solar Hour:** `12:00` (Generation: `{h_df['Solar Generation (kW)'][12]:.2f} kW`)")

with tab2:
    st.markdown("### ☀️ Rooftop Solar & Net-Metering Economics")
    if solar_kw > 0:
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            st.metric("Total Solar Capacity", f"{solar_kw:.1f} kW", "Installed PV")
        with sc2:
            solar_coverage = min(100.0, (res['daily_solar_kwh'] / max(0.1, res['daily_gross_kwh'])) * 100)
            st.metric("Solar Grid Offset", f"{solar_coverage:.1f}%", f"{res['monthly_solar_kwh']:,.0f} kWh/mo generated")
        with sc3:
            st.metric("Estimated Monthly Savings", f"${res['monthly_savings']:,.0f}", f"Annual: ${res['monthly_savings'] * 12:,.0f}/yr")

        st.markdown(f"""
        <div class="kpi-card">
            <h4 style="margin:0 0 8px 0; color:#00FFA3;">⚡ Net-Metering Performance Analysis</h4>
            <p style="margin:0; font-size:0.95rem; line-height:1.5;">
            Your <b>{solar_kw} kW</b> array generates approx. <b>{res['daily_solar_kwh']:.1f} kWh/day</b>. 
            During peak sunlight hours (11:00 AM - 3:00 PM), surplus electricity is fed back to the utility grid at credit rates.
            This setup offsets <b>{solar_coverage:.1f}%</b> of your facility's electrical requirements, saving approx. <b>${res['monthly_savings']:,.0f} every month</b>.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("💡 No Solar PV capacity is currently configured. Adjust the 'Rooftop Solar PV Capacity' slider in the sidebar to simulate potential solar savings.")

with tab3:
    st.markdown("### 🔌 Electrical Infrastructure & Appliance Distribution")
    eq_data = res["equipment_breakdown"]

    col_b1, col_b2 = st.columns([1, 1])
    with col_b1:
        fig_pie = go.Figure(go.Pie(
            labels=list(eq_data.keys()),
            values=list(eq_data.values()),
            hole=0.45,
            marker=dict(colors=["#00FFA3", "#00C9A7", "#42A5F5", "#AB47BC", "#FFA726"]),
            hovertemplate="%{label}: %{value:.1f} kWh (%{percent})<extra></extra>"
        ))
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#f1f5f9"),
            margin=dict(l=10, r=10, t=15, b=10),
            height=300
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b2:
        st.markdown("#### Category Consumption Breakdown")
        tot = sum(eq_data.values())
        if tot > 0:
            for cat, val in eq_data.items():
                pct = (val / tot) * 100
                st.write(f"• **{cat}:** `{val:.1f} kWh/day` ({pct:.1f}%)")
        else:
            st.write("No active appliances loaded.")

with tab4:
    st.markdown("### 🌱 Carbon Emissions & Efficiency Star Rating")
    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown(f"""
        <div class="kpi-card" style="text-align:center;">
            <div style="font-size:0.95rem; color:#94a3b8;">Energy Use Intensity (EUI)</div>
            <div style="font-size:1.8rem; font-weight:700; color:#00FFA3;">{res['eui']:.2f} <span style="font-size:0.9rem;">kWh/sqft/yr</span></div>
            <div class="rating-pill" style="background:rgba(0,255,163,0.15); color:#00FFA3;">
                {'⭐' * res['stars']} {res['rating_text']}
            </div>
        </div>
        """, unsafe_allow_html=True)
    with r2:
        st.markdown(f"""
        <div class="kpi-card" style="text-align:center;">
            <div style="font-size:0.95rem; color:#94a3b8;">Annual Carbon Footprint</div>
            <div style="font-size:1.8rem; font-weight:700; color:#FF7043;">{res['annual_co2_tons']:.2f} <span style="font-size:0.9rem;">Tons CO₂/year</span></div>
            <div style="font-size:0.85rem; color:#cbd5e1; margin-top:5px;">Based on {carbon_region}</div>
        </div>
        """, unsafe_allow_html=True)
    with r3:
        st.markdown(f"""
        <div class="kpi-card" style="text-align:center;">
            <div style="font-size:0.95rem; color:#94a3b8;">Carbon Neutral Offset</div>
            <div style="font-size:1.8rem; font-weight:700; color:#42A5F5;">{res['trees_offset']:,} <span style="font-size:0.9rem;">Trees Needed</span></div>
            <div style="font-size:0.85rem; color:#cbd5e1; margin-top:5px;">To sequester annual building emissions</div>
        </div>
        """, unsafe_allow_html=True)

with tab5:
    st.markdown("### 💡 AI Energy Optimization Action Plan")
    for rec in res["recommendations"]:
        card_class = "rec-card-high" if rec["impact"] == "High" else "rec-card-med"
        st.markdown(f"""
        <div class="{card_class}">
            <h4 style="margin:0 0 6px 0; color:#00FFA3;">{rec['title']} <span style="font-size:0.75rem; background:rgba(255,255,255,0.15); padding:2px 8px; border-radius:10px;">{rec['impact']} Impact</span></h4>
            <p style="margin:0; color:#e2e8f0; font-size:0.92rem;">{rec['desc']}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📁 Audit Logs & Export")
    if os.path.exists(HISTORY_PATH):
        try:
            h_df_all = pd.read_csv(HISTORY_PATH)
            if not h_df_all.empty:
                disp_df = h_df_all.tail(8).iloc[::-1]
                tbl = "<table class='history-table'><thead><tr>"
                for col in disp_df.columns:
                    tbl += f"<th>{col}</th>"
                tbl += "</tr></thead><tbody>"
                for _, r in disp_df.iterrows():
                    tbl += "<tr>"
                    for v in r:
                        tbl += f"<td>{v}</td>"
                    tbl += "</tr>"
                tbl += "</tbody></table>"
                st.markdown(tbl, unsafe_allow_html=True)

                csv_bytes = h_df_all.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Audit History (CSV)", csv_bytes, "building_energy_audit_history.csv", "text/csv")
        except Exception:
            pass

# ================= FOOTER =================
st.markdown("---")
st.markdown("<p style='text-align:center; color:#64748b; font-size:0.85rem;'>Smart Energy AI | Built for Sustainable Energy Optimization & Net-Zero Transition</p>", unsafe_allow_html=True)
