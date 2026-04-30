import streamlit as st
import numpy as np
import joblib
import requests
import pandas as pd
import os
from datetime import datetime

# ================= CONFIG =================
st.set_page_config(page_title="Smart Energy AI", layout="wide")

# ================= LOAD MODEL =================
model = joblib.load("model.pkl")

# ================= API =================
WEATHER_API = "7ac08099485e24e52ce7c3725e107353"

# ================= UI STYLE =================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}
.card {
    background: rgba(255,255,255,0.06);
    padding: 18px;
    border-radius: 15px;
    margin-bottom: 15px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.3);
}
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.05);
    padding: 10px;
    border-radius: 10px;
}
section[data-testid="stSidebar"] {
    background: rgba(0,0,0,0.3);
}
.stButton>button {
    background: linear-gradient(135deg, #00FFA3, #00C9A7);
    color: black;
    border-radius: 10px;
    font-weight: bold;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown("<h1 style='text-align:center;color:#00FFA3;'>⚡ Smart Energy Optimization</h1>", unsafe_allow_html=True)

# ================= TIME =================
now = datetime.now()
hour = now.hour
day = now.weekday()

st.markdown(f"""
<div class="card" style="text-align:center;">
📅 {now.strftime("%A, %d %B %Y")} <br>
⏰ {now.strftime("%I:%M:%S %p")}
</div>
""", unsafe_allow_html=True)

# ================= WEATHER =================
def get_weather(city):
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API}&units=metric"
        res = requests.get(url, timeout=3)
        if res.status_code != 200:
            return 30, 50, "Clear"
        data = res.json()
        return data['main']['temp'], data['main']['humidity'], data['weather'][0]['main']
    except:
        return 30, 50, "Clear"

# ================= LOCATION =================
city = st.text_input("📍 Enter City", "Delhi")

# ================= SIDEBAR =================
st.sidebar.markdown("## 🏢 Building Details")

building_type_name = st.sidebar.selectbox("Building", ["House","Office","Mall"])
building_type = {"House":0,"Office":1,"Mall":2}[building_type_name]

area = float(st.sidebar.text_input("Area (sq ft)", "2000"))

if building_type == 2:
    occupancy = 2000 if day < 5 else 4000
    st.sidebar.info(f"👥 Mall: {occupancy} people")
else:
    occupancy = float(st.sidebar.text_input("People", "4"))

# ================= APPLIANCES =================
st.sidebar.markdown("## 🔌 Appliances")

fan = st.sidebar.number_input("Fans", 0, 20, 2)
ac = st.sidebar.number_input("AC", 0, 10, 1)
tv = st.sidebar.number_input("TV", 0, 10, 1)
fridge = st.sidebar.number_input("Fridge", 0, 5, 1)
light = st.sidebar.number_input("Lights", 0, 50, 5)
washing_machine = st.sidebar.number_input("Washing Machine", 0, 5, 0)

# ================= POWER =================
appliance_power = {
    "fan": 0.075,
    "ac": 1.5,
    "tv": 0.1,
    "fridge": 0.2,
    "light": 0.02,
    "washing_machine": 0.5
}

# ================= PREDICT =================
if st.button("🚀 Predict"):

    temp, humidity, condition = get_weather(city)

    inp = np.array([[temp, humidity, hour, day, building_type, occupancy, area]])
    model_pred = abs(model.predict(inp)[0])

    usage_hours = {
        "fan": 10,
        "ac": 6 if temp > 30 else 2,
        "tv": 4,
        "fridge": 24,
        "light": 6,
        "washing_machine": 1
    }

    appliance_units = (
        fan * appliance_power["fan"] * usage_hours["fan"] +
        ac * appliance_power["ac"] * usage_hours["ac"] +
        tv * appliance_power["tv"] * usage_hours["tv"] +
        fridge * appliance_power["fridge"] * usage_hours["fridge"] +
        light * appliance_power["light"] * usage_hours["light"] +
        washing_machine * appliance_power["washing_machine"] * usage_hours["washing_machine"]
    )

    final_pred = (model_pred * 0.4) + (appliance_units * 0.6)

    if building_type == 0:
        final_pred *= 0.85
    elif building_type == 2:
        final_pred *= 1.2

    final_pred = max(5, final_pred)

    monthly_units = final_pred * 30

    # ================= DASHBOARD =================
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("📊 Energy Dashboard")

    c1, c2, c3 = st.columns(3)
    c1.metric("⚡ Daily Units", f"{final_pred:.2f}")
    c2.metric("🌡 Temperature", f"{temp:.1f}°C")
    c3.metric("💧 Humidity", f"{humidity}%")

    st.success(f"⚡ Approx {final_pred:.2f} units/day")
    st.info(f"Weather: {condition}")

    st.metric("📊 Monthly Units", f"{monthly_units:.0f}")

    st.markdown('</div>', unsafe_allow_html=True)

    # ================= GRAPH =================
    st.markdown("## 📈 Usage Pattern")

    pattern = [0.4,0.3,0.5,0.7,1.0,1.3,1.5,1.2,0.9,0.6]
    df = pd.DataFrame({
        "Time": range(1,11),
        "Usage": [final_pred * p for p in pattern]
    })

    st.line_chart(df.set_index("Time"))

    # ================= HISTORY =================
    new = pd.DataFrame([[city, final_pred, datetime.now()]], columns=["City","Energy","Time"])

    if os.path.exists("history.csv"):
        hist = pd.concat([pd.read_csv("history.csv"), new], ignore_index=True)
    else:
        hist = new

    hist.to_csv("history.csv", index=False)

    st.markdown("## 📊 History")
    st.line_chart(hist["Energy"])
    st.metric("Average", f"{hist['Energy'].mean():.2f}")
    st.metric("Max", f"{hist['Energy'].max():.2f}")

# ================= FOOTER =================
st.markdown("---\n<center>⚡ Smart Energy AI | Save Energy • Save Future</center>", unsafe_allow_html=True)
