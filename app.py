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

# ================= API KEY =================
WEATHER_API = "7ac08099485e24e52ce7c3725e107353"

# ================= UI =================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}
.card {
    background: rgba(255,255,255,0.05);
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 15px;
}
.stButton>button {
    background: linear-gradient(135deg, #00FFA3, #00C9A7);
    color: black;
    border-radius: 10px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown("<h1 style='text-align:center;color:#00FFA3;'>⚡ Smart Energy Optimization</h1>", unsafe_allow_html=True)

# ================= DATE =================
now = datetime.now()
hour = now.hour
day = now.weekday()

st.markdown(f"""
<div class="card">
📅 {now.strftime("%A")} | {now.strftime("%d %B %Y")} <br>
⏰ {now.strftime("%I:%M %p")}
</div>
""", unsafe_allow_html=True)

# ================= FUNCTIONS =================
def get_location():
    try:
        return requests.get("http://ip-api.com/json/").json().get("city", "Delhi")
    except:
        return "Delhi"

def get_city_from_pincode(pincode):
    try:
        url = f"http://api.openweathermap.org/geo/1.0/zip?zip={pincode},IN&appid={WEATHER_API}"
        res = requests.get(url).json()
        return res["name"] if "name" in res else "Delhi"
    except:
        return "Delhi"

def get_weather(city):
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API}&units=metric"
        data = requests.get(url).json()
        if "main" in data:
            return data['main']['temp'], data['main']['humidity'], data['weather'][0]['main']
        return 30, 50, "Clear"
    except:
        return 30, 50, "Clear"

# ================= LOCATION =================
mode = st.radio("Location Input", ["Auto", "City", "Pincode"])

if mode == "Auto":
    city = get_location()
elif mode == "City":
    city = st.text_input("City", "Delhi")
else:
    pincode = st.text_input("Pincode")
    city = get_city_from_pincode(pincode)

st.success(f"Using: {city}")

# ================= SIDEBAR =================
st.sidebar.header("🏢 Controls")

building_type = {"House":0,"Office":1,"Mall":2}[st.sidebar.selectbox("Building",["House","Office","Mall"])]

area_input = st.sidebar.text_input("Area (sq ft)", "2000")
occ_input = st.sidebar.text_input("People", "4")

# validation
try:
    area = float(area_input)
except:
    st.sidebar.error("Invalid Area")
    area = 2000

try:
    occupancy = float(occ_input)
except:
    st.sidebar.error("Invalid People")
    occupancy = 4

# ================= PREDICTION =================
if st.button("🚀 Predict"):

    with st.spinner("Analyzing smart energy usage..."):
        temp, humidity, condition = get_weather(city)

        inp = np.array([[temp, humidity, hour, day, building_type, occupancy, area]])
        base_pred = model.predict(inp)[0]
        base_pred = max(50, min(abs(base_pred), 500))

        # ================= SMART SCALING =================
        hour_factor = 1.1 if 12 <= hour <= 16 else 1.0
        temp_factor = 1.2 if temp > 32 else 1.0
        occupancy_factor = 1 + (occupancy / 30)

        final_pred = base_pred * hour_factor * temp_factor * occupancy_factor
        final_pred = max(50, min(final_pred, 600))

    # ================= DASHBOARD =================
    st.subheader("📊 Smart Energy Dashboard")

    c1,c2,c3 = st.columns(3)
    c1.metric("⚡ Daily Units", f"{final_pred:.2f}")
    c2.metric("🌡 Temp", f"{temp}°C")
    c3.metric("💧 Humidity", f"{humidity}%")

    st.info(f"Weather: {condition}")

    # Monthly estimate (no ₹)
    st.metric("📊 Monthly Estimate", f"{final_pred * 30:.0f} units")

    # efficiency
    score = 9 if final_pred <150 else 6 if final_pred<300 else 3
    st.progress(score*10)

    if final_pred<150:
        st.success("🟢 Efficient Usage")
    elif final_pred<300:
        st.warning("🟠 Moderate Usage")
    else:
        st.error("🔴 High Usage")

    # ================= DAILY PATTERN =================
    st.markdown("### 📈 Daily Usage Pattern")

    pattern = [0.5, 0.4, 0.6, 0.8, 1.2, 1.5, 1.3, 1.0, 0.8, 0.6]
    simulated = [final_pred * p for p in pattern]

    st.line_chart(pd.DataFrame(simulated, columns=["Usage"]))

    # ================= RECOMMENDATIONS =================
    st.markdown("### 💡 Smart Recommendations")

    if final_pred > 300:
        st.warning("Reduce heavy appliances")

    if temp > 32:
        st.warning("Set AC to 24–26°C")

    if 12 <= hour <= 16:
        st.warning("Avoid peak hours")

    if area > 4000:
        st.warning("Use zone cooling")

    st.success("Use LED & energy-efficient appliances")

    # ================= SAFE HISTORY =================
    new = pd.DataFrame([[city, final_pred, datetime.now()]], columns=["City","Energy","Time"])

    if os.path.exists("history.csv"):
        try:
            old = pd.read_csv("history.csv")

            if len(old.columns)==2:
                old["Time"]=datetime.now()

            hist = pd.concat([old,new], ignore_index=True)
            hist.to_csv("history.csv", index=False)

        except:
            new.to_csv("history.csv", index=False)
            hist = new
    else:
        new.to_csv("history.csv", index=False)
        hist = new

    # ================= HISTORY =================
    st.markdown("### 📊 Usage History")

    st.line_chart(hist["Energy"])

    st.metric("Average", f"{hist['Energy'].mean():.2f}")
    st.metric("Max", f"{hist['Energy'].max():.2f}")

    st.bar_chart(hist.tail(5)["Energy"])

    st.download_button("📥 Download Report", hist.to_csv(), "report.csv")

# ================= FOOTER =================
st.markdown("---\n<center>⚡ Smart Energy AI | Save Energy • Save Future</center>", unsafe_allow_html=True)