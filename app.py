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

# ================= WEATHER (FORECAST FIX) =================
def get_weather(city):
    try:
        url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={WEATHER_API}&units=metric"
        res = requests.get(url)

        if res.status_code != 200:
            return None, None, None, "Invalid city"

        data = res.json()

        temps = [i['main']['temp'] for i in data['list'][:8]]  # 24 hrs
        humidity = data['list'][0]['main']['humidity']
        condition = data['list'][0]['weather'][0]['main']

        avg_temp = sum(temps) / len(temps)

        return avg_temp, humidity, condition, None

    except:
        return None, None, None, "API Error"

# ================= LOCATION (FIXED) =================
st.markdown("### 📍 Enter Location")

city = st.text_input("City", "Delhi")

if city.strip() == "":
    st.warning("Please enter a valid city")

# ================= SIDEBAR =================
st.sidebar.header("🏢 Controls")

building_type_name = st.sidebar.selectbox("Building", ["House","Office","Mall"])
building_type = {"House":0,"Office":1,"Mall":2}[building_type_name]

area_input = st.sidebar.text_input("Area (sq ft)", "2000")

try:
    area = float(area_input)
except:
    st.sidebar.error("Invalid Area")
    area = 2000

# ================= OCCUPANCY (UPDATED) =================
if building_type == 2:  # Mall
    if day < 5:
        occupancy = 2000
        st.sidebar.info("👥 Mall (Weekday): 2000 people")
    else:
        occupancy = 4000
        st.sidebar.info("👥 Mall (Weekend): 4000 people")
else:
    occ_input = st.sidebar.text_input("People", "4")
    try:
        occupancy = float(occ_input)
    except:
        st.sidebar.error("Invalid People")
        occupancy = 4

# ================= PREDICTION =================
if st.button("🚀 Predict"):

    if city.strip() == "":
        st.error("Enter a valid city first")
    else:
        with st.spinner("Analyzing smart energy usage..."):

            temp, humidity, condition, error = get_weather(city)

            if error:
                st.error(error)
            else:
                inp = np.array([[temp, humidity, hour, day, building_type, occupancy, area]])
                base_pred = model.predict(inp)[0]
                base_pred = max(20, min(abs(base_pred), 400))

                # ================= SMART SCALING =================

                if building_type == 0:
                    usage_hours = 8
                    base_factor = 0.6
                elif building_type == 1:
                    usage_hours = 10
                    base_factor = 1.0
                else:
                    usage_hours = 14
                    base_factor = 1.4

                temp_factor = 1.2 if temp > 32 else 1.0
                occupancy_factor = 1 + (occupancy / 50)

                final_pred = base_pred * base_factor * temp_factor * occupancy_factor
                final_pred = (final_pred / 24) * usage_hours

                # caps
                if building_type == 0:
                    final_pred = min(final_pred, 50)
                elif building_type == 1:
                    final_pred = min(final_pred, 300)
                else:
                    final_pred = min(final_pred, 600)

                final_pred = max(10, final_pred)

        # ================= DASHBOARD =================
        st.subheader("📊 Smart Energy Dashboard")

        c1,c2,c3 = st.columns(3)
        c1.metric("⚡ Daily Units", f"{final_pred:.2f}")
        c2.metric("🌡 Temp", f"{temp:.1f}°C")
        c3.metric("💧 Humidity", f"{humidity}%")

        st.info(f"Weather: {condition}")
        st.metric("📊 Monthly Estimate", f"{final_pred * 30:.0f} units")

        # efficiency
        score = 9 if final_pred <150 else 6 if final_pred<300 else 3
        st.progress(score*10)

        if final_pred <150:
            st.success("🟢 Efficient Usage")
        elif final_pred <300:
            st.warning("🟠 Moderate Usage")
        else:
            st.error("🔴 High Usage")

        # ================= DAILY PATTERN =================
        st.markdown("### 📈 Daily Usage Pattern")

        pattern = [0.4,0.3,0.5,0.7,1.0,1.3,1.5,1.2,0.9,0.6]
        simulated = [final_pred * p for p in pattern]

        st.line_chart(pd.DataFrame(simulated, columns=["Usage"]))

        # ================= RECOMMENDATIONS =================
        st.markdown("### 💡 Smart Recommendations")

        if final_pred > 300:
            st.warning("Reduce heavy appliances")
        if temp > 32:
            st.warning("Set AC to 24–26°C")
        if building_type == 0:
            st.info("Turn off unused home appliances")
        if building_type == 1:
            st.info("Optimize office energy usage")
        if building_type == 2:
            st.info("Use efficient cooling for large spaces")

        st.success("Use LED & energy-efficient appliances")

        # ================= HISTORY =================
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

        st.markdown("### 📊 Usage History")
        st.line_chart(hist["Energy"])
        st.metric("Average", f"{hist['Energy'].mean():.2f}")
        st.metric("Max", f"{hist['Energy'].max():.2f}")
        st.bar_chart(hist.tail(5)["Energy"])

        st.download_button("📥 Download Report", hist.to_csv(), "report.csv")

# ================= FOOTER =================
st.markdown("---\n<center>⚡ Smart Energy AI | Save Energy • Save Future</center>", unsafe_allow_html=True)


