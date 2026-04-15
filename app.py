import streamlit as st
import numpy as np
import joblib
import requests
import pandas as pd
import os

# Load model
model = joblib.load("model.pkl")

# API key
API_KEY = "7ac08099485e24e52ce7c3725e107353"

# 🌙 DARK THEME + STYLE
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}
h1, h2, h3 {
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# 🌍 Auto location
def get_location():
    try:
        data = requests.get("http://ip-api.com/json/").json()
        return data.get('city', 'Delhi')
    except:
        return "Delhi"

# 📍 Pincode → City
def get_city_from_pincode(pincode):
    try:
        url = f"http://api.openweathermap.org/geo/1.0/zip?zip={pincode},IN&appid={API_KEY}"
        data = requests.get(url).json()
        return data.get("name", "Delhi")
    except:
        return "Delhi"

# 🌦 Weather
def get_weather(city):
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
        data = requests.get(url).json()
        return data['main']['temp'], data['main']['humidity']
    except:
        return 30, 50

# PAGE CONFIG
st.set_page_config(page_title="Smart Energy AI", layout="centered")

# HEADER
st.markdown("""
<h1 style='color:#00FFAA;'>⚡ Smart Energy Optimization System</h1>
<p style='text-align:center;'>AI-powered energy prediction & optimization</p>
""", unsafe_allow_html=True)

st.markdown("---")

# 📍 LOCATION
st.subheader("📍 Location Input")

mode = st.radio("Choose input type", ["City", "Pincode"])

if mode == "City":
    city = st.text_input("Enter City", get_location())
else:
    pincode = st.text_input("Enter Pincode", "160055")
    city = get_city_from_pincode(pincode) if pincode else "Delhi"

st.write(f"📌 Using location: **{city}**")

st.markdown("---")

# SIDEBAR
st.sidebar.header("🏢 Building Controls")

building_type_name = st.sidebar.selectbox("Building Type", ["House", "Office", "Mall"])
bt_map = {"House": 0, "Office": 1, "Mall": 2}
building_type = bt_map[building_type_name]

area = float(st.sidebar.text_input("Area (sq ft)", "2000"))

if building_type_name == "House":
    occupancy = float(st.sidebar.text_input("People", "4"))
else:
    occupancy = 50 if building_type_name == "Office" else 200
    st.sidebar.info(f"👥 Occupancy: {occupancy}")

# TIME
st.subheader("⏰ Time Details")

col1, col2 = st.columns(2)

with col1:
    hour = int(st.text_input("Hour", "12"))

with col2:
    day = int(st.text_input("Day", "2"))

st.markdown("---")

# BUTTON
if st.button("🚀 Predict Energy"):

    with st.spinner("🔄 Predicting energy usage..."):
        temp, humidity = get_weather(city)

        input_data = np.array([[temp, humidity, hour, day, building_type, occupancy, area]])
        prediction = model.predict(input_data)[0]

        if prediction < 0:
            prediction = abs(prediction)

        # Clamp realistic
        prediction = max(50, min(prediction, 500))

    # RESULTS
    st.markdown("## ⚡ Results")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("⚡ Energy (units)", f"{prediction:.2f}")

    with col2:
        score = 9 if prediction < 150 else 6 if prediction < 300 else 3
        st.metric("⚡ Efficiency", f"{score}/10")

    # WEATHER
    st.markdown("### 🌦 Weather")
    st.write(f"🌡 Temp: {temp}°C | 💧 Humidity: {humidity}%")

    # ALERT LEVEL
    if prediction > 350:
        st.error("🔴 CRITICAL USAGE")
    elif prediction > 250:
        st.warning("🟠 MODERATE USAGE")
    else:
        st.success("🟢 EFFICIENT USAGE")

    # SMART TIPS
    st.markdown("### 🚨 Smart Recommendations")

    if prediction > 300:
        st.warning("Reduce heavy appliances")

    if temp > 32:
        st.warning("Set AC to 24–26°C")

    if 12 <= hour <= 16:
        st.warning("Avoid peak-hour usage")

    if building_type_name == "House" and occupancy > 6:
        st.warning("Turn off unused devices")

    if area > 4000:
        st.warning("Use zone cooling")

    st.info("Use LED and efficient appliances")

    # TREND
    st.markdown("### 📊 Energy Trend")
    trend = [prediction + i*5 for i in range(10)]
    st.line_chart(pd.DataFrame(trend, columns=["Trend"]))

    # FORECAST
    st.markdown("### 🔮 Forecast")
    future = [prediction + i*10 for i in range(5)]
    st.line_chart(pd.DataFrame(future, columns=["Forecast"]))

    # SAVE HISTORY
    new = pd.DataFrame([[city, prediction]], columns=["City", "Energy"])

    if os.path.exists("history.csv"):
        new.to_csv("history.csv", mode='a', header=False, index=False)
    else:
        new.to_csv("history.csv", index=False)

    # SHOW HISTORY
    st.markdown("### 📈 History")
    hist = pd.read_csv("history.csv")
    st.line_chart(hist["Energy"])

    # DOWNLOAD
    st.download_button("📥 Download Report", hist.to_csv(), "energy_report.csv")

    st.markdown("---")
    st.success("💚 Optimize energy → Save money + environment")