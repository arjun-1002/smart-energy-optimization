Use the bellow link
https://smart-energy-optimization-tjwebrzbvsjfnceqjpsgeb.streamlit.app/

⚡ Smart Energy Optimization using Machine Learning
📌 Overview
This project is an AI-powered energy prediction system that estimates electricity consumption based on real-world factors like weather, building type, and occupancy.

The goal is to help users understand, predict, and optimize energy usage for better efficiency.

🚀 Features
🔮 Energy Prediction (ML-based)

🌡 Weather Integration (Temperature & Humidity)

🏢 Building Type Support

House

Office

Mall

👥 Dynamic Occupancy Logic

Mall: Weekday (2000), Weekend (4000)

📊 Daily & Monthly Energy Estimates

📈 Usage Pattern Graph

💡 Smart Recommendations

🕒 Time-based Energy Scaling

📁 History Tracking & CSV Export

🧠 Machine Learning Model
Model: Random Forest Regressor

Type: Regression Model

Loaded using: joblib

Purpose:

Predict energy consumption based on input features

📥 Input Parameters
Temperature (from API)

Humidity (from API)

Building Type

Area (sq ft)

Number of People

Time & Day

⚙️ Technologies Used
Python 🐍

Streamlit 🎨 (Frontend UI)

Scikit-learn 🤖 (ML Model)

Pandas & NumPy 📊

OpenWeather API 🌐

🧩 How It Works
User enters:

City

Building details

App fetches weather forecast

ML model predicts base energy

Custom logic adjusts:

Temperature effect

Occupancy

Usage hours

Final output:

Daily units

Monthly estimate

Graph + recommendations

📊 Output
⚡ Daily Energy Consumption

📅 Monthly Estimate

📈 Energy Trend Graph

💡 Optimization Suggestions

▶️ How to Run
pip install -r requirements.txt
streamlit run app.py
⚠️ Challenges Faced
Auto-location not working in cloud

Unrealistic high predictions

Data inconsistency

✅ Solutions Implemented
Manual city input

Forecast-based weather data

Smart scaling logic

Value capping for realistic output

🔮 Future Improvements

IoT-based real-time data

React + FastAPI deployment

Electricity bill calculation

AI chatbot assistant

👨‍💻 Author
Arjun Rajesh

⭐ Conclusion
This project combines Machine Learning + Real-world logic to create a smart, adaptive system for energy prediction and optimization.
