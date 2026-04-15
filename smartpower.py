import pandas as pd
import numpy as np

np.random.seed(42)
n = 2000

# Building type: 0=House, 1=Office, 2=Mall
building_type = np.random.randint(0, 3, n)

data = pd.DataFrame({
    'temperature': np.random.randint(20, 40, n),
    'humidity': np.random.randint(30, 80, n),
    'hour': np.random.randint(0, 24, n),
    'day': np.random.randint(0, 7, n),
    'building_type': building_type,
    'occupancy': np.random.randint(1, 100, n),
    'area': np.random.randint(500, 5000, n)
})

# Energy logic (IMPORTANT)
data['energy'] = (
    data['temperature'] * 2 +
    data['humidity'] * 0.5 +
    data['hour'] * 1.5 +
    data['occupancy'] * 0.8 +
    data['area'] * 0.05 +
    data['building_type'] * 50 +
    np.random.normal(0, 10, n)
)

print(data.head())



import requests

API_KEY = "7ac08099485e24e52ce7c3725e107353"
CITY = "Delhi"

def get_weather():
    url = f"http://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    data = response.json()

    if 'main' not in data:
        print("API Error:", data)
        return 30, 50   # fallback values

    temp = data['main']['temp']
    humidity = data['main']['humidity']

    return temp, humidity



from datetime import datetime

def get_time():
    now = datetime.now()
    hour = now.hour
    day = now.weekday()
    return hour, day


temp, humidity = get_weather()
hour, day = get_time()

# Example fixed values (you can improve later)
building_type = 1   # office
occupancy = 50
area = 2000

input_data = [[temp, humidity, hour, day, building_type, occupancy, area]]

# Train a simple model
from sklearn.ensemble import RandomForestRegressor
best_model = RandomForestRegressor(random_state=42)
best_model.fit(data[['temperature', 'humidity', 'hour', 'day', 'building_type', 'occupancy', 'area']], data['energy'])

prediction = best_model.predict(input_data)

print("\n--- REAL-TIME PREDICTION ---")
print("Temperature:", temp)
print("Humidity:", humidity)
print("Predicted Energy:", prediction[0])

def suggest(temp, hour):
    print("\nSuggestions:")
    
    if temp > 30:
        print("- Reduce AC usage")
    if 12 <= hour <= 16:
        print("- Avoid peak hour usage")

suggest(temp, hour)




import joblib

joblib.dump(best_model, "model.pkl")
print("Model saved successfully ✅")