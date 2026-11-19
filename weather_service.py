"""
Weather service module.
Handles OpenWeatherMap API communication with caching, timeout resilience,
and climate fallbacks.
"""

import os
import urllib.parse
import requests
import streamlit as st

@st.cache_data(ttl=600, show_spinner=False)
def fetch_weather_cached(city_name: str, api_key: str):
    """
    Fetches real-time temperature, humidity, and atmospheric conditions.
    Results are cached for 10 minutes to respect API rate limits.
    """
    clean_city = city_name.strip() if city_name else "Delhi"

    if not api_key:
        return 28.0, 50.0, "Clear (Demo Mode)", False, "No API key configured. Showing standard demo weather."

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
            return 28.0, 50.0, "Clear (Fallback)", False, f"City '{clean_city}' not found on OpenWeatherMap. Using standard climate data."
        else:
            return 28.0, 50.0, "Clear (Fallback)", False, f"Weather API error (status code {response.status_code})."
    except requests.exceptions.Timeout:
        return 28.0, 50.0, "Clear (Fallback)", False, "Weather request timed out. Using offline climate defaults."
    except Exception as e:
        return 28.0, 50.0, "Clear (Fallback)", False, f"Network error: {str(e)}"
