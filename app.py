import streamlit as st
import pandas as pd
import joblib
import numpy as np
import folium
from streamlit_folium import folium_static
from geopy.geocoders import Nominatim

# Load trained model and encoders
model = joblib.load("real_estate_model.pkl")
scaler = joblib.load("scaler.pkl")
le_city = joblib.load("le_city.pkl")
le_property = joblib.load("le_property.pkl")
le_area = joblib.load("le_area.pkl")
le_amenities = joblib.load("le_amenities.pkl")

# Initialize geolocator
geolocator = Nominatim(user_agent="real_estate_app")

# Custom Styling
st.markdown("""
    <style>
        body, .main, .stApp {
            background-color: #8B4513 !important; /* Brown background */
            color: #ffffff !important;
        }
        .title {
            font-size: 40px;
            font-weight: bold;
            color: #ffffff;
            text-align: center;
            background-color: #D2691E;
            padding: 15px;
            border-radius: 10px;
        }
        .stSidebar {
            background-color: #5A3825 !important; /* Dark brown sidebar */
            color: #ffffff !important;
            padding: 20px;
            border-radius: 10px;
        }
        .stButton>button {
            background-color: #D2691E;
            color: white;
            font-size: 18px;
            padding: 10px 20px;
            border-radius: 10px;
        }
        .stButton>button:hover {
            background-color: #A0522D;
        }
        .custom-box {
            background-color: #5A3825;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.markdown("<div class='stSidebar'><h2>🔍 Search Property</h2></div>", unsafe_allow_html=True)
st.sidebar.write("Enter property details below to predict the price per unit area.")

# User Inputs
bedrooms = st.sidebar.number_input("🛏️ Number of Bedrooms", min_value=1, max_value=10, value=2)
total_floors = st.sidebar.number_input("🏢 Total Floors", min_value=1, max_value=50, value=10)
age = st.sidebar.number_input("🏚️ Property Age (Years)", min_value=0, max_value=100, value=5)
max_area = st.sidebar.number_input("📏 Max Area (sqft)", min_value=100, max_value=10000, value=500)
min_area = st.sidebar.number_input("📐 Min Area (sqft)", min_value=100, max_value=10000, value=300)

# Dropdown Inputs
city = st.sidebar.selectbox("📍 Select City", le_city.classes_)
property_type = st.sidebar.selectbox("🏠 Property Type", le_property.classes_)
area = st.sidebar.selectbox("🗺️ Area", le_area.classes_)
amenities = st.sidebar.text_input("✨ Amenities (comma-separated)", "Swimming Pool, Gym")

# Convert inputs to encoded values
city_encoded = le_city.transform([city])[0]
property_type_encoded = le_property.transform([property_type])[0]
area_encoded = le_area.transform([area])[0]

if amenities in le_amenities.classes_:
    amenities_encoded = le_amenities.transform([amenities])[0]
else:
    amenities_encoded = le_amenities.transform(["Unknown"])[0]

# Prepare input data
input_data = pd.DataFrame([[bedrooms, total_floors, age, max_area, min_area, city_encoded, property_type_encoded, area_encoded, amenities_encoded]],
                          columns=["BEDROOM_NUM", "TOTAL_FLOOR", "AGE", "MAX_AREA_SQFT", "MIN_AREA_SQFT", "CITY", "PROPERTY_TYPE", "AREA", "AMENITIES"])

# Standardize numeric features
num_features = ["BEDROOM_NUM", "TOTAL_FLOOR", "AGE", "MAX_AREA_SQFT", "MIN_AREA_SQFT"]
input_data[num_features] = scaler.transform(input_data[num_features])

# Page Header
st.markdown("<p class='title'>🏡 Real Estate Price Prediction</p>", unsafe_allow_html=True)

# Prediction Button
if st.sidebar.button("🔍 Predict Price"):
    prediction = model.predict(input_data)[0]
    st.markdown(f"<div class='custom-box'><h2>🏠 Predicted Square Foot Price:</h2><h1>₹{prediction:,.2f}</h1></div>", unsafe_allow_html=True)
    
    # Get Accurate Latitude & Longitude for Selected Area
    location = geolocator.geocode(f"{area}, {city}, India")
    if location:
        latitude, longitude = location.latitude, location.longitude
    else:
        latitude, longitude = 19.0760, 72.8777  # Default to Mumbai

    # Interactive Map
    st.write("📍 **Property Location on Map:**")
    map_ = folium.Map(location=[latitude, longitude], zoom_start=14)
    folium.Marker(
        location=[latitude, longitude],
        popup=f"🏠 Predicted Price: ₹{prediction:,.2f}",
        tooltip="Click for details",
        icon=folium.Icon(color="blue", icon="home"),
    ).add_to(map_)
    folium_static(map_)

# Price Trends Section
with st.expander("📈 View Price Trends"):
    st.write("🔹 Prices are influenced by location, amenities, and property type.")
    st.line_chart(np.random.randint(5000, 50000, size=10))  # Placeholder random data for trends

# Footer
st.markdown("""
    <hr>
    <p style='text-align: center; font-size: 18px; color: #ffffff;'>
    Made with ❤️ using Streamlit & Folium
    </p>
""", unsafe_allow_html=True)
