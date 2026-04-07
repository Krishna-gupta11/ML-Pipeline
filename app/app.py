import streamlit as st
import pandas as pd
import joblib
import datetime
import folium
from streamlit_folium import st_folium
from math import radians, sin, cos, sqrt, atan2

st.set_page_config(
    page_title="Uber Fare Prediction",
    page_icon="🚕",
    layout="wide"
)

# Load model
try:
    model = joblib.load("models/uber_price_model.pkl")
except Exception as e:
    st.error(f"Error loading model: {e}")

st.title("🚕 Uber Fare Prediction Dashboard")
st.markdown("Predict Uber ride fares using Machine Learning")

# Sidebar
st.sidebar.header("Ride Details")

pickup_lat = st.sidebar.number_input("Pickup Latitude", value=40.7614327)
pickup_lon = st.sidebar.number_input("Pickup Longitude", value=-73.9798156)
drop_lat = st.sidebar.number_input("Dropoff Latitude", value=40.6513111)
drop_lon = st.sidebar.number_input("Dropoff Longitude", value=-73.8803331)

passenger_count = st.sidebar.slider("Passengers", 1, 6, 1)

pickup_time = st.sidebar.datetime_input(
    "Pickup Time",
    datetime.datetime.now()
)

hour = pickup_time.hour
day = pickup_time.day
month = pickup_time.month


# Distance function
def haversine(lat1, lon1, lat2, lon2):

    R = 6371

    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))

    return R * c


distance_km = haversine(pickup_lat, pickup_lon, drop_lat, drop_lon)


# Session state
if "prediction" not in st.session_state:
    st.session_state.prediction = None


# Button
if st.sidebar.button("Predict Fare"):

    input_df = pd.DataFrame([{
        "distance_km": distance_km,
        "hour": hour,
        "day": day,
        "month": month,
        "passenger_count": passenger_count
    }])

    with st.spinner("Calculating best Uber price... 🚕"):
        prediction = model.predict(input_df)[0]
        st.session_state.prediction = prediction


# OUTPUT (outside button block)

if st.session_state.prediction is not None:

    st.subheader("💰 Predicted Fare")

    st.metric(
        label="Estimated Fare",
        value=f"${st.session_state.prediction:.2f}"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Ride Information")
        st.write("Distance (km):", round(distance_km, 2))
        st.write("Passengers:", passenger_count)
        st.write("Pickup Hour:", hour)

    with col2:

        st.subheader("Pickup → Dropoff Map")

        m = folium.Map(location=[pickup_lat, pickup_lon], zoom_start=12)

        folium.Marker(
            [pickup_lat, pickup_lon],
            tooltip="Pickup",
            icon=folium.Icon(color="green")
        ).add_to(m)

        folium.Marker(
            [drop_lat, drop_lon],
            tooltip="Dropoff",
            icon=folium.Icon(color="red")
        ).add_to(m)

        st_folium(m, width=500, height=400)