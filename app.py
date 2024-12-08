import streamlit as st
from joblib import load
import numpy as np

# Load the trained model
model_filename = "C:\\Users\\Sahil\\Desktop\\Project\\taxi_fare_model.pkl"
model = load(model_filename)

# Define the Streamlit App
st.title("Taxi Fare Prediction")
st.markdown("""
Predict taxi fares based on trip details using a trained machine learning model.
""")

# Input Fields for User Input
st.sidebar.header("Enter Trip Details:")
pickup_longitude = st.sidebar.number_input("Pickup Longitude", min_value=-180.0, max_value=180.0, value=-73.95, step=0.01)
pickup_latitude = st.sidebar.number_input("Pickup Latitude", min_value=-90.0, max_value=90.0, value=40.78, step=0.01)
dropoff_longitude = st.sidebar.number_input("Dropoff Longitude", min_value=-180.0, max_value=180.0, value=-73.96, step=0.01)
dropoff_latitude = st.sidebar.number_input("Dropoff Latitude", min_value=-90.0, max_value=90.0, value=40.79, step=0.01)
passenger_count = st.sidebar.slider("Passenger Count", min_value=1, max_value=10, value=1)

# Predict Fare Button
if st.sidebar.button("Predict Fare"):
    # Prepare the input for the model
    input_features = np.array([[pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count]])
    
    # Make the prediction
    predicted_fare = model.predict(input_features)[0]
    
    # Display the prediction
    st.success(f"Predicted Taxi Fare: ${predicted_fare:.2f}")

# Footer
st.markdown("""
---
Developed with ❤️ using **Streamlit**.
""")
