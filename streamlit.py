import streamlit as st
import pandas as pd
import pickle
import requests
import logging

# Setting up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Loading encoders and features
with open('encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)
with open('features.pkl', 'rb') as f:
    features = pickle.load(f)

# Streamlit app
st.title("Flight Price Prediction")

# Input form
st.header("Enter Flight Details")
airline = st.selectbox("Airline", encoders['Airline'].classes_)
source = st.selectbox("Source", encoders['Source'].classes_)
destination = st.selectbox("Destination", encoders['Destination'].classes_)
total_stops = st.selectbox("Total Stops", [0, 1, 2, 3])
date_of_journey = st.date_input("Date of Journey")
duration = st.text_input("Duration (e.g., '2h 30m')")
additional_info = st.selectbox("Additional Info", encoders['Additional_Info'].classes_)

if st.button("Predict Price"):
    try:
        # Preparing input data
        input_data = {
            "Airline": airline,
            "Source": source,
            "Destination": destination,
            "Total_Stops": total_stops,
            "Date_of_Journey": date_of_journey.strftime('%d/%m/%Y'),
            "Duration": duration,
            "Additional_Info": additional_info
        }
        
        # Sending request to FastAPI
        response = requests.post("http://localhost:8000/predict", json=input_data)
        
        if response.status_code == 200:
            prediction = response.json()['price']
            st.success(f"Predicted Price: ₹{prediction:.2f}")
        else:
            st.error("Error in prediction. Please try again.")
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        st.error(f"An error occurred: {str(e)}")