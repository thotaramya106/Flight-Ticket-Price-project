from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle
import logging
from datetime import datetime

# Setting up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI setup
app = FastAPI()

# Pydantic model for input validation
class FlightData(BaseModel):
    Airline: str
    Source: str
    Destination: str
    Total_Stops: int
    Date_of_Journey: str
    Duration: str
    Additional_Info: str

# Loading model, scaler, encoders, and features
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)
with open('features.pkl', 'rb') as f:
    features = pickle.load(f)

def duration_to_minutes(duration):
    try:
        hours, minutes = 0, 0
        if 'h' in duration:
            hours = int(duration.split('h')[0])
            if 'm' in duration.split('h')[1]:
                minutes = int(duration.split('h')[1].split('m')[0])
        elif 'm' in duration:
            minutes = int(duration.split('m')[0])
        return hours * 60 + minutes
    except:
        return 0

@app.post("/predict")
async def predict_price(flight: FlightData):
    try:
        # Preparing input data
        input_data = pd.DataFrame([{
            'Airline': flight.Airline,
            'Source': flight.Source,
            'Destination': flight.Destination,
            'Total_Stops': flight.Total_Stops,
            'Date_of_Journey': flight.Date_of_Journey,
            'Duration': flight.Duration,
            'Additional_Info': flight.Additional_Info
        }])
        
        # Preprocessing input
        input_data['Date_of_Journey'] = pd.to_datetime(input_data['Date_of_Journey'], format='%d/%m/%Y')
        input_data['Day'] = input_data['Date_of_Journey'].dt.day
        input_data['Month'] = input_data['Date_of_Journey'].dt.month
        input_data['Weekday'] = input_data['Date_of_Journey'].dt.weekday
        input_data['Duration_Minutes'] = input_data['Duration'].apply(duration_to_minutes)
        
        # Encoding categorical variables
        for col in ['Airline', 'Source', 'Destination', 'Additional_Info']:
            input_data[col] = encoders[col].transform(input_data[col])
        
        # Selecting features and scaling
        X = input_data[features]
        X_scaled = scaler.transform(X)
        
        # Making prediction
        prediction = model.predict(X_scaled)[0]
        
        return {"price": float(prediction)}
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)