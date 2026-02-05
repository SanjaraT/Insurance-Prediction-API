from fastapi import FastAPI,Path,HTTPException,Query
import joblib
import pandas as pd
from fastapi.responses import JSONResponse
import numpy as np
from schema.user_input import UserInput
from model.predict import predict_output, model, MODEL_VERSION

#Import ML model
model = joblib.load("model/insurance_model.pkl")

app = FastAPI()
@app.get("/")
def home():
    return {"message": "Insurance Prediction API is running"}

MODEL_VERSION = '1.0.0'
@app.get('/health')
def health_check():
    return{
        'status' : 'OKAY',
        'version': MODEL_VERSION,
        'update': model is not None

    }
@app.post("/predict")
def predict_premium(data: UserInput):
    
    user_input = {
        'bmi': data.bmi,
        'age_group': data.age_group,
        'lifestyle_risk': data.lifestyle_risk,
        'city_tier': data.city_tier,
        'income_lpa': data.income_lpa,
        'occupation': data.occupation
    }

    try:
        result = predict_output(user_input)
        return JSONResponse(status_code=200, content=result)

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )