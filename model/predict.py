import joblib
import pandas as pd


#Import ML model
model = joblib.load("model/insurance_model.pkl")

# MLFlow
MODEL_VERSION = '1.0.0'

class_labels = model.classes_.tolist()

RISK_MAP = {
    0: "low",
    1: "medium",
    2: "high"
}

def predict_output(user_input: dict):

    df = pd.DataFrame([user_input])

    predicted_class = model.predict(df)[0]
    probabilities = model.predict_proba(df)[0]

    confidence = float(max(probabilities))

    class_probs = {
        RISK_MAP[int(cls)]: round(float(prob), 4)
        for cls, prob in zip(class_labels, probabilities)
    }

    return {
        "predicted_category": RISK_MAP[int(predicted_class)],
        "confidence": round(confidence, 4),
        "class_probabilities": class_probs
    }