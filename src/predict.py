"""
predict.py — Load saved model and run inference on a single customer dict.
Useful for testing the model outside of the Streamlit app.

Usage:
    python -c "from src.predict import predict_customer; predict_customer({...})"
"""
import os
import pickle
import numpy as np


MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'churn_model.pkl')


def load_model():
    with open(MODEL_PATH, 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['features']


def predict_customer(customer: dict) -> dict:
    """
    Predict churn for a single customer.

    Parameters
    ----------
    customer : dict
        Keys must match the 19 training features (already label-encoded).

    Returns
    -------
    dict with keys: prediction (0/1), churn_probability, risk_level
    """
    model, features = load_model()
    row = np.array([[customer[f] for f in features]])

    pred = model.predict(row)[0]
    prob = model.predict_proba(row)[0][1]
    risk = 'High' if prob >= 0.6 else ('Medium' if prob >= 0.35 else 'Low')

    return {
        'prediction': int(pred),
        'churn_probability': round(float(prob), 4),
        'risk_level': risk,
    }
