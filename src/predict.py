import joblib
import numpy as np

model = joblib.load("models/uber_price_model.pkl")

def predict_price(distance, hour, day, month, passengers):

    features = np.array([[distance, hour, day, month, passengers]])

    price = model.predict(features)

    return price[0]