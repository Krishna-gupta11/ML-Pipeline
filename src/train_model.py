import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_preprocessing import load_data, clean_data
from feature_engineering import add_features


# Load raw data
df = load_data("data/raw/uber.csv")

# Clean data
df = clean_data(df)

# Feature engineering
df = add_features(df)

# Save processed data
df.to_csv("data/processed/clean_data.csv", index=False)


# Features for model
features = [
    "distance_km",
    "hour",
    "day",
    "month",
    "passenger_count"
]

X = df[features]
y = df["fare_amount"]


# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Decision Tree model
model = DecisionTreeRegressor(
    max_depth=10,
    min_samples_split=10,
    random_state=42
)

model.fit(X_train, y_train)


# Predictions
y_pred = model.predict(X_test)


# Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)


# Save trained model
joblib.dump(model, "models/uber_price_model.pkl")

print("Model saved successfully!")