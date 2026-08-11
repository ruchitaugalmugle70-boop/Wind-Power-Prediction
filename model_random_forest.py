import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# Load dataset
df = pd.read_csv("T1.csv")

# Convert Date/Time
df["Date/Time"] = pd.to_datetime(
    df["Date/Time"],
    format="%d %m %Y %H:%M"
)

# Create time-based features
df["Hour"] = df["Date/Time"].dt.hour
df["Day"] = df["Date/Time"].dt.day
df["Month"] = df["Date/Time"].dt.month
df["DayOfWeek"] = df["Date/Time"].dt.dayofweek

# Features
X = df[
    [
        "Wind Speed (m/s)",
        "Theoretical_Power_Curve (KWh)",
        "Wind Direction (°)",
        "Hour",
        "Day",
        "Month",
        "DayOfWeek"
    ]
]

# Target
y = df["LV ActivePower (kW)"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Random Forest model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

# Train model
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Random Forest Regression Results")
print("--------------------------------")
print("MAE:", mae)
print("MSE:", mse)
print("R2 Score:", r2)

# Save the trained model
joblib.dump(model, "wind_power_model.pkl")

print("Model saved successfully!")