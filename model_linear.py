import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================
# 1. Load Dataset
# ==========================================

df = pd.read_csv("T1.csv")

print("Dataset loaded successfully!")
print("Shape:", df.shape)


# ==========================================
# 2. Convert Date/Time
# ==========================================

df["Date/Time"] = pd.to_datetime(
    df["Date/Time"],
    format="%d %m %Y %H:%M"
)


# ==========================================
# 3. Create Time Features
# ==========================================

df["Hour"] = df["Date/Time"].dt.hour
df["Day"] = df["Date/Time"].dt.day
df["Month"] = df["Date/Time"].dt.month
df["DayOfWeek"] = df["Date/Time"].dt.dayofweek


# ==========================================
# 4. Select Features
# ==========================================

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


# ==========================================
# 5. Select Target
# ==========================================

y = df["LV ActivePower (kW)"]


# ==========================================
# 6. Split Dataset
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)


# ==========================================
# 7. Create Linear Regression Model
# ==========================================

model = LinearRegression()


# ==========================================
# 8. Train Model
# ==========================================

model.fit(X_train, y_train)

print("Model training completed!")


# ==========================================
# 9. Make Predictions
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# 10. Evaluate Model
# ==========================================

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

r2 = r2_score(y_test, y_pred)


# ==========================================
# 11. Display Results
# ==========================================

print("\n==============================")
print("Linear Regression Results")
print("==============================")

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R2 Score:", r2)


# ==========================================
# 12. Show Sample Predictions
# ==========================================

results = pd.DataFrame({
    "Actual Power": y_test.values[:10],
    "Predicted Power": y_pred[:10]
})

print("\nSample Predictions:")
print(results)