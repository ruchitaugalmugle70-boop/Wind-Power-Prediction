import os
import sys
from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, template_folder="Templates")

# Load trained model safely
model_path = os.path.join(BASE_DIR, "wind_power_model.pkl")
try:
    model = joblib.load(model_path)
except Exception as e:
    print(f"[ERROR] Failed to load model from {model_path}: {e}")
    sys.exit(1)


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    wind_speed = data["wind_speed"]
    theoretical_power = data["theoretical_power"]
    wind_direction = data["wind_direction"]
    hour = data["hour"]
    day = data["day"]
    month = data["month"]
    day_of_week = data["day_of_week"]

    features = np.array([[
        wind_speed,
        theoretical_power,
        wind_direction,
        hour,
        day,
        month,
        day_of_week
    ]])

    prediction = model.predict(features)

    return jsonify({
        "predicted_power": float(prediction[0])
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)