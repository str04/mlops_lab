# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

import joblib
import pandas as pd

# Load trained model
model = joblib.load("telecom_tower_model.pkl")

# New tower data
new_data = pd.DataFrame({
    "Temperature_C": [55],
    "Battery_Voltage": [11.2],
    "Power_Consumption_W": [480],
    "Signal_Strength_Percent": [72],
    "Fan_Speed_RPM": [2800],
    "Humidity_Percent": [68],
    "Traffic_Load": [85],
    "Tower_Age_Years": [7]
})

# Predict
prediction = model.predict(new_data)

if prediction[0] == 1:
    print("Hardware Failure Predicted")
else:
    print("Tower is Healthy")