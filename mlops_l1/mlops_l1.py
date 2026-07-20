# Generated from: mlops_l1.ipynb
# Converted at: 2026-07-13T05:09:16.489Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import json


df = pd.read_csv("mlops_l1/Telecom_Tower_Failure_Dataset_10000-1.csv")
df.head()

x = df.drop(['Tower_ID', 'Failure_Within_48Hrs'], axis=1)
y = df['Failure_Within_48Hrs']

x.columns

y

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

print("training model....")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy}")

joblib.dump(model, 'telecom_tower_failure_model.pkl')
metrics = {
    "accuracy": accuracy
}
with open('metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)

print("Model and metrics saved successfully.")