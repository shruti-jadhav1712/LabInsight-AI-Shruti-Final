
import joblib
import pandas as pd
from pathlib import Path
from shap_explainer import explain_diabetes

BASE_DIR = Path(__file__).resolve().parent.parent

model = joblib.load(
    BASE_DIR / "models" / "diabetes_model.pkl"
)

sample = pd.DataFrame([{
    "Pregnancies": 6,
    "Glucose": 190,
    "BloodPressure": 95,
    "SkinThickness": 35,
    "Insulin": 250,
    "BMI": 36.5,
    "DiabetesPedigreeFunction": 1.20,
    "Age": 52
}])

result = explain_diabetes(model, sample)

print("\nSHAP Explanation:")
print(result.to_string(index=False))