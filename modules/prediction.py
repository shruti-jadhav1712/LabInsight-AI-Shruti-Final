import joblib
import pandas as pd
from pathlib import Path


# ==========================================================
# MODEL PATHS
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DIABETES_MODEL_PATH = BASE_DIR / "models" / "diabetes_model.pkl"
KIDNEY_MODEL_PATH = BASE_DIR / "models" / "kidney_model.pkl"
HEART_MODEL_PATH = BASE_DIR / "models" / "heart_model.pkl"


# ==========================================================
# LOAD MODELS
# ==========================================================

diabetes_model = joblib.load(DIABETES_MODEL_PATH)

kidney_model = joblib.load(KIDNEY_MODEL_PATH)

heart_model = joblib.load(HEART_MODEL_PATH)


# ==========================================================
# DIABETES PREDICTION
# ==========================================================

def predict_diabetes(
    pregnancies,
    glucose,
    blood_pressure,
    skin_thickness,
    insulin,
    bmi,
    diabetes_pedigree,
    age
):

    data = pd.DataFrame([{
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": diabetes_pedigree,
        "Age": age
    }])

    prediction = diabetes_model.predict(data)[0]

    probability = diabetes_model.predict_proba(data)[0]

    confidence = max(probability) * 100

    if prediction == 1:
        result = "Diabetes Risk Detected"
    else:
        result = "No Diabetes Risk Detected"

    return result, confidence


# ==========================================================
# KIDNEY DISEASE PREDICTION
# ==========================================================

def predict_kidney(
    age,
    blood_pressure,
    specific_gravity,
    albumin,
    sugar,
    red_blood_cells,
    pus_cells,
    pus_cell_clumps,
    bacteria,
    blood_glucose,
    blood_urea,
    creatinine,
    sodium,
    potassium,
    hemoglobin,
    packed_cell_volume,
    white_blood_cell_count,
    red_blood_cell_count,
    hypertension,
    diabetes_mellitus,
    coronary_artery_disease,
    appetite,
    pedal_edema,
    anemia
):

    data = pd.DataFrame([{
        "age": age,
        "bp": blood_pressure,
        "sg": specific_gravity,
        "al": albumin,
        "su": sugar,
        "rbc": red_blood_cells,
        "pc": pus_cells,
        "pcc": pus_cell_clumps,
        "ba": bacteria,
        "bgr": blood_glucose,
        "bu": blood_urea,
        "sc": creatinine,
        "sod": sodium,
        "pot": potassium,
        "hemo": hemoglobin,
        "pcv": packed_cell_volume,
        "wc": white_blood_cell_count,
        "rc": red_blood_cell_count,
        "htn": hypertension,
        "dm": diabetes_mellitus,
        "cad": coronary_artery_disease,
        "appet": appetite,
        "pe": pedal_edema,
        "ane": anemia
    }])

    prediction = kidney_model.predict(data)[0]

    probability = kidney_model.predict_proba(data)[0]

    confidence = max(probability) * 100

    if prediction == 1:
        result = "Kidney Disease Risk Detected"
    else:
        result = "No Kidney Disease Risk Detected"

    return result, confidence


# ==========================================================
# HEART DISEASE PREDICTION
# ==========================================================

def predict_heart(
    age,
    sex,
    chest_pain_type,
    resting_blood_pressure,
    cholesterol,
    fasting_blood_sugar,
    resting_ecg,
    max_heart_rate,
    exercise_angina,
    oldpeak,
    slope,
    ca,
    thal
):

    data = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "cp": chest_pain_type,
        "trestbps": resting_blood_pressure,
        "chol": cholesterol,
        "fbs": fasting_blood_sugar,
        "restecg": resting_ecg,
        "thalach": max_heart_rate,
        "exang": exercise_angina,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }])

    prediction = heart_model.predict(data)[0]

    probability = heart_model.predict_proba(data)[0]

    confidence = max(probability) * 100

    if prediction == 1:
        result = "Heart Disease Risk Detected"
    else:
        result = "No Heart Disease Risk Detected"

    return result, confidence