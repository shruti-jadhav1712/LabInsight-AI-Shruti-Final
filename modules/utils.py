"""
LabInsight AI - Utilities Module
Provides clinical reference ranges, parameter mappings, input validation,
and display formatting helpers for laboratory data.
"""

from typing import Dict, Any, Optional, Tuple


# ============================================================
# CLINICAL REFERENCE RANGES (Standard Diagnostic Guidelines)
# ============================================================

REFERENCE_RANGES: Dict[str, Dict[str, Any]] = {
    # Common / Shared
    "glucose": {"min": 70.0, "max": 99.0, "unit": "mg/dL", "name": "Fasting Blood Glucose"},
    "blood_pressure": {"min": 90.0, "max": 120.0, "unit": "mmHg", "name": "Systolic Blood Pressure"},
    "cholesterol": {"min": 125.0, "max": 200.0, "unit": "mg/dL", "name": "Total Serum Cholesterol"},
    "creatinine": {"min": 0.6, "max": 1.2, "unit": "mg/dL", "name": "Serum Creatinine"},
    "blood_urea": {"min": 7.0, "max": 20.0, "unit": "mg/dL", "name": "Blood Urea Nitrogen (BUN)"},
    "hemoglobin": {"min": 12.0, "max": 17.5, "unit": "g/dL", "name": "Hemoglobin"},
    "bmi": {"min": 18.5, "max": 24.9, "unit": "kg/m²", "name": "Body Mass Index (BMI)"},

    # Diabetes Specific
    "pregnancies": {"min": 0.0, "max": 15.0, "unit": "count", "name": "Number of Pregnancies"},
    "skin_thickness": {"min": 10.0, "max": 50.0, "unit": "mm", "name": "Triceps Skin Fold Thickness"},
    "insulin": {"min": 16.0, "max": 166.0, "unit": "µIU/mL", "name": "2-Hour Serum Insulin"},
    "diabetes_pedigree": {"min": 0.08, "max": 2.42, "unit": "score", "name": "Diabetes Pedigree Function"},

    # Kidney Specific
    "specific_gravity": {"min": 1.010, "max": 1.025, "unit": "", "name": "Specific Gravity"},
    "albumin": {"min": 0.0, "max": 0.0, "unit": "grade", "name": "Urine Albumin (0-5)"},
    "sugar": {"min": 0.0, "max": 0.0, "unit": "grade", "name": "Urine Sugar (0-5)"},
    "blood_glucose": {"min": 70.0, "max": 125.0, "unit": "mg/dL", "name": "Blood Glucose Random"},
    "sodium": {"min": 135.0, "max": 145.0, "unit": "mEq/L", "name": "Serum Sodium"},
    "potassium": {"min": 3.5, "max": 5.0, "unit": "mEq/L", "name": "Serum Potassium"},
    "packed_cell_volume": {"min": 36.0, "max": 50.0, "unit": "%", "name": "Packed Cell Volume (PCV)"},
    "white_blood_cell_count": {"min": 4000.0, "max": 11000.0, "unit": "/µL", "name": "White Blood Cell Count (WBC)"},
    "red_blood_cell_count": {"min": 4.2, "max": 5.9, "unit": "M/µL", "name": "Red Blood Cell Count (RBC)"},

    # Heart Specific
    "resting_blood_pressure": {"min": 90.0, "max": 120.0, "unit": "mmHg", "name": "Resting Blood Pressure"},
    "max_heart_rate": {"min": 60.0, "max": 180.0, "unit": "bpm", "name": "Maximum Heart Rate Achieved"},
    "oldpeak": {"min": 0.0, "max": 1.0, "unit": "mm", "name": "ST Depression (Oldpeak)"},
}

# Parameter friendly name mapping for charts and reports
FEATURE_DISPLAY_NAMES: Dict[str, str] = {
    # Diabetes
    "Pregnancies": "Pregnancies",
    "Glucose": "Blood Glucose (mg/dL)",
    "BloodPressure": "Blood Pressure (mmHg)",
    "SkinThickness": "Skin Thickness (mm)",
    "Insulin": "Serum Insulin (µIU/mL)",
    "BMI": "Body Mass Index (kg/m²)",
    "DiabetesPedigreeFunction": "Diabetes Pedigree Score",
    "Age": "Patient Age (years)",

    # Kidney
    "bp": "Blood Pressure (mmHg)",
    "sg": "Specific Gravity",
    "al": "Albumin",
    "su": "Sugar",
    "rbc": "Red Blood Cells",
    "pc": "Pus Cells",
    "pcc": "Pus Cell Clumps",
    "ba": "Bacteria",
    "bgr": "Blood Glucose Random",
    "bu": "Blood Urea (mg/dL)",
    "sc": "Serum Creatinine (mg/dL)",
    "sod": "Sodium (mEq/L)",
    "pot": "Potassium (mEq/L)",
    "hemo": "Hemoglobin (g/dL)",
    "pcv": "Packed Cell Volume (%)",
    "wc": "White Blood Cell Count (/µL)",
    "rc": "Red Blood Cell Count (M/µL)",
    "htn": "Hypertension",
    "dm": "Diabetes Mellitus",
    "cad": "Coronary Artery Disease",
    "appet": "Appetite",
    "pe": "Pedal Edema",
    "ane": "Anemia",

    # Heart
    "sex": "Biological Sex",
    "cp": "Chest Pain Type",
    "trestbps": "Resting Blood Pressure (mmHg)",
    "chol": "Serum Cholesterol (mg/dL)",
    "fbs": "Fasting Blood Sugar > 120",
    "restecg": "Resting ECG Results",
    "thalach": "Max Heart Rate (bpm)",
    "exang": "Exercise Induced Angina",
    "slope": "ST Segment Slope",
    "ca": "Major Vessels Colored",
    "thal": "Thalassemia Condition",
}


def get_parameter_status(key: str, value: Any) -> Tuple[str, str]:
    """
    Check if a numeric lab parameter falls within standard reference range.
    Returns a tuple of (status_text, badge_style):
        - ("Normal", "normal")
        - ("High", "high")
        - ("Low", "low")
        - ("-", "unknown")
    """
    if value is None or key not in REFERENCE_RANGES:
        return "-", "unknown"

    try:
        val = float(value)
        ref = REFERENCE_RANGES[key]
        min_v = ref.get("min")
        max_v = ref.get("max")

        if min_v is not None and val < min_v:
            return "Low", "low"
        if max_v is not None and val > max_v:
            return "High", "high"
        return "Normal", "normal"
    except (ValueError, TypeError):
        return "-", "unknown"


def format_parameter_display(key: str, value: Any) -> str:
    """Format a parameter value with its clinical unit if known."""
    if value is None or str(value).strip() in ("", "None", "nan"):
        return "N/A"

    ref = REFERENCE_RANGES.get(key)
    unit = f" {ref['unit']}" if ref and ref.get("unit") else ""

    try:
        f_val = float(value)
        if f_val.is_integer():
            return f"{int(f_val)}{unit}"
        return f"{f_val:.2f}{unit}"
    except (ValueError, TypeError):
        return f"{value}{unit}"


def validate_patient_id(patient_id: str) -> Tuple[bool, str]:
    """Validate patient ID format."""
    if not patient_id or not patient_id.strip():
        return False, "Patient ID cannot be empty."
    if len(patient_id.strip()) > 30:
        return False, "Patient ID must not exceed 30 characters."
    return True, ""


def validate_patient_name(name: str) -> Tuple[bool, str]:
    """Validate patient name."""
    if not name or not name.strip():
        return False, "Patient Name cannot be empty."
    if len(name.strip()) < 2:
        return False, "Patient Name must be at least 2 characters long."
    return True, ""
