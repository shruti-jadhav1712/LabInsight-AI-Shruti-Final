-- LabInsight AI Database Schema
-- Defines tables for patients and comprehensive laboratory reports

CREATE TABLE IF NOT EXISTS patients (
    patient_id TEXT PRIMARY KEY,
    patient_name TEXT NOT NULL,
    age INTEGER,
    gender TEXT,
    phone TEXT,
    phone_number TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS lab_reports (
    report_id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id TEXT NOT NULL,

    -- Common Laboratory Measurements
    glucose REAL,
    blood_pressure REAL,
    cholesterol REAL,
    creatinine REAL,
    blood_urea REAL,
    hemoglobin REAL,
    bmi REAL,

    -- Diabetes Specific Parameters
    pregnancies REAL,
    skin_thickness REAL,
    insulin REAL,
    diabetes_pedigree REAL,

    -- Kidney Specific Parameters
    specific_gravity REAL,
    albumin REAL,
    sugar REAL,
    red_blood_cells TEXT,
    pus_cells TEXT,
    pus_cell_clumps TEXT,
    bacteria TEXT,
    blood_glucose REAL,
    sodium REAL,
    potassium REAL,
    packed_cell_volume REAL,
    white_blood_cell_count REAL,
    red_blood_cell_count REAL,
    hypertension TEXT,
    diabetes_mellitus TEXT,
    coronary_artery_disease TEXT,
    appetite TEXT,
    pedal_edema TEXT,
    anemia TEXT,

    -- Heart Disease Specific Parameters
    sex REAL,
    chest_pain_type REAL,
    resting_blood_pressure REAL,
    fasting_blood_sugar REAL,
    resting_ecg REAL,
    max_heart_rate REAL,
    exercise_angina REAL,
    oldpeak REAL,
    slope REAL,
    ca REAL,
    thal REAL,

    -- Legacy prediction columns
    prediction TEXT,
    confidence REAL,

    -- AI Predictions and Model Confidences
    diabetes_prediction TEXT,
    diabetes_confidence REAL,
    kidney_prediction TEXT,
    kidney_confidence REAL,
    heart_prediction TEXT,
    heart_confidence REAL,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY(patient_id) REFERENCES patients(patient_id)
);

CREATE INDEX IF NOT EXISTS idx_reports_patient ON lab_reports(patient_id);
CREATE INDEX IF NOT EXISTS idx_reports_created ON lab_reports(created_at);