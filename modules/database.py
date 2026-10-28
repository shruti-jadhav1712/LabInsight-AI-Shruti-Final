"""
LabInsight AI - Database Module
Handles SQLite connection, schema initialization, migrations,
patient management, and laboratory report persistence.
"""

import sqlite3
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "database" / "labinsight.db"


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_connection():
    """Establish and return a connection to the SQLite database."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ============================================================
# INITIALIZE / MIGRATE DATABASE
# ============================================================

def initialize_database():
    """
    Create tables if they do not exist and run migrations safely.
    Preserves all existing records and schema.
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Patients table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            patient_id TEXT PRIMARY KEY,
            patient_name TEXT NOT NULL,
            age INTEGER,
            gender TEXT,
            phone TEXT,
            phone_number TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Lab Reports table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS lab_reports (
            report_id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT,

            -- Common Measurements
            glucose REAL,
            blood_pressure REAL,
            cholesterol REAL,
            creatinine REAL,
            blood_urea REAL,
            hemoglobin REAL,
            bmi REAL,

            -- Legacy columns
            prediction TEXT,
            confidence REAL,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            -- Diabetes Parameters
            pregnancies REAL,
            skin_thickness REAL,
            insulin REAL,
            diabetes_pedigree REAL,

            -- Kidney Parameters
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

            -- Heart Parameters
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

            -- AI Predictions
            diabetes_prediction TEXT,
            diabetes_confidence REAL,
            kidney_prediction TEXT,
            kidney_confidence REAL,
            heart_prediction TEXT,
            heart_confidence REAL,

            FOREIGN KEY(patient_id) REFERENCES patients(patient_id)
        )
    """)

    # Indexes
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_reports_patient
        ON lab_reports(patient_id)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_reports_created
        ON lab_reports(created_at)
    """)

    conn.commit()
    conn.close()

    migrate_database()


def migrate_database():
    """
    Safely inspect existing tables and add any missing columns.
    Ensures zero data loss for existing patients and reports.
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Check patients columns
    cursor.execute("PRAGMA table_info(patients)")
    patient_columns = [row["name"] for row in cursor.fetchall()]

    if "phone_number" not in patient_columns:
        cursor.execute("ALTER TABLE patients ADD COLUMN phone_number TEXT")

    # Check lab_reports columns
    cursor.execute("PRAGMA table_info(lab_reports)")
    report_columns = [row["name"] for row in cursor.fetchall()]

    required_columns = {
        # Diabetes
        "pregnancies": "REAL",
        "skin_thickness": "REAL",
        "insulin": "REAL",
        "diabetes_pedigree": "REAL",

        # Kidney
        "specific_gravity": "REAL",
        "albumin": "REAL",
        "sugar": "REAL",
        "red_blood_cells": "TEXT",
        "pus_cells": "TEXT",
        "pus_cell_clumps": "TEXT",
        "bacteria": "TEXT",
        "blood_glucose": "REAL",
        "sodium": "REAL",
        "potassium": "REAL",
        "packed_cell_volume": "REAL",
        "white_blood_cell_count": "REAL",
        "red_blood_cell_count": "REAL",
        "hypertension": "TEXT",
        "diabetes_mellitus": "TEXT",
        "coronary_artery_disease": "TEXT",
        "appetite": "TEXT",
        "pedal_edema": "TEXT",
        "anemia": "TEXT",

        # Heart
        "sex": "REAL",
        "chest_pain_type": "REAL",
        "resting_blood_pressure": "REAL",
        "fasting_blood_sugar": "REAL",
        "resting_ecg": "REAL",
        "max_heart_rate": "REAL",
        "exercise_angina": "REAL",
        "oldpeak": "REAL",
        "slope": "REAL",
        "ca": "REAL",
        "thal": "REAL",

        # Predictions
        "diabetes_prediction": "TEXT",
        "diabetes_confidence": "REAL",
        "kidney_prediction": "TEXT",
        "kidney_confidence": "REAL",
        "heart_prediction": "TEXT",
        "heart_confidence": "REAL",
    }

    added_columns = 0
    for column_name, column_type in required_columns.items():
        if column_name not in report_columns:
            cursor.execute(
                f"ALTER TABLE lab_reports ADD COLUMN {column_name} {column_type}"
            )
            added_columns += 1

    conn.commit()
    conn.close()

    if added_columns > 0:
        print(f"Database migration completed. Added {added_columns} new columns.")


# ============================================================
# PATIENT MANAGEMENT
# ============================================================

def add_patient(
    patient_id,
    name,
    age=None,
    phone_number=None,
    gender=None,
):
    """
    Insert a new patient or update details if the patient already exists.
    Maintains both phone and phone_number columns for compatibility.
    """
    patient_id = str(patient_id).strip()
    name = str(name).strip()
    phone_clean = str(phone_number).strip() if phone_number is not None else None

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT patient_id FROM patients WHERE patient_id = ?",
        (patient_id,)
    )
    existing_patient = cursor.fetchone()

    if existing_patient:
        cursor.execute("""
            UPDATE patients
            SET
                patient_name = ?,
                age = ?,
                gender = ?,
                phone = ?,
                phone_number = ?
            WHERE patient_id = ?
        """, (
            name,
            age,
            gender,
            phone_clean,
            phone_clean,
            patient_id,
        ))
    else:
        cursor.execute("""
            INSERT INTO patients (
                patient_id,
                patient_name,
                age,
                gender,
                phone,
                phone_number
            )
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            patient_id,
            name,
            age,
            gender,
            phone_clean,
            phone_clean,
        ))

    conn.commit()
    conn.close()


def get_patient(patient_id):
    """
    Retrieve a patient by exact patient_id.
    Returns a tuple (patient_id, patient_name, age, phone, gender) or None.
    """
    if not patient_id:
        return None

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            patient_id,
            patient_name,
            age,
            COALESCE(phone_number, phone) AS phone,
            gender
        FROM patients
        WHERE patient_id = ?
    """, (str(patient_id).strip(),))

    row = cursor.fetchone()
    conn.close()

    if row:
        return (
            row["patient_id"],
            row["patient_name"],
            row["age"],
            row["phone"],
            row["gender"],
        )
    return None


def search_patients_by_name(patient_name):
    """
    Search for patients whose name contains the search term (case-insensitive).
    Returns a list of tuples (patient_id, patient_name, age, phone, gender).
    """
    if not patient_name or not patient_name.strip():
        return []

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            patient_id,
            patient_name,
            age,
            COALESCE(phone_number, phone) AS phone,
            gender
        FROM patients
        WHERE LOWER(patient_name) LIKE LOWER(?)
        ORDER BY patient_name ASC
    """, (f"%{patient_name.strip()}%",))

    rows = cursor.fetchall()
    conn.close()

    return [
        (
            row["patient_id"],
            row["patient_name"],
            row["age"],
            row["phone"],
            row["gender"],
        )
        for row in rows
    ]


def get_all_patients():
    """
    Retrieve all registered patients sorted by patient_id.
    Returns list of dicts with patient details.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            patient_id,
            patient_name,
            age,
            COALESCE(phone_number, phone) AS phone,
            gender,
            created_at
        FROM patients
        ORDER BY patient_id ASC
    """)

    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]


def suggest_next_patient_id():
    """
    Suggest the next available patient ID formatted like 'P008'.
    Finds existing IDs with the 'P' prefix, extracts highest number, and increments.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT patient_id FROM patients")
    rows = cursor.fetchall()
    conn.close()

    max_num = 0
    for row in rows:
        pid = str(row["patient_id"]).strip()
        if pid.upper().startswith("P"):
            digits = pid[1:]
            if digits.isdigit():
                max_num = max(max_num, int(digits))
        elif pid.isdigit():
            max_num = max(max_num, int(pid))

    return f"P{max_num + 1:03d}"


# ============================================================
# LABORATORY REPORTS MANAGEMENT
# ============================================================

def add_lab_report(
    patient_id,
    # Common parameters
    glucose=None,
    blood_pressure=None,
    cholesterol=None,
    creatinine=None,
    blood_urea=None,
    hemoglobin=None,
    bmi=None,
    # Diabetes
    pregnancies=None,
    skin_thickness=None,
    insulin=None,
    diabetes_pedigree=None,
    # Kidney
    specific_gravity=None,
    albumin=None,
    sugar=None,
    red_blood_cells=None,
    pus_cells=None,
    pus_cell_clumps=None,
    bacteria=None,
    blood_glucose=None,
    sodium=None,
    potassium=None,
    packed_cell_volume=None,
    white_blood_cell_count=None,
    red_blood_cell_count=None,
    hypertension=None,
    diabetes_mellitus=None,
    coronary_artery_disease=None,
    appetite=None,
    pedal_edema=None,
    anemia=None,
    # Heart
    sex=None,
    chest_pain_type=None,
    resting_blood_pressure=None,
    fasting_blood_sugar=None,
    resting_ecg=None,
    max_heart_rate=None,
    exercise_angina=None,
    oldpeak=None,
    slope=None,
    ca=None,
    thal=None,
    # Predictions (optional atomic insert)
    diabetes_prediction=None,
    diabetes_confidence=None,
    kidney_prediction=None,
    kidney_confidence=None,
    heart_prediction=None,
    heart_confidence=None,
):
    """
    Insert a complete laboratory report into the database.
    Can store predictions atomically during insert or later via update functions.
    Returns the newly created report_id (integer).
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO lab_reports (
            patient_id,
            glucose,
            blood_pressure,
            cholesterol,
            creatinine,
            blood_urea,
            hemoglobin,
            bmi,
            pregnancies,
            skin_thickness,
            insulin,
            diabetes_pedigree,
            specific_gravity,
            albumin,
            sugar,
            red_blood_cells,
            pus_cells,
            pus_cell_clumps,
            bacteria,
            blood_glucose,
            sodium,
            potassium,
            packed_cell_volume,
            white_blood_cell_count,
            red_blood_cell_count,
            hypertension,
            diabetes_mellitus,
            coronary_artery_disease,
            appetite,
            pedal_edema,
            anemia,
            sex,
            chest_pain_type,
            resting_blood_pressure,
            fasting_blood_sugar,
            resting_ecg,
            max_heart_rate,
            exercise_angina,
            oldpeak,
            slope,
            ca,
            thal,
            diabetes_prediction,
            diabetes_confidence,
            kidney_prediction,
            kidney_confidence,
            heart_prediction,
            heart_confidence
        )
        VALUES (
            ?,
            ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?
        )
    """, (
        str(patient_id).strip(),
        glucose,
        blood_pressure,
        cholesterol,
        creatinine,
        blood_urea,
        hemoglobin,
        bmi,
        pregnancies,
        skin_thickness,
        insulin,
        diabetes_pedigree,
        specific_gravity,
        albumin,
        sugar,
        red_blood_cells,
        pus_cells,
        pus_cell_clumps,
        bacteria,
        blood_glucose,
        sodium,
        potassium,
        packed_cell_volume,
        white_blood_cell_count,
        red_blood_cell_count,
        hypertension,
        diabetes_mellitus,
        coronary_artery_disease,
        appetite,
        pedal_edema,
        anemia,
        sex,
        chest_pain_type,
        resting_blood_pressure,
        fasting_blood_sugar,
        resting_ecg,
        max_heart_rate,
        exercise_angina,
        oldpeak,
        slope,
        ca,
        thal,
        diabetes_prediction,
        diabetes_confidence,
        kidney_prediction,
        kidney_confidence,
        heart_prediction,
        heart_confidence,
    ))

    conn.commit()
    report_id = cursor.lastrowid
    conn.close()

    return report_id


def get_lab_reports(patient_id):
    """
    Retrieve all lab reports for a patient, ordered by report_id DESC.
    Returns:
        reports: list of row tuples
        column_names: list of column names
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT *
        FROM lab_reports
        WHERE patient_id = ?
        ORDER BY report_id DESC
    """, (str(patient_id).strip(),))

    rows = cursor.fetchall()
    column_names = [col[0] for col in cursor.description] if cursor.description else []

    # Convert Row objects to tuples for backward compatibility
    reports = [tuple(row) for row in rows]

    conn.close()
    return reports, column_names


def get_report_by_id(report_id):
    """
    Retrieve a single lab report by report_id.
    Returns a dictionary of column_name -> value, or None if not found.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT *
        FROM lab_reports
        WHERE report_id = ?
    """, (report_id,))

    row = cursor.fetchone()
    conn.close()

    if row:
        return dict(row)
    return None


# ============================================================
# UPDATE PREDICTIONS
# ============================================================

def update_diabetes_prediction(
    patient_id,
    prediction,
    confidence,
    report_id=None,
):
    """
    Update diabetes prediction for a specific report_id or the patient's latest report.
    """
    conn = get_connection()
    cursor = conn.cursor()

    if report_id is not None:
        cursor.execute("""
            UPDATE lab_reports
            SET
                diabetes_prediction = ?,
                diabetes_confidence = ?
            WHERE report_id = ?
        """, (prediction, confidence, report_id))
    else:
        cursor.execute("""
            UPDATE lab_reports
            SET
                diabetes_prediction = ?,
                diabetes_confidence = ?
            WHERE report_id = (
                SELECT report_id
                FROM lab_reports
                WHERE patient_id = ?
                ORDER BY report_id DESC
                LIMIT 1
            )
        """, (prediction, confidence, str(patient_id).strip()))

    conn.commit()
    conn.close()


def update_kidney_prediction(
    patient_id,
    prediction,
    confidence,
    report_id=None,
):
    """
    Update kidney prediction for a specific report_id or the patient's latest report.
    """
    conn = get_connection()
    cursor = conn.cursor()

    if report_id is not None:
        cursor.execute("""
            UPDATE lab_reports
            SET
                kidney_prediction = ?,
                kidney_confidence = ?
            WHERE report_id = ?
        """, (prediction, confidence, report_id))
    else:
        cursor.execute("""
            UPDATE lab_reports
            SET
                kidney_prediction = ?,
                kidney_confidence = ?
            WHERE report_id = (
                SELECT report_id
                FROM lab_reports
                WHERE patient_id = ?
                ORDER BY report_id DESC
                LIMIT 1
            )
        """, (prediction, confidence, str(patient_id).strip()))

    conn.commit()
    conn.close()


def update_heart_prediction(
    patient_id,
    prediction,
    confidence,
    report_id=None,
):
    """
    Update heart prediction for a specific report_id or the patient's latest report.
    """
    conn = get_connection()
    cursor = conn.cursor()

    if report_id is not None:
        cursor.execute("""
            UPDATE lab_reports
            SET
                heart_prediction = ?,
                heart_confidence = ?
            WHERE report_id = ?
        """, (prediction, confidence, report_id))
    else:
        cursor.execute("""
            UPDATE lab_reports
            SET
                heart_prediction = ?,
                heart_confidence = ?
            WHERE report_id = (
                SELECT report_id
                FROM lab_reports
                WHERE patient_id = ?
                ORDER BY report_id DESC
                LIMIT 1
            )
        """, (prediction, confidence, str(patient_id).strip()))

    conn.commit()
    conn.close()


# Ensure database and tables are ready upon module import
initialize_database()