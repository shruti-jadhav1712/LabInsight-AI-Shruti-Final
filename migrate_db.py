import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "database" / "labinsight.db"

# New columns to add
new_columns = {
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
    "sex": "INTEGER",
    "chest_pain_type": "INTEGER",
    "resting_blood_pressure": "REAL",
    "fasting_blood_sugar": "INTEGER",
    "resting_ecg": "INTEGER",
    "max_heart_rate": "REAL",
    "exercise_angina": "INTEGER",
    "oldpeak": "REAL",
    "slope": "INTEGER",
    "ca": "INTEGER",
    "thal": "INTEGER",

    # Predictions
    "diabetes_prediction": "TEXT",
    "diabetes_confidence": "REAL",
    "kidney_prediction": "TEXT",
    "kidney_confidence": "REAL",
    "heart_prediction": "TEXT",
    "heart_confidence": "REAL",
}

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Get existing columns
cursor.execute("PRAGMA table_info(lab_reports)")
existing_columns = {row[1] for row in cursor.fetchall()}

added = 0

for column, data_type in new_columns.items():
    if column not in existing_columns:
        cursor.execute(
            f"ALTER TABLE lab_reports ADD COLUMN {column} {data_type}"
        )
        print(f"Added: {column}")
        added += 1
    else:
        print(f"Already exists: {column}")

conn.commit()
conn.close()

print()
print(f"Migration completed. {added} new columns added.")