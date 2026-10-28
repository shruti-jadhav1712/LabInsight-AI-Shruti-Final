"""
Integration Tests for LabInsight AI
Verifies the complete end-to-end lifecycle:
1. Patient Registration (Technician)
2. Laboratory Test Entry & Multi-Disease ML Predictions (Technician)
3. Report Persistence to SQLite (Technician)
4. Patient Search & Disambiguation (Doctor)
5. Report Retrieval & Verification (Doctor)
6. Explainable AI SHAP Attribution Analysis (Doctor)
7. Final Clinical PDF Report Generation & Validation
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import pandas as pd

from modules import database
from modules.prediction import (
    predict_diabetes,
    predict_kidney,
    predict_heart,
    diabetes_model,
)
from modules.shap_explainer import explain_diabetes
from modules.report_generator import generate_pdf_report


class TestEndToEndWorkflow(unittest.TestCase):
    def setUp(self):
        """Isolate database in temporary directory."""
        self.test_dir = tempfile.mkdtemp()
        self.test_db_path = Path(self.test_dir) / "integration_labinsight.db"
        self.original_db = database.DB_PATH
        database.DB_PATH = self.test_db_path
        database.initialize_database()

    def tearDown(self):
        database.DB_PATH = self.original_db
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_complete_technician_to_doctor_workflow(self):
        # Step 1: Technician registers patient
        patient_id = "INT_P001"
        patient_name = "Eleanor Vance"
        age = 52
        gender = "Female"
        phone = "555-987-6543"

        database.add_patient(
            patient_id=patient_id,
            name=patient_name,
            age=age,
            phone_number=phone,
            gender=gender,
        )

        # Step 2: Technician inputs lab values and computes predictions
        # Diabetes parameters
        d_res, d_conf = predict_diabetes(
            pregnancies=3.0,
            glucose=175.0,
            blood_pressure=85.0,
            skin_thickness=30.0,
            insulin=200.0,
            bmi=32.0,
            diabetes_pedigree=0.85,
            age=age,
        )

        # Kidney parameters
        k_res, k_conf = predict_kidney(
            age=age,
            blood_pressure=85.0,
            specific_gravity=1.015,
            albumin=1.0,
            sugar=0.0,
            red_blood_cells="normal",
            pus_cells="normal",
            pus_cell_clumps="notpresent",
            bacteria="notpresent",
            blood_glucose=175.0,
            blood_urea=42.0,
            creatinine=1.4,
            sodium=138.0,
            potassium=4.2,
            hemoglobin=13.0,
            packed_cell_volume=39.0,
            white_blood_cell_count=7500.0,
            red_blood_cell_count=4.6,
            hypertension="yes",
            diabetes_mellitus="yes",
            coronary_artery_disease="no",
            appetite="good",
            pedal_edema="no",
            anemia="no",
        )

        # Heart parameters
        h_res, h_conf = predict_heart(
            age=age,
            sex=0,  # Female
            chest_pain_type=2,
            resting_blood_pressure=140.0,
            cholesterol=245.0,
            fasting_blood_sugar=1,
            resting_ecg=1,
            max_heart_rate=145.0,
            exercise_angina=0,
            oldpeak=1.4,
            slope=1,
            ca=1,
            thal=2,
        )

        # Step 3: Technician commits complete report to database
        report_id = database.add_lab_report(
            patient_id=patient_id,
            glucose=175.0,
            blood_pressure=85.0,
            cholesterol=245.0,
            creatinine=1.4,
            blood_urea=42.0,
            hemoglobin=13.0,
            bmi=32.0,
            pregnancies=3.0,
            skin_thickness=30.0,
            insulin=200.0,
            diabetes_pedigree=0.85,
            specific_gravity=1.015,
            albumin=1.0,
            sugar=0.0,
            red_blood_cells="normal",
            pus_cells="normal",
            pus_cell_clumps="notpresent",
            bacteria="notpresent",
            blood_glucose=175.0,
            sodium=138.0,
            potassium=4.2,
            packed_cell_volume=39.0,
            white_blood_cell_count=7500.0,
            red_blood_cell_count=4.6,
            hypertension="yes",
            diabetes_mellitus="yes",
            coronary_artery_disease="no",
            appetite="good",
            pedal_edema="no",
            anemia="no",
            sex=0,
            chest_pain_type=2,
            resting_blood_pressure=140.0,
            fasting_blood_sugar=1,
            resting_ecg=1,
            max_heart_rate=145.0,
            exercise_angina=0,
            oldpeak=1.4,
            slope=1,
            ca=1,
            thal=2,
            diabetes_prediction=d_res,
            diabetes_confidence=d_conf,
            kidney_prediction=k_res,
            kidney_confidence=k_conf,
            heart_prediction=h_res,
            heart_confidence=h_conf,
        )
        self.assertIsInstance(report_id, int)
        self.assertGreater(report_id, 0)

        # Step 4: Doctor searches for patient by ID
        doc_patient = database.get_patient(patient_id)
        self.assertIsNotNone(doc_patient)
        self.assertEqual(doc_patient[1], patient_name)

        # Also verify search by partial name
        name_results = database.search_patients_by_name("eleanor")
        self.assertEqual(len(name_results), 1)
        self.assertEqual(name_results[0][0], patient_id)

        # Step 5: Doctor retrieves reports
        reports, cols = database.get_lab_reports(patient_id)
        self.assertEqual(len(reports), 1)
        retrieved_report = dict(zip(cols, reports[0]))
        self.assertEqual(retrieved_report["report_id"], report_id)
        self.assertEqual(retrieved_report["glucose"], 175.0)
        self.assertEqual(retrieved_report["diabetes_prediction"], d_res)

        # Step 6: Doctor generates SHAP explanation on the retrieved diabetes features
        shap_input = pd.DataFrame([{
            "Pregnancies": retrieved_report["pregnancies"],
            "Glucose": retrieved_report["glucose"],
            "BloodPressure": retrieved_report["blood_pressure"],
            "SkinThickness": retrieved_report["skin_thickness"],
            "Insulin": retrieved_report["insulin"],
            "BMI": retrieved_report["bmi"],
            "DiabetesPedigreeFunction": retrieved_report["diabetes_pedigree"],
            "Age": doc_patient[2],  # Age
        }])

        shap_df = explain_diabetes(diabetes_model, shap_input)
        self.assertEqual(len(shap_df), 8)
        self.assertIn("Glucose", shap_df["Parameter"].values)

        # High glucose (175 mg/dL) should have positive SHAP impact (increasing diabetes risk)
        glucose_row = shap_df[shap_df["Parameter"] == "Glucose"].iloc[0]
        self.assertGreater(glucose_row["SHAP Impact"], 0.0)

        # Step 7: Doctor generates downloadable PDF report
        pdf_bytes = generate_pdf_report(
            patient=doc_patient,
            report_data=retrieved_report,
            shap_df=shap_df,
        )
        self.assertIsInstance(pdf_bytes, bytes)
        self.assertTrue(pdf_bytes.startswith(b"%PDF"))
        self.assertGreater(len(pdf_bytes), 2500)


if __name__ == "__main__":
    unittest.main()
