"""
Unit tests for Machine Learning Models (modules.prediction)
Tests loading, feature expectations, inference execution, and probability bounds
for Diabetes, Kidney Disease, and Heart Disease models.
"""

import unittest
import numpy as np
import pandas as pd

from modules.prediction import (
    diabetes_model,
    kidney_model,
    heart_model,
    predict_diabetes,
    predict_kidney,
    predict_heart,
)


class TestPredictionModels(unittest.TestCase):
    def test_models_loaded(self):
        """Ensure all three pre-trained models are loaded into memory."""
        self.assertIsNotNone(diabetes_model, "Diabetes model failed to load.")
        self.assertIsNotNone(kidney_model, "Kidney model failed to load.")
        self.assertIsNotNone(heart_model, "Heart model failed to load.")

    def test_diabetes_prediction(self):
        """Test diabetes prediction function returns valid label and confidence."""
        result, confidence = predict_diabetes(
            pregnancies=1.0,
            glucose=120.0,
            blood_pressure=70.0,
            skin_thickness=20.0,
            insulin=80.0,
            bmi=25.0,
            diabetes_pedigree=0.5,
            age=30,
        )
        self.assertIn(result, ["Diabetes Risk Detected", "No Diabetes Risk Detected"])
        self.assertTrue(0.0 <= confidence <= 100.0, f"Confidence {confidence} out of range [0, 100]")

    def test_kidney_prediction(self):
        """Test kidney disease prediction function returns valid label and confidence."""
        result, confidence = predict_kidney(
            age=45,
            blood_pressure=80.0,
            specific_gravity=1.020,
            albumin=0.0,
            sugar=0.0,
            red_blood_cells="normal",
            pus_cells="normal",
            pus_cell_clumps="notpresent",
            bacteria="notpresent",
            blood_glucose=100.0,
            blood_urea=30.0,
            creatinine=1.0,
            sodium=140.0,
            potassium=4.5,
            hemoglobin=15.0,
            packed_cell_volume=45.0,
            white_blood_cell_count=8000.0,
            red_blood_cell_count=5.0,
            hypertension="no",
            diabetes_mellitus="no",
            coronary_artery_disease="no",
            appetite="good",
            pedal_edema="no",
            anemia="no",
        )
        self.assertIn(result, ["Kidney Disease Risk Detected", "No Kidney Disease Risk Detected"])
        self.assertTrue(0.0 <= confidence <= 100.0)

    def test_heart_prediction(self):
        """Test heart disease prediction function returns valid label and confidence."""
        result, confidence = predict_heart(
            age=55,
            sex=1,
            chest_pain_type=0,
            resting_blood_pressure=130.0,
            cholesterol=240.0,
            fasting_blood_sugar=0,
            resting_ecg=0,
            max_heart_rate=150.0,
            exercise_angina=0,
            oldpeak=1.2,
            slope=1,
            ca=0,
            thal=2,
        )
        self.assertIn(result, ["Heart Disease Risk Detected", "No Heart Disease Risk Detected"])
        self.assertTrue(0.0 <= confidence <= 100.0)

    def test_feature_order_integrity(self):
        """Ensure model feature names match documented order."""
        expected_diabetes = [
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
        ]
        self.assertEqual(list(diabetes_model.feature_names_in_), expected_diabetes)


if __name__ == "__main__":
    unittest.main()
