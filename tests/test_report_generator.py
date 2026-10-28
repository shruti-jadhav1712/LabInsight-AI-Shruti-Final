"""
Unit tests for PDF Report Generator (modules.report_generator)
Tests ReportLab PDF document compilation, byte output, file persistence,
and handling of missing predictions/SHAP data.
"""

import unittest
import tempfile
import os
from pathlib import Path
import pandas as pd

from modules.report_generator import generate_pdf_report


class TestReportGenerator(unittest.TestCase):
    def setUp(self):
        self.patient = {
            "patient_id": "P_TEST",
            "name": "Jane Doe",
            "age": 42,
            "gender": "Female",
            "phone": "555-1234",
        }
        self.report_data = {
            "report_id": 99,
            "glucose": 135.0,
            "blood_pressure": 125.0,
            "cholesterol": 210.0,
            "creatinine": 0.9,
            "blood_urea": 18.0,
            "hemoglobin": 13.8,
            "bmi": 27.4,
            "diabetes_prediction": "Diabetes Risk Detected",
            "diabetes_confidence": 82.5,
            "kidney_prediction": "No Kidney Disease Risk Detected",
            "kidney_confidence": 94.0,
            "heart_prediction": "No Heart Disease Risk Detected",
            "heart_confidence": 71.0,
            "created_at": "2026-10-10 12:00:00",
        }

    def test_generate_pdf_bytes(self):
        """Test generating PDF returns valid non-empty bytes with PDF magic header."""
        pdf_bytes = generate_pdf_report(self.patient, self.report_data)
        self.assertIsInstance(pdf_bytes, bytes)
        self.assertGreater(len(pdf_bytes), 1000)
        # PDF documents start with %PDF- header
        self.assertTrue(pdf_bytes.startswith(b"%PDF"))

    def test_generate_pdf_with_shap(self):
        """Test PDF generation including SHAP attribution table."""
        shap_df = pd.DataFrame([
            {"Parameter": "Glucose", "SHAP Impact": 1.85, "Direction": "Increases model output", "Absolute Impact": 1.85},
            {"Parameter": "BMI", "SHAP Impact": 0.42, "Direction": "Increases model output", "Absolute Impact": 0.42},
            {"Parameter": "BloodPressure", "SHAP Impact": -0.15, "Direction": "Decreases model output", "Absolute Impact": 0.15},
        ])
        pdf_bytes = generate_pdf_report(self.patient, self.report_data, shap_df=shap_df)
        self.assertTrue(pdf_bytes.startswith(b"%PDF"))
        self.assertGreater(len(pdf_bytes), 2000)

    def test_generate_pdf_to_disk(self):
        """Test generating PDF with direct output_path writes file to disk."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            pdf_bytes = generate_pdf_report(self.patient, self.report_data, output_path=tmp_path)
            self.assertTrue(os.path.exists(tmp_path))
            self.assertEqual(os.path.getsize(tmp_path), len(pdf_bytes))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_generate_pdf_missing_optional_fields(self):
        """Test PDF generation handles missing predictions or sparse lab values gracefully."""
        sparse_report = {"report_id": 100}
        sparse_patient = {"patient_id": "P_SPARSE"}
        pdf_bytes = generate_pdf_report(sparse_patient, sparse_report)
        self.assertTrue(pdf_bytes.startswith(b"%PDF"))


if __name__ == "__main__":
    unittest.main()
