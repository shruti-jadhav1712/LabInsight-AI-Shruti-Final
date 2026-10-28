"""
Unit tests for Database Operations (modules.database)
Tests patient creation, retrieval, search, report insertion, ordering, and migration.
"""

import unittest
import sqlite3
import tempfile
import os
from pathlib import Path
import shutil

from modules import database


class TestDatabaseOperations(unittest.TestCase):
    def setUp(self):
        """Create a temporary database for test isolation."""
        self.test_dir = tempfile.mkdtemp()
        self.test_db_path = Path(self.test_dir) / "test_labinsight.db"
        self.original_db_path = database.DB_PATH
        database.DB_PATH = self.test_db_path
        database.initialize_database()

    def tearDown(self):
        """Restore original database path and clean up temporary directory."""
        database.DB_PATH = self.original_db_path
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_add_and_get_patient(self):
        """Test adding a new patient and retrieving by ID."""
        database.add_patient(
            patient_id="TEST_P001",
            name="Alice Smith",
            age=32,
            phone_number="1234567890",
            gender="Female",
        )
        patient = database.get_patient("TEST_P001")
        self.assertIsNotNone(patient)
        self.assertEqual(patient[0], "TEST_P001")
        self.assertEqual(patient[1], "Alice Smith")
        self.assertEqual(patient[2], 32)
        self.assertEqual(patient[3], "1234567890")
        self.assertEqual(patient[4], "Female")

    def test_update_existing_patient(self):
        """Test that re-adding an existing patient updates their information."""
        database.add_patient("TEST_P002", "Bob Jones", age=40)
        database.add_patient("TEST_P002", "Bob Jones Jr.", age=41, phone_number="9998887777")
        patient = database.get_patient("TEST_P002")
        self.assertEqual(patient[1], "Bob Jones Jr.")
        self.assertEqual(patient[2], 41)
        self.assertEqual(patient[3], "9998887777")

    def test_search_patients_by_name(self):
        """Test case-insensitive partial name search."""
        database.add_patient("TEST_P010", "Michael Brown", age=50)
        database.add_patient("TEST_P011", "Michelle White", age=45)
        database.add_patient("TEST_P012", "David Green", age=28)

        results = database.search_patients_by_name("mich")
        self.assertEqual(len(results), 2)
        names = [r[1] for r in results]
        self.assertIn("Michael Brown", names)
        self.assertIn("Michelle White", names)

    def test_add_and_retrieve_lab_reports(self):
        """Test creating multiple lab reports for one patient and chronological ordering."""
        database.add_patient("TEST_P020", "Charlie Chaplin", age=60)

        rep1 = database.add_lab_report(
            patient_id="TEST_P020",
            glucose=110.0,
            blood_pressure=120.0,
            diabetes_prediction="No Diabetes Risk Detected",
            diabetes_confidence=85.0,
        )
        rep2 = database.add_lab_report(
            patient_id="TEST_P020",
            glucose=180.0,
            blood_pressure=140.0,
            diabetes_prediction="Diabetes Risk Detected",
            diabetes_confidence=92.5,
        )

        reports, columns = database.get_lab_reports("TEST_P020")
        self.assertEqual(len(reports), 2)
        # Check ordering: report 2 should be first because ORDER BY report_id DESC
        rep_id_idx = columns.index("report_id")
        self.assertEqual(reports[0][rep_id_idx], rep2)
        self.assertEqual(reports[1][rep_id_idx], rep1)

    def test_get_report_by_id(self):
        """Test retrieving a single report by its unique report_id."""
        database.add_patient("TEST_P030", "Diana Prince", age=30)
        rep_id = database.add_lab_report(
            patient_id="TEST_P030",
            cholesterol=210.0,
            heart_prediction="Heart Disease Risk Detected",
            heart_confidence=78.0,
        )

        report = database.get_report_by_id(rep_id)
        self.assertIsNotNone(report)
        self.assertEqual(report["patient_id"], "TEST_P030")
        self.assertEqual(report["cholesterol"], 210.0)
        self.assertEqual(report["heart_prediction"], "Heart Disease Risk Detected")

    def test_update_predictions(self):
        """Test updating predictions on existing lab reports."""
        database.add_patient("TEST_P040", "Edward Norton", age=48)
        rep_id = database.add_lab_report("TEST_P040", glucose=100.0)

        database.update_diabetes_prediction(
            patient_id="TEST_P040",
            prediction="No Diabetes Risk Detected",
            confidence=90.0,
            report_id=rep_id,
        )
        report = database.get_report_by_id(rep_id)
        self.assertEqual(report["diabetes_prediction"], "No Diabetes Risk Detected")
        self.assertAlmostEqual(report["diabetes_confidence"], 90.0)

    def test_suggest_next_patient_id(self):
        """Test patient ID suggestion logic."""
        database.add_patient("P001", "P1")
        database.add_patient("P005", "P5")
        next_id = database.suggest_next_patient_id()
        self.assertEqual(next_id, "P006")


if __name__ == "__main__":
    unittest.main()
