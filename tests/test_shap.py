"""
Unit tests for Explainable AI (SHAP) Module (modules.shap_explainer)
Tests calculation accuracy, feature count, missing input validation,
data types, and directional mapping.
"""

import unittest
import pandas as pd
import numpy as np

from modules.prediction import diabetes_model
from modules.shap_explainer import (
    explain_diabetes,
    get_diabetes_model_output,
    DIABETES_FEATURES,
)


class TestShapExplainer(unittest.TestCase):
    def setUp(self):
        self.valid_input = pd.DataFrame([{
            "Pregnancies": 6,
            "Glucose": 190,
            "BloodPressure": 95,
            "SkinThickness": 35,
            "Insulin": 250,
            "BMI": 36.5,
            "DiabetesPedigreeFunction": 1.20,
            "Age": 52,
        }])

    def test_explain_diabetes_valid_input(self):
        """Test SHAP explanation generation on standard patient data."""
        explanation = explain_diabetes(diabetes_model, self.valid_input)
        self.assertIsInstance(explanation, pd.DataFrame)
        self.assertEqual(len(explanation), 8)

        # Expected columns
        expected_cols = ["Parameter", "SHAP Impact", "Direction", "Absolute Impact"]
        self.assertEqual(list(explanation.columns), expected_cols)

        # Verify all 8 features are present
        returned_params = set(explanation["Parameter"])
        self.assertEqual(returned_params, set(DIABETES_FEATURES))

        # Check values are finite numbers
        self.assertTrue(np.isfinite(explanation["SHAP Impact"].to_numpy(dtype=float)).all())

        # Check sorting by absolute impact descending
        abs_impacts = explanation["Absolute Impact"].tolist()
        self.assertEqual(abs_impacts, sorted(abs_impacts, reverse=True))

    def test_missing_feature_raises_error(self):
        """Test that missing required features raises ValueError."""
        incomplete_input = self.valid_input.drop(columns=["Glucose"])
        with self.assertRaises(ValueError):
            explain_diabetes(diabetes_model, incomplete_input)

    def test_nan_value_raises_error(self):
        """Test that NaN values in inputs raise ValueError."""
        nan_input = self.valid_input.copy()
        nan_input.loc[0, "Glucose"] = np.nan
        with self.assertRaises(ValueError):
            explain_diabetes(diabetes_model, nan_input)

    def test_non_dataframe_raises_type_error(self):
        """Test that non-DataFrame inputs raise TypeError."""
        with self.assertRaises(TypeError):
            explain_diabetes(diabetes_model, {"Glucose": 120})

    def test_direction_column_values(self):
        """Test that direction labels match impact signs."""
        explanation = explain_diabetes(diabetes_model, self.valid_input)
        for _, row in explanation.iterrows():
            if row["SHAP Impact"] > 0:
                self.assertEqual(row["Direction"], "Increases model output")
            elif row["SHAP Impact"] < 0:
                self.assertEqual(row["Direction"], "Decreases model output")
            else:
                self.assertEqual(row["Direction"], "No impact")


if __name__ == "__main__":
    unittest.main()
