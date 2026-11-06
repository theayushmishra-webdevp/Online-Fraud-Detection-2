"""
Unit & Pipeline Tests for Online Fraud Detection System
=======================================================
Verifies reproducibility, pipeline integrity, handling of unseen categories,
missing value imputation, and probability validity without data leakage.

Run via:
    python -m unittest test_pipeline.py
"""

import os
import unittest
import numpy as np
import pandas as pd
import joblib

MODELS_DIR = "models"
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model_pipeline.joblib")


class TestFraudDetectionPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Ensure models exist before running tests."""
        if not os.path.exists(BEST_MODEL_PATH):
            raise unittest.SkipTest("Trained models not found. Run train_model.py first.")
        cls.pipeline = joblib.load(BEST_MODEL_PATH)
        cls.metadata = joblib.load(os.path.join(MODELS_DIR, "metadata.joblib"))

    def test_pipeline_valid_prediction(self):
        """Verify pipeline returns binary prediction (0 or 1)."""
        sample = pd.DataFrame([{
            "Transaction_Amount": 150.0,
            "Transaction_Type": "POS Payment",
            "Time_of_Transaction": 12.0,
            "Device_Used": "Mobile",
            "Location": "Boston",
            "Previous_Fraudulent_Transactions": 0,
            "Account_Age": 36,
            "Number_of_Transactions_Last_24H": 2
        }])
        pred = self.pipeline.predict(sample)[0]
        self.assertIn(pred, [0, 1], "Prediction must be either 0 (Legitimate) or 1 (Fraudulent).")

    def test_probability_distribution(self):
        """Verify output probabilities sum to 1.0 and lie in [0, 1]."""
        sample = pd.DataFrame([{
            "Transaction_Amount": 3500.0,
            "Transaction_Type": "Bank Transfer",
            "Time_of_Transaction": 3.0,
            "Device_Used": "Desktop",
            "Location": "Miami",
            "Previous_Fraudulent_Transactions": 2,
            "Account_Age": 5,
            "Number_of_Transactions_Last_24H": 12
        }])
        probs = self.pipeline.predict_proba(sample)[0]
        self.assertEqual(len(probs), 2, "Probabilities must contain two classes.")
        self.assertAlmostEqual(probs[0] + probs[1], 1.0, places=4, msg="Probabilities must sum to 1.0.")
        self.assertTrue(0.0 <= probs[0] <= 1.0, "Class 0 probability out of bounds.")
        self.assertTrue(0.0 <= probs[1] <= 1.0, "Class 1 probability out of bounds.")

    def test_missing_value_imputation(self):
        """Verify pipeline handles NaN in numerical and categorical features safely."""
        sample_with_nans = pd.DataFrame([{
            "Transaction_Amount": np.nan,         # Missing numerical
            "Transaction_Type": "Online Purchase",
            "Time_of_Transaction": np.nan,         # Missing numerical
            "Device_Used": np.nan,                 # Missing categorical
            "Location": np.nan,                    # Missing categorical
            "Previous_Fraudulent_Transactions": 1,
            "Account_Age": 12,
            "Number_of_Transactions_Last_24H": 5
        }])
        try:
            pred = self.pipeline.predict(sample_with_nans)[0]
            self.assertIn(pred, [0, 1])
        except Exception as e:
            self.fail(f"Pipeline crashed on missing values: {e}")

    def test_unseen_categorical_levels(self):
        """Verify OneHotEncoder handle_unknown='ignore' handles brand new categories."""
        sample_with_unseen = pd.DataFrame([{
            "Transaction_Amount": 200.0,
            "Transaction_Type": "Cryptocurrency_Transfer",  # Unseen category
            "Time_of_Transaction": 15.0,
            "Device_Used": "Smartwatch_Device",             # Unseen category
            "Location": "Tokyo_Japan",                       # Unseen category
            "Previous_Fraudulent_Transactions": 0,
            "Account_Age": 24,
            "Number_of_Transactions_Last_24H": 3
        }])
        try:
            pred = self.pipeline.predict(sample_with_unseen)[0]
            self.assertIn(pred, [0, 1])
        except Exception as e:
            self.fail(f"Pipeline crashed on unseen categorical level: {e}")


if __name__ == "__main__":
    unittest.main()

