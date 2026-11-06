"""
Online Fraud Detection System - Model Training Script
=====================================================
This script prepares data, builds preprocessing pipelines, handles class imbalance,
and trains three supervised machine learning algorithms:
1. Random Forest Classifier
2. Support Vector Machine (Linear SVM with Platt Calibration)
3. Gaussian Naive Bayes

Crucial ML Integrity Constraint:
---------------------------------
Supervised classification requires authentic ground-truth labels.
If the dataset lacks a genuine target column (e.g. 'Fraudulent' or 'Is_Fraud'),
this script explicitly halts training, reports the diagnostic schema,
and instructs the user on providing the labeled dataset without fabricating fake labels.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# Configuration & Paths
DEFAULT_DATA_PATH = "Fraud Detection Dataset-selected-columns.csv"
MODELS_DIR = "models"
OUTPUTS_DIR = "outputs"

CANDIDATE_TARGET_COLUMNS = [
    "Fraudulent", "Is_Fraud", "is_fraud", "Class", "Target",
    "fraud", "fraudulent", "isFraud", "IsFraud"
]

NUMERICAL_FEATURES = [
    "Transaction_Amount",
    "Time_of_Transaction",
    "Previous_Fraudulent_Transactions",
    "Account_Age",
    "Number_of_Transactions_Last_24H"
]

CATEGORICAL_FEATURES = [
    "Transaction_Type",
    "Device_Used",
    "Location"
]

EXCLUDED_COLUMNS = ["Transaction_ID", "User_ID"]


def ensure_directories():
    """Ensure required model and output directories exist."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)


def load_dataset(file_path: str) -> pd.DataFrame:
    """Load dataset from disk with error handling."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file '{file_path}' was not found.")
    print(f"\n[INFO] Loading dataset from: {file_path}")
    df = pd.read_csv(file_path)
    print(f"[INFO] Dataset loaded successfully. Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


def inspect_and_identify_target(df: pd.DataFrame, user_target: str = None) -> str:
    """
    Inspect columns and verify the presence of a genuine ground-truth target variable.
    Returns target column name if found, else returns None.
    """
    print("\n" + "=" * 80)
    print("STEP 1: DATASET INSPECTION & TARGET IDENTIFICATION")
    print("=" * 80)
    print("Columns detected:")
    for idx, col in enumerate(df.columns, 1):
        print(f"  {idx:2d}. {col:35s} (dtype: {df[col].dtype})")

    # If user explicitly supplied target column
    if user_target:
        if user_target in df.columns:
            print(f"\n[INFO] User specified target column: '{user_target}'")
            return user_target
        else:
            print(f"\n[ERROR] Specified target column '{user_target}' not found in dataset.")
            return None

    # Auto-detect from candidate target list
    for candidate in CANDIDATE_TARGET_COLUMNS:
        if candidate in df.columns:
            print(f"\n[INFO] Identified genuine target column: '{candidate}'")
            return candidate

    return None


def print_missing_target_notice(df: pd.DataFrame, file_path: str):
    """
    Display a comprehensive academic integrity notice when target labels are missing.
    Explains the difference between features (e.g. Previous_Fraudulent_Transactions)
    and the target variable, and halts training without fabricating fake data.
    """
    print("\n" + "!" * 80)
    print("CRITICAL MACHINE LEARNING NOTICE: GROUND-TRUTH TARGET COLUMN MISSING")
    print("!" * 80)
    print(f"\nDataset inspected: '{file_path}'")
    print(f"Total rows: {len(df):,} | Total columns: {len(df.columns)}")
    print("\nAnalysis of columns present:")
    print("  - Identifiers (Excluded from ML): Transaction_ID, User_ID")
    print("  - Numerical Input Features: Transaction_Amount, Time_of_Transaction,")
    print("                             Account_Age, Number_of_Transactions_Last_24H")
    print("  - Categorical Input Features: Transaction_Type, Device_Used, Location")
    print("  - Historical Profile Feature: Previous_Fraudulent_Transactions")
    
    if "Previous_Fraudulent_Transactions" in df.columns:
        counts = df["Previous_Fraudulent_Transactions"].value_counts().to_dict()
        print("\n  [Key Distinction] 'Previous_Fraudulent_Transactions':")
        print(f"    Value distribution: {counts}")
        print("    This column records historical integer counts (0-4) of past infractions")
        print("    associated with the user profile. It is NOT the binary target label (0 or 1)")
        print("    for whether THIS current transaction is fraudulent.")

    print("\nACADEMIC & PROFESSIONAL INTEGRITY POLICY:")
    print("  1. No synthetic fraud labels will be generated or hallucinated.")
    print("  2. Arbitrary heuristics (e.g. assuming high amounts = fraud) will NOT be used.")
    print("  3. Supervised classification algorithms (Random Forest, SVM, Naive Bayes)")
    print("     cannot be trained without genuine ground-truth target labels.")
    print("  4. The rest of the project pipeline (preprocessing, pipelines, testing,")
    print("     Streamlit app) is structured and ready.")

    print("\nACTION REQUIRED TO COMPLETE MODEL TRAINING:")
    print("  Please provide the complete dataset containing the binary target column,")
    print("  typically named 'Fraudulent' (0 = Legitimate, 1 = Fraudulent).")
    print("  In the original Kaggle 'Fraud Detection Dataset' (~51,000 rows), this column")
    print("  is present alongside 'Payment_Method'.")
    print("\n  Once the labeled CSV is available, run:")
    print("      python train_model.py --data \"path/to/Fraud_Detection_Dataset_with_labels.csv\"")
    print("=" * 80 + "\n")


def build_preprocessing_pipeline(num_cols, cat_cols):
    """
    Construct ColumnTransformer for reproducible preprocessing without data leakage:
    - Numerical: Median imputation + StandardScaler
    - Categorical: Most frequent imputation + OneHotEncoder
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols)
        ],
        remainder="drop"
    )

    return preprocessor


def train_and_evaluate(df: pd.DataFrame, target_col: str):
    """
    End-to-end model training, evaluation, comparison, and serialization.
    """
    ensure_directories()

    print("\n" + "=" * 80)
    print("STEP 2: DATA CLEANING & PREPROCESSING")
    print("=" * 80)

    # Validate target distribution
    target_counts = df[target_col].value_counts()
    print(f"Target column: '{target_col}'")
    print(f"Class distribution:\n{target_counts}")
    fraud_rate = (target_counts.get(1, 0) / len(df)) * 100
    print(f"Fraud rate: {fraud_rate:.2f}% (Handling class imbalance during training)")

    # Exclude identifiers
    drop_cols = [c for c in EXCLUDED_COLUMNS if c in df.columns] + [target_col]
    X = df.drop(columns=drop_cols)
    y = df[target_col].astype(int)

    # Verify feature sets
    num_cols = [c for c in NUMERICAL_FEATURES if c in X.columns]
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in X.columns]
    print(f"\nFeatures selected for modeling ({len(num_cols) + len(cat_cols)} total):")
    print(f"  Numerical ({len(num_cols)}):   {num_cols}")
    print(f"  Categorical ({len(cat_cols)}): {cat_cols}")
    print(f"  Excluded:      {drop_cols[:-1]}")

    print("\n" + "=" * 80)
    print("STEP 3: STRATIFIED TRAIN-TEST SPLIT (80:20)")
    print("=" * 80)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
    print(f"Training set: {X_train.shape[0]:,} samples")
    print(f"Testing set:  {X_test.shape[0]:,} samples")

    # Save isolated test dataset for test_model.py (Strict zero-leakage guarantee)
    test_df = X_test.copy()
    test_df[target_col] = y_test
    test_path = os.path.join(MODELS_DIR, "test_data.csv")
    test_df.to_csv(test_path, index=False)
    print(f"[INFO] Saved separated test set to: {test_path}")

    # Build model pipelines
    print("\n" + "=" * 80)
    print("STEP 4: TRAINING MACHINE LEARNING ALGORITHMS")
    print("=" * 80)

    preprocessor = build_preprocessing_pipeline(num_cols, cat_cols)

    models = {
        "Random Forest": Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            ))
        ]),
        "Support Vector Machine": Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", CalibratedClassifierCV(
                estimator=LinearSVC(
                    class_weight="balanced",
                    dual=False,
                    max_iter=2000,
                    random_state=42
                ),
                cv=3
            ))
        ]),
        "Gaussian Naive Bayes": Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", GaussianNB())
        ])
    }

    results = {}
    saved_paths = {}

    for model_name, pipeline in models.items():
        print(f"\n--> Training {model_name}...")
        pipeline.fit(X_train, y_train)

        # Predictions on test split
        y_pred = pipeline.predict(X_test)
        
        # Probabilities if available
        if hasattr(pipeline, "predict_proba"):
            y_proba = pipeline.predict_proba(X_test)[:, 1]
            auc_val = float(roc_auc_score(y_test, y_proba))
        else:
            auc_val = 0.0

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))

        results[model_name] = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "ROC-AUC": auc_val
        }

        # Save individual model pipeline
        slug = model_name.lower().replace(" ", "_")
        model_path = os.path.join(MODELS_DIR, f"{slug}_pipeline.joblib")
        joblib.dump(pipeline, model_path)
        saved_paths[model_name] = model_path
        print(f"    Saved: {model_path}")
        print(f"    Metrics -> Acc: {acc:.4f} | Prec: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | AUC: {auc_val:.4f}")

    print("\n" + "=" * 80)
    print("STEP 5: MODEL COMPARISON & BEST MODEL SELECTION")
    print("=" * 80)
    results_df = pd.DataFrame(results).T
    print(results_df.to_string())

    # In financial fraud detection, F1-Score (harmonic mean of Precision & Recall)
    # is the preferred selection criterion because false negatives (missed fraud)
    # carry immense direct financial liability.
    best_model_name = results_df["F1-Score"].idxmax()
    best_pipeline = models[best_model_name]
    best_model_path = os.path.join(MODELS_DIR, "best_model_pipeline.joblib")
    joblib.dump(best_pipeline, best_model_path)
    print(f"\n[BEST MODEL SELECTED] '{best_model_name}' (Highest F1-Score: {results[best_model_name]['F1-Score']:.4f})")
    print(f"Saved best model separately to: {best_model_path}")

    # Save metadata for web application and testing script
    metadata = {
        "target_col": target_col,
        "numerical_features": num_cols,
        "categorical_features": cat_cols,
        "best_model_name": best_model_name,
        "results": results,
        "categorical_options": {
            col: sorted([str(x) for x in df[col].dropna().unique().tolist()])
            for col in cat_cols
        }
    }
    metadata_path = os.path.join(MODELS_DIR, "metadata.joblib")
    joblib.dump(metadata, metadata_path)

    # Save summary JSON
    summary_path = os.path.join(OUTPUTS_DIR, "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"[INFO] Training summary saved to: {summary_path}")

    print("\n" + "=" * 80)
    print("TRAINING PROCESS COMPLETED SUCCESSFULLY")
    print("=" * 80)
    return results


def main():
    parser = argparse.ArgumentParser(description="Train Online Fraud Detection ML Models")
    parser.add_argument(
        "--data",
        type=str,
        default=DEFAULT_DATA_PATH,
        help="Path to CSV dataset (default: Fraud Detection Dataset-selected-columns.csv)"
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Name of ground-truth target column (e.g. Fraudulent, Is_Fraud)"
    )
    args = parser.parse_args()

    ensure_directories()

    try:
        df = load_dataset(args.data)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    # Inspect columns and identify target variable
    target_col = inspect_and_identify_target(df, user_target=args.target)

    if target_col is None:
        # Ground-truth target column is missing. Stop supervised training stage cleanly.
        print_missing_target_notice(df, args.data)
        sys.exit(2)

    # Target is present: proceed with full training pipeline
    train_and_evaluate(df, target_col)


if __name__ == "__main__":
    main()

