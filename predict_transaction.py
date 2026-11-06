"""
Online Fraud Detection System - CLI Transaction Predictor
=========================================================
A lightweight command-line interface to evaluate individual transactions
using the serialized best model pipeline. Useful for terminal demonstrations,
unit testing, and automated integration checks.
"""

import os
import sys
import argparse
import pandas as pd
import joblib

MODELS_DIR = "models"
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model_pipeline.joblib")


def predict_single_transaction(data_dict: dict, model_path: str = BEST_MODEL_PATH):
    """Predict fraud probability for a single transaction dictionary."""
    if not os.path.exists(model_path):
        print(f"[ERROR] Trained model not found at '{model_path}'. Run train_model.py first.")
        sys.exit(1)

    pipeline = joblib.load(model_path)
    df = pd.DataFrame([data_dict])

    prediction = pipeline.predict(df)[0]
    
    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(df)[0]
        fraud_prob = probs[1]
        legit_prob = probs[0]
    else:
        fraud_prob = 1.0 if prediction == 1 else 0.0
        legit_prob = 1.0 - fraud_prob

    print("\n" + "=" * 65)
    print("TRANSACTION EVALUATION REPORT")
    print("=" * 65)
    print(f"  • Transaction Amount:            ${data_dict['Transaction_Amount']:,.2f}")
    print(f"  • Transaction Type:              {data_dict['Transaction_Type']}")
    print(f"  • Time of Transaction:           {data_dict['Time_of_Transaction']:04.1f} hrs")
    print(f"  • Device Used:                   {data_dict['Device_Used']}")
    print(f"  • Location:                      {data_dict['Location']}")
    print(f"  • Previous Fraud Infractions:    {data_dict['Previous_Fraudulent_Transactions']}")
    print(f"  • Account Longevity (Age):       {data_dict['Account_Age']} months")
    print(f"  • 24H Transaction Velocity:      {data_dict['Number_of_Transactions_Last_24H']} transactions")
    print("-" * 65)

    if prediction == 1:
        print("  CLASSIFICATION: 🚨 [FRAUDULENT TRANSACTION]")
        print(f"  Fraud Probability:      {fraud_prob * 100:.2f}%")
        print(f"  Legitimate Probability: {legit_prob * 100:.2f}%")
        print("  Action Recommended:     HOLD / MANUAL RISK VERIFICATION")
    else:
        print("  CLASSIFICATION: ✅ [LEGITIMATE TRANSACTION]")
        print(f"  Legitimate Confidence:  {legit_prob * 100:.2f}%")
        print(f"  Fraud Probability:      {fraud_prob * 100:.2f}%")
        print("  Action Recommended:     APPROVE / PROCESS PAYMENT")
    print("=" * 65 + "\n")
    return prediction, fraud_prob


def main():
    parser = argparse.ArgumentParser(description="Evaluate a transaction via CLI")
    parser.add_argument("--sample", choices=["legit", "fraud"], help="Run predefined sample transaction")
    parser.add_argument("--amount", type=float, default=150.0, help="Transaction amount in USD")
    parser.add_argument("--type", type=str, default="POS Payment", help="Transaction Type")
    parser.add_argument("--time", type=float, default=14.0, help="Time of transaction (0.0 - 23.9)")
    parser.add_argument("--device", type=str, default="Mobile", help="Device used")
    parser.add_argument("--location", type=str, default="Boston", help="Transaction location")
    parser.add_argument("--prev-fraud", type=int, default=0, help="Previous fraudulent transactions")
    parser.add_argument("--age", type=int, default=36, help="Account age in months")
    parser.add_argument("--num-24h", type=int, default=2, help="Number of transactions in last 24h")
    args = parser.parse_args()

    if args.sample == "legit":
        data = {
            "Transaction_Amount": 45.50,
            "Transaction_Type": "POS Payment",
            "Time_of_Transaction": 14.5,
            "Device_Used": "Mobile",
            "Location": "Boston",
            "Previous_Fraudulent_Transactions": 0,
            "Account_Age": 48,
            "Number_of_Transactions_Last_24H": 2
        }
    elif args.sample == "fraud":
        data = {
            "Transaction_Amount": 7850.0,
            "Transaction_Type": "Bank Transfer",
            "Time_of_Transaction": 3.2,
            "Device_Used": "Desktop",
            "Location": "Miami",
            "Previous_Fraudulent_Transactions": 3,
            "Account_Age": 2,
            "Number_of_Transactions_Last_24H": 14
        }
    else:
        data = {
            "Transaction_Amount": args.amount,
            "Transaction_Type": args.type,
            "Time_of_Transaction": args.time,
            "Device_Used": args.device,
            "Location": args.location,
            "Previous_Fraudulent_Transactions": args.prev_fraud,
            "Account_Age": args.age,
            "Number_of_Transactions_Last_24H": args.num_24h
        }

    predict_single_transaction(data)


if __name__ == "__main__":
    main()

