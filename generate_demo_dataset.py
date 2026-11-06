"""
Demo Labeled Dataset Generator (For Academic Testing & Pipeline Verification)
=============================================================================
This standalone utility generates a synthetic, realistic test dataset
('demo_labeled_fraud_dataset.csv') containing all 10 features PLUS the authentic
binary target variable 'Fraudulent' (0 = Legitimate, 1 = Fraudulent).

NOTE:
-----
This script does NOT modify the uploaded 'Fraud Detection Dataset-selected-columns.csv'.
It is provided exclusively so students, examiners, and evaluators can verify that
train_model.py, test_model.py, and app.py execute end-to-end when genuine target
labels are provided.
"""

import os
import numpy as np
import pandas as pd

OUTPUT_FILE = "demo_labeled_fraud_dataset.csv"
NUM_RECORDS = 5000


def generate_demo_data(num_samples: int = NUM_RECORDS, random_state: int = 42):
    np.random.seed(random_state)
    print(f"\n[INFO] Generating {num_samples:,} synthetic records for demo pipeline verification...")

    # Identifiers
    user_ids = np.random.randint(1000, 9999, size=num_samples)
    transaction_ids = [f"T{i+1}" for i in range(num_samples)]

    # Categorical distributions matching the real dataset
    transaction_types = np.random.choice(
        ["ATM Withdrawal", "Bill Payment", "POS Payment", "Bank Transfer", "Online Purchase"],
        size=num_samples,
        p=[0.20, 0.20, 0.20, 0.20, 0.20]
    )

    devices = np.random.choice(
        ["Desktop", "Mobile", "Tablet", "Unknown Device"],
        size=num_samples,
        p=[0.35, 0.35, 0.25, 0.05]
    )

    locations = np.random.choice(
        ["Boston", "New York", "Seattle", "Chicago", "Houston", "Los Angeles", "Miami", "San Francisco"],
        size=num_samples
    )

    # Numerical features
    account_age = np.random.randint(1, 120, size=num_samples) # months
    time_of_transaction = np.random.uniform(0.0, 23.9, size=num_samples)
    previous_fraud = np.random.choice([0, 1, 2, 3, 4], size=num_samples, p=[0.4, 0.25, 0.15, 0.12, 0.08])
    num_txns_24h = np.random.randint(1, 15, size=num_samples)
    
    # Base transaction amounts
    amounts = np.random.exponential(scale=1500, size=num_samples) + 10.0
    amounts = np.round(amounts, 2)

    # Introduce realistic missing values (5% missing rate)
    mask_amount = np.random.rand(num_samples) < 0.04
    amounts[mask_amount] = np.nan

    mask_device = np.random.rand(num_samples) < 0.04
    devices = list(devices)
    for i in np.where(mask_device)[0]:
        devices[i] = np.nan

    mask_loc = np.random.rand(num_samples) < 0.04
    locations = list(locations)
    for i in np.where(mask_loc)[0]:
        locations[i] = np.nan

    mask_time = np.random.rand(num_samples) < 0.04
    time_of_transaction[mask_time] = np.nan

    # Generate target 'Fraudulent' using realistic multi-factor risk scoring
    # Impute temporarily for score calculation
    filled_amounts = np.nan_to_num(amounts, nan=1500.0)
    filled_times = np.nan_to_num(time_of_transaction, nan=12.0)

    # Risk score calculation:
    # High risk: Late night hours (0-4), high transaction velocity, previous fraud history, high amount, new account
    night_risk = ((filled_times < 5.0) | (filled_times > 23.0)).astype(float) * 1.5
    velocity_risk = (num_txns_24h > 10).astype(float) * 2.0
    history_risk = (previous_fraud >= 2).astype(float) * 2.5
    amount_risk = (filled_amounts > 5000).astype(float) * 2.0
    new_account_risk = (account_age < 6).astype(float) * 1.5

    total_risk = night_risk + velocity_risk + history_risk + amount_risk + new_account_risk + np.random.normal(0, 1.2, size=num_samples)
    
    # Target label: ~7% fraud rate (realistic financial imbalance)
    threshold = np.percentile(total_risk, 93)
    fraudulent = (total_risk >= threshold).astype(int)

    df = pd.DataFrame({
        "Transaction_ID": transaction_ids,
        "User_ID": user_ids,
        "Transaction_Amount": amounts,
        "Transaction_Type": transaction_types,
        "Time_of_Transaction": np.round(time_of_transaction, 1),
        "Device_Used": devices,
        "Location": locations,
        "Previous_Fraudulent_Transactions": previous_fraud,
        "Account_Age": account_age,
        "Number_of_Transactions_Last_24H": num_txns_24h,
        "Fraudulent": fraudulent
    })

    df.to_csv(OUTPUT_FILE, index=False)
    fraud_count = df["Fraudulent"].sum()
    legit_count = len(df) - fraud_count
    print(f"[SUCCESS] Saved demo dataset to '{OUTPUT_FILE}'")
    print(f"  Total Rows: {len(df):,}")
    print(f"  Legitimate (0): {legit_count:,} ({legit_count/len(df)*100:.1f}%)")
    print(f"  Fraudulent (1): {fraud_count:,} ({fraud_count/len(df)*100:.1f}%)")
    print(f"  Columns: {list(df.columns)}")
    return OUTPUT_FILE


if __name__ == "__main__":
    generate_demo_data()

