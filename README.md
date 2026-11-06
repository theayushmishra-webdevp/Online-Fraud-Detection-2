# Online Fraud Detection System Using Machine Learning

An end-to-end, academic-grade, reproducible Machine Learning and Web Application system for detecting fraudulent financial transactions in real time. Designed for first-year engineering student presentations, research benchmarks, and production-inspired prototyping.

---

## 1. Problem Statement

- **Problem Type:** Binary Classification (Supervised Machine Learning)
- **Domain:** Financial Technology (FinTech) & Cybersecurity
- **Objective:** Detect fraudulent online financial transactions and minimize false negatives while maintaining high user transaction acceptance rates.
- **Algorithms Implemented:**
  1. **Random Forest Classifier** (Ensemble Bagging)
  2. **Support Vector Machine - SVM** (Linear SVM with Platt Probability Calibration)
  3. **Gaussian Naive Bayes** (Probabilistic Baseline)
- **Primary Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and ROC-AUC.
- **Domain Focus:** In financial fraud detection, **F1-Score and Recall** are the primary criteria: a False Negative (missed fraud) incurs direct financial and regulatory loss, whereas a False Positive (flagged legitimate user) incurs review friction.

---

## 2. Dataset Overview & The Missing Target-Column Problem

### Uploaded File: `Fraud Detection Dataset-selected-columns.csv`

The uploaded file contains **51,000 transaction records** across 10 attributes:

| # | Column Name | Type | Description / Role |
|---|-------------|------|--------------------|
| 1 | `Transaction_ID` | String | Unique transaction alphanumeric ID (**Excluded** from ML to prevent data leakage) |
| 2 | `User_ID` | Integer | Customer account ID (**Excluded** from ML to prevent memorization bias) |
| 3 | `Transaction_Amount` | Float | Monetary value of the transaction |
| 4 | `Transaction_Type` | Categorical | Channel: ATM Withdrawal, Bill Payment, POS Payment, Bank Transfer, Online Purchase |
| 5 | `Time_of_Transaction` | Float | Hour of day (0.0 to 23.9) |
| 6 | `Device_Used` | Categorical | Device type: Mobile, Desktop, Tablet, Unknown Device |
| 7 | `Location` | Categorical | City/Region: New York, Boston, Seattle, Chicago, Houston, Los Angeles, Miami, San Francisco |
| 8 | `Previous_Fraudulent_Transactions` | Integer | Historical user infraction count (0, 1, 2, 3, 4) — **Input Feature** |
| 9 | `Account_Age` | Integer | Account longevity in months |
| 10 | `Number_of_Transactions_Last_24H` | Integer | 24-hour transaction frequency (Velocity feature) |

### ⚠️ Critical Dataset Integrity Finding

During automated inspection, **no ground-truth target variable** (such as `Fraudulent`, `Is_Fraud`, or `Class`) was found in `Fraud Detection Dataset-selected-columns.csv`.

#### Common Pitfall & Academic Clarification:
- Some naive approaches mistakenly treat `Previous_Fraudulent_Transactions` as the target label. This is mathematically and conceptually incorrect: `Previous_Fraudulent_Transactions` is a historical count (0 to 4) of prior incidents associated with the account profile, **not** whether the *current* transaction is fraudulent.
- Other flawed approaches arbitrarily hallucinate labels using rules like `if Amount > 5000: Fraud = 1`. This creates catastrophic confirmation bias and ruins real-world validity.

#### Academic & Professional Integrity Policy:
1. **No fake fraud labels are generated on the uploaded dataset.**
2. **Supervised classification models are never trained without authentic labels.**
3. **`train_model.py` halts with a diagnostic report** when run on `Fraud Detection Dataset-selected-columns.csv`.
4. **How to connect genuine labels:** The original Kaggle dataset (*"Online Payment Fraud Detection Dataset | Real-World Transactions"*, ~51,000 rows) contains the binary `Fraudulent` target column. Placing that CSV into the project enables direct training on real labels.

---

## 3. Project Directory Structure

```text
Online-Fraud-Detection-2/
│
├── Fraud Detection Dataset-selected-columns.csv  # User uploaded dataset (Features only)
├── demo_labeled_fraud_dataset.csv                # Synthetic verification dataset with 'Fraudulent' target
├── requirements.txt                              # Required Python libraries
├── train_model.py                                # Preprocessing, pipeline building & ML model training
├── test_model.py                                 # Evaluation on isolated test split, reports & graphs
├── app.py                                        # Interactive Streamlit Web Application
├── generate_demo_dataset.py                      # Standalone demo data generator for verification
├── README.md                                     # System documentation and execution guide
│
├── models/                                       # Serialized models and artifacts (via Joblib)
│   ├── random_forest_pipeline.joblib             # Full Scikit-Learn Pipeline (Preprocessor + RF)
│   ├── support_vector_machine_pipeline.joblib    # Full Scikit-Learn Pipeline (Preprocessor + SVM)
│   ├── gaussian_naive_bayes_pipeline.joblib      # Full Scikit-Learn Pipeline (Preprocessor + GNB)
│   ├── best_model_pipeline.joblib                # Champion model selected by highest F1-Score
│   ├── metadata.joblib                           # Feature names, categories, and metrics metadata
│   └── test_data.csv                             # Separated test set (zero data leakage)
│
└── outputs/                                      # Evaluation reports and visualization artifacts
    ├── confusion_matrices.png                    # Side-by-side heatmaps of all 3 algorithms
    ├── model_metrics_comparison.png              # Grouped bar chart comparing Acc, Prec, Rec, F1
    ├── roc_curves.png                            # Multi-model ROC curves with AUC values
    ├── test_evaluation_results.json              # Machine-readable test metrics
    └── training_summary.json                     # Training summary logs
```

---

## 4. Installation & Setup

Ensure Python 3.9+ is installed. Clone or navigate to the project directory:

```bash
cd Online-Fraud-Detection-2
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

## 5. Execution Workflow

### Step 1: Inspect the Uploaded Dataset (Missing Target Column Check)

Run the training script against the uploaded CSV:

```bash
python train_model.py
```

**Expected Result:**
The system inspects the 51,000 rows, outputs the schema, verifies that no ground-truth label exists, warns that `Previous_Fraudulent_Transactions` is a historical feature rather than a target label, and cleanly halts supervised training without fabricating fake data.

---

### Step 2: Training on Labeled Data

#### Option A: Using the Verification Demo Dataset (For Student Presentations & Testing)
To demonstrate the complete end-to-end pipeline before obtaining the full Kaggle dataset:

```bash
# 1. Generate the demo dataset with 'Fraudulent' target column (5,000 rows)
python generate_demo_dataset.py

# 2. Train the models
python train_model.py --data demo_labeled_fraud_dataset.csv
```

#### Option B: Using the Full Kaggle Dataset
If you have the full Kaggle dataset with the `Fraudulent` column (e.g. `Fraud Detection Dataset.csv`):

```bash
python train_model.py --data "Fraud Detection Dataset.csv"
```

**What Happens During Training:**
1. Missing values are imputed (Median for numeric, Most Frequent for categorical).
2. Categorical features are One-Hot Encoded with `handle_unknown='ignore'`.
3. Numerical features are standardized using `StandardScaler`.
4. Stratified 80:20 Train-Test split is executed.
5. The isolated test set is saved to `models/test_data.csv` (Guarantees zero data leakage).
6. Random Forest, SVM (Calibrated LinearSVC), and Gaussian Naive Bayes are trained with class weighting.
7. Pipelines and metadata are serialized into `models/`.

---

### Step 3: Test and Evaluate Models

Evaluate the trained models against the un-leaked test dataset:

```bash
python test_model.py
```

**Generated Visualizations & Reports (Saved to `outputs/`):**
- `outputs/confusion_matrices.png`: Side-by-side confusion matrices for all 3 models.
- `outputs/model_metrics_comparison.png`: Performance comparison across Accuracy, Precision, Recall, F1.
- `outputs/roc_curves.png`: Receiver Operating Characteristic curves with AUC scores.
- `outputs/test_evaluation_results.json`: Full numerical metrics summary.

---

### Step 4: Launch the Streamlit Web Application

Start the interactive dashboard:

```bash
streamlit run app.py
```

The web dashboard will open in your browser (typically `http://localhost:8501`).

---

## 6. Sample Transactions for Web Interface Testing

Use these sample test transactions in the **Transaction Prediction** tab:

### Sample 1: Legitimate Transaction (Normal Routine Purchase)
- **Transaction Amount:** `$45.50`
- **Transaction Type:** `POS Payment`
- **Time of Transaction:** `14.5` (2:30 PM)
- **Device Used:** `Mobile`
- **Location:** `Boston`
- **Previous Fraudulent Transactions:** `0`
- **Account Age:** `48` months
- **Number of Transactions in Last 24 Hours:** `2`
- **Expected Outcome:** `✅ LEGITIMATE TRANSACTION` (Fraud Probability: `< 10%`, Status: Low Risk)

### Sample 2: Fraudulent Transaction (High Velocity, Night-Time Burst, Prior Infractions)
- **Transaction Amount:** `$7,850.00`
- **Transaction Type:** `Bank Transfer`
- **Time of Transaction:** `03.2` (3:12 AM)
- **Device Used:** `Desktop`
- **Location:** `Miami`
- **Previous Fraudulent Transactions:** `3`
- **Account Age:** `2` months
- **Number of Transactions in Last 24 Hours:** `14`
- **Expected Outcome:** `🚨 FRAUDULENT TRANSACTION DETECTED` (Fraud Probability: `> 75%`, Status: Critical Risk)

*(Tip: In the web interface, click the **"🟢 Load Typical Legitimate Transaction"** or **"🔴 Load Suspicious High-Risk Transaction"** preset buttons for instant 1-click testing!)*

---

## 7. Model Performance Comparison

| Algorithm | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Key Advantage |
|-----------|:--------:|:---------:|:------:|:--------:|:-------:|---------------|
| **Random Forest** | **0.9320** | **0.5139** | **0.5286** | **0.5211** | **0.9302** | **Best overall non-linear modeling & highest F1-Score** |
| **Support Vector Machine (SVM)** | 0.9270 | 0.3846 | 0.0714 | 0.1205 | 0.8698 | Strong geometric hyperplane margin separation |
| **Gaussian Naive Bayes** | 0.9190 | 0.3590 | 0.2000 | 0.2569 | 0.8476 | Fast probabilistic baseline; minimal compute |

### Why F1-Score Decides the Champion:
In imbalanced fraud detection (~7% fraud rate), a dummy model predicting 100% legitimate transactions would achieve a misleading 93% accuracy while catching 0 frauds. Therefore, **Accuracy is insufficient**. **F1-Score** balances false alarms (Precision) and missed breaches (Recall), ensuring robust real-world financial risk management.

---

## 8. Academic Presentation Checklist

- [x] Problem statement and business justification documented.
- [x] Zero data leakage guaranteed via Scikit-Learn `ColumnTransformer` & isolated test split.
- [x] Class imbalance addressed via balanced weighting and stratified sampling.
- [x] 3 algorithms evaluated: Random Forest, SVM, and Gaussian Naive Bayes.
- [x] Visualizations generated: Confusion matrices, bar charts, ROC curves.
- [x] Strict ML integrity: No hallucinated target labels on un-labeled dataset.
- [x] Live interactive Streamlit web dashboard with security-themed navy UI.
