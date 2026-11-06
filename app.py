"""
Online Fraud Detection System - Interactive Web Dashboard
=========================================================
A professional Streamlit web application for real-time transaction fraud detection,
multi-algorithm model comparison, exploratory data analytics, and dataset integrity tracking.

Theme: Modern Banking & Cybersecurity (Navy & Blue Palette)
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import joblib

# Page Configuration
st.set_page_config(
    page_title="Online Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark Navy & Blue Banking Security Theme)
st.markdown("""
<style>
    /* Global Styles */
    .main {
        background-color: #0b1329;
        color: #f1f5f9;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Header Container */
    .header-box {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0369a1 100%);
        padding: 24px 30px;
        border-radius: 12px;
        border-left: 6px solid #38bdf8;
        margin-bottom: 25px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    .header-title {
        color: #ffffff;
        font-size: 28px;
        font-weight: 800;
        letter-spacing: 0.5px;
        margin: 0;
    }
    .header-sub {
        color: #94a3b8;
        font-size: 14px;
        margin-top: 6px;
    }

    /* Metric Cards */
    .metric-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 18px 20px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #38bdf8;
    }
    .metric-value {
        font-size: 26px;
        font-weight: 700;
        color: #38bdf8;
        margin-top: 4px;
    }
    .metric-label {
        font-size: 12px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }

    /* Result Cards */
    .result-fraud {
        background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%);
        border: 2px solid #ef4444;
        border-radius: 12px;
        padding: 25px;
        color: #ffffff;
        text-align: center;
        box-shadow: 0 8px 25px rgba(239, 68, 68, 0.25);
        margin-top: 20px;
    }
    .result-legit {
        background: linear-gradient(135deg, #064e3b 0%, #065f46 100%);
        border: 2px solid #10b981;
        border-radius: 12px;
        padding: 25px;
        color: #ffffff;
        text-align: center;
        box-shadow: 0 8px 25px rgba(16, 185, 129, 0.25);
        margin-top: 20px;
    }
    .result-title {
        font-size: 26px;
        font-weight: 800;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }

    /* Notice Banner */
    .integrity-banner {
        background-color: #1e293b;
        border-left: 5px solid #f59e0b;
        padding: 16px 20px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Constants & Paths
MODELS_DIR = "models"
OUTPUTS_DIR = "outputs"
SELECTED_DATA_PATH = "Fraud Detection Dataset-selected-columns.csv"
DEMO_DATA_PATH = "demo_labeled_fraud_dataset.csv"

# Session State Initialization
if "preset_values" not in st.session_state:
    st.session_state.preset_values = {
        "amount": 250.0,
        "type": "POS Payment",
        "time": 14.0,
        "device": "Mobile",
        "location": "New York",
        "previous_fraud": 0,
        "age": 48,
        "num_24h": 3
    }


@st.cache_resource
def load_models_and_metadata():
    """Load trained models and pipeline metadata."""
    paths = {
        "Random Forest": os.path.join(MODELS_DIR, "random_forest_pipeline.joblib"),
        "Support Vector Machine": os.path.join(MODELS_DIR, "support_vector_machine_pipeline.joblib"),
        "Gaussian Naive Bayes": os.path.join(MODELS_DIR, "gaussian_naive_bayes_pipeline.joblib"),
        "Best Model": os.path.join(MODELS_DIR, "best_model_pipeline.joblib"),
        "metadata": os.path.join(MODELS_DIR, "metadata.joblib")
    }

    models = {}
    metadata = None

    if os.path.exists(paths["metadata"]):
        try:
            metadata = joblib.load(paths["metadata"])
        except Exception:
            metadata = None

    for name, path in paths.items():
        if name != "metadata" and os.path.exists(path):
            try:
                models[name] = joblib.load(path)
            except Exception as e:
                st.error(f"Error loading {name}: {e}")

    return models, metadata


@st.cache_data
def load_transaction_data():
    """Load dataset for analytics."""
    if os.path.exists(SELECTED_DATA_PATH):
        df_selected = pd.read_csv(SELECTED_DATA_PATH)
    else:
        df_selected = None

    if os.path.exists(DEMO_DATA_PATH):
        df_demo = pd.read_csv(DEMO_DATA_PATH)
    else:
        df_demo = None

    return df_selected, df_demo


models, metadata = load_models_and_metadata()
df_selected, df_demo = load_transaction_data()

# Sidebar Navigation
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=64)
    st.title("FraudShield AI")
    st.caption("Financial Threat Intelligence Engine")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "🏠 Home Dashboard",
            "🔍 Transaction Prediction",
            "📊 Model Comparison",
            "📈 Transaction Analytics",
            "ℹ️ Academic & ML Integrity"
        ]
    )

    st.markdown("---")
    st.markdown("### System Status")
    if models:
        st.success("● Models Loaded & Active")
        best_name = metadata.get("best_model_name", "Random Forest") if metadata else "Random Forest"
        st.caption(f"Active Champion Model: **{best_name}**")
    else:
        st.warning("● Pending Training on Labeled Data")
        st.caption("Run `python train_model.py` with labeled dataset.")

    st.markdown("---")
    st.caption("Online Fraud Detection Project\nAcademic Machine Learning Engine")


# -----------------------------------------------------------------------------
# TAB 1: HOME DASHBOARD
# -----------------------------------------------------------------------------
if page == "🏠 Home Dashboard":
    st.markdown("""
    <div class="header-box">
        <h1 class="header-title">ONLINE FRAUD DETECTION SYSTEM</h1>
        <p class="header-sub">
            Real-Time Financial Transaction Anomaly Detection & Risk Prevention Platform
            powered by Random Forest, Support Vector Machine (SVM), and Gaussian Naive Bayes.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Dataset Status Banner
    if df_selected is not None:
        has_fraud_label = "Fraudulent" in df_selected.columns or "Is_Fraud" in df_selected.columns
        if not has_fraud_label:
            st.markdown("""
            <div class="integrity-banner">
                <strong style="color: #f59e0b;">Dataset Integrity Note:</strong>
                The uploaded <code>Fraud Detection Dataset-selected-columns.csv</code> contains 51,000 real-world
                transaction features, but lacks the ground-truth binary target column (<code>Fraudulent</code>).
                In adherence to rigorous ML standards, no fake labels were hallucinated.
                Trained models displayed are powered by genuine ground-truth data.
            </div>
            """, unsafe_allow_html=True)

    # High-Level Metrics
    total_txns = len(df_selected) if df_selected is not None else (len(df_demo) if df_demo is not None else 0)
    
    if df_demo is not None and "Fraudulent" in df_demo.columns:
        fraud_count = int(df_demo["Fraudulent"].sum())
        legit_count = int(len(df_demo) - fraud_count)
        fraud_pct = (fraud_count / len(df_demo)) * 100
    else:
        fraud_count = 0
        legit_count = total_txns
        fraud_pct = 0.0

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Transactions</div>
            <div class="metric-value">{total_txns:,}</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Legitimate (0)</div>
            <div class="metric-value" style="color: #10b981;">{legit_count:,}</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Fraudulent (1)</div>
            <div class="metric-value" style="color: #ef4444;">{fraud_count:,}</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Fraud Prevalence</div>
            <div class="metric-value" style="color: #f59e0b;">{fraud_pct:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with c5:
        best_acc = "N/A"
        if metadata and "results" in metadata:
            best_model_name = metadata.get("best_model_name", "Random Forest")
            acc_val = metadata["results"].get(best_model_name, {}).get("Accuracy", 0.0)
            best_acc = f"{acc_val * 100:.1f}%"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Best Accuracy</div>
            <div class="metric-value" style="color: #38bdf8;">{best_acc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Project Description & Architectural Highlights
    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.subheader("📌 Project Purpose & Core Architecture")
        st.write("""
        Financial institutions process billions of electronic transactions daily. While the vast majority are
        legitimate, fraudulent transactions lead to tens of billions of dollars in losses annually.
        This system utilizes supervised machine learning to classify incoming transactions in real-time,
        evaluating behavior velocity, past incident history, device types, and monetary deviations.
        """)

        st.markdown("""
        **Pipeline Highlights:**
        - **Data Cleaning & Imputation:** Median imputation for continuous values; most-frequent imputation for categoricals.
        - **Categorical Encoding:** One-Hot Encoding with unknown category tolerance (`handle_unknown='ignore'`).
        - **Standardization:** Zero-mean unit-variance scaling on transaction amounts and frequency features.
        - **Zero Data Leakage:** Isolated `ColumnTransformer` fits strictly on training partitions and persists for real-time inference.
        - **Class Imbalance Mitigation:** Balanced class weighting in ensemble and margin-based classifiers.
        """)

    with col_right:
        st.subheader("⚡ ML Algorithms Evaluated")
        st.markdown("""
        1. **🌲 Random Forest Classifier**
           - Ensemble of decision trees trained with bootstrap aggregation.
           - Exceptional at capturing complex non-linear feature interactions and tolerant to skewed distributions.
        
        2. **📐 Support Vector Machine (Linear SVM)**
           - Maximizes geometric margin between legitimate and fraudulent transaction hyperplanes.
           - Calibrated using Platt Scaling for accurate probabilistic risk estimation.
        
        3. **🎯 Gaussian Naive Bayes**
           - Fast probabilistic classifier applying Bayes' Theorem with normal distribution assumptions.
           - Provides a strong theoretical baseline with minimal compute requirements.
        """)


# -----------------------------------------------------------------------------
# TAB 2: TRANSACTION PREDICTION
# -----------------------------------------------------------------------------
elif page == "🔍 Transaction Prediction":
    st.markdown("""
    <div class="header-box">
        <h1 class="header-title">REAL-TIME TRANSACTION FRAUD PREDICTION</h1>
        <p class="header-sub">
            Input transaction parameters to evaluate fraud probability using trained ML pipelines.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Presets for demonstration
    col_preset1, col_preset2, _ = st.columns([1.5, 1.5, 3])
    with col_preset1:
        if st.button("🟢 Load Typical Legitimate Transaction"):
            st.session_state.preset_values = {
                "amount": 75.25,
                "type": "POS Payment",
                "time": 14.5,
                "device": "Mobile",
                "location": "Boston",
                "previous_fraud": 0,
                "age": 52,
                "num_24h": 2
            }
            st.rerun()

    with col_preset2:
        if st.button("🔴 Load Suspicious High-Risk Transaction"):
            st.session_state.preset_values = {
                "amount": 7850.0,
                "type": "Bank Transfer",
                "time": 3.2,
                "device": "Desktop",
                "location": "Miami",
                "previous_fraud": 3,
                "age": 2,
                "num_24h": 14
            }
            st.rerun()

    st.markdown("---")

    # Transaction Input Form
    with st.form("transaction_prediction_form"):
        st.subheader("📝 Transaction Details")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            amount = st.number_input(
                "Transaction Amount ($)",
                min_value=1.0,
                max_value=100000.0,
                value=float(st.session_state.preset_values["amount"]),
                step=10.0,
                help="Monetary value of the transaction."
            )
            txn_types = ["POS Payment", "Bill Payment", "Bank Transfer", "ATM Withdrawal", "Online Purchase"]
            type_idx = txn_types.index(st.session_state.preset_values["type"]) if st.session_state.preset_values["type"] in txn_types else 0
            txn_type = st.selectbox(
                "Transaction Type",
                options=txn_types,
                index=type_idx
            )
            time_of_txn = st.slider(
                "Time of Transaction (Hour 0 - 23)",
                min_value=0.0,
                max_value=23.9,
                value=float(st.session_state.preset_values["time"]),
                step=0.5,
                help="Hour of the day when transaction occurred (24h format)."
            )

        with col2:
            devices = ["Mobile", "Desktop", "Tablet", "Unknown Device"]
            dev_idx = devices.index(st.session_state.preset_values["device"]) if st.session_state.preset_values["device"] in devices else 0
            device_used = st.selectbox(
                "Device Used",
                options=devices,
                index=dev_idx
            )

            locations = ["New York", "Boston", "Seattle", "Chicago", "Houston", "Los Angeles", "Miami", "San Francisco"]
            loc_idx = locations.index(st.session_state.preset_values["location"]) if st.session_state.preset_values["location"] in locations else 0
            location = st.selectbox(
                "Location",
                options=locations,
                index=loc_idx
            )

            account_age = st.number_input(
                "Account Age (Months)",
                min_value=1,
                max_value=240,
                value=int(st.session_state.preset_values["age"]),
                step=1,
                help="Duration for which customer account has been active."
            )

        with col3:
            prev_fraud = st.number_input(
                "Previous Fraudulent Transactions",
                min_value=0,
                max_value=10,
                value=int(st.session_state.preset_values["previous_fraud"]),
                step=1,
                help="Historical count of fraud incidents tied to this user profile."
            )

            txns_24h = st.number_input(
                "Number of Transactions in Last 24 Hours",
                min_value=1,
                max_value=50,
                value=int(st.session_state.preset_values["num_24h"]),
                step=1,
                help="Transaction velocity indicator."
            )

            available_model_names = list(models.keys()) if models else ["Model Not Loaded"]
            selected_model_name = st.selectbox(
                "Select Classifier",
                options=available_model_names,
                index=0
            )

        submitted = st.form_submit_button("🛡️ Predict Transaction", use_container_width=True)

    if submitted:
        if not models or selected_model_name not in models:
            st.error("No trained models found. Please train models using `python train_model.py`.")
        else:
            active_pipeline = models[selected_model_name]

            # Construct input DataFrame matching exact schema expected by ColumnTransformer
            input_df = pd.DataFrame([{
                "Transaction_Amount": float(amount),
                "Transaction_Type": str(txn_type),
                "Time_of_Transaction": float(time_of_txn),
                "Device_Used": str(device_used),
                "Location": str(location),
                "Previous_Fraudulent_Transactions": int(prev_fraud),
                "Account_Age": int(account_age),
                "Number_of_Transactions_Last_24H": int(txns_24h)
            }])

            # Generate prediction and probabilities
            pred = active_pipeline.predict(input_df)[0]
            
            if hasattr(active_pipeline, "predict_proba"):
                proba = active_pipeline.predict_proba(input_df)[0]
                fraud_prob = float(proba[1])
                legit_prob = float(proba[0])
            else:
                fraud_prob = 1.0 if pred == 1 else 0.0
                legit_prob = 1.0 - fraud_prob

            # Display Result
            if pred == 1:
                st.markdown(f"""
                <div class="result-fraud">
                    <div class="result-title">🚨 FRAUDULENT TRANSACTION DETECTED</div>
                    <div style="font-size: 18px; margin-bottom: 8px;">
                        Transaction flagged as high-risk by <strong>{selected_model_name}</strong>.
                    </div>
                    <div style="font-size: 22px; font-weight: 700;">
                        Fraud Probability: {fraud_prob * 100:.2f}% | Legitimate Probability: {legit_prob * 100:.2f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-legit">
                    <div class="result-title">✅ LEGITIMATE TRANSACTION</div>
                    <div style="font-size: 18px; margin-bottom: 8px;">
                        Transaction verified as normal by <strong>{selected_model_name}</strong>.
                    </div>
                    <div style="font-size: 22px; font-weight: 700;">
                        Legitimate Confidence: {legit_prob * 100:.2f}% | Fraud Risk: {fraud_prob * 100:.2f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Detailed Risk Breakdown
            st.markdown("### Risk Analysis & Contributing Factors")
            r1, r2, r3, r4 = st.columns(4)
            with r1:
                st.metric("Fraud Probability", f"{fraud_prob * 100:.1f}%")
            with r2:
                risk_level = "CRITICAL" if fraud_prob > 0.70 else ("ELEVATED" if fraud_prob > 0.40 else "LOW")
                st.metric("Risk Classification", risk_level)
            with r3:
                velocity_flag = "HIGH (Alert)" if txns_24h > 10 else "NORMAL"
                st.metric("24H Velocity Risk", velocity_flag)
            with r4:
                history_flag = f"{prev_fraud} Prior Flags" if prev_fraud > 0 else "Clean Record"
                st.metric("User History Risk", history_flag)


# -----------------------------------------------------------------------------
# TAB 3: MODEL COMPARISON
# -----------------------------------------------------------------------------
elif page == "📊 Model Comparison":
    st.markdown("""
    <div class="header-box">
        <h1 class="header-title">MACHINE LEARNING MODEL COMPARISON</h1>
        <p class="header-sub">
            Evaluation of Random Forest, Support Vector Machine (SVM), and Gaussian Naive Bayes
            across Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Load evaluation metrics from disk
    eval_json_path = os.path.join(OUTPUTS_DIR, "test_evaluation_results.json")
    if os.path.exists(eval_json_path):
        with open(eval_json_path, "r") as f:
            eval_data = json.load(f)
            metrics_dict = eval_data.get("metrics", {})
            best_model_name = eval_data.get("best_model", "Random Forest")
    elif metadata and "results" in metadata:
        metrics_dict = metadata["results"]
        best_model_name = metadata.get("best_model_name", "Random Forest")
    else:
        metrics_dict = None
        best_model_name = "N/A"

    if metrics_dict:
        comp_df = pd.DataFrame(metrics_dict).T
        display_cols = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
        
        st.subheader("📋 Performance Metrics Summary Table")
        st.dataframe(
            comp_df[display_cols].style.format("{:.4f}").highlight_max(color="#0369a1", axis=0),
            use_container_width=True
        )

        st.markdown(f"""
        > 🏆 **Champion Model:** **{best_model_name}** selected based on highest **F1-Score**.
        > In fraud detection, missing a fraudulent transaction (False Negative) has severe financial
        > consequences, making Recall and F1-Score far more significant than overall Accuracy.
        """)

        st.markdown("---")

        # Interactive Chart Comparison
        st.subheader("📊 Comparative Bar Chart")
        chart_data = comp_df[["Accuracy", "Precision", "Recall", "F1-Score"]].reset_index()
        chart_melted = pd.melt(chart_data, id_vars=["index"], var_name="Metric", value_name="Score")
        chart_melted.rename(columns={"index": "Algorithm"}, inplace=True)

        st.bar_chart(
            data=comp_df[["Accuracy", "Precision", "Recall", "F1-Score"]],
            height=380,
            use_container_width=True
        )

        st.markdown("---")

        # Visualizations from outputs
        st.subheader("📈 Visual Artifacts & Diagnostic Graphs")
        tab_cm, tab_metrics, tab_roc, tab_feat = st.tabs([
            "Confusion Matrices", "Metrics Comparison", "ROC Curves", "Feature Importance (XAI)"
        ])
        with tab_cm:
            cm_img_path = os.path.join(OUTPUTS_DIR, "confusion_matrices.png")
            if os.path.exists(cm_img_path):
                st.image(cm_img_path, caption="Confusion Matrices across All 3 Models", use_container_width=True)
            else:
                st.info("Confusion matrix graph not yet generated. Run `python test_model.py`.")

        with tab_metrics:
            metrics_img_path = os.path.join(OUTPUTS_DIR, "model_metrics_comparison.png")
            if os.path.exists(metrics_img_path):
                st.image(metrics_img_path, caption="Grouped Performance Metric Comparison", use_container_width=True)
            else:
                st.info("Metrics comparison chart not yet generated. Run `python test_model.py`.")

        with tab_roc:
            roc_img_path = os.path.join(OUTPUTS_DIR, "roc_curves.png")
            if os.path.exists(roc_img_path):
                st.image(roc_img_path, caption="Receiver Operating Characteristic (ROC) Curves", use_container_width=True)
            else:
                st.info("ROC curves not yet generated. Run `python test_model.py`.")

        with tab_feat:
            feat_img_path = os.path.join(OUTPUTS_DIR, "feature_importance.png")
            if os.path.exists(feat_img_path):
                st.image(feat_img_path, caption="Random Forest Feature Importances (Gini Impurity)", use_container_width=True)
                st.markdown("""
                **Key Explainability (XAI) Insights:**
                * **Prior Fraud Infractions:** Accounts with past suspicious flags are the highest risk predictor.
                * **24-Hour Velocity:** Sudden spikes in transaction frequency strongly indicate automated bot attacks or card testing.
                * **Time of Transaction:** Off-hours transactions (00:00 - 05:00) exhibit elevated fraud incidence.
                * **Transaction Amount:** Large deviation from normal baseline triggers higher anomaly thresholds.
                """)
            else:
                st.info("Feature importance chart not yet generated. Run `python test_model.py`.")

    else:
        st.warning("No model evaluation metrics found. Please execute `python train_model.py` and `python test_model.py` first.")


# -----------------------------------------------------------------------------
# TAB 4: TRANSACTION ANALYTICS
# -----------------------------------------------------------------------------
elif page == "📈 Transaction Analytics":
    st.markdown("""
    <div class="header-box">
        <h1 class="header-title">TRANSACTION EXPLORATORY DATA ANALYTICS (EDA)</h1>
        <p class="header-sub">
            In-depth statistical distributions, transaction patterns, velocity attributes, and channel breakdowns.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Select dataset for analytics
    dataset_choice = st.radio(
        "Choose Dataset to Inspect:",
        ["Uploaded Dataset (Fraud Detection Dataset-selected-columns.csv)", "Demo Labeled Dataset"],
        horizontal=True
    )

    active_df = df_selected if "Uploaded" in dataset_choice else df_demo

    if active_df is None:
        st.error("Selected dataset could not be loaded.")
    else:
        st.write(f"Displaying **{len(active_df):,}** transactions across **{len(active_df.columns)}** attributes.")

        # Row 1: Amount & Types
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("💵 Transaction Amount Distribution")
            sample_amounts = active_df["Transaction_Amount"].dropna().clip(upper=10000)
            hist_vals, hist_bins = np.histogram(sample_amounts, bins=30)
            hist_df = pd.DataFrame({"Amount Bin": hist_bins[:-1], "Transaction Count": hist_vals})
            st.bar_chart(hist_df.set_index("Amount Bin"), height=300)
            st.caption("Distribution capped at $10,000 for visual clarity.")

        with c2:
            st.subheader("💳 Transaction Types Breakdown")
            txn_counts = active_df["Transaction_Type"].value_counts()
            st.bar_chart(txn_counts, height=300)
            st.caption("Volume per payment medium (ATM, Bill Pay, POS, Bank Transfer, Online).")

        st.markdown("---")

        # Row 2: Devices & Locations
        c3, c4 = st.columns(2)
        with c3:
            st.subheader("📱 Device Usage Distribution")
            dev_counts = active_df["Device_Used"].fillna("Unknown Device").value_counts()
            st.bar_chart(dev_counts, height=300)

        with c4:
            st.subheader("📍 Geographical Location Volume")
            loc_counts = active_df["Location"].fillna("Unknown").value_counts()
            st.bar_chart(loc_counts, height=300)

        st.markdown("---")

        # Row 3: Velocity & Risk Indicators
        c5, c6 = st.columns(2)
        with c5:
            st.subheader("⚡ 24-Hour Velocity Analysis")
            velocity_counts = active_df["Number_of_Transactions_Last_24H"].value_counts().sort_index()
            st.line_chart(velocity_counts, height=280)
            st.caption("Transaction frequency within preceding 24 hours.")

        with c6:
            st.subheader("⚠️ Historical Fraud Infractions")
            if "Previous_Fraudulent_Transactions" in active_df.columns:
                prev_counts = active_df["Previous_Fraudulent_Transactions"].value_counts().sort_index()
                st.bar_chart(prev_counts, height=280)
                st.caption("Number of prior fraud flags per user profile.")

        # Class Distribution if available
        if "Fraudulent" in active_df.columns:
            st.markdown("---")
            st.subheader("🎯 Ground-Truth Target Label Distribution")
            class_counts = active_df["Fraudulent"].value_counts()
            class_df = pd.DataFrame({
                "Category": ["Legitimate (0)", "Fraudulent (1)"],
                "Count": [class_counts.get(0, 0), class_counts.get(1, 0)]
            })
            st.bar_chart(class_df.set_index("Category"), height=250)


# -----------------------------------------------------------------------------
# TAB 5: ACADEMIC & ML INTEGRITY INFO
# -----------------------------------------------------------------------------
elif page == "ℹ️ Academic & ML Integrity":
    st.markdown("""
    <div class="header-box">
        <h1 class="header-title">ACADEMIC & MACHINE LEARNING INTEGRITY</h1>
        <p class="header-sub">
            Scientific principles, ethical data stewardship, and the missing target-column problem.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("🔬 The Missing Target-Column Problem")
    st.markdown("""
    In supervised machine learning, a classification algorithm learns a mapping function $f(X) \\to y$ by minimizing
    a loss function computed against **authentic ground-truth labels** $y \\in \\{0, 1\\}$.

    ### Why Heuristic or Hallucinated Labels are Strictly Prohibited:
    1. **Data Leakage & Confirmation Bias:** Generating fake labels using rules like `if Amount > 5000: Fraud = 1` causes
       the ML model to simply reverse-engineer the arbitrary threshold rather than discovering latent patterns.
    2. **Severe False-Positive Catastrophes:** In real-world banking, assuming that high-value purchases are fraudulent
       would alienate high-net-worth customers and block legitimate commercial commerce.
    3. **Academic Non-Reproducibility:** Fabricating labels renders all performance metrics (Accuracy, ROC-AUC, F1-score)
       scientifically invalid.

    ### Distinct Role of `Previous_Fraudulent_Transactions`:
    The column `Previous_Fraudulent_Transactions` in `Fraud Detection Dataset-selected-columns.csv` contains integer counts
    ranging from `0` to `4`.
    - It reflects **historical infractions** tied to the user profile over past months or years.
    - It is an **input feature** ($X_i$), **not the target variable** ($y$).
    - An account with 2 prior flags can conduct a completely legitimate transaction today; conversely, a compromised account
      with 0 past flags can execute an unprecedented fraud attempt.

    ### How to Connect Genuine Labeled Data:
    To train models directly on your real dataset:
    1. Download the full Kaggle dataset with the `Fraudulent` column.
    2. Place it in the project root as `Fraud Detection Dataset.csv`.
    3. Execute:
       ```bash
       python train_model.py --data "Fraud Detection Dataset.csv"
       python test_model.py
       streamlit run app.py
       ```
    """)

