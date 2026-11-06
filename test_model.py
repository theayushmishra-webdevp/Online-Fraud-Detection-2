"""
Online Fraud Detection System - Model Testing & Evaluation Script
=================================================================
This script evaluates the three saved models on the separated test dataset:
1. Random Forest Classifier
2. Support Vector Machine (Calibrated Linear SVM)
3. Gaussian Naive Bayes

Key Requirements:
-----------------
- No training on the test set (Strict zero-leakage guarantee).
- Calculates: Accuracy, Precision, Recall, F1-Score, Confusion Matrix.
- Visualizations: Confusion matrices, metric comparisons, ROC curves.
- Identifies the best model based on domain-justified criteria (F1-score & Recall).
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd

# Direct matplotlib cache to writable directory
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

MODELS_DIR = "models"
OUTPUTS_DIR = "outputs"
DEFAULT_TEST_PATH = os.path.join(MODELS_DIR, "test_data.csv")


def ensure_directories():
    os.makedirs(OUTPUTS_DIR, exist_ok=True)


def check_models_exist():
    """Verify that models and metadata exist in models/ directory."""
    required_files = [
        os.path.join(MODELS_DIR, "random_forest_pipeline.joblib"),
        os.path.join(MODELS_DIR, "support_vector_machine_pipeline.joblib"),
        os.path.join(MODELS_DIR, "gaussian_naive_bayes_pipeline.joblib"),
        os.path.join(MODELS_DIR, "best_model_pipeline.joblib"),
        os.path.join(MODELS_DIR, "metadata.joblib"),
        DEFAULT_TEST_PATH
    ]
    missing = [f for f in required_files if not os.path.exists(f)]
    return missing


def load_test_data(test_path: str):
    """Load separated test set and extract features & target."""
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test dataset not found at '{test_path}'.")
    
    metadata_path = os.path.join(MODELS_DIR, "metadata.joblib")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError("Model metadata not found. Run train_model.py first.")
        
    metadata = joblib.load(metadata_path)
    target_col = metadata.get("target_col", "Fraudulent")

    df_test = pd.read_csv(test_path)
    if target_col not in df_test.columns:
        raise ValueError(f"Target column '{target_col}' missing from test dataset.")

    y_test = df_test[target_col].astype(int)
    X_test = df_test.drop(columns=[target_col])
    return X_test, y_test, metadata


def plot_feature_importance(pipeline, save_path):
    """Generate horizontal bar chart of Random Forest feature importances."""
    try:
        preprocessor = pipeline.named_steps["preprocessor"]
        classifier = pipeline.named_steps["classifier"]
        if hasattr(classifier, "feature_importances_"):
            feature_names = preprocessor.get_feature_names_out()
            clean_names = [
                f.replace("num__", "").replace("cat__", "").replace("_", " ")
                for f in feature_names
            ]
            importances = classifier.feature_importances_
            feat_df = pd.DataFrame({"Feature": clean_names, "Importance": importances})
            feat_df = feat_df.sort_values(by="Importance", ascending=True).tail(10)

            plt.figure(figsize=(10, 6))
            plt.barh(feat_df["Feature"], feat_df["Importance"], color="#0284C7", edgecolor="#0369A1")
            plt.title("Top 10 Feature Importances (Random Forest Gini Importance)", fontsize=14, weight="bold", pad=15)
            plt.xlabel("Mean Decrease in Impurity (Feature Importance)", fontsize=11)
            plt.ylabel("Transaction Attribute", fontsize=11)
            plt.grid(axis="x", linestyle="--", alpha=0.4)

            for i, (val, feat) in enumerate(zip(feat_df["Importance"], feat_df["Feature"])):
                plt.text(val + 0.005, i, f"{val:.3f}", va="center", fontsize=9, fontweight="semibold")

            plt.tight_layout()
            plt.savefig(save_path, dpi=300)
            plt.close()
            print(f"[INFO] Feature importance plot saved to: {save_path}")
    except Exception as e:
        print(f"[WARNING] Could not plot feature importance: {e}")


def plot_confusion_matrices(cms, model_names, save_path):
    """Generate side-by-side seaborn heatmaps of confusion matrices."""
    fig, axes = plt.subplots(1, len(model_names), figsize=(18, 5))
    if len(model_names) == 1:
        axes = [axes]

    for ax, name in zip(axes, model_names):
        cm = cms[name]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=["Legit (0)", "Fraud (1)"],
            yticklabels=["Legit (0)", "Fraud (1)"],
            annot_kws={"size": 14, "weight": "bold"}
        )
        ax.set_title(f"{name}\nConfusion Matrix", fontsize=13, weight="bold", pad=10)
        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.set_ylabel("True Label", fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[INFO] Confusion matrices plot saved to: {save_path}")


def plot_metrics_comparison(metrics_df, save_path):
    """Generate grouped bar chart comparing Accuracy, Precision, Recall, and F1-score."""
    plot_df = metrics_df[["Accuracy", "Precision", "Recall", "F1-Score"]].reset_index()
    melted = pd.melt(plot_df, id_vars=["index"], var_name="Metric", value_name="Score")
    melted.rename(columns={"index": "Model"}, inplace=True)

    plt.figure(figsize=(12, 6))
    palette = ["#1E3A8A", "#2563EB", "#0284C7", "#0D9488"]
    ax = sns.barplot(data=melted, x="Model", y="Score", hue="Metric", palette=palette)

    plt.title("Model Performance Metrics Comparison", fontsize=15, weight="bold", pad=15)
    plt.xlabel("Machine Learning Algorithm", fontsize=12, labelpad=10)
    plt.ylabel("Score (0.0 to 1.0)", fontsize=12)
    plt.ylim(0, 1.15)
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.legend(loc="upper right", frameon=True)

    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.annotate(
                f"{height:.3f}",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center", va="bottom",
                fontsize=9,
                fontweight="semibold",
                xytext=(0, 3),
                textcoords="offset points"
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[INFO] Model metrics comparison chart saved to: {save_path}")


def plot_roc_curves(roc_data, save_path):
    """Generate superimposed ROC curves with AUC values."""
    plt.figure(figsize=(9, 7))
    colors = ["#1D4ED8", "#DC2626", "#059669"]

    for (name, data), color in zip(roc_data.items(), colors):
        plt.plot(
            data["fpr"],
            data["tpr"],
            color=color,
            lw=2.5,
            label=f"{name} (AUC = {data['auc']:.4f})"
        )

    plt.plot([0, 1], [0, 1], color="#64748B", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.50)")
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    plt.ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=12)
    plt.title("Receiver Operating Characteristic (ROC) Curves", fontsize=15, weight="bold", pad=15)
    plt.legend(loc="lower right", fontsize=11, frameon=True)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[INFO] ROC curves plot saved to: {save_path}")


def run_testing(test_path=DEFAULT_TEST_PATH):
    """Execute model evaluation on test set."""
    ensure_directories()
    missing_files = check_models_exist()

    if missing_files:
        print("\n" + "!" * 80)
        print("ERROR: TRAINED MODELS OR TEST DATASET NOT FOUND")
        print("!" * 80)
        print(f"Missing components ({len(missing_files)}):")
        for f in missing_files:
            print(f"  - {f}")
        print("\nREASON:")
        print("  Supervised models have not been trained yet because the uploaded dataset")
        print("  'Fraud Detection Dataset-selected-columns.csv' lacks the ground-truth")
        print("  target column ('Fraudulent').")
        print("\nHOW TO RESOLVE:")
        print("  1. Provide the complete labeled dataset containing 'Fraudulent' column.")
        print("  2. Run the training script:")
        print("         python train_model.py --data \"path/to/complete_dataset.csv\"")
        print("  3. Then rerun this evaluation script:")
        print("         python test_model.py")
        print("=" * 80 + "\n")
        return False

    print("\n" + "=" * 80)
    print("STEP 1: LOADING TEST DATASET (ZERO DATA LEAKAGE)")
    print("=" * 80)
    X_test, y_test, metadata = load_test_data(test_path)
    print(f"Test samples: {X_test.shape[0]:,} | Features: {X_test.shape[1]}")
    test_fraud_count = int(y_test.sum())
    test_legit_count = len(y_test) - test_fraud_count
    print(f"Test class distribution: Legit={test_legit_count:,}, Fraud={test_fraud_count:,}")

    # Load pipelines
    model_configs = {
        "Random Forest": os.path.join(MODELS_DIR, "random_forest_pipeline.joblib"),
        "Support Vector Machine": os.path.join(MODELS_DIR, "support_vector_machine_pipeline.joblib"),
        "Gaussian Naive Bayes": os.path.join(MODELS_DIR, "gaussian_naive_bayes_pipeline.joblib")
    }

    print("\n" + "=" * 80)
    print("STEP 2: GENERATING PREDICTIONS & EVALUATION METRICS")
    print("=" * 80)

    metrics_dict = {}
    cms = {}
    roc_data = {}
    reports = {}

    for name, path in model_configs.items():
        print(f"\n[EVALUATING] {name}...")
        pipeline = joblib.load(path)

        y_pred = pipeline.predict(X_test)
        
        # Probabilities
        if hasattr(pipeline, "predict_proba"):
            y_proba = pipeline.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc_val = float(auc(fpr, tpr))
            roc_data[name] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": auc_val}
        else:
            auc_val = 0.0

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        cm = confusion_matrix(y_test, y_pred)
        cms[name] = cm

        metrics_dict[name] = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "ROC-AUC": auc_val,
            "True Negatives": int(cm[0, 0]),
            "False Positives": int(cm[0, 1]),
            "False Negatives": int(cm[1, 0]),
            "True Positives": int(cm[1, 1])
        }

        report_str = classification_report(
            y_test, y_pred,
            target_names=["Legitimate (0)", "Fraudulent (1)"],
            digits=4,
            zero_division=0
        )
        reports[name] = report_str
        print(f"Classification Report for {name}:\n{report_str}")

    print("\n" + "=" * 80)
    print("STEP 3: COMPREHENSIVE ALGORITHM COMPARISON TABLE")
    print("=" * 80)
    metrics_df = pd.DataFrame(metrics_dict).T
    summary_cols = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    print(metrics_df[summary_cols].to_string())

    # Evaluation criterion justification
    print("\n" + "=" * 80)
    print("STEP 4: BEST MODEL SELECTION & EVALUATION CRITERION")
    print("=" * 80)
    best_model_by_f1 = metrics_df["F1-Score"].idxmax()
    best_f1 = metrics_dict[best_model_by_f1]["F1-Score"]
    best_rec = metrics_dict[best_model_by_f1]["Recall"]

    print(f"SELECTED BEST MODEL: '{best_model_by_f1}'")
    print(f"  F1-Score: {best_f1:.4f} | Recall: {best_rec:.4f} | Accuracy: {metrics_dict[best_model_by_f1]['Accuracy']:.4f}")
    print("\nEVALUATION CRITERION JUSTIFICATION:")
    print("  In financial fraud detection:")
    print("  - False Negatives (FN): Fraudulent transactions that pass undetected.")
    print("    Result: Immediate financial loss, regulatory fines, customer churn.")
    print("  - False Positives (FP): Legitimate transactions incorrectly flagged.")
    print("    Result: Temporary user inconvenience, operational review costs.")
    print("  Because financial institutions prioritize intercepting fraudulent transactions,")
    print("  Recall is essential. However, to prevent flagging all legitimate users, the")
    print("  F1-Score (harmonic mean of Precision and Recall) serves as the primary benchmark.")

    print("\n" + "=" * 80)
    print("STEP 5: GENERATING & SAVING VISUALIZATIONS")
    print("=" * 80)
    plot_confusion_matrices(cms, list(model_configs.keys()), os.path.join(OUTPUTS_DIR, "confusion_matrices.png"))
    plot_metrics_comparison(metrics_df, os.path.join(OUTPUTS_DIR, "model_metrics_comparison.png"))
    if roc_data:
        plot_roc_curves(roc_data, os.path.join(OUTPUTS_DIR, "roc_curves.png"))

    # Plot Random Forest Feature Importance
    rf_path = model_configs.get("Random Forest")
    if rf_path and os.path.exists(rf_path):
        rf_pipeline = joblib.load(rf_path)
        plot_feature_importance(rf_pipeline, os.path.join(OUTPUTS_DIR, "feature_importance.png"))

    # Save complete evaluation output to JSON
    output_json_path = os.path.join(OUTPUTS_DIR, "test_evaluation_results.json")
    with open(output_json_path, "w") as f:
        json.dump({
            "metrics": metrics_dict,
            "best_model": best_model_by_f1,
            "evaluation_criterion": "F1-Score (Harmonic Mean of Precision and Recall)"
        }, f, indent=4)
    print(f"[INFO] Detailed evaluation JSON saved to: {output_json_path}")

    print("\n" + "=" * 80)
    print("EVALUATION PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)
    return True


def main():
    parser = argparse.ArgumentParser(description="Test and Evaluate Online Fraud Detection Models")
    parser.add_argument(
        "--test-data",
        type=str,
        default=DEFAULT_TEST_PATH,
        help="Path to separated test CSV (default: models/test_data.csv)"
    )
    args = parser.parse_args()
    success = run_testing(args.test_data)
    if not success:
        sys.exit(2)


if __name__ == "__main__":
    main()

