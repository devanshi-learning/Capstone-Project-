"""Generate EDA plots from the prepared insurance fraud dataset."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import RocCurveDisplay, confusion_matrix

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from preprocessing import build_model_pipeline, get_feature_columns, load_raw_data, split_data
from train import evaluate_model, find_decision_threshold, fit_with_validation_threshold
from utils import OUTPUTS_DIR, TARGET_COLUMN

EDA_DIR = OUTPUTS_DIR / "eda"


def main() -> None:
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    df = load_raw_data()
    target_counts = df[TARGET_COLUMN].value_counts()

    fig, ax = plt.subplots(figsize=(5, 4))
    target_counts.plot(kind="bar", ax=ax, color=["#2ca02c", "#d62728"])
    ax.set_title("Fraud Reported Class Distribution")
    ax.set_xlabel("fraud_reported")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "target_distribution.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x=TARGET_COLUMN, y="total_claim_amount", ax=ax)
    ax.set_title("Total Claim Amount by Fraud Label")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "claim_amount_by_fraud.png", dpi=150)
    plt.close()

    numeric_df = df.copy()
    for col in numeric_df.columns:
        if numeric_df[col].dtype == "object" and col != TARGET_COLUMN:
            continue
        numeric_df[col] = pd.to_numeric(numeric_df[col], errors="coerce")
    corr = numeric_df.select_dtypes(include="number").corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Numeric Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "correlation_heatmap.png", dpi=150)
    plt.close()

    X_train, X_test, y_train, y_test = split_data(df)
    numeric_features, categorical_features = get_feature_columns(df)
    pipeline = build_model_pipeline(numeric_features, categorical_features, "voting_ensemble", True)
    pipeline, threshold = fit_with_validation_threshold(pipeline, X_train, y_train)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    fig, ax = plt.subplots(figsize=(5, 4))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Random Forest Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_test, y_prob, ax=ax)
    ax.set_title("Random Forest ROC Curve")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "roc_curve.png", dpi=150)
    plt.close()

    print(f"EDA plots saved to {EDA_DIR}")


if __name__ == "__main__":
    main()
