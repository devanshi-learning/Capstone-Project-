"""Train fraud detection model and export artifacts for local inference."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from preprocessing import (
    build_metadata,
    build_model_pipeline,
    get_feature_columns,
    load_raw_data,
    split_data,
)
from utils import METADATA_PATH, METRICS_PATH, MODEL_PATH, save_json

MODEL_CONFIGS = [
    ("voting_ensemble", True),
    ("random_forest", True),
    ("gradient_boosting", True),
    ("random_forest", False),
    ("logistic", False),
]


def find_decision_threshold(
    y_true: pd.Series,
    y_prob: np.ndarray,
    min_recall: float = 0.75,
) -> tuple[float, dict]:
    best_threshold = 0.5
    best_score = -1.0
    best_metrics: dict = {}

    for threshold in np.arange(0.05, 0.96, 0.01):
        y_pred = (y_prob >= threshold).astype(int)
        recall = recall_score(y_true, y_pred, zero_division=0)
        if recall < min_recall:
            continue

        precision = precision_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        score = f1 + (0.05 * precision)

        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
            best_metrics = {
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }

    if not best_metrics:
        for threshold in np.arange(0.05, 0.96, 0.01):
            y_pred = (y_prob >= threshold).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_score:
                best_score = f1
                best_threshold = float(threshold)

    return best_threshold, best_metrics


def evaluate_model(
    pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    decision_threshold: float = 0.5,
) -> dict:
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= decision_threshold).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    return {
        "decision_threshold": float(decision_threshold),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "confusion_matrix": {
            "true_negative": int(cm[0, 0]),
            "false_positive": int(cm[0, 1]),
            "false_negative": int(cm[1, 0]),
            "true_positive": int(cm[1, 1]),
        },
    }


def fit_with_validation_threshold(
    pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    min_recall: float = 0.75,
) -> tuple[object, float]:
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train,
    )
    pipeline.fit(X_fit, y_fit)
    val_prob = pipeline.predict_proba(X_val)[:, 1]
    threshold, _ = find_decision_threshold(y_val, val_prob, min_recall=min_recall)

    pipeline.fit(X_train, y_train)
    return pipeline, threshold


def compare_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    numeric_features: list[str],
    categorical_features: list[str],
) -> list[dict]:
    rows = []
    for model_type, use_smote in MODEL_CONFIGS:
        pipeline = build_model_pipeline(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            model_type=model_type,
            use_smote=use_smote,
        )
        pipeline, threshold = fit_with_validation_threshold(pipeline, X_train, y_train)
        metrics = evaluate_model(pipeline, X_test, y_test, threshold)
        rows.append(
            {
                "model_type": model_type,
                "use_smote": use_smote,
                **metrics,
            }
        )
    rows.sort(key=lambda row: (row["f1"], row["precision"], row["recall"]), reverse=True)
    return rows


def select_best_model(model_comparison: list[dict]) -> dict:
    return model_comparison[0]


def cross_validate_model(pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    scores = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
    return {
        metric.replace("test_", ""): float(np.mean(values))
        for metric, values in scores.items()
        if metric.startswith("test_")
    }


def train(
    model_type: str | None = None,
    use_smote: bool | None = None,
    auto_select: bool = True,
) -> dict:
    df = load_raw_data()
    X_train, X_test, y_train, y_test = split_data(df)
    numeric_features, categorical_features = get_feature_columns(df)

    model_comparison = compare_models(
        X_train, X_test, y_train, y_test, numeric_features, categorical_features
    )
    best = select_best_model(model_comparison)

    if auto_select:
        model_type = best["model_type"]
        use_smote = best["use_smote"]
    else:
        model_type = model_type or "voting_ensemble"
        use_smote = True if use_smote is None else use_smote

    pipeline = build_model_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        model_type=model_type,
        use_smote=use_smote,
    )

    cv_metrics = cross_validate_model(pipeline, X_train, y_train)
    pipeline, decision_threshold = fit_with_validation_threshold(pipeline, X_train, y_train)

    test_metrics_default = evaluate_model(pipeline, X_test, y_test, decision_threshold=0.5)
    test_metrics = evaluate_model(pipeline, X_test, y_test, decision_threshold=decision_threshold)

    metadata = build_metadata(df)
    metadata["model_type"] = model_type
    metadata["use_smote"] = use_smote
    metadata["decision_threshold"] = decision_threshold
    metadata["auto_selected"] = auto_select
    metadata["trained_at"] = datetime.now(timezone.utc).isoformat()

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    save_json(METADATA_PATH, metadata)

    metrics_payload = {
        "model_type": model_type,
        "use_smote": use_smote,
        "auto_selected": auto_select,
        "trained_at": metadata["trained_at"],
        "decision_threshold": decision_threshold,
        "cross_validation": cv_metrics,
        "test_metrics_default_threshold": test_metrics_default,
        "test_metrics": test_metrics,
        "model_comparison": model_comparison,
        "selected_model": best,
        "dataset_summary": {
            "rows": int(len(df)),
            "fraud_cases": int((df["fraud_reported"] == "Y").sum()),
            "non_fraud_cases": int((df["fraud_reported"] == "N").sum()),
        },
    }
    save_json(METRICS_PATH, metrics_payload)

    print("Training complete.")
    print(f"Selected model: {model_type} (SMOTE={use_smote})")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metadata saved to: {METADATA_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")
    print(json.dumps(test_metrics, indent=2))
    return metrics_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Train insurance fraud detection model.")
    parser.add_argument(
        "--model",
        choices=["random_forest", "logistic", "gradient_boosting", "voting_ensemble"],
        default=None,
        help="Classifier to train. Defaults to best model from comparison.",
    )
    parser.add_argument(
        "--no-smote",
        action="store_true",
        help="Disable SMOTE oversampling when a specific model is chosen.",
    )
    parser.add_argument(
        "--no-auto-select",
        action="store_true",
        help="Disable automatic best-model selection.",
    )
    args = parser.parse_args()

    try:
        train(
            model_type=args.model,
            use_smote=False if args.no_smote else None,
            auto_select=not args.no_auto_select and args.model is None,
        )
    except ValueError as exc:
        if "SMOTE" in str(exc):
            print("SMOTE failed; retrying with class_weight only.")
            train(
                model_type=args.model or "random_forest",
                use_smote=False,
                auto_select=not args.no_auto_select and args.model is None,
            )
        else:
            raise


if __name__ == "__main__":
    main()
