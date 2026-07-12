"""Load trained model and score single claims or batches."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from preprocessing import engineer_features
from reason_codes import build_result
from utils import METADATA_PATH, METRICS_PATH, MODEL_PATH, load_json


class ModelNotFoundError(FileNotFoundError):
    pass


def load_model():
    if not MODEL_PATH.exists():
        raise ModelNotFoundError(
            f"Model not found at {MODEL_PATH}. Train on the lab computer first: python src/train.py"
        )
    return joblib.load(MODEL_PATH)


def load_metadata() -> dict[str, Any]:
    if not METADATA_PATH.exists():
        raise ModelNotFoundError(
            f"Metadata not found at {METADATA_PATH}. Train on the lab computer first: python src/train.py"
        )
    return load_json(METADATA_PATH)


def _prepare_claim_frame(claim: dict[str, Any], metadata: dict[str, Any]) -> pd.DataFrame:
    frame = pd.DataFrame([claim])
    engineered = engineer_features(frame)
    feature_columns = metadata.get("feature_columns", engineered.columns.tolist())
    for column in feature_columns:
        if column not in engineered.columns:
            engineered[column] = pd.NA
    return engineered[feature_columns]


def score_claim(claim: dict[str, Any], model=None, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    model = model or load_model()
    metadata = metadata or load_metadata()
    features = _prepare_claim_frame(claim, metadata)
    risk_score = float(model.predict_proba(features)[0, 1])
    enriched_claim = engineer_features(pd.DataFrame([claim])).iloc[0].to_dict()
    return build_result(enriched_claim, risk_score, metadata)


def score_batch(df: pd.DataFrame, model=None, metadata: dict[str, Any] | None = None) -> pd.DataFrame:
    model = model or load_model()
    metadata = metadata or load_metadata()
    engineered = engineer_features(df)
    feature_columns = metadata.get("feature_columns", engineered.columns.tolist())
    for column in feature_columns:
        if column not in engineered.columns:
            engineered[column] = pd.NA

    probabilities = model.predict_proba(engineered[feature_columns])[:, 1]
    results = []
    for index, probability in enumerate(probabilities):
        claim = engineered.iloc[index].to_dict()
        results.append(build_result(claim, probability, metadata))

    output = pd.DataFrame(results)
    output = output.sort_values("risk_score", ascending=False).reset_index(drop=True)
    return output


def run_sample() -> None:
    metadata = load_metadata()
    sample_key = "high_risk" if "high_risk" in metadata.get("sample_records", {}) else "low_risk"
    claim = metadata["sample_records"][sample_key]
    result = score_claim(claim)
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Score insurance claims with the trained model.")
    parser.add_argument("--sample", action="store_true", help="Score a sample claim from metadata.")
    args = parser.parse_args()

    if args.sample:
        run_sample()
        return

    parser.print_help()


if __name__ == "__main__":
    main()
