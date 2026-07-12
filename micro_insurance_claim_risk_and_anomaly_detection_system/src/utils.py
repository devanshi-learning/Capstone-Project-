"""Shared paths, constants, and review-routing helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

DATA_PATH = DATA_DIR / "insurance_fraud_detection.csv"
MODEL_PATH = MODELS_DIR / "fraud_model.joblib"
METADATA_PATH = MODELS_DIR / "feature_metadata.json"
METRICS_PATH = OUTPUTS_DIR / "metrics.json"

TARGET_COLUMN = "fraud_reported"
RISK_THRESHOLDS = {
    "high": 0.70,
    "medium": 0.40,
}

DROP_COLUMNS = [
    "policy_number",
    "policy_bind_date",
    "incident_date",
    "_c39",
]

CATEGORICAL_COLUMNS = [
    "policy_state",
    "policy_csl",
    "insured_sex",
    "insured_education_level",
    "insured_occupation",
    "insured_hobbies",
    "insured_relationship",
    "incident_type",
    "collision_type",
    "incident_severity",
    "authorities_contacted",
    "incident_state",
    "incident_city",
    "incident_location",
    "property_damage",
    "police_report_available",
    "auto_make",
    "auto_model",
]

NUMERIC_COLUMNS = [
    "months_as_customer",
    "age",
    "policy_deductible",
    "policy_annual_premium",
    "umbrella_limit",
    "insured_zip",
    "capital-gains",
    "capital-loss",
    "incident_hour_of_the_day",
    "number_of_vehicles_involved",
    "bodily_injuries",
    "witnesses",
    "total_claim_amount",
    "injury_claim",
    "property_claim",
    "vehicle_claim",
    "auto_year",
    "policy_tenure_days",
    "incident_month",
    "incident_day_of_week",
    "capital_net",
    "injury_to_total_ratio",
    "vehicle_to_total_ratio",
    "property_to_total_ratio",
    "claim_parts_sum",
    "claim_sum_gap",
    "claim_sum_gap_ratio",
    "incident_is_night",
    "incident_is_weekend",
    "customer_tenure_ratio",
    "premium_deductible_ratio",
    "police_report_available_missing",
    "property_damage_missing",
    "collision_type_missing",
]


def risk_tier(score: float) -> str:
    if score >= RISK_THRESHOLDS["high"]:
        return "HIGH"
    if score >= RISK_THRESHOLDS["medium"]:
        return "MEDIUM"
    return "LOW"


def recommend_review(risk_score: float, reason_codes: list[str]) -> str:
    tier = risk_tier(risk_score)
    if tier == "HIGH" or "MODEL_HIGH_RISK" in reason_codes:
        return "ESCALATE_TO_FRAUD_TEAM"
    if tier == "MEDIUM" or len(reason_codes) >= 2:
        return "REVIEW_RECOMMENDED"
    return "AUTO_APPROVE"


def fraud_label(score: float) -> str:
    return "SUSPICIOUS" if score >= RISK_THRESHOLDS["medium"] else "LIKELY_GENUINE"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
