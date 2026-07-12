"""Rule-based and model-based reason code generation."""

from __future__ import annotations

from typing import Any

import pandas as pd

from utils import RISK_THRESHOLDS, fraud_label, recommend_review, risk_tier

REASON_DESCRIPTIONS = {
    "HIGH_CLAIM_AMOUNT": "Total claim amount is above the training 75th percentile.",
    "NO_POLICE_REPORT": "No police report is available for this incident.",
    "PROPERTY_DAMAGE_MISMATCH": "Property damage flag does not align with property claim amount.",
    "LOW_WITNESS_COUNT": "No witnesses were recorded for the incident.",
    "MODEL_HIGH_RISK": "Model fraud probability exceeds the high-risk threshold.",
    "HIGH_INJURY_RATIO": "Injury claim is unusually high relative to total claim amount.",
    "HIGH_VEHICLE_RATIO": "Vehicle claim is unusually high relative to total claim amount.",
    "HIGH_PROPERTY_RATIO": "Property claim is unusually high relative to total claim amount.",
    "CLAIM_SUM_MISMATCH": "Injury, property, and vehicle claims do not align with total claim amount.",
    "NIGHT_INCIDENT": "Incident occurred during night hours.",
    "WEEKEND_INCIDENT": "Incident occurred on a weekend.",
}


def generate_reason_codes(claim: dict[str, Any], risk_score: float, metadata: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    thresholds = metadata.get("thresholds", {})

    total_claim = float(claim.get("total_claim_amount", 0) or 0)
    high_claim_threshold = float(thresholds.get("high_claim_amount", 0))
    if total_claim > high_claim_threshold:
        reasons.append("HIGH_CLAIM_AMOUNT")

    police_report = str(claim.get("police_report_available", "")).upper()
    if police_report in {"NO", "N", "FALSE", "0"}:
        reasons.append("NO_POLICE_REPORT")

    property_damage = str(claim.get("property_damage", "")).upper()
    property_claim = float(claim.get("property_claim", 0) or 0)
    if property_damage in {"NO", "N", "FALSE", "0"} and property_claim > 0:
        reasons.append("PROPERTY_DAMAGE_MISMATCH")
    if property_damage in {"YES", "Y", "TRUE", "1"} and property_claim == 0:
        reasons.append("PROPERTY_DAMAGE_MISMATCH")

    witnesses = claim.get("witnesses", 0)
    if pd.isna(witnesses) or int(witnesses) == 0:
        reasons.append("LOW_WITNESS_COUNT")

    injury_ratio = float(claim.get("injury_to_total_ratio", 0) or 0)
    vehicle_ratio = float(claim.get("vehicle_to_total_ratio", 0) or 0)
    property_ratio = float(claim.get("property_to_total_ratio", 0) or 0)
    claim_sum_gap_ratio = float(claim.get("claim_sum_gap_ratio", 0) or 0)

    if injury_ratio >= 0.6:
        reasons.append("HIGH_INJURY_RATIO")
    if vehicle_ratio >= 0.7:
        reasons.append("HIGH_VEHICLE_RATIO")
    if property_ratio >= 0.5:
        reasons.append("HIGH_PROPERTY_RATIO")
    if claim_sum_gap_ratio >= 0.15:
        reasons.append("CLAIM_SUM_MISMATCH")

    if int(claim.get("incident_is_night", 0) or 0) == 1:
        reasons.append("NIGHT_INCIDENT")
    if int(claim.get("incident_is_weekend", 0) or 0) == 1:
        reasons.append("WEEKEND_INCIDENT")

    if risk_score >= thresholds.get("high_risk_score", RISK_THRESHOLDS["high"]):
        reasons.append("MODEL_HIGH_RISK")

    return reasons


def build_explanation(reason_codes: list[str]) -> str:
    if not reason_codes:
        return "No major risk signals detected. Claim can proceed with routine monitoring."

    readable = [REASON_DESCRIPTIONS.get(code, code.replace("_", " ").lower()) for code in reason_codes]
    return "Claim flagged because: " + "; ".join(readable)


def build_result(claim: dict[str, Any], risk_score: float, metadata: dict[str, Any]) -> dict[str, Any]:
    reason_codes = generate_reason_codes(claim, risk_score, metadata)
    review_action = recommend_review(risk_score, reason_codes)

    return {
        "risk_score": round(float(risk_score), 4),
        "risk_tier": risk_tier(risk_score),
        "fraud_prediction": fraud_label(risk_score),
        "reason_codes": reason_codes,
        "review_action": review_action,
        "explanation": build_explanation(reason_codes),
    }
