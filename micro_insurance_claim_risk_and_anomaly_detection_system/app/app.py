"""Streamlit demo for insurance claim risk and anomaly detection."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from predict import ModelNotFoundError, load_metadata, score_batch, score_claim  # noqa: E402
from utils import METRICS_PATH, MODEL_PATH, load_json  # noqa: E402

st.set_page_config(
    page_title="Insurance Claim Risk Detection",
    page_icon="🛡️",
    layout="wide",
)

TIER_COLORS = {
    "HIGH": "#d62728",
    "MEDIUM": "#ff7f0e",
    "LOW": "#2ca02c",
}

ACTION_LABELS = {
    "ESCALATE_TO_FRAUD_TEAM": "Escalate to fraud review team",
    "REVIEW_RECOMMENDED": "Recommend manual review",
    "AUTO_APPROVE": "Auto-approve with routine monitoring",
}


def render_missing_model_help() -> None:
    st.error("Trained model artifacts were not found.")
    st.markdown(
        """
        **Train on the lab computer first:**

        1. Copy this project to the lab machine
        2. `pip install -r requirements-train.txt`
        3. Place Kaggle CSV at `data/insurance_fraud_detection.csv`
        4. `python src/train.py`
        5. Copy `models/` and `outputs/metrics.json` back to this machine
        6. `pip install -r requirements-app.txt`
        7. `streamlit run app/app.py`
        """
    )


def render_sidebar() -> None:
    st.sidebar.title("Model Status")
    if MODEL_PATH.exists():
        st.sidebar.success("Model artifact loaded path is available.")
    else:
        st.sidebar.warning("Model artifact missing.")

    if METRICS_PATH.exists():
        metrics = load_json(METRICS_PATH)
        test_metrics = metrics.get("test_metrics", {})
        st.sidebar.markdown("**Test metrics (tuned threshold)**")
        st.sidebar.write(f"Decision threshold: {test_metrics.get('decision_threshold', 'n/a')}")
        st.sidebar.write(f"Recall: {test_metrics.get('recall', 'n/a'):.3f}" if isinstance(test_metrics.get('recall'), (int,float)) else "Recall: n/a")
        st.sidebar.write(f"ROC-AUC: {test_metrics.get('roc_auc', 'n/a'):.3f}" if isinstance(test_metrics.get('roc_auc'), (int,float)) else "ROC-AUC: n/a")
        st.sidebar.write(f"Precision: {test_metrics.get('precision', 'n/a'):.3f}" if isinstance(test_metrics.get('precision'), (int,float)) else "Precision: n/a")
        st.sidebar.write(f"F1: {test_metrics.get('f1', 'n/a'):.3f}" if isinstance(test_metrics.get('f1'), (int,float)) else "F1: n/a")
        dataset = metrics.get("dataset_summary", {})
        if dataset:
            st.sidebar.markdown("**Dataset**")
            st.sidebar.write(f"Claims: {dataset.get('rows', 'n/a')}")
            st.sidebar.write(f"Fraud cases: {dataset.get('fraud_cases', 'n/a')}")
    else:
        st.sidebar.info("Metrics file not found. Train the model on the lab computer.")

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Responsible use:** This tool supports human reviewers. "
        "It does not automatically deny claims or replace investigator judgment."
    )


def render_result(result: dict) -> None:
    tier = result["risk_tier"]
    color = TIER_COLORS.get(tier, "#333333")

    col1, col2, col3 = st.columns(3)
    col1.metric("Risk Score", f"{result['risk_score']:.2%}")
    col2.markdown(
        f"<div style='padding:8px;border-radius:8px;background:{color};color:white;"
        f"text-align:center;font-weight:600;'>{tier} RISK</div>",
        unsafe_allow_html=True,
    )
    col3.metric("Prediction", result["fraud_prediction"])

    st.subheader("Recommended Action")
    st.info(ACTION_LABELS.get(result["review_action"], result["review_action"]))

    st.subheader("Reason Codes")
    if result["reason_codes"]:
        st.write(", ".join(result["reason_codes"]))
    else:
        st.write("No major risk signals detected.")

    st.subheader("Explanation")
    st.write(result["explanation"])


def render_single_claim_tab(metadata: dict) -> None:
    st.subheader("Single Claim Review")
    samples = metadata.get("sample_records", {})
    preset = st.selectbox(
        "Load sample claim",
        options=["Custom", "High risk sample", "Low risk sample"],
    )

    default_claim = samples.get("high_risk", {})
    if preset == "Low risk sample":
        default_claim = samples.get("low_risk", default_claim)
    elif preset == "Custom":
        default_claim = samples.get("high_risk", {})

    with st.form("claim_form"):
        st.markdown("#### Customer and Policy")
        c1, c2, c3 = st.columns(3)
        months_as_customer = c1.number_input("Months as customer", value=int(default_claim.get("months_as_customer", 100)))
        age = c2.number_input("Age", value=int(default_claim.get("age", 40)))
        policy_deductible = c3.number_input("Policy deductible", value=float(default_claim.get("policy_deductible", 1000)))

        c4, c5, c6 = st.columns(3)
        policy_annual_premium = c4.number_input("Annual premium", value=float(default_claim.get("policy_annual_premium", 1200)))
        insured_zip = c5.number_input("Insured ZIP", value=int(default_claim.get("insured_zip", 10001)))
        auto_year = c6.number_input("Auto year", value=int(default_claim.get("auto_year", 2015)))

        st.markdown("#### Incident")
        d1, d2, d3 = st.columns(3)
        incident_type = d1.selectbox(
            "Incident type",
            ["Single Vehicle Collision", "Multi-vehicle Collision", "Parked Car", "Vehicle Theft"],
            index=0,
        )
        incident_severity = d2.selectbox(
            "Incident severity",
            ["Trivial Damage", "Minor Damage", "Major Damage", "Total Loss"],
            index=1,
        )
        collision_type = d3.selectbox(
            "Collision type",
            ["Front Collision", "Rear Collision", "Side Collision", "N/A"],
            index=0,
        )

        d4, d5, d6 = st.columns(3)
        incident_hour = d4.number_input("Incident hour", min_value=0, max_value=23, value=int(default_claim.get("incident_hour_of_the_day", 12)))
        vehicles_involved = d5.number_input("Vehicles involved", min_value=1, value=int(default_claim.get("number_of_vehicles_involved", 1)))
        witnesses = d6.number_input("Witnesses", min_value=0, value=int(default_claim.get("witnesses", 0)))

        property_damage = st.selectbox("Property damage", ["YES", "NO"], index=0)
        police_report_available = st.selectbox("Police report available", ["YES", "NO"], index=1)

        st.markdown("#### Claim Amounts")
        e1, e2, e3, e4 = st.columns(4)
        total_claim_amount = e1.number_input("Total claim", value=float(default_claim.get("total_claim_amount", 50000)))
        injury_claim = e2.number_input("Injury claim", value=float(default_claim.get("injury_claim", 5000)))
        property_claim = e3.number_input("Property claim", value=float(default_claim.get("property_claim", 10000)))
        vehicle_claim = e4.number_input("Vehicle claim", value=float(default_claim.get("vehicle_claim", 35000)))

        submitted = st.form_submit_button("Score claim")

    if submitted:
        claim = {
            "months_as_customer": months_as_customer,
            "age": age,
            "policy_state": default_claim.get("policy_state", "NY"),
            "policy_csl": default_claim.get("policy_csl", "250/500"),
            "policy_deductible": policy_deductible,
            "policy_annual_premium": policy_annual_premium,
            "umbrella_limit": default_claim.get("umbrella_limit", 0),
            "insured_zip": insured_zip,
            "insured_sex": default_claim.get("insured_sex", "MALE"),
            "insured_education_level": default_claim.get("insured_education_level", "High School"),
            "insured_occupation": default_claim.get("insured_occupation", "craft-repair"),
            "insured_hobbies": default_claim.get("insured_hobbies", "reading"),
            "insured_relationship": default_claim.get("insured_relationship", "own-child"),
            "capital-gains": default_claim.get("capital-gains", 0),
            "capital-loss": default_claim.get("capital-loss", 0),
            "incident_type": incident_type,
            "collision_type": collision_type,
            "incident_severity": incident_severity,
            "authorities_contacted": default_claim.get("authorities_contacted", "Police"),
            "incident_state": default_claim.get("incident_state", "NY"),
            "incident_city": default_claim.get("incident_city", "Springfield"),
            "incident_location": default_claim.get("incident_location", "Highway"),
            "incident_hour_of_the_day": incident_hour,
            "number_of_vehicles_involved": vehicles_involved,
            "property_damage": property_damage,
            "bodily_injuries": default_claim.get("bodily_injuries", 1),
            "witnesses": witnesses,
            "police_report_available": police_report_available,
            "total_claim_amount": total_claim_amount,
            "injury_claim": injury_claim,
            "property_claim": property_claim,
            "vehicle_claim": vehicle_claim,
            "auto_make": default_claim.get("auto_make", "Honda"),
            "auto_model": default_claim.get("auto_model", "Civic"),
            "auto_year": auto_year,
        }
        result = score_claim(claim)
        render_result(result)


def render_batch_tab() -> None:
    st.subheader("Batch Claim Review")
    uploaded = st.file_uploader("Upload claims CSV", type=["csv"])
    if uploaded is None:
        st.info("Upload a CSV with the same columns as the Kaggle insurance fraud dataset.")
        return

    raw_df = pd.read_csv(uploaded)
    with st.spinner("Scoring claims..."):
        results = score_batch(raw_df)

    high = int((results["risk_tier"] == "HIGH").sum())
    medium = int((results["risk_tier"] == "MEDIUM").sum())
    low = int((results["risk_tier"] == "LOW").sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("High risk", high)
    c2.metric("Medium risk", medium)
    c3.metric("Low risk", low)

    st.dataframe(results, use_container_width=True)
    st.download_button(
        "Download scored results",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name="scored_claims.csv",
        mime="text/csv",
    )


def render_about_tab() -> None:
    st.subheader("About this system")
    st.markdown(
        """
        This prototype helps insurance operations teams prioritize suspicious micro-insurance
        claims for human review.

        **Workflow**
        1. Claim data enters the scoring pipeline
        2. A trained ML model estimates fraud/anomaly risk
        3. Rule-based reason codes explain why a claim was flagged
        4. The app recommends review, escalation, or routine approval

        **AI/ML component**
        - Random Forest classifier with class-imbalance handling (SMOTE or balanced weights)
        - Risk tiers and reason codes for decision support

        **Limitations**
        - Trained on a small public dataset; not production-ready
        - Should assist reviewers, not replace them
        - Performance depends on training data quality and class balance
        """
    )


def main() -> None:
    st.title("Micro Insurance Claim Risk and Anomaly Detection")
    st.caption("Decision-support tool for fraud and anomaly review")

    render_sidebar()

    try:
        metadata = load_metadata()
    except ModelNotFoundError:
        render_missing_model_help()
        return

    tab_single, tab_batch, tab_about = st.tabs(["Single Claim", "Batch Upload", "About"])
    with tab_single:
        render_single_claim_tab(metadata)
    with tab_batch:
        render_batch_tab()
    with tab_about:
        render_about_tab()


if __name__ == "__main__":
    main()
