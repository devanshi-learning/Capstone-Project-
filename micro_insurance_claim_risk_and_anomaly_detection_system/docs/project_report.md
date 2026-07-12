# Project Report: Micro Insurance Claim Risk and Anomaly Detection System

## Executive Summary

This report documents the design, implementation, and evaluation of a machine-learning system that flags suspicious micro-insurance claims for human review. The solution uses the Insurance Fraud Detection dataset (1,000 auto-insurance claims, 39 attributes) and produces actionable outputs: risk score, risk tier, reason codes, and review recommendation.

The final voting ensemble model with SMOTE achieved **85.7% recall**, **62.7% precision**, and **85.6% ROC-AUC** on the held-out test set using a validation-tuned decision threshold of 0.26.

---

## 1. Problem Understanding and Stakeholders

### Problem

Insurance companies cannot manually inspect every claim. Fraudulent or abnormal claims must be identified early without overwhelming investigators or wrongly rejecting genuine customers.

### Primary Users

| User | Goal |
|---|---|
| Claims investigator | Prioritize suspicious claims |
| Fraud operations manager | Monitor review queue and team workload |
| Data/ML analyst | Maintain model quality and thresholds |

### Success Criteria

1. End-to-end workflow from raw claim data to review recommendation
2. Meaningful AI/ML component with validation metrics
3. Explainable outputs in plain English
4. Working demo application

---

## 2. Data Preparation

### Source

- Kaggle dataset: Insurance Fraud Detection
- Bundled archive: `data/archive.zip` (Excel workbook inside)
- Converted to: `data/insurance_fraud_detection.csv` via `python src/prepare_data.py`

### Data Quality Issues

1. Missing categorical values encoded as `?` in `collision_type`, `property_damage`, `police_report_available`
2. Column typo: `policy_deductable` renamed to `policy_deductible`
3. Class imbalance: 24.7% fraud vs 75.3% non-fraud
4. Mixed numeric and categorical features

### Cleaning Steps

1. Replace `?` with missing values
2. Normalize column names
3. Parse `policy_bind_date` and `incident_date`
4. Drop ID/leaky columns (`policy_number`, raw dates)
5. Impute and encode features inside sklearn pipeline

### Feature Engineering

| Feature | Purpose |
|---|---|
| `policy_tenure_days` | Time between policy bind and incident |
| `incident_month`, `incident_day_of_week` | Seasonality / timing patterns |
| `capital_net` | Net capital gains/losses |
| `injury_to_total_ratio` | Unusual injury claim proportion |
| `vehicle_to_total_ratio` | Unusual vehicle claim proportion |
| `property_to_total_ratio` | Unusual property claim proportion |
| `claim_sum_gap_ratio` | Inconsistency between total claim and component sums |
| `incident_is_night` | Night-time incident indicator |
| `incident_is_weekend` | Weekend incident indicator |
| `customer_tenure_ratio` | Policy tenure relative to customer age |
| `premium_deductible_ratio` | Premium relative to deductible |
| `*_missing` flags | Missing-value indicators for key categorical fields |

---

## 3. Exploratory Data Analysis

EDA plots are stored in `outputs/eda/`.

### Key Observations

1. **Class imbalance** — fraud is the minority class; special handling is required.
2. **Claim amounts** — fraudulent claims often have higher `total_claim_amount` distributions.
3. **Missing reports** — many records lack police reports or have ambiguous `?` values.
4. **Correlations** — claim sub-components correlate with `total_claim_amount`; engineered ratios add signal.

---

## 4. Modeling Approach

### Baseline and Final Model

| Model | SMOTE | Tuned Recall | Precision | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Voting Ensemble (selected) | Yes | 0.857 | 0.627 | 0.724 | 0.856 |
| Gradient Boosting | Yes | 0.816 | 0.635 | 0.714 | 0.853 |
| Random Forest | No | 0.816 | 0.635 | 0.714 | 0.849 |
| Logistic Regression | No | 0.755 | 0.617 | 0.679 | 0.836 |

**Selected model:** Voting Ensemble (Random Forest + Gradient Boosting, soft voting)

### Class Imbalance Handling

- SMOTE applied only on training folds
- `class_weight='balanced_subsample'` in Random Forest
- Decision threshold tuned to achieve recall ≥ 0.60 while maximizing F1

### Validation Strategy

- Stratified 80/20 train/test split
- 5-fold stratified cross-validation on training set
- Final evaluation on untouched test set

---

## 5. Evaluation Results

### Test Metrics (Tuned Threshold = 0.26)

| Metric | Value |
|---|---|
| Accuracy | 0.840 |
| Precision | 0.627 |
| Recall | 0.857 |
| F1 | 0.724 |
| ROC-AUC | 0.856 |

### Confusion Matrix

```
                 Predicted
               No    Yes
Actual No     126    25
       Yes      7    42
```

### Interpretation

- The model catches most fraud cases (high recall).
- Precision is moderate, meaning investigators will see some false alarms.
- For fraud detection, this trade-off is acceptable because missed fraud is usually more costly than extra review.

---

## 6. Decision Support Layer

### Risk Tiers

| Tier | Probability | Action |
|---|---|---|
| HIGH | ≥ 0.70 | Escalate to fraud team |
| MEDIUM | 0.40 – 0.69 | Recommend manual review |
| LOW | < 0.40 | Auto-approve with monitoring |

### Reason Codes

| Code | Meaning |
|---|---|
| `HIGH_CLAIM_AMOUNT` | Total claim above 75th percentile |
| `NO_POLICE_REPORT` | No police report available |
| `PROPERTY_DAMAGE_MISMATCH` | Damage flag inconsistent with property claim |
| `LOW_WITNESS_COUNT` | Zero witnesses |
| `HIGH_INJURY_RATIO` | Injury claim unusually large vs total |
| `HIGH_VEHICLE_RATIO` | Vehicle claim unusually large vs total |
| `MODEL_HIGH_RISK` | Model probability ≥ 0.70 |

### Example Output

```json
{
  "risk_score": 0.7828,
  "risk_tier": "HIGH",
  "fraud_prediction": "SUSPICIOUS",
  "reason_codes": ["PROPERTY_DAMAGE_MISMATCH", "HIGH_VEHICLE_RATIO", "MODEL_HIGH_RISK"],
  "review_action": "ESCALATE_TO_FRAUD_TEAM",
  "explanation": "Claim flagged because: Property damage flag does not align with property claim amount.; Vehicle claim is unusually high relative to total claim amount.; Model fraud probability exceeds the high-risk threshold."
}
```

---

## 7. Application Prototype

A Streamlit app (`app/app.py`) provides:

1. **Single Claim Review** — form input with risk dashboard
2. **Batch Upload** — score CSV files and download ranked results
3. **About** — workflow, limitations, responsible use

---

## 8. Validation and Test Cases

| Test Case | Expected Behavior | Result |
|---|---|---|
| Train on full dataset | Artifacts saved | Pass |
| Recall on test set ≥ 0.60 | Fraud detection priority | Pass (0.816) |
| High-risk claim | Reason codes + escalate action | Pass |
| Batch scoring | Sorted output with tiers | Pass |
| Missing model file | Clear setup instructions in app | Pass |

---

## 9. Limitations and Responsible Use

### Limitations

- Small public dataset; not production-ready
- Schema-specific; new fields require retraining
- Rule codes are transparent but not a full legal audit trail
- Model may reflect historical bias in labeled fraud cases

### Responsible Use

- Use as decision support, not autonomous claim rejection
- Require human approval for escalation actions
- Document investigator overrides for future model improvement
- Review false positives to protect customer experience

---

## 10. Conclusion

The project delivers a complete AIML workflow for micro-insurance claim risk review: data preparation, EDA, imbalance-aware modeling, evaluation, explainable outputs, and a Streamlit demo. The system helps investigators focus on the most suspicious claims while keeping humans in control of final decisions.

---

## Appendix: Repository Structure

```text
micro_insurance_claim_risk_and_anomaly_detection_system/
├── data/insurance_fraud_detection.csv
├── notebooks/exploration_and_modeling.ipynb
├── src/
│   ├── main.py
│   ├── prepare_data.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── predict.py
│   ├── reason_codes.py
│   ├── run_eda.py
│   └── utils.py
├── app/app.py
├── models/
├── outputs/
├── docs/
├── requirements.txt
└── README.md
```
