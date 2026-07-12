# Presentation: Micro Insurance Claim Risk and Anomaly Detection System

> Export to PDF: open this file in any Markdown-to-PDF tool, or copy each slide into PowerPoint/Google Slides.

---

## Slide 1 — Title and Team

**Micro Insurance Claim Risk and Anomaly Detection System**

- Track: AIML
- Domain: Insurance operations and fraud-risk review
- Team: _[Add names]_

---

## Slide 2 — Problem and Real-World Impact

**Problem**

Insurance teams cannot manually inspect every claim in detail.

**Impact**

- Reduces operational risk
- Helps investigators focus on suspicious claims
- Supports responsible, human-in-the-loop review

**Users:** claims investigators, fraud operations managers, ML analysts

---

## Slide 3 — Dataset and Source Material

**Dataset:** Insurance Fraud Detection (Kaggle)

- 1,000 claims
- 39 features
- Target: `fraud_reported` (Y/N)

**Distribution**

- Non-fraud: 753
- Fraud: 247

**Fields:** policy, customer, incident, and claim amount data

---

## Slide 4 — System Workflow

```text
Data → Clean/Engineer → EDA → Train Model → Evaluate → Score Claim → Recommend Review → Streamlit Demo
```

**Core modules**

- Claim EDA
- Class imbalance handling
- Fraud model
- Reason codes
- Human review recommendation

---

## Slide 5 — AI/ML Innovation

**Model:** Voting Ensemble (Random Forest + Gradient Boosting) + SMOTE

**Outputs**

- Fraud probability (risk score)
- Risk tier: HIGH / MEDIUM / LOW
- Reason codes
- Review action

**Why ML?** Fraud patterns involve many interacting variables that are difficult to capture with fixed rules alone.

---

## Slide 6 — Prototype Demo

**Streamlit app features**

- Single-claim scoring form
- Batch CSV upload
- Risk dashboard with explanation
- Sidebar with model metrics

Screenshot paths:

- `outputs/eda/target_distribution.png`
- `outputs/eda/confusion_matrix.png`
- Streamlit UI screenshots (capture during demo)

---

## Slide 7 — Results

**Tuned test performance**

| Metric | Value |
|---|---|
| Recall | 85.7% |
| Precision | 62.7% |
| F1 | 72.4% |
| ROC-AUC | 85.6% |

**Sample output:** HIGH risk claim → `ESCALATE_TO_FRAUD_TEAM` with reason codes

---

## Slide 8 — Limitations and Responsible Use

**Limitations**

- Small dataset
- Not production-certified
- Schema-dependent
- Moderate precision → some false positives

**Responsible use**

- Assists reviewers; does not auto-deny claims
- Human investigators make final decisions
- Document overrides for audit and improvement

---

## Slide 9 — Future Improvements

- SHAP-based explanations
- API integration with claims systems
- Active learning from investigator feedback
- Model monitoring and drift detection
- Larger and insurer-specific datasets

---

## Slide 10 — Conclusion

We built an end-to-end claim risk detection system that:

1. Processes real insurance claim data
2. Uses ML to score fraud/anomaly risk
3. Explains flags with reason codes
4. Recommends review actions through a working demo

**Thank you**
