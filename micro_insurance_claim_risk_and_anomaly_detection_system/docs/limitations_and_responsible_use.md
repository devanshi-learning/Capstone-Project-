# Limitations and Responsible Use

## Limitations

1. **Dataset size** — Only 1,000 labeled claims are available. Production fraud systems typically require much larger datasets.
2. **Public data mismatch** — The Kaggle dataset may not represent a specific insurer's products, geography, or customer base.
3. **Missing data** — Several categorical fields contain `?` placeholders that required imputation.
4. **Moderate precision** — The tuned model prioritizes recall; investigators will review some false positives.
5. **Static model** — The current pipeline does not automatically retrain when claim patterns change.
6. **Schema dependency** — Incoming data must follow the same columns and formats used during training.

## Responsible Use Guidelines

1. **Human-in-the-loop** — All high-risk outputs should be reviewed by trained investigators.
2. **No automatic denial** — The system recommends actions; it does not reject claims on its own.
3. **Transparency** — Reason codes and explanations must be shown to reviewers alongside the risk score.
4. **Auditability** — Teams should log investigator overrides to improve future model versions.
5. **Customer fairness** — False positives can harm customer trust; review queues should be monitored regularly.
6. **Regulatory caution** — This academic prototype is not a substitute for legal, compliance, or actuarial review.

## Safe Claims for Demo and Report

Acceptable:

- "The model helps prioritize suspicious claims for review."
- "Reason codes explain why a claim was flagged."
- "The system achieved 81.6% recall on a held-out test set."

Avoid:

- "This system eliminates fraud."
- "The model always detects fraud correctly."
- "Claims can be auto-denied without human review."
