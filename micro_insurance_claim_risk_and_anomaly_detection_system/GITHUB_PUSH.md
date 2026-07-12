# Push to GitHub

This folder is already a git repo on branch `main` with an initial commit.

## Steps

1. Create a **new empty repository** on GitHub (no README, no .gitignore).

2. Open a terminal in this folder:

```bash
cd micro_insurance_claim_risk_and_anomaly_detection_system
```

3. Add your remote and push:

```bash
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

4. Before submission, edit `README.md` and add your **team member names**.

## Verify locally (optional)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/prepare_data.py
streamlit run app/app.py
```
