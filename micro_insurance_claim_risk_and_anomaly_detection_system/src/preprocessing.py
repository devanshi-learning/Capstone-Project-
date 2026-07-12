"""Data loading, feature engineering, and sklearn preprocessing pipeline."""

from __future__ import annotations

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from utils import (
    CATEGORICAL_COLUMNS,
    DATA_PATH,
    DROP_COLUMNS,
    NUMERIC_COLUMNS,
    TARGET_COLUMN,
)


COLUMN_ALIASES = {
    "policy_deductable": "policy_deductible",
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data.rename(columns=COLUMN_ALIASES, inplace=True)
    return data


def load_raw_data(path=DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = normalize_columns(df)
    df.replace("?", pd.NA, inplace=True)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    if "policy_bind_date" in data.columns and "incident_date" in data.columns:
        bind_dates = pd.to_datetime(data["policy_bind_date"], errors="coerce")
        incident_dates = pd.to_datetime(data["incident_date"], errors="coerce")
        data["policy_tenure_days"] = (incident_dates - bind_dates).dt.days
        data["incident_month"] = incident_dates.dt.month
        data["incident_day_of_week"] = incident_dates.dt.dayofweek
    else:
        for column, default in {
            "policy_tenure_days": 0,
            "incident_month": 0,
            "incident_day_of_week": 0,
        }.items():
            if column not in data.columns:
                data[column] = default

    gains = pd.to_numeric(data.get("capital-gains", 0), errors="coerce").fillna(0)
    losses = pd.to_numeric(data.get("capital-loss", 0), errors="coerce").fillna(0)
    data["capital_net"] = gains - losses

    total_claim = pd.to_numeric(data.get("total_claim_amount", 0), errors="coerce").fillna(0)
    injury_claim = pd.to_numeric(data.get("injury_claim", 0), errors="coerce").fillna(0)
    property_claim = pd.to_numeric(data.get("property_claim", 0), errors="coerce").fillna(0)
    vehicle_claim = pd.to_numeric(data.get("vehicle_claim", 0), errors="coerce").fillna(0)

    data["injury_to_total_ratio"] = injury_claim / total_claim.replace(0, pd.NA)
    data["vehicle_to_total_ratio"] = vehicle_claim / total_claim.replace(0, pd.NA)
    data["property_to_total_ratio"] = property_claim / total_claim.replace(0, pd.NA)
    data["injury_to_total_ratio"] = data["injury_to_total_ratio"].fillna(0)
    data["vehicle_to_total_ratio"] = data["vehicle_to_total_ratio"].fillna(0)
    data["property_to_total_ratio"] = data["property_to_total_ratio"].fillna(0)

    data["claim_parts_sum"] = injury_claim + property_claim + vehicle_claim
    data["claim_sum_gap"] = (total_claim - data["claim_parts_sum"]).abs()
    data["claim_sum_gap_ratio"] = data["claim_sum_gap"] / total_claim.replace(0, pd.NA)
    data["claim_sum_gap_ratio"] = data["claim_sum_gap_ratio"].fillna(0)

    incident_hour = pd.to_numeric(data.get("incident_hour_of_the_day", 0), errors="coerce").fillna(0)
    data["incident_is_night"] = ((incident_hour < 6) | (incident_hour >= 20)).astype(int)

    incident_dow = pd.to_numeric(data.get("incident_day_of_week", 0), errors="coerce").fillna(0)
    data["incident_is_weekend"] = incident_dow.isin([5, 6]).astype(int)

    age = pd.to_numeric(data.get("age", 1), errors="coerce").fillna(1).replace(0, 1)
    months = pd.to_numeric(data.get("months_as_customer", 0), errors="coerce").fillna(0)
    data["customer_tenure_ratio"] = months / age

    premium = pd.to_numeric(data.get("policy_annual_premium", 0), errors="coerce").fillna(0)
    deductible = pd.to_numeric(data.get("policy_deductible", 1), errors="coerce").fillna(1).replace(0, 1)
    data["premium_deductible_ratio"] = premium / deductible

    for column in ["police_report_available", "property_damage", "collision_type"]:
        if column in data.columns:
            data[f"{column}_missing"] = (
                data[column].astype(str).str.upper().eq("UNKNOWN").astype(int)
            )

    for column in DROP_COLUMNS:
        if column in data.columns:
            data.drop(columns=column, inplace=True)

    high_null = [col for col in data.columns if data[col].isna().mean() > 0.3]
    if high_null:
        data.drop(columns=high_null, inplace=True)

    for column in data.columns:
        if column == TARGET_COLUMN:
            continue
        if data[column].dtype == "object":
            data[column] = data[column].fillna("Unknown").astype(str)
        else:
            data[column] = pd.to_numeric(data[column], errors="coerce").fillna(0)

    return data


def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    data = engineer_features(df)
    y = data[TARGET_COLUMN].map({"Y": 1, "N": 0, "Yes": 1, "No": 0})
    X = data.drop(columns=[TARGET_COLUMN])
    return X, y


def get_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    engineered = engineer_features(df)
    feature_cols = [col for col in engineered.columns if col != TARGET_COLUMN]
    numeric = [col for col in NUMERIC_COLUMNS if col in feature_cols]
    categorical = [col for col in CATEGORICAL_COLUMNS if col in feature_cols]
    remaining_numeric = [
        col
        for col in feature_cols
        if col not in categorical and col not in numeric and engineered[col].dtype != "object"
    ]
    numeric = numeric + remaining_numeric
    remaining_categorical = [
        col for col in feature_cols if col not in numeric and col not in categorical
    ]
    categorical = categorical + remaining_categorical
    return numeric, categorical


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )


def _build_classifier(model_type: str):
    if model_type == "logistic":
        return LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            C=0.8,
            random_state=42,
        )
    if model_type == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            random_state=42,
        )
    if model_type == "voting_ensemble":
        random_forest = RandomForestClassifier(
            n_estimators=500,
            max_depth=14,
            min_samples_leaf=1,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
        gradient_boosting = GradientBoostingClassifier(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            random_state=42,
        )
        return VotingClassifier(
            estimators=[
                ("random_forest", random_forest),
                ("gradient_boosting", gradient_boosting),
            ],
            voting="soft",
            weights=[2, 1],
        )

    return RandomForestClassifier(
        n_estimators=500,
        max_depth=14,
        min_samples_leaf=1,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )


def build_model_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    model_type: str = "voting_ensemble",
    use_smote: bool = True,
) -> Pipeline | ImbPipeline:
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    classifier = _build_classifier(model_type)

    if use_smote and model_type != "logistic":
        return ImbPipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("smote", SMOTE(random_state=42)),
                ("classifier", classifier),
            ]
        )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X, y = prepare_xy(df)
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def build_metadata(df: pd.DataFrame) -> dict:
    engineered = engineer_features(df)
    total_claim = pd.to_numeric(engineered["total_claim_amount"], errors="coerce")
    numeric, categorical = get_feature_columns(df)

    return {
        "feature_columns": [col for col in engineered.columns if col != TARGET_COLUMN],
        "numeric_features": numeric,
        "categorical_features": categorical,
        "thresholds": {
            "high_claim_amount": float(total_claim.quantile(0.75)),
            "high_risk_score": 0.70,
            "medium_risk_score": 0.40,
        },
        "sample_records": {
            "high_risk": _sample_record(engineered, prefer_fraud=True),
            "low_risk": _sample_record(engineered, prefer_fraud=False),
        },
    }


def _sample_record(df: pd.DataFrame, prefer_fraud: bool) -> dict:
    if TARGET_COLUMN not in df.columns:
        row = df.iloc[0]
    else:
        label = "Y" if prefer_fraud else "N"
        subset = df[df[TARGET_COLUMN] == label]
        row = subset.iloc[0] if not subset.empty else df.iloc[0]

    if TARGET_COLUMN in row.index:
        row = row.drop(labels=[TARGET_COLUMN])

    record = {}
    for key, value in row.to_dict().items():
        if pd.isna(value):
            record[key] = 0 if key in NUMERIC_COLUMNS else "Unknown"
        elif hasattr(value, "item"):
            record[key] = value.item()
        else:
            record[key] = value
    return record
