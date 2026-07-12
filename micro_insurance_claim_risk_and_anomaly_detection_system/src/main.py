"""Project entry point: prepare data, train model, score claims, or launch checks."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from predict import run_sample, score_batch  # noqa: E402
from prepare_data import DEFAULT_SOURCE, prepare_dataset  # noqa: E402
from preprocessing import load_raw_data  # noqa: E402
from train import train  # noqa: E402
from utils import DATA_PATH, METRICS_PATH  # noqa: E402


def run_prepare(source: Path) -> None:
    prepare_dataset(source, DATA_PATH)


def run_train(model: str, use_smote: bool) -> None:
    try:
        train(model_type=model, use_smote=use_smote)
    except ValueError as exc:
        if "SMOTE" in str(exc):
            print("SMOTE failed; retrying with class_weight only.")
            train(model_type=model, use_smote=False)
        else:
            raise


def run_eda() -> None:
    subprocess.run([sys.executable, str(SRC_DIR / "run_eda.py")], check=True)


def run_batch(limit: int | None = None) -> None:
    df = load_raw_data().drop(columns=["fraud_reported"])
    if limit:
        df = df.head(limit)
    results = score_batch(df)
    print(results[["risk_score", "risk_tier", "review_action", "reason_codes"]].head(10).to_string())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Micro Insurance Claim Risk and Anomaly Detection System"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="Prepare dataset CSV from source file.")
    prepare_parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)

    train_parser = subparsers.add_parser("train", help="Train fraud detection model.")
    train_parser.add_argument("--model", choices=["random_forest", "logistic"], default="random_forest")
    train_parser.add_argument("--no-smote", action="store_true")

    subparsers.add_parser("eda", help="Execute EDA notebook and save plots.")
    subparsers.add_parser("sample", help="Score a sample claim.")
    batch_parser = subparsers.add_parser("batch", help="Score all claims in prepared CSV.")
    batch_parser.add_argument("--limit", type=int, default=20)

    pipeline_parser = subparsers.add_parser("pipeline", help="Run prepare -> train -> sample.")
    pipeline_parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    pipeline_parser.add_argument("--model", choices=["random_forest", "logistic"], default="random_forest")
    pipeline_parser.add_argument("--no-smote", action="store_true")
    pipeline_parser.add_argument("--skip-eda", action="store_true")

    args = parser.parse_args()

    if args.command == "prepare":
        run_prepare(args.source)
    elif args.command == "train":
        run_train(args.model, use_smote=not args.no_smote)
    elif args.command == "eda":
        run_eda()
    elif args.command == "sample":
        run_sample()
    elif args.command == "batch":
        run_batch(args.limit)
    elif args.command == "pipeline":
        run_prepare(args.source)
        run_train(args.model, use_smote=not args.no_smote)
        if not args.skip_eda:
            try:
                run_eda()
            except subprocess.CalledProcessError:
                print("EDA notebook execution failed; continuing with saved training artifacts.")
        run_sample()
        if METRICS_PATH.exists():
            print(json.dumps(json.loads(METRICS_PATH.read_text()), indent=2))


if __name__ == "__main__":
    main()
