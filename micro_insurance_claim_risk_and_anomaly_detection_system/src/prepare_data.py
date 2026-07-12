"""Convert source dataset (Excel/CSV/ZIP archive) into normalized project CSV."""

from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

import pandas as pd

from utils import DATA_DIR, DATA_PATH, TARGET_COLUMN

COLUMN_ALIASES = {
    "policy_deductable": "policy_deductible",
}

ARCHIVE_PATH = DATA_DIR / "archive.zip"
DEFAULT_SOURCE = ARCHIVE_PATH


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data.rename(columns=COLUMN_ALIASES, inplace=True)
    return data


def load_from_archive(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        spreadsheet_names = [
            name for name in archive.namelist() if name.lower().endswith((".xlsx", ".xls"))
        ]
        if not spreadsheet_names:
            raise ValueError(f"No Excel file found in archive: {path}")
        with archive.open(spreadsheet_names[0]) as spreadsheet:
            return pd.read_excel(io.BytesIO(spreadsheet.read()), sheet_name=0)


def load_source(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".zip":
        return load_from_archive(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=0)
    return pd.read_csv(path)


def prepare_dataset(source_path: Path, output_path: Path = DATA_PATH) -> pd.DataFrame:
    df = normalize_columns(load_source(source_path))

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Expected target column '{TARGET_COLUMN}' in source data.")

    for date_col in ["policy_bind_date", "incident_date"]:
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.strftime("%Y-%m-%d")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Prepared dataset: {output_path}")
    print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
    print(df[TARGET_COLUMN].value_counts().to_string())
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare insurance fraud dataset CSV.")
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Path to bundled archive.zip, Excel, or CSV file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DATA_PATH,
        help="Output CSV path.",
    )
    args = parser.parse_args()
    prepare_dataset(args.source, args.output)


if __name__ == "__main__":
    main()
