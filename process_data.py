"""
Enrich raw phone call data with per-call fines and summary statistics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

SPAM_FINE_EURO = 0.20


def main() -> None:
    parser = argparse.ArgumentParser(description="Process raw call CSV and compute fines.")
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=Path("german_phone_calls.csv"),
        help="Input CSV from generate_data.py (default: german_phone_calls.csv)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("processed_calls.csv"),
        help="Enriched output CSV (default: processed_calls.csv)",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input, dtype={"Number": "string"})
    df["fine_euro"] = (df["Label"] == "Spam").astype(float) * SPAM_FINE_EURO

    total_calls = len(df)
    total_spam = int((df["Label"] == "Spam").sum())
    total_fines_euro = float(df["fine_euro"].sum())

    print("--- Summary ---")
    print(f"Total calls:        {total_calls}")
    print(f"Total Spam calls:   {total_spam}")
    print(f"Total Fines (Euro): {total_fines_euro:.2f}")

    df.to_csv(args.output, index=False)
    print(f"\nSaved enriched data to {args.output}")


if __name__ == "__main__":
    main()
