"""
Enrich raw phone call data with ML-based spam classification and per-call fines.

Pipeline
--------
1. Load raw call CSV (output of generate_data.py).
2. Engineer features from available columns.
3. Train a Random-Forest classifier (Label column used as ground truth).
4. Predict probabilities & labels on the full dataset.
5. Add fine_euro column (€0.20 per predicted spam call).
6. Persist the enriched CSV and the trained model (.pkl).

Usage
-----
    python process_data.py                        # defaults
    python process_data.py -i my_calls.csv -o out.csv --save-model model.pkl
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SPAM_FINE_EURO: float = 0.20
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.20

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *df* with ML-ready feature columns appended."""
    out = df.copy()

    # Temporal features
    ts = pd.to_datetime(out["Timestamp"])
    out["hour_of_day"] = ts.dt.hour                       # 0-23
    out["day_of_week"] = ts.dt.dayofweek                  # 0=Mon, 6=Sun
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    out["is_business_hours"] = (
        (out["hour_of_day"] >= 9) & (out["hour_of_day"] < 18)
    ).astype(int)

    # Number-based features
    out["is_mobile"] = out["Number"].str.startswith(
        ("+4915", "+4916", "+4917")
    ).astype(int)

    # Duration-based features
    out["duration_log"] = np.log1p(out["Duration"])       # right-skewed → log
    out["is_very_short"] = (out["Duration"] <= 45).astype(int)

    return out


FEATURE_COLS: list[str] = [
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "is_business_hours",
    "is_mobile",
    "duration_log",
    "is_very_short",
    "Duration",
]


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------
def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=8,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def train(df: pd.DataFrame) -> tuple[Pipeline, dict]:
    """Train classifier and return (fitted_pipeline, metrics_dict)."""
    df_feat = engineer_features(df)
    X = df_feat[FEATURE_COLS]
    y = (df_feat["Label"] == "Spam").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=["Normal", "Spam"], output_dict=True)

    log.info("Classification report on held-out test set:")
    log.info("\n" + classification_report(y_test, y_pred, target_names=["Normal", "Spam"]))

    return pipe, report


# ---------------------------------------------------------------------------
# Enrichment
# ---------------------------------------------------------------------------
def enrich(df: pd.DataFrame, pipe: Pipeline) -> pd.DataFrame:
    """Add predicted label, spam probability, and fine columns."""
    df_feat = engineer_features(df)
    X = df_feat[FEATURE_COLS]

    proba = pipe.predict_proba(X)[:, 1]          # P(Spam)
    predicted = pipe.predict(X)

    out = df.copy()
    out["spam_probability"] = proba.round(4)
    out["predicted_label"] = np.where(predicted == 1, "Spam", "Normal")
    out["fine_euro"] = (predicted == 1).astype(float) * SPAM_FINE_EURO
    return out


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def print_summary(df: pd.DataFrame) -> None:
    total = len(df)
    spam_n = int((df["predicted_label"] == "Spam").sum())
    fines = float(df["fine_euro"].sum())

    log.info("--- Pipeline Summary ---")
    log.info(f"Total calls processed : {total:,}")
    log.info(f"Predicted spam calls  : {spam_n:,}  ({spam_n/total*100:.1f} %)")
    log.info(f"Total fines (€)       : {fines:,.2f}")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="ML-based spam enrichment pipeline for German call records."
    )
    parser.add_argument(
        "-i", "--input",
        type=Path,
        default=Path("german_phone_calls.csv"),
        help="Raw input CSV from generate_data.py (default: german_phone_calls.csv)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("processed_calls.csv"),
        help="Enriched output CSV (default: processed_calls.csv)",
    )
    parser.add_argument(
        "--save-model",
        type=Path,
        default=Path("spam_model.pkl"),
        help="Where to persist the trained model (default: spam_model.pkl)",
    )
    args = parser.parse_args()

    log.info(f"Loading data from {args.input} …")
    df_raw = pd.read_csv(args.input, dtype={"Number": "string"})
    log.info(f"Loaded {len(df_raw):,} rows.")

    log.info("Training Random-Forest classifier …")
    pipe, _ = train(df_raw)

    log.info("Enriching full dataset with predictions …")
    df_out = enrich(df_raw, pipe)

    print_summary(df_out)

    df_out.to_csv(args.output, index=False)
    log.info(f"Saved enriched data → {args.output}")

    joblib.dump(pipe, args.save_model)
    log.info(f"Saved trained model  → {args.save_model}")


if __name__ == "__main__":
    main()
