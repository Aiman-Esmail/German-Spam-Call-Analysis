"""
Generate a synthetic CSV of German-style phone call records for ML experiments.
"""

from __future__ import annotations

import argparse
import csv
import random
from datetime import datetime, timedelta


# Common German mobile network prefixes (national form without leading 0)
MOBILE_PREFIXES = (
    "151",
    "152",
    "155",
    "157",
    "159",
    "160",
    "162",
    "163",
    "170",
    "171",
    "172",
    "173",
    "175",
    "176",
    "177",
    "178",
    "179",
)

# Sample geographic area codes (national, without leading 0)
LANDLINE_AREAS = (
    "30",
    "40",
    "69",
    "89",
    "201",
    "211",
    "221",
    "228",
    "231",
    "911",
)


def _random_mobile_e164(rng: random.Random) -> str:
    prefix = rng.choice(MOBILE_PREFIXES)
    rest = "".join(rng.choice("0123456789") for _ in range(8))
    return f"+49{prefix}{rest}"


def _random_landline_e164(rng: random.Random) -> str:
    ac = rng.choice(LANDLINE_AREAS)
    width = 11 - len(ac)
    rest = "".join(rng.choice("0123456789") for _ in range(width))
    return f"+49{ac}{rest}"


def random_german_number(rng: random.Random) -> str:
    return _random_mobile_e164(rng) if rng.random() < 0.72 else _random_landline_e164(rng)


def random_timestamp(rng: random.Random, end: datetime, span_days: int) -> str:
    delta = timedelta(seconds=rng.randint(0, span_days * 24 * 3600))
    t = end - delta
    return t.strftime("%Y-%m-%d %H:%M:%S")


def random_duration_seconds(rng: random.Random) -> int:
    # Short spam/automated vs typical conversation
    if rng.random() < 0.22:
        return rng.randint(3, 45)
    return rng.randint(15, 600)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate fake German phone call CSV.")
    parser.add_argument("-n", "--rows", type=int, default=500, help="Number of rows (default: 500)")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="german_phone_calls.csv",
        help="Output CSV path (default: german_phone_calls.csv)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    end = datetime.now()
    span_days = 120

    rows = []
    for _ in range(args.rows):
        label = "Spam" if rng.random() < 0.28 else "Normal"
        rows.append(
            {
                "Timestamp": random_timestamp(rng, end, span_days),
                "Number": random_german_number(rng),
                "Duration": random_duration_seconds(rng),
                "Label": label,
            }
        )

    fieldnames = ["Timestamp", "Number", "Duration", "Label"]
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {args.rows} rows to {args.output}")


if __name__ == "__main__":
    main()
