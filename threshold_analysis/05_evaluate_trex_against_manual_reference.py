#!/usr/bin/env python3
"""Compute the approved Phase-2 A-core for 78 manual-reference snippets.

This program intentionally computes only the approved A-core metrics: bias
(mean signed error) and MAE.  It never selects a threshold, changes counts,
or performs cleaning, imputation, inferential tests, bootstrap resampling,
correlation analysis, or secondary error metrics.

The required JSON configuration names three CSV inputs (paths may be relative
to the configuration file)::

  {
    "expected_ids_csv": "expected_ids_80_manifest.csv",
    "counts_csv": "normalized_threshold_counts.csv",
    "reference_csv": "manual_reference_counts.csv"
  }

``expected_ids_csv`` needs ``snippet_id`` and exactly 80 unique rows.
``counts_csv`` needs ``snippet_id``, ``track_conf_threshold`` and
``trex_net_count``: exactly 240 rows, one for every expected ID at each of
0.10, 0.20 and 0.30.  ``reference_csv`` needs ``snippet_id``,
``manual_reference_count`` and ``manual_reference_estimated``: exactly 80
rows, of which exactly two must be marked true.  IDs are compared exactly;
the script never strips suffixes or otherwise normalizes them.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


EXPECTED_POPULATION = 80
MAIN_POPULATION = 78
EXPECTED_THRESHOLDS = (Decimal("0.10"), Decimal("0.20"), Decimal("0.30"))
THRESHOLD_LABELS = {value: format(value, ".2f") for value in EXPECTED_THRESHOLDS}
ESTIMATED_REFERENCE_COUNTS = {
    "test-camera-MusolePath2-clip-2-firstframe-34763-cliptime-7-scaled": 314,
    "test-camera-MusolePath2-clip-3-firstframe-36083-cliptime-7-scaled": 213,
}
OUTPUT_NAMES = (
    "phase2_core_78_detail.csv",
    "phase2_core_78_summary.csv",
    "phase2_core_78_summary.json",
)


class InputError(ValueError):
    """Input does not meet the fixed Phase-2 contract."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="JSON configuration naming the three CSV inputs.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for the three fixed Phase-2 outputs.")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace all three existing Phase-2 outputs.")
    return parser.parse_args()


def resolve_path(value: Any, base_dir: Path, key: str) -> Path:
    if not isinstance(value, str) or not value:
        raise InputError(f"{key} must be a non-empty path string.")
    path = Path(value)
    return path if path.is_absolute() else (base_dir / path).resolve()


def read_config(path: Path) -> dict[str, Path]:
    try:
        with path.open(encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise InputError(f"Cannot read JSON configuration {path}: {error}") from error
    if not isinstance(raw, dict):
        raise InputError("Configuration root must be a JSON object.")
    required = ("expected_ids_csv", "counts_csv", "reference_csv")
    missing = [key for key in required if key not in raw]
    if missing:
        raise InputError("Configuration is missing required keys: " + ", ".join(missing))
    return {key: resolve_path(raw[key], path.parent.resolve(), key) for key in required}


def read_csv(path: Path, required_columns: tuple[str, ...], label: str) -> pd.DataFrame:
    if not path.is_file():
        raise InputError(f"{label} does not exist or is not a file: {path}")
    try:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError) as error:
        raise InputError(f"Cannot read {label} as UTF-8 CSV: {path}: {error}") from error
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise InputError(f"{label} is missing required columns: {', '.join(missing)}")
    return frame


def require_nonempty_ids(frame: pd.DataFrame, label: str) -> None:
    invalid = frame.index[frame["snippet_id"].map(lambda value: not isinstance(value, str) or not value)].tolist()
    if invalid:
        raise InputError(f"{label} has missing or empty snippet_id values at CSV rows {', '.join(str(index + 2) for index in invalid)}.")


def require_unique_ids(frame: pd.DataFrame, label: str) -> None:
    duplicates = sorted(frame.loc[frame["snippet_id"].duplicated(keep=False), "snippet_id"].unique())
    if duplicates:
        raise InputError(f"{label} contains duplicate snippet_id values: {duplicates}")


def parse_nonnegative_integer(value: str, *, label: str, row_number: int) -> int:
    if not isinstance(value, str) or not value or not value.isascii() or not value.isdecimal():
        raise InputError(f"{label} at CSV row {row_number} must be a non-negative integer; received {value!r}.")
    return int(value)


def parse_threshold(value: str, row_number: int) -> Decimal:
    try:
        parsed = Decimal(value)
    except (InvalidOperation, TypeError) as error:
        raise InputError(f"track_conf_threshold at CSV row {row_number} is invalid: {value!r}.") from error
    if not parsed.is_finite() or parsed not in EXPECTED_THRESHOLDS:
        expected = ", ".join(THRESHOLD_LABELS[item] for item in EXPECTED_THRESHOLDS)
        raise InputError(f"track_conf_threshold at CSV row {row_number} must be one of {expected}; received {value!r}.")
    return parsed


def parse_bool(value: str, row_number: int) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise InputError(f"manual_reference_estimated at CSV row {row_number} must be literal true or false; received {value!r}.")


def validate_expected_ids(frame: pd.DataFrame) -> list[str]:
    require_nonempty_ids(frame, "expected_ids_csv")
    require_unique_ids(frame, "expected_ids_csv")
    if len(frame) != EXPECTED_POPULATION:
        raise InputError(f"expected_ids_csv must contain exactly {EXPECTED_POPULATION} rows; received {len(frame)}.")
    return frame["snippet_id"].tolist()


def validate_counts(frame: pd.DataFrame, expected_ids: list[str]) -> pd.DataFrame:
    require_nonempty_ids(frame, "counts_csv")
    if len(frame) != EXPECTED_POPULATION * len(EXPECTED_THRESHOLDS):
        raise InputError(f"counts_csv must contain exactly 240 rows (80 IDs × 3 thresholds); received {len(frame)}.")
    records: list[dict[str, Any]] = []
    for index, row in frame.iterrows():
        records.append({
            "snippet_id": row["snippet_id"],
            "track_conf_threshold": parse_threshold(row["track_conf_threshold"], index + 2),
            "trex_net_count": parse_nonnegative_integer(row["trex_net_count"], label="trex_net_count", row_number=index + 2),
        })
    parsed = pd.DataFrame(records)
    duplicate_mask = parsed.duplicated(["snippet_id", "track_conf_threshold"], keep=False)
    if duplicate_mask.any():
        duplicate_pairs = [
            f"{row.snippet_id}@{THRESHOLD_LABELS[row.track_conf_threshold]}"
            for row in parsed.loc[duplicate_mask, ["snippet_id", "track_conf_threshold"]].itertuples(index=False)
        ]
        raise InputError("counts_csv contains duplicate snippet_id/track_conf_threshold pairs: " + ", ".join(sorted(duplicate_pairs)))
    expected_set = set(expected_ids)
    actual_set = set(parsed["snippet_id"])
    if actual_set != expected_set:
        missing = sorted(expected_set - actual_set)
        unexpected = sorted(actual_set - expected_set)
        raise InputError(f"counts_csv ID population differs from expected IDs; missing={missing}, unexpected={unexpected}.")
    by_threshold = {threshold: set(parsed.loc[parsed["track_conf_threshold"] == threshold, "snippet_id"]) for threshold in EXPECTED_THRESHOLDS}
    for threshold, observed_ids in by_threshold.items():
        if observed_ids != expected_set:
            raise InputError(
                f"counts_csv has a false population for threshold {THRESHOLD_LABELS[threshold]}; "
                f"missing={sorted(expected_set - observed_ids)}, unexpected={sorted(observed_ids - expected_set)}."
            )
    observed_thresholds = set(parsed["track_conf_threshold"])
    if observed_thresholds != set(EXPECTED_THRESHOLDS):
        raise InputError(f"counts_csv threshold set must be exactly {list(THRESHOLD_LABELS.values())}; received {sorted(map(str, observed_thresholds))}.")
    return parsed


def validate_references(frame: pd.DataFrame, expected_ids: list[str]) -> pd.DataFrame:
    require_nonempty_ids(frame, "reference_csv")
    require_unique_ids(frame, "reference_csv")
    if len(frame) != EXPECTED_POPULATION:
        raise InputError(f"reference_csv must contain exactly {EXPECTED_POPULATION} rows; received {len(frame)}.")
    expected_set = set(expected_ids)
    actual_set = set(frame["snippet_id"])
    if actual_set != expected_set:
        raise InputError(
            f"reference_csv ID population differs from expected IDs; missing={sorted(expected_set - actual_set)}, "
            f"unexpected={sorted(actual_set - expected_set)}."
        )
    records: list[dict[str, Any]] = []
    for index, row in frame.iterrows():
        records.append({
            "snippet_id": row["snippet_id"],
            "manual_reference_count": parse_nonnegative_integer(row["manual_reference_count"], label="manual_reference_count", row_number=index + 2),
            "manual_reference_estimated": parse_bool(row["manual_reference_estimated"], index + 2),
        })
    parsed = pd.DataFrame(records)
    estimated_count = int(parsed["manual_reference_estimated"].sum())
    if estimated_count != 2:
        raise InputError(f"reference_csv must mark exactly two manual_reference_estimated rows true; received {estimated_count}.")
    estimated = parsed.loc[parsed["manual_reference_estimated"]].set_index("snippet_id")
    if set(estimated.index) != set(ESTIMATED_REFERENCE_COUNTS):
        raise InputError("reference_csv must mark exactly the two audited estimated-reference IDs.")
    for snippet_id, audited_count in ESTIMATED_REFERENCE_COUNTS.items():
        if int(estimated.at[snippet_id, "manual_reference_count"]) != audited_count:
            raise InputError(
                f"Audited estimated-reference ID {snippet_id!r} must retain manual_reference_count={audited_count}."
            )
    return parsed


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    try:
        paths = read_config(args.config)
        expected_ids = validate_expected_ids(read_csv(paths["expected_ids_csv"], ("snippet_id",), "expected_ids_csv"))
        counts = validate_counts(read_csv(paths["counts_csv"], ("snippet_id", "track_conf_threshold", "trex_net_count"), "counts_csv"), expected_ids)
        references = validate_references(read_csv(paths["reference_csv"], ("snippet_id", "manual_reference_count", "manual_reference_estimated"), "reference_csv"), expected_ids)
    except InputError as error:
        print(f"INPUT ERROR: {error}", file=sys.stderr)
        return 2

    output_dir = args.output_dir.resolve()
    outputs = {name: output_dir / name for name in OUTPUT_NAMES}
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not args.overwrite:
        print("OUTPUT COLLISION: use --overwrite to replace all fixed Phase-2 outputs: " + ", ".join(map(str, existing)), file=sys.stderr)
        return 2
    output_dir.mkdir(parents=True, exist_ok=True)

    main_references = references.loc[~references["manual_reference_estimated"]].copy()
    if len(main_references) != MAIN_POPULATION:
        print(f"INPUT ERROR: expected exactly {MAIN_POPULATION} non-estimated references; received {len(main_references)}.", file=sys.stderr)
        return 2
    reference_by_id = main_references.set_index("snippet_id")
    records: list[dict[str, Any]] = []
    for threshold in EXPECTED_THRESHOLDS:
        threshold_counts = counts.loc[counts["track_conf_threshold"] == threshold].set_index("snippet_id")
        for snippet_id in expected_ids:
            if snippet_id not in reference_by_id.index:
                continue
            trex_count = int(threshold_counts.at[snippet_id, "trex_net_count"])
            manual_count = int(reference_by_id.at[snippet_id, "manual_reference_count"])
            signed_error = trex_count - manual_count
            records.append({
                "snippet_id": snippet_id,
                "track_conf_threshold": THRESHOLD_LABELS[threshold],
                "trex_net_count": trex_count,
                "manual_reference_count": manual_count,
                "manual_reference_estimated": "false",
                "signed_error": signed_error,
                "absolute_error": abs(signed_error),
            })
    if len(records) != MAIN_POPULATION * len(EXPECTED_THRESHOLDS):
        print("INPUT ERROR: internal population check failed; no outputs were written.", file=sys.stderr)
        return 2

    summary_rows: list[dict[str, Any]] = []
    for threshold in EXPECTED_THRESHOLDS:
        threshold_records = [row for row in records if row["track_conf_threshold"] == THRESHOLD_LABELS[threshold]]
        signed_errors = np.asarray([row["signed_error"] for row in threshold_records], dtype=np.int64)
        absolute_errors = np.asarray([row["absolute_error"] for row in threshold_records], dtype=np.int64)
        summary_rows.append({
            "track_conf_threshold": THRESHOLD_LABELS[threshold],
            "n_snippets": MAIN_POPULATION,
            "bias_mean_signed_error": float(np.mean(signed_errors)),
            "mae": float(np.mean(absolute_errors)),
        })

    detail_fields = ["snippet_id", "track_conf_threshold", "trex_net_count", "manual_reference_count", "manual_reference_estimated", "signed_error", "absolute_error"]
    summary_fields = ["track_conf_threshold", "n_snippets", "bias_mean_signed_error", "mae"]
    try:
        write_csv(outputs["phase2_core_78_detail.csv"], detail_fields, records)
        write_csv(outputs["phase2_core_78_summary.csv"], summary_fields, summary_rows)
        summary_document = {
            "analysis": "phase_2_manual_reference_accuracy_a_core",
            "population": {"expected_snippets": EXPECTED_POPULATION, "main_snippets": MAIN_POPULATION, "excluded_estimated_manual_reference_counts": 2},
            "thresholds": list(THRESHOLD_LABELS.values()),
            "metrics": {"signed_error": "trex_net_count - manual_reference_count", "absolute_error": "abs(trex_net_count - manual_reference_count)", "summary": ["bias_mean_signed_error", "mae"]},
            "summary": summary_rows,
            "inputs": {key: str(value) for key, value in paths.items()},
            "outputs": {name: str(path) for name, path in outputs.items()},
            "scope_limitations": ["No cleaning, imputation, automatic exclusion, threshold selection, secondary error metrics, bootstrap interval, correlation, or significance test was performed.", "This output is not a final result and does not substitute for Data Freeze or result authorization."],
        }
        with outputs["phase2_core_78_summary.json"].open("w", encoding="utf-8", newline="") as handle:
            json.dump(summary_document, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    except OSError as error:
        print(f"OUTPUT ERROR: {error}", file=sys.stderr)
        return 2
    print(f"PASS: wrote deterministic Phase-2 A-core outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
