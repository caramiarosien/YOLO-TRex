#!/usr/bin/env python3
"""Compute the approved Phase-3 A-core Koger comparison for 78 snippets.

This program makes exactly one pre-specified comparison: TRex at
``track_conf_threshold == 0.10`` versus the declared raw Koger net crossing
count, against the same manual reference counts as Phase 2.  It computes only
signed errors, absolute errors, paired absolute-error differences, Bias, MAE,
and the specified descriptive paired-difference summaries.  It never cleans,
imputes, selects a winner or threshold, performs tests, bootstrap resampling,
correlation, WAPE, RMSE, or secondary metrics.

Required JSON configuration (input paths may be relative to the config)::

  {
    "expected_ids_csv": "expected_ids_80_manifest.csv",
    "counts_csv": "normalized_threshold_counts.csv",
    "reference_csv": "manual_reference_counts.csv",
    "koger_csv": "koger_raw_net_counts.csv",
    "koger_raw_net_count_provenance_confirmed": true,
    "koger_raw_net_count_provenance_evidence_path": "koger_provenance.json",
    "koger_raw_net_count_provenance_evidence_sha256": "<64 hex characters>"
  }

The provenance gate is intentional. A separately labelled provisional mode is
available only when Cara explicitly authorizes use of the unchanged Working
Master values while raw provenance remains open. Such an output is always
marked ``NOT_FROZEN_PROVISIONAL_ANALYSIS_ONLY`` and never passes the raw-
provenance or results-release gate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
from decimal import Decimal, InvalidOperation
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import pandas as pd


EXPECTED_POPULATION = 80
MAIN_POPULATION = 78
PRIMARY_THRESHOLD = Decimal("0.10")
PRIMARY_THRESHOLD_LABEL = "0.10"
EXPECTED_THRESHOLDS = (Decimal("0.10"), Decimal("0.20"), Decimal("0.30"))
ESTIMATED_REFERENCE_COUNTS = {
    "test-camera-MusolePath2-clip-2-firstframe-34763-cliptime-7-scaled": 314,
    "test-camera-MusolePath2-clip-3-firstframe-36083-cliptime-7-scaled": 213,
}
KOGER_PROVENANCE_SCOPE = "RAW_KOGER_NET_CROSSING_COUNT_PER_IDENTICAL_SNIPPET"
OUTPUT_NAMES = (
    "phase3_core_78_detail.csv",
    "phase3_core_78_summary.csv",
    "phase3_core_78_summary.json",
)


class InputError(ValueError):
    """Input does not meet the fixed Phase-3 contract."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="JSON configuration naming the four CSV inputs and the provenance gate.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for the three fixed Phase-3 outputs.")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace all three existing Phase-3 outputs.")
    return parser.parse_args()


def resolve_path(value: Any, base_dir: Path, key: str) -> Path:
    if not isinstance(value, str) or not value:
        raise InputError(f"{key} must be a non-empty path string.")
    path = Path(value)
    return path if path.is_absolute() else (base_dir / path).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_config(path: Path) -> tuple[dict[str, Path], dict[str, str]]:
    try:
        with path.open(encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise InputError(f"Cannot read JSON configuration {path}: {error}") from error
    if not isinstance(raw, dict):
        raise InputError("Configuration root must be a JSON object.")
    required_paths = ("expected_ids_csv", "counts_csv", "reference_csv", "koger_csv")
    missing = [key for key in required_paths if key not in raw]
    if missing:
        raise InputError("Configuration is missing required keys: " + ", ".join(missing))
    paths = {key: resolve_path(raw[key], path.parent.resolve(), key) for key in required_paths}
    if not paths["koger_csv"].is_file():
        raise InputError(f"koger_csv does not exist or is not a file: {paths['koger_csv']}")
    koger_hash = sha256_file(paths["koger_csv"])
    confirmation_key = "koger_raw_net_count_provenance_confirmed"
    if raw.get(confirmation_key) is not True:
        provisional_key = "provisional_master_koger_use_authorized_by_cara"
        decision_source = raw.get("provisional_decision_source")
        if (
            raw.get(provisional_key) is not True
            or raw.get("release_status") != "NOT_FROZEN_PROVISIONAL_ANALYSIS_ONLY"
            or not isinstance(decision_source, str)
            or not decision_source
        ):
            raise InputError(
                "Koger provenance gate is closed. Raw provenance requires "
                f"{confirmation_key}=true with hash-bound evidence. A provisional Working-Master "
                f"comparison additionally requires {provisional_key}=true, a non-empty "
                "provisional_decision_source, and release_status="
                "'NOT_FROZEN_PROVISIONAL_ANALYSIS_ONLY'."
            )
        return paths, {
            "status": "RAW_PROVENANCE_NOT_VERIFIED",
            "mode": "CARA_AUTHORIZED_UNCHANGED_WORKING_MASTER_VALUES_PROVISIONAL_ONLY",
            "evidence_path": "",
            "evidence_sha256": "",
            "koger_csv_sha256": koger_hash,
            "decision_source": decision_source,
            "release_status": "NOT_FROZEN_PROVISIONAL_ANALYSIS_ONLY",
        }
    evidence_path = resolve_path(
        raw.get("koger_raw_net_count_provenance_evidence_path"), path.parent.resolve(),
        "koger_raw_net_count_provenance_evidence_path",
    )
    declared_evidence_hash = raw.get("koger_raw_net_count_provenance_evidence_sha256")
    if not evidence_path.is_file() or not isinstance(declared_evidence_hash, str) or len(declared_evidence_hash) != 64:
        raise InputError("Koger provenance gate requires an existing evidence JSON and its 64-character SHA-256.")
    if sha256_file(evidence_path) != declared_evidence_hash:
        raise InputError("Koger provenance evidence SHA-256 does not match the declared hash.")
    try:
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise InputError(f"Cannot read Koger provenance evidence JSON: {error}") from error
    if not isinstance(evidence, dict):
        raise InputError("Koger provenance evidence root must be a JSON object.")
    if evidence.get("status") != "VERIFIED" or evidence.get("scope") != KOGER_PROVENANCE_SCOPE:
        raise InputError("Koger provenance evidence must explicitly verify the approved raw snippet-level scope.")
    if evidence.get("koger_csv_sha256") != koger_hash:
        raise InputError("Koger provenance evidence is not bound to the configured koger_csv SHA-256.")
    source = evidence.get("decision_source")
    if not isinstance(source, str) or not source:
        raise InputError("Koger provenance evidence requires a non-empty decision_source.")
    return paths, {
        "status": "VERIFIED",
        "mode": "HASH_BOUND_RAW_PROVENANCE",
        "evidence_path": str(evidence_path),
        "evidence_sha256": declared_evidence_hash,
        "koger_csv_sha256": koger_hash,
        "decision_source": source,
    }


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


def parse_signed_integer(value: str, *, label: str, row_number: int) -> int:
    if not isinstance(value, str) or not value or not value.isascii():
        raise InputError(f"{label} at CSV row {row_number} must be an integer; received {value!r}.")
    digits = value[1:] if value.startswith("-") else value
    if not digits.isdecimal():
        raise InputError(f"{label} at CSV row {row_number} must be an integer; received {value!r}.")
    return int(value)


def parse_threshold(value: str, row_number: int) -> Decimal:
    try:
        parsed = Decimal(value)
    except (InvalidOperation, TypeError) as error:
        raise InputError(f"track_conf_threshold at CSV row {row_number} is invalid: {value!r}.") from error
    if not parsed.is_finite() or parsed not in EXPECTED_THRESHOLDS:
        raise InputError(f"track_conf_threshold at CSV row {row_number} must be one of 0.10, 0.20, 0.30; received {value!r}.")
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
        pairs = sorted(f"{row.snippet_id}@{row.track_conf_threshold:.2f}" for row in parsed.loc[duplicate_mask, ["snippet_id", "track_conf_threshold"]].itertuples(index=False))
        raise InputError("counts_csv contains duplicate snippet_id/track_conf_threshold pairs: " + ", ".join(pairs))
    expected_set = set(expected_ids)
    actual_set = set(parsed["snippet_id"])
    if actual_set != expected_set:
        raise InputError(f"counts_csv ID population differs from expected IDs; missing={sorted(expected_set - actual_set)}, unexpected={sorted(actual_set - expected_set)}.")
    for threshold in EXPECTED_THRESHOLDS:
        observed_ids = set(parsed.loc[parsed["track_conf_threshold"] == threshold, "snippet_id"])
        if observed_ids != expected_set:
            raise InputError(f"counts_csv has a false population for threshold {threshold:.2f}; missing={sorted(expected_set - observed_ids)}, unexpected={sorted(observed_ids - expected_set)}.")
    return parsed


def validate_references(frame: pd.DataFrame, expected_ids: list[str]) -> pd.DataFrame:
    require_nonempty_ids(frame, "reference_csv")
    require_unique_ids(frame, "reference_csv")
    if len(frame) != EXPECTED_POPULATION:
        raise InputError(f"reference_csv must contain exactly {EXPECTED_POPULATION} rows; received {len(frame)}.")
    expected_set = set(expected_ids)
    actual_set = set(frame["snippet_id"])
    if actual_set != expected_set:
        raise InputError(f"reference_csv ID population differs from expected IDs; missing={sorted(expected_set - actual_set)}, unexpected={sorted(actual_set - expected_set)}.")
    records: list[dict[str, Any]] = []
    for index, row in frame.iterrows():
        records.append({
            "snippet_id": row["snippet_id"],
            "manual_reference_count": parse_nonnegative_integer(row["manual_reference_count"], label="manual_reference_count", row_number=index + 2),
            "manual_reference_estimated": parse_bool(row["manual_reference_estimated"], index + 2),
        })
    parsed = pd.DataFrame(records)
    if int(parsed["manual_reference_estimated"].sum()) != 2:
        raise InputError("reference_csv must mark exactly two manual_reference_estimated rows true.")
    estimated = parsed.loc[parsed["manual_reference_estimated"]].set_index("snippet_id")
    if set(estimated.index) != set(ESTIMATED_REFERENCE_COUNTS):
        raise InputError("reference_csv must mark exactly the two audited estimated-reference IDs.")
    for snippet_id, audited_count in ESTIMATED_REFERENCE_COUNTS.items():
        if int(estimated.at[snippet_id, "manual_reference_count"]) != audited_count:
            raise InputError(
                f"Audited estimated-reference ID {snippet_id!r} must retain manual_reference_count={audited_count}."
            )
    return parsed


def validate_koger(frame: pd.DataFrame, expected_ids: list[str]) -> pd.DataFrame:
    require_nonempty_ids(frame, "koger_csv")
    require_unique_ids(frame, "koger_csv")
    if len(frame) != EXPECTED_POPULATION:
        raise InputError(f"koger_csv must contain exactly {EXPECTED_POPULATION} rows; received {len(frame)}.")
    expected_set = set(expected_ids)
    actual_set = set(frame["snippet_id"])
    if actual_set != expected_set:
        raise InputError(f"koger_csv ID population differs from expected IDs; missing={sorted(expected_set - actual_set)}, unexpected={sorted(actual_set - expected_set)}.")
    records = [
        {"snippet_id": row["snippet_id"], "koger_net_count": parse_signed_integer(row["koger_net_count"], label="koger_net_count", row_number=index + 2)}
        for index, row in frame.iterrows()
    ]
    return pd.DataFrame(records)


def atomic_write_all(outputs: dict[str, Path], csv_documents: dict[str, tuple[list[str], Iterable[dict[str, Any]]]], json_document: dict[str, Any]) -> None:
    temporary_paths: list[tuple[Path, Path]] = []
    try:
        for name, (fieldnames, rows) in csv_documents.items():
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=outputs[name].parent, prefix=f".{name}.", delete=False) as handle:
                temporary = Path(handle.name)
                writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
                writer.writeheader()
                writer.writerows(rows)
            temporary_paths.append((temporary, outputs[name]))
        json_name = "phase3_core_78_summary.json"
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=outputs[json_name].parent, prefix=f".{json_name}.", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(json_document, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        temporary_paths.append((temporary, outputs[json_name]))
        for temporary, destination in temporary_paths:
            os.replace(temporary, destination)
    except OSError:
        for temporary, _ in temporary_paths:
            temporary.unlink(missing_ok=True)
        raise


def method_summary(signed_errors: list[int], absolute_errors: list[int]) -> dict[str, float]:
    return {"bias_mean_signed_error": sum(signed_errors) / MAIN_POPULATION, "mae": sum(absolute_errors) / MAIN_POPULATION}


def main() -> int:
    args = parse_args()
    try:
        paths, provenance_confirmation = read_config(args.config)
        expected_ids = validate_expected_ids(read_csv(paths["expected_ids_csv"], ("snippet_id",), "expected_ids_csv"))
        counts = validate_counts(read_csv(paths["counts_csv"], ("snippet_id", "track_conf_threshold", "trex_net_count"), "counts_csv"), expected_ids)
        references = validate_references(read_csv(paths["reference_csv"], ("snippet_id", "manual_reference_count", "manual_reference_estimated"), "reference_csv"), expected_ids)
        koger = validate_koger(read_csv(paths["koger_csv"], ("snippet_id", "koger_net_count"), "koger_csv"), expected_ids)
    except InputError as error:
        print(f"INPUT ERROR: {error}", file=sys.stderr)
        return 2

    output_dir = args.output_dir.resolve()
    outputs = {name: output_dir / name for name in OUTPUT_NAMES}
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not args.overwrite:
        print("OUTPUT COLLISION: use --overwrite to replace all fixed Phase-3 outputs: " + ", ".join(map(str, existing)), file=sys.stderr)
        return 2
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        print(f"OUTPUT ERROR: {error}", file=sys.stderr)
        return 3

    main_references = references.loc[~references["manual_reference_estimated"]].set_index("snippet_id")
    if len(main_references) != MAIN_POPULATION:
        print(f"INPUT ERROR: expected exactly {MAIN_POPULATION} non-estimated references; received {len(main_references)}.", file=sys.stderr)
        return 2
    count_by_id = counts.loc[counts["track_conf_threshold"] == PRIMARY_THRESHOLD].set_index("snippet_id")
    koger_by_id = koger.set_index("snippet_id")
    records: list[dict[str, Any]] = []
    for snippet_id in expected_ids:
        if snippet_id not in main_references.index:
            continue
        trex_count = int(count_by_id.at[snippet_id, "trex_net_count"])
        manual_count = int(main_references.at[snippet_id, "manual_reference_count"])
        koger_count = int(koger_by_id.at[snippet_id, "koger_net_count"])
        trex_signed_error = trex_count - manual_count
        koger_signed_error = koger_count - manual_count
        trex_absolute_error = abs(trex_signed_error)
        koger_absolute_error = abs(koger_signed_error)
        records.append({
            "snippet_id": snippet_id,
            "track_conf_threshold": PRIMARY_THRESHOLD_LABEL,
            "manual_reference_count": manual_count,
            "manual_reference_estimated": "false",
            "trex_net_count": trex_count,
            "trex_signed_error": trex_signed_error,
            "trex_absolute_error": trex_absolute_error,
            "koger_net_count": koger_count,
            "koger_signed_error": koger_signed_error,
            "koger_absolute_error": koger_absolute_error,
            "paired_absolute_error_difference_trex_minus_koger": trex_absolute_error - koger_absolute_error,
        })
    if len(records) != MAIN_POPULATION:
        print("INPUT ERROR: internal population check failed; no outputs were written.", file=sys.stderr)
        return 2

    trex_signed = [row["trex_signed_error"] for row in records]
    trex_absolute = [row["trex_absolute_error"] for row in records]
    koger_signed = [row["koger_signed_error"] for row in records]
    koger_absolute = [row["koger_absolute_error"] for row in records]
    paired = [row["paired_absolute_error_difference_trex_minus_koger"] for row in records]
    summary_row: dict[str, Any] = {
        "track_conf_threshold": PRIMARY_THRESHOLD_LABEL,
        "n_snippets": MAIN_POPULATION,
        "trex_bias_mean_signed_error": method_summary(trex_signed, trex_absolute)["bias_mean_signed_error"],
        "trex_mae": method_summary(trex_signed, trex_absolute)["mae"],
        "koger_bias_mean_signed_error": method_summary(koger_signed, koger_absolute)["bias_mean_signed_error"],
        "koger_mae": method_summary(koger_signed, koger_absolute)["mae"],
        "paired_absolute_error_difference_mean_trex_minus_koger": sum(paired) / MAIN_POPULATION,
        "paired_absolute_error_difference_median_trex_minus_koger": median(paired),
        "paired_absolute_error_difference_trex_lower_count": sum(value < 0 for value in paired),
        "paired_absolute_error_difference_equal_count": sum(value == 0 for value in paired),
        "paired_absolute_error_difference_trex_higher_count": sum(value > 0 for value in paired),
    }
    detail_fields = list(records[0])
    summary_fields = list(summary_row)
    summary_document = {
        "analysis": (
            "phase_3_primary_trex_0_10_vs_raw_koger_a_core"
            if provenance_confirmation["status"] == "VERIFIED"
            else "phase_3_primary_trex_0_10_vs_working_master_koger_provisional_a_core"
        ),
        "inputs": {key: str(value) for key, value in paths.items()},
        "koger_raw_net_count_provenance_evidence": provenance_confirmation,
        "metrics": {
            "signed_error": "method_net_count - manual_reference_count",
            "absolute_error": "abs(method_net_count - manual_reference_count)",
            "paired_absolute_error_difference": "trex_absolute_error - koger_absolute_error",
            "summary": ["bias_mean_signed_error", "mae", "paired_absolute_error_difference_mean_trex_minus_koger", "paired_absolute_error_difference_median_trex_minus_koger", "paired_absolute_error_difference_trex_lower_count", "paired_absolute_error_difference_equal_count", "paired_absolute_error_difference_trex_higher_count"],
        },
        "outputs": {name: str(path) for name, path in outputs.items()},
        "population": {"expected_snippets": EXPECTED_POPULATION, "main_snippets": MAIN_POPULATION, "excluded_estimated_manual_reference_counts": 2},
        "scope_limitations": ["TRex is restricted to track_conf_threshold 0.10; no 0.20 or 0.30 Koger comparison was performed.", "No cleaning, imputation, automatic exclusion, threshold selection, winner selection, WAPE, RMSE, bootstrap interval, correlation, or significance test was performed.", "This output is not a final result and does not substitute for Data Freeze or result authorization."],
        "summary": summary_row,
    }
    try:
        atomic_write_all(outputs, {
            "phase3_core_78_detail.csv": (detail_fields, records),
            "phase3_core_78_summary.csv": (summary_fields, [summary_row]),
        }, summary_document)
    except OSError as error:
        print(f"OUTPUT ERROR: {error}", file=sys.stderr)
        return 3
    print(f"PASS: wrote deterministic Phase-3 A-core outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
