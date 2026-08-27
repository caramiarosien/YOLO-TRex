#!/usr/bin/env python3
"""Create only the Phase-1 A-core threshold-sensitivity outputs (N=80).

Inputs are deliberately explicit: the authoritative 80-ID manifest CSV with a
``snippet_id`` column and a count CSV containing ``snippet_id``, ``track_conf_threshold``, and
``trex_net_count``. IDs are compared as supplied--this program never strips a
suffix, changes case, or otherwise normalizes them.  Counts must be
non-negative integers.  It performs no B/C diagnostics or statistical tests.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable


THRESHOLDS = (Decimal("0.10"), Decimal("0.20"), Decimal("0.30"))
REQUIRED_COLUMNS = ("snippet_id", "track_conf_threshold", "trex_net_count")
OUTPUT_NAMES = (
    "phase1_core_threshold_summary.csv",
    "phase1_core_pairwise_differences.csv",
    "phase1_core_pairwise_summary.csv",
    "phase1_core_summary.json",
)


class ContractError(ValueError):
    """Raised when supplied data do not satisfy the approved Phase-1 contract."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts-csv", required=True, type=Path)
    parser.add_argument("--expected-ids-csv", required=True, type=Path,
                        help="CSV containing a snippet_id column with exactly the approved 80 IDs.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true",
                        help="Explicitly replace all fixed outputs if they already exist.")
    return parser.parse_args()


def load_expected_ids(path: Path) -> list[str]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or "snippet_id" not in reader.fieldnames:
                raise ContractError("Expected-ID CSV must contain a snippet_id column.")
            value = [row["snippet_id"] for row in reader]
    except OSError as error:
        raise ContractError(f"Cannot read expected-ID CSV {path}: {error}") from error
    if any(not item for item in value):
        raise ContractError("Expected-ID CSV must contain only non-empty snippet_id values.")
    if len(value) != 80:
        raise ContractError(f"Expected exactly 80 IDs, received {len(value)}.")
    duplicates = sorted(item for item, count in Counter(value).items() if count > 1)
    if duplicates:
        raise ContractError(f"Expected-ID CSV contains duplicate IDs: {duplicates}")
    return value


def parse_threshold(value: str) -> Decimal:
    try:
        threshold = Decimal(value)
    except (InvalidOperation, ValueError) as error:
        raise ContractError(f"Invalid track_conf_threshold {value!r}.") from error
    if not threshold.is_finite() or threshold not in THRESHOLDS:
        raise ContractError("track_conf_threshold must be exactly one of 0.10, 0.20, 0.30.")
    return threshold


def parse_count(value: str, *, context: str) -> int:
    try:
        count = int(value)
    except (TypeError, ValueError) as error:
        raise ContractError(f"{context} must be an integer; received {value!r}.") from error
    if str(count) != value.strip():
        raise ContractError(f"{context} must be an integer literal; received {value!r}.")
    if count < 0:
        raise ContractError(f"{context} must not be negative; received {count}.")
    return count


def load_counts(path: Path, expected_ids: set[str]) -> dict[Decimal, dict[str, int]]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or set(REQUIRED_COLUMNS) - set(reader.fieldnames):
                missing = sorted(set(REQUIRED_COLUMNS) - set(reader.fieldnames or ()))
                raise ContractError(f"Counts CSV is missing required columns: {missing}")
            records: dict[Decimal, dict[str, int]] = {threshold: {} for threshold in THRESHOLDS}
            for line_number, row in enumerate(reader, start=2):
                snippet_id = row["snippet_id"]
                if not snippet_id:
                    raise ContractError(f"Row {line_number} has an empty snippet_id.")
                if snippet_id not in expected_ids:
                    raise ContractError(f"Row {line_number} has undeclared snippet_id {snippet_id!r}.")
                threshold = parse_threshold(row["track_conf_threshold"])
                if snippet_id in records[threshold]:
                    raise ContractError(f"Duplicate row for ID {snippet_id!r} at threshold {threshold}.")
                records[threshold][snippet_id] = parse_count(
                    row["trex_net_count"], context=f"Row {line_number} trex_net_count"
                )
    except OSError as error:
        raise ContractError(f"Cannot read counts CSV {path}: {error}") from error
    for threshold in THRESHOLDS:
        found = set(records[threshold])
        if found != expected_ids:
            missing, unexpected = sorted(expected_ids - found), sorted(found - expected_ids)
            raise ContractError(
                f"Threshold {threshold} does not have exactly the approved 80 IDs; "
                f"missing={missing}, unexpected={unexpected}."
            )
    return records


def atomic_write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent,
                                     prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    try:
        expected_ids = load_expected_ids(args.expected_ids_csv)
        counts = load_counts(args.counts_csv, set(expected_ids))
        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        outputs = {name: output_dir / name for name in OUTPUT_NAMES}
        collisions = [str(path) for path in outputs.values() if path.exists()]
        if collisions and not args.overwrite:
            raise ContractError("Output collision; use --overwrite: " + ", ".join(collisions))

        threshold_rows = []
        for threshold in THRESHOLDS:
            values = [counts[threshold][snippet_id] for snippet_id in sorted(expected_ids)]
            threshold_rows.append({
                "track_conf_threshold": f"{threshold:.2f}",
                "n_snippets": len(values),
                "net_count_sum": sum(values),
                "net_count_mean": sum(values) / len(values),
                "net_count_median": sorted(values)[(len(values) - 1) // 2] if len(values) % 2 else
                (sorted(values)[len(values) // 2 - 1] + sorted(values)[len(values) // 2]) / 2,
            })

        difference_rows: list[dict[str, Any]] = []
        comparison_summaries: list[dict[str, Any]] = []
        for left, right in ((THRESHOLDS[0], THRESHOLDS[1]), (THRESHOLDS[0], THRESHOLDS[2]), (THRESHOLDS[1], THRESHOLDS[2])):
            differences = [counts[left][snippet_id] - counts[right][snippet_id] for snippet_id in sorted(expected_ids)]
            absolute = [abs(value) for value in differences]
            for snippet_id, difference, absolute_difference in zip(sorted(expected_ids), differences, absolute):
                difference_rows.append({
                    "left_track_conf_threshold": f"{left:.2f}",
                    "right_track_conf_threshold": f"{right:.2f}",
                    "snippet_id": snippet_id,
                    "directed_count_difference_left_minus_right": difference,
                    "absolute_count_difference": absolute_difference,
                })
            ordered_differences, ordered_absolute = sorted(differences), sorted(absolute)
            midpoint = len(differences) // 2
            median = ordered_differences[midpoint] if len(differences) % 2 else (ordered_differences[midpoint - 1] + ordered_differences[midpoint]) / 2
            median_absolute = ordered_absolute[midpoint] if len(absolute) % 2 else (ordered_absolute[midpoint - 1] + ordered_absolute[midpoint]) / 2
            comparison_summaries.append({
                "left_track_conf_threshold": f"{left:.2f}",
                "right_track_conf_threshold": f"{right:.2f}",
                "n_snippets": len(differences),
                "mean_directed_difference_left_minus_right": sum(differences) / len(differences),
                "median_directed_difference_left_minus_right": median,
                "mean_absolute_difference": sum(absolute) / len(absolute),
                "median_absolute_difference": median_absolute,
                "identical_count_n": sum(value == 0 for value in differences),
                "identical_count_proportion": sum(value == 0 for value in differences) / len(differences),
                "left_higher_n": sum(value > 0 for value in differences),
                "left_lower_n": sum(value < 0 for value in differences),
                "total_net_count_difference_left_minus_right": sum(differences),
            })

        atomic_write_csv(outputs["phase1_core_threshold_summary.csv"], list(threshold_rows[0]), threshold_rows)
        atomic_write_csv(outputs["phase1_core_pairwise_differences.csv"], list(difference_rows[0]), difference_rows)
        atomic_write_csv(outputs["phase1_core_pairwise_summary.csv"], list(comparison_summaries[0]), comparison_summaries)
        atomic_write_json(outputs["phase1_core_summary.json"], {
            "analysis": "phase_1_threshold_sensitivity_A_core",
            "population_n": 80,
            "threshold_summaries": threshold_rows,
            "pairwise_comparison_summaries": comparison_summaries,
            "scope": "A-core only: no B/C diagnostics, visualizations, parameter selection, or statistical tests.",
        })
    except ContractError as error:
        print(f"CONTRACT ERROR: {error}", file=sys.stderr)
        return 2
    except OSError as error:
        print(f"I/O ERROR: {error}", file=sys.stderr)
        return 3
    print(f"PASS: wrote Phase-1 A-core outputs to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
