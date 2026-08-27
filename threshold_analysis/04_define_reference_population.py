#!/usr/bin/env python3
"""Mark the two audited estimated manual references and form the 78-ID population.

The program joins an explicit 80-ID population table to a reference table.
It copies stored reference fields verbatim, validates count fields without
changing them, and makes only the two audit-documented exclusions.  It does
not infer IDs from file names, normalize IDs, clean data, or replace counts.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable


ESTIMATED_REFERENCE_COUNTS = {
    "test-camera-MusolePath2-clip-2-firstframe-34763-cliptime-7-scaled": 314,
    "test-camera-MusolePath2-clip-3-firstframe-36083-cliptime-7-scaled": 213,
}
ESTIMATED_IDS = set(ESTIMATED_REFERENCE_COUNTS)
OUTPUT_NAMES = (
    "reference_quality_marked_80.csv",
    "reference_based_main_population_78.csv",
    "reference_quality_exclusions.csv",
    "reference_quality_population_summary.json",
)


class ContractError(ValueError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population-csv", required=True, type=Path,
                        help="CSV with exactly the declared 80 IDs and an ID column.")
    parser.add_argument("--reference-csv", required=True, type=Path,
                        help="CSV containing stored manual reference counts.")
    parser.add_argument("--expected-ids-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--population-id-column", default="snippet_id")
    parser.add_argument("--reference-id-column", default="snippet_id")
    parser.add_argument("--manual-count-column", default="manual_reference_count")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def expected_ids(path: Path) -> list[str]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or "snippet_id" not in reader.fieldnames:
                raise ContractError("Expected-ID CSV must contain a snippet_id column.")
            values = [row["snippet_id"] for row in reader]
    except OSError as error:
        raise ContractError(f"Cannot read expected-ID CSV {path}: {error}") from error
    if not all(isinstance(value, str) and value for value in values):
        raise ContractError("Expected-ID CSV must contain non-empty snippet_id values.")
    if len(values) != 80 or len(set(values)) != 80:
        raise ContractError("Expected-ID CSV must contain exactly 80 unique IDs.")
    if not ESTIMATED_IDS <= set(values):
        raise ContractError("The 80-ID manifest does not contain both audited estimated-reference IDs.")
    return values


def read_unique_csv(path: Path, id_column: str, *, label: str) -> tuple[list[str], dict[str, dict[str, str]]]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or id_column not in reader.fieldnames:
                raise ContractError(f"{label} CSV must contain column {id_column!r}.")
            fields = list(reader.fieldnames)
            rows: dict[str, dict[str, str]] = {}
            for line_number, row in enumerate(reader, start=2):
                identifier = row[id_column]
                if not identifier:
                    raise ContractError(f"{label} CSV row {line_number} has an empty ID.")
                if identifier in rows:
                    raise ContractError(f"{label} CSV has duplicate ID {identifier!r}.")
                rows[identifier] = row
    except OSError as error:
        raise ContractError(f"Cannot read {label} CSV {path}: {error}") from error
    return fields, rows


def validate_nonnegative_integer(value: str, *, context: str) -> None:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ContractError(f"{context} must be an integer; received {value!r}.") from error
    if str(parsed) != value.strip():
        raise ContractError(f"{context} must be an integer literal; received {value!r}.")
    if parsed < 0:
        raise ContractError(f"{context} must not be negative; received {parsed}.")


def validate_count_columns(row: dict[str, str], *, table: str, row_id: str, required_manual_column: str | None = None) -> None:
    """Reject negatives in every supplied count column without rewriting values."""
    for column, value in row.items():
        if "count" in column.casefold() and value != "":
            validate_nonnegative_integer(value, context=f"{table} ID {row_id!r} column {column!r}")
    if required_manual_column is not None:
        value = row.get(required_manual_column, "")
        if value == "":
            raise ContractError(f"Reference ID {row_id!r} has an empty {required_manual_column!r}.")
        validate_nonnegative_integer(value, context=f"Reference ID {row_id!r} manual count")


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
        declared_ids = expected_ids(args.expected_ids_csv)
        population_fields, population = read_unique_csv(args.population_csv, args.population_id_column, label="Population")
        reference_fields, reference = read_unique_csv(args.reference_csv, args.reference_id_column, label="Reference")
        if args.manual_count_column not in reference_fields:
            raise ContractError(f"Reference CSV must contain manual-count column {args.manual_count_column!r}.")
        declared_set = set(declared_ids)
        if set(population) != declared_set:
            raise ContractError("Population CSV IDs must match the declared 80-ID manifest exactly.")
        if set(reference) != declared_set:
            raise ContractError("Reference CSV IDs must match the declared 80-ID manifest exactly; no partial join is allowed.")
        for identifier in declared_ids:
            validate_count_columns(population[identifier], table="Population", row_id=identifier)
            validate_count_columns(reference[identifier], table="Reference", row_id=identifier,
                                   required_manual_column=args.manual_count_column)
        for identifier, audited_count in ESTIMATED_REFERENCE_COUNTS.items():
            stored_count = int(reference[identifier][args.manual_count_column])
            if stored_count != audited_count:
                raise ContractError(
                    f"Audited estimated-reference ID {identifier!r} must retain stored manual count "
                    f"{audited_count}; received {stored_count}. No substitute or correction is allowed."
                )

        reference_output_fields = ["snippet_id", "manual_reference_count"]
        reference_output_fields += [f"population__{field}" for field in population_fields]
        reference_output_fields += [f"reference__{field}" for field in reference_fields]
        reference_output_fields += ["manual_reference_estimated", "main_population_included"]
        marked_rows: list[dict[str, str]] = []
        for identifier in sorted(declared_ids):
            marked = {
                "snippet_id": identifier,
                # The value is copied as stored; the name is standardized only
                # for downstream contract consumers.
                "manual_reference_count": reference[identifier][args.manual_count_column],
            }
            marked.update({f"population__{field}": population[identifier][field] for field in population_fields})
            marked.update({f"reference__{field}": reference[identifier][field] for field in reference_fields})
            estimated = identifier in ESTIMATED_IDS
            marked["manual_reference_estimated"] = "true" if estimated else "false"
            marked["main_population_included"] = "false" if estimated else "true"
            marked_rows.append(marked)
        main_rows = [row for row in marked_rows if row["main_population_included"] == "true"]
        if len(marked_rows) != 80 or len(main_rows) != 78:
            raise ContractError(f"Population invariant failed: marked={len(marked_rows)}, main={len(main_rows)}.")
        exclusions = [{
            "snippet_id": identifier,
            "manual_reference_estimated": "true",
            "stored_manual_reference_count": reference[identifier][args.manual_count_column],
            "exclusion_reason": "audited_estimated_manual_reference_count_excluded_from_78_snippet_main_population",
            "audit_note": "Stored count retained unchanged; no substitute value applied.",
        } for identifier in sorted(ESTIMATED_IDS)]

        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        outputs = {name: output_dir / name for name in OUTPUT_NAMES}
        collisions = [str(path) for path in outputs.values() if path.exists()]
        if collisions and not args.overwrite:
            raise ContractError("Output collision; use --overwrite: " + ", ".join(collisions))
        atomic_write_csv(outputs["reference_quality_marked_80.csv"], reference_output_fields, marked_rows)
        atomic_write_csv(outputs["reference_based_main_population_78.csv"], reference_output_fields, main_rows)
        atomic_write_csv(outputs["reference_quality_exclusions.csv"], list(exclusions[0]), exclusions)
        atomic_write_json(outputs["reference_quality_population_summary.json"], {
            "analysis": "reference_quality_marking_and_78_snippet_main_population",
            "declared_population_n": 80,
            "estimated_manual_reference_n": 2,
            "main_population_n": 78,
            "excluded_snippet_ids": sorted(ESTIMATED_IDS),
            "audited_estimated_reference_counts": ESTIMATED_REFERENCE_COUNTS,
            "manual_count_handling": "Stored manual reference counts are copied unchanged; no replacement, correction, or normalization is performed.",
            "exclusion_rule": "Only the two IDs documented by the manual-reference-quality audit are excluded from the 78-snippet main population.",
        })
    except ContractError as error:
        print(f"CONTRACT ERROR: {error}", file=sys.stderr)
        return 2
    except OSError as error:
        print(f"I/O ERROR: {error}", file=sys.stderr)
        return 3
    print(f"PASS: wrote reference-quality marking and the 78-ID main population to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
