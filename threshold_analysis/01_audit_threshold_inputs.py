#!/usr/bin/env python3
"""Read-only entrance audit for the three TRex threshold output sets.

The program deliberately has no defaults for project-specific IDs, paths, or
run metadata.  Those items must be supplied in a JSON configuration so that
the audit cannot silently reconstruct, correct, or exclude an input.

Minimal configuration shape (all paths may be absolute or relative to this
configuration file)::

  {
    "expected_snippet_ids_csv": ".../expected_ids_80_manifest.csv",
    "countline_configuration": {
      "split_segments": true,
      "quality_filter": "active",
      "min_frames": 5,
      "min_distance_px": 50,
      "flip_direction": false
    },
    "runs": [{
      "name": "thre_01",
      "track_conf_threshold": 0.10,
      "data_dir": "/read-only/path/to/thre_01/data",
      "metadata_path": "/read-only/path/to/thre_01/metadata.json",
      "required_metadata": {
        "metadata_scope": "RAW_RUN_SIDECAR",
        "detect_conf_threshold": 0.10,
        "track_conf_threshold": 0.10
      },
      "track_npz_regex": "^(?P<snippet_container>data-(?P<snippet_id>.+?))/.*_id[0-9]+\\.npz$"
    }]
  }

``track_npz_regex`` is applied to each NPZ path relative to ``data_dir``.
Matching files are treated as trajectory NPZ files and must contain the required
TRex fields; non-matching NPZ files are recorded as auxiliary NPZ files and
only checked for readability.  The expression must contain the named group
``snippet_id`` and ``snippet_container``.  A snippet may contain multiple
trajectory files (one per track), but it must map to exactly one declared
container.  Both groups are intentionally required:
file names or directory names are not assumed to be canonical IDs.  The three
run names and thresholds must be exactly thre_01/0.10, thre_02/0.20 and
thre_03/0.30. The file at ``metadata_path`` must contain a JSON object with
``metadata_scope == RAW_RUN_SIDECAR``; each dotted key in ``required_metadata``
is compared exactly to its declared expected value.

This audit only reads configured inputs.  It never computes counts, changes
thresholds, modifies source data, or infers missing metadata.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


EXPECTED_RUNS = {"thre_01": 0.10, "thre_02": 0.20, "thre_03": 0.30}
REQUIRED_COUNTLINE = {
    "split_segments": True,
    "quality_filter": "active",
    "min_frames": 5,
    "min_distance_px": 50,
    "flip_direction": False,
}
REQUIRED_NPZ_FIELDS = ("X#wcentroid", "Y#wcentroid", "frame", "tracklets", "video_size")
REQUIRED_METADATA_SCOPE = "RAW_RUN_SIDECAR"
OUTPUT_NAMES = (
    "audit_threshold_inputs.csv",
    "audit_expected_ids.csv",
    "audit_countline_configuration.csv",
    "audit_summary.json",
    "audit_threshold_inputs.log",
)


class ConfigurationError(ValueError):
    """The declared audit contract is incomplete or internally inconsistent."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Path to the audit JSON configuration.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for the five fixed audit outputs.")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace all pre-existing audit outputs.")
    return parser.parse_args()


def resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base_dir / path).resolve()


def dotted_get(document: dict[str, Any], dotted_key: str) -> Any:
    current: Any = document
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(dotted_key)
        current = current[part]
    return current


def read_expected_ids_csv(path: Path) -> list[str]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or "snippet_id" not in reader.fieldnames:
                raise ConfigurationError("expected_snippet_ids_csv must contain a snippet_id column.")
            return [row["snippet_id"] for row in reader]
    except OSError as error:
        raise ConfigurationError(f"Cannot read expected_snippet_ids_csv {path}: {error}") from error


def ensure_config(config: Any, config_dir: Path) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise ConfigurationError("The configuration root must be a JSON object.")
    ids = config.get("expected_snippet_ids")
    if ids is None and isinstance(config.get("expected_snippet_ids_csv"), str):
        manifest_path = resolve_path(config["expected_snippet_ids_csv"], config_dir)
        ids = read_expected_ids_csv(manifest_path)
        config["_expected_snippet_ids_csv"] = manifest_path
        config["expected_snippet_ids"] = ids
    if not isinstance(ids, list) or not all(isinstance(value, str) and value for value in ids):
        raise ConfigurationError("Provide expected_snippet_ids or expected_snippet_ids_csv with non-empty strings.")
    if len(ids) != 80:
        raise ConfigurationError(f"Expected exactly 80 snippet IDs, received {len(ids)}.")
    duplicates = sorted(value for value, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise ConfigurationError(f"expected_snippet_ids contains duplicates: {duplicates}")

    countline = config.get("countline_configuration")
    if not isinstance(countline, dict):
        raise ConfigurationError("countline_configuration must be an object.")
    for key, expected in REQUIRED_COUNTLINE.items():
        if countline.get(key) != expected:
            raise ConfigurationError(
                f"countline_configuration.{key} must be {expected!r}; received {countline.get(key)!r}."
            )

    runs = config.get("runs")
    if not isinstance(runs, list) or len(runs) != 3:
        raise ConfigurationError("runs must contain exactly three run objects.")
    names = [run.get("name") for run in runs if isinstance(run, dict)]
    if set(names) != set(EXPECTED_RUNS) or len(names) != 3:
        raise ConfigurationError("runs must use each of thre_01, thre_02, and thre_03 exactly once.")
    for run in runs:
        if not isinstance(run, dict):
            raise ConfigurationError("Every runs entry must be an object.")
        name = run["name"]
        if run.get("track_conf_threshold") != EXPECTED_RUNS[name]:
            raise ConfigurationError(
                f"{name}.track_conf_threshold must be {EXPECTED_RUNS[name]!r}; received "
                f"{run.get('track_conf_threshold')!r}."
            )
        for key in ("data_dir", "metadata_path", "track_npz_regex", "required_metadata"):
            if key not in run:
                raise ConfigurationError(f"{name} is missing required key: {key}.")
        if not isinstance(run["required_metadata"], dict) or not run["required_metadata"]:
            raise ConfigurationError(f"{name}.required_metadata must be a non-empty object.")
        for key, expected in (
            ("metadata_scope", REQUIRED_METADATA_SCOPE),
            ("detect_conf_threshold", 0.10),
            ("track_conf_threshold", EXPECTED_RUNS[name]),
        ):
            if run["required_metadata"].get(key) != expected:
                raise ConfigurationError(
                    f"{name}.required_metadata.{key} must declare {expected!r} for metadata verification."
                )
        try:
            expression = re.compile(run["track_npz_regex"])
        except (TypeError, re.error) as error:
            raise ConfigurationError(f"{name}.track_npz_regex is invalid: {error}") from error
        missing_groups = {"snippet_id", "snippet_container"} - set(expression.groupindex)
        if missing_groups:
            raise ConfigurationError(
                f"{name}.track_npz_regex needs named groups: {', '.join(sorted(missing_groups))}."
            )
        run["_data_dir"] = resolve_path(run["data_dir"], config_dir)
        run["_metadata_path"] = resolve_path(run["metadata_path"], config_dir)
        run["_snippet_id_expression"] = expression
    return config


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def inspect_npz(path: Path, *, require_track_fields: bool) -> tuple[list[str], list[str]]:
    """Return errors and warnings; no array values are changed or persisted."""
    errors: list[str] = []
    warnings: list[str] = []
    try:
        with np.load(path, allow_pickle=True) as archive:
            fields = set(archive.files)
            if not require_track_fields:
                return errors, warnings
            missing = [field for field in REQUIRED_NPZ_FIELDS if field not in fields]
            if missing:
                errors.append("missing_npz_fields:" + ",".join(missing))
                return errors, warnings
            for field in ("X#wcentroid", "Y#wcentroid", "frame"):
                if np.asarray(archive[field]).size == 0:
                    errors.append(f"empty_npz_field:{field}")
            video_size = np.asarray(archive["video_size"])
            if video_size.size < 2:
                errors.append("invalid_video_size:requires_width_and_height")
            elif not np.all(np.isfinite(video_size.astype(float, copy=False))) or np.any(video_size[:2] <= 0):
                errors.append("invalid_video_size:non_positive_or_non_finite")
    except Exception as error:  # NumPy exposes several exception classes by format/version.
        errors.append(f"npz_unreadable:{type(error).__name__}:{error}")
    return errors, warnings


def metadata_errors(run: dict[str, Any]) -> list[str]:
    path: Path = run["_metadata_path"]
    if not path.is_file():
        return [f"metadata_file_missing:{path}"]
    try:
        with path.open(encoding="utf-8") as handle:
            metadata = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        return [f"metadata_file_unreadable:{type(error).__name__}:{error}"]
    if not isinstance(metadata, dict):
        return ["metadata_not_a_json_object"]
    errors: list[str] = []
    for key, expected in run["required_metadata"].items():
        try:
            actual = dotted_get(metadata, key)
        except KeyError:
            errors.append(f"metadata_missing:{key}")
        else:
            if actual != expected:
                errors.append(f"metadata_mismatch:{key}:expected={expected!r}:actual={actual!r}")
    return errors


def audit_run(run: dict[str, Any], expected_ids: set[str]) -> tuple[list[dict[str, Any]], dict[str, list[Path]], dict[str, set[str]], list[str]]:
    name = run["name"]
    data_dir: Path = run["_data_dir"]
    rows: list[dict[str, Any]] = []
    by_id: dict[str, list[Path]] = defaultdict(list)
    containers_by_id: dict[str, set[str]] = defaultdict(set)
    declared_metadata_errors = metadata_errors(run)
    run_errors = list(declared_metadata_errors)
    if not data_dir.is_dir():
        run_errors.append(f"data_directory_missing:{data_dir}")
        rows.append({"run": name, "check": "data_directory", "status": "FAIL", "detail": run_errors[-1], "path": str(data_dir)})
        return rows, by_id, containers_by_id, run_errors

    track_paths = sorted(data_dir.rglob("*.npz"))
    if not track_paths:
        run_errors.append("no_npz_files_found")
    expression: re.Pattern[str] = run["_snippet_id_expression"]
    for path in track_paths:
        relative = path.relative_to(data_dir).as_posix()
        match = expression.search(relative)
        if not match:
            errors, warnings = inspect_npz(path, require_track_fields=False)
            rows.append({
                "run": name,
                "check": "auxiliary_npz",
                "status": "FAIL" if errors else "PASS",
                "detail": ";".join(errors or warnings or ["readable_auxiliary_npz"]),
                "path": str(path),
            })
            run_errors.extend(f"{relative}:{error}" for error in errors)
            continue
        snippet_id = match.group("snippet_id")
        snippet_container = match.group("snippet_container")
        if not snippet_id or not snippet_container:
            rows.append({"run": name, "check": "id_mapping", "status": "FAIL", "detail": "empty_snippet_id_or_container", "path": str(path)})
            run_errors.append(f"empty_snippet_id:{relative}")
            continue
        by_id[snippet_id].append(path)
        containers_by_id[snippet_id].add(snippet_container)
        errors, warnings = inspect_npz(path, require_track_fields=True)
        rows.append({
            "run": name,
            "check": "track_npz",
            "status": "FAIL" if errors else "PASS",
            "detail": ";".join(errors or warnings or ["readable_required_fields_present"]),
            "path": str(path),
        })
        run_errors.extend(f"{relative}:{error}" for error in errors)

    for error in declared_metadata_errors:
        rows.append({"run": name, "check": "required_metadata", "status": "FAIL", "detail": error, "path": str(run["_metadata_path"])})
    if not declared_metadata_errors:
        rows.append({"run": name, "check": "required_metadata", "status": "PASS", "detail": "all_declared_metadata_matches", "path": str(run["_metadata_path"])})

    for snippet_id, paths in sorted(by_id.items()):
        containers = containers_by_id[snippet_id]
        if len(containers) != 1:
            error = f"non_unique_id_mapping:{snippet_id}:containers={sorted(containers)}"
            run_errors.append(error)
            rows.append({"run": name, "check": "id_uniqueness", "status": "FAIL", "detail": error, "path": "|".join(map(str, paths))})
        if snippet_id not in expected_ids:
            error = f"unexpected_id:{snippet_id}"
            run_errors.append(error)
            rows.append({"run": name, "check": "id_membership", "status": "FAIL", "detail": error, "path": "|".join(map(str, paths))})
    for snippet_id in sorted(expected_ids - set(by_id)):
        error = f"missing_expected_id:{snippet_id}"
        run_errors.append(error)
        rows.append({"run": name, "check": "id_completeness", "status": "FAIL", "detail": error, "path": ""})
    return rows, by_id, containers_by_id, run_errors


def main() -> int:
    args = parse_args()
    try:
        with args.config.open(encoding="utf-8") as handle:
            config = ensure_config(json.load(handle), args.config.parent.resolve())
    except (OSError, json.JSONDecodeError, ConfigurationError) as error:
        print(f"CONFIGURATION ERROR: {error}", file=sys.stderr)
        return 2

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {name: output_dir / name for name in OUTPUT_NAMES}
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not args.overwrite:
        print("OUTPUT COLLISION: use --overwrite to replace existing audit outputs: " + ", ".join(map(str, existing)), file=sys.stderr)
        return 2

    logging.basicConfig(filename=outputs["audit_threshold_inputs.log"], filemode="w", encoding="utf-8", level=logging.INFO, format="%(levelname)s %(message)s", force=True)
    logging.info("Starting read-only threshold-input audit with configuration %s", args.config)
    expected_ids = config["expected_snippet_ids"]
    expected_set = set(expected_ids)
    input_rows: list[dict[str, Any]] = []
    all_by_run: dict[str, dict[str, list[Path]]] = {}
    containers_by_run: dict[str, dict[str, set[str]]] = {}
    errors_by_run: dict[str, list[str]] = {}
    for run in sorted(config["runs"], key=lambda item: item["name"]):
        rows, by_id, containers, errors = audit_run(run, expected_set)
        input_rows.extend(rows)
        all_by_run[run["name"]] = by_id
        containers_by_run[run["name"]] = containers
        errors_by_run[run["name"]] = errors
        logging.info("%s: %d mapped IDs, %d errors", run["name"], len(by_id), len(errors))

    expected_rows: list[dict[str, Any]] = []
    for snippet_id in expected_ids:
        row: dict[str, Any] = {"snippet_id": snippet_id}
        for run_name in sorted(EXPECTED_RUNS):
            paths = all_by_run[run_name].get(snippet_id, [])
            containers = containers_by_run[run_name].get(snippet_id, set())
            row[f"{run_name}_track_file_count"] = len(paths)
            row[f"{run_name}_container_count"] = len(containers)
            row[f"{run_name}_containers"] = "|".join(sorted(containers))
            row[f"{run_name}_paths"] = "|".join(map(str, paths))
            row[f"{run_name}_status"] = "PASS" if len(containers) == 1 and len(paths) > 0 else "FAIL"
        expected_rows.append(row)

    cross_run_errors: list[str] = []
    id_sets = {name: set(values) for name, values in all_by_run.items()}
    if all(id_sets.get(name, set()) == expected_set for name in EXPECTED_RUNS):
        input_rows.append({"run": "all", "check": "cross_run_id_set", "status": "PASS", "detail": "all_three_runs_match_the_80_expected_ids", "path": ""})
    else:
        cross_run_errors.append("cross_run_id_set_does_not_match_expected_ids")
        input_rows.append({"run": "all", "check": "cross_run_id_set", "status": "FAIL", "detail": cross_run_errors[-1], "path": ""})

    countline_rows = []
    for key, expected in REQUIRED_COUNTLINE.items():
        actual = config["countline_configuration"].get(key)
        countline_rows.append({"parameter": key, "expected": expected, "declared": actual, "status": "PASS" if actual == expected else "FAIL", "evidence_scope": "declared configuration; not a claim that count extraction was executed"})

    write_csv(outputs["audit_threshold_inputs.csv"], ["run", "check", "status", "detail", "path"], input_rows)
    write_csv(outputs["audit_expected_ids.csv"], list(expected_rows[0]), expected_rows)
    write_csv(outputs["audit_countline_configuration.csv"], ["parameter", "expected", "declared", "status", "evidence_scope"], countline_rows)
    errors = {name: values for name, values in errors_by_run.items() if values}
    if cross_run_errors:
        errors["all_runs"] = cross_run_errors
    status = "PASS" if not errors else "FAIL"
    summary = {
        "audit_status": status,
        "audit_kind": "read_only_input_audit",
        "config_path": str(args.config.resolve()),
        "expected_snippet_ids_csv": str(config.get("_expected_snippet_ids_csv", "embedded_in_config")),
        "expected_snippet_id_count": len(expected_ids),
        "expected_runs": EXPECTED_RUNS,
        "mapped_id_count_by_run": {name: len(all_by_run[name]) for name in sorted(EXPECTED_RUNS)},
        "errors": errors,
        "outputs": {name: str(path) for name, path in outputs.items()},
        "limitations": ["No count extraction, cleaning, correction, exclusion, or threshold adjustment was performed.", "Countline rows validate the declared configuration only; execution requires a separate reproducible count-extraction step."],
    }
    with outputs["audit_summary.json"].open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    logging.info("Audit completed with status %s", status)
    print(f"{status}: wrote fixed audit outputs to {output_dir}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
