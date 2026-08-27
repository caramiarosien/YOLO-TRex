#!/usr/bin/env python3
"""Extract audited TRex net crossing counts without modifying source inputs.

This is deliberately a separate implementation rather than a wrapper around
``countline_bat_crossings.py``.  The reference script's batch mode writes event
CSVs next to each input video; this program never does.  It reads only NPZ
files, applies the approved countline rules, and writes five fixed artefacts to
an explicitly supplied output directory:

* ``net_counts_long_80.csv`` -- one deterministic row per snippet and threshold;
* ``net_counts_wide_80.csv`` -- one deterministic row per audited snippet;
* ``input_hash_manifest.csv`` -- SHA-256 provenance for every consumed input;
* ``net_count_extraction_summary.json`` -- machine-readable run result; and
* ``extract_net_counts.log`` -- execution log.

The CLI accepts JSON only (no PyYAML dependency).  Paths inside the JSON are
resolved relative to the JSON file.  Minimal configuration::

  {
    "audit_summary": ".../audit_summary.json",
    "audit_expected_ids": ".../audit_expected_ids.csv",
    "runs": [{
      "name": "thre_01", "track_conf_threshold": 0.10,
      "data_dir": "/read-only/thre_01/data",
      "track_npz_regex": "^(?P<snippet_container>data-(?P<snippet_id>.+?))/.*_id[0-9]+\\\\.npz$"
    }],
    "day_by_snippet": {"audited-snippet-id": "16Nov"}
  }

All 80 day values are intentionally required.  They must be supplied from an
audited source rather than inferred from file names.  The audit summary must
say ``PASS`` and the expected-ID CSV must contain exactly the same 80 passing
IDs for all three approved thresholds.  Any mismatch, malformed NPZ, output
collision, or source/output overlap is a non-zero exit; no manual correction,
ID normalisation, or silent exclusion is performed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


EXPECTED_RUNS = {"thre_01": 0.10, "thre_02": 0.20, "thre_03": 0.30}
DETECT_CONF_THRESHOLD = 0.10
EXPECTED_ID_COUNT = 80
MIN_FRAME_INDEX_DIFFERENCE = 5
MIN_ENDPOINT_DISTANCE_PX = 50.0
REQUIRED_NPZ_FIELDS = ("X#wcentroid", "Y#wcentroid", "frame", "tracklets", "video_size")
OUTPUT_NAMES = (
    "net_counts_long_80.csv",
    "net_counts_wide_80.csv",
    "input_hash_manifest.csv",
    "net_count_extraction_summary.json",
    "extract_net_counts.log",
)


class ContractError(ValueError):
    """A required provenance or input-contract condition was not met."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="JSON extraction configuration.")
    parser.add_argument("--output-dir", required=True, type=Path, help="New or explicit output directory.")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace all five fixed outputs.")
    return parser.parse_args()


def resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base_dir / path).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(f"{label} is unreadable JSON: {path}: {error}") from error
    if not isinstance(value, dict):
        raise ContractError(f"{label} must be a JSON object: {path}")
    return value


def load_audited_ids(summary_path: Path, ids_path: Path) -> list[str]:
    summary = read_json_object(summary_path, "audit_summary")
    if summary.get("audit_status") != "PASS":
        raise ContractError("audit_summary.audit_status must be exactly 'PASS'.")
    if summary.get("expected_snippet_id_count") != EXPECTED_ID_COUNT:
        raise ContractError("audit_summary must confirm exactly 80 expected snippet IDs.")
    mapped = summary.get("mapped_id_count_by_run")
    if not isinstance(mapped, dict) or mapped != {name: EXPECTED_ID_COUNT for name in EXPECTED_RUNS}:
        raise ContractError("audit_summary must confirm 80 mapped IDs for thre_01, thre_02, and thre_03.")
    if summary.get("errors") not in ({}, None):
        raise ContractError("audit_summary contains errors despite PASS status.")

    try:
        with ids_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as error:
        raise ContractError(f"audit_expected_ids is unreadable: {ids_path}: {error}") from error
    if len(rows) != EXPECTED_ID_COUNT or not rows or "snippet_id" not in rows[0]:
        raise ContractError("audit_expected_ids must contain exactly 80 rows with a snippet_id column.")
    ids = [row.get("snippet_id", "") for row in rows]
    if any(not value for value in ids) or len(set(ids)) != EXPECTED_ID_COUNT:
        raise ContractError("audit_expected_ids must contain 80 unique, non-empty IDs.")
    for row in rows:
        for name in EXPECTED_RUNS:
            if row.get(f"{name}_status") != "PASS":
                raise ContractError(f"audit_expected_ids has non-PASS status for {row['snippet_id']} in {name}.")
            if row.get(f"{name}_container_count") != "1" or int(row.get(f"{name}_track_file_count", "0")) < 1:
                raise ContractError(f"audit_expected_ids has incomplete track mapping for {row['snippet_id']} in {name}.")
    return sorted(ids)


def validate_config(config: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    for key in ("audit_summary", "audit_expected_ids", "runs", "day_by_snippet"):
        if key not in config:
            raise ContractError(f"Configuration is missing required key: {key}.")
    config["_audit_summary"] = resolve_path(config["audit_summary"], base_dir)
    config["_audit_expected_ids"] = resolve_path(config["audit_expected_ids"], base_dir)
    runs = config["runs"]
    if not isinstance(runs, list) or len(runs) != len(EXPECTED_RUNS):
        raise ContractError("runs must contain exactly the three approved run objects.")
    names = [run.get("name") for run in runs if isinstance(run, dict)]
    if sorted(names) != sorted(EXPECTED_RUNS):
        raise ContractError("runs must use thre_01, thre_02, and thre_03 exactly once.")
    for run in runs:
        if not isinstance(run, dict):
            raise ContractError("Every run must be a JSON object.")
        name = run["name"]
        if run.get("track_conf_threshold") != EXPECTED_RUNS[name]:
            raise ContractError(f"{name}.track_conf_threshold must be {EXPECTED_RUNS[name]!r}.")
        if not isinstance(run.get("data_dir"), str) or not isinstance(run.get("track_npz_regex"), str):
            raise ContractError(f"{name} requires string data_dir and track_npz_regex.")
        try:
            expression = re.compile(run["track_npz_regex"])
        except re.error as error:
            raise ContractError(f"{name}.track_npz_regex is invalid: {error}") from error
        required_groups = {"snippet_id", "snippet_container"}
        if not required_groups.issubset(expression.groupindex):
            raise ContractError(f"{name}.track_npz_regex requires named groups snippet_id and snippet_container.")
        run["_data_dir"] = resolve_path(run["data_dir"], base_dir)
        run["_expression"] = expression
    if not isinstance(config["day_by_snippet"], dict):
        raise ContractError("day_by_snippet must be a JSON object mapping every audited ID to a non-empty day label.")
    return config


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def discover_run_files(run: dict[str, Any], expected_ids: set[str]) -> dict[str, list[Path]]:
    data_dir: Path = run["_data_dir"]
    if not data_dir.is_dir():
        raise ContractError(f"{run['name']}.data_dir does not exist or is not a directory: {data_dir}")
    grouped: dict[str, list[Path]] = defaultdict(list)
    containers: dict[str, set[str]] = defaultdict(set)
    for path in sorted(data_dir.rglob("*.npz")):
        relative = path.relative_to(data_dir).as_posix()
        match = run["_expression"].search(relative)
        if not match:
            continue  # Auxiliary NPZ files were separately checked by the required entrance audit.
        snippet_id, container = match.group("snippet_id"), match.group("snippet_container")
        if not snippet_id or not container:
            raise ContractError(f"{run['name']}: empty snippet mapping in {path}")
        if snippet_id not in expected_ids:
            raise ContractError(f"{run['name']}: non-audited snippet ID {snippet_id!r} in {path}")
        grouped[snippet_id].append(path)
        containers[snippet_id].add(container)
    if set(grouped) != expected_ids:
        missing, extra = sorted(expected_ids - set(grouped)), sorted(set(grouped) - expected_ids)
        raise ContractError(f"{run['name']}: audited IDs no longer match input files; missing={missing}, extra={extra}")
    for snippet_id in sorted(expected_ids):
        if len(containers[snippet_id]) != 1 or not grouped[snippet_id]:
            raise ContractError(f"{run['name']}: non-unique or empty container mapping for {snippet_id}.")
    return {snippet_id: sorted(paths) for snippet_id, paths in grouped.items()}


def one_dimensional_float(value: Any, field: str, path: Path) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError) as error:
        raise ContractError(f"{path}: {field} cannot be converted to a one-dimensional float array.") from error
    if array.size == 0:
        raise ContractError(f"{path}: {field} is empty.")
    return array


def track_segments(path: Path) -> tuple[list[tuple[np.ndarray, np.ndarray, np.ndarray]], float, str]:
    """Load one NPZ read-only and return finite split segments plus its line height."""
    try:
        with np.load(path, allow_pickle=True) as archive:
            missing = [field for field in REQUIRED_NPZ_FIELDS if field not in archive.files]
            if missing:
                raise ContractError(f"{path}: missing required NPZ fields: {', '.join(missing)}")
            x = one_dimensional_float(archive["X#wcentroid"], "X#wcentroid", path)
            y = one_dimensional_float(archive["Y#wcentroid"], "Y#wcentroid", path)
            frames = one_dimensional_float(archive["frame"], "frame", path)
            if not (x.size == y.size == frames.size):
                raise ContractError(f"{path}: X, Y, and frame arrays have unequal lengths.")
            video_size = one_dimensional_float(archive["video_size"], "video_size", path)
            if video_size.size < 2 or not np.isfinite(video_size[1]) or video_size[1] <= 0:
                raise ContractError(f"{path}: video_size must provide a positive height at index 1.")
            tracklets = np.asarray(archive["tracklets"])
    except (OSError, ValueError, TypeError) as error:
        if isinstance(error, ContractError):
            raise
        raise ContractError(f"{path}: NPZ is unreadable or invalid: {type(error).__name__}: {error}") from error

    if tracklets.ndim != 2 or tracklets.shape[1] != 2:
        raise ContractError(f"{path}: tracklets must be an N-by-2 segment table.")
    segments: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for number, pair in enumerate(tracklets):
        try:
            start, end = int(pair[0]), int(pair[1])
        except (TypeError, ValueError, OverflowError) as error:
            raise ContractError(f"{path}: tracklets row {number} is not integer-indexed.") from error
        # The project reference loader applies ``array[start:end + 1]``
        # without an upper-bounds rejection. Real TRex files contain final
        # end markers at or slightly beyond ``array.size``; NumPy intentionally
        # truncates those slices. Mirror that loader exactly and validate only
        # integer, non-negative, ordered markers here.
        if start < 0 or end < start or pair[0] != start or pair[1] != end:
            raise ContractError(f"{path}: invalid inclusive tracklet bounds at row {number}: {pair!r}")
        # This exactly mirrors the approved split-segment loader: discard non-finite centroids.
        selection = np.isfinite(x[start : end + 1]) & np.isfinite(y[start : end + 1])
        segments.append((x[start : end + 1][selection], y[start : end + 1][selection], frames[start : end + 1][selection]))
    return segments, float(video_size[1]), path.stem.rsplit("_id", 1)[0]


def count_snippet(paths: list[Path], snippet_id: str) -> dict[str, Any]:
    forward = backward = total_segments = qualifying_segments = 0
    heights: set[float] = set()
    video_names: set[str] = set()
    for path in paths:
        segments, height, video_name = track_segments(path)
        heights.add(height)
        video_names.add(video_name)
        total_segments += len(segments)
        for x, y, frames in segments:
            # The approved quality filter is active: lifespan is endpoint frame-index difference.
            if x.size < 2 or not np.all(np.isfinite(frames)):
                continue
            lifespan = frames[-1] - frames[0]
            distance = float(np.hypot(x[-1] - x[0], y[-1] - y[0]))
            if lifespan < MIN_FRAME_INDEX_DIFFERENCE or distance < MIN_ENDPOINT_DISTANCE_PX:
                continue
            qualifying_segments += 1
            line_y = int(height) / 2.0
            # flip_direction=False: screen-Y decreasing is forward (+1).
            if y[0] >= line_y and y[-1] <= line_y:
                forward += 1
            elif y[0] <= line_y and y[-1] >= line_y:
                backward += 1
    if len(heights) != 1:
        raise ContractError(f"{snippet_id}: source files do not agree on video height: {sorted(heights)}")
    if len(video_names) != 1:
        raise ContractError(f"{snippet_id}: source files do not agree on video name: {sorted(video_names)}")
    net_count = forward - backward
    if net_count < 0:
        raise ContractError(
            f"{snippet_id}: extracted net count is negative ({net_count}); "
            "the Priority-A data contract requires non-negative counts and forbids automatic correction."
        )
    return {
        "video_name": next(iter(video_names)),
        "counting_forward": forward,
        "counting_backward": backward,
        "net_count": net_count,
        "source_file_count": len(paths),
        "total_segments": total_segments,
        "qualifying_segments": qualifying_segments,
    }


def main() -> int:
    args = parse_args()
    try:
        config_path = args.config.resolve()
        config = validate_config(read_json_object(config_path, "configuration"), config_path.parent)
        expected_ids = load_audited_ids(config["_audit_summary"], config["_audit_expected_ids"])
        expected_set = set(expected_ids)
        days = config["day_by_snippet"]
        if set(days) != expected_set or any(not isinstance(days[item], str) or not days[item] for item in expected_ids):
            raise ContractError("day_by_snippet must contain exactly the 80 audited IDs, each with a non-empty day label.")
        files_by_run = {run["name"]: discover_run_files(run, expected_set) for run in sorted(config["runs"], key=lambda item: item["name"])}
        output_dir = args.output_dir.resolve()
        input_roots = [run["_data_dir"] for run in config["runs"]]
        if any(is_relative_to(output_dir, source) or is_relative_to(source, output_dir) for source in input_roots):
            raise ContractError("output_dir must neither equal nor contain nor be nested within a read-only input directory.")
        outputs = {name: output_dir / name for name in OUTPUT_NAMES}
        existing = [path for path in outputs.values() if path.exists()]
        if existing and not args.overwrite:
            raise ContractError("output collision; use --overwrite to replace all fixed outputs: " + ", ".join(map(str, existing)))
    except ContractError as error:
        print(f"CONTRACT ERROR: {error}", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=outputs["extract_net_counts.log"], filemode="w", encoding="utf-8", level=logging.INFO, format="%(levelname)s %(message)s", force=True)
    try:
        logging.info("Starting audited read-only count extraction with %s", config_path)
        long_rows: list[dict[str, Any]] = []
        manifest_paths = {config_path, config["_audit_summary"], config["_audit_expected_ids"]}
        for run_name in sorted(EXPECTED_RUNS):
            for paths in files_by_run[run_name].values():
                manifest_paths.update(paths)
        manifest_rows = [{"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size, "role": "track_npz" if path.suffix == ".npz" else "contract_input"} for path in sorted(manifest_paths)]

        for run_name in sorted(EXPECTED_RUNS):
            for snippet_id in expected_ids:
                result = count_snippet(files_by_run[run_name][snippet_id], snippet_id)
                long_rows.append({
                    "snippet_id": snippet_id, "day": days[snippet_id], "source_video_name": result["video_name"],
                    "run_name": run_name, "track_conf_threshold": EXPECTED_RUNS[run_name], "detect_conf_threshold": DETECT_CONF_THRESHOLD,
                    "split_segments": True, "quality_filter_enabled": True, "min_frame_index_difference": MIN_FRAME_INDEX_DIFFERENCE,
                    "min_endpoint_distance_px": MIN_ENDPOINT_DISTANCE_PX, "flip_direction": False,
                    "trex_forward_count": result["counting_forward"], "trex_backward_count": result["counting_backward"], "trex_net_count": result["net_count"],
                    "source_file_count": result["source_file_count"], "total_segments": result["total_segments"], "qualifying_segments": result["qualifying_segments"],
                })
        long_rows.sort(key=lambda row: (row["snippet_id"], row["track_conf_threshold"]))
        long_fields = list(long_rows[0])
        write_csv(outputs["net_counts_long_80.csv"], long_fields, long_rows)
        by_key = {(row["snippet_id"], row["run_name"]): row for row in long_rows}
        wide_rows = []
        for snippet_id in expected_ids:
            row: dict[str, Any] = {"snippet_id": snippet_id, "day": days[snippet_id]}
            for run_name in sorted(EXPECTED_RUNS):
                source = by_key[(snippet_id, run_name)]
                threshold_label = f"{EXPECTED_RUNS[run_name]:.2f}".replace(".", "_")
                for column in ("trex_forward_count", "trex_backward_count", "trex_net_count"):
                    row[f"{column}_{threshold_label}"] = source[column]
            wide_rows.append(row)
        write_csv(outputs["net_counts_wide_80.csv"], list(wide_rows[0]), wide_rows)
        write_csv(outputs["input_hash_manifest.csv"], ["path", "sha256", "bytes", "role"], manifest_rows)
        summary = {
            "status": "GEPRUEFT_TECHNICAL_OUTPUT_PENDING_DATA_FREEZE",
            "config_path": str(config_path), "config_sha256": sha256_file(config_path),
            "audit_summary_path": str(config["_audit_summary"]), "audit_summary_sha256": sha256_file(config["_audit_summary"]),
            "audit_expected_ids_path": str(config["_audit_expected_ids"]), "audit_expected_ids_sha256": sha256_file(config["_audit_expected_ids"]),
            "input_manifest_sha256": sha256_file(outputs["input_hash_manifest.csv"]), "expected_snippet_count": EXPECTED_ID_COUNT,
            "rows_written": len(long_rows), "runs": EXPECTED_RUNS,
            "approved_countline_configuration": {"split_segments": True, "quality_filter_enabled": True, "min_frame_index_difference": MIN_FRAME_INDEX_DIFFERENCE, "min_endpoint_distance_px": MIN_ENDPOINT_DISTANCE_PX, "flip_direction": False},
            "outputs": {name: str(path) for name, path in outputs.items()},
            "limitations": ["No event CSVs were written to input directories.", "No manual correction, ID normalization, or exclusion was performed.", "This output is not a Frozen Analysis Dataset and is not authorized for final Results before Data Freeze."],
        }
        with outputs["net_count_extraction_summary.json"].open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        logging.info("Completed %d deterministic long rows", len(long_rows))
        print(f"PASS: wrote audited net counts for {len(long_rows)} rows to {output_dir}")
        return 0
    except (ContractError, OSError, ValueError, TypeError) as error:
        logging.exception("Extraction failed")
        print(f"EXTRACTION ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
