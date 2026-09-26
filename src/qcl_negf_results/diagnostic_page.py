"""Standalone diagnostic view derived from captured native objects.

Full histories remain in the immutable inventory. This small first-page view
never substitutes for source arrays or claims that a calculation has converged.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import h5py

from qcl_negf_contracts.artifacts import CONTRACT_SET
from .commits import json_bytes

TAIL_ROWS = 32
MAX_COLUMNS = 256


def _scalar(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else "Infinity" if value > 0 else "-Infinity"
    return value


def _tables(handle: Any) -> dict[str, Any]:
    tables: dict[str, Any] = {}
    for name in ("scba", "outer", "physical_markers", "physical_markers/channels", "physical_markers/collisions"):
        if name not in handle or not isinstance(handle[name], h5py.Group):
            continue
        group = handle[name]
        columns = {key: value for key, value in group.items()
                   if isinstance(value, h5py.Dataset) and value.ndim == 1}
        selected = sorted(columns)[:MAX_COLUMNS]
        table: dict[str, Any] = {"scope": "last 32 rows per scalar column; complete data in source object",
                 "columns": {}, "omitted_columns": max(0, len(columns) - len(selected))}
        for key in selected:
            column = columns[key]
            count = len(column)
            start = max(0, count - TAIL_ROWS)
            table["columns"][key] = {"rows": count, "start_row": start,
                "values": [_scalar(value) for value in column[start:]],
                "units": _scalar(column.attrs.get("units", "unspecified"))}
        tables[name] = table
    return tables


def summary(manifest: dict[str, Any], files: dict[str, tuple[Path, dict[str, Any]]]) -> bytes:
    rows = []
    seen: set[str] = set()
    for record in manifest["records"]:
        for item in record["included"]:
            name = item["object"]
            if name in seen or item["role"] != "science.history":
                continue
            seen.add(name)
            path, description = files[name]
            if description["media_type"] != "application/x-hdf5":
                continue
            with h5py.File(path, "r") as handle:
                rows.append({"identity": record["identity"], "generation": record["generation"],
                    "source_object": name, "source_sha256": description["sha256"],
                    "scientific_accepted": record["scientific_accepted"],
                    "quality": record["quality"], "tables": _tables(handle)})
    return json_bytes({"schema": "qcl-negf.diagnostic-page.v1", "contract_set": CONTRACT_SET,
        "snapshot_identity": manifest["snapshot_identity"], "job_status": manifest["job_status"],
        "snapshot_consistent": True, "scientific_acceptance": "per-record, never inferred from export success",
        "coverage": manifest["coverage"], "missing_records": manifest["missing_records"],
        "freshness": manifest["freshness"],
        "scalar_history_tails": rows,
        "nonfinite_values": "NaN / Infinity / -Infinity are explicit strings in this derived JSON view",
        "complete_sources": "manifest.json preserves full role/object/dependency inventory; export-index.json locates all parts"})
