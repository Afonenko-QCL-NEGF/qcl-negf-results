"""Bounded proof that a cumulative native history preserves an earlier cutoff."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np

from qcl_negf_contracts.artifacts import digest_value
from qcl_negf_contracts.messages import ContractError
from .commits import decode_json, json_bytes
from .witnesses import verify_selected_witnesses
from .native import HISTORY_SCHEMA, MARKER_CHILD_FIELDS, validate_native_handle

CHUNK_ROWS = 131_072  # At most 1 MiB for each native Int64/Float64 column.


def _text(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _sources(handle: Any, identity: dict[str, Any]) -> tuple[dict[str, bytes], dict[str, int]]:
    validate_native_handle(handle, "science.history", HISTORY_SCHEMA)
    metadata = handle["metadata"]
    if (_text(metadata.attrs["schema"]) != HISTORY_SCHEMA
            or _text(metadata.attrs["artifact_role"]) != "science.history"
            or _text(metadata.attrs["representation"]) != "lossless consolidation of every local closed history segment"):
        raise ValueError("history is not the native cumulative representation")
    payload = metadata["source_segments_json"][()]
    if isinstance(payload, str):
        payload = payload.encode()
    if not isinstance(payload, bytes):
        raise ValueError("cumulative history source inventory must be UTF-8 JSON")
    segments = decode_json(b'{"segments":' + payload + b'}')["segments"]
    if not isinstance(segments, list):
        raise ValueError("history source segments must be a list")
    sources: dict[str, bytes] = {}
    counts = {"scba": 0, "outer": 0}
    for segment in segments:
        source_identity = segment["identity"]
        if any(source_identity.get(key) != identity[key]
               for key in ("point_id", "execution_id", "plan_fingerprint")):
            raise ValueError("cumulative history source belongs to another scientific point")
        attempt = source_identity.get("attempt")
        if not isinstance(attempt, int) or isinstance(attempt, bool) or not 1 <= attempt <= identity["attempt"]:
            raise ValueError("cumulative history source attempt is outside the committed cutoff")
        digest = digest_value(segment["sha256"])
        if digest in sources:
            raise ValueError("duplicate cumulative history source segment")
        sources[digest] = json_bytes(segment)
        for table in counts:
            count = segment[table + "_rows"]
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                raise ValueError("invalid cumulative history source row count")
            counts[table] += count
    for table, count in counts.items():
        if metadata.attrs[table + "_rows"] != count:
            raise ValueError("history table row count disagrees with its source inventory")
        group = handle[table]
        if "sequence" not in group:
            raise ValueError("history table has no immutable sequence")
        for column in group.values():
            if (not isinstance(column, h5py.Dataset) or column.shape != (count,)
                    or column.dtype.kind not in "if" or column.dtype.itemsize != 8):
                raise ValueError("history table is not complete native Int64/Float64 columns")
        sequence = group["sequence"]
        if sequence.dtype.kind != "i":
            raise ValueError("history sequence is not Int64")
        last = 0
        for start in range(0, count, CHUNK_ROWS):
            rows = sequence[start:start + CHUNK_ROWS]
            if rows[0] <= last or np.any(rows[1:] <= rows[:-1]):
                raise ValueError("history sequence is not strictly increasing")
            last = int(rows[-1])
    return sources, counts


def _annotations(previous: Any, current: Any, *, growing: bool = False) -> None:
    for key, value in previous.attrs.items():
        if growing and key in {"logical_shape", "nonfinite_count"}:
            continue
        if key not in current.attrs or not np.array_equal(value, current.attrs[key]):
            raise ValueError("new cumulative history changes scientific annotations")


def _equal_payload(previous: Any, current: Any) -> bool:
    old, new = np.asarray(previous), np.asarray(current)
    if old.dtype.kind in "OSU":
        # Vlen strings are object arrays: their pointer bytes are not their data.
        return bool(np.array_equal(old, new))
    return old.tobytes() == new.tobytes()


def _column_prefix(previous: Any, current: Any, count: int) -> None:
    if previous.dtype != current.dtype:
        raise ValueError("new cumulative history changes native dtype")
    _annotations(previous, current, growing=True)
    for start in range(0, count, CHUNK_ROWS):
        stop = min(start + CHUNK_ROWS, count)
        if not _equal_payload(previous[start:stop], current[start:stop]):
            raise ValueError("new cumulative history changes or omits earlier numerical rows")


def _table_prefix(previous: Any, current: Any, old_count: int, new_count: int) -> None:
    if new_count < old_count:
        raise ValueError("new cumulative history is shorter than its earlier cutoff")
    if not set(previous) <= set(current):
        raise ValueError("new cumulative history omits earlier scientific columns")
    for name, column in previous.items():
        _column_prefix(column, current[name], old_count)


def _psd_scalar_columns(previous: Any, current: Any, old_count: int, new_count: int) -> None:
    for name, old_column in previous.items():
        if name == "selected_blocks":
            continue
        new_column = current[name]
        if (not isinstance(old_column, h5py.Dataset) or not isinstance(new_column, h5py.Dataset)
                or old_column.shape != (old_count,) or new_column.shape != (new_count,)):
            raise ValueError("PSD history no longer matches the SCBA row coordinates")
        _column_prefix(old_column, new_column, old_count)


def _psd_sequence(handle: Any, count: int) -> None:
    for start in range(0, count, CHUNK_ROWS):
        stop = min(start + CHUNK_ROWS, count)
        if not _equal_payload(handle["scba/sequence"][start:stop],
                              handle["psd_history/sequence"][start:stop]):
            raise ValueError("PSD history sequence differs from SCBA history")


def _psd_selected(previous: Any, current: Any) -> None:
    selected = previous.get("selected_blocks")
    if selected is None:
        raise ValueError("native format 4.0 requires packed selected PSD witnesses")
    newer = current.get("selected_blocks")
    if not isinstance(selected, h5py.Group) or not isinstance(newer, h5py.Group):
        raise ValueError("new cumulative history omits selected PSD witnesses")
    verify_selected_witnesses(selected, newer)


def _marker_column_prefix(name: str, column: Any, replacement: Any, old_count: int, new_count: int) -> None:
    if name in MARKER_CHILD_FIELDS:
        _annotations(column, replacement, growing=True)
        _table_prefix(column, replacement, len(column["sequence"]), len(replacement["sequence"]))
        return
    if (not isinstance(column, h5py.Dataset) or not isinstance(replacement, h5py.Dataset)
            or column.shape != (old_count,) or replacement.shape != (new_count,)):
        raise ValueError("physical markers no longer match the SCBA row coordinates")
    _column_prefix(column, replacement, old_count)


def _physical_marker_prefix(previous: Any, current: Any, old_count: int, new_count: int) -> int:
    if "physical_markers" not in previous:
        raise ValueError("native format 4.0 requires physical marker history")
    if "physical_markers" not in current:
        raise ValueError("new cumulative history omits physical marker history")
    old, new = previous["physical_markers"], current["physical_markers"]
    if not isinstance(old, h5py.Group) or not isinstance(new, h5py.Group):
        raise ValueError("physical marker history must be a native group")
    _annotations(old, new, growing=True)
    if not {"sequence", "available"} <= set(old) or not set(old) <= set(new):
        raise ValueError("new cumulative history omits physical marker columns")
    for name, column in old.items():
        _marker_column_prefix(name, column, new[name], old_count, new_count)
    for handle, count in ((previous, old_count), (current, new_count)):
        for start in range(0, count, CHUNK_ROWS):
            stop = min(start + CHUNK_ROWS, count)
            if not _equal_payload(handle["scba/sequence"][start:stop],
                                  handle["physical_markers/sequence"][start:stop]):
                raise ValueError("physical marker sequence differs from SCBA history")
    return old_count


def _psd_prefix(previous: Any, current: Any, old_count: int, new_count: int) -> int:
    if "psd_history" not in previous:
        raise ValueError("native format 4.0 requires PSD history")
    if "psd_history" not in current:
        raise ValueError("new cumulative history omits PSD history")
    old, new = previous["psd_history"], current["psd_history"]
    if not isinstance(old, h5py.Group) or not isinstance(new, h5py.Group):
        raise ValueError("PSD history must be a native group")
    _annotations(old, new, growing=True)
    if "sequence" not in old or not set(old) <= set(new):
        raise ValueError("new cumulative history omits PSD columns")
    _psd_scalar_columns(old, new, old_count, new_count)
    _psd_sequence(previous, old_count)
    _psd_sequence(current, new_count)
    _psd_selected(old, new)
    return old_count


def verify_cumulative_history(previous: Path, current: Path, identity: dict[str, Any]) -> dict[str, Any]:
    """Both inputs must already be privately captured, hash-checked and parsed.

    Segment provenance alone is insufficient: every previous numerical column is
    compared in bounded chunks, byte for byte, including NaN payload and signed zero.
    """
    try:
        with h5py.File(previous, "r") as old, h5py.File(current, "r") as new:
            if set(old) - {"metadata", "scba", "outer", "psd_history", "physical_markers"}:
                raise ValueError("earlier history has unrecognized evidence; lossless replacement cannot be proven")
            old_sources, old_counts = _sources(old, identity)
            new_sources, new_counts = _sources(new, identity)
            if any(new_sources.get(digest) != value for digest, value in old_sources.items()):
                raise ValueError("new cumulative history omits or changes an earlier source segment")
            for table, old_count in old_counts.items():
                _table_prefix(old[table], new[table], old_count, new_counts[table])
            psd_rows = _psd_prefix(old, new, old_counts["scba"], new_counts["scba"])
            marker_rows = _physical_marker_prefix(old, new, old_counts["scba"], new_counts["scba"])
            return {"schema": "qcl-negf.cumulative-history-proof.v2",
                "previous_rows": old_counts, "replacement_rows": new_counts,
                "previous_psd_rows": psd_rows,
                "previous_physical_marker_rows": marker_rows,
                "previous_source_segments": len(old_sources), "replacement_source_segments": len(new_sources),
                "comparison": "bytewise native columns and unchanged source provenance",
                "previous_cutoff": "first previous_rows entries of each sequence-ordered table"}
    except (OSError, ValueError, KeyError, TypeError, AttributeError, OverflowError) as error:
        raise ContractError(f"cumulative history replacement is not lossless: {error}", "corrupt_result") from error
