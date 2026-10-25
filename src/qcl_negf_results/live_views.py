"""Selected live history: one captured index, closed same-open copies, local HDF only."""

from __future__ import annotations

import math
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from ._capture_io import beneath, copy_closed, digest, fail, integer, small_json
from .commits import json_bytes
from .diagnostic_views import DiagnosticReader, DiagnosticViewLimits, _exact
from .saved_views import _bounded
from .state import _IOBudget

SHA = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class LiveHistoryLimits(DiagnosticViewLimits):
    maximum_segments: int = 16
    maximum_spool_bytes: int = 64 * 1024 * 1024

    def __post_init__(self):
        defaults = LiveHistoryLimits.__dataclass_fields__
        if any(not integer(v, 1, defaults[k].default) for k, v in vars(self).items()):
            raise ValueError("live history limits may only tighten public bounds")


def _selection(value):
    keys = {
        "run_uuid",
        "calcjob_uuid",
        "execution_id",
        "attempt",
        "point_id",
        "source_attempt",
        "plan_fingerprint",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != keys
        or any(not integer(value[k], 1) for k in ("attempt", "source_attempt"))
        or any(
            not isinstance(value[k], str) or not value[k]
            for k in keys - {"attempt", "source_attempt"}
        )
        or not SHA.fullmatch(value["plan_fingerprint"])
    ):
        fail(
            "selection_required", "exact live provider/point/source selection required"
        )
    return dict(value)


def _index(path, selection, limits, budget):
    value, raw = small_json(path, limits.maximum_metadata_bytes, budget=budget)
    identity = value.get("identity")
    if (
        value.get("schema") != "qcl-negf-scientific-history-index-v2"
        or value.get("contract_set") != "qcl-negf.results.v1"
        or not isinstance(identity, dict)
    ):
        fail("unsupported_contract", "closed native history index v2 required")
    expected = {
        "point_id": selection["point_id"],
        "execution_id": selection["execution_id"],
        "attempt": selection["source_attempt"],
        "plan_fingerprint": selection["plan_fingerprint"],
    }
    if set(identity) - set(expected) - {"run_uuid", "calcjob_uuid"}:
        fail("unsupported_contract", "unknown native history identity fields")
    if any(
        type(identity.get(k)) is not type(v) or identity[k] != v
        for k, v in expected.items()
    ):
        fail(
            "identity_mismatch",
            "live native source differs from the selected point attempt",
        )
    for key in ("calcjob_uuid", "run_uuid"):
        if key in identity and identity[key] != selection.get(key):
            fail(
                "identity_mismatch",
                "native provider UUID contradicts exact selected owner",
            )
    rows = value.get("segments")
    if not isinstance(rows, list) or len(rows) > limits.maximum_source_entries:
        fail("budget_exceeded", "live index reference inventory exceeds cap")
    seen = set()
    totals = {"scba": 0, "outer": 0}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "path",
            "bytes",
            "sha256",
            "scba_rows",
            "outer_rows",
        }:
            fail("corrupt_result", "invalid closed history reference")
        name = row["path"]
        beneath(Path(path).parent, name)
        if name in seen or not name.endswith(".h5"):
            fail("corrupt_result", "duplicate/non-history closed leaf")
        seen.add(name)
        if (
            not integer(row["bytes"], 1, limits.maximum_owner_bytes)
            or not isinstance(row["sha256"], str)
            or not SHA.fullmatch(row["sha256"])
        ):
            fail(
                "budget_exceeded",
                "closed history length/digest exceeds public contract",
            )
        for phase in ("scba", "outer"):
            count = row[phase + "_rows"]
            if not integer(count):
                fail("corrupt_result", "invalid declared live row count")
            totals[phase] += count
            if not integer(totals[phase]):
                fail(
                    "budget_exceeded", "live row arithmetic exceeds signed native bound"
                )
    if any(
        value.get(phase + "_rows") != count
        or type(value.get(phase + "_rows")) is not int
        for phase, count in totals.items()
    ):
        fail(
            "corrupt_result", "closed inventory counts differ from native index cutoff"
        )
    if (
        not integer(value.get("generation"), 1)
        or type(value.get("updated_unix")) not in (int, float)
        or not math.isfinite(value["updated_unix"])
    ):
        fail("corrupt_result", "live generation/publication clock unavailable")
    return value, raw


def _base(value, raw, selection, limits, budget):
    return {
        "selection": selection,
        "producer_identity": value["identity"],
        "index_sha256": digest(raw),
        "generation": value["generation"],
        "updated_unix": value["updated_unix"],
        "observed_unix": time.time(),
        "clock_skew_possible": True,
        "cutoff": {k: value[k] for k in ("scba_rows", "outer_rows")},
        "scope": "closed_prefix",
        "access_mode": "selected_closed_copy",
        "limits": vars(limits),
        "coverage": {
            "scientific_acceptance": "not_assessed",
            "freshness": "publication not heartbeat",
        },
        "integrity": {"raw_index_captured": True, "native_validation": "not_performed"},
        "io_counters": budget.counters(),
    }


def describe_live_history(index_path, *, expected_selection, limits=None):
    limits = limits or LiveHistoryLimits()
    selection = _selection(expected_selection)
    if not isinstance(limits, LiveHistoryLimits):
        raise TypeError("limits must be LiveHistoryLimits")
    budget = _IOBudget(limits.maximum_io_bytes)
    value, raw = _index(index_path, selection, limits, budget)
    return _bounded(
        _exact(
            {
                "schema": "qcl-negf.live-history-source.v1",
                **_base(value, raw, selection, limits, budget),
                "segments": value["segments"],
                "phases": [
                    {
                        "phase": phase,
                        "total_rows": value[phase + "_rows"],
                        "columns": None,
                        "reason": "descriptor_does_not_open_closed_payload",
                    }
                    for phase in ("scba", "outer")
                ],
            }
        ),
        limits.maximum_json_bytes,
    )


def read_live_history(
    index_path,
    phase,
    selection,
    *,
    expected_selection,
    expected_index_sha256,
    expected_generation,
    spool,
    limits=None,
):
    limits = limits or LiveHistoryLimits()
    owner = _selection(expected_selection)
    if not isinstance(limits, LiveHistoryLimits):
        raise TypeError("limits must be LiveHistoryLimits")
    budget = _IOBudget(limits.maximum_io_bytes)
    value, raw = _index(index_path, owner, limits, budget)
    if (
        digest(raw) != expected_index_sha256
        or type(expected_generation) is not int
        or value["generation"] != expected_generation
    ):
        fail(
            "source_changed",
            "source_changed: index advanced before selected payload capture",
        )
    if (
        phase not in ("scba", "outer")
        or not isinstance(selection, Mapping)
        or set(selection) != {"start", "stop", "step", "columns"}
    ):
        fail("selection_required", "explicit native phase and row selection required")
    start, stop, step = (selection[k] for k in ("start", "stop", "step"))
    columns = selection["columns"]
    if (
        not all(integer(v) for v in (start, stop, step))
        or not 0 <= start < stop <= value[phase + "_rows"]
        or step < 1
        or not isinstance(columns, list)
        or not 1 <= len(columns) <= limits.maximum_columns
        or any(
            not isinstance(c, str)
            or not 1 <= len(c) <= 128
            or "/" in c
            or "\\" in c
            or any(ord(x) < 32 for x in c)
            for c in columns
        )
        or len(set(columns)) != len(columns)
        or {"sequence", "iteration", "outer_iteration"} & set(columns)
    ):
        fail("selection_required", "invalid explicit live row/value-column selection")
    count = (stop - start - 1) // step + 1
    if count > limits.maximum_rows:
        fail("budget_exceeded", "live row cap exceeded")
    chosen = []
    offset = 0
    for row in value["segments"]:
        end = offset + row[phase + "_rows"]
        first = start + max(0, (offset - start + step - 1) // step) * step
        if first < min(stop, end):
            chosen.append((row, offset, first, min(stop, end)))
        offset = end
    if not chosen or len(chosen) > limits.maximum_segments:
        fail("budget_exceeded", "selected closed segment cap exceeded")
    payload_bytes = sum(row["bytes"] for row, *_ in chosen)
    spool_allowance = payload_bytes + 4096 * len(chosen)
    if spool_allowance > limits.maximum_spool_bytes:
        fail("budget_exceeded", "selected local spool exceeds cap")
    # Reserve spool writes inside the same whole-request I/O ceiling; remaining
    # _IOBudget meters source reads + local integrity/hash/HDF delivery.
    budget.maximum = limits.maximum_io_bytes - spool_allowance
    budget.require(
        2 * payload_bytes
        + limits.maximum_selected_bytes
        + limits.maximum_metadata_bytes
    )
    spool = Path(spool)
    if not spool.is_dir() or spool.is_symlink() or any(spool.iterdir()):
        fail("selection_required", "caller-owned empty local request spool required")
    answers = []
    copied = 0
    adapter_written = 0
    per_metadata = (limits.maximum_metadata_bytes - len(raw)) // len(chosen)
    per_selected = limits.maximum_selected_bytes // len(chosen)
    if per_metadata < 1 or per_selected < 1:
        fail("budget_exceeded", "whole-request selected/metadata allowance exhausted")
    reader_limits = DiagnosticViewLimits(
        maximum_metadata_bytes=per_metadata,
        maximum_owner_bytes=limits.maximum_owner_bytes,
        maximum_io_bytes=limits.maximum_io_bytes,
        maximum_selected_bytes=per_selected,
        maximum_json_bytes=limits.maximum_json_bytes,
        maximum_objects=max(1, limits.maximum_objects // len(chosen)),
        maximum_rows=limits.maximum_rows,
        maximum_columns=limits.maximum_columns,
        maximum_source_entries=limits.maximum_source_entries,
    )
    for number, (row, offset, first, end) in enumerate(chosen):
        local = spool / f"leaf-{number:04d}.h5"
        previous_phase = budget.phase
        try:
            budget.phase = "sha"
            copy_closed(
                beneath(Path(index_path).parent, row["path"]),
                local,
                size=row["bytes"],
                sha256=row["sha256"],
                budget=budget,
            )
        finally:
            budget.phase = previous_phase
        copied += row["bytes"]
        commit = {
            "schema": "qcl-negf.artifact-commit.v2",
            "contract_set": "qcl-negf.results.v1",
            "identity": value["identity"],
            "generation": value["generation"],
            "closed": False,
            "artifacts": [
                {
                    "path": local.name,
                    "bytes": row["bytes"],
                    "sha256": row["sha256"],
                    "role": "science.history",
                    "schema": "qcl-negf-scientific-history-v4",
                    "media_type": "application/x-hdf5",
                    "profile": "science",
                    "dependencies": [],
                }
            ],
        }
        # Private adapter metadata only; never exposed as a native scientific commit.
        metadata = json_bytes(commit)
        if len(metadata) > 4096:
            fail(
                "budget_exceeded",
                "private adapter metadata exceeds reserved spool allowance",
            )
        target = spool / f"leaf-{number:04d}.json"
        target.write_bytes(metadata)
        adapter_written += len(metadata)
        expected = {**value["identity"], "source_commit_sha256": digest(metadata)}
        expected = {
            k: v
            for k, v in expected.items()
            if k
            in {
                "point_id",
                "execution_id",
                "attempt",
                "plan_fingerprint",
                "source_commit_sha256",
                "run_uuid",
                "calcjob_uuid",
            }
        }
        with DiagnosticReader(
            target, expected_identity=expected, limits=reader_limits, budget=budget
        ) as reader:
            answer = reader.read(
                phase,
                {
                    "start": first - offset,
                    "stop": end - offset,
                    "step": step,
                    "columns": columns,
                },
            )
            if (
                next(p for p in reader._phases if p["phase"] == phase)["total_rows"]
                != row[phase + "_rows"]
            ):
                fail(
                    "identity_mismatch",
                    "closed segment native shape differs from index",
                )
            answers.append(answer)
    coords = {
        key: [] if key != "outer_iteration" or phase == "scba" else None
        for key in ("sequence", "iteration", "outer_iteration", "source_attempt")
    }
    values = [
        {"name": name, "units": None, "values": [], "invalid_mask": []}
        for name in columns
    ]
    for answer in answers:
        for key, items in coords.items():
            if items is not None:
                items.extend(answer["coordinates"][key])
        for old, new in zip(values, answer["columns"]):
            if old["values"] and old["units"] != new["units"]:
                fail("identity_mismatch", "native units changed within selected prefix")
            old["units"] = new["units"]
            old["values"].extend(new["values"])
            old["invalid_mask"].extend(new["invalid_mask"])
    if any(items is not None and len(items) != count for items in coords.values()):
        fail("corrupt_result", "selected closed segment row count differs from index")
    breaks = [True] * count
    for i in range(1, count):
        # Unsafe tagged coordinates are retained exactly but cannot be assumed
        # adjacent for plotting. No decimal→float conversion or invented zero.
        prev, now = coords["sequence"][i - 1 : i + 1]
        sequence_ok = type(prev) is int and type(now) is int and now == prev + 1
        iteration_ok = (
            type(coords["iteration"][i - 1]) is int
            and type(coords["iteration"][i]) is int
            and coords["iteration"][i] > coords["iteration"][i - 1]
        )
        invalid = any(c["invalid_mask"][i] or c["invalid_mask"][i - 1] for c in values)
        breaks[i] = (
            step != 1
            or not sequence_ok
            or not iteration_ok
            or invalid
            or coords["source_attempt"][i] != coords["source_attempt"][i - 1]
            or (
                phase == "scba"
                and coords["outer_iteration"][i] != coords["outer_iteration"][i - 1]
            )
        )
    counters = {
        **budget.counters(),
        "maximum_io_bytes": limits.maximum_io_bytes,
        "read_allowance_bytes": budget.maximum,
        "source_copy_bytes": copied,
        "spool_written_bytes": copied + adapter_written,
        "total_read_and_spool_write_bytes": budget.bytes_read_total
        + copied
        + adapter_written,
    }
    answer = {
        "schema": "qcl-negf.live-history-slice.v1",
        **_base(value, raw, owner, limits, budget),
        "phase": phase,
        "source_selection": owner,
        "selection": dict(selection),
        "total_rows": value[phase + "_rows"],
        "coordinates": coords,
        "columns": values,
        "break_before": breaks,
        "io_counters": counters,
        "integrity": {
            "raw_index_captured": True,
            "selected_same_open_copies_verified": True,
            "native_validation": "not_performed",
        },
    }
    return _bounded(_exact(answer), limits.maximum_json_bytes)
