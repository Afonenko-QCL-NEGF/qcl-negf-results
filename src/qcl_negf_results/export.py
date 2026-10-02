"""Role-based, generation-pinned exports with independent container validation.

The exporter never copies a live file using an earlier stat length. It captures
one complete descriptor, validates the committed size/hash and parses the exact
captured bytes before publishing an archive. Scientific arrays are not converted.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable, Mapping
import hashlib
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any

from qcl_negf_contracts.artifacts import (Artifact, CONTRACT_SET, EXPORT_SCHEMA, MODEL_SCHEMA,
    POINTER_SCHEMA, RECOVERY_SCHEMA, digest_value, relative_path, validate_export_receipt,
    require_contract_set, validate_commit)
from qcl_negf_contracts.messages import TERMINAL, MAX_PLAN_BYTES, ContractError, decode, scientific_plan
from .commits import atomic_write, json_bytes, read_json, safe_path
from .catalog import catalog_artifacts
from .native import dataset_blocks as _dataset_blocks, validate_native_handle
from ._atomic_io import fsync_directory

CHUNK_BYTES = 1024 * 1024


@dataclass(frozen=True)
class PinnedCommit:
    path: Path
    value: dict[str, Any]
    sha256: str
    artifacts: tuple[Artifact, ...]


def _pointer(path: Path) -> tuple[Path, str]:
    value, _ = read_json(path)
    require_contract_set(value)
    if value.get("schema") != POINTER_SCHEMA:
        raise ContractError("invalid artifact pointer schema", "corrupt_result")
    return safe_path(path.parent, value["commit_path"]), digest_value(value.get("sha256"))


def _pin(path: Path, expected_sha256: str | None = None) -> PinnedCommit:
    try:
        value, payload = read_json(path)
    except FileNotFoundError as error:
        raise ContractError(f"committed artifact index is missing: {path}", "corrupt_result") from error
    digest = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ContractError("artifact commit hash does not match its pointer", "corrupt_result")
    artifacts = validate_commit(value)
    catalog = value.get("telemetry_catalog")
    if catalog is not None:
        if not isinstance(catalog, dict):
            raise ContractError("invalid telemetry catalog watermark", "corrupt_result")
        artifacts = catalog_artifacts(path.parent, catalog) + artifacts
        if len({item.path for item in artifacts}) != len(artifacts):
            raise ContractError("duplicate expanded telemetry artifact", "corrupt_result")
        paths = {item.path for item in artifacts}
        if any(dependency not in paths for item in artifacts for dependency in item.dependencies):
            raise ContractError("expanded telemetry dependency is absent", "corrupt_result")
    return PinnedCommit(path, value, digest, artifacts)


def _verify_series_reference(point: dict[str, Any], series: dict[str, Any],
                              commit: PinnedCommit) -> dict[str, Any]:
    """Bind this series row to its exact reference, including borrowed checkpoints."""
    identity = commit.value["identity"]
    checks: dict[str, Any] = {}
    expected = {"point_id": point.get("id")}
    if not isinstance(expected["point_id"], str) or not expected["point_id"]:
        raise ContractError("series commit reference requires a point ID", "corrupt_result")
    for key, container in (("execution_id", point), ("plan_fingerprint", series)):
        if key in container:
            expected[key] = container[key]
        else:
            checks[key] = "not_available"
    for key, value in expected.items():
        if identity.get(key) != value:
            raise ContractError(f"series point differs from its referenced commit {key}", "corrupt_result")
        checks[key] = "matched"
    data = point.get("data") or {}
    row_attempt = point.get("attempt")
    source_attempt = data.get("checkpoint_source_attempt", row_attempt)
    if "checkpoint_source_attempt" in data:
        if (point.get("status") != "paused" or data.get("pause_reason") != "resource_pressure"
                or data.get("resume_kind") != "checkpoint"
                or data.get("recovery_origin") != "last_committed_before_resource_pause"
                or type(source_attempt) is not int or type(row_attempt) is not int
                or not 0 < source_attempt < row_attempt):
            raise ContractError("invalid historical checkpoint lineage in series commit reference", "corrupt_result")
        checks["checkpoint_lineage"] = {"source_attempt": source_attempt, "current_attempt": row_attempt,
            "pause_reason": data["pause_reason"], "resume_kind": data["resume_kind"],
            "recovery_origin": data["recovery_origin"]}
    if "attempt" in point:
        if type(row_attempt) is not int or row_attempt < 1 or type(identity.get("attempt")) is not int:
            raise ContractError("series commit reference requires positive integer attempts", "corrupt_result")
        if identity["attempt"] != source_attempt:
            raise ContractError("series point differs from its referenced commit attempt", "corrupt_result")
        checks["attempt"] = "matched"
    else:
        checks["attempt"] = "not_available"
    return checks


def _collect(root: Path) -> tuple[list[PinnedCommit], list[dict[str, Any]], dict[str, Any] | None]:
    """Capture the series once, then follow only explicit committed references."""
    commits: list[PinnedCommit] = []
    coverage: list[dict[str, Any]] = []
    seen: set[Path] = set()
    series = None
    root_pointer = root / "current-commit.json"
    if root_pointer.exists():
        root_path, root_sha = _pointer(root_pointer)
        root_commit = _pin(root_path, root_sha)
        commits.append(root_commit)
        seen.add(root_path)
        indices = [item for item in root_commit.artifacts if item.role == "control.series"]
        if len(indices) != 1:
            raise ContractError("campaign commit must contain exactly one immutable series index", "corrupt_result")
        index_artifact = indices[0]
        index = safe_path(root_commit.path.parent, index_artifact.path)
        series, index_payload = read_json(index)
        if len(index_payload) != index_artifact.size or hashlib.sha256(index_payload).hexdigest() != index_artifact.sha256:
            raise ContractError("committed series index bytes changed", "corrupt_result")
    else:
        index = root / "series_result.json"
        if index.exists():
            series, _ = read_json(index)
    if series is not None:
        require_contract_set(series)
        if series.get("schema") != "qcl-negf-series-result-v3":
            raise ContractError("unsupported scientific series result schema", "incompatible_contract")
        rows = series.get("points", [])
        history = series.get("attempt_history", [])
        if not isinstance(rows, list) or not isinstance(history, list):
            raise ContractError("series points and attempt history must be lists", "corrupt_result")
        for row_number, point in enumerate(rows + history):
            if not isinstance(point, dict):
                raise ContractError("series point must be an object", "corrupt_result")
            data = point.get("data") or {}
            if not isinstance(data, dict):
                raise ContractError("series point data must be an object", "corrupt_result")
            reference = data.get("result_commit")
            if "checkpoint_source_attempt" in data and not reference:
                raise ContractError("historical checkpoint lineage requires a series commit reference", "corrupt_result")
            entry = {key: point[key] for key in ("id", "execution_id", "attempt", "status", "quality", "converged") if key in point}
            entry["series_section"] = "points" if row_number < len(rows) else "attempt_history"
            if reference:
                if isinstance(reference, dict):
                    path = safe_path(root, reference["path"])
                    expected = digest_value(reference.get("sha256"))
                else:
                    path, expected = safe_path(root, reference), None
                commit = _pin(path, expected)
                entry["reference_identity_checks"] = _verify_series_reference(point, series, commit)
                if path not in seen:
                    commits.append(commit)
                    seen.add(path)
                entry["commit_sha256"] = commit.sha256
                entry["source_commit_path"] = str(path.relative_to(root))
                entry["scientific_accepted"] = commit.value.get("scientific_accepted", False)
            else:
                entry["availability"] = "no_committed_physical_record"
            if reference or row_number < len(rows):
                coverage.append(entry)
    # Standalone CLI point and explicit campaign commit use the same contract.
    for pointer in (root / "artifacts" / "current.json",):
        if pointer.exists():
            path, expected = _pointer(pointer)
            if path not in seen:
                commits.append(_pin(path, expected))
                seen.add(path)
    # Performance sessions are explicitly registered after their first commit.
    performance = root / "performance" / "index.json"
    if performance.exists():
        sessions, _ = read_json(performance)
        require_contract_set(sessions)
        if sessions.get("schema") != "qcl-negf.performance-index.v2":
            raise ContractError("invalid performance index", "corrupt_result")
        for session in sessions.get("sessions", []):
            pointer = safe_path(performance.parent, session["pointer"])
            path, expected = _pointer(pointer)
            if path not in seen:
                commits.append(_pin(path, expected))
                seen.add(path)
    # A cheap recovery generation can point to its last complete native analysis.
    # Follow immutable ancestry explicitly; do not search directories by filename.
    for commit in commits:
        parent = commit.value.get("science_parent_commit")
        if parent is None:
            continue
        if not isinstance(parent, dict):
            raise ContractError("science parent must be a committed reference", "corrupt_result")
        parent_path = safe_path(commit.path.parent.parent, relative_path(parent.get("path")))
        expected = digest_value(parent.get("sha256"))
        ancestor = _pin(parent_path, expected)
        if not any(item.role == "physics.analysis" for item in ancestor.artifacts):
            raise ContractError("science parent has no native analysis", "corrupt_result")
        for key in ("point_id", "execution_id", "attempt"):
            if ancestor.value["identity"].get(key) != commit.value["identity"].get(key):
                raise ContractError("science parent belongs to another point attempt", "corrupt_result")
        current_generation = commit.value["generation"]
        prior_generation = ancestor.value["generation"]
        if not isinstance(current_generation, int) or not isinstance(prior_generation, int) or prior_generation >= current_generation:
            raise ContractError("science ancestry must precede its recovery generation", "corrupt_result")
        if parent_path not in seen:
            commits.append(ancestor)
            seen.add(parent_path)
    return commits, coverage, series


def _history_replacements(commits: list[PinnedCommit], profile: str) -> dict[tuple[str, str], tuple[PinnedCommit, Artifact]]:
    """Propose only ancestry-scoped replacements; verify bytes before publication.

    An explicitly depended-on historical payload remains part of its exact closure.
    Full-state preserves every explicitly referenced generation without coalescing.
    """
    replacements: dict[tuple[str, str], tuple[PinnedCommit, Artifact]] = {}
    if profile != "science":
        return replacements
    by_sha = {commit.sha256: commit for commit in commits}
    identity_keys = ("point_id", "execution_id", "attempt", "plan_fingerprint")
    for current in commits:
        histories = [item for item in current.artifacts if item.role == "science.history" and item.included(profile)]
        if not histories or not all(key in current.value["identity"] for key in identity_keys):
            continue
        if len(histories) != 1:
            raise ContractError("cumulative point commit must contain exactly one history", "corrupt_result")
        parent = current.value.get("science_parent_commit")
        while parent is not None:
            ancestor = by_sha.get(parent["sha256"])
            if ancestor is None:
                raise ContractError("pinned science ancestry is incomplete", "corrupt_result")
            if any(ancestor.value["identity"].get(key) != current.value["identity"][key] for key in identity_keys):
                raise ContractError("history replacement crosses scientific point provenance", "corrupt_result")
            required = {dependency for item in ancestor.artifacts if item.included(profile)
                        for dependency in item.dependencies}
            for previous in ancestor.artifacts:
                if previous.role != "science.history" or not previous.included(profile) or previous.path in required:
                    continue
                key = (ancestor.sha256, previous.path)
                candidate = replacements.get(key)
                if candidate is None or candidate[0].value["generation"] < current.value["generation"]:
                    replacements[key] = (current, histories[0])
            parent = ancestor.value.get("science_parent_commit")
    return replacements


def _selection(commit: PinnedCommit, profile: str,
               replacements: dict[tuple[str, str], tuple[PinnedCommit, Artifact]] | None = None) -> tuple[list[Artifact], list[dict[str, Any]]]:
    selected = [item for item in commit.artifacts if item.included(profile)]
    if profile == "science" and commit.value.get("telemetry_catalog"):
        # Preserve the complete committed performance history, including idle
        # intervals needed to distinguish a blocked queue from a slow solver.
        for table in ("resources", "spans"):
            selected.extend(item for item in commit.artifacts if item.role == "performance.full"
                            and item.path.startswith(table + "-"))
    # Selection windows do not reorder the committed catalog. Compaction must
    # preserve the producer's prefix order even when earlier batches were full.
    positions = {item.path: index for index, item in enumerate(commit.artifacts)}
    selected.sort(key=lambda item: positions[item.path])
    selected_paths = {item.path for item in selected}
    for item in selected:
        if not set(item.dependencies) <= selected_paths:
            raise ContractError(f"profile omits a dependency required by {item.path}", "corrupt_result")
    # Hash the canonical excluded stream once, retaining neither its individual
    # rows in the science manifest nor a quadratic membership comparison.
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    digests: dict[tuple[str, str], Any] = {}
    for item in commit.artifacts:
        if item.path in selected_paths:
            continue
        reason = "available_in_full_state" if item.profile == "full-state" else "omitted_by_policy"
        key = (item.role, reason)
        group = groups.setdefault(key, {"role": item.role, "reason": reason,
            "count": 0, "bytes": 0, "first_source_path": item.path,
            "last_source_path": item.path,
            "selection_rule": "not selected by requested scientific role/profile",
            "digest_format": "sha256 of ordered canonical Artifact rows, each followed by LF"})
        group["count"] += 1
        group["bytes"] += item.size
        group["last_source_path"] = item.path
        digests.setdefault(key, hashlib.sha256()).update(json_bytes({
            "path": item.path, "role": item.role, "bytes": item.size, "sha256": item.sha256,
            "media_type": item.media_type, "profile": item.profile,
            "dependencies": list(item.dependencies)}) + b"\n")
    omitted = [{**value, "catalog_digest": digests[key].hexdigest()}
               for key, value in groups.items()]
    for item in list(selected):
        replacement = (replacements or {}).get((commit.sha256, item.path))
        if replacement is not None:
            selected.remove(item)
            omitted.append({"role": item.role, "sha256": item.sha256,
                "source_path": item.path,
                "reason": "superseded_by_verified_cumulative_history",
                "replacement_sha256": replacement[1].sha256,
                "replacement_commit_sha256": replacement[0].sha256,
                "coverage": "all earlier source segments and exact SCBA/outer row prefixes"})
    return selected, omitted


def _validate_json_stream(path: Path) -> None:
    """Parse all native JSON without retaining numerical arrays in memory.

    The buffer is bounded; memory additionally follows nesting, object keys and
    the largest individual scalar. Scientific matrices belong in native HDF5,
    but a large JSON result is valid and must not be silently excluded.
    """
    import ijson  # type: ignore[import-untyped]  # Streaming parser exposes no PEP 561 marker.

    stack: list[set[str] | None] = [set()]
    try:
        with path.open("rb") as stream:
            events = iter(ijson.basic_parse(stream, buf_size=CHUNK_BYTES))
            first = next(events, None)
            if first is None or first[0] != "start_map":
                raise ValueError("expected JSON object")
            for event, value in events:
                _validate_json_event(stack, event, value)
            if stack:
                raise ValueError("incomplete JSON object")
    except (ValueError, OverflowError, ijson.JSONError) as error:
        raise ContractError(f"{path.name}: invalid complete UTF-8 JSON: {error}", "corrupt_result") from error


def _validate_json_event(stack: list[set[str] | None], event: str, value: Any) -> None:
    if event in {"start_map", "start_array"}:
        # Preserve the existing finite-depth JSON schema contract.
        if len(stack) > 64:
            raise ValueError("JSON nesting exceeds 64")
        stack.append(set() if event == "start_map" else None)
    elif event in {"end_map", "end_array"}:
        stack.pop()
    elif event == "map_key":
        keys = stack[-1]
        assert keys is not None
        if value in keys:
            raise ValueError(f"duplicate JSON key {value}")
        keys.add(value)
    elif event == "number" and not isinstance(value, int):
        if not math.isfinite(float(value)):
            raise ValueError("JSON number must be finite")


def _validate_json_contract(path: Path, declared_schema: str | None) -> None:
    if declared_schema is None:
        return
    value, _ = read_json(path)
    require_contract_set(value)
    if value.get("schema") != declared_schema:
        raise ContractError("JSON schema differs from committed artifact schema", "incompatible_contract")
    if declared_schema == "qcl-negf-operator-diagnostics-v4" and value.get("schema_version") != "4.0":
        raise ContractError("operator diagnostics require native format 4.0", "incompatible_contract")
    if declared_schema == MODEL_SCHEMA:
        from .model import validate_model
        validate_model(value)
    if declared_schema == RECOVERY_SCHEMA:
        payload = value.get("payload")
        if not isinstance(payload, dict) or payload.get("path") != "physics.h5":
            raise ContractError("invalid canonical recovery reference", "corrupt_result")
        digest_value(payload.get("sha256"))
        if not isinstance(payload.get("bytes"), int) or payload["bytes"] < 0:
            raise ContractError("invalid recovery payload size", "corrupt_result")


def _validate_container(path: Path, media_type: str, *, role: str | None = None,
                        declared_schema: str | None = None) -> None:
    if media_type == "application/json":
        _validate_json_stream(path)
        _validate_json_contract(path, declared_schema)
    elif media_type == "application/x-hdf5":
        import h5py
        try:
            with h5py.File(path, "r") as handle:
                validate_native_handle(handle, role, declared_schema)
                seen: set[int] = set()
                def validate_group(group: Any) -> None:
                    address = h5py.h5o.get_info(group.id).addr
                    if address in seen:
                        return
                    seen.add(address)
                    for name in group:
                        link = group.get(name, getlink=True)
                        if isinstance(link, h5py.ExternalLink):
                            raise ContractError("external HDF5 links are not portable", "corrupt_result")
                        try:
                            child = group[name]
                        except KeyError as error:
                            raise ContractError("dangling HDF5 link", "corrupt_result") from error
                        if isinstance(child, h5py.Group):
                            validate_group(child)
                        elif isinstance(child, h5py.Dataset):
                            # Full payload readability, including every compressed chunk.
                            if child.shape is None or child.size == 0:
                                continue
                            if child.shape == ():
                                child[()]
                            else:
                                for block in _dataset_blocks(child):
                                    child[block]
                validate_group(handle)
        except (OSError, ValueError) as error:
            raise ContractError(f"invalid HDF5 artifact: {error}", "corrupt_result") from error
    elif media_type == "application/vnd.apache.parquet":
        import pyarrow.parquet as pq
        try:
            parquet = pq.ParquetFile(path)
            from .compaction import validate_performance_schema
            validate_performance_schema(parquet.schema_arrow)
            rows = sum(batch.num_rows for batch in parquet.iter_batches(batch_size=4096))
            if rows != parquet.metadata.num_rows:
                raise ContractError("Parquet row count mismatch", "corrupt_result")
        except (OSError, ValueError) as error:
            raise ContractError(f"invalid closed Parquet artifact: {error}", "corrupt_result") from error
    elif media_type in {"text/plain", "text/markdown"}:
        with path.open("r", encoding="utf-8") as stream:
            while stream.read(CHUNK_BYTES):
                pass
    else:
        raise ContractError(f"no parser for included artifact media type: {media_type}", "incompatible_contract")


def _capture(source: Path, destination: Path, artifact: Artifact,
             on_bytes: Callable[[int], None] | None = None) -> None:
    """Pin bytes to a private spool and reject any mismatch, never truncate."""
    count = 0
    digest = hashlib.sha256()
    try:
        descriptor = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(descriptor, "rb") as stream, destination.open("xb") as captured:
            while chunk := stream.read(CHUNK_BYTES):
                count += len(chunk)
                if count > artifact.size:
                    raise ContractError("artifact grew beyond committed byte length", "corrupt_result")
                captured.write(chunk)
                digest.update(chunk)
                if on_bytes is not None:
                    on_bytes(count)
    except FileNotFoundError as error:
        raise ContractError("committed artifact disappeared before capture", "corrupt_result") from error
    if count != artifact.size or digest.hexdigest() != artifact.sha256:
        raise ContractError("artifact bytes do not match immutable commit", "corrupt_result")
    _validate_container(destination, artifact.media_type, role=artifact.role, declared_schema=artifact.schema)


def _verify_recovery_closure(commit: PinnedCommit, artifact: Artifact, captured: Path) -> None:
    if artifact.schema != RECOVERY_SCHEMA:
        return
    value, _ = read_json(captured)
    payload = value["payload"]
    physical = [item for item in commit.artifacts if item.role == "physics.full" and item.path == payload["path"]]
    if len(physical) != 1 or artifact.dependencies != (payload["path"],):
        raise ContractError("recovery reference has no exact canonical dependency", "corrupt_result")
    if payload != {"path": physical[0].path, "sha256": physical[0].sha256, "bytes": physical[0].size}:
        raise ContractError("recovery reference differs from its committed canonical payload", "corrupt_result")


def _check_profile(profile: str) -> None:
    if profile not in {"science", "full-state"}:
        raise ContractError("profile must be science or full-state")


def _telemetry_freshness(commit: PinnedCommit, spool: Path,
                         selected: list[tuple[PinnedCommit, Artifact]], cutoff: float) -> list[dict[str, Any]]:
    """Report only the committed telemetry clock, never an inferred live tail."""
    telemetry: list[dict[str, Any]] = []
    if commit.value.get("telemetry_catalog") is not None:
        published = commit.value.get("published_unix")
        summaries = [artifact for owner, artifact in selected
                     if owner.sha256 == commit.sha256 and artifact.role == "performance.summary"
                     and artifact.media_type == "application/json"]
        for artifact in summaries:
            summary, _ = read_json(spool / (artifact.sha256 + ".json"))
            if "recorded_until_monotonic_ns" not in summary:
                continue  # Hardware passport is not a temporal observation.
            telemetry.append({
                "session_id": summary.get("session_id"), "generation": commit.value["generation"],
                "published_unix": published,
                "committed_age_seconds": cutoff - published if isinstance(published, (int, float)) and published <= cutoff else None,
                "closed": summary.get("closed"),
                "maximum_buffer_seconds": summary.get("durability", {}).get("maximum_buffer_seconds"),
                "pending_rows": None,
                "tail_scope": "uncommitted rows excluded; unknown to this snapshot observer",
                "unpublished_resource_tail_at_risk": summary.get("uncommitted_resource_batch_at_risk"),
            })
    return telemetry


def _freshness(commits: list[PinnedCommit], spool: Path,
               selected: list[tuple[PinnedCommit, Artifact]], cutoff: float) -> dict[str, Any]:
    """Report source clocks separately; a newer recovery never dates old analysis."""
    import h5py

    records = []
    telemetry = []
    for commit in commits:
        telemetry.extend(_telemetry_freshness(commit, spool, selected, cutoff))
        native = [(owner, artifact) for owner, artifact in selected
                  if owner.sha256 == commit.sha256 and artifact.role in {"physics.analysis", "science.history"}]
        if not native:
            continue
        published = commit.value.get("published_unix")
        row: dict[str, Any] = {"identity": commit.value["identity"],
            "source_commit_sha256": commit.sha256, "generation": commit.value["generation"],
            "published_unix": published, "analysis_unix": None, "analysis_age_seconds": None,
            "analysis_present": False, "analysis_coordinates": None, "last_history": None}
        for _, artifact in native:
            if artifact.role == "physics.analysis":
                row["analysis_present"] = True
                row["analysis_unix"] = published
                row["analysis_coordinates"] = commit.value.get("analysis_coordinates")
                if isinstance(published, (int, float)) and not isinstance(published, bool) and published <= cutoff:
                    row["analysis_age_seconds"] = cutoff - published
            elif artifact.media_type == "application/x-hdf5":
                with h5py.File(spool / (artifact.sha256 + ".h5"), "r") as handle:
                    groups = {}
                    for phase in ("scba", "outer"):
                        if phase not in handle or not isinstance(handle[phase], h5py.Group):
                            continue
                        group = handle[phase]
                        groups[phase] = {name: int(group[name][-1]) for name in (
                            "sequence", "iteration", "outer_iteration")
                            if name in group and group[name].shape and len(group[name])}
                    row["last_history"] = groups or None
        records.append(row)
    return {"cutoff_unix": cutoff, "records": records, "telemetry": telemetry,
        "missing_timestamp_policy": "unknown; filesystem mtime and checkpoint time never substitute analysis time",
        "history_timestamp_policy": "native last row coordinates; UTC unavailable unless explicitly recorded"}


def _cached_receipt(destination: Path, identity: str) -> dict[str, Any] | None:
    receipt_path = destination / f"{identity}.json"
    if not receipt_path.exists():
        return None
    receipt, _ = read_json(receipt_path)
    validate_export_receipt(receipt)
    if receipt["snapshot_identity"] != identity:
        raise ContractError("cached receipt identifies another snapshot", "corrupt_result")
    existing = destination / receipt["archive"]
    if not existing.exists() or existing.stat().st_size != receipt["bytes"]:
        return None
    with existing.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != receipt["sha256"]:
            return None
    return receipt


def _download_filename(label: str, profile: str, identity: str) -> str:
    safe = "".join(char if char.isascii() and (char.isalnum() or char in "-_") else "_" for char in label)
    return f"{safe[:100] or 'snapshot'}-{profile}-{identity[:12]}.tar.xz"


def _frozen_plan(plan: Mapping[str, Any] | bytes | None, source: str | None
                 ) -> tuple[dict[str, Any] | None, bytes | None, dict[str, Any] | None]:
    if plan is None:
        if source is not None:
            raise ContractError("frozen scientific plan source requires plan bytes", "corrupt_result")
        return None, None, None
    if not isinstance(plan, (bytes, Mapping)):
        raise ContractError("frozen scientific plan must be exact bytes or a mapping", "corrupt_result")
    source = source if source is not None else ("caller.bytes" if isinstance(plan, bytes) else "caller.mapping")
    if (not isinstance(source, str) or not source or len(source) > 512
            or any(ord(char) < 32 or ord(char) == 127 for char in source)):
        raise ContractError("invalid frozen scientific plan source", "corrupt_result")
    from jsonschema import ValidationError
    from qcl_negf_contracts import schema_validator
    try:
        payload = plan if isinstance(plan, bytes) else json_bytes(dict(plan))
        value = scientific_plan(decode(payload, maximum=MAX_PLAN_BYTES))
        schema_validator("scientific-plan.schema.json").validate(value)
    except (ContractError, ValidationError) as error:
        raise ContractError(f"invalid frozen scientific plan: {str(error)[:500]}", "corrupt_result") from error
    return value, payload, {"path": "plan.json", "schema": value["schema"], "source": source,
        "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload),
        "fingerprint": value["fingerprint"], "scientific_fingerprint": value["scientific_fingerprint"],
        "model_revision": value["model_revision"]}


def _verify_plan_identity(plan: dict[str, Any], commits: list[PinnedCommit],
                          series: dict[str, Any] | None) -> list[str]:
    """Compare saved identities; Julia's canonical fingerprints are not recomputed here."""
    verified: set[str] = set()
    executions = {item["id"]: item for item in plan["executions"]}
    points = {item["id"]: item for item in plan["points"]}
    for point in points.values():
        if point["execution_id"] not in executions:
            raise ContractError("frozen plan point identifies an unknown execution", "corrupt_result")
    for identifier, execution in executions.items():
        if set(execution["point_ids"]) != {key for key, point in points.items() if point["execution_id"] == identifier}:
            raise ContractError("frozen plan execution point membership differs", "corrupt_result")

    def compare(container: Mapping[str, Any], key: str, expected: Any, locator: str) -> None:
        if key in container:
            if container[key] != expected:
                raise ContractError(f"frozen plan identity differs from {locator}.{key}", "corrupt_result")
            verified.add(f"{locator}.{key}")

    def point_identity(identity: Mapping[str, Any], locator: str) -> dict[str, Any] | None:
        identifier = identity.get("point_id", identity.get("id"))
        point = points.get(identifier) if isinstance(identifier, str) else None
        if identifier is not None and point is None:
            raise ContractError(f"frozen plan does not contain {locator} point", "corrupt_result")
        execution = identity.get("execution_id")
        if execution is not None and (execution not in executions or
                point is not None and execution != point["execution_id"]):
            raise ContractError(f"frozen plan differs from {locator} execution", "corrupt_result")
        if point is not None:
            verified.add(f"{locator}.point_id")
        if execution is not None:
            verified.add(f"{locator}.execution_id")
        return point

    if series is not None:
        for key, expected in (("plan_fingerprint", plan["fingerprint"]),
                              ("plan_scientific_fingerprint", plan["scientific_fingerprint"]),
                              ("root_definition_id", plan["root_definition_id"])):
            compare(series, key, expected, "series")
        selected = series.get("selected_execution_id")
        if selected is not None and selected not in executions:
            raise ContractError("frozen plan does not contain selected execution", "corrupt_result")
        for row in series.get("points", []) + series.get("attempt_history", []):
            point = point_identity(row, "series.point")
            if point is not None and "coordinates" in row:
                coordinates = row["coordinates"]
                if not isinstance(coordinates, dict):
                    raise ContractError("frozen plan requires saved point coordinates to be an object", "corrupt_result")
                for key in ("temperature_K", "voltage_per_period_V", "branch", "order"):
                    compare(coordinates, key, point[key], "series.point.coordinates")
    for commit in commits:
        identity = commit.value["identity"]
        compare(identity, "plan_fingerprint", plan["fingerprint"], "commit")
        compare(identity, "plan_scientific_fingerprint", plan["scientific_fingerprint"], "commit")
        point_identity(identity, "commit")
    return sorted(verified)


def export_snapshot(job_root: Path, destination: Path, *, profile: str = "science",
                    job_id: str | None = None, job_status: str = "running",
                    plan: Mapping[str, Any] | bytes | None = None,
                    plan_source: str | None = None,
                    progress: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    operation_started = time.monotonic()
    captured_unix = time.time()
    phase_started = operation_started
    current_phase: str | None = None
    phase_seconds: dict[str, float] = {}

    def report(phase: str, **counts: Any) -> None:
        nonlocal phase_started, current_phase
        now = time.monotonic()
        if phase != current_phase:
            if current_phase is not None:
                phase_seconds[current_phase] = phase_seconds.get(current_phase, 0.0) + now - phase_started
            phase_started, current_phase = now, phase
        if progress is not None:
            progress({"phase": phase, "elapsed_seconds": now - operation_started,
                      "phase_seconds": dict(phase_seconds), **counts})

    _check_profile(profile)
    plan_value, plan_payload, plan_metadata = _frozen_plan(plan, plan_source)
    root, destination = Path(job_root).resolve(), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    report("collecting")
    commits, coverage, series = _collect(root)
    if plan_value is not None:
        assert plan_metadata is not None
        plan_metadata["identity_verified_against"] = _verify_plan_identity(plan_value, commits, series)
    captured_unix = time.time()  # The cutoff describes the now fully pinned reference set.
    replacements = _history_replacements(commits, profile)
    selected: list[tuple[PinnedCommit, Artifact]] = []
    records: list[dict[str, Any]] = []
    for commit in commits:
        artifacts, omitted = _selection(commit, profile, replacements)
        selected.extend((commit, item) for item in artifacts)
        records.append({"source_commit_sha256": commit.sha256,
            "identity": commit.value["identity"], "generation": commit.value["generation"],
            "scientific_accepted": commit.value.get("scientific_accepted"),
            "terminal_status": commit.value.get("terminal_status"),
            "quality": commit.value.get("quality"),
            "science_parent_sha256": (commit.value.get("science_parent_commit") or {}).get("sha256"),
            "telemetry_catalog_watermark": commit.value.get("telemetry_catalog"),
            "included": [], "omitted_by_policy": omitted})
    object_contracts: dict[str, tuple[int, str]] = {}
    for _, item in selected:
        contract = (item.size, item.media_type)
        if item.sha256 in object_contracts and object_contracts[item.sha256] != contract:
            raise ContractError("one object digest has conflicting size or media type", "corrupt_result")
        object_contracts[item.sha256] = contract
    unique_size = sum(size for size, _ in object_contracts.values())
    # This identity pins the committed prefix. A later producer generation is a new export.
    compressor: dict[str, Any] = {"format": "xz", "preset": 1,
        "extreme": False, "check": "CRC64",
        "tar_format": "pax",
        "implementation": "Python lzma / liblzma"}
    from .witness_selection import POLICY as witness_policy
    derivation = {"telemetry": "qcl-negf.telemetry-compaction-proof.v1",
                  "witnesses": witness_policy if profile == "science" else "unchanged source evidence"}
    identity = hashlib.sha256(json_bytes({"schema": EXPORT_SCHEMA, "contract_set": CONTRACT_SET, "profile": profile,
        "commits": [item.sha256 for item in commits], "coverage": coverage,
        "native_format": "4.0", "history_closure": "verified-cumulative-physical-markers-psd-v4", "inventory": "complete-performance-v2",
        "derivation": derivation, "exporter_revision": 4,
        "size_policy": "single-complete-archive-v1",
        "compressor": compressor,
        "plan": plan_metadata, "job_id": job_id or root.name, "job_status": job_status})).hexdigest()
    receipt_path = destination / f"{identity}.json"
    cached = _cached_receipt(destination, identity)
    if cached is not None:
        report("cached", completed_bytes=cached["bytes"], total_bytes=cached["bytes"])
        return cached
    with tempfile.TemporaryDirectory(prefix=".export-pin-", dir=destination) as directory:
        spool = Path(directory)
        files: dict[str, tuple[Path, dict[str, Any]]] = {}
        record_map = {record["source_commit_sha256"]: record for record in records}
        captured_bytes = 0
        report("capturing", completed_bytes=0, total_bytes=unique_size,
               completed_files=0, total_files=len(object_contracts))
        for commit, artifact in selected:
            suffix = {"application/x-hdf5": ".h5", "application/vnd.apache.parquet": ".parquet",
                      "application/json": ".json", "text/markdown": ".md", "text/plain": ".txt"}[artifact.media_type]
            name = f"objects/{artifact.sha256}{suffix}"
            if name not in files:
                target = spool / (artifact.sha256 + suffix)
                _capture(safe_path(commit.path.parent, artifact.path), target, artifact,
                    lambda count: report("capturing", completed_bytes=captured_bytes + count,
                        total_bytes=unique_size, completed_files=len(files), total_files=len(object_contracts)))
                files[name] = (target, {"path": name, "bytes": artifact.size,
                    "sha256": artifact.sha256, "media_type": artifact.media_type})
                captured_bytes += artifact.size
                report("capturing", completed_bytes=captured_bytes, total_bytes=unique_size,
                       completed_files=len(files), total_files=len(object_contracts))
            _verify_recovery_closure(commit, artifact, files[name][0])
            window = profile == "science" and artifact.role == "performance.full"
            record_map[commit.sha256]["included"].append({
                "role": "performance.window" if window else artifact.role,
                **({"source_role": artifact.role, "selection": "all committed segments per table"} if window else {}),
                "object": name,
                "source_path": artifact.path, "sha256": artifact.sha256,
                **({"schema": artifact.schema} if artifact.schema is not None else {}),
                "dependencies": [next(f"objects/{item.sha256}" + {
                    "application/x-hdf5": ".h5", "application/vnd.apache.parquet": ".parquet",
                    "application/json": ".json", "text/plain": ".txt", "text/markdown": ".md"}[item.media_type]
                    for item in commit.artifacts if item.path == dep) for dep in artifact.dependencies]})
        if replacements:
            report("verifying_history", completed_files=0, total_files=len(replacements))
            from .history import verify_cumulative_history
            by_sha = {commit.sha256: commit for commit in commits}
            for proof_index, ((old_sha, old_path), (new_commit, new_artifact)) in enumerate(replacements.items(), 1):
                old_commit = by_sha[old_sha]
                old_artifact = next(item for item in old_commit.artifacts if item.path == old_path)
                if old_artifact.media_type != "application/x-hdf5" or new_artifact.media_type != "application/x-hdf5":
                    raise ContractError("cumulative history replacement requires native HDF5", "corrupt_result")
                old_capture = spool / ("history-proof-" + old_artifact.sha256 + ".h5")
                if not old_capture.exists():
                    _capture(safe_path(old_commit.path.parent, old_artifact.path), old_capture, old_artifact)
                new_capture = spool / (new_artifact.sha256 + ".h5")
                proof = verify_cumulative_history(old_capture, new_capture, new_commit.value["identity"])
                proof["applies_to"] = "captured cumulative source before declared science-export witness selection"
                for omission in record_map[old_sha]["omitted_by_policy"]:
                    if omission.get("source_path") == old_path:
                        omission["replacement_verification"] = proof
                report("verifying_history", completed_files=proof_index, total_files=len(replacements))
        from .compaction import compact_export_records
        if any(item.media_type == "application/vnd.apache.parquet" for _, item in selected):
            report("compacting_telemetry")
        compaction_proofs = compact_export_records(records, files, spool)
        from .witness_selection import select_export_witnesses
        witness_selection = select_export_witnesses(records, files, spool) if profile == "science" else []
        report("validating_closure")
        all_objects = set(files)
        if any(dep not in all_objects for record in records for item in record["included"] for dep in item["dependencies"]):
            raise ContractError("export closure contains a dangling object reference", "corrupt_result")
        missing = [item for item in coverage if item.get("availability") == "no_committed_physical_record"]
        job_complete = job_status in TERMINAL
        has_science = any(item.role in {"physics.analysis", "science.comparison"} for _, item in selected)
        if not has_science:
            missing.append({"availability": "no_committed_scientific_records"})
        manifest = {"schema": EXPORT_SCHEMA, "contract_set": CONTRACT_SET, "job_id": job_id or root.name, "profile": profile,
            "snapshot_identity": identity, "job_status": job_status,
            "job_complete": job_complete, "snapshot_consistent": True,
            "complete": job_complete and not missing, "scientific_accepted": None,
            "capabilities": (["stored_observables", "scientific_diagnostics", "complete_committed_performance"]
                if profile == "science" else ["full_physical_state", "declared_recovery_capabilities"]),
            "requires_full_state": ["arbitrary_new_optical_response", "exact_restart"] if profile == "science" else [],
            "cutoff": "exact committed records named below; active uncommitted work is absent",
            "records": records, "coverage": coverage, "missing_records": missing,
            "telemetry_compaction": compaction_proofs,
            "derivation": derivation,
            "witness_selection": witness_selection,
            "termination": {"status": job_status, "scope": "series", "cause": (series or {}).get("stop_reason"),
                            "cause_observed": (series or {}).get("stop_reason") is not None},
            "completeness": {"series_terminal": job_complete, "all_planned_points_recorded": not missing,
                             "committed_cut_verified": True, "uncommitted_tail": "unknown"},
            "files": [value[1] for _, value in sorted(files.items())],
            "captured_unix": captured_unix,
            "freshness": _freshness(commits, spool, selected, captured_unix),
            "compressor": compressor,
            "size_policy": {"measurement": "actual compressed archive including container metadata",
                            "scope": "one complete archive, all profiles",
                            "archive_byte_limit": None, "storage_failure": "fail without publishing a receipt"}}
        if plan_metadata is not None:
            manifest["frozen_plan"] = plan_metadata
        metadata: dict[str, bytes] = {"manifest.json": json_bytes(manifest),
            "README.md": ("# QCLNEGF scientific snapshot\n\n"
                f"Profile: {profile}. Job complete: {job_complete}. Snapshot consistent: true.\n\n"
                "Objects are immutable, byte-verified and independently parsed before publication.\n"
                "manifest.json maps scientific roles and native numerical sources to content-addressed files.\n"
                "The science profile supports declared analyses, not arbitrary new full-matrix optics or exact restart.\n"
                "Scientific acceptance is explicit per record; successful export does not imply convergence.\n\n"
                "One .tar.xz contains the complete selected committed snapshot.\n"
                "export-index.json describes whole native objects and metadata with their hashes.\n"
                "Verify: python -m qcl_negf_results.archive verify ARCHIVE --receipt RECEIPT\n"
                "Restore native files: python -m qcl_negf_results.archive reassemble --destination recovered ARCHIVE\n"
                "Keep the receipt to additionally verify the finalized archive hash and snapshot identity.\n"
                "The receiver also supports legacy multipart sets.\n").encode()}
        from .diagnostic_page import summary as diagnostic_summary
        metadata["diagnostics.json"] = diagnostic_summary(manifest, files)
        if plan_payload is not None:
            metadata["plan.json"] = plan_payload
        metadata_bytes = sum(map(len, metadata.values()))
        raw_bytes = sum(path.stat().st_size for path, _ in files.values()) + metadata_bytes
        from .archive import build_archive, verify_archive
        started = time.monotonic()
        temporary_archive, archive_info, transport_index = build_archive(spool, files, metadata,
            identity=identity, profile=profile, preset=int(compressor["preset"]), report=report)
        compression_seconds = time.monotonic() - started
        # Readback has its own inventory parser and checks every complete member
        # plus the XZ footer; source containers and object closure were checked above.
        verified_index = verify_archive(temporary_archive, report=report)
        if verified_index != transport_index:
            raise ContractError("final archive inventory changed", "corrupt_result")
        transport_metadata_bytes = len(json_bytes(transport_index))
        report("publishing")
        digest, size = archive_info["sha256"], archive_info["bytes"]
        archive_path = destination / f"{digest}.tar.xz"
        receipt = {"schema": EXPORT_SCHEMA, "contract_set": CONTRACT_SET,
            "snapshot_id": digest, "id": job_id or root.name, "profile": profile,
            "sha256": digest, "bytes": size, "payload_bytes": raw_bytes + transport_metadata_bytes,
            "source_payload_bytes": raw_bytes, "transport_metadata_bytes": transport_metadata_bytes,
            "total_archive_bytes": size, "transport_schema": transport_index["schema"],
            "filename": _download_filename(job_id or root.name, profile, identity),
            "complete": manifest["complete"], "job_complete": job_complete,
            "snapshot_consistent": True, "snapshot_identity": identity,
            "export_seconds": time.monotonic() - operation_started,
            "compression_seconds": compression_seconds,
            "phase_seconds": {**phase_seconds, "publishing": time.monotonic() - phase_started},
            "freshness": manifest["freshness"], "manifest_bytes": len(metadata["manifest.json"]),
            "captured_unix": captured_unix, "committed_records": len(commits),
            "compressor": manifest["compressor"], "size_policy": manifest["size_policy"],
            "archive": archive_path.name}
        validate_export_receipt(receipt)
        temporary_receipt = spool / "receipt.json"
        atomic_write(temporary_receipt, json_bytes(receipt))
        os.chmod(temporary_archive, 0o640)
        archive_existed = archive_path.exists()
        previous_receipt = receipt_path.read_bytes() if receipt_path.exists() else None
        receipt_published = False
        os.replace(temporary_archive, archive_path)
        try:
            os.replace(temporary_receipt, receipt_path)
            receipt_published = True
            fsync_directory(destination)
        except BaseException:
            try:
                if receipt_published:
                    if previous_receipt is None:
                        receipt_path.unlink(missing_ok=True)
                    else:
                        atomic_write(receipt_path, previous_receipt)
            finally:
                if not archive_existed:
                    archive_path.unlink(missing_ok=True)
            raise
        report("completed", completed_bytes=size, total_bytes=size)
        return receipt


def preview(job_root: Path, *, profile: str = "science", job_id: str | None = None,
            job_status: str = "running") -> dict[str, Any]:
    """Inspect committed metadata without reading numerical payloads.

    Availability is not an integrity verdict: export_snapshot validates each
    payload before publishing a receipt. The caller supplies scheduler state.
    """
    _check_profile(profile)
    commits, coverage, _ = _collect(Path(job_root).resolve())
    artifacts = [item for commit in commits for item in _selection(commit, profile)[0]]
    return {"id": job_id, "profile": profile,
            "files": len({item.sha256 for item in artifacts}),
            "bytes": sum({item.sha256: item.size for item in artifacts}.values()),
            "complete": job_status in TERMINAL, "committed_records": len(commits),
            "selected_roles": sorted({item.role for item in artifacts}),
            "coverage": coverage, "transport_schema": "qcl-negf.export-archive.v1"}
