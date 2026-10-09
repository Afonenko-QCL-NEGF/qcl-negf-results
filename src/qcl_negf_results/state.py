"""Verified immutable state access with bounded HDF5 hyperslabs.

Integrity verification streams each file once; it never deserializes solver
arrays. Keep a reader open for multiple block reads to avoid repeated hashing.
Receipts establish local publication and integrity, not scientific acceptance
or remote storage availability.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import h5py

from qcl_negf_contracts.artifacts import Artifact, validate_commit, validate_execution_progress
from qcl_negf_contracts.messages import ContractError
from .commits import read_json, safe_path
from .native import validate_native_handle
from .provenance import _identity_fields, _json_scalar, _state_identity


def _fail(message: str) -> None:
    raise ContractError(message, "corrupt_result")


def _verify_stream(stream: Any, size: int, digest: str) -> None:
    stream.seek(0, 2)
    if stream.tell() != size:
        _fail("state payload byte length differs from its commit")
    stream.seek(0)
    if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
        _fail("state payload checksum differs from its commit")
    stream.seek(0)


def _receipt(root: Path, commit: Mapping[str, Any], payload: bytes) -> dict[str, Any]:
    receipt, _ = read_json(safe_path(root, "receipt.json"), maximum=1024 * 1024)
    if (receipt.get("schema") != "qcl-negf-recovery-receipt-v1"
            or receipt.get("status") != "verified"
            or receipt.get("publication_scope") != "local_filesystem"):
        _fail("unsupported recovery publication receipt")
    if receipt.get("commit_sha256") != hashlib.sha256(payload).hexdigest():
        _fail("recovery receipt commit checksum differs")
    for field in ("identity", "state_id", "state_sequence"):
        if field not in commit or receipt.get(field) != commit[field]:
            _fail(f"recovery receipt {field} differs from its commit")
    return receipt


def _same_json_value(left: Any, right: Any) -> bool:
    """Compare decoded JSON values without Python's bool/int/float coercion."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _same_json_value(left[key], right[key]) for key in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _same_json_value(a, b) for a, b in zip(left, right))
    return left == right


def _prior_declarations(files: list[Mapping[str, Any]], owned: tuple[Artifact, ...],
                        identity: Mapping[str, Any], locator: str) -> tuple[dict[str, Any], ...]:
    """Normalize one validated final's declarations, preserving both SHA obligations.

    This metadata preflight grants no byte lifetime or reusable verification proof.
    The caller must still run the original index, owner and native passes.
    """
    declarations = {row["path"]: {**row, "origins": ("index",)} for row in files}
    for owner in owned:
        row = declarations.get(owner.path)
        if row is None:
            row = {"path": owner.path, "bytes": owner.size, "sha256": owner.sha256,
                   "origins": ("owner",)}
            declarations[owner.path] = row
        else:
            claims = {"bytes": owner.size, "sha256": owner.sha256, "role": owner.role,
                      "schema": owner.schema, "media_type": owner.media_type,
                      "identity": identity}
            for field, expected in claims.items():
                if field not in row:
                    continue
                matches = (_same_json_value(row[field], expected) if field == "identity"
                           else row[field] == expected)
                if not matches:
                    _fail(f"prior final {locator}: {owner.path} {field} declaration conflict")
            row["origins"] += ("owner",)
        row["owner"] = owner
    return tuple(declarations.values())


def verify_recovery_bundle(directory: str | Path,
                           expected_identity: Mapping[str, Any] | None = None, *,
                           archive_directory: str | Path | None = None) -> dict[str, Any]:
    """Verify every declared dependency and receipt, without running Julia.

    The caller owns compatibility decisions for application/model contracts.
    A successful checksum verification alone does not certify resumability.
    """
    root = Path(directory)
    commit, payload = read_json(safe_path(root, "commit.json"), maximum=16 * 1024 * 1024)
    artifacts = validate_commit(commit)
    if expected_identity is not None and any(
            commit["identity"].get(key) != value for key, value in expected_identity.items()):
        _fail("recovery bundle identity differs from the selected attempt")
    receipt = _receipt(root, commit, payload)
    for artifact in artifacts:
        with safe_path(root, artifact.path).open("rb") as stream:
            _verify_stream(stream, artifact.size, artifact.sha256)
            if artifact.media_type == "application/x-hdf5":
                with h5py.File(stream, "r") as handle:
                    _internal_storage(handle)
                    validate_native_handle(handle, artifact.role, artifact.schema)
    prior_finals = 0
    for artifact in artifacts:
        if artifact.role != "execution.progress":
            continue
        progress, _ = read_json(safe_path(root, artifact.path), maximum=16 * 1024 * 1024)
        entries = validate_execution_progress(progress)
        if progress["identity"] != commit["identity"]:
            _fail("execution progress state identity differs from its commit")
        if entries and archive_directory is None:
            _fail("recovery requires its prior archive dependencies")
        for entry in entries:
            final_path = safe_path(Path(archive_directory), entry["final_commit"])
            final, final_bytes = read_json(final_path, maximum=16 * 1024 * 1024)
            owned = validate_commit(final)
            if final.get("storage_class") != "archive" or not {"physics.full", "model", "science.history"} <= {item.role for item in owned}:
                _fail("prior final must own full state, resolved model and history in the scientific archive")
            proof = entry["receipt"]
            if hashlib.sha256(final_bytes).hexdigest() != proof["commit_sha256"]:
                _fail("prior archive commit checksum differs from progress")
            for field in ("identity", "state_id", "state_sequence"):
                if final.get(field) != proof.get(field):
                    _fail("prior archive identity differs from progress")
            _receipt(final_path.parent, final, final_bytes)
            _prior_declarations(entry["files"], owned, final["identity"], entry["final_commit"])
            for dependency in (*entry["files"], *({"path": item.path, "bytes": item.size,
                    "sha256": item.sha256} for item in owned)):
                with safe_path(final_path.parent, dependency["path"]).open("rb") as stream:
                    _verify_stream(stream, dependency["bytes"], dependency["sha256"])
            for item in owned:
                if item.media_type == "application/x-hdf5":
                    with h5py.File(safe_path(final_path.parent, item.path), "r") as handle:
                        _internal_storage(handle)
                        validate_native_handle(handle, item.role, item.schema)
            prior_finals += 1
    return {"commit": commit, "receipt": receipt, "verified_artifacts": len(artifacts),
            "verified_prior_finals": prior_finals}


@dataclass(frozen=True)
class StateBlock:
    values: Any
    axes: tuple[str, ...]
    units: str
    coordinates: Mapping[str, Any]
    weights: Mapping[str, Any]
    source_identity: Mapping[str, Any]


class StateReader:
    """Read one committed physics.full owner in its external HDF5 axis order."""

    def __init__(self, commit_path: str | Path, *,
                 expected_identity: Mapping[str, Any] | None = None):
        path = Path(commit_path)
        self.commit, payload = read_json(path, maximum=16 * 1024 * 1024)
        artifacts = validate_commit(self.commit)
        if expected_identity is not None and any(
                self.commit["identity"].get(key) != value for key, value in expected_identity.items()):
            _fail("state identity differs from the selected attempt")
        owners = [item for item in artifacts if item.role == "physics.full"]
        if len(owners) != 1:
            _fail("state must have exactly one physics.full owner")
        if "state_id" in self.commit:
            _receipt(path.parent, self.commit, payload)
        owner = owners[0]
        self._stream = safe_path(path.parent, owner.path).open("rb")
        self._handle = None
        try:
            _verify_stream(self._stream, owner.size, owner.sha256)
            self._handle = h5py.File(self._stream, "r")
            _internal_storage(self._handle)
            validate_native_handle(self._handle, owner.role, owner.schema)
            if "state_id" in self.commit and "metadata/point_identity_json" not in self._handle:
                _fail("native state identity is missing")
            for name in ("metadata/identity_json", "metadata/point_identity_json"):
                if name in self._handle:
                    identity = _json_scalar(self._handle, name)
                    _state_identity(identity, self.commit["identity"])
                    _identity_fields(identity, self.commit["identity"])
        except Exception:
            self.close()
            raise

    def __enter__(self) -> "StateReader":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        self._stream.close()

    def read(self, dataset_path: str, selection: tuple[int | slice, ...], *,
             maximum_bytes: int = 1024 * 1024) -> StateBlock:
        if self._handle is None:
            raise ValueError("state reader is closed")
        if type(maximum_bytes) is not int or maximum_bytes < 1:
            raise ValueError("maximum_bytes must be positive")
        if not isinstance(dataset_path, str) or not dataset_path or ".." in dataset_path.split("/"):
            raise ValueError("dataset path must be a native absolute or relative HDF5 path")
        dataset = self._handle[dataset_path]
        if not isinstance(dataset, h5py.Dataset) or dataset.dtype.hasobject:
            _fail("state block requires a fixed width native dataset")
        if not isinstance(selection, tuple) or len(selection) != dataset.ndim:
            raise ValueError("selection must name every stored axis")
        normalized: list[int | slice] = []
        counts: list[int] = []
        for item, size in zip(selection, dataset.shape):
            if type(item) is int:
                position = item + size if item < 0 else item
                if not 0 <= position < size:
                    raise IndexError("state selection is outside its axis")
                normalized.append(position)
                counts.append(1)
            elif isinstance(item, slice):
                if item.step is not None and item.step <= 0:
                    raise ValueError("state selection requires a positive slice step")
                start, stop, step = item.indices(size)
                normalized.append(slice(start, stop, step))
                counts.append(len(range(start, stop, step)))
            else:
                raise ValueError("state selection supports integers and positive slices")
        axes = tuple(_text(dataset.attrs.get("logical_axis_order", "")).split(","))
        if len(axes) != dataset.ndim or any(not axis for axis in axes):
            _fail("state block has no explicit stored axis order")
        units = _text(dataset.attrs.get("units", ""))
        if not units:
            _fail("state block units are missing")
        coordinate_paths = json.loads(_text(dataset.attrs.get("axis_coordinate_paths_json", "null")))
        if not isinstance(coordinate_paths, list) or len(coordinate_paths) != dataset.ndim:
            _fail("state block coordinate inventory differs from its rank")
        # Include coordinate and weight vectors in the allocation budget, not
        # just the multidimensional payload. Missing measures remain None.
        vectors: list[tuple[str, str, h5py.Dataset, int | slice]] = []
        for axis, source, item, size in zip(axes, coordinate_paths, normalized, dataset.shape):
            if source is not None:
                vector = self._handle[source]
                if not isinstance(vector, h5py.Dataset) or vector.shape != (size,) or vector.dtype.hasobject:
                    _fail("state coordinate shape or dtype differs from its axis")
                vectors.append(("coordinate", axis, vector, item))
            for candidate in _weight_paths(axis):
                if candidate in self._handle:
                    vector = self._handle[candidate]
                    if not isinstance(vector, h5py.Dataset) or vector.shape != (size,) or vector.dtype.hasobject:
                        _fail("state quadrature weight shape or dtype differs from its axis")
                    vectors.append(("weight", axis, vector, item))
                    break
        required = math.prod(counts) * dataset.dtype.itemsize + sum(
            (1 if type(item) is int else len(range(item.start, item.stop, item.step))) * vector.dtype.itemsize
            for _, _, vector, item in vectors)
        if required > maximum_bytes:
            raise ContractError(f"state block needs {required} bytes; budget is {maximum_bytes}", "budget_exceeded")
        coordinates: dict[str, Any] = dict.fromkeys(axes)
        weights: dict[str, Any] = dict.fromkeys(axes)
        for kind, axis, vector, item in vectors:
            (coordinates if kind == "coordinate" else weights)[axis] = vector[item]
        return StateBlock(dataset[tuple(normalized)], axes, units, coordinates, weights,
                          dict(self.commit["identity"]))


def _text(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _internal_storage(handle: Any) -> None:
    """Reject payloads whose values can change outside the committed HDF5 file."""
    seen: set[int] = set()
    pending = [handle]
    while pending:
        group = pending.pop()
        address = h5py.h5o.get_info(group.id).addr
        if address in seen:
            continue
        seen.add(address)
        for name in group:
            link = group.get(name, getlink=True)
            if not isinstance(link, h5py.HardLink):
                _fail("state payload rejects unowned external HDF5 or soft links")
            child = group[name]
            if isinstance(child, h5py.Group):
                pending.append(child)
            elif isinstance(child, h5py.Dataset) and (child.external or child.is_virtual):
                _fail("state payload rejects unowned external or virtual storage")


def _weight_paths(axis: str) -> tuple[str, ...]:
    return {"E": ("grids_dimensionless/wE", "axes/energy_weights"),
            "k": ("grids_dimensionless/wk", "axes/momentum_weights"),
            "z": ("grids_dimensionless/wx", "axes/spatial_weights")}.get(axis, ())
