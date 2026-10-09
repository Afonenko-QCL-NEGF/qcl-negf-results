"""Verified immutable state access with bounded HDF5 hyperslabs.

Integrity verification streams each file once; it never deserializes solver
arrays. Keep a reader open for multiple block reads to avoid repeated hashing.
Receipts establish local publication and integrity, not scientific acceptance
or remote storage availability.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any, Mapping
from types import MappingProxyType

import h5py

from qcl_negf_contracts.artifacts import Artifact, validate_commit, validate_execution_progress
from qcl_negf_contracts.messages import ContractError
from .commits import decode_json, read_json, safe_path
from .native import validate_native_handle
from .provenance import _identity_fields, _json_scalar, _state_identity


def _fail(message: str) -> None:
    raise ContractError(message, "corrupt_result")


class _IOBudget:
    """Physical stream delivery, including integrity and native validation.

    This counts delivered bytes, not disk traffic or logical dataset bytes.
    Checking an EOF-clipped request before the callback prevents overshoot.
    """
    def __init__(self, maximum: int | None):
        if maximum is not None and (type(maximum) is not int or maximum < 1):
            raise ValueError("maximum_io_bytes must be a positive integer")
        self.maximum = maximum
        self.phase = "metadata"
        self.bytes_read_total = 0
        self.sha_bytes = 0
        self.hdf_bytes = 0
        self.metadata_bytes = 0
        self.logical_selected_bytes = 0

    def require(self, count: int) -> None:
        if self.maximum is not None and count > self.maximum - self.bytes_read_total:
            raise ContractError("total physical stream I/O budget exceeded", "budget_exceeded")

    def account(self, count: int) -> None:
        self.bytes_read_total += count
        if self.phase == "sha":
            self.sha_bytes += count
        elif self.phase == "hdf":
            self.hdf_bytes += count
        else:
            self.metadata_bytes += count

    def counters(self) -> dict[str, int | None]:
        return {name: getattr(self, name) for name in (
            "bytes_read_total", "sha_bytes", "hdf_bytes", "metadata_bytes", "logical_selected_bytes")} | {
                "maximum_io_bytes": self.maximum}


class _BudgetStream(io.RawIOBase):
    """A single read-only file object used by SHA and the HDF5 fileobj driver."""
    def __init__(self, raw: Any, budget: _IOBudget):
        super().__init__()
        self._raw, self._budget = raw, budget

    @property
    def name(self) -> str:
        return self._raw.name

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._raw.tell()

    def seek(self, offset: int, whence: int = 0) -> int:
        return self._raw.seek(offset, whence)

    def effective_request(self, size: int) -> int:
        position = self._raw.tell()
        end = self._raw.seek(0, 2)
        self._raw.seek(position)
        remaining = max(0, end - position)
        return remaining if size < 0 else min(size, remaining)

    def read(self, size: int = -1) -> bytes:
        self._budget.require(self.effective_request(size))
        data = self._raw.read(size)
        self._budget.account(len(data))
        return data

    def readinto(self, target: Any) -> int:
        self._budget.require(self.effective_request(len(target)))
        count = self._raw.readinto(target)
        self._budget.account(count)
        return count

    def close(self) -> None:
        if not self.closed:
            self._raw.close()
        super().close()


def _budget_json(path: Path, maximum: int, budget: _IOBudget) -> tuple[dict[str, Any], bytes]:
    with _BudgetStream(path.open("rb", buffering=0), budget) as stream:
        payload = stream.read(maximum + 1)
    if len(payload) > maximum:
        _fail("JSON exceeds its declared budget")
    return decode_json(payload, str(path)), payload


def _immutable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _immutable(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_immutable(item) for item in value)
    return value


def _verify_stream(stream: Any, size: int, digest: str) -> None:
    stream.seek(0, 2)
    if stream.tell() != size:
        _fail("state payload byte length differs from its commit")
    stream.seek(0)
    if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
        _fail("state payload checksum differs from its commit")
    stream.seek(0)


def _receipt(root: Path, commit: Mapping[str, Any], payload: bytes,
             budget: _IOBudget | None = None) -> dict[str, Any]:
    path = safe_path(root, "receipt.json")
    receipt, _ = (read_json(path, maximum=1024 * 1024) if budget is None
                  else _budget_json(path, 1024 * 1024, budget))
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


@dataclass(frozen=True)
class StateDescription:
    object_kind: str
    shape: tuple[int, ...] = ()
    dtype: Any = None
    itemsize: int = 0
    chunks: tuple[int, ...] | None = None
    compression: str | None = None
    axes: tuple[str, ...] = ()
    units: str = ""
    coordinate_paths: tuple[str | None, ...] | None = None
    child_names: tuple[str, ...] = ()
    filters: tuple[int, ...] = ()


@dataclass(frozen=True)
class StateScalar:
    value: Any
    units: str
    source_identity: Mapping[str, Any]


class StateReader:
    """Read one committed physics.full owner in its external HDF5 axis order."""

    def __init__(self, commit_path: str | Path, *,
                 expected_identity: Mapping[str, Any] | None = None,
                 maximum_io_bytes: int | None = None):
        path = Path(commit_path).resolve()
        self._budget = _IOBudget(maximum_io_bytes)
        self.commit, payload = _budget_json(path, 16 * 1024 * 1024, self._budget)
        artifacts = validate_commit(self.commit)
        if expected_identity is not None and any(
                not _same_json_value(self.commit["identity"].get(key), value)
                for key, value in expected_identity.items()):
            _fail("state identity differs from the selected attempt")
        owners = [item for item in artifacts if item.role == "physics.full"]
        if len(owners) != 1:
            _fail("state must have exactly one physics.full owner")
        receipt_verified = False
        if "state_id" in self.commit:
            _receipt(path.parent, self.commit, payload, self._budget)
            receipt_verified = True
        owner = owners[0]
        # The entire owner is mandatory for SHA; fail before opening physics if
        # the remaining cap cannot even cover that irreducible obligation.
        self._budget.require(owner.size)
        self._stream = _BudgetStream(safe_path(path.parent, owner.path).open("rb", buffering=0), self._budget)
        self._handle = None
        try:
            self._budget.phase = "sha"
            _verify_stream(self._stream, owner.size, owner.sha256)
            self._budget.phase = "hdf"
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
            metadata = self._handle["metadata"]
            self._source = _immutable({
                "commit_path": str(path), "commit_sha256": hashlib.sha256(payload).hexdigest(),
                "identity": dict(self.commit["identity"]), "expected_identity": dict(expected_identity or {}),
                "state_id": self.commit.get("state_id"), "state_sequence": self.commit.get("state_sequence"),
                "storage_class": self.commit.get("storage_class"),
                "owner": {"path": owner.path, "bytes": owner.size, "sha256": owner.sha256,
                          "role": owner.role, "schema": owner.schema},
                "producer_format": _text(metadata.attrs["schema_version"]),
                "contract_set": _text(metadata.attrs["contract_set"]),
                "native_verified": True, "receipt_verified": receipt_verified})
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

    @property
    def source(self) -> Mapping[str, Any]:
        """Immutable certificate from the exact constructor verification passes."""
        return _immutable({**self._source, "budgets": self._budget.counters()})

    @property
    def io_counters(self) -> Mapping[str, int | None]:
        return MappingProxyType(self._budget.counters())

    def _object(self, path: str) -> Any:
        if self._handle is None:
            raise ValueError("state reader is closed")
        if not isinstance(path, str) or not path or ".." in path.split("/"):
            raise ValueError("dataset path must be a native HDF5 path")
        try:
            return self._handle[path]
        except KeyError as error:
            raise ContractError(f"missing native object: {path}", "invalid_data") from error

    def describe(self, path: str, *, maximum_children: int = 128) -> StateDescription:
        """Immutable metadata; no numerical payload or raw HDF handle escapes."""
        if type(maximum_children) is not int or maximum_children < 1:
            raise ValueError("maximum_children must be positive")
        obj = self._object(path)
        if isinstance(obj, h5py.Group):
            if len(obj) > maximum_children:
                raise ContractError("group inventory exceeds its budget", "budget_exceeded")
            return StateDescription("group", child_names=tuple(obj))
        if not isinstance(obj, h5py.Dataset):
            _fail("unsupported native object")
        coordinate = obj.attrs.get("axis_coordinate_paths_json")
        try:
            paths = None if coordinate is None else json.loads(_text(coordinate))
        except (ValueError, UnicodeError) as error:
            raise ContractError("invalid coordinate inventory", "unsupported_layout") from error
        if paths is not None and (not isinstance(paths, list) or any(
                item is not None and not isinstance(item, str) for item in paths)):
            raise ContractError("invalid coordinate inventory", "unsupported_layout")
        plist = obj.id.get_create_plist()
        return StateDescription("dataset", tuple(obj.shape), obj.dtype, obj.dtype.itemsize,
            obj.chunks, obj.compression, tuple(_text(obj.attrs.get("logical_axis_order", "")).split(",")),
            _text(obj.attrs.get("units", "")), None if paths is None else tuple(paths),
            filters=tuple(plist.get_filter(i)[0] for i in range(plist.get_nfilters())))

    def _numeric(self, path: str, rank: int, maximum_bytes: int) -> Any:
        if type(maximum_bytes) is not int or maximum_bytes < 1:
            raise ValueError("maximum_bytes must be positive")
        dataset = self._object(path)
        if (not isinstance(dataset, h5py.Dataset) or dataset.ndim != rank
                or dataset.dtype.hasobject or dataset.dtype.kind not in "biufc"):
            raise ContractError("requires a fixed-width numeric dataset of the declared rank", "unsupported_layout")
        if not _text(dataset.attrs.get("units", "")):
            raise ContractError("numeric dataset units missing", "invalid_data")
        count = math.prod(dataset.shape) * dataset.dtype.itemsize
        if count > maximum_bytes:
            raise ContractError("numeric read exceeds its byte budget", "budget_exceeded")
        return dataset

    def read_scalar(self, path: str, *, maximum_bytes: int = 64) -> StateScalar:
        dataset = self._numeric(path, 0, maximum_bytes)
        value = dataset[()]
        self._budget.logical_selected_bytes += dataset.dtype.itemsize
        return StateScalar(value.item(), _text(dataset.attrs["units"]), _immutable(self.commit["identity"]))

    def read_vector(self, path: str, *, maximum_bytes: int) -> StateBlock:
        dataset = self._numeric(path, 1, maximum_bytes)
        values = dataset[:]
        self._budget.logical_selected_bytes += values.nbytes
        return StateBlock(values, tuple(_text(dataset.attrs.get("logical_axis_order", "")).split(",")),
            _text(dataset.attrs["units"]), {}, {}, _immutable(self.commit["identity"]))

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
        values = dataset[tuple(normalized)]
        self._budget.logical_selected_bytes += required
        return StateBlock(values, axes, units, coordinates, weights,
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
