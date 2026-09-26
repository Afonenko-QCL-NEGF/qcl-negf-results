"""Lossless selected-PSD evidence in the current packed native format.

The packed layout changes only HDF5 object organization. Matrix shapes, units,
coordinates, and Float64 payload bits remain scientifically identical.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import h5py
import numpy as np

from .commits import decode_json

PACKED_SCHEMA = "qcl-negf-psd-selected-packed-v2"
CHUNK_VALUES = 131_072
_FIELDS = {"record_sequence", "record_attributes_json", "group_metadata_json",
           "record_group_metadata_index", "dataset_metadata_json", "payload_record_index",
           "payload_metadata_index", "payload_offsets", "payload_values"}


def _text(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _json(value: Any) -> dict[str, Any]:
    return decode_json(_text(value).encode("utf-8"), "packed PSD metadata")


def _attribute_value(value: Any) -> np.ndarray:
    """HDF5 fixed UTF-8 bytes and JSON Unicode express the same text annotation."""
    array = np.asarray(value)
    if array.dtype.kind in "OSU":
        decoded = [_text(item) if isinstance(item, (bytes, str)) else item for item in array.flat]
        return np.asarray(decoded, dtype=object).reshape(array.shape)
    return array


def _attrs(previous: Any, current: Any) -> None:
    for key, value in previous.items():
        if key not in current or not np.array_equal(_attribute_value(value), _attribute_value(current[key])):
            raise ValueError("new cumulative history changes selected PSD annotations")


@dataclass(frozen=True)
class ArrayEvidence:
    dataset: Any
    shape: tuple[int, ...]
    attributes: Any
    offset: int

    @property
    def size(self) -> int:
        return math.prod(self.shape)


@dataclass
class RecordEvidence:
    attributes: Any
    groups: dict[str, Any]
    arrays: dict[str, ArrayEvidence]


def _indices(group: Any, name: str, count: int) -> np.ndarray:
    column = group[name]
    if column.shape != (count,) or column.dtype.kind != "i" or column.dtype.itemsize != 8:
        raise ValueError("packed PSD index must be a native Int64 vector")
    return column[()]


def _metadata(group: Any, name: str) -> list[dict[str, Any]]:
    column = group[name]
    if column.ndim != 1 or h5py.check_string_dtype(column.dtype) is None:
        raise ValueError("packed PSD metadata must be a UTF-8 string vector")
    return [_json(item) for item in column]


def _packed_records(group: Any) -> tuple[np.ndarray, list[RecordEvidence]]:
    """Validate record identities and dictionary references before payload views."""
    sequence_column = group["record_sequence"]
    if sequence_column.ndim != 1:
        raise ValueError("packed PSD record sequence must be a vector")
    sequences = _indices(group, "record_sequence", len(sequence_column))
    record_attributes = _metadata(group, "record_attributes_json")
    group_metadata = _metadata(group, "group_metadata_json")
    group_indices = _indices(group, "record_group_metadata_index", len(sequences))
    if len(record_attributes) != len(sequences) or len(set(sequences)) != len(sequences):
        raise ValueError("packed PSD record coordinates are inconsistent")
    records: list[RecordEvidence] = []
    for index, sequence in enumerate(sequences):
        metadata_index = int(group_indices[index])
        if not 0 <= metadata_index < len(group_metadata):
            raise ValueError("packed PSD group index is outside the metadata dictionary")
        attributes = record_attributes[index]
        if attributes.get("sequence") != int(sequence):
            raise ValueError("packed PSD sequence differs from its record annotations")
        records.append(RecordEvidence(attributes, group_metadata[metadata_index], {}))
    return sequences, records


def _attach_packed_payloads(group: Any, records: list[RecordEvidence]) -> None:
    """Validate a complete flat payload inventory and attach lazy matrix views."""
    descriptors = _metadata(group, "dataset_metadata_json")
    record_column = group["payload_record_index"]
    if record_column.ndim != 1:
        raise ValueError("packed PSD payload indices must be vectors")
    count = len(record_column)
    record_indices = _indices(group, "payload_record_index", count)
    descriptor_indices = _indices(group, "payload_metadata_index", count)
    offsets = _indices(group, "payload_offsets", count + 1)
    values = group["payload_values"]
    if values.ndim != 1 or values.dtype.kind != "f" or values.dtype.itemsize != 8:
        raise ValueError("packed PSD values must preserve native Float64")
    if offsets[0] != 0 or offsets[-1] != len(values) or np.any(offsets[1:] < offsets[:-1]):
        raise ValueError("packed PSD offsets do not cover the complete payload")
    for index in range(count):
        record_index, descriptor_index = int(record_indices[index]), int(descriptor_indices[index])
        if not 0 <= record_index < len(records) or not 0 <= descriptor_index < len(descriptors):
            raise ValueError("packed PSD payload index is outside the metadata dictionary")
        descriptor = descriptors[descriptor_index]
        if set(descriptor) != {"path", "shape", "attributes"}:
            raise ValueError("packed PSD dataset descriptor is incomplete")
        path, dimensions = descriptor["path"], descriptor["shape"]
        if (not isinstance(path, str) or not path or path.startswith("/")
                or any(part in {"", ".", ".."} for part in path.split("/"))
                or not isinstance(dimensions, list)
                or any(not isinstance(n, int) or isinstance(n, bool) or n < 0 for n in dimensions)
                or not isinstance(descriptor["attributes"], dict)):
            raise ValueError("invalid packed PSD dataset coordinates")
        arrays = records[record_index].arrays
        if path in arrays or math.prod(dimensions) != int(offsets[index + 1] - offsets[index]):
            raise ValueError("packed PSD payload size or path is inconsistent")
        arrays[path] = ArrayEvidence(values, tuple(dimensions), descriptor["attributes"], int(offsets[index]))


def _packed(group: Any) -> dict[int, RecordEvidence]:
    if set(group) != _FIELDS:
        raise ValueError("packed PSD container has missing or unrecognized evidence")
    if group.attrs.get("index_origin", 0) != 0:
        raise ValueError("packed PSD indices must be zero based")
    sequences, records = _packed_records(group)
    _attach_packed_payloads(group, records)
    return {int(sequence): record for sequence, record in zip(sequences, records)}


def _array_equal(previous: ArrayEvidence, current: ArrayEvidence) -> None:
    if previous.shape != current.shape or previous.dataset.dtype != current.dataset.dtype:
        raise ValueError("new cumulative history changes selected PSD matrix shape or dtype")
    _attrs(previous.attributes, current.attributes)
    for index in range(0, previous.size, CHUNK_VALUES):
        length = min(CHUNK_VALUES, previous.size - index)
        old = previous.dataset[previous.offset + index:previous.offset + index + length]
        new = current.dataset[current.offset + index:current.offset + index + length]
        if old.tobytes() != new.tobytes():
            raise ValueError("new cumulative history changes selected PSD evidence")


def selected_witness_records(group: Any) -> dict[int, RecordEvidence]:
    """Expose current-format record coordinates and lazy packed matrix views."""
    if not isinstance(group, h5py.Group) or _text(group.attrs.get("schema", "")) != PACKED_SCHEMA:
        raise ValueError("native format 4.0 requires packed selected PSD witnesses")
    return _packed(group)


def verify_selected_witnesses(previous: Any, current: Any) -> None:
    """Compare current-format packed values without expanding matrices."""
    _attrs(previous.attrs, current.attrs)
    old, new = selected_witness_records(previous), selected_witness_records(current)
    if not set(old) <= set(new):
        raise ValueError("new cumulative history omits selected PSD witnesses")
    for sequence, record in old.items():
        replacement = new[sequence]
        _attrs(record.attributes, replacement.attributes)
        if set(record.groups) != set(replacement.groups) or set(record.arrays) != set(replacement.arrays):
            raise ValueError("new cumulative history changes a selected PSD witness")
        for path, attributes in record.groups.items():
            _attrs(attributes, replacement.groups[path])
        for path, array in record.arrays.items():
            _array_equal(array, replacement.arrays[path])
