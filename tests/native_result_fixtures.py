"""Small explicitly unavailable native v4 diagnostics for contract/IO tests.

This is test support, never a numerical producer or data migration adapter.
"""
from __future__ import annotations

import h5py
import numpy as np

from qcl_negf_results.native import MARKER_CHILD_FIELDS, MARKER_FIELDS
from qcl_negf_results.witnesses import PACKED_SCHEMA
from qcl_negf_contracts.artifacts import CONTRACT_SET


def empty_packed(parent):
    table = parent.create_group("selected_blocks")
    table.attrs.update(schema=PACKED_SCHEMA, index_origin=0)
    for name in ("record_sequence", "record_group_metadata_index", "payload_record_index", "payload_metadata_index"):
        table[name] = np.zeros(0, dtype=np.int64)
    for name in ("record_attributes_json", "group_metadata_json", "dataset_metadata_json"):
        table.create_dataset(name, shape=(0,), dtype=h5py.string_dtype("utf-8"))
    table["payload_offsets"] = np.array([0], dtype=np.int64)
    table["payload_values"] = np.zeros(0, dtype=np.float64)
    return table


def diagnostic_tables(parent, count=0, sequence=None):
    sequence = np.arange(1, count + 1, dtype=np.int64) if sequence is None else sequence
    markers = parent.create_group("physical_markers")
    markers.attrs["schema"] = "qcl-negf-physical-markers-v3"
    markers["sequence"] = sequence
    markers["available"] = np.zeros(count, dtype=np.int8)
    for name, kind in MARKER_FIELDS.items():
        if kind == "s":
            markers.create_dataset(name, data=["not_recorded"] * count, dtype=h5py.string_dtype("utf-8"))
        else:
            dtype = np.int8 if name == "equilibrium_applicable" else np.int64
            markers[name] = np.full(count, np.nan if kind == "f" else 0,
                                    dtype=np.float64 if kind == "f" else dtype)
    for name, fields in MARKER_CHILD_FIELDS.items():
        child = markers.create_group(name)
        child.attrs["schema"] = f"qcl-negf-physical-marker-{name}-v1"
        for field, kind in fields.items():
            child.create_dataset(field, shape=(0,), dtype=h5py.string_dtype("utf-8") if kind == "s"
                                 else np.int64 if kind == "i" else np.float64)
    psd = parent.create_group("psd_history")
    psd.attrs["schema"] = "qcl-negf-psd-history-v2"
    psd["sequence"] = sequence
    psd["available"] = np.zeros(count, dtype=np.int8)
    psd.create_dataset("matrix_kind", data=["not_recorded"] * count, dtype=h5py.string_dtype("utf-8"))
    empty_packed(psd)


def declare_native(handle, schema, role, *, scba_rows=0):
    metadata = handle.require_group("metadata")
    metadata.attrs.update(schema=schema, schema_version="4.0", artifact_role=role,
                          contract_set=CONTRACT_SET)
    containers = {"qcl-negf-scientific-history-v4": "", "qcl-negf-physics-analysis-v4": "diagnostics",
                  "qcl-negf-checkpoint-v4": "convergence", "qcl-negf-physics-v4": "convergence"}
    if schema in containers:
        path = containers[schema]
        parent = handle.require_group(path) if path else handle
        if "scba" not in parent:
            if path == "convergence":
                parent["scba"] = np.zeros((scba_rows, 18), dtype=np.float64)
            else:
                scba = parent.create_group("scba")
                scba["sequence"] = np.arange(1, scba_rows + 1, dtype=np.int64)
        scba = parent["scba"]
        sequence = scba["sequence"][()] if isinstance(scba, h5py.Group) and "sequence" in scba else None
        diagnostic_tables(parent, scba_rows, sequence)
    return metadata
