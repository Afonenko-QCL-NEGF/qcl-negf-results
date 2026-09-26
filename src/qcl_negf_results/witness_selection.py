"""Declared science-export selection; immutable local witness evidence is retained."""
from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
from typing import Any

import h5py
import numpy as np

from qcl_negf_contracts.artifacts import CONTRACT_SET
from .commits import decode_json, json_bytes
from .native import validate_native_handle

POLICY = "per-attempt-first-first-positive-ratio-worst-last-v1"


def _selected_indices(table: Any, candidates: np.ndarray) -> np.ndarray:
    records = table["selected_blocks/record_sequence"][:]
    if not len(records):
        return np.zeros(0, dtype=np.int64)
    sequence = table["sequence"][:]
    ratios = table["ratio"][:]
    positions = np.searchsorted(sequence, records)
    if np.any(positions >= len(sequence)) or not np.array_equal(sequence[positions], records):
        raise ValueError("selected PSD evidence is outside its scalar history")
    ordered = candidates[np.argsort(records[candidates])]
    finite = ordered[np.isfinite(ratios[positions[ordered]])]
    selected = {int(ordered[0]), int(ordered[-1])}
    if len(finite):
        selected.add(int(finite[np.argmax(ratios[positions[finite]])]))
        positive = finite[ratios[positions[finite]] > 0]
        if len(positive):
            selected.add(int(positive[0]))
    return np.asarray(sorted(selected), dtype=np.int64)


def _attempt_intervals(handle: Any) -> list[tuple[int, int, int]]:
    if "source_segments_json" not in handle["metadata"]:
        raise ValueError("PSD selection requires explicit source attempt coordinates")
    encoded = handle["metadata/source_segments_json"][()]
    payload = encoded.encode() if isinstance(encoded, str) else encoded
    sources = decode_json(b'{"sources":' + payload + b'}')["sources"]
    intervals = []
    coordinates = handle["scba/sequence"][:]
    for source in sources:
        count = source["scba_rows"]
        if count == 0:
            continue
        first, last = source.get("scba_sequence_first"), source.get("scba_sequence_last")
        attempt = source["identity"].get("attempt")
        if any(type(value) is not int or value < 1 for value in (count, first, last, attempt)) or first > last:
            raise ValueError("PSD source segment has invalid attempt coordinates")
        if int(np.searchsorted(coordinates, last, side="right") - np.searchsorted(coordinates, first)) != count:
            raise ValueError("PSD source segment coordinates do not cover its declared rows")
        intervals.append((first, last, attempt))
    ordered = sorted(intervals)
    if any(older[1] >= newer[0] for older, newer in zip(ordered, ordered[1:])):
        raise ValueError("PSD source attempt intervals overlap")
    return ordered


def _attempt_selection(handle: Any) -> tuple[np.ndarray, list[dict[str, Any]]]:
    table = handle["psd_history"]
    records = table["selected_blocks/record_sequence"][:]
    if not len(records):
        return np.zeros(0, dtype=np.int64), []
    grouped: dict[int, list[int]] = {}
    assigned = np.zeros(len(records), dtype=bool)
    for first, last, attempt in _attempt_intervals(handle):
        indices = np.flatnonzero((records >= first) & (records <= last))
        grouped.setdefault(attempt, []).extend(indices.tolist())
        assigned[indices] = True
    if not np.all(assigned):
        raise ValueError("PSD witness has no source attempt")
    selected, proof = [], []
    for attempt, indices in sorted(grouped.items()):
        if not indices:
            continue
        chosen = _selected_indices(table, np.asarray(indices, dtype=np.int64))
        selected.extend(chosen.tolist())
        proof.append({"attempt": attempt, "source_witnesses": len(indices), "selected_witnesses": len(chosen),
                      "omitted_witnesses": len(indices) - len(chosen), "selected_sequences": records[chosen].tolist()})
    return np.asarray(sorted(selected), dtype=np.int64), proof


def _replace_dataset(table: Any, name: str, values: Any, dtype: Any = None) -> None:
    attrs = dict(table[name].attrs)
    del table[name]
    column = table.create_dataset(name, data=values, dtype=dtype,
                                  **({"compression": "gzip", "compression_opts": 1, "shuffle": True}
                                     if len(values) and dtype is None else {}))
    for key, value in attrs.items():
        if key == "logical_shape":
            column.attrs[key] = str(len(values))
        elif key == "nonfinite_count" and np.asarray(values).dtype.kind == "f":
            column.attrs[key] = int(np.count_nonzero(~np.isfinite(values)))
        else:
            column.attrs[key] = value


def _select_packed(table: Any, selected: np.ndarray) -> None:
    utf8 = h5py.string_dtype("utf-8")
    remap = {int(index): offset for offset, index in enumerate(selected)}
    record_index = table["payload_record_index"][:]
    kept = np.asarray([index for index, record in enumerate(record_index) if int(record) in remap], dtype=np.int64)
    offsets = table["payload_offsets"][:]
    values = np.concatenate([table["payload_values"][offsets[index]:offsets[index + 1]] for index in kept]) if len(kept) else np.zeros(0)
    for name in ("record_sequence", "record_group_metadata_index", "record_attributes_json"):
        _replace_dataset(table, name, table[name][:][selected], utf8 if name.endswith("json") else None)
    _replace_dataset(table, "payload_record_index", np.asarray([remap[int(record_index[index])] for index in kept], dtype=np.int64))
    _replace_dataset(table, "payload_metadata_index", table["payload_metadata_index"][:][kept])
    _replace_dataset(table, "payload_offsets", np.concatenate(([0], np.cumsum(offsets[kept + 1] - offsets[kept]))).astype(np.int64))
    _replace_dataset(table, "payload_values", values.astype(np.float64, copy=False))


def select_history_witnesses(source: Path, destination: Path) -> dict[str, Any] | None:
    with h5py.File(source, "r") as handle:
        validate_native_handle(handle, "science.history")
        sequences = handle["psd_history/selected_blocks/record_sequence"][:]
        if len(sequences) <= 4:
            return None
        selected, attempts = _attempt_selection(handle)
        if len(selected) == len(sequences):
            return None
    shutil.copyfile(source, destination)
    with h5py.File(destination, "r+") as handle:
        psd = handle["psd_history"]
        _select_packed(psd["selected_blocks"], selected)
        psd.attrs["export_witness_policy"] = POLICY
        psd.attrs["matrix_sampling"] = "first, first positive defect ratio, worst and last matrix witness separately per source attempt"
        psd.attrs["source_matrix_witnesses"] = len(sequences)
        psd.attrs["selected_matrix_witnesses"] = len(selected)
        psd.attrs["omitted_matrix_witnesses"] = len(sequences) - len(selected)
        validate_native_handle(handle, "science.history")
    # Repack removes free HDF5 blocks left by replacing large packed datasets.
    packed = destination.with_suffix(".repacked.h5")
    with h5py.File(destination, "r") as old, h5py.File(packed, "w") as new:
        for name in old:
            old.copy(name, new)
        for key, value in old.attrs.items():
            new.attrs[key] = value
    packed.replace(destination)
    omitted = np.delete(sequences, selected).tolist()
    return {"schema": "qcl-negf.witness-selection-proof.v1", "contract_set": CONTRACT_SET,
            "policy": POLICY, "scope": "each source attempt independently", "attempts": attempts,
            "source_witnesses": len(sequences), "selected_witnesses": len(selected),
            "omitted_witnesses": len(omitted), "selected_sequences": sequences[selected].tolist(),
            "omitted_sequences_sha256": hashlib.sha256(json_bytes(omitted)).hexdigest(),
            "scalar_history": "all rows unchanged", "local_source_evidence": "retained unchanged",
            "matrix_payload_scope": "selected witnesses only; omitted matrices require original local history"}


def select_export_witnesses(records: list[dict[str, Any]], files: dict[str, Any], spool: Path) -> list[dict[str, Any]]:
    proofs = []
    depended_on = {name for record in records for item in record["included"] for name in item["dependencies"]}
    for record in records:
        for item in record["included"]:
            if item["role"] != "science.history" or item["object"] in depended_on:
                continue
            source_name = item["object"]
            source, descriptor = files[source_name]
            target = spool / f"selected-witnesses-{len(proofs):06d}.h5"
            proof = select_history_witnesses(source, target)
            if proof is None:
                continue
            with target.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            name = f"objects/{digest}.h5"
            proof.update(source_sha256=descriptor["sha256"], derived_sha256=digest,
                         source_bytes=source.stat().st_size, derived_bytes=target.stat().st_size)
            proofs.append(proof)
            files[name] = (target, {"path": name, "sha256": digest, "bytes": target.stat().st_size,
                                   "media_type": "application/x-hdf5"})
            item.update(object=name, sha256=digest, source_sha256=descriptor["sha256"],
                        witness_selection_proof=len(proofs) - 1)
    retained = {item["object"] for record in records for item in record["included"]} | depended_on
    for name in list(files):
        if name not in retained:
            del files[name]
    return proofs
