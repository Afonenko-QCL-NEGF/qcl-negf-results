"""Bounded, bit-preserving compaction of an exact committed telemetry prefix."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from qcl_negf_contracts.artifacts import CONTRACT_SET
from qcl_negf_contracts.messages import ContractError
from qcl_negf_contracts.telemetry import TABLE_FIELDS, TELEMETRY_SCHEMA
from .commits import json_bytes

BATCH_ROWS = 4096
ROW_GROUP_ROWS = 65536
_TYPES = {"string": pa.string(), "int64": pa.int64(), "float64": pa.float64(), "bool": pa.bool_()}


def validate_performance_schema(schema: pa.Schema) -> str:
    metadata = schema.metadata or {}
    if metadata.get(b"contract_set") != CONTRACT_SET.encode() or metadata.get(b"schema") != TELEMETRY_SCHEMA.encode():
        raise ContractError("unsupported performance contract set or schema", "incompatible_contract")
    table = metadata.get(b"table", b"").decode("utf-8")
    fields = TABLE_FIELDS.get(table)
    if fields is None or schema.remove_metadata() != pa.schema([pa.field(name, _TYPES[kind]) for name, kind in fields]):
        raise ContractError("performance table differs from its exact typed contract", "incompatible_contract")
    return table


def _batches(paths: list[Path]) -> Iterator[pa.Table]:
    pending: list[pa.RecordBatch] = []
    count = 0
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=BATCH_ROWS):
            offset = 0
            while offset < batch.num_rows:
                take = min(BATCH_ROWS - count, batch.num_rows - offset)
                pending.append(batch.slice(offset, take))
                offset += take
                count += take
                if count == BATCH_ROWS:
                    yield pa.Table.from_batches(pending).combine_chunks()
                    pending, count = [], 0
    if pending:
        yield pa.Table.from_batches(pending).combine_chunks()


def _payload_hash(paths: list[Path]) -> str:
    """Hash validity and native scalar bits; null payload bytes have no meaning."""
    digest = hashlib.sha256()
    for batch in _batches(paths):
        digest.update(batch.num_rows.to_bytes(8, "little"))
        for column in batch.columns:
            array = column.combine_chunks()
            digest.update(array.is_valid().to_numpy(zero_copy_only=False).tobytes())
            default: Any = "" if pa.types.is_string(array.type) else False if pa.types.is_boolean(array.type) else 0
            filled = pc.fill_null(array, default)
            if pa.types.is_string(array.type):
                buffers = filled.buffers()
                offsets = np.frombuffer(buffers[1], dtype="<i4", count=len(filled) + 1, offset=filled.offset * 4)
                digest.update(np.diff(offsets).astype("<i4", copy=False).tobytes())
                digest.update(memoryview(buffers[2])[int(offsets[0]):int(offsets[-1])])
            else:
                values = filled.to_numpy(zero_copy_only=False)
                digest.update(values.astype(values.dtype.newbyteorder("<"), copy=False).tobytes())
    return digest.hexdigest()


def compact_performance(paths: list[Path], destination: Path) -> dict[str, Any]:
    """Keep order, every row, nulls, NaN bits and signed zeros; prove by rereading."""
    if not paths:
        raise ValueError("compaction requires committed source segments")
    first = pq.ParquetFile(paths[0])
    schema = first.schema_arrow
    table = validate_performance_schema(schema)
    sources = []
    rows = 0
    for path in paths:
        file = pq.ParquetFile(path)
        validate_performance_schema(file.schema_arrow)
        if not schema.equals(file.schema_arrow, check_metadata=True):
            raise ContractError("telemetry compaction cannot mix table schemas", "incompatible_contract")
        with path.open("rb") as stream:
            sha = hashlib.file_digest(stream, "sha256").hexdigest()
        count = file.metadata.num_rows
        sources.append({"sha256": sha, "bytes": path.stat().st_size,
                        "row_start": rows, "row_stop": rows + count})
        rows += count
    with pq.ParquetWriter(destination, schema, compression="zstd", use_dictionary=True) as writer:
        pending: list[pa.Table] = []
        count = 0
        for batch in _batches(paths):
            pending.append(batch)
            count += batch.num_rows
            if count >= ROW_GROUP_ROWS:
                writer.write_table(pa.concat_tables(pending), row_group_size=ROW_GROUP_ROWS)
                pending, count = [], 0
        if pending:
            writer.write_table(pa.concat_tables(pending), row_group_size=ROW_GROUP_ROWS)
    before = _payload_hash(paths)
    after = _payload_hash([destination])
    if before != after or pq.ParquetFile(destination).metadata.num_rows != rows:
        raise ContractError("telemetry compaction changed committed rows", "corrupt_result")
    return {"schema": "qcl-negf.telemetry-compaction-proof.v1", "contract_set": CONTRACT_SET,
            "table": table, "source_segments": sources, "rows": rows,
            "source_files": len(paths), "source_bytes": sum(item["bytes"] for item in sources),
            "compacted_bytes": destination.stat().st_size,
            "schema_sha256": hashlib.sha256(json_bytes([list(field) for field in TABLE_FIELDS[table]])).hexdigest(),
            "logical_payload_sha256": before, "verified_payload_sha256": after,
            "proof_encoding": "fixed-4096-row batches; validity bytes; little-endian native values; UTF8 lengths and bytes",
            "row_order": "source catalog order, no sorting, aggregation or deduplication"}


def compact_export_records(records: list[dict[str, Any]], files: dict[str, Any],
                           spool: Path) -> list[dict[str, Any]]:
    """Replace segment containers with proven compact objects in each pinned record."""
    proofs = []
    depended_on = {name for record in records for item in record["included"] for name in item["dependencies"]}
    for record in records:
        groups: dict[str, list[dict[str, Any]]] = {}
        for item in record["included"]:
            path, descriptor = files[item["object"]]
            if descriptor["media_type"] == "application/vnd.apache.parquet" and item["object"] not in depended_on:
                table = validate_performance_schema(pq.ParquetFile(path).schema_arrow)
                groups.setdefault(table, []).append(item)
        for table, items in groups.items():
            if len(items) < 2:
                continue
            target = spool / f"compacted-{len(proofs):06d}.parquet"
            proof = compact_performance([files[item["object"]][0] for item in items], target)
            with target.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            name = f"objects/{digest}.parquet"
            proof.update(object=name, source_commit_sha256=record["source_commit_sha256"])
            proofs.append(proof)
            files[name] = (target, {"path": name, "sha256": digest, "bytes": target.stat().st_size,
                                   "media_type": "application/vnd.apache.parquet"})
            replaced = {id(item) for item in items}
            record["included"] = [item for item in record["included"] if id(item) not in replaced]
            record["included"].append({"role": items[0]["role"], "object": name, "sha256": digest,
                "schema": TELEMETRY_SCHEMA, "table": table, "dependencies": [],
                "selection": "every committed row, lossless compaction", "compaction_proof": len(proofs) - 1})
    retained = {item["object"] for record in records for item in record["included"]} | depended_on
    for name in list(files):
        if name not in retained:
            del files[name]
    return proofs
