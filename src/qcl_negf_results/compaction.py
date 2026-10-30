"""Bounded, bit-preserving compaction of an exact committed telemetry prefix."""
from __future__ import annotations

import hashlib
from contextlib import ExitStack, contextmanager
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
from .export_budget import ExportBudget
from ._capture_io import regular, stamp
from .state import _BudgetStream, _IOBudget

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


class _CompactionRead:
    """One delivered-I/O allowance and bounded immutable source witnesses.

    The buffer limit is an engineering Arrow/Python workspace allowance, not an
    RSS or native allocator bound. Dictionary decoding avoids expanding repeated
    long strings before their actual selected length has been admitted.
    """
    def __init__(self, maximum, budget):
        if type(maximum) is not int or maximum < 1:
            raise ValueError("maximum_buffer_bytes must be positive")
        self.maximum = maximum
        self.slot = (maximum - 1024 * 1024) // 6
        self.witness_weight = 0
        self.budget = budget if budget is not None else _IOBudget(256 * 1024 * 1024)
        self.witnesses = {}

    def admit(self, count):
        if count + self.witness_weight > self.slot:
            raise ContractError("telemetry decoded buffer allowance exceeded", "budget_exceeded")

    def guard(self, path, expected):
        with regular(path) as raw:
            if stamp(raw, include_change=True) != expected:
                raise ContractError("consumed telemetry source changed", "source_changed")

    @contextmanager
    def open(self, path):
        path = Path(path)
        with regular(path) as raw:
            before = stamp(raw, include_change=True)
            expected = self.witnesses.get(path)
            if expected is not None and expected != before:
                raise ContractError("consumed telemetry source changed", "source_changed")
            if expected is None:
                # Path witnesses and file metadata share the finite allowance.
                self.admit(len(str(path).encode()) + 512)
                self.witness_weight += len(str(path).encode()) + 512
                self.witnesses[path] = before
            stream = _BudgetStream(raw, self.budget)
            try:
                yield stream
            except BaseException as primary:
                try:
                    if stamp(raw, include_change=True) != before:
                        raise ContractError("telemetry source changed during read", "source_changed")
                    self.guard(path, before)
                except BaseException as cleanup:
                    primary.add_note("source guard: " + str(cleanup))
                raise
            else:
                if stamp(raw, include_change=True) != before:
                    raise ContractError("telemetry source changed during read", "source_changed")
                self.guard(path, before)

    def parquet(self, stream):
        self.admit(1024)
        end=stream.seek(0,2)
        if end<12:raise ContractError('invalid Parquet closed size','corrupt_result')
        stream.seek(-8,2);trailer=stream.read(8)
        if len(trailer)!=8 or trailer[4:]!=b'PAR1':raise ContractError('invalid Parquet footer','corrupt_result')
        self.admit(2*int.from_bytes(trailer[:4],'little')+4096)
        stream.seek(0)
        options = dict(pre_buffer=False, buffer_size=0, thrift_string_size_limit=max(1, self.slot),
            thrift_container_size_limit=max(1, self.slot // 128))
        with pq.ParquetFile(stream, **options) as header:
            metadata = header.metadata
            if metadata.num_columns>max(map(len,TABLE_FIELDS.values())):
                raise ContractError('telemetry table column bound differs','incompatible_contract')
            self.admit(2*metadata.serialized_size+metadata.num_columns*1024)
            strings = [field.name for field in header.schema_arrow if pa.types.is_string(field.type)]
        stream.seek(0)
        return pq.ParquetFile(stream, metadata=metadata, read_dictionary=strings, **options)

    def geometry(self, file):
        logical = file.schema_arrow
        # Only dictionary string storage requested by this reader is normalized.
        normalized = pa.schema([pa.field(f.name, pa.string() if pa.types.is_dictionary(f.type) and pa.types.is_string(f.type.value_type) else f.type,
            nullable=f.nullable) for f in logical], metadata=logical.metadata)
        table = validate_performance_schema(normalized)
        widths = sum(4 if k == "string" else 1 if k == "bool" else 8 for _, k in TABLE_FIELDS[table])
        for index in range(file.metadata.num_row_groups):
            group = file.metadata.row_group(index)
            uncompressed = sum(group.column(n).total_uncompressed_size for n in range(group.num_columns))
            if uncompressed < 0:
                raise ContractError("invalid telemetry row-group size", "corrupt_result")
            self.admit(2 * uncompressed + group.num_rows * (widths + len(normalized)) + 4096)
        return normalized, table

    def plain(self, batch, schema):
        # Six independently bounded lifetimes: input, pending, combined/yielded,
        # previous yielded/caller batch, and two hash/cast scratch slots; plus 1 MiB SHA.
        # Source/path witness metadata is included in each conservative slot check.
        self.admit(batch.get_total_buffer_size() + len(batch.schema) * 1024)
        output = 0
        for column in batch.columns:
            if pa.types.is_dictionary(column.type):
                dictionary = column.dictionary
                offsets = np.frombuffer(dictionary.buffers()[1], dtype="<i4", count=len(dictionary) + 1, offset=dictionary.offset * 4)
                # Scalar iteration keeps scratch independent of string expansion.
                length = sum(int(offsets[i.as_py()+1] - offsets[i.as_py()]) for i in column.indices if i.is_valid)
                output += length + (len(column) + 1) * 4 + (len(column) + 7) // 8
            else:
                output += column.get_total_buffer_size()
        self.admit(output + len(schema) * 1024)
        result = pa.Table.from_batches([batch]).cast(schema)
        self.admit(result.get_total_buffer_size() + len(schema) * 1024)
        return result.to_batches()[0]

    def finish(self):
        for path, expected in self.witnesses.items():
            self.guard(path, expected)


def _batches(paths: list[Path], context: _CompactionRead) -> Iterator[pa.Table]:
    pending: list[pa.RecordBatch] = []
    count = 0
    for path in paths:
        with context.open(path) as stream:
            with context.parquet(stream) as file:
                schema, _ = context.geometry(file)
                for batch in file.iter_batches(batch_size=BATCH_ROWS, use_threads=False):
                    batch = context.plain(batch, schema)
                    offset = 0
                    while offset < batch.num_rows:
                        take = min(BATCH_ROWS - count, batch.num_rows - offset)
                        piece = batch.slice(offset, take)
                        context.admit(sum(b.get_total_buffer_size() + len(b.schema) * 1024 for b in pending)
                            + piece.get_total_buffer_size() + len(piece.schema) * 1024)
                        pending.append(piece)
                        offset += take
                        count += take
                        if count == BATCH_ROWS:
                            result = pa.Table.from_batches(pending).combine_chunks()
                            context.admit(result.get_total_buffer_size() + len(schema) * 1024)
                            yield result
                            pending, count = [], 0
    if pending:
        result = pa.Table.from_batches(pending).combine_chunks()
        context.admit(result.get_total_buffer_size() + len(result.schema) * 1024)
        yield result


def _payload_hash(paths: list[Path], context: _CompactionRead) -> str:
    """Hash validity and native scalar bits; null payload bytes have no meaning."""
    digest = hashlib.sha256()
    for batch in _batches(paths, context):
        digest.update(batch.num_rows.to_bytes(8, "little"))
        for column in batch.columns:
            context.admit(2*sum(chunk.get_total_buffer_size() for chunk in column.chunks)+len(column)*24+4096)
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


def compact_performance(paths: list[Path], destination: Path, *,
                        budget: ExportBudget | None = None,
                        maximum_buffer_bytes: int = 32 * 1024 * 1024,
                        io_budget: _IOBudget | None = None) -> dict[str, Any]:
    """Keep every row and native bits under one actual-delivery/workspace ledger."""
    if not paths:
        raise ValueError("compaction requires committed source segments")
    if destination.exists():
        raise ValueError("compaction destination must be new")
    context = _CompactionRead(maximum_buffer_bytes, io_budget)
    sources, rows, schema, table = [], 0, None, None
    for path in paths:
        with context.open(path) as stream:
            with context.parquet(stream) as file:
                candidate, name = context.geometry(file)
                if schema is None:
                    schema, table = candidate, name
                elif not schema.equals(candidate, check_metadata=True):
                    raise ContractError("telemetry compaction cannot mix table schemas", "incompatible_contract")
                count = file.metadata.num_rows
            stream.seek(0)
            sha = hashlib.file_digest(stream, "sha256").hexdigest()
            sources.append({"sha256": sha, "bytes": stream.seek(0, 2), "row_start": rows, "row_stop": rows + count})
            rows += count
    created = False
    try:
        with ExitStack() as stack:
            sink = stack.enter_context(budget.open(destination, "xb")) if budget is not None else stack.enter_context(destination.open("xb"))
            created = True
            writer = stack.enter_context(pq.ParquetWriter(sink, schema, compression="zstd", use_dictionary=True))
            for batch in _batches(paths, context):
                writer.write_table(batch, row_group_size=min(ROW_GROUP_ROWS, batch.num_rows))
        before = _payload_hash(paths, context)
        after = _payload_hash([destination], context)
        with context.open(destination) as stream:
            with context.parquet(stream) as file:
                compacted_rows = file.metadata.num_rows
        if before != after or compacted_rows != rows:
            raise ContractError("telemetry compaction changed committed rows", "corrupt_result")
        context.finish()
        return {"schema": "qcl-negf.telemetry-compaction-proof.v1", "contract_set": CONTRACT_SET,
                "table": table, "source_segments": sources, "rows": rows,
                "source_files": len(paths), "source_bytes": sum(item["bytes"] for item in sources),
                "compacted_bytes": destination.stat().st_size,
                "schema_sha256": hashlib.sha256(json_bytes([list(field) for field in TABLE_FIELDS[table]])).hexdigest(),
                "logical_payload_sha256": before, "verified_payload_sha256": after,
                "proof_encoding": "fixed-4096-row batches; validity bytes; little-endian native values; UTF8 lengths and bytes",
                "row_order": "source catalog order, no sorting, aggregation or deduplication"}
    except BaseException:
        if created:
            destination.unlink(missing_ok=True)
        raise


def compact_export_records(records: list[dict[str, Any]], files: dict[str, Any],
                           spool: Path, *, budget: ExportBudget | None = None,io_budget=None,retain_original=False) -> list[dict[str, Any]]:
    """Replace segment containers with proven compact objects in each pinned record."""
    proofs = []
    context=_CompactionRead(32*1024*1024,io_budget)
    depended_on = {name for record in records for item in record["included"] for name in item["dependencies"]}
    for record in records:
        groups: dict[str, list[dict[str, Any]]] = {}
        for item in record["included"]:
            path, descriptor = files[item["object"]]
            if descriptor["media_type"] == "application/vnd.apache.parquet" and item["object"] not in depended_on:
                with context.open(path) as stream:
                    with context.parquet(stream) as file:
                        _,table=context.geometry(file)
                groups.setdefault(table, []).append(item)
        for table, items in groups.items():
            if len(items) < 2:
                continue
            target = spool / f"compacted-{len(proofs):06d}.parquet"
            proof = compact_performance([files[item["object"]][0] for item in items], target, budget=budget,io_budget=io_budget)
            digest_state=hashlib.sha256()
            with context.open(target) as stream:
                while block:=stream.read(1024*1024):digest_state.update(block)
            digest=digest_state.hexdigest()
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
        if not retain_original and name not in retained:
            del files[name]
    context.finish()
    return proofs
