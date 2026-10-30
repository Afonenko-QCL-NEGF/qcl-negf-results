"""Streaming result-v2 single-archive transport.

The shared inventory describes whole native objects. Independent readback checks
every member and the complete object closure before publication or restoration.
Only IO buffers and the file inventory occupy memory; arrays stay in their files.
"""
from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
import hashlib
import io
import json
import lzma
import os
from pathlib import Path
import tarfile
import tempfile
from typing import Any

from qcl_negf_contracts.artifacts import (CONTRACT_SET, DIAGNOSTIC_EXPORT_SCHEMA,
    EXPORT_SCHEMA, EXPORT_TRANSPORT_SCHEMA,
    digest_value, relative_path, validate_export_receipt)
from qcl_negf_contracts.messages import ContractError
from .commits import decode_json, json_bytes
from .export_budget import ExportBudget

BLOCK = 1024 * 1024
INDEX_NAME = "export-index.json"
METADATA_NAMES = frozenset({"manifest.json", "plan.json", "diagnostics.json", "README.md"})


def _quiet(phase: str, **values: Any) -> None:
    pass


def _add_bytes(archive: tarfile.TarFile, name: str, value: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size, info.mode = len(value), 0o640
    archive.addfile(info, io.BytesIO(value))


class _Writer:
    def __init__(self, stream: Any, update: Callable[[int], None]):
        self.stream, self.update, self.count = stream, update, 0

    def write(self, value: bytes) -> int:
        count = self.stream.write(value)
        self.count += count
        self.update(self.count)
        return count

    def flush(self) -> None:
        self.stream.flush()


class _Reader:
    def __init__(self, source: Any, update: Callable[[int], None]):
        self.source, self.update = source, update

    def read(self, count: int = -1) -> bytes:
        value = self.source.read(BLOCK if count < 0 else min(count, BLOCK))
        self.update(len(value))
        return value


def _write_archive(path: Path, files: dict[str, tuple[Path, dict[str, Any]]],
                   metadata: dict[str, bytes], index_payload: bytes, preset: int,
                   report: Callable[..., None], budget: ExportBudget | None = None) -> None:
    total = len(index_payload) + sum(len(value) for value in metadata.values()) + sum(
        row["bytes"] for _, row in files.values())
    completed, archive_bytes = 0, 0

    def update(count: int = 0, compressed: int | None = None) -> None:
        nonlocal completed, archive_bytes
        completed += count
        if compressed is not None:
            archive_bytes = compressed
        report("compressing", completed_bytes=completed, total_bytes=total,
               archive_bytes=archive_bytes,
               **({"temporary_bytes": budget.used, "byte_budget": budget.byte_budget,
                   "reserve_bytes": budget.reserve_bytes} if budget is not None else {}))

    update()
    with (budget.open(path, "wb") if budget is not None else path.open("wb")) as raw:
        sink = _Writer(raw, lambda count: update(compressed=count))
        with lzma.LZMAFile(sink, "wb", preset=preset, check=lzma.CHECK_CRC64) as compressed:
            with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT) as archive:
                _add_bytes(archive, INDEX_NAME, index_payload)
                update(len(index_payload))
                for name, payload in metadata.items():
                    _add_bytes(archive, name, payload)
                    update(len(payload))
                for name, (source_path, row) in sorted(files.items()):
                    info = tarfile.TarInfo(name)
                    info.size, info.mode = row["bytes"], 0o640
                    with source_path.open("rb") as source:
                        archive.addfile(info, _Reader(source, update))
        raw.flush()
        os.fsync(raw.fileno())
    update(compressed=path.stat().st_size)


def build_archive(spool: Path, files: dict[str, tuple[Path, dict[str, Any]]],
                  metadata: dict[str, bytes], *, identity: str, profile: str,
                  preset: int = 1, report: Callable[..., None] = _quiet,
                  budget: ExportBudget | None = None
                  ) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Write one temporary tar/XZ stream; never paginate or omit an object."""
    index = {"schema": EXPORT_TRANSPORT_SCHEMA, "contract_set": CONTRACT_SET,
             "snapshot_identity": identity, "profile": profile,
             "metadata": [{"path": name, "bytes": len(payload),
                           "sha256": hashlib.sha256(payload).hexdigest()}
                          for name, payload in metadata.items()],
             "objects": [{"path": name, "bytes": row["bytes"], "sha256": row["sha256"]}
                         for name, (_, row) in sorted(files.items())]}
    _inventory(index)
    index_payload = json_bytes(index)
    path = spool / "snapshot.tar.xz"
    if budget is not None:
        budget.check()
    _write_archive(path, files, metadata, index_payload, preset, report, budget)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return path, {"sha256": digest, "bytes": path.stat().st_size}, index


def _rows(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ContractError("transport inventory must contain object rows", "corrupt_result")
    return value


def _inventory(index: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if index.get("schema") != EXPORT_TRANSPORT_SCHEMA or index.get("contract_set") != CONTRACT_SET:
        raise ContractError("unsupported single archive transport", "incompatible_contract")
    digest_value(index.get("snapshot_identity"))
    if index.get("profile") not in {"science", "full-state", "diagnostic"}:
        raise ContractError("invalid archive profile", "corrupt_result")
    inventory: dict[str, dict[str, Any]] = {}
    for key in ("objects", "metadata"):
        for row in _rows(index.get(key)):
            name = relative_path(row.get("path"))
            size = row.get("bytes")
            if (name in inventory or not isinstance(size, int) or isinstance(size, bool) or size < 0
                    or (key == "objects" and not name.startswith("objects/"))
                    or (key == "metadata" and name not in METADATA_NAMES)):
                raise ContractError("duplicate or invalid archive inventory member", "corrupt_result")
            digest_value(row.get("sha256"))
            inventory[name] = row
    if "manifest.json" not in inventory:
        raise ContractError("index omitted the mandatory manifest", "corrupt_result")
    return inventory


def _json_object(payload: bytes) -> dict[str, Any]:
    return decode_json(payload, "archive metadata")


def _manifest_closure(manifest: dict[str, Any], index: dict[str, Any]) -> None:
    objects = {row["path"]: row for row in index["objects"]}
    expected_schema = DIAGNOSTIC_EXPORT_SCHEMA if index["profile"] == "diagnostic" else EXPORT_SCHEMA
    if manifest.get("schema") != expected_schema:
        raise ContractError("manifest schema differs from its transport profile", "corrupt_result")
    if (manifest.get("contract_set") != CONTRACT_SET
            or manifest.get("snapshot_identity") != index["snapshot_identity"]
            or manifest.get("profile") != index["profile"]):
        raise ContractError("manifest and transport identify different snapshots", "corrupt_result")
    files = _rows(manifest.get("files"))
    if len(files) != len(objects) or {row.get("path") for row in files} != set(objects):
        raise ContractError("manifest and transport object inventories differ", "corrupt_result")
    for row in files:
        other = objects[row["path"]]
        if row.get("sha256") != other["sha256"] or row.get("bytes") != other["bytes"]:
            raise ContractError("manifest object identity differs from transport", "corrupt_result")
    for record in _rows(manifest.get("records")):
        for row in _rows(record.get("included")):
            dependencies = row.get("dependencies")
            if (row.get("object") not in objects or not isinstance(dependencies, list)
                    or any(dep not in objects for dep in dependencies)):
                raise ContractError("snapshot has dangling dependencies", "corrupt_result")
    if index['profile'] != 'diagnostic':
        complete = index['profile'] == 'full-state'
        if manifest.get('canonical_source_complete') is not complete:
            raise ContractError('subset archive cannot claim complete canonical source', 'corrupt_result')
        mapping = {}
        from qcl_negf_contracts import schema_validator
        from .state import _schema_check
        common = schema_validator('result-common.schema.json')
        validator = common.evolve(schema={'$ref': '#/$defs/FileRef', '$defs': common.schema['$defs']})
        for row in _rows(manifest.get('source_locator')):
            if set(row) != {'reference','object'}:
                raise ContractError('invalid source locator row', 'corrupt_result')
            ref = row['reference']
            _schema_check(validator, ref)
            name = relative_path(ref['path'])
            if name in mapping or row['object'] not in objects:
                raise ContractError('duplicate or dangling source locator', 'corrupt_result')
            item = objects[row['object']]
            if ref['sha256'] != item['sha256'] or int(ref['bytes']) != item['bytes']:
                raise ContractError('source locator differs from transport bytes', 'corrupt_result')
            mapping[name] = row
        technical={};technical_objects={}
        for row in _rows(manifest.get('technical_locator', [])):
            if set(row)!={'scope','reference','object'} or row['scope'] not in ('telemetry','canonical_plan'):
                raise ContractError('invalid technical locator scope/fields','corrupt_result')
            ref=row['reference'];_schema_check(validator,ref);key=(row['scope'],ref['path'])
            if key in technical or row['object'] not in objects:
                raise ContractError('duplicate or dangling technical locator','corrupt_result')
            obj=objects[row['object']]
            if ref['sha256']!=obj['sha256'] or int(ref['bytes'])!=obj['bytes']:
                raise ContractError('technical locator differs from exact original bytes','corrupt_result')
            technical[key]=ref;technical_objects[key]=row['object']
        proofs=_rows(manifest.get('telemetry_compaction', []))
        for record in _rows(manifest.get('records')):
            if record.get('technical') is not True:continue
            for row in _rows(record.get('included')):
                if 'compaction_proof' in row:
                    number=row['compaction_proof']
                    if type(number) is not int or not 0<=number<len(proofs):
                        raise ContractError('technical projection proof locator differs','corrupt_result')
                    proof=proofs[number]
                    if row['object']!=proof.get('object') or row.get('table')!=proof.get('table'):
                        raise ContractError('technical projection object/table differs','corrupt_result')
                else:
                    key=(row.get('source_scope'),row.get('source_path'));ref=technical.get(key)
                    if ref is None or technical_objects[key]!=row['object'] or ref['sha256']!=row.get('sha256') or ref['schema']!=row.get('schema'):
                        raise ContractError('technical record scope/locator differs','corrupt_result')
        original_bytes={(ref['sha256'],int(ref['bytes'])) for ref in technical.values() if ref['media_type']=='application/vnd.apache.parquet'}
        cutoff=manifest.get('cutoff')
        if cutoff is not None:
            ref=cutoff.get('commit')
            if not isinstance(ref,dict) or technical.get(('telemetry',ref.get('path')))!=ref:
                raise ContractError('technical cutoff has no exact original commit locator','corrupt_result')
            if cutoff.get('uncommitted')!='not_included' or cutoff.get('pending_rows') is not None:
                raise ContractError('technical cutoff fabricates uncommitted tail evidence','corrupt_result')
        elif technical:
            raise ContractError('technical locator has no selected transaction cutoff','corrupt_result')
        for proof in _rows(manifest.get('telemetry_compaction', [])):
            if (proof.get('object') not in objects or proof.get('logical_payload_sha256')!=proof.get('verified_payload_sha256')
                    or proof.get('row_order')!='source catalog order, no sorting, aggregation or deduplication'):
                raise ContractError('compaction projection proof/closure differs','corrupt_result')
            position=0
            for source in _rows(proof.get('source_segments')):
                if source.get('row_start')!=position or type(source.get('row_stop')) is not int or source['row_stop']<position:
                    raise ContractError('compaction source row ranges differ','corrupt_result')
                position=source['row_stop']
                if (source.get('sha256'),source.get('bytes')) not in original_bytes:
                    raise ContractError('compaction original source bytes were not retained','corrupt_result')
            if position!=proof.get('rows'):
                raise ContractError('compaction total row range differs','corrupt_result')
        state = manifest.get('source_state')
        _schema_check(common.evolve(schema={'$ref': '#/$defs/StateRef', '$defs': common.schema['$defs']}), state)
        if state['manifest']['path'] not in mapping or mapping[state['manifest']['path']]['reference'] != state['manifest']:
            raise ContractError('source manifest locator missing or changed', 'corrupt_result')
        _retained_v2_cutoff(manifest)


def _receipt_closure(receipt: dict[str, Any] | None, index: dict[str, Any]) -> None:
    if receipt is not None and (
            receipt["snapshot_identity"] != index["snapshot_identity"]
            or receipt["profile"] != index["profile"]
            or receipt["transport_schema"] != index["schema"]):
        raise ContractError("receipt and archive identify different snapshots or profiles", "corrupt_result")


def _read_archive(path: Path, root: Path | None, report: Callable[..., None], *, manifest_out=None, technical_out=None) -> dict[str, Any]:
    try:
        return _read_archive_contents(path, root, report, manifest_out=manifest_out, technical_out=technical_out)
    except (tarfile.TarError, lzma.LZMAError, EOFError) as error:
        raise ContractError('invalid complete archive stream: ' + str(error), 'corrupt_result') from error


def _read_archive_contents(path: Path, root: Path | None, report: Callable[..., None], *, manifest_out=None, technical_out=None) -> dict[str, Any]:
    with tarfile.open(path, "r:xz") as archive:
        first = archive.next()
        if first is not None and first.name == 'part.json':
            raise ContractError('retired multipart transport is unsupported', 'unsupported_format')
        if first is None or first.name != INDEX_NAME or not first.isfile():
            raise ContractError("archive omitted its leading transport index", "corrupt_result")
        source = archive.extractfile(first)
        assert source is not None
        if first.size > 64 * 1024 * 1024:
            raise ContractError('archive index exceeds metadata limit', 'budget_exceeded')
        index_payload = source.read(64 * 1024 * 1024 + 1)
        index = _json_object(index_payload)
        remaining = _inventory(index)
        total = sum(row["bytes"] for row in remaining.values())
        completed = 0
        manifest = None
        report("verifying", completed_bytes=0, total_bytes=total)
        for member in archive:
            # TarFile's iterator includes first even after next() in stream mode.
            if member is first:
                continue
            if not member.isfile() or member.name not in remaining:
                raise ContractError("archive contains an unindexed or duplicate member", "corrupt_result")
            row = remaining.pop(member.name)
            if member.size != row["bytes"]:
                raise ContractError("archive member size mismatch", "corrupt_result")
            source = archive.extractfile(member)
            assert source is not None
            target = None
            if root is not None:
                target_path = root / relative_path(member.name)
                target_path.parent.mkdir(parents=True, exist_ok=True)
                target = target_path.open("wb")
            digest = hashlib.sha256()
            if member.name == 'manifest.json' and member.size > 64 * 1024 * 1024:
                raise ContractError('export manifest exceeds metadata limit', 'budget_exceeded')
            captured = bytearray() if member.name == "manifest.json" else None
            try:
                while value := source.read(BLOCK):
                    digest.update(value)
                    if target is not None:
                        target.write(value)
                    if captured is not None:
                        captured.extend(value)
                    completed += len(value)
                    report("verifying", completed_bytes=completed, total_bytes=total)
            finally:
                if target is not None:
                    target.close()
            if digest.hexdigest() != row["sha256"]:
                raise ContractError("archive member checksum mismatch", "corrupt_result")
            if captured is not None:
                manifest = _json_object(captured)
        if remaining or manifest is None:
            raise ContractError("archive omitted mandatory members", "corrupt_result")
        assert archive.fileobj is not None
        while archive.fileobj.read(BLOCK):
            pass  # Force XZ footer and checksum verification beyond tar padding.
        _manifest_closure(manifest, index)
        technical = _verify_retained_v2(manifest, index,
            lambda name: archive.extractfile(name) if root is None else (root / relative_path(name)).open('rb'))
        if technical_out is not None and technical is not None: technical_out.update(technical)
        # Old hash-valid exports may contain HDF5 values stored outside their
        # object. Verify ownership independently before claiming closure or
        # publishing a restored tree. Tar members provide seekable file objects;
        # verification does not allocate a second numerical payload copy.
        from .state import _internal_storage
        import h5py
        for row in manifest["files"]:
            name = row["path"]
            if row.get("media_type") != "application/x-hdf5" and not name.endswith(".h5"):
                continue
            source = archive.extractfile(name) if root is None else root / relative_path(name)
            assert source is not None
            try:
                with h5py.File(source, "r") as handle:
                    _internal_storage(handle)
                    if index['profile'] != 'diagnostic':
                        from .native import validate_native_handle
                        schemas = {entry['reference']['schema'] for entry in manifest['source_locator']
                                   if entry['object'] == name}
                        if len(schemas) != 1:
                            raise ContractError('native object lacks one exact declared schema', 'corrupt_result')
                        validate_native_handle(handle, declared_schema=schemas.pop())
            except (OSError, ValueError) as error:
                raise ContractError(f"invalid HDF5 archive object: {error}", "corrupt_result") from error
            finally:
                if root is None:
                    source.close()
    if root is not None:
        (root / INDEX_NAME).write_bytes(index_payload)
    if manifest_out is not None:
        manifest_out.update(manifest)
    return index


def verify_archive(path: Path, *, report: Callable[..., None] = _quiet) -> dict[str, Any]:
    """Verify hashes and self-contained HDF5 storage without a second payload tree.

    Random access to compressed members may repeat decompression; native arrays
    remain on disk and no numerical values are read for this ownership check.
    """
    return _read_archive(path, None, report)


def _verify_source_tree(root, state_ref, *, limits=None, budget=None, selected_reader=None):
    """Explicit full-source metadata/dtype proof with the same bounded reader.

    The operation streams integrity hashes; no numerical tensor is materialized.
    """
    from .state import StateReader, ResultReadLimits, _IOBudget, _arrays
    limits = limits or ResultReadLimits()
    budget = budget or _IOBudget(limits.maximum_io_bytes)
    queue, seen = [state_ref], set()
    while queue:
        state = queue.pop()
        digest = state['manifest']['sha256']
        if digest in seen:
            continue
        seen.add(digest)
        from contextlib import nullcontext
        reader = selected_reader if selected_reader is not None and selected_reader._state_ref == state else None
        if reader is None:
            reader = StateReader.__new__(StateReader)
            reader._initialize(root, state_ref=state, expected_identity=state['identity'], limits=limits,
                               budget=budget)
        with (nullcontext(reader) if reader is selected_reader else reader):
            if len(seen) > reader.limits.maximum_catalog_pages:
                raise ContractError('restored lineage exceeds metadata limit', 'budget_exceeded')
            reader.import_history()
            checkpoint = reader.checkpoint()
            for declaration in _arrays(checkpoint):
                reader.describe_array(declaration)
            cursor = None
            while True:
                batch = reader.history_batch(cursor)
                for record in batch.records:
                    for declaration in _arrays(record):
                        reader.describe_array(declaration)
                cursor = batch.cursor
                if batch.at_cutoff:
                    break
            parent = reader.source.lineage['parent_state']
            if parent is not None:
                queue.append(parent)


def _restore_source(root, manifest):
    from ._capture_io import beneath
    from .state import StateReader
    source = root / 'source'
    source.mkdir()
    for row in manifest['source_locator']:
        target = beneath(source, row['reference']['path'])
        target.parent.mkdir(parents=True, exist_ok=True)
        # Both paths are in our freshly owned private extraction tree. Link the
        # same immutable bytes instead of allocating another scientific copy.
        os.link(root / row['object'], target)
    _verify_source_tree(source, manifest['source_state'])
    from .export import _selection
    state = manifest['source_state']
    with StateReader(source, state_ref=state, expected_identity=state['identity']) as reader:
        required, _, _ = _selection(reader, 'full-state')
    supplied = {row['reference']['path']: row['reference'] for row in manifest['source_locator']}
    if set(supplied) != set(required):
        raise ContractError('restored source inventory contains unrelated or missing paths', 'corrupt_result')
    for name, reference in required.items():
        if name not in supplied or any(supplied[name][key] != reference[key]
                                      for key in ('path','bytes','sha256','media_type','schema')):
            raise ContractError('restored source graph has missing/conflicting dependency', 'corrupt_result')


def receive(paths: Sequence[Path], destination: Path | None = None, *,
            receipt: dict[str, Any] | None = None) -> dict[str, Any]:
    """Verify one result-v2 archive; full-state reconstructs its original source."""
    paths = [Path(path) for path in paths]
    if not paths:
        raise ContractError("no export archives supplied", "corrupt_result")
    if len(paths) != 1:
        raise ContractError("single archive transport requires exactly one archive", "unsupported_format")
    path = paths[0]
    if receipt is not None:
        validate_export_receipt(receipt)
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != receipt["sha256"]:
                raise ContractError("archive checksum differs from receipt", "corrupt_result")
        if path.stat().st_size != receipt["bytes"]:
            raise ContractError("archive size differs from receipt", "corrupt_result")
    manifest = {}; technical = {}
    if destination is None:
        index = _read_archive(path, None, _quiet, manifest_out=manifest, technical_out=technical)
        _receipt_closure(receipt, index)
    else:
        destination = Path(destination)
        if destination.exists():
            raise ContractError("restore destination already exists")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".qcl-receive-", dir=destination.parent) as temporary:
            root = Path(temporary) / "restored"
            root.mkdir()
            index = _read_archive(path, root, _quiet, technical_out=technical)
            _receipt_closure(receipt, index)
            manifest = _json_object((root / 'manifest.json').read_bytes())
            if index['profile'] == 'full-state':
                _restore_source(root, manifest)
            os.replace(root, destination)
    return {"snapshot_identity": index["snapshot_identity"], "verified": True,
            "verification_scope": ('transport_hashes_native5_storage_and_source_closure'
                if destination is not None and index['profile'] == 'full-state' else 'transport_hashes_and_hdf5_storage'),
            "archive_count": 1, "objects": len(index["objects"]),
            "destination": str(destination) if destination is not None else None,
            'profile': index['profile'],
            'capabilities': manifest.get('capabilities', []),
            'source_root': str(destination / 'source') if destination is not None and index['profile'] == 'full-state' else None,
            'source_state': manifest['source_state'] if manifest and index['profile'] == 'full-state' else None,
            **({'technical_verification': technical} if technical else {})}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("verify", "reassemble"))
    parser.add_argument("archives", nargs="+", type=Path)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    if (args.action == "reassemble") != (args.destination is not None):
        parser.error("only reassemble requires --destination")
    try:
        receipt = json.loads(args.receipt.read_bytes()) if args.receipt else None
        print(json.dumps(receive(args.archives, args.destination, receipt=receipt), indent=2))
    except (ContractError, OSError, ValueError, EOFError, tarfile.TarError, lzma.LZMAError) as error:
        parser.exit(2, str(error) + "\n")



def _retained_v2_cutoff(manifest):
    """Classify the explicit new boundary; never reinterpret it as legacy v1."""
    cutoff = manifest.get('cutoff')
    records = [r for r in _rows(manifest.get('records')) if r.get('technical') is True]
    schema = cutoff.get('commit', {}).get('schema') if isinstance(cutoff, dict) else None
    declared = (schema == 'qcl-negf.telemetry-commit.v2' or
        isinstance(cutoff, dict) and 'technical_contract' in cutoff or
        any('technical_contract' in r for r in records))
    if schema is not None and schema not in ('qcl-negf.telemetry-commit.v1', 'qcl-negf.telemetry-commit.v2'):
        raise ContractError('unsupported explicit technical cutoff version', 'unsupported_format')
    if not declared: return None
    if schema != 'qcl-negf.telemetry-commit.v2' or len(records) != 1:
        raise ContractError('retained technical cutoff requires exactly one v2 record', 'corrupt_result')
    record = records[0]
    expected = dict(technical_contract='qcl-negf.telemetry-retained.v2', scope='retained_technical_window',
                   cumulative_scope='enclosing_transport_since_v2_genesis')
    for key, value in expected.items():
        if cutoff.get(key) != value or record.get(key) != value:
            raise ContractError('retained technical record/cutoff scope differs', 'corrupt_result')
    if (cutoff.get('prior_capture') not in ('none', 'unimported')
            or record.get('prior_capture') != cutoff['prior_capture']
            or record.get('catalog') != cutoff.get('catalog')
            or record.get('source_commit_sha256') != cutoff['commit']['sha256']
            or record.get('source_state') != manifest['source_state']
            or record.get('omitted_by_policy') != []):
        raise ContractError('retained technical selection/capture inventory differs', 'corrupt_result')
    return cutoff


def _verify_retained_v2(manifest, index, open_object):
    """Verify contained original v2 bytes/relations, without original source roots.

    This is declared source consistency, never authentication of AiiDA origin,
    a read lease, active native access, or scientific acceptance.
    """
    cutoff = _retained_v2_cutoff(manifest)
    if cutoff is None: return None
    from qcl_negf_contracts import schema_validator
    from qcl_negf_contracts.results import validate_manifest
    from qcl_negf_contracts.telemetry import (validate_telemetry_commit, validate_telemetry_catalog,
        validate_telemetry_source_binding, validate_telemetry_binding)
    from .state import _IOBudget, _BudgetStream, _metadata_weight, _schema_check
    from .catalog import _retained_parquet_header
    from .performance_views import _bounded_objects
    common = schema_validator('result-common.schema.json')
    reference_validator = common.evolve(schema={'$ref':'#/$defs/FileRef', '$defs':common.schema['$defs']})
    technical = {(r['scope'], r['reference']['path']):r for r in manifest['technical_locator']}
    scientific = {r['reference']['path']:r for r in manifest['source_locator']}
    cache = {}; used = set(); payloads = {}; budget = _IOBudget(4*1024*1024)
    def read(scope, ref, *, promise=True):
        _schema_check(reference_validator, ref); relative_path(ref['path'])
        row = scientific.get(ref['path']) if scope == 'scientific' else technical.get((scope, ref['path']))
        if row is None or row['reference'] != ref:
            raise ContractError('missing or conflicting exact retained locator', 'corrupt_result')
        key = (scope, ref['path'])
        if scope != 'scientific' and promise: used.add(key)
        if key in cache: return cache[key]
        size = int(ref['bytes']); budget.require(size)
        retained = _metadata_weight((technical, scientific, cache, used, payloads))
        if ref['schema'] == 'qcl-negf.telemetry-catalog.v3' and size > 65536:
            raise ContractError('retained catalog raw ceiling exceeded before open', 'budget_exceeded')
        if ref['media_type'] == 'application/json' and 2*size+retained > 1024*1024:
            raise ContractError('retained archive metadata credit exhausted before open', 'budget_exceeded')
        with open_object(row['object']) as raw:
            stream = _BudgetStream(raw, budget)
            if stream.seek(0, 2) != size:
                raise ContractError('retained original object size differs', 'corrupt_result')
            stream.seek(0)
            if ref['media_type'] == 'application/json':
                data = stream.read(size)
                if len(data) != size or hashlib.sha256(data).hexdigest() != ref['sha256']:
                    raise ContractError('retained original metadata bytes differ', 'corrupt_result')
                value = decode_json(data, 'retained archive metadata')
                if value.get('schema') != ref['schema']:
                    raise ContractError('retained metadata content/reference version differs', 'unsupported_format')
                cache[key] = value, data
            else:
                descriptor = payloads.get(json_bytes(ref))
                if descriptor is None or ref['media_type'] != 'application/vnd.apache.parquet':
                    raise ContractError('unselected or unsupported retained payload', 'corrupt_result')
                digest = hashlib.sha256()
                while block := stream.read(65536): digest.update(block)
                if digest.hexdigest() != ref['sha256']:
                    raise ContractError('retained original Parquet checksum differs', 'corrupt_result')
                _retained_parquet_header(stream, descriptor, maximum_metadata_bytes=1024*1024-retained)
                cache[key] = None, None
        if _metadata_weight((technical, scientific, cache, used, payloads)) > 1024*1024:
            raise ContractError('combined retained archive metadata exceeds allowance', 'budget_exceeded')
        return cache[key]
    commit, _ = read('telemetry', cutoff['commit'])
    envelope = schema_validator('telemetry-envelope.schema.json')
    _schema_check(envelope.evolve(schema={'$ref':'#/$defs/TelemetryCommit2', '$defs':envelope.schema['$defs']}), commit)
    summary, _ = read('telemetry', commit['summary'])
    catalog, _ = read('telemetry', commit['catalog'])
    validate_telemetry_catalog(catalog, expected_transport=commit['transport'])
    previous = previous_summary = previous_catalog = None
    if commit['previous'] is not None:
        pair = commit['previous']; previous, _ = read('telemetry', pair['commit']); previous_summary, _ = read('telemetry', pair['summary'])
        _schema_check(envelope.evolve(schema={'$ref':'#/$defs/TelemetryCommit2', '$defs':envelope.schema['$defs']}), previous)
        previous_catalog, _ = read('telemetry', previous['catalog'])
    contexts = []
    for row in commit['contexts']['phase_context_refs']:
        publication, _ = read('telemetry', row['retained_publication_ref']); segment, _ = read('telemetry', row['retained_segment_ref'])
        contexts.append(dict(publication=publication, publication_ref=row['publication_ref'], segment=segment, segment_ref=row['segment_ref']))
    resume = []
    for pair in commit['contexts']['resume_context_refs']:
        c, _ = read('telemetry', pair['commit']); s, _ = read('telemetry', pair['summary'])
        resume.append(dict(commit=c, commit_ref=pair['commit'], summary=s, summary_ref=pair['summary']))
    kwargs = {} if previous is None else dict(previous=previous, previous_ref=commit['previous']['commit'],
        previous_summary=previous_summary, previous_summary_ref=commit['previous']['summary'], previous_catalog=previous_catalog)
    validate_telemetry_commit(commit, summary=summary, summary_ref=commit['summary'], catalog=catalog,
        phase_contexts=contexts, resume_contexts=resume, **kwargs)
    if (cutoff['catalog'] != commit['catalog'] or cutoff.get('source_cursors') != commit['source_cursors']
            or cutoff['prior_capture'] != summary['cumulative']['prior_capture']
            or cutoff['cumulative_scope'] != summary['cumulative']['scope']):
        raise ContractError('retained cutoff differs from exact selected transaction', 'corrupt_result')
    for row in catalog['artifacts']:
        if row['ref']['media_type'] == 'application/vnd.apache.parquet': payloads[json_bytes(row['ref'])] = row
        value, _ = read('telemetry', row['ref'])
        if row['ref']['media_type'] == 'application/json':
            from .catalog import _retained_metadata
            _retained_metadata(value, row['ref'], expected_transport=commit['transport'])
    selected = manifest['source_state']; source, _ = read('scientific', selected['manifest']); validate_manifest(source)
    # Match a returned close only to this exact transaction, never a newest page.
    bindings = []
    for key, row in technical.items():
        if key[0] == 'telemetry' and row['reference']['schema'] == 'qcl-negf.telemetry-binding.v2':
            candidate, _ = read('telemetry', row['reference'], promise=False)
            _schema_check(envelope.evolve(schema={'$ref':'#/$defs/TelemetryBinding2', '$defs':envelope.schema['$defs']}), candidate)
            if isinstance(candidate.get('commit'), dict) and candidate['commit'].get('ref') == cutoff['commit']:
                bindings.append((row['reference'], candidate))
    if len(bindings) > 1 or commit['closed'] and len(bindings) != 1:
        raise ContractError('retained close binding is missing or ambiguous', 'corrupt_result')
    binding = bindings[0][1] if bindings else None
    if binding is not None:
        read('telemetry', bindings[0][0])
        proof_ref = binding['source_binding']['ref']; proof, _ = read('telemetry', proof_ref)
    else:
        candidates = []
        for key, row in technical.items():
            if key[0] == 'telemetry' and row['reference']['schema'] == 'qcl-negf.telemetry-source-binding.v1':
                candidate, _ = read('telemetry', row['reference'], promise=False)
                _schema_check(envelope.evolve(schema={'$ref':'#/$defs/TelemetrySourceBinding', '$defs':envelope.schema['$defs']}), candidate)
                if candidate.get('selected_state') == selected and candidate.get('enclosing_transport') == commit['transport']:
                    candidates.append((row['reference'], candidate))
        if len(candidates) != 1:
            raise ContractError('exact source binding is missing or ambiguous', 'corrupt_result')
        proof_ref, proof = candidates[0]
        read('telemetry', proof_ref)
    _schema_check(envelope.evolve(schema={'$ref':'#/$defs/TelemetrySourceBinding', '$defs':envelope.schema['$defs']}), proof)
    plan, _ = read('canonical_plan', proof['canonical_plan']); _schema_check(schema_validator('scientific-plan.schema.json'), plan)
    context = dict(expected_state=selected, expected_transport=commit['transport'], manifest=source,
        manifest_ref=selected['manifest'], source_transport=proof['source_transport'], canonical_plan_ref=proof['canonical_plan'],
        expected_producer_provenance=source['provenance'])
    validate_telemetry_source_binding(proof, **context)
    if binding is not None:
        validate_telemetry_binding(binding, commit=commit, commit_ref=cutoff['commit'], summary=summary,
            summary_ref=commit['summary'], source_binding=proof, source_binding_ref=proof_ref, **context)
    if used != set(technical):
        raise ContractError('extra unselected retained technical locator', 'corrupt_result')
    selected_objects = {r['object'] for r in manifest['source_locator']} | {r['object'] for r in manifest['technical_locator']}
    selected_objects.update(p['object'] for p in manifest.get('telemetry_compaction', []))
    if selected_objects != {r['path'] for r in index['objects']}:
        raise ContractError('extra unselected retained transport object', 'corrupt_result')
    # Existing compaction algorithm/encoding stays unchanged. Bind its declared
    # original order and exact row coverage to this authenticated after snapshot.
    for proof_row in manifest.get('telemetry_compaction', []):
        table = proof_row.get('table'); expected = []; originals = []; position = 0
        for row in catalog['artifacts']:
            if row['table'] == table:
                stop = position+int(row['rows'])
                expected.append(dict(sha256=row['ref']['sha256'], bytes=int(row['ref']['bytes']), row_start=position, row_stop=stop)); position=stop
                originals.append(technical[('telemetry', row['ref']['path'])]['object'])
        if (not expected or json_bytes(proof_row.get('source_segments')) != json_bytes(expected)
                or type(proof_row.get('rows')) is not int or proof_row['rows'] != position
                or type(proof_row.get('source_files')) is not int or proof_row['source_files'] != len(expected)
                or proof_row.get('source_commit_sha256') != cutoff['commit']['sha256']):
            raise ContractError('compaction is outside retained catalog order/row coverage', 'corrupt_result')
        # Reuse the unchanged fixed-4096 validity/native-bit codec on actual
        # contained streams. No temporary payload copies or source-root paths.
        from contextlib import contextmanager
        from .compaction import _CompactionRead, _payload_hash
        class ArchiveBits(_CompactionRead):
            @contextmanager
            def open(self, name):
                with open_object(name) as raw:
                    yield _BudgetStream(raw, self.budget)
        bits = ArchiveBits(32*1024*1024, budget)
        before = _payload_hash(originals, bits)
        after = _payload_hash([proof_row['object']], bits)
        if (before != after or before != proof_row.get('logical_payload_sha256')
                or after != proof_row.get('verified_payload_sha256')):
            raise ContractError('contained compaction changed retained scalar bits', 'corrupt_result')
    _bounded_objects((manifest['technical_locator'], cache, commit, summary, catalog, previous, previous_summary, previous_catalog, contexts, resume, proof, binding, plan), 16384)
    return dict(technical_contract='qcl-negf.telemetry-retained.v2', scope='retained_technical_window',
        commit=cutoff['commit'], catalog=cutoff['catalog'], cumulative_scope=cutoff['cumulative_scope'],
        prior_capture=cutoff['prior_capture'], provenance_origin='not_authenticated_offline',
        historical_catalog_descriptors='opaque_not_recursive_retention_promises')


if __name__ == "__main__":
    main()
