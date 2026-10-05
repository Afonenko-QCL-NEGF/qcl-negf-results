"""Streaming single-archive transport, with a receiver for legacy multipart sets.

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


def _receipt_closure(receipt: dict[str, Any] | None, index: dict[str, Any]) -> None:
    if receipt is not None and (
            receipt["snapshot_identity"] != index["snapshot_identity"]
            or receipt["profile"] != index["profile"]
            or receipt["transport_schema"] != index["schema"]):
        raise ContractError("receipt and archive identify different snapshots or profiles", "corrupt_result")


def _read_archive(path: Path, root: Path | None, report: Callable[..., None]) -> dict[str, Any]:
    with tarfile.open(path, "r:xz") as archive:
        first = archive.next()
        if first is None or first.name != INDEX_NAME or not first.isfile():
            raise ContractError("archive omitted its leading transport index", "corrupt_result")
        source = archive.extractfile(first)
        assert source is not None
        index_payload = source.read()
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
            except (OSError, ValueError) as error:
                raise ContractError(f"invalid HDF5 archive object: {error}", "corrupt_result") from error
            finally:
                if root is None:
                    source.close()
    if root is not None:
        (root / INDEX_NAME).write_bytes(index_payload)
    return index


def verify_archive(path: Path, *, report: Callable[..., None] = _quiet) -> dict[str, Any]:
    """Verify hashes and self-contained HDF5 storage without a second payload tree.

    Random access to compressed members may repeat decompression; native arrays
    remain on disk and no numerical values are read for this ownership check.
    """
    return _read_archive(path, None, report)


def receive(paths: Sequence[Path], destination: Path | None = None, *,
            receipt: dict[str, Any] | None = None) -> dict[str, Any]:
    """Verify/restore one new archive, or independently verify a legacy part set."""
    paths = [Path(path) for path in paths]
    if not paths:
        raise ContractError("no export archives supplied", "corrupt_result")
    with tarfile.open(paths[0], "r|xz") as archive:
        first = archive.next()
        legacy = first is not None and first.name == "part.json"
    if legacy:
        from .multipart import receive as receive_legacy
        return receive_legacy(paths, destination, receipt=receipt)
    if len(paths) != 1:
        raise ContractError("single archive transport requires exactly one archive", "corrupt_result")
    path = paths[0]
    if receipt is not None:
        validate_export_receipt(receipt)
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != receipt["sha256"]:
                raise ContractError("archive checksum differs from receipt", "corrupt_result")
        if path.stat().st_size != receipt["bytes"]:
            raise ContractError("archive size differs from receipt", "corrupt_result")
    if destination is None:
        index = verify_archive(path)
        _receipt_closure(receipt, index)
    else:
        destination = Path(destination)
        if destination.exists():
            raise ContractError("restore destination already exists")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".qcl-receive-", dir=destination.parent) as temporary:
            root = Path(temporary) / "restored"
            root.mkdir()
            index = _read_archive(path, root, _quiet)
            _receipt_closure(receipt, index)
            os.replace(root, destination)
    return {"snapshot_identity": index["snapshot_identity"], "verified": True,
            "verification_scope": "transport_hashes_and_hdf5_storage",
            "archive_count": 1, "objects": len(index["objects"]),
            "destination": str(destination) if destination is not None else None}


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


if __name__ == "__main__":
    main()
