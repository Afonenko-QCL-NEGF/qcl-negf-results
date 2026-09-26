"""Operational evidence capture using the scientific multipart transport.

The caller owns source selection and safe file descriptors. This module owns
bounded capture, content identities, archive framing and independent verification.
No queue, API or scientific execution implementation is imported here.
"""
from __future__ import annotations

from collections.abc import Callable
import hashlib
import os
from pathlib import Path
import shutil
import tarfile
import tempfile
from typing import Any, BinaryIO, Protocol

from qcl_negf_contracts.artifacts import CONTRACT_SET, SCIENCE_MAX_BYTES, relative_path
from qcl_negf_contracts.messages import ContractError
from .commits import json_bytes
from .multipart import BLOCK, build_parts, verify_parts

DIAGNOSTIC_SCHEMA = "qcl-negf-operational-evidence.v1"


class ArchiveSink(Protocol):
    """The small archive-writing port also implemented by tarfile.TarFile."""

    def addfile(self, tarinfo: tarfile.TarInfo, fileobj: Any = None) -> None: ...


def _quiet(phase: str, **values: Any) -> None:
    pass


class DiagnosticCapture:
    """Capture sources once with bounded IO buffers/disk; inventory scales with files."""

    def __init__(self, spool: Path, *, maximum_source_bytes: int):
        if maximum_source_bytes < 1:
            raise ValueError("diagnostic source byte limit must be positive")
        self.spool = spool
        self.maximum_source_bytes = maximum_source_bytes
        self.source_bytes = 0
        self.files: dict[str, tuple[Path, dict[str, Any]]] = {}
        self.entries: list[dict[str, Any]] = []
        self._names: set[str] = set()

    def addfile(self, tarinfo: tarfile.TarInfo, fileobj: Any = None) -> None:
        name = relative_path(tarinfo.name)
        if not tarinfo.isfile() or fileobj is None or name in self._names or tarinfo.size < 0:
            raise ContractError("diagnostic capture requires unique regular files")
        if self.source_bytes + tarinfo.size > self.maximum_source_bytes:
            raise ContractError("diagnostic source exceeds temporary storage budget; no partial capture published", "storage_failure")
        target = self.spool / "capture.tmp"
        digest = self._copy(fileobj, target, tarinfo.size)
        object_name = f"objects/{digest}.bin"
        object_path = self.spool / object_name
        object_path.parent.mkdir(exist_ok=True)
        os.replace(target, object_path)
        description = {"path": object_name, "bytes": tarinfo.size, "sha256": digest}
        self.files[object_name] = (object_path, description)
        self.entries.append({"source_path": name, "object": object_name,
                             "role": "diagnostic.evidence", "dependencies": []})
        self.source_bytes += tarinfo.size
        self._names.add(name)

    @staticmethod
    def _copy(source: BinaryIO, target: Path, count: int) -> str:
        digest = hashlib.sha256()
        with target.open("wb") as stream:
            remaining = count
            while remaining:
                value = source.read(min(remaining, BLOCK))
                if not value:
                    raise OSError("diagnostic source ended during capture")
                stream.write(value)
                digest.update(value)
                remaining -= len(value)
        return digest.hexdigest()

    def manifest(self, context: dict[str, Any]) -> dict[str, Any]:
        records = [{"record_id": "operational-evidence", "included": self.entries}]
        files = [description for _, description in sorted(self.files.values(), key=lambda row: row[1]["path"])]
        identity = hashlib.sha256(json_bytes({"context": context, "files": files, "records": records})).hexdigest()
        return {"schema": DIAGNOSTIC_SCHEMA, "contract_set": CONTRACT_SET,
                "profile": "diagnostic", "snapshot_identity": identity,
                "context": context, "files": files, "records": records,
                "source_consistency": "captured regular files; a live job is not an atomic scientific checkpoint",
                "scientific_acceptance": "not asserted by operational evidence capture"}


def _metadata(manifest: dict[str, Any]) -> dict[str, bytes]:
    return {"manifest.json": json_bytes(manifest), "README.md": (
        "# QCLNEGF operational evidence\n\n"
        "Every finalized part is at most 200000000 decimal bytes. Keep all parts.\n"
        "The first part contains this manifest and the shared transport index.\n"
        "manifest.json records the original evidence paths in records[].included[].source_path,\n"
        "mapped to immutable content-addressed objects. The captured manifest.yaml object\n"
        "lists unavailable and excluded source files and the capture time.\n"
        "This is operational evidence; live files are not an atomic scientific checkpoint.\n\n"
        "Verify: python -m qcl_negf_results.multipart verify PART...\n"
        "Restore: python -m qcl_negf_results.multipart reassemble --destination recovered PART...\n"
        "The same receiver accepts scientific and diagnostic transport parts.\n"
    ).encode()}


def export_diagnostics(destination: Path, capture: Callable[[ArchiveSink], dict[str, Any]], *,
                       label: str, maximum_bytes: int = SCIENCE_MAX_BYTES,
                       maximum_source_bytes: int | None = None,
                       progress: Callable[..., None] = _quiet) -> dict[str, Any]:
    """Publish a new directory only after every captured object and page verifies.

    Temporary data belongs to the caller, never the queue. An incomplete capture
    is removed on every failure. Compression uses one worker and XZ preset 1.
    """
    if destination.exists():
        raise FileExistsError("diagnostic destination already exists")
    if not label or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for character in label):
        raise ValueError("invalid diagnostic filename label")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".diagnostic-", dir=destination.parent) as temporary:
        spool = Path(temporary)
        free_budget = max(0, (shutil.disk_usage(spool).free - 64 * 1024**2) // 3)
        source_budget = free_budget if maximum_source_bytes is None else min(maximum_source_bytes, free_budget)
        if source_budget < 1:
            raise ContractError("insufficient temporary disk space for diagnostic capture", "storage_failure")
        sink = DiagnosticCapture(spool, maximum_source_bytes=source_budget)
        progress("capturing")
        context = capture(sink)
        manifest = sink.manifest(context)
        metadata = _metadata(manifest)
        # Exact source inventory is now stable. Reserve space for compression
        # and bounded metadata before making any public result visible.
        if shutil.disk_usage(spool).free < sink.source_bytes + 64 * 1024**2:
            raise ContractError("insufficient temporary disk space for diagnostic parts", "storage_failure")
        finalized, index = build_parts(spool, sink.files, metadata, records=manifest["records"],
            identity=manifest["snapshot_identity"], profile="diagnostic", maximum=maximum_bytes,
            preset=1, report=progress)
        verify_parts(finalized, index, metadata, progress)
        publication = spool / "published"
        publication.mkdir()
        parts = []
        for path, part in finalized:
            filename = f"{label}.part-{part['index']:04d}-of-{len(finalized):04d}.tar.xz"
            os.chmod(path, 0o640)
            os.replace(path, publication / filename)
            parts.append({**part, "filename": filename})
        receipt = {"schema": DIAGNOSTIC_SCHEMA, "contract_set": CONTRACT_SET,
                   "profile": "diagnostic", "snapshot_identity": manifest["snapshot_identity"],
                   "parts": parts, "part_count": len(parts), "maximum_part_bytes": maximum_bytes,
                   "total_archive_bytes": sum(part["bytes"] for part in parts),
                   "source_payload_bytes": sink.source_bytes, "context": context}
        (publication / "receipt.json").write_bytes(json_bytes(receipt))
        os.rename(publication, destination)
    return receipt
