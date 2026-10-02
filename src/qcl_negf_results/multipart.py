"""Read and independently verify legacy qcl-negf.export-parts.v1 sets.

New producers use qcl_negf_results.archive and never paginate. This receiver
retains compatibility with archived multipart data, including chunk checksums.
"""
from __future__ import annotations

import argparse
from collections.abc import Sequence
import hashlib
import json
import lzma
import os
from pathlib import Path
import tarfile
import tempfile
from typing import Any

from qcl_negf_contracts.artifacts import CONTRACT_SET, EXPORT_PART_MAX_BYTES, digest_value, relative_path
from qcl_negf_contracts.messages import ContractError
from .commits import json_bytes

INDEX_SCHEMA = "qcl-negf.export-parts.v1"
PART_SCHEMA = "qcl-negf.export-part.v1"
BLOCK = 1024 * 1024


def _read_json_member(archive: tarfile.TarFile, name: str) -> dict[str, Any]:
    try:
        member = archive.getmember(name)
    except KeyError as error:
        raise ContractError(f"part is missing {name}", "corrupt_result") from error
    if not member.isfile():
        raise ContractError("transport metadata must be a regular file", "corrupt_result")
    stream = archive.extractfile(member)
    assert stream is not None
    value = json.load(stream)
    if not isinstance(value, dict):
        raise ContractError("transport metadata must be an object", "corrupt_result")
    return value


def _integer(value: Any, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ContractError("invalid transport integer", "corrupt_result")
    return value


def _validate_header(header: dict[str, Any], size: int) -> int:
    if header.get("schema") != PART_SCHEMA or header.get("contract_set") != CONTRACT_SET:
        raise ContractError("unsupported multipart transport", "incompatible_contract")
    number = _integer(header.get("index"), 1)
    count = _integer(header.get("count"), number)
    maximum = _integer(header.get("maximum_part_bytes"), 1)
    if maximum > EXPORT_PART_MAX_BYTES or size > maximum:
        raise ContractError("part exceeds its declared byte limit", "corrupt_result")
    digest_value(header.get("snapshot_identity"))
    digest_value(header.get("index_sha256"))
    assert count >= number
    return number


def _validate_part_set(index: dict[str, Any], parts: dict[int, tuple[Path, dict[str, Any]]]) -> None:
    if index.get("schema") != INDEX_SCHEMA or index.get("contract_set") != CONTRACT_SET:
        raise ContractError("unsupported shared transport index", "incompatible_contract")
    count = _integer(index.get("part_count"), 1)
    if len(parts) != count or set(parts) != set(range(1, count + 1)):
        missing = [str(i) for i in range(1, min(count + 1, 65)) if i not in parts]
        raise ContractError(f"missing export parts: received {len(parts)} of {count}; " +
                            ", ".join(missing), "corrupt_result")
    digest = hashlib.sha256(json_bytes(index)).hexdigest()
    for _, header in parts.values():
        if (header.get("snapshot_identity") != index.get("snapshot_identity") or
            header.get("index_sha256") != digest or header.get("count") != count or
            header.get("maximum_part_bytes") != index.get("maximum_part_bytes")):
            raise ContractError("parts belong to different snapshots or indexes", "corrupt_result")


def _read_headers(paths: Sequence[Path]) -> tuple[dict[str, Any], dict[int, tuple[Path, dict[str, Any]]]]:
    parts: dict[int, tuple[Path, dict[str, Any]]] = {}
    index = None
    for path in paths:
        if path.stat().st_size > EXPORT_PART_MAX_BYTES:
            raise ContractError("part exceeds 200000000 bytes", "corrupt_result")
        with tarfile.open(path, "r:xz") as archive:
            header = _read_json_member(archive, "part.json")
            number = _validate_header(header, path.stat().st_size)
            if number in parts:
                raise ContractError(f"duplicate part {number}", "corrupt_result")
            parts[number] = (path, header)
            if number == 1:
                index = _read_json_member(archive, "export-index.json")
    if index is None:
        raise ContractError("missing diagnostic part 1", "corrupt_result")
    _validate_part_set(index, parts)
    return index, parts


def _validate_receipt(receipt: dict[str, Any], index: dict[str, Any],
                      parts: dict[int, tuple[Path, dict[str, Any]]]) -> None:
    expected_parts = _rows(receipt.get("parts"))
    if (len(expected_parts) != len(parts) or
        {row.get("index") for row in expected_parts} != set(parts) or
        receipt.get("snapshot_identity") != index.get("snapshot_identity")):
        raise ContractError("receipt does not describe these parts", "corrupt_result")
    for row in expected_parts:
        path, _ = parts[_integer(row.get("index"), 1)]
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != digest_value(row.get("sha256")):
                raise ContractError("archive checksum differs from receipt", "corrupt_result")
        if path.stat().st_size != _integer(row.get("bytes")):
            raise ContractError("archive size differs from receipt", "corrupt_result")


Inventory = dict[int, dict[str, tuple[dict[str, Any], dict[str, Any] | None]]]


def _rows(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ContractError("transport inventory must contain object rows", "corrupt_result")
    return value


def _object_segments(row: dict[str, Any], expected: Inventory) -> None:
    offset = 0
    digest_value(row.get("sha256"))
    for segment in _rows(row.get("segments")):
        member = relative_path(segment.get("member"))
        part = _integer(segment.get("part"), 1)
        count = _integer(segment.get("bytes"))
        if part not in expected or member in expected[part] or _integer(segment.get("offset")) != offset:
            raise ContractError("noncontiguous or duplicate object segments", "corrupt_result")
        if not member.startswith(("objects/", "chunks/")):
            raise ContractError("invalid transport object member", "corrupt_result")
        digest_value(segment.get("sha256"))
        expected[part][member] = (segment, row)
        offset += count
    if offset != _integer(row.get("bytes")) or not row["segments"]:
        raise ContractError("object segments do not cover the declared size", "corrupt_result")


def _inventory(index: dict[str, Any], parts: dict[int, tuple[Path, dict[str, Any]]]) -> Inventory:
    expected: Inventory = {i: {} for i in parts}
    names: set[str] = set()
    for row in _rows(index.get("objects")):
        name = relative_path(row.get("path"))
        if name in names or not name.startswith("objects/"):
            raise ContractError("duplicate or invalid object name", "corrupt_result")
        names.add(name)
        _object_segments(row, expected)
    for row in _rows(index.get("metadata")):
        name = relative_path(row.get("path"))
        if name in expected[1] or name not in {"manifest.json", "plan.json", "diagnostics.json", "README.md"}:
            raise ContractError("duplicate or unknown transport metadata", "corrupt_result")
        _integer(row.get("bytes"))
        digest_value(row.get("sha256"))
        expected[1][name] = (row, None)
    if "manifest.json" not in expected[1]:
        raise ContractError("index omitted the mandatory manifest", "corrupt_result")
    return expected


def _restore_member(archive: tarfile.TarFile, member: tarfile.TarInfo, root: Path,
                    segment: dict[str, Any], object_row: dict[str, Any] | None) -> None:
    if member.size != segment["bytes"]:
        raise ContractError("transport member size mismatch", "corrupt_result")
    source = archive.extractfile(member)
    assert source is not None
    name = object_row["path"] if object_row is not None else member.name
    target = root / relative_path(name)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("r+b" if target.exists() else "wb") as output:
        output.seek(segment.get("offset", 0))
        digest = hashlib.sha256()
        while block := source.read(BLOCK):
            digest.update(block)
            output.write(block)
    if digest.hexdigest() != segment["sha256"]:
        raise ContractError("transport member checksum mismatch", "corrupt_result")


def _restore_part(path: Path, number: int, expected: Inventory, root: Path) -> None:
    remaining = dict(expected[number])
    allowed = {"part.json", "export-index.json"} if number == 1 else {"part.json"}
    seen: set[str] = set()
    with tarfile.open(path, "r:xz") as archive:
        for member in archive:
            if not member.isfile() or member.name in seen:
                raise ContractError("invalid or duplicate final archive member", "corrupt_result")
            seen.add(member.name)
            if member.name in allowed:
                continue
            if member.name not in remaining:
                raise ContractError("archive contains an unindexed member", "corrupt_result")
            segment, object_row = remaining.pop(member.name)
            _restore_member(archive, member, root, segment, object_row)
        if remaining or not allowed <= seen:
            raise ContractError("archive omitted mandatory members", "corrupt_result")
        assert archive.fileobj is not None
        while archive.fileobj.read(BLOCK):
            pass  # Force XZ footer verification after tar padding.


def _verify_restored(root: Path, index: dict[str, Any]) -> None:
    for row in index["objects"]:
        path = root / row["path"]
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != row["sha256"] or path.stat().st_size != row["bytes"]:
            raise ContractError("reassembled object checksum mismatch", "corrupt_result")
    manifest = json.loads((root / "manifest.json").read_bytes())
    inventory = {row["path"]: row for row in index["objects"]}
    if manifest.get("snapshot_identity") != index["snapshot_identity"]:
        raise ContractError("manifest and transport identify different snapshots", "corrupt_result")
    if {row["path"] for row in manifest["files"]} != set(inventory):
        raise ContractError("manifest and transport object inventories differ", "corrupt_result")
    for row in manifest["files"]:
        if row["sha256"] != inventory[row["path"]]["sha256"] or row["bytes"] != inventory[row["path"]]["bytes"]:
            raise ContractError("manifest object identity differs from transport", "corrupt_result")
    for record in manifest["records"]:
        for row in record["included"]:
            if row["object"] not in inventory or any(dep not in inventory for dep in row["dependencies"]):
                raise ContractError("restored snapshot has dangling dependencies", "corrupt_result")


def receive(paths: Sequence[Path], destination: Path | None = None,
            *, receipt: dict[str, Any] | None = None) -> dict[str, Any]:
    """Verify all parts and optionally restore native objects without a giant tar.

    Input order is immaterial. Missing, duplicate, corrupt and mixed-generation
    parts are rejected. Reassembly is published as one new directory only after
    both transport-member and complete-object hashes pass.
    """
    index, parts = _read_headers(paths)
    if receipt is not None:
        _validate_receipt(receipt, index, parts)
    expected = _inventory(index, parts)
    if destination is not None:
        destination = Path(destination)
        if destination.exists():
            raise ContractError("reassembly destination already exists")
        destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".qcl-receive-", dir=destination.parent if destination else None) as temporary:
        root = Path(temporary) / "restored"
        root.mkdir()
        for number, (path, _) in sorted(parts.items()):
            _restore_part(path, number, expected, root)
        _verify_restored(root, index)
        (root / "export-index.json").write_bytes(json_bytes(index))
        if destination is not None:
            os.replace(root, destination)
    return {"snapshot_identity": index["snapshot_identity"], "verified": True,
            "part_count": len(parts), "objects": len(index["objects"]),
            "destination": str(destination) if destination is not None else None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("verify", "reassemble"))
    parser.add_argument("parts", nargs="+", type=Path)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    if args.action == "reassemble" and args.destination is None:
        parser.error("reassemble requires --destination")
    if args.action == "verify" and args.destination is not None:
        parser.error("verify does not accept --destination")
    try:
        receipt = json.loads(args.receipt.read_bytes()) if args.receipt else None
        print(json.dumps(receive(args.parts, args.destination, receipt=receipt), indent=2))
    except (ContractError, OSError, ValueError, EOFError, tarfile.TarError, lzma.LZMAError) as error:
        parser.exit(2, str(error) + "\n")


if __name__ == "__main__":
    main()
