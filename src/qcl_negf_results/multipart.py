"""Bounded transport shared by scientific and operational diagnostic snapshots.

The scientific manifest describes immutable objects. The transport index maps
those objects to whole members or explicitly ordered chunks without changing
native arrays. Archive hashes live in the receipt, avoiding a circular hash of
the first archive which contains the shared object index.
"""
from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
import hashlib
import io
import json
import lzma
import os
from pathlib import Path
import tarfile
import tempfile
from typing import Any, IO, cast

from qcl_negf_contracts.artifacts import CONTRACT_SET, SCIENCE_MAX_BYTES, digest_value, relative_path
from qcl_negf_contracts.messages import ContractError
from .commits import json_bytes

INDEX_SCHEMA = "qcl-negf.export-parts.v1"
PART_SCHEMA = "qcl-negf.export-part.v1"
BLOCK = 1024 * 1024


class _PartTooLarge(Exception):
    pass


class _BoundedWriter(io.BufferedIOBase):
    def __init__(self, stream: Any, maximum: int) -> None:
        self.stream, self.maximum, self.count = stream, maximum, 0

    def writable(self) -> bool:
        return True

    def write(self, value: Any) -> int:
        if self.count + len(value) > self.maximum:
            raise _PartTooLarge
        written: int = self.stream.write(value)
        self.count += written
        return written

    def flush(self) -> None:
        if not self.stream.closed:
            self.stream.flush()


@dataclass(frozen=True)
class Member:
    name: str
    source: Path
    offset: int
    size: int
    sha256: str
    object_name: str
    object_size: int
    object_sha256: str
    weight: int
    priority: int = 10

    def descriptor(self) -> dict[str, Any]:
        return {"member": self.name, "offset": self.offset, "bytes": self.size,
                "sha256": self.sha256}


class _SliceReader:
    def __init__(self, stream: IO[bytes], count: int) -> None:
        self.stream, self.remaining = stream, count

    def read(self, size: int = -1) -> bytes:
        block = self.stream.read(self.remaining if size < 0 else min(size, self.remaining))
        self.remaining -= len(block)
        return block


def _add_bytes(archive: tarfile.TarFile, name: str, value: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size, info.mode = len(value), 0o640
    archive.addfile(info, io.BytesIO(value))


def _write_page(path: Path, metadata: dict[str, bytes], members: list[Member],
                maximum: int, preset: int) -> int:
    # Enforce the actual complete stream, including its footer; an unsuccessful
    # attempt never creates even a temporary archive larger than the cap.
    with path.open("wb") as raw:
        sink = _BoundedWriter(raw, maximum)
        with lzma.LZMAFile(cast(IO[bytes], sink), "wb", preset=preset,
                           check=lzma.CHECK_CRC64) as compressed:
            with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT) as archive:
                for name, payload in metadata.items():
                    _add_bytes(archive, name, payload)
                for member in members:
                    info = tarfile.TarInfo(member.name)
                    info.size, info.mode = member.size, 0o640
                    with member.source.open("rb") as source:
                        source.seek(member.offset)
                        archive.addfile(info, _SliceReader(source, member.size))
        raw.flush()
        os.fsync(raw.fileno())
    return path.stat().st_size


def _chunks(member: Member, size: int) -> list[Member]:
    result = []
    with member.source.open("rb") as stream:
        stream.seek(member.offset)
        end = member.offset + member.size
        offset = member.offset
        while offset < end:
            count = min(size, end - offset)
            digest, remaining = hashlib.sha256(), count
            while remaining:
                block = stream.read(min(remaining, BLOCK))
                if not block:
                    raise ContractError("captured object truncated during pagination", "corrupt_result")
                digest.update(block)
                remaining -= len(block)
            result.append(replace(member,
                name=f"chunks/{member.object_sha256}/{offset:016d}.bin",
                offset=offset, size=count, sha256=digest.hexdigest(), weight=count + 1024))
            offset += count
    return result


def _index(identity: str, profile: str, maximum: int, pages: list[list[Member]],
           metadata: dict[str, bytes]) -> dict[str, Any]:
    objects: dict[str, dict[str, Any]] = {}
    for number, page in enumerate(pages, 1):
        for member in page:
            row = objects.setdefault(member.object_name, {"path": member.object_name,
                "bytes": member.object_size, "sha256": member.object_sha256, "segments": []})
            row["segments"].append({**member.descriptor(), "part": number})
    for row in objects.values():
        row["segments"].sort(key=lambda item: item["offset"])
        row["transport"] = "whole" if (len(row["segments"]) == 1 and
            row["segments"][0]["member"] == row["path"]) else "chunks"
    return {"schema": INDEX_SCHEMA, "contract_set": CONTRACT_SET,
        "snapshot_identity": identity, "profile": profile, "part_count": len(pages),
        "maximum_part_bytes": maximum, "index_part": 1,
        "metadata": [{"path": name, "bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}
                     for name, value in metadata.items()],
        "objects": list(objects.values()),
        "archive_hashes": "finalized part hashes are in the export receipt; object and chunk hashes are here",
        "receive": "python -m qcl_negf_results.multipart reassemble --destination recovered PART..."}


def _priorities(records: list[dict[str, Any]]) -> dict[str, int]:
    result: dict[str, int] = {}
    for record in records:
        for item in record["included"]:
            role = item["role"]
            priority = (0 if role in {"model", "plan", "science.comparison", "performance.summary", "control.state"}
                        else 1 if role == "science.history"
                        else 3 if role in {"performance.window", "performance.full"} else 5)
            result[item["object"]] = min(priority, result.get(item["object"], 10))
    return result


def _check_metadata_fits(spool: Path, metadata: dict[str, bytes], maximum: int, preset: int) -> None:
    probe = spool / ".part-probe.tar.xz"
    try:
        _write_page(probe, metadata, [], maximum, preset)
    except _PartTooLarge as error:
        raise ContractError("diagnostic index/metadata alone exceeds the requested part limit; "
                            "increase the limit up to 200000000 bytes", "storage_failure") from error
    finally:
        probe.unlink(missing_ok=True)


def _transport_units(spool: Path, files: dict[str, tuple[Path, dict[str, Any]]],
                     priorities: dict[str, int], budget: int, preset: int) -> list[Member]:
    units: list[Member] = []
    probe = spool / ".part-probe.tar.xz"
    for name, (path, description) in sorted(files.items(), key=lambda item: (priorities.get(item[0], 10), item[0])):
        size = path.stat().st_size
        member = Member(name, path, 0, size, description["sha256"], name, size,
                        description["sha256"], size + 1024, priorities.get(name, 10))
        if member.weight <= budget:
            units.append(member)
            continue
        # Preserve even very large whole objects when their real compressed
        # stream fits. Incompressible objects use explicit byte-range chunks.
        try:
            compressed = _write_page(probe, {}, [member], budget, preset)
            units.append(replace(member, weight=compressed + 1024))
        except _PartTooLarge:
            units.extend(_chunks(member, max(1, budget - 1024)))
        finally:
            probe.unlink(missing_ok=True)
    return units


def _layout(units: list[Member], metadata: dict[str, bytes], budget: int) -> list[list[Member]]:
    pages: list[list[Member]] = [[]]
    used = sum(len(payload) + 1024 for payload in metadata.values())
    for member in units:
        if (pages[-1] or len(pages) == 1) and used + member.weight > budget:
            pages.append([])
            used = 0
        pages[-1].append(member)
        used += member.weight
    return pages


def _refine_layout(pages: list[list[Member]], overflow: int) -> None:
    page = pages[overflow]
    if not page:
        raise ContractError("diagnostic index/metadata alone exceeds the requested part limit; "
                            "increase the limit up to 200000000 bytes", "storage_failure")
    if len(page) > 1:
        split = max(1, len(page) // 2)
        pages[overflow:overflow + 1] = [page[:split], page[split:]]
    elif overflow == 0:
        pages[0:1] = [[], page]
    else:
        member = page[0]
        if member.size <= 1:
            raise ContractError("part limit cannot contain transport headers", "storage_failure")
        pages[overflow:overflow + 1] = [[item] for item in _chunks(member, max(1, member.size // 2))]


def part_header(index: dict[str, Any], number: int, index_digest: str | None = None) -> bytes:
    return json_bytes({"schema": PART_SCHEMA, "contract_set": CONTRACT_SET,
        "snapshot_identity": index["snapshot_identity"],
        "index_sha256": index_digest or hashlib.sha256(json_bytes(index)).hexdigest(),
        "index": number, "count": index["part_count"], "maximum_part_bytes": index["maximum_part_bytes"]})


def _encode_parts(spool: Path, pages: list[list[Member]], metadata: dict[str, bytes],
                  index: dict[str, Any], preset: int, report: Callable[..., None]
                  ) -> tuple[list[tuple[Path, dict[str, Any]]], int | None]:
    finalized: list[tuple[Path, dict[str, Any]]] = []
    maximum = index["maximum_part_bytes"]
    index_payload = json_bytes(index)
    index_digest = hashlib.sha256(index_payload).hexdigest()
    total = sum(m.size for page in pages for m in page)
    completed_bytes, archive_bytes = 0, 0
    for number, members in enumerate(pages, 1):
        page_metadata = {"part.json": part_header(index, number, index_digest)}
        if number == 1:
            page_metadata.update(metadata)
            page_metadata["export-index.json"] = index_payload
        target = spool / f"part-{number:06d}.tar.xz"
        try:
            size = _write_page(target, page_metadata, members, maximum, preset)
        except _PartTooLarge:
            target.unlink(missing_ok=True)
            return finalized, number - 1
        with target.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        finalized.append((target, {"index": number, "count": len(pages), "bytes": size,
            "size_bytes": size, "sha256": digest, "status": "ready", "diagnostics": number == 1}))
        completed_bytes += sum(m.size for m in members)
        archive_bytes += size
        report("compressing", completed_bytes=completed_bytes, total_bytes=total,
               archive_bytes=archive_bytes, archive_limit_bytes=maximum,
               completed_parts=number, total_parts=len(pages))
    return finalized, None


def build_parts(spool: Path, files: dict[str, tuple[Path, dict[str, Any]]],
                metadata: dict[str, bytes], *, records: list[dict[str, Any]],
                identity: str, profile: str, maximum: int, preset: int,
                report: Callable[..., None]) -> tuple[list[tuple[Path, dict[str, Any]]], dict[str, Any]]:
    """Plan whole objects first, split only oversized transport, finalize bounded pages."""
    if not isinstance(maximum, int) or isinstance(maximum, bool) or not 0 < maximum <= SCIENCE_MAX_BYTES:
        raise ContractError("part limit must be within 1..200000000 bytes")
    report("compressing", completed_bytes=0, total_bytes=sum(p.stat().st_size for p, _ in files.values()),
           archive_bytes=0, archive_limit_bytes=maximum, completed_parts=0, total_parts=None)
    try:
        _check_metadata_fits(spool, metadata, maximum, preset)
        budget = max(1024, maximum * 9 // 10 - 2048)
        units = _transport_units(spool, files, _priorities(records), budget, preset)
        pages = _layout(units, metadata, budget)
        while True:
            index = _index(identity, profile, maximum, pages, metadata)
            finalized, overflow = _encode_parts(spool, pages, metadata, index, preset, report)
            if overflow is None:
                return finalized, index
            _refine_layout(pages, overflow)
    except ContractError:
        report("failed", reason="archive_size_limit", archive_limit_bytes=maximum, published=False)
        raise


def _verify_archive(path: Path, expected_hashes: dict[str, str],
                    on_bytes: Callable[[int], None]) -> None:
    """Validate the final transport independently, including the XZ footer."""
    expected = set(expected_hashes)
    verified_bytes = 0
    with tarfile.open(path, "r:xz") as archive:
        for member in archive:
            if not member.isfile() or member.name not in expected:
                raise ContractError("invalid final archive member", "corrupt_result")
            expected.remove(member.name)
            member_stream = archive.extractfile(member)
            assert member_stream is not None
            member_digest = hashlib.sha256()
            while chunk := member_stream.read(BLOCK):
                member_digest.update(chunk)
                verified_bytes += len(chunk)
                on_bytes(verified_bytes)
            if member_digest.hexdigest() != expected_hashes[member.name]:
                raise ContractError("final archive member checksum mismatch", "corrupt_result")
        if expected:
            raise ContractError("final archive omitted mandatory members", "corrupt_result")
        # Tar ends before transport padding. Reading the underlying XZ stream to
        # EOF forces its integrity check even when all member hashes were valid.
        assert archive.fileobj is not None
        while archive.fileobj.read(BLOCK):
            pass


def verify_parts(finalized: list[tuple[Path, dict[str, Any]]], index: dict[str, Any],
                        metadata: dict[str, bytes], report: Callable[..., None]) -> int:
    index_payload = json_bytes(index)
    index_digest = hashlib.sha256(index_payload).hexdigest()
    verify_total = sum(row["bytes"] for row in index["metadata"]) + sum(row["bytes"] for row in index["objects"])
    verified_bytes, transport_metadata_bytes = 0, len(index_payload)
    report("verifying", completed_bytes=0, total_bytes=verify_total,
           completed_parts=0, total_parts=len(finalized))
    for path, part in finalized:
        number = part["index"]
        header = part_header(index, number, index_digest)
        transport_metadata_bytes += len(header)
        expected = {"part.json": hashlib.sha256(header).hexdigest()}
        segments = [segment for row in index["objects"] for segment in row["segments"] if segment["part"] == number]
        expected.update({segment["member"]: segment["sha256"] for segment in segments})
        if number == 1:
            expected.update({name: hashlib.sha256(value).hexdigest() for name, value in metadata.items()})
            expected["export-index.json"] = index_digest
            verified_bytes += sum(row["bytes"] for row in index["metadata"])
        _verify_archive(path, expected, lambda _: None)
        verified_bytes += sum(segment["bytes"] for segment in segments)
        report("verifying", completed_bytes=verified_bytes, total_bytes=verify_total,
               completed_parts=number, total_parts=len(finalized))
    return transport_metadata_bytes


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
    if maximum > SCIENCE_MAX_BYTES or size > maximum:
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
        if path.stat().st_size > SCIENCE_MAX_BYTES:
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
