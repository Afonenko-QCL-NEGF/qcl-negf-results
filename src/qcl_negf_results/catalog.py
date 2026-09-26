"""Hash-linked, append-only telemetry inventory with a committed watermark.

Each chunk is written once. A snapshot pins its terminal chunk, entry count and
hash; later appends cannot change the prefix. The session index is only a locator.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from qcl_negf_contracts.artifacts import Artifact, digest_value, relative_path, require_contract_set
from qcl_negf_contracts.messages import ContractError
from .commits import read_json, safe_path

CATALOG_SCHEMA = "qcl-negf.telemetry-catalog.v2"


def catalog_artifacts(root: Path, watermark: dict[str, Any]) -> tuple[Artifact, ...]:
    """Read the exact closed prefix and reject truncation, substitution or cycles."""
    count = watermark.get("entries")
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise ContractError("invalid telemetry catalog watermark", "corrupt_result")
    chunks: list[tuple[Artifact, ...]] = []
    reference: Any = watermark
    seen: set[str] = set()
    expected_entries = count
    while reference is not None:
        if not isinstance(reference, dict):
            raise ContractError("invalid telemetry catalog reference", "corrupt_result")
        name = relative_path(reference.get("path"))
        if name in seen:
            raise ContractError("telemetry catalog cycle", "corrupt_result")
        seen.add(name)
        value, payload = read_json(safe_path(root, name))
        require_contract_set(value)
        if hashlib.sha256(payload).hexdigest() != digest_value(reference.get("sha256")):
            raise ContractError("telemetry catalog checksum mismatch", "corrupt_result")
        rows = value.get("artifacts")
        if (value.get("schema") != CATALOG_SCHEMA or not isinstance(rows, list)
                or not rows or not all(isinstance(row, dict) for row in rows)
                or value.get("entries") != expected_entries
                or reference.get("entries") != expected_entries):
            raise ContractError("telemetry catalog prefix is inconsistent", "corrupt_result")
        parsed = tuple(Artifact.parse(row) for row in rows)
        expected_entries -= len(parsed)
        if expected_entries < 0:
            raise ContractError("telemetry catalog exceeds watermark", "corrupt_result")
        chunks.append(parsed)
        reference = value.get("previous")
    if expected_entries != 0:
        raise ContractError("telemetry catalog is incomplete", "corrupt_result")
    artifacts = tuple(item for chunk in reversed(chunks) for item in chunk)
    if len({item.path for item in artifacts}) != count:
        raise ContractError("duplicate telemetry catalog path", "corrupt_result")
    return artifacts
