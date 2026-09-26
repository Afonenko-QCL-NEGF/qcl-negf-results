"""Durable immutable artifact publication shared by telemetry and result adapters."""
from __future__ import annotations
from ._atomic_io import atomic_write as atomic_write, fsync_directory as fsync_directory

import hashlib
import json
from pathlib import Path
from typing import Any

from qcl_negf_contracts.artifacts import CONTRACT_SET, POINTER_SCHEMA, relative_path
from qcl_negf_contracts.messages import ContractError, validate_json


def json_bytes(value: object) -> bytes:
    return json.dumps(validate_json(value), ensure_ascii=False, allow_nan=False,
                      sort_keys=True, separators=(",", ":")).encode("utf-8")


def decode_json(payload: bytes, source: str = "JSON") -> dict[str, Any]:
    try:
        def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, item in pairs:
                if key in result:
                    raise ValueError(f"duplicate JSON key {key}")
                result[key] = item
            return result
        value = validate_json(json.loads(payload, object_pairs_hook=unique_object))
    except (ValueError, UnicodeError) as error:
        raise ContractError(f"{source}: invalid complete UTF-8 JSON: {error}", "corrupt_result") from error
    if not isinstance(value, dict):
        raise ContractError(f"{source}: expected JSON object", "corrupt_result")
    return value


def read_json(path: Path, maximum: int | None = None) -> tuple[dict[str, Any], bytes]:
    # Open once: atomic rename by a producer cannot mix old length with new bytes.
    with path.open("rb") as stream:
        payload = stream.read() if maximum is None else stream.read(maximum + 1)
    if maximum is not None and len(payload) > maximum:
        raise ContractError(f"{path.name}: JSON exceeds its declared budget", "corrupt_result")
    return decode_json(payload, str(path)), payload






def safe_path(root: Path, relative: str) -> Path:
    relative_path(relative)
    root = root.resolve()
    candidate = root.joinpath(*relative.split("/"))
    current = root
    for part in relative.split("/"):
        current = current / part
        if current.is_symlink():
            raise ContractError("artifact path traverses a symbolic link", "corrupt_result")
    if not candidate.is_relative_to(root):
        raise ContractError("artifact path escapes its root", "corrupt_result")
    return candidate


def artifact_row(path: Path, *, relative: str, role: str, media_type: str,
                 profile: str = "science", schema: str | None = None) -> dict[str, Any]:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": relative, "role": role, "media_type": media_type,
            "profile": profile, "bytes": path.stat().st_size,
            "sha256": digest, "dependencies": [],
            **({"schema": schema} if schema is not None else {})}


def publish_pointer(path: Path, commit_path: str, commit_payload: bytes, generation: int) -> None:
    atomic_write(path, json_bytes({"schema": POINTER_SCHEMA, "contract_set": CONTRACT_SET,
        "commit_path": relative_path(commit_path), "generation": generation,
        "sha256": hashlib.sha256(commit_payload).hexdigest()}))
