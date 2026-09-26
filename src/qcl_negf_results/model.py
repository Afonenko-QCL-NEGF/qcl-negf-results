"""Resolved scientific input with a language-neutral, independently checked identity."""
from __future__ import annotations

import hashlib
import math
import struct
from typing import Any

from qcl_negf_contracts.artifacts import MODEL_SCHEMA, require_contract_set
from qcl_negf_contracts.messages import ContractError


def canonical_bytes(value: Any) -> bytes:
    """Exactly Julia's qcl-negf-canonical-bytes-v1, including binary64 bits."""
    if value is None:
        return b"n;"
    if isinstance(value, bool):
        return b"b1;" if value else b"b0;"
    if isinstance(value, (str, int, float)):
        if isinstance(value, str):
            tag, payload = b"s", value.encode("utf-8")
        elif isinstance(value, int):
            tag, payload = b"i", str(value).encode("ascii")
        else:
            if not math.isfinite(value):
                raise ValueError("nonfinite model identity input")
            tag, payload = b"f", struct.pack(">d", value).hex().encode("ascii")
        return tag + str(len(payload)).encode("ascii") + b":" + payload
    if isinstance(value, list):
        return b"a" + str(len(value)).encode() + b"[" + b"".join(map(canonical_bytes, value)) + b"]"
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return b"m" + str(len(value)).encode() + b"{" + b"".join(
            canonical_bytes(key) + canonical_bytes(value[key]) for key in sorted(value)) + b"}"
    raise ValueError("invalid portable model identity input")


def validate_model(value: dict[str, Any]) -> dict[str, Any]:
    require_contract_set(value)
    if value.get("schema") != MODEL_SCHEMA or value.get("hash_encoding") != "qcl-negf-canonical-bytes-v1":
        raise ContractError("unsupported resolved configuration envelope", "incompatible_contract")
    configuration = value.get("configuration")
    if not isinstance(configuration, dict):
        raise ContractError("resolved configuration is missing", "corrupt_result")
    if hashlib.sha256(canonical_bytes(configuration)).hexdigest() != value.get("configuration_hash"):
        raise ContractError("resolved configuration identity differs", "corrupt_result")
    return configuration
