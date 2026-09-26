"""Small immutable presentation contracts. No numerical or HDF5 dependency."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from qcl_negf_contracts.artifacts import require_contract_set, validate_commit

SCHEMA = "qcl-negf.presentation-index.v3"


def accepted_result(record: dict[str, Any]) -> bool:
    """Project the native point contract; acceptance is never inferred from readiness.

    Native Julia result rows publish ``converged`` and ``quality``. The explicit
    acceptance flag belongs to artifact commits and is optional on those rows.
    An explicit rejection must nevertheless remain a rejection.
    """
    return (record.get("status") == "completed" and record.get("quality") == "strict"
            and record.get("converged") is True
            and record.get("scientific_accepted", True) is True)


def acceptance_kind(record: dict[str, Any]) -> str:
    """Analytical operator checks are distinct from stationary reference evidence."""
    return "operator" if "operator_checks_passed" in (record.get("observables") or {}) or "operator_checks" in (record.get("data") or {}) else "stationary"


def inside(root: Path, relative: str) -> Path:
    if root.is_symlink():
        raise ValueError("artifact root is a symbolic link")
    part = Path(relative)
    if part.is_absolute() or ".." in part.parts or not part.parts:
        raise ValueError("artifact path escaped result root")
    candidate = root
    for component in part.parts:
        candidate = candidate / component
        if candidate.is_symlink():
            raise ValueError("artifact path contains a symbolic link")
    return candidate


def read_json(path: Path, maximum: int | None = None) -> dict[str, Any]:
    if path.is_symlink():
        raise ValueError("published index cannot be a symbolic link")
    with path.open("rb") as stream:
        payload = stream.read() if maximum is None else stream.read(maximum + 1)
    if maximum is not None and len(payload) > maximum:
        raise ValueError("published index exceeds its byte budget")
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError("published index must be an object")
    return value


def read_contract_json(path: Path, schema: str, maximum: int | None = None) -> dict[str, Any]:
    value = read_json(path, maximum)
    require_contract_set(value)
    if value.get("schema") != schema:
        raise ValueError(f"unsupported result schema: requires {schema}")
    return value


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def committed_generation(artifact_root: Path) -> tuple[Path, dict[str, Any], str]:
    pointer = read_contract_json(inside(artifact_root, "current.json"), "qcl-negf.artifact-pointer.v2", 64_000)
    relative = pointer.get("commit_path")
    if not isinstance(relative, str):
        raise ValueError("artifact pointer requires explicit commit_path")
    path = inside(artifact_root, relative)
    commit = read_json(path)
    if commit.get("schema") != "qcl-negf.artifact-commit.v2":
        raise ValueError("unsupported artifact commit schema")
    validate_commit(commit)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = pointer.get("sha256")
    if not isinstance(expected, str) or len(expected) != 64 or digest != expected:
        raise ValueError("artifact commit hash differs from current pointer")
    return path.parent, commit, digest


def read_performance_summary(root: Path) -> dict[str, Any]:
    """Read registered immutable performance summaries, never enumerate a tree."""
    index_path = root / "performance" / "index.json"
    if not index_path.is_file():
        return {"available": False, "sessions": []}
    index = read_contract_json(index_path, "qcl-negf.performance-index.v2")
    sessions = []
    for session in index.get("sessions", []):
        pointer_path = inside(index_path.parent, session["pointer"])
        pointer = read_contract_json(pointer_path, "qcl-negf.artifact-pointer.v2", 64_000)
        commit_path = inside(pointer_path.parent, pointer["commit_path"])
        payload = commit_path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != pointer["sha256"]:
            raise ValueError("performance commit hash differs from pointer")
        commit = json.loads(payload)
        validate_commit(commit)
        summaries = [artifact for artifact in commit.get("artifacts", []) if artifact.get("role") == "performance.summary" and Path(artifact["path"]).name.startswith("summary")]
        for artifact in summaries:
            path = inside(commit_path.parent, artifact["path"])
            value = read_json(path)
            require_contract_set(value)
            if hashlib.sha256(path.read_bytes()).hexdigest() != artifact["sha256"]:
                raise ValueError("performance summary hash differs from commit")
            sessions.append({"id": session["id"], **{key:value.get(key) for key in ("rows", "max_observed_rss_bytes", "phase_seconds", "closed", "phase_sum_is_not_wall_time")}})
    return {"available": bool(sessions), "sessions": sessions, "warning": "Inclusive phase intervals overlap; their sum is not elapsed campaign time."}
