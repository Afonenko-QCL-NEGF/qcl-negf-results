"""Startup credit is the existing validated locator, not a new admission service."""
import hashlib
import os

import pytest

from qcl_negf_results.capture import observe
from qcl_negf_results.commits import json_bytes


def setup(tmp_path):
    plan = json_bytes({"fingerprint": "a" * 64, "points": [{"id": "P", "execution_id": "E"}]})
    (tmp_path / "plan.json").write_bytes(plan)
    spec = {"schema": "qcl-negf.capture-spec.v1",
            "selection": {"calcjob_uuid": "job", "execution_id": "E", "attempt": 2, "plan_fingerprint": "a" * 64},
            "plan": {"path": "plan.json", "bytes": len(plan), "sha256": hashlib.sha256(plan).hexdigest()},
            "capture_token": "b" * 32, "allocation_id": "calcjob-process:job:2",
            "allocated_logical_cpus": None, "scratch_root": None,
            "credits": {"performance_bytes": 32 * 1024**2, "scalar_bytes": 64 * 1024**2,
                        "io_bytes": 128 * 1024**2, "reserve_bytes": 0}}
    path = tmp_path / "capture-spec.json"
    path.write_bytes(json_bytes(spec))
    return path, spec


def test_no_locator_before_terminal_means_zero_observer_output(tmp_path, monkeypatch):
    path, _ = setup(tmp_path)
    monkeypatch.setattr("qcl_negf_results.telemetry.TelemetryWriter", lambda *a, **k: pytest.fail("writer started without credit"))
    read_fd, write_fd = os.pipe2(os.O_NONBLOCK | os.O_CLOEXEC)
    os.write(write_fd, b"exit 0\n"); os.close(write_fd)
    assert observe(path, 42, 7, read_fd) == 2
    assert set(p.name for p in tmp_path.iterdir()) == {"plan.json", "capture-spec.json"}


def test_corrupt_startup_spec_has_no_writer_or_inherited_log_write(tmp_path, monkeypatch):
    path, _ = setup(tmp_path)
    path.write_bytes(b"{")
    class ForbiddenStderr:
        def write(self, *_): pytest.fail("pre-credit inherited log write")
    read_fd, write_fd = os.pipe2(os.O_NONBLOCK | os.O_CLOEXEC)
    try:
        with monkeypatch.context() as patch:
            patch.setattr("qcl_negf_results.capture.sys.stderr", ForbiddenStderr())
            assert observe(path, 42, 7, read_fd) == 2
    finally:
        os.close(write_fd)
    assert set(p.name for p in tmp_path.iterdir()) == {"plan.json", "capture-spec.json"}


@pytest.mark.parametrize("foreign", [False, True])
def test_delayed_locator_is_checked_before_first_writer_call(tmp_path, monkeypatch, foreign):
    path, spec = setup(tmp_path)
    read_fd, write_fd = os.pipe2(os.O_NONBLOCK | os.O_CLOEXEC)
    def publish(_):
        assert not (tmp_path / "performance").exists()
        locator = {"schema": "qcl-negf.capture-source.v1", "capture_token": "c" * 32 if foreign else spec["capture_token"],
                   "execution_id": "E", "plan_fingerprint": "a" * 64, "mode": "direct",
                   "actual_root": str(tmp_path / "result"), "epoch": 0}
        (tmp_path / "capture-source.json").write_bytes(json_bytes(locator))
    calls = []
    def writer(*a, **k):
        calls.append(True)
        raise ValueError("fixture stop after permitted writer entry")
    monkeypatch.setattr("qcl_negf_results.capture.time.sleep", publish)
    monkeypatch.setattr("qcl_negf_results.telemetry.TelemetryWriter", writer)
    try:
        assert observe(path, 42, 7, read_fd) == 2
    finally:
        os.close(write_fd)
    assert calls == ([] if foreign else [True])
