"""No-solve wrapper/observer controls, executed only by the separate finite gate."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from qcl_negf_results.capture import journal_batch, proc_sample, supervise


def test_proc_root_process_counter_identity_and_missing_are_not_zero(tmp_path):
    root = tmp_path / "proc"
    point = root / "42"
    point.mkdir(parents=True)
    fields = ["S"] + ["0"] * 49
    fields[11], fields[12], fields[19] = "100", "200", "500"
    (point / "stat").write_text("42 (name with ) spaces) " + " ".join(fields))
    (point / "status").write_text("VmRSS:\t12 kB\nVmHWM:\t18 kB\n")
    row = proc_sample(42, 500, ticks=100, proc_root=root, now_ns=10, wall_seconds=2)
    assert row["cpu_time_seconds"] == 3.0
    assert row["rss_bytes"] == 12288 and row["peak_rss_bytes"] == 18432
    assert row["cgroup_scope"] == "root_julia_process_including_threads"
    foreign = proc_sample(42, 501, ticks=100, proc_root=root, now_ns=11, wall_seconds=3)
    assert foreign["cpu_time_seconds"] is None and foreign["missing_reason"]
    (point / "stat").unlink()
    gone = proc_sample(42, 500, ticks=100, proc_root=root, now_ns=12, wall_seconds=4)
    assert gone["cpu_time_seconds"] is None and gone["rss_bytes"] is None


@pytest.mark.parametrize("mode, expected", [("normal", 7), ("term", 23)])
def test_supervisor_ready_child_exit_and_term_reap_stalled_observer(tmp_path, mode, expected):
    helper = Path(__file__).parent / "helpers/capture_lifecycle_probe.py"
    child = subprocess.Popen(
        [sys.executable, str(helper), str(tmp_path), mode],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        deadline = time.monotonic() + 3
        while not all((tmp_path / name).exists() for name in ("solver-ready", "observer-ready")):
            assert time.monotonic() < deadline, "stand-ins failed to become ready"
            assert child.poll() is None, "helper exited before readiness"
            time.sleep(.01)
        if mode == "term":
            child.send_signal(signal.SIGTERM)
        output, errors = child.communicate(timeout=5)
        assert child.returncode == 0, errors
        row = json.loads(output)
        assert row["code"] == expected and row["observer_reaped"] is True
        assert row["elapsed"] < 5
    finally:
        if child.poll() is None:
            for name in ("solver-ready", "observer-ready"):
                marker = tmp_path / name
                if marker.exists():
                    try:
                        os.killpg(int(marker.read_text()), signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            try:
                child.communicate(timeout=2)
            except subprocess.TimeoutExpired:
                child.kill()
                child.communicate(timeout=1)


def test_supervisor_never_relaunches_after_observer_spawn_error(tmp_path, monkeypatch):
    marker = tmp_path / "once"
    code = 'from pathlib import Path; p=Path(__import__("sys").argv[1]); p.write_text("once")'

    def fail(*args):
        raise OSError("observer unavailable")

    class ForbiddenStderr:
        def write(self, *args):
            raise AssertionError("supervisor must not write an inherited log")

    with monkeypatch.context() as patch:
        patch.setattr("qcl_negf_results.capture.sys.stderr", ForbiddenStderr())
        code = supervise(
            [sys.executable, "-c", code, str(marker)], observer_factory=fail,
            drain_seconds=0.1,
        )
    assert code == 0
    assert marker.read_text() == "once"


def test_observer_prefix_and_repeated_metadata_refuse_before_read(tmp_path, monkeypatch):
    from contextlib import contextmanager
    import qcl_negf_results._capture_io as io
    import qcl_negf_results.capture as capture

    reads = []
    original = io.regular

    class Spy:
        def __init__(self, stream):
            self.stream = stream
        def fileno(self):
            return self.stream.fileno()
        def read(self, count):
            reads.append(count)
            return self.stream.read(count)

    @contextmanager
    def tracked(*args, **kwargs):
        with original(*args, **kwargs) as stream:
            yield Spy(stream)

    monkeypatch.setattr(io, "regular", tracked)
    monkeypatch.setattr(capture, "regular", tracked)
    journal = tmp_path / "journal"
    journal.write_bytes(b"x" * 1024 + b"\n")
    budget = io.ObserverIO(100)
    with pytest.raises(ValueError):
        journal_batch(journal, {"offset": 1024}, source_key="p", budget=budget)
    assert reads == [] and budget.read_bytes == 0
    metadata = tmp_path / "metadata.json"
    metadata.write_bytes(b'{"a":1}')
    budget = io.ObserverIO(7)
    assert io.small_json(metadata, 100, budget=budget)[0] == {"a": 1}
    first = list(reads)
    with pytest.raises(ValueError):
        io.small_json(metadata, 100, budget=budget)
    assert reads == first and budget.read_bytes == 7


def test_writer_startup_metadata_and_binding_share_observer_credit(tmp_path):
    from qcl_negf_results._capture_io import ObserverIO
    from qcl_negf_results.capture import make_binding
    from qcl_negf_results.telemetry import TelemetryWriter

    selection = {"calcjob_uuid": "job", "execution_id": "ex", "attempt": 1,
                 "plan_fingerprint": "a" * 64}
    budget = ObserverIO(0)
    with pytest.raises(ValueError):
        TelemetryWriter(tmp_path / "denied", selection, {}, metadata_budget=budget)
    assert not list((tmp_path / "denied").rglob("*.json"))
    budget = ObserverIO(64 * 1024)
    writer = TelemetryWriter(tmp_path / "performance", selection, {}, metadata_budget=budget)
    closed = writer.close()
    assert budget.read_bytes > 0 and budget.write_bytes > 0
    before = budget.used
    budget.maximum = before
    with pytest.raises(ValueError):
        make_binding(tmp_path, selection=selection, result_path="result.json",
                     point_pins=[], closed_path=closed, capture_status={}, budget=budget)
    assert budget.used == before and not (tmp_path / "performance-binding.json").exists()


def test_complete_native_lines_are_atomic_cursor_batches(tmp_path):
    source = tmp_path / "phases.jsonl"
    row = {
        "schema": "qcl-negf.producer-phase.v2",
        "contract_set": "qcl-negf.results.v1",
        "source_session_id": "native",
        "source_sequence": 1,
        "clock_domain_id": "native-clock",
        "phase": "basis",
        "event_kind": "end",
        "duration_seconds": 2.0,
        "monotonic_ns": 20,
        "begin_monotonic_ns": 10,
        "end_monotonic_ns": 20,
    }
    raw = json.dumps(row).encode() + b"\n"
    source.write_bytes(raw + b'{"partial":')
    rows, cursor, consumed = journal_batch(source, {}, source_key="p/attempt-2")
    assert len(rows) == 1 and rows[0]["measurement_kind"] == "phase_boundary"
    assert rows[0]["source_sequence"] == 1
    assert cursor["offset"] == len(raw) and cursor["complete_at_observation"] is False
    assert consumed == len(raw) + len(b'{"partial":')
    rows2, cursor2, _ = journal_batch(source, cursor, source_key="p/attempt-2")
    assert rows2 == [] and cursor2["offset"] == len(raw)
    source.write_bytes(raw.replace(b'"native"', b'"foreign"') + b"\n")
    with pytest.raises(ValueError):
        journal_batch(source, cursor, source_key="p/attempt-2")


def test_actual_scratch_closed_publication_then_direct_route_no_whole_copy(tmp_path):
    from test_live_views import fixture

    from qcl_negf_results.capture import ScalarPublisher
    from qcl_negf_results._capture_io import ObserverIO

    job = tmp_path / "job"
    job.mkdir()
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    point_root = scratch / "p"
    point_root.mkdir()
    index = fixture(point_root / "history")
    (point_root / "history" / ".open.h5").write_bytes(b"not closed")
    selection = {
        "calcjob_uuid": "job",
        "execution_id": "ex",
        "attempt": 10,
        "plan_fingerprint": "a" * 64,
    }
    budget = ObserverIO(128 * 1024 * 1024)
    publisher = ScalarPublisher(
        job, selection, scalar_credit=64 * 1024 * 1024, reserve_bytes=0, budget=budget
    )
    point = {"point_id": "p0", "source_attempt": 10, "artifact_root": "p"}
    value = publisher.publish(scratch, point, mode="scratch", terminal=True)
    mirror = job / value["index_path"]
    assert mirror.read_bytes() == index.read_bytes()
    assert not (mirror.parent / ".open.h5").exists()
    native = json.loads(index.read_bytes())
    for row in native["segments"]:
        assert (mirror.parent / row["path"]).read_bytes() == (
            index.parent / row["path"]
        ).read_bytes()
    # Generation/index visibility follows every durable leaf, descriptor last.
    assert (
        json.loads((job / "live/current.json").read_bytes())["index_sha256"]
        == value["index_sha256"]
    )
    assert budget.write_bytes == (
        sum(row["bytes"] for row in native["segments"]) + len(index.read_bytes())
        + len((job / "live/current.json").read_bytes())
    )
    prior = index.read_bytes()
    denied = ScalarPublisher(job, selection, scalar_credit=0, reserve_bytes=0)
    with pytest.raises(ValueError):
        denied.publish(scratch, point, mode="scratch", terminal=True)
    assert denied.disabled and index.read_bytes() == prior
    result = job / "result"
    result.mkdir()
    direct_point = result / "p"
    direct_point.mkdir()
    fixture(direct_point / "history")
    direct = ScalarPublisher(job, selection, scalar_credit=0, reserve_bytes=0)
    shown = direct.publish(result, point, mode="direct", terminal=True)
    assert shown["index_path"] == "result/p/history/index.json" and direct.bytes == 0


def test_real_observer_direct_close_sidecar_and_scientific_sentinel(
    tmp_path, monkeypatch
):
    import os

    from test_live_views import fixture

    from qcl_negf_results.capture import observe
    from qcl_negf_results.commits import json_bytes

    job = tmp_path / "job"
    job.mkdir()
    result = job / "result"
    result.mkdir()
    point = result / "p"
    point.mkdir()
    index = fixture(point / "history")
    commit = point / "history/part-1/commit.json"
    sentinel = commit.read_bytes()
    plan = {"fingerprint": "a" * 64, "points": [{"id": "p0", "execution_id": "ex"}]}
    raw = json_bytes(plan)
    (job / "plan.json").write_bytes(raw)
    import hashlib

    selection = {
        "calcjob_uuid": "job",
        "execution_id": "ex",
        "attempt": 10,
        "plan_fingerprint": "a" * 64,
    }
    spec = {
        "schema": "qcl-negf.capture-spec.v1",
        "selection": selection,
        "plan": {
            "path": "plan.json",
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        },
        "capture_token": "c" * 32,
        "allocation_id": "allocation",
        "allocated_logical_cpus": 2,
        "scratch_root": None,
        "credits": {
            "performance_bytes": 32 * 1024 * 1024,
            "scalar_bytes": 0,
            "io_bytes": 32 * 1024 * 1024,
            "reserve_bytes": 0,
        },
    }
    (job / "capture-spec.json").write_bytes(json_bytes(spec))
    locator = {
        "schema": "qcl-negf.capture-source.v1",
        "capture_token": "c" * 32,
        "execution_id": "ex",
        "plan_fingerprint": "a" * 64,
        "mode": "direct",
        "actual_root": str(result),
        "epoch": 0,
    }
    (job / "capture-source.json").write_bytes(json_bytes(locator))
    series = {
        "schema": "qcl-negf-series-result-v3",
        "plan_fingerprint": "a" * 64,
        "points": [
            {
                "id": "p0",
                "execution_id": "ex",
                "attempt": 10,
                "data": {
                    "result_commit": "p/history/part-1/commit.json",
                    "artifact_root": "p",
                },
            }
        ],
    }
    (result / "series_result.json").write_bytes(json_bytes(series))
    progress = point / "progress"
    progress.mkdir()
    phase = {
        "schema": "qcl-negf.producer-phase.v2",
        "contract_set": "qcl-negf.results.v1",
        "source_session_id": "native",
        "source_sequence": 1,
        "clock_domain_id": "clock",
        "phase": "basis",
        "event_kind": "end",
        "duration_seconds": 2.0,
        "monotonic_ns": 2_000_000_010,
        "begin_monotonic_ns": 10,
        "end_monotonic_ns": 2_000_000_010,
    }
    (progress / "phases.jsonl").write_bytes(json_bytes(phase) + b"\n")
    read_fd, write_fd = os.pipe2(os.O_NONBLOCK | os.O_CLOEXEC)
    os.write(write_fd, b"exit 0\n")
    os.close(write_fd)
    monkeypatch.setattr(
        "qcl_negf_results.capture.proc_sample",
        lambda *a, **k: {
            "wall_seconds": 2.0,
            "monotonic_ns": 2_000_000_000,
            "cpu_time_seconds": None,
            "missing_reason": "root_process_missing_or_identity_changed",
        },
    )
    assert observe(job / "capture-spec.json", 42, 7, read_fd) == 0
    assert commit.read_bytes() == sentinel
    descriptor = json.loads((job / "live/current.json").read_bytes())
    assert descriptor["index_path"] == "result/p/history/index.json"
    assert descriptor["index_sha256"] == hashlib.sha256(index.read_bytes()).hexdigest()
    sidecar = json.loads((job / "performance-binding.json").read_bytes())
    native = json.loads(sidecar["commit"]["raw_utf8"])
    assert native["closed"] is True
    assert native["source_cursors"]["p0/attempt-10"]["source_sequence"] == 1
    assert sidecar["point_pins"][0]["source_attempt"] == 10
    assert sidecar["capture_status"]["complete"] is True


def test_root_switch_same_native_prefix_does_not_replay_rows(tmp_path):
    first = tmp_path / "scratch.jsonl"
    second = tmp_path / "staged.jsonl"
    row = {
        "schema": "qcl-negf.producer-phase.v2",
        "contract_set": "qcl-negf.results.v1",
        "source_session_id": "native",
        "source_sequence": 1,
        "clock_domain_id": "clock",
        "phase": "basis",
        "event_kind": "end",
        "duration_seconds": 1.0,
    }
    raw = json.dumps(row).encode() + b"\n"
    first.write_bytes(raw)
    second.write_bytes(raw)
    rows, cursor, _ = journal_batch(first, {}, source_key="p/attempt-2")
    replay, advanced, _ = journal_batch(second, cursor, source_key="p/attempt-2")
    assert len(rows) == 1 and replay == [] and advanced["offset"] == len(raw)
    row["source_sequence"] = 2
    second.write_bytes(raw + json.dumps(row).encode() + b"\n")
    newer, cutoff, _ = journal_batch(second, advanced, source_key="p/attempt-2")
    assert [item["source_sequence"] for item in newer] == [2]
    assert cutoff["source_sequence"] == 2
