"""One solver reaper and one disposable I/O observer; no solver retry."""

from __future__ import annotations

import hashlib
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from ._atomic_io import atomic_write, fsync_directory
from ._capture_io import (
    beneath,
    copy_closed,
    digest,
    fail,
    integer,
    ObserverIO,
    regular,
    small_json,
)
from .commits import decode_json, json_bytes

MIB = 1024 * 1024


def _start_ticks(pid):
    with open(f"/proc/{pid}/stat", encoding="ascii") as stream:
        return int(stream.read(16384).rpartition(") ")[2].split()[19])


def proc_sample(
    pid,
    start_ticks,
    *,
    ticks=None,
    proc_root=Path("/proc"),
    now_ns=None,
    wall_seconds=0.0,
    budget=None,
):
    row = {
        "pid": pid,
        "process_start_ticks": start_ticks,
        "monotonic_ns": time.monotonic_ns() if now_ns is None else now_ns,
        "wall_seconds": wall_seconds,
        "cgroup_scope": "root_julia_process_including_threads",
        "cpu_time_seconds": None,
        "rss_bytes": None,
        "peak_rss_bytes": None,
    }
    try:
        if budget is not None:
            budget.require(16384)
        with (Path(proc_root) / str(pid) / "stat").open("rb") as stream:
            payload = stream.read(16384)
        if budget is not None:
            budget.account(len(payload))
        raw = payload.decode("utf-8", errors="replace")
        fields = raw.rpartition(") ")[2].split()
        if int(fields[19]) != start_ticks:
            raise ValueError("PID identity changed")
        clock = os.sysconf("SC_CLK_TCK") if ticks is None else ticks
        row["cpu_time_seconds"] = (int(fields[11]) + int(fields[12])) / clock
    except OSError, ValueError, IndexError:
        row["missing_reason"] = "root_process_missing_or_identity_changed"
        return row
    try:
        if budget is not None:
            budget.require(65536)
        with (Path(proc_root) / str(pid) / "status").open("rb") as stream:
            payload = stream.read(65536)
            if budget is not None:
                budget.account(len(payload))
            for line in payload.decode("ascii", errors="replace").splitlines():
                key, _, value = line.partition(":")
                if key in ("VmRSS", "VmHWM"):
                    number, unit = value.split()
                    if unit != "kB" or not integer(int(number)):
                        raise ValueError("unsupported RSS unit")
                    row["rss_bytes" if key == "VmRSS" else "peak_rss_bytes"] = (
                        int(number) * 1024
                    )
        if row["rss_bytes"] is None:
            row["capability_reasons"] = "root_rss_unavailable"
    except OSError, ValueError:
        row["capability_reasons"] = "root_rss_unavailable"
    return row


def _kill_group(child, signum):
    if child is not None:
        try:
            os.killpg(child.pid, signum)
        except ProcessLookupError:
            pass


def supervise(solver_argv, *, observer_factory=None, drain_seconds=5.0):
    """No filesystem I/O after spawn except local proc. D-state observer is not awaited."""
    if not 0 <= drain_seconds <= 5:
        raise ValueError("observer drain must be in 0..5 seconds")
    child = subprocess.Popen(solver_argv, start_new_session=True)
    observer = None
    read_fd = write_fd = None
    termination = None
    previous = {}

    def forward(signum, frame):
        nonlocal termination
        if termination is None:
            termination = time.monotonic()
        _kill_group(child, signum)

    try:
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous[signum] = signal.signal(signum, forward)
        try:
            start = _start_ticks(child.pid)
            if observer_factory is not None:
                read_fd, write_fd = os.pipe2(os.O_NONBLOCK | os.O_CLOEXEC)
                argv = observer_factory(child.pid, start, read_fd)
                observer = subprocess.Popen(
                    argv, pass_fds=(read_fd,), start_new_session=True
                )
        except OSError, ValueError, IndexError:
            # Julia was already spawned: observational failure can never retry it.
            # Inherited stderr can itself be an uninterruptible NFS write.
            pass
        finally:
            if read_fd is not None:
                os.close(read_fd)
                read_fd = None
        while child.poll() is None:
            if termination is not None and time.monotonic() - termination >= 5:
                _kill_group(child, signal.SIGKILL)
            time.sleep(0.025)
        code = child.wait()
        if write_fd is not None:
            try:
                os.write(write_fd, f"exit {code}\n".encode())
            except BlockingIOError, BrokenPipeError:
                pass
        deadline = time.monotonic() + drain_seconds
        while (
            observer is not None
            and observer.poll() is None
            and time.monotonic() < deadline
        ):
            time.sleep(0.025)
        if observer is not None and observer.poll() is None:
            _kill_group(observer, signal.SIGKILL)
            # Ordinary killed observer gets a short nonblocking reap opportunity;
            # uninterruptible remote I/O cannot hold the solver supervisor forever.
            deadline = time.monotonic() + 0.1
            while observer.poll() is None and time.monotonic() < deadline:
                time.sleep(0.005)
            # No log write/ack: an unreaped observer remains scheduler authority.
        return code if code >= 0 else 128 - code
    finally:
        if write_fd is not None:
            os.close(write_fd)
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def journal_batch(path, cursor, *, source_key, maximum_bytes=65536, maximum_rows=128, budget=None):
    """Validate complete native lines and replay prefix on every source reopen."""
    from qcl_negf_contracts.telemetry import SPAN_FIELDS

    allowed = {key for key, _ in SPAN_FIELDS}
    offset = cursor.get("offset", 0)
    if (
        not integer(offset, 0, 64 * MIB)
        or not 1 <= maximum_bytes <= 65536
        or not 1 <= maximum_rows <= 128
    ):
        raise ValueError("journal cursor/batch exceeds bound")
    with regular(path) as stream:
        size = os.fstat(stream.fileno()).st_size
        if size < offset:
            raise ValueError("native journal truncated")
        suffix = min(maximum_bytes + 1, size - offset)
        if budget is not None:
            budget.require(offset + suffix)
        prefix = hashlib.sha256()
        consumed = 0
        remaining = offset
        while remaining:
            block = stream.read(min(MIB, remaining))
            if budget is not None:
                budget.account(len(block))
            consumed += len(block)
            if not block:
                raise ValueError("native journal prefix truncated")
            prefix.update(block)
            remaining -= len(block)
        if offset and prefix.hexdigest() != cursor.get("prefix_sha256"):
            raise ValueError("native journal prefix changed across source epochs")
        raw = stream.read(suffix)
        if budget is not None:
            budget.account(len(raw))
        consumed += len(raw)
        rows = []
        used = 0
        session = cursor.get("source_session_id")
        sequence = cursor.get("source_sequence", 0)
        clock = cursor.get("clock_domain_id")
        for line in raw.splitlines(keepends=True):
            if (
                not line.endswith(b"\n")
                or used + len(line) > maximum_bytes
                or len(rows) >= maximum_rows
            ):
                break
            if len(line) > 65536:
                raise ValueError("native journal line exceeds cap")
            value = decode_json(line, "native phase line")
            if (
                not isinstance(value, dict)
                or value.get("schema") != "qcl-negf.producer-phase.v2"
                or value.get("contract_set") != "qcl-negf.results.v1"
                or not isinstance(value.get("source_session_id"), str)
                or not value["source_session_id"]
                or not isinstance(value.get("clock_domain_id"), str)
                or not integer(value.get("source_sequence"), 1)
                or value["source_sequence"] != sequence + 1
                or (session is not None and value["source_session_id"] != session)
                or (clock is not None and value["clock_domain_id"] != clock)
            ):
                raise ValueError("native phase session/sequence/clock differs")
            session = value["source_session_id"]
            sequence = value["source_sequence"]
            clock = value["clock_domain_id"]
            row = {k: v for k, v in value.items() if k in allowed}
            row.update(
                measurement_kind="phase_boundary",
                timestamp_kind="measurement",
                source_path=source_key,
                source_offset=offset + used,
            )
            rows.append(row)
            prefix.update(line)
            used += len(line)
        if not rows and len(raw) > maximum_bytes and b"\n" not in raw[:maximum_bytes]:
            raise ValueError("native journal line exceeds cap")
        return (
            rows,
            {
                "offset": offset + used,
                "prefix_sha256": prefix.hexdigest(),
                "source_session_id": session,
                "source_sequence": sequence,
                "clock_domain_id": clock,
                "complete_at_observation": len(raw) == used,
            },
            consumed,
        )


def make_binding(
    job_root, *, selection, result_path, point_pins, closed_path, capture_status, budget=None
):
    """Only returned close path is authoritative. Caller supplies independently proven point pins."""
    from .performance_views import validate_performance_binding

    job_root = Path(job_root)
    closed_path = Path(closed_path)
    commit, raw = small_json(closed_path, MIB, budget=budget)
    owners = [
        r for r in commit.get("artifacts", []) if r.get("role") == "performance.summary"
    ]
    if len(owners) != 1:
        fail("corrupt_result", "unique closed summary required")
    owner = owners[0]
    summary_path = beneath(closed_path.parent, owner["path"])
    _, summary_raw = small_json(summary_path, MIB - len(raw), budget=budget)
    _, result_raw = small_json(beneath(job_root, result_path), MIB, budget=budget)

    def retained(path, payload):
        return {
            "native_path": str(path.relative_to(job_root)),
            "bytes": len(payload),
            "sha256": digest(payload),
            "raw_utf8": payload.decode("utf-8"),
        }

    value = {
        "schema": "qcl-negf.performance-binding.v1",
        "scope": "execution_attempt",
        "selection": selection,
        "runner_result": {
            "path": result_path,
            "bytes": len(result_raw),
            "sha256": digest(result_raw),
        },
        "point_pins": point_pins,
        "commit": retained(closed_path, raw),
        "summary": retained(summary_path, summary_raw),
        "capture_status": capture_status,
    }
    validate_performance_binding(
        value, expected_selection=selection, expected_point_pins=point_pins
    )
    payload = json_bytes(value)
    if len(payload) > MIB:
        fail("budget_exceeded", "retained performance sidecar exceeds cap")
    if budget is not None:
        budget.write(len(payload))
    atomic_write(job_root / "performance-binding.json", payload)
    return value


class ScalarPublisher:
    """Actual scratch route only. Direct route points at the existing closed index."""

    def __init__(self, job_root, selection, *, scalar_credit, reserve_bytes, budget=None):
        if not integer(scalar_credit, 0, 64 * MIB) or not integer(reserve_bytes):
            raise ValueError("invalid pre-admitted scalar partition")
        self.job_root = Path(job_root)
        self.selection = selection
        self.maximum = scalar_credit
        self.reserve = reserve_bytes
        self.copied = {}
        self.bytes = 0
        self.disabled = False
        self.metadata = {}
        self.prefixes = {}
        self.io_bytes = 0
        self.last = None
        self.last_publish = -float("inf")
        self.budget = budget

    def publish(
        self, actual_root, point, *, mode, terminal=False, io_remaining=128 * MIB
    ):
        if self.disabled or (
            not terminal and time.monotonic() - self.last_publish < 30
        ):
            return self.last
        budget = self.budget if self.budget is not None else ObserverIO(io_remaining)
        initial_io = budget.used
        try:
            if mode not in ("direct", "scratch", "staged"):
                raise ValueError("unknown actual source route")
            expected = {
                "point_id": point["point_id"],
                "execution_id": self.selection["execution_id"],
                "attempt": point["source_attempt"],
                "plan_fingerprint": self.selection["plan_fingerprint"],
            }
            history = beneath(actual_root, point["artifact_root"]) / "history"
            from .live_views import LiveHistoryLimits, _index

            owner = {
                **self.selection,
                "point_id": point["point_id"],
                "source_attempt": point["source_attempt"],
            }
            if not (history / "index.json").exists():
                return self.last
            value, raw = _index(
                history / "index.json", owner, LiveHistoryLimits(), budget
            )
            target = (
                self.job_root
                / "live"
                / self.selection["execution_id"]
                / point["point_id"]
                / f"attempt-{point['source_attempt']}"
                / "history"
            )
            metadata = {**self.metadata, str(target): len(raw)}
            if sum(metadata.values()) > MIB:
                raise ValueError("whole-job live metadata cap exhausted")
            previous = self.prefixes.get(str(target))
            if previous is not None and (
                value["generation"] < previous["generation"]
                or value["segments"][: len(previous["segments"])]
                != previous["segments"]
            ):
                raise ValueError("closed prefix changed across source root epochs")
            new = []
            if mode == "scratch":
                new = [
                    r
                    for r in value["segments"]
                    if (str(target / r["path"]), r["sha256"]) not in self.copied
                ]
            budget.require((len(raw) if mode == "scratch" else 0) + 16384 + 2 * sum(r["bytes"] for r in new))
            if mode == "scratch":
                if (
                    len(self.copied) + len(new) > 4096
                    or self.bytes + sum(r["bytes"] for r in new) > self.maximum
                ):
                    raise ValueError("admitted scalar partition exhausted")
                free = os.statvfs(self.job_root)
                if (
                    free.f_bavail * free.f_frsize
                    < self.reserve + sum(r["bytes"] for r in new) + MIB
                ):
                    raise ValueError("protected output reserve unavailable")
                target.mkdir(parents=True, exist_ok=True)
                for row in new:
                    destination = beneath(target, row["path"])
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    temporary = destination.with_name(".copy-" + destination.name)
                    budget.require(2 * row["bytes"])
                    budget.write(row["bytes"])
                    copy_closed(
                        beneath(history, row["path"]),
                        temporary,
                        size=row["bytes"],
                        sha256=row["sha256"],
                        budget=budget,
                    )
                    os.link(temporary, destination)
                    temporary.unlink()
                    fsync_directory(destination.parent)
                    self.copied[(str(destination), row["sha256"])] = True
                    self.bytes += row["bytes"]
                budget.write(len(raw))
                atomic_write(target / "index.json", raw)
                published = target / "index.json"
            else:
                published = history / "index.json"
            descriptor = {
                "schema": "qcl-negf.live-prefix.v1",
                "identity": expected,
                "access_mode": "selected_closed_copy",
                "index_path": str(published.relative_to(self.job_root)),
                "index_sha256": digest(raw),
                "generation": value["generation"],
                "scba_rows": value["scba_rows"],
                "outer_rows": value["outer_rows"],
                "updated_unix": value["updated_unix"],
                "observed_unix": time.time(),
                "route": mode,
                "live_publication_incomplete": False,
            }
            payload = json_bytes(descriptor)
            budget.write(len(payload))
            atomic_write(self.job_root / "live" / "current.json", payload)
            self.last = descriptor
            self.last_publish = time.monotonic()
            self.metadata = metadata
            self.prefixes[str(target)] = {
                "generation": value["generation"],
                "segments": value["segments"],
            }
            return descriptor
        except OSError, ValueError:
            self.disabled = True
            if self.last is not None:
                self.last = {**self.last, "live_publication_incomplete": True}
                payload = json_bytes(self.last)
                budget.write(len(payload))
                atomic_write(
                    self.job_root / "live" / "current.json", payload
                )
            raise
        finally:
            self.io_bytes += budget.used - initial_io


def _spec(path, *, budget=None):
    """Private operational port, written by AiiDA before launch, never HTTP input."""
    budget = budget if budget is not None else ObserverIO(16384)
    value, _ = small_json(path, 16384, budget=budget)
    keys = {
        "schema",
        "selection",
        "plan",
        "capture_token",
        "allocation_id",
        "allocated_logical_cpus",
        "scratch_root",
        "credits",
    }
    from .performance_views import _selection

    if (
        not isinstance(value, dict)
        or set(value) != keys
        or value["schema"] != "qcl-negf.capture-spec.v1"
    ):
        raise ValueError("unsupported capture spec")
    # A valid grant prepays the one-time bounded spec read before revealing itself.
    credits = value["credits"]
    if (not isinstance(credits, dict) or set(credits) != {"performance_bytes", "scalar_bytes", "io_bytes", "reserve_bytes"}
        or not integer(credits["performance_bytes"], 8 * MIB, 32 * MIB)
        or not integer(credits["scalar_bytes"], 0, 64 * MIB)
        or not integer(credits["io_bytes"], 16384) or not integer(credits["reserve_bytes"])):
        raise ValueError("existing admitted observer partitions/startup credit required")
    budget.maximum = credits["io_bytes"]
    _selection(value["selection"])
    for identifier in (value["selection"]["execution_id"],):
        if "/" in identifier or "\\" in identifier or identifier in (".", ".."):
            raise ValueError("execution identity is not one path component")
    token = value["capture_token"]
    if (
        not isinstance(token, str)
        or len(token) != 32
        or any(c not in "0123456789abcdef" for c in token)
    ):
        raise ValueError("invalid launch token")
    width = value["allocated_logical_cpus"]
    if width is not None and not integer(width, 1, 65536):
        raise ValueError("actual allocation width invalid")
    if not isinstance(value["allocation_id"], str) or not value["allocation_id"]:
        raise ValueError("actual allocation identity required")
    plan = value["plan"]
    if (
        not isinstance(plan, dict)
        or set(plan) != {"path", "bytes", "sha256"}
        or not integer(plan["bytes"], 1, MIB)
    ):
        raise ValueError("bounded frozen plan pin required")
    root = Path(path).absolute().parent
    parsed, raw = small_json(beneath(root, plan["path"]), MIB, budget=budget)
    if len(raw) != plan["bytes"] or digest(raw) != plan["sha256"]:
        raise ValueError("frozen plan bytes differ")
    rows = parsed.get("points")
    if (
        parsed.get("fingerprint") != value["selection"]["plan_fingerprint"]
        or not isinstance(rows, list)
        or len(rows) > 256
        or any(
            not isinstance(r, dict) or not isinstance(r.get("id"), str) for r in rows
        )
    ):
        raise ValueError("frozen point inventory unavailable")
    known = {
        r["id"]
        for r in rows
        if r.get("execution_id") == value["selection"]["execution_id"]
    }
    if (
        not known
        or any(not p or "/" in p or "\\" in p or p in (".", "..") for p in known)
        or len(known)
        != sum(
            r.get("execution_id") == value["selection"]["execution_id"] for r in rows
        )
    ):
        raise ValueError("frozen execution points invalid")
    scratch = value["scratch_root"]
    if scratch is not None and (
        not isinstance(scratch, str)
        or not Path(scratch).is_absolute()
        or os.path.abspath(scratch) != scratch
    ):
        raise ValueError("trusted scratch parent invalid")
    return root, value, known


def _locator(root, spec, *, budget=None):
    value, _ = small_json(root / "capture-source.json", 16384, budget=budget)
    if (
        not isinstance(value, dict)
        or set(value)
        != {
            "schema",
            "capture_token",
            "execution_id",
            "plan_fingerprint",
            "mode",
            "actual_root",
            "epoch",
        }
        or value["schema"] != "qcl-negf.capture-source.v1"
        or value["capture_token"] != spec["capture_token"]
        or value["execution_id"] != spec["selection"]["execution_id"]
        or value["plan_fingerprint"] != spec["selection"]["plan_fingerprint"]
        or not integer(value["epoch"])
        or value["mode"] not in ("direct", "scratch", "staged")
        or not isinstance(value["actual_root"], str)
        or not Path(value["actual_root"]).is_absolute()
    ):
        raise ValueError("actual-root locator differs from launch")
    actual = Path(value["actual_root"])
    if str(actual) != os.path.abspath(actual):
        raise ValueError("actual source root is not canonical")
    if value["mode"] in ("direct", "staged"):
        if actual != root / "result":
            raise ValueError("direct/staged locator outside selected job result")
    elif (
        spec["scratch_root"] is None
        or not actual.is_relative_to(Path(spec["scratch_root"]))
        or actual == Path(spec["scratch_root"])
    ):
        raise ValueError("scratch locator outside trusted worker parent")
    return value, actual


def _current(actual, selection, known, *, budget=None):
    value, _ = small_json(actual / "current_point.json", 16384, budget=budget)
    if (
        value.get("schema") != "qcl-negf-current-point-v1"
        or value.get("execution_id") != selection["execution_id"]
        or value.get("point_id") not in known
        or not integer(value.get("attempt"), 1)
    ):
        raise ValueError("current point differs from frozen execution")
    beneath(actual, value.get("artifact_root"))
    return {
        "point_id": value["point_id"],
        "source_attempt": value["attempt"],
        "artifact_root": value["artifact_root"],
    }


def _final_points(actual, selection, known, *, budget=None):
    value, raw = small_json(actual / "series_result.json", MIB, budget=budget)
    if (
        value.get("schema") != "qcl-negf-series-result-v3"
        or value.get("plan_fingerprint") != selection["plan_fingerprint"]
    ):
        raise ValueError("final result differs from frozen execution")
    rows = value.get("points")
    if not isinstance(rows, list) or len(rows) > 256:
        raise ValueError("bounded final point inventory required")
    points = []
    pins = []
    seen = set()
    metadata = len(raw)
    for row in rows:
        if not isinstance(row, dict):
            fail("corrupt_result", "invalid final point row")
        if row.get("execution_id") != selection["execution_id"]:
            continue
        point = row.get("id")
        data = row.get("data") or {}
        if point not in known or point in seen or not isinstance(data, dict):
            raise ValueError("unknown/duplicate final point")
        seen.add(point)
        if data.get("result_commit") is None:
            continue
        commit, raw_commit = small_json(
            beneath(actual, data["result_commit"]), MIB - metadata, budget=budget
        )
        metadata += len(raw_commit)
        identity = commit.get("identity", {})
        source = data.get("checkpoint_source_attempt", row.get("attempt"))
        expected = {
            "point_id": point,
            "execution_id": selection["execution_id"],
            "attempt": source,
            "plan_fingerprint": selection["plan_fingerprint"],
        }
        if (
            not isinstance(identity, dict)
            or not integer(source, 1)
            or any(
                type(identity.get(k)) is not type(v) or identity[k] != v
                for k, v in expected.items()
            )
        ):
            raise ValueError("native source point commit differs")
        relative = str((actual.relative_to(actual.parent)) / data["result_commit"])
        # Source attempt is independent of enclosing input attempt; no UUID stamping.
        pins.append(
            {
                "point_id": point,
                "source_attempt": source,
                "point_commit_path": relative,
                "point_commit_sha256": digest(raw_commit),
            }
        )
        beneath(actual, data.get("artifact_root"))
        points.append(
            {
                "point_id": point,
                "source_attempt": source,
                "artifact_root": data["artifact_root"],
            }
        )
    return points, pins


def observe(spec_path, pid, start_ticks, control_fd):
    """Only this disposable process imports Arrow or touches job/scratch/NFS files."""
    from .telemetry import TelemetryWriter

    budget = ObserverIO(16384)
    root, spec, known = _spec(spec_path, budget=budget)
    selection = spec["selection"]
    credits = spec["credits"]
    began = time.monotonic()
    cursors = {}
    journal_bytes = 0
    previous_epoch = -1
    missing_sources = set()
    publisher = ScalarPublisher(
        root,
        selection,
        scalar_credit=credits["scalar_bytes"],
        reserve_bytes=credits["reserve_bytes"],
        budget=budget,
    )
    writer = None
    failure = None
    closed = None

    def admit(estimate=8 * MIB):
        # Software forecast, not atomic filesystem quota/IO enforcement. Conservative
        # 8 MiB burst is inside the previously admitted 32 MiB partition.
        count = 0
        used = 0
        if writer is not None:
            for path in writer.directory.iterdir():
                count += 1
                if count > 4096 or path.is_symlink() or not path.is_file():
                    raise ValueError("bounded session inventory exceeded")
                used += path.stat().st_size
        free = os.statvfs(root)
        budget.require(estimate)
        if (
            used + estimate > credits["performance_bytes"]
            or free.f_bavail * free.f_frsize < credits["reserve_bytes"] + estimate
        ):
            raise ValueError("observer admitted storage/IO partition exhausted")

    def writer_call(method, *args, **kwargs):
        # Local finite forecast for unchanged Arrow serialization/footer/hash I/O.
        # Exact JSON metadata has its own before-read/write hook on the same grant.
        admit()
        reservation = 4 * (writer.batch_bytes if writer is not None else 4 * MIB) + MIB
        budget.reserve(reservation)
        before = set(writer.directory.glob("*.parquet")) if writer is not None else set()
        try:
            return method(*args, **kwargs)
        finally:
            paths = set(writer.directory.glob("*.parquet")) if writer is not None else set()
            forecast = 4 * sum(path.stat().st_size for path in paths - before)
            budget.settle(reservation, forecast)
            if forecast > reservation:
                raise ValueError("writer exceeded its finite I/O forecast; capture unknown")

    def capture_point(actual, point, mode, terminal):
        nonlocal journal_bytes
        key = f"{point['point_id']}/attempt-{point['source_attempt']}"
        path = beneath(actual, point["artifact_root"]) / "progress" / "phases.jsonl"
        # Prefix replay also consumes the whole-job journal/IO cap. A long run
        # can deliberately disable observation rather than exceed its credit.
        cursor = cursors.get(key, {})
        offset = cursor.get("offset", 0)
        allowance = min(65536, 64 * MIB - journal_bytes - offset - 1, budget.remaining - offset - 1)
        if allowance < 1:
            raise ValueError("whole-job journal processed-byte cap exhausted")
        while True:
            rows, next_cursor, consumed = journal_batch(
                path, cursor, source_key=key, maximum_bytes=allowance, budget=budget
            )
            journal_bytes += consumed
            if journal_bytes > 64 * MIB:
                raise ValueError("observer journal/IO credit exhausted")
            if rows or next_cursor != cursor:
                admit()
                writer_call(writer.ingest_spans, rows, source_cursors={key: next_cursor})
                cursors[key] = next_cursor
            if (
                not terminal
                or next_cursor["complete_at_observation"]
                or next_cursor["offset"] == cursor.get("offset", 0)
            ):
                break
            cursor = next_cursor
            allowance = min(
                65536,
                64 * MIB - journal_bytes - cursor["offset"] - 1,
                budget.remaining - cursor["offset"] - 1,
            )
            if allowance < 1:
                raise ValueError("whole-job final journal drain exhausted")
        publisher.publish(
            actual,
            point,
            mode=mode,
            terminal=terminal,
            io_remaining=budget.remaining,
        )

    try:
        admit()
        writer = writer_call(TelemetryWriter,
            root / "performance",
            {**selection, "allocation_id": spec["allocation_id"]},
            {
                "scope": "root_julia_process_including_threads",
                "allocated_logical_cpus": spec["allocated_logical_cpus"],
            },
            metadata_budget=budget,
        )
        while True:
            try:
                os.read(control_fd, 64)
                terminal = True
            except BlockingIOError:
                terminal = False
            sample = proc_sample(
                pid, start_ticks, wall_seconds=time.monotonic() - began, budget=budget
            )
            sample.update(
                allocation_id=spec["allocation_id"],
                execution_id=selection["execution_id"],
                cpu_count=spec["allocated_logical_cpus"],
            )
            admit()
            writer_call(writer.record_resource, sample)
            try:
                if not (root / "capture-source.json").exists():
                    if terminal:
                        raise ValueError("actual source locator never published")
                    admit()
                    writer_call(writer.flush_due)
                    time.sleep(2)
                    continue
                locator, actual = _locator(root, spec, budget=budget)
                if locator["epoch"] < previous_epoch:
                    raise ValueError("actual source epoch reversed")
                previous_epoch = locator["epoch"]
                if not terminal and not (actual / "current_point.json").exists():
                    admit()
                    writer_call(writer.flush_due)
                    time.sleep(2)
                    continue
                points, pins = (
                    _final_points(actual, selection, known, budget=budget)
                    if terminal
                    else ([_current(actual, selection, known, budget=budget)], None)
                )
                for point in points:
                    if (
                        beneath(actual, point["artifact_root"])
                        / "progress"
                        / "phases.jsonl"
                    ).exists():
                        capture_point(actual, point, locator["mode"], terminal)
                    else:
                        missing_sources.add(point["point_id"])
                        # Scalar publication still works when phase telemetry is absent.
                        publisher.publish(
                            actual,
                            point,
                            mode=locator["mode"],
                            terminal=terminal,
                            io_remaining=budget.remaining,
                        )
            except FileNotFoundError:
                # Most errors are wrapped safely below; no current point at launch
                # is an explicit capture gap, never fabricated rows.
                pass
            admit()
            writer_call(writer.flush_due)
            if terminal:
                admit()
                closed = writer_call(writer.close)
                if locator["mode"] in ("direct", "staged"):
                    make_binding(
                        root,
                        selection=selection,
                        result_path="result/series_result.json",
                        point_pins=pins,
                        closed_path=closed,
                        budget=budget,
                        capture_status={
                            "complete": not missing_sources
                            and bool(cursors)
                            and all(
                                c.get("complete_at_observation")
                                for c in cursors.values()
                            ),
                            "terminal_tail": "may_be_unobserved",
                            "root_scope": "root_julia_process_including_threads",
                            "journal_processed_bytes": journal_bytes,
                            "live_publication_incomplete": publisher.disabled,
                            "phase_sources_missing": bool(missing_sources),
                            "observer_io_bytes_before_binding": budget.used,
                            "writer_io_is_local_forecast": True,
                        },
                    )
                break
            time.sleep(2)
    except (OSError, ValueError) as error:
        failure = error
        # Missing/incomplete capture is evidence; no unbudgeted inherited-log write.
    finally:
        if writer is not None and closed is None:
            try:
                admit()
                writer_call(writer.close)
            except OSError, ValueError:
                pass
        os.close(control_fd)
    return 0 if failure is None else 2


def capture_run(spec_path, solver, arguments):
    spec_path = str(Path(spec_path).absolute())

    def factory(pid, start, fd):
        return [
            sys.executable,
            "-m",
            "qcl_negf_results.capture",
            "observe",
            spec_path,
            str(pid),
            str(start),
            str(fd),
        ]

    return supervise([solver, *arguments], observer_factory=factory)


if __name__ == "__main__":
    if len(sys.argv) != 6 or sys.argv[1] != "observe":
        raise SystemExit(2)
    raise SystemExit(
        observe(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
    )
