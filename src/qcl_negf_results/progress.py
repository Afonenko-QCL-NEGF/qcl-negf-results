"""Render durable scalar histories without reading full-state matrices.

The live index only references closed HDF5 segments. A rendered generation is
immutable and keyed by their inventory, never by transient browser samples.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from qcl_negf_contracts.artifacts import CONTRACT_SET, require_contract_set

from .presentation import atomic_json, inside, read_contract_json, read_json
from .render import RENDERER_REVISION, _svg, _text
from .native import HISTORY_SCHEMA, validate_native_handle

HISTORY_INDEX_SCHEMA = "qcl-negf-scientific-history-index-v2"


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def live_history(root: Path, identity: dict[str, Any]) -> tuple[list[dict[str, Any]], str, dict[str, Any]] | None:
    """Read a bounded inventory; numerical files are opened only by the worker."""
    pointer = inside(root, "index.json")
    if not pointer.is_file():
        return None
    value = read_json(pointer)
    require_contract_set(value)
    if value.get("schema") != HISTORY_INDEX_SCHEMA or value.get("identity") != identity:
        raise ValueError("live history index belongs to another point attempt")
    segments = value.get("segments")
    if not isinstance(segments, list):
        raise ValueError("live history segment inventory must be a list")
    paths = set()
    for item in segments:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ValueError("invalid live history segment")
        path = inside(root, item["path"])
        if path.suffix != ".h5" or item["path"] in paths:
            raise ValueError("duplicate or invalid live history segment path")
        paths.add(item["path"])
        for key in ("scba_rows", "outer_rows", "bytes"):
            if not isinstance(item.get(key), int) or isinstance(item[key], bool) or item[key] < 0:
                raise ValueError("invalid live history segment size")
        if not isinstance(item.get("sha256"), str) or len(item["sha256"]) != 64:
            raise ValueError("invalid live history segment digest")
    for phase in ("scba", "outer"):
        if sum(row[phase + "_rows"] for row in segments) != value.get(phase + "_rows"):
            raise ValueError("live history row count differs from its inventory")
    digest = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return segments, digest, value


def _identity(actual: dict[str, Any], expected: dict[str, Any], *, cumulative: bool) -> None:
    if any(actual.get(key) != expected.get(key) for key in ("point_id", "execution_id", "plan_fingerprint")):
        raise ValueError("scientific history belongs to another point")
    attempt = actual.get("attempt")
    cutoff = expected.get("attempt")
    if not isinstance(attempt, int) or isinstance(attempt, bool) or not isinstance(cutoff, int) or not 1 <= attempt <= cutoff or (not cumulative and attempt != cutoff):
        raise ValueError("scientific history belongs to another attempt")


def _inspect(handle: Any, identity: dict[str, Any]) -> dict[str, int]:
    import h5py
    import numpy as np

    validate_native_handle(handle, "science.history", HISTORY_SCHEMA)
    metadata = handle["metadata"]
    if _text(metadata.attrs.get("schema", "")) != HISTORY_SCHEMA or _text(metadata.attrs.get("artifact_role", "")) != "science.history":
        raise ValueError("unsupported scalar scientific history schema")
    if "identity_json" in metadata:
        encoded = metadata["identity_json"][()]
        _identity(json.loads(_text(encoded)), identity, cumulative=False)
    elif "source_segments_json" in metadata:
        encoded = metadata["source_segments_json"][()]
        inventory = json.loads(_text(encoded))
        if not isinstance(inventory, list):
            raise ValueError("invalid consolidated history source inventory")
        for source in inventory:
            _identity(source["identity"], identity, cumulative=True)
    else:
        raise ValueError("scientific history has no source identity")
    rows = {}
    for phase in ("scba", "outer"):
        count = int(metadata.attrs[phase + "_rows"])
        group = handle[phase]
        if count < 0 or not isinstance(group, h5py.Group):
            raise ValueError("invalid scientific history table")
        if "sequence" not in group:
            raise ValueError("scientific history is missing its sequence")
        for dataset in group.values():
            if not isinstance(dataset, h5py.Dataset) or dataset.shape != (count,) or dataset.dtype.kind not in "if" or dataset.dtype.itemsize != 8:
                raise ValueError("scientific history is not a native scalar table")
        previous = 0
        for start in range(0, count, 131_072):
            sequence = group["sequence"][start:start + 131_072]
            if len(sequence) and (sequence[0] <= previous or np.any(sequence[1:] <= sequence[:-1])):
                raise ValueError("scientific history sequence is not strictly increasing")
            if len(sequence):
                previous = int(sequence[-1])
        rows[phase] = count
    return rows


def _attempt_spans(handle: Any, phase: str, count: int) -> list[tuple[int, int, int]]:
    """Map cumulative row ordinals to attempts using lossless source counts.

    Consolidation sorts immutable sequence numbers. Attempts are chronological,
    whereas inventory filenames may be lexicographic (attempt-10 before -2).
    """
    metadata = handle["metadata"]
    if "identity_json" in metadata:
        return [(0, count, int(json.loads(_text(metadata["identity_json"][()]))["attempt"]))]
    inventory = json.loads(_text(metadata["source_segments_json"][()]))
    counts: dict[int, int] = {}
    for source in inventory:
        size = source.get(phase + "_rows")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ValueError("consolidated history lacks native attempt row counts")
        attempt = int(source["identity"]["attempt"])
        counts[attempt] = counts.get(attempt, 0) + size
    if sum(counts.values()) != count:
        raise ValueError("consolidated history attempt counts differ from its table")
    spans = []
    start = 0
    for attempt, size in sorted(counts.items()):
        spans.append((start, start + size, attempt))
        start += size
    return spans


def _coordinate_blocks(handle: Any, phase: str):
    """Yield bounded native coordinates; never infer SCBA indices from row count."""
    import numpy as np

    group = handle[phase]
    count = len(group["sequence"])
    if not count:
        return
    for name in ("iteration", "outer_iteration") if phase == "scba" else ("iteration",):
        if name not in group or group[name].dtype.kind != "i":
            raise ValueError("scientific history is missing native iteration coordinates")
    for first, last, attempt in _attempt_spans(handle, phase, count):
        for start in range(first, last, 131_072):
            stop = min(last, start + 131_072)
            sequence = group["sequence"][start:stop]
            iteration = group["iteration"][start:stop]
            outer = group["outer_iteration"][start:stop] if phase == "scba" else None
            if np.any(iteration < 1) or (outer is not None and np.any(outer < 1)):
                raise ValueError("scientific history has invalid iteration coordinates")
            yield start, stop, attempt, sequence, iteration, outer


class _CycleIndex:
    """Locate the final native cycle and retain a bounded set of axis markers."""

    def __init__(self, phase: str):
        self.phase = phase
        self.last: dict[str, Any] | None = None
        self.count = 0
        self.markers: list[tuple[float, str]] = []

    def observe(self, offset: int, attempt: int, sequence: Any, iteration: Any, outer: Any) -> list[int]:
        import numpy as np

        if not len(sequence):
            return []
        split = np.zeros(len(sequence), dtype=bool)
        # A checkpoint resume can retain μ and continue ν; the attempt boundary
        # still separates the persisted sources. Never draw a joining segment.
        previous = self.last
        split[0] = previous is None or attempt != previous["attempt"] or int(iteration[0]) <= previous["last_iteration"] or (
            self.phase == "scba" and int(outer[0]) != previous["outer_iteration"])
        split[1:] = iteration[1:] <= iteration[:-1]
        if self.phase == "scba":
            split[1:] |= outer[1:] != outer[:-1]
        starts = np.flatnonzero(split).tolist()
        for start, stop in zip([0, *[value for value in starts if value > 0]],
                               [*[value for value in starts if value > 0], len(sequence)], strict=True):
            if split[start]:
                self.count += 1
                self.last = {"phase": self.phase, "attempt": attempt,
                             "outer_iteration": int(outer[start]) if outer is not None else None,
                             "first_iteration": int(iteration[start]), "last_iteration": int(iteration[start]),
                             "first_sequence": int(sequence[start]), "last_sequence": int(sequence[start]),
                             "rows": 0, "offset": offset + start}
                if self.count > 1 and len(self.markers) < 24:
                    label = f"μ={int(outer[start])} · a={attempt}" if outer is not None else f"a={attempt}"
                    self.markers.append((float(sequence[start]), label))
            assert self.last is not None
            self.last.update(last_iteration=int(iteration[stop - 1]), last_sequence=int(sequence[stop - 1]),
                             rows=self.last["rows"] + stop - start)
        return starts

    def context(self) -> dict[str, Any] | None:
        return {key: value for key, value in self.last.items() if key != "offset"} if self.last else None


class _HistoryEnvelope:
    """Global display bins span every segment and fit the SVG input bound.

    At most 400 bins emit four extrema/endpoints each. A bin containing any
    invalid value emits isolated witnesses (up to nine entries including NaN
    separators), so a display curve never bridges an invalid source interval.
    """

    def __init__(self, count: int, *, logarithmic: bool):
        self.width = max(1, (count + 399) // 400)
        self.logarithmic = logarithmic
        self.bins: dict[int, dict[str, Any]] = {}

    def missing(self, offset: int, count: int) -> None:
        if count:
            for index in range(offset // self.width, (offset + count - 1) // self.width + 1):
                self.bins.setdefault(index, {})["invalid"] = True

    def break_before(self, offset: int) -> None:
        # A bin straddling a cycle boundary emits isolated extrema: downsampling
        # must not invent a connection between different self-consistency loops.
        self.bins.setdefault(offset // self.width, {})["invalid"] = True

    def add(self, sequence: Any, values: Any, offset: int) -> None:
        import numpy as np

        # The leading slice can start inside a bin left open by an earlier
        # segment. Boundaries always use the global row ordinal.
        start = 0
        while start < len(sequence):
            index = (offset + start) // self.width
            stop = min(len(sequence), (index + 1) * self.width - offset)
            xx = np.asarray(sequence[start:stop], dtype=float)
            yy = np.asarray(values[start:stop], dtype=float)
            valid = np.isfinite(xx) & np.isfinite(yy)
            if self.logarithmic:
                valid &= yy > 0
            bucket = self.bins.setdefault(index, {})
            bucket["invalid"] = bucket.get("invalid", False) or not bool(valid.all())
            observed = np.flatnonzero(valid)
            if len(observed):
                points = [(float(xx[i]), float(yy[i])) for i in
                          (observed[0], observed[-1], observed[np.argmin(yy[observed])], observed[np.argmax(yy[observed])])]
                bucket.setdefault("first", points[0])
                bucket["last"] = points[1]
                if "minimum" not in bucket or points[2][1] < bucket["minimum"][1]:
                    bucket["minimum"] = points[2]
                if "maximum" not in bucket or points[3][1] > bucket["maximum"][1]:
                    bucket["maximum"] = points[3]
            start = stop

    def curve(self) -> tuple[list[float], list[float]]:
        xs: list[float] = []
        ys: list[float] = []
        for _, bucket in sorted(self.bins.items()):
            points = sorted({bucket[name] for name in ("first", "last", "minimum", "maximum") if name in bucket})
            for x, y in points:
                if bucket.get("invalid"):
                    xs.append(float("nan"))
                    ys.append(float("nan"))
                xs.append(x)
                ys.append(y)
            if bucket.get("invalid") or not points:
                xs.append(float("nan"))
                ys.append(float("nan"))
        return xs, ys


def materialize_history(root: Path, artifacts: list[dict[str, Any]], source_hash: str,
                        identity: dict[str, Any], destination: Path, *, quality: str,
                        complete: bool, updated_unix: float | None = None) -> dict[str, Any]:
    import h5py
    import numpy as np

    # Both representations are immutable products of the same inventory. The
    # policy is part of the key; a pre-r5 global monitor asset cannot be reused.
    key = hashlib.sha256((source_hash + CONTRACT_SET + RENDERER_REVISION + "-history-v4-last-native-cycle-" + str(complete)).encode()).hexdigest()
    target = destination / key
    if (target / "index.json").is_file():
        return read_contract_json(target / "index.json", "qcl-negf.materialized-history.v1")
    figures = []
    rows = {"scba": 0, "outer": 0}
    paths = []
    cycles = {phase: _CycleIndex(phase) for phase in rows}
    last_sequence = {"scba": 0, "outer": 0}
    for artifact in artifacts:
        path = inside(root, artifact["path"])
        if path.stat().st_size != artifact["bytes"] or _hash(path) != artifact["sha256"]:
            raise ValueError("scientific history hash differs from its published inventory")
        with h5py.File(path, "r") as handle:
            counts = _inspect(handle, identity)
            for phase in rows:
                if phase + "_rows" in artifact and artifact[phase + "_rows"] != counts[phase]:
                    raise ValueError("scientific history row count differs from its published inventory")
                sequence = handle[phase]["sequence"]
                if len(sequence):
                    if int(sequence[0]) <= last_sequence[phase]:
                        raise ValueError("scientific history segments overlap or are out of sequence")
                    last_sequence[phase] = int(sequence[-1])
                for start, _, attempt, sequence, iteration, outer in _coordinate_blocks(handle, phase):
                    cycles[phase].observe(rows[phase] + start, attempt, sequence, iteration, outer)
                rows[phase] += counts[phase]
        paths.append(path)
    # One bounded envelope per scalar column and representation. The monitor
    # selects the last published cycle; worker progress is a separate snapshot.
    tables: dict[str, dict[str, _HistoryEnvelope]] = {"scba": {}, "outer": {}}
    monitor_tables: dict[str, dict[str, _HistoryEnvelope]] = {"scba": {}, "outer": {}}
    offsets = {"scba": 0, "outer": 0}
    scan = {phase: _CycleIndex(phase) for phase in rows}
    last_x = {"scba": 0, "outer": 0}
    for path in paths:
        with h5py.File(path, "r") as handle:
            for phase in tables:
                group = handle[phase]
                count = len(group["sequence"])
                if not count:
                    continue
                names = {name for name in group if name.startswith("r_") or name == "J"}
                context = cycles[phase].last
                assert context is not None
                cycle_start = context["offset"]
                for name in names - tables[phase].keys():
                    tables[phase][name] = _HistoryEnvelope(rows[phase], logarithmic=name != "J")
                    tables[phase][name].missing(0, offsets[phase])
                for name in names - monitor_tables[phase].keys():
                    monitor_tables[phase][name] = _HistoryEnvelope(context["rows"], logarithmic=name != "J")
                    monitor_tables[phase][name].missing(0, max(0, offsets[phase] - cycle_start))
                for start, stop, attempt, sequence, iteration, outer in _coordinate_blocks(handle, phase):
                    offset = offsets[phase] + start
                    boundaries = scan[phase].observe(offset, attempt, sequence, iteration, outer)
                    gaps = np.flatnonzero(np.diff(sequence, prepend=last_x[phase]) > 1).tolist()
                    last_x[phase] = int(sequence[-1])
                    selected = max(0, cycle_start - offset)
                    for name, envelope in tables[phase].items():
                        if name not in names:
                            envelope.missing(offset, stop - start)
                        else:
                            envelope.add(sequence, group[name][start:stop], offset)
                        for boundary in set([*boundaries, *gaps]):
                            if offset + boundary > 0:
                                envelope.break_before(offset + boundary)
                    if selected >= len(sequence):
                        continue
                    monitor_offset = offset + selected - cycle_start
                    for name, envelope in monitor_tables[phase].items():
                        if name not in names:
                            envelope.missing(monitor_offset, len(sequence) - selected)
                        else:
                            envelope.add(iteration[selected:], group[name][start + selected:stop], monitor_offset)
                        for gap in gaps:
                            if gap >= selected and monitor_offset + gap - selected > 0:
                                envelope.break_before(monitor_offset + gap - selected)
                offsets[phase] += count
    target.mkdir(parents=True, exist_ok=True)
    for phase, label in (("scba", "SCBA"), ("outer", "Хартри / Пуассон")):
        names = sorted(name for name in tables[phase] if name.startswith("r_"))
        context = cycles[phase].context()
        for kind, columns, logarithmic, title, units in (
            ("convergence", names, True, f"Сходимость: {label}", "Невязка, log₁₀"),
            ("current-history", ["J"], False, f"Ток по итерациям: {label}", "J, A/m²"),
        ):
            curves = [(name, *tables[phase][name].curve()) for name in columns if name in tables[phase]]
            monitor_curves = [(name, *monitor_tables[phase][name].curve()) for name in columns if name in monitor_tables[phase]]
            if not curves:
                continue
            identifier = f"{kind}-{phase}"
            filename = identifier + ".svg"
            x_label = "Сквозной номер записи (все внешние циклы)" if phase == "scba" else "Сквозной номер записи (все попытки)"
            monitor_x = "Итерация SCBA, ν" if phase == "scba" else "Внешняя итерация, μ"
            (target / filename).write_text(_svg(title, x_label, units, curves, quality, logarithmic=logarithmic,
                                               boundaries=cycles[phase].markers), encoding="utf-8")
            monitor_filename = identifier + "-monitor.svg"
            (target / monitor_filename).write_text(_svg(title, monitor_x, units, monitor_curves, quality,
                                                       logarithmic=logarithmic, monitor=True), encoding="utf-8")
            figures.append({"id": identifier, "title": title, "path": f"{key}/{filename}", "monitor_path": f"{key}/{monitor_filename}",
                            "media_type": "image/svg+xml", "source_datasets": [f"{phase}/{name}" for name in columns if name in tables[phase]],
                            "source_commit": source_hash, "quality": quality,
                            "units": {"x": x_label, "y": units}, "monitor_units": {"x": monitor_x, "y": units},
                            "sampling": "display-only envelope; complete scalar history remains in HDF5; cycle boundaries are disconnected",
                            "history_rows": rows[phase], "history_cycle_count": cycles[phase].count,
                            "monitor_context": context})
    value = {"schema": "qcl-negf.materialized-history.v1", "contract_set": CONTRACT_SET, "source_commit": source_hash, "figures": figures, "history_rows": rows,
             "history_context": {phase: cycles[phase].context() for phase in rows},
             "history_complete": complete, "history_updated_unix": updated_unix}
    atomic_json(target / "index.json", value)
    return value
