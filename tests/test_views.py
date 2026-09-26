from __future__ import annotations
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from qcl_negf_results.presentation import atomic_json, committed_generation
from qcl_negf_results.render import materialize, materialize_series
from native_result_fixtures import declare_native


def generation_fixture(root: Path, *, generation=1, quality="unconverged"):
    destination = root / "artifacts" / f"generation-{generation:06d}"
    destination.mkdir(parents=True)
    path = destination / "analysis.h5"
    with h5py.File(path, "w") as file:
        file["axes/z_nm"] = np.array([0., 1., 2.])
        file["axes/energy_eV"] = np.array([-.1, .2])
        file["axes/k_per_nm"] = np.array([0., 1.])
        for name in ("structure", "external", "hartree", "total"):
            file[f"observables/potential_{name}_eV"] = [0., .1, .2]
        file["observables/density_per_m3"] = [1e20, -1e-10, 2e20]
        file["observables/spectral"] = [[1., 2.], [3., 4.]]
        file["observables/occupied_spectral"] = [[.1, .2], [.3, .4]]
        file["observables/spatial_energy_density"] = [[1., -1., np.nan], [0., .5, 2.]]
        file["observables/spatial_energy_density"].attrs["units"] = "m^-3/eV"
        file["basis/effective_wavefunctions/real"] = np.array([[1.,0.],[.5,.5],[0.,1.]])
        file["diagnostics/scba/r_K"] = [1e-2, 0, 1e-4, float("nan"), 1e-6]
        file["diagnostics/scba/sequence"] = np.arange(1, 6, dtype=np.int64)
        file["diagnostics/outer/r_U"] = [1e-2, 1e-3]
        file["diagnostics/validation_json"] = json.dumps({"evaluated": True, "strict_passed": False, "metrics": [{"metric": "r_sum", "value": .43, "strict_threshold": .05, "approximate_threshold": .05, "strict_passed": False}], "messages": ["r_sum exceeds threshold"]})
        file["diagnostics/positivity_json"] = json.dumps({"matrices": [{"matrix_kind": name, "minimum_eigenvalue": -.001, "unit": "eV" if name == "broadening" else "1/eV", "strict_passed": False} for name in ("spectral", "occupied", "unoccupied", "broadening")]})
        for name, axes in {"axes/z_nm":"z", "axes/energy_eV":"E", "axes/k_per_nm":"k",
                           "observables/spatial_energy_density":"E,z", "observables/spectral":"E,k",
                           "observables/occupied_spectral":"E,k", "basis/effective_wavefunctions/real":"z,state",
                           "observables/density_per_m3":"z",
                           **{f"observables/potential_{name}_eV":"z" for name in ("structure", "external", "hartree", "total")}}.items():
            file[name].attrs["logical_axis_order"]=axes
            file[name].attrs["logical_shape"]=",".join(str(size) for size in file[name].shape)
        declare_native(file, "qcl-negf-physics-analysis-v4", "physics.analysis", scba_rows=5)

    identity = {"point_id":"p1", "execution_id":"ex1", "attempt":1, "plan_fingerprint":"f"*64}
    commit = {"schema":"qcl-negf.artifact-commit.v2", "contract_set": "qcl-negf.results.v1", "identity":identity, "generation":generation,
              "quality":quality, "terminal_status":"completed", "scientific_accepted":False,
              "physics_ready":True, "checkpoint_ready":True, "artifacts":[{"role":"physics.analysis", "path":"analysis.h5", "schema":"qcl-negf-physics-analysis-v4", "contract_set": "qcl-negf.results.v1", "media_type":"application/x-hdf5", "profile":"science", "dependencies":[], "sha256":hashlib.sha256(path.read_bytes()).hexdigest(), "bytes":path.stat().st_size}]}
    atomic_json(destination / "commit.json", commit)
    digest = hashlib.sha256((destination / "commit.json").read_bytes()).hexdigest()
    atomic_json(root / "artifacts/current.json", {"schema":"qcl-negf.artifact-pointer.v2", "contract_set": "qcl-negf.results.v1", "generation":generation, "commit_path":f"generation-{generation:06d}/commit.json", "sha256":digest})
    return destination, commit, digest


def test_native_fields_materialize_without_json_arrays_and_preserve_scientific_evidence(tmp_path):
    generation, commit, digest = generation_fixture(tmp_path / "point")
    source_before = (generation / "analysis.h5").read_bytes()
    view = materialize(generation, commit, digest, tmp_path / "views")
    assert view["scientific_accepted"] is False and view["presentation_ready"] is True
    assert view["validation"]["metrics"][0]["value"] == .43
    assert len(view["positivity"]["matrices"]) == 4
    assert {figure["id"] for figure in view["figures"]} >= {"potentials", "density", "wavefunctions", "spectra", "energy-density", "convergence-scba", "convergence-outer"}
    for figure in view["figures"]:
        assert figure["source_commit"] == digest
        assert (tmp_path / "views" / figure["path"]).is_file()
        assert "values" not in figure and "x" not in figure
    convergence = next(f for f in view["figures"] if f["id"] == "convergence-scba")
    svg = (tmp_path / "views" / convergence["path"]).read_text()
    assert "log" in svg and "разрывы/неположительные значения: 2" in svg
    assert "nan" not in svg.lower()
    image = next(f for f in view["figures"] if f["id"] == "energy-density")
    assert image["native_shape"] == [2, 3] and image["color_range"] == [-2, 2]
    assert image["media_type"] == "image/svg+xml"
    svg = (tmp_path / "views" / image["path"]).read_text()
    assert 'viewBox="0 0 840 560"' in svg and "z, nm" in svg and "E, eV" in svg
    assert "линейная шкала" in svg and "m⁻³/eV" in svg
    assert 'href="data:image/png;base64,' in svg
    assert (tmp_path / "views" / image["raw_path"]).read_bytes().startswith(b"\x89PNG")
    assert (generation / "analysis.h5").read_bytes() == source_before
    assert len(json.dumps(view).encode()) < 100_000


def test_immutable_renderer_cache_and_corrupt_hdf5_rejected(tmp_path, monkeypatch):
    generation, commit, digest = generation_fixture(tmp_path / "point")
    view = materialize(generation, commit, digest, tmp_path / "views")
    monkeypatch.setattr(h5py, "File", lambda *args, **kwargs: pytest.fail("cache hit reopened numerical HDF5"))
    assert materialize(generation, commit, digest, tmp_path / "views") == view
    (generation / "analysis.h5").write_bytes(b"truncated")
    with pytest.raises(ValueError, match="hash"):
        materialize(generation, commit, digest, tmp_path / "fresh-cache")


def test_pointer_hash_and_path_escape_are_rejected(tmp_path):
    generation_fixture(tmp_path)
    pointer = tmp_path / "artifacts/current.json"
    value = json.loads(pointer.read_text()); value["sha256"]="0"*64;atomic_json(pointer,value)
    with pytest.raises(ValueError,match="hash"):
        committed_generation(tmp_path / "artifacts")
    value["commit_path"]="../../escape.json"; atomic_json(pointer,value)
    with pytest.raises(ValueError,match="escaped"):
        committed_generation(tmp_path / "artifacts")


def test_series_graph_has_explicit_unaccepted_marker_not_interpolated_line(tmp_path):
    rows=[{"id":f"p{i}","execution_id":"ex","scientific_accepted":i!=1,"coordinates":{"temperature_K":70,"voltage_per_period_V":i*.01,"branch":"forward","order":i},"observables":{"current_density_A_per_m2":i+1}} for i in range(3)]
    figures=materialize_series(rows,tmp_path)
    svg=(tmp_path/figures[0]["path"]).read_text()
    assert "непринята" in svg and svg.count("<circle") == 3
    assert "<polyline" not in svg  # neither isolated accepted endpoint disappears or crosses the failure
    assert rows[1]["observables"]["current_density_A_per_m2"]==2


def test_undeclared_axis_order_is_rejected_without_transpose_guess(tmp_path):
    generation, commit, digest = generation_fixture(tmp_path / "point")
    path = generation / "analysis.h5"
    with h5py.File(path, "r+") as record:
        record["observables/spatial_energy_density"].attrs["logical_axis_order"] = "z,E"
    commit["artifacts"][0]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="axis annotation"):
        materialize(generation, commit, digest, tmp_path / "views")


def test_empty_outer_history_does_not_hide_committed_scba_physics(tmp_path):
    generation, commit, digest = generation_fixture(tmp_path / "point")
    path = generation / "analysis.h5"
    with h5py.File(path, "r+") as record:
        del record["diagnostics/outer/r_U"]
        record["diagnostics/outer/r_U"] = np.array([], dtype=float)
    commit["artifacts"][0]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    result = materialize(generation, commit, digest, tmp_path / "views")
    assert result["presentation_ready"] and any(figure["id"] == "convergence-scba" for figure in result["figures"])


def history_fixture(root, *, sequence=(1, 2, 3), cumulative=False, residual=(.1, .01, .001), identity=None, iteration=None, outer=None, sources=None, currents=None):
    from qcl_negf_results.progress import HISTORY_INDEX_SCHEMA

    identity = identity or {"point_id": "p1", "execution_id": "ex1", "attempt": 1, "plan_fingerprint": "f" * 64}
    root.mkdir(parents=True, exist_ok=True)
    path = root / ("history.h5" if cumulative else "segment-000001.h5")
    with h5py.File(path, "w") as file:
        metadata = file.create_group("metadata")
        metadata.attrs.update(scba_rows=len(sequence), outer_rows=0)
        if cumulative:
            metadata["source_segments_json"] = json.dumps(sources or [{"identity": identity, "scba_rows": len(sequence), "outer_rows": 0}])
        else:
            metadata["identity_json"] = json.dumps(identity)
        file["scba/sequence"] = np.asarray(sequence, dtype="int64")
        file["scba/iteration"] = np.asarray(iteration if iteration is not None else range(1, len(sequence) + 1), dtype="int64")
        file["scba/outer_iteration"] = np.asarray(outer if outer is not None else [1] * len(sequence), dtype="int64")
        file["scba/r_K"] = np.asarray(residual, dtype="float64")
        file["scba/J"] = np.asarray(currents if currents is not None else np.arange(1, len(sequence) + 1, dtype="float64"), dtype="float64")
        file["outer/sequence"] = np.array([], dtype="int64")
        declare_native(file, "qcl-negf-scientific-history-v4", "science.history", scba_rows=len(sequence))
    artifact = {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size,
                "scba_rows": len(sequence), "outer_rows": 0, "role": "science.history",
                "schema": "qcl-negf-scientific-history-v4", "contract_set": "qcl-negf.results.v1", "media_type": "application/x-hdf5",
                "profile": "science", "dependencies": []}
    pointer = {"schema": HISTORY_INDEX_SCHEMA, "contract_set": "qcl-negf.results.v1", "identity": identity, "generation": 1, "updated_unix": 10.,
               "scba_rows": len(sequence), "outer_rows": 0, "segments": [artifact]}
    if not cumulative:
        atomic_json(root / "index.json", pointer)
    return artifact, pointer


@pytest.mark.parametrize("damage", ["hash", "identity", "sequence", "path"])
def test_live_history_rejects_corrupt_or_foreign_segments(tmp_path, damage):
    from qcl_negf_results.progress import live_history, materialize_history

    artifact, pointer = history_fixture(tmp_path / "history")
    identity = pointer["identity"].copy()
    if damage == "identity":
        identity["point_id"] = "other"
    elif damage == "path":
        pointer["segments"][0]["path"] = "../../foreign.h5"
    elif damage == "hash":
        pointer["segments"][0]["sha256"] = "0" * 64
    else:
        with h5py.File(tmp_path / "history" / artifact["path"], "r+") as file:
            file["scba/sequence"][...] = [1, 1, 3]
        pointer["segments"][0]["sha256"] = hashlib.sha256((tmp_path / "history" / artifact["path"]).read_bytes()).hexdigest()
    atomic_json(tmp_path / "history/index.json", pointer)
    with pytest.raises(ValueError):
        artifacts, digest, _ = live_history(tmp_path / "history", identity)
        materialize_history(tmp_path / "history", artifacts, digest, identity, tmp_path / "views", quality="not_evaluated", complete=False)


def test_live_log_graph_does_not_bridge_nonfinite_or_nonpositive_history(tmp_path):
    from qcl_negf_results.progress import materialize_history

    artifact, pointer = history_fixture(tmp_path / "history", residual=(.1, float("nan"), .001))
    result = materialize_history(tmp_path / "history", [artifact], "f" * 64, pointer["identity"],
                                 tmp_path / "views", quality="not_evaluated", complete=False)
    figure = next(figure for figure in result["figures"] if figure["id"] == "convergence-scba")
    svg = (tmp_path / "views" / figure["path"]).read_text()
    assert "<polyline" not in svg and svg.count("<circle") == 2
    assert "nan" not in svg.lower()


def test_global_envelope_preserves_spike_and_gap_across_many_tiny_segments():
    from qcl_negf_results.progress import _HistoryEnvelope
    from qcl_negf_results.render import MAX_CURVE_POINTS
    import math

    count = 25_000
    envelope = _HistoryEnvelope(count, logarithmic=True)
    spike, gap = 12_345, 12_349
    for index in range(count):
        # Every input is one separate closed segment: bin boundaries must not reset.
        value = 1e12 if index == spike else float("nan") if index == gap else 1e-6
        envelope.add([index + 1], [value], index)
    xx, yy = envelope.curve()
    assert len(xx) <= 3600 < MAX_CURVE_POINTS
    assert (spike + 1, 1e12) in list(zip(xx, yy, strict=True))
    segment = []
    for x, y in zip([*xx, float("nan")], [*yy, float("nan")], strict=True):
        if math.isfinite(x) and math.isfinite(y):
            segment.append(x)
        else:
            assert not segment or not min(segment) < gap + 1 < max(segment)
            segment.clear()


def test_monitor_uses_native_latest_outer_cycle_and_full_history_keeps_boundaries(tmp_path, monkeypatch):
    from qcl_negf_results import progress

    artifact, pointer = history_fixture(tmp_path / "history", sequence=(100, 101, 102, 103, 104),
        iteration=(1, 2, 3, 1, 2), outer=(1, 1, 1, 2, 2), residual=(.1, .001, .0001, .2, .02),
        currents=(9000., 9100., 9200., 7000., 7100.))
    captured = []
    original = progress._svg
    def capture(*args, **kwargs):
        captured.append((args, kwargs))
        return original(*args, **kwargs)
    monkeypatch.setattr(progress, "_svg", capture)
    source = (tmp_path / "history" / artifact["path"]).read_bytes()
    value = progress.materialize_history(tmp_path / "history", [artifact], "c" * 64,
        pointer["identity"], tmp_path / "views", quality="not_evaluated", complete=False, updated_unix=123.)
    convergence = next(figure for figure in value["figures"] if figure["id"] == "convergence-scba")
    assert convergence["history_rows"] == 5 and convergence["history_cycle_count"] == 2
    assert convergence["monitor_context"] == {"phase": "scba", "attempt": 1, "outer_iteration": 2,
        "first_iteration": 1, "last_iteration": 2, "first_sequence": 103, "last_sequence": 104, "rows": 2}
    monitor_args, monitor_kwargs = next(item for item in captured if item[1].get("monitor") and item[1].get("logarithmic"))
    assert monitor_args[1] == "Итерация SCBA, ν"
    assert monitor_args[3] == [("r_K", [1., 2.], [.2, .02])]
    full_args, full_kwargs = next(item for item in captured if not item[1].get("monitor") and item[1].get("logarithmic"))
    xx, yy = full_args[3][0][1:]
    assert set(x for x in xx if np.isfinite(x)) == {100, 101, 102, 103, 104}
    assert any(not np.isfinite(x) for x in xx)
    assert full_kwargs["boundaries"] == [(103., "μ=2 · a=1")]
    assert (tmp_path / "history" / artifact["path"]).read_bytes() == source


def test_resume_keeps_native_indices_and_excludes_previous_attempt_from_monitor(tmp_path, monkeypatch):
    from qcl_negf_results import progress

    identity = {"point_id": "p1", "execution_id": "ex1", "attempt": 10, "plan_fingerprint": "f" * 64}
    # Source inventory can be in filename order, while consolidated sequence is
    # chronological. The latest resumed attempt continues ν=3 rather than 1.
    sources = [{"identity": identity, "scba_rows": 2, "outer_rows": 0},
               {"identity": {**identity, "attempt": 2}, "scba_rows": 2, "outer_rows": 0}]
    artifact, pointer = history_fixture(tmp_path / "history", cumulative=True, sequence=(31, 32, 33, 34),
        iteration=(1, 2, 3, 4), outer=(7, 7, 7, 7), residual=(.1, .01, .009, .008), identity=identity, sources=sources)
    captured = []
    original = progress._svg
    def capture(*args, **kwargs):
        if kwargs.get("monitor") and kwargs.get("logarithmic"):
            captured.extend(args[3])
        return original(*args, **kwargs)
    monkeypatch.setattr(progress, "_svg", capture)
    value = progress.materialize_history(tmp_path / "history", [artifact], "d" * 64,
        pointer["identity"], tmp_path / "views", quality="not_evaluated", complete=False)
    context = value["history_context"]["scba"]
    assert context["attempt"] == 10 and context["outer_iteration"] == 7
    assert context["first_iteration"] == 3 and context["last_iteration"] == 4
    assert captured == [("r_K", [3., 4.], [.009, .008])]


def test_cycle_spanning_closed_segments_does_not_reset_at_file_boundary(tmp_path, monkeypatch):
    from qcl_negf_results import progress

    first, pointer = history_fixture(tmp_path / "history/first", sequence=(1, 2, 3),
        iteration=(1, 2, 1), outer=(1, 1, 2), residual=(.1, .01, .2))
    second, _ = history_fixture(tmp_path / "history/second", sequence=(4, 5, 6),
        iteration=(2, 3, 4), outer=(2, 2, 2), residual=(.1, float("nan"), .01))
    first["path"] = "first/" + first["path"]
    second["path"] = "second/" + second["path"]
    value = progress.materialize_history(tmp_path / "history", [first, second], "e" * 64,
        pointer["identity"], tmp_path / "views", quality="not_evaluated", complete=False)
    figure = next(figure for figure in value["figures"] if figure["id"] == "convergence-scba")
    assert figure["history_cycle_count"] == 2
    assert figure["monitor_context"]["rows"] == 4
    assert figure["monitor_context"]["first_iteration"] == 1 and figure["monitor_context"]["last_iteration"] == 4
    svg = (tmp_path / "views" / figure["monitor_path"]).read_text()
    # The valid samples before the NaN connect, while the final sample is isolated.
    assert svg.count("<polyline") == 1 and svg.count("<circle") == 1


def test_heatmap_uses_all_native_extrema_and_preserves_missing_and_signed_witnesses():
    from qcl_negf_results.render import _png

    values = np.ones((2050, 4))
    values[1, 1] = -1e22  # discarded by old stride sampling before scale selection
    values[4, 2] = np.nan
    original = values.copy()
    payload, annotation = _png(values)
    assert annotation["color_range"] == [-1e22, 1e22]
    assert annotation["native_range"] == [-1e22, 1.]
    assert annotation["missing_count"] == 1
    assert annotation["sampling"] == [1, 1] and annotation["display_shape"] == [2050, 4]
    assert payload.startswith(b"\x89PNG")
    import struct
    import zlib
    offset, blocks = 8, []
    while offset < len(payload):
        length = struct.unpack(">I", payload[offset:offset + 4])[0]
        if payload[offset + 4:offset + 8] == b"IDAT":
            blocks.append(payload[offset + 8:offset + 8 + length])
        offset += length + 12
    pixels = np.frombuffer(zlib.decompress(b"".join(blocks)), dtype=np.uint8).reshape(2050, 13)[:, 1:].reshape(2050, 4, 3)[::-1]
    assert pixels[1, 1].tolist() == [0, 0, 255]
    assert pixels[4, 2].tolist() == [155, 155, 155]
    np.testing.assert_equal(values, original)


def test_heatmap_places_nonuniform_axes_in_physical_coordinates():
    from qcl_negf_results.render import _heatmap_svg

    z = np.array([0., 1., 9.])
    energy = np.array([-.15, -.14, .05, .2])
    values = np.array([[1., 2., 3.], [4., -5., 6.], [7., 8., 9.], [10., 11., np.nan]])
    original = values.copy()
    svg, raw, annotation = _heatmap_svg(values, z, energy, "m^-3/eV", "unconverged")
    assert annotation["axes_extent"] == [0., 9., -.15, .2]
    assert annotation["display_projection"] == "nearest native cell on physical axes; no interpolation"
    assert annotation["color_range"] == [-11., 11.]
    assert "z, nm" in svg and "E, eV" in svg and "-0.15" in svg and "0.2" in svg
    assert "nan" not in svg.lower() and raw.startswith(b"\x89PNG")
    np.testing.assert_equal(values, original)


def test_tall_native_heatmap_materializes_as_landscape_without_changing_hdf5(tmp_path):
    generation, commit, _ = generation_fixture(tmp_path / "point")
    source = generation / "analysis.h5"
    with h5py.File(source, "r+") as record:
        for path in ("axes/energy_eV", "observables/spectral", "observables/occupied_spectral", "observables/spatial_energy_density"):
            del record[path]
        energy = np.linspace(-.1473913140194004, .263098718019446, 1921)
        record["axes/energy_eV"] = energy
        record["axes/energy_eV"].attrs["logical_axis_order"] = "E"
        values = np.sin(np.arange(1921)[:, None] / 100) * np.array([[1., 2., 3.]]) * 1e22
        record["observables/spatial_energy_density"] = values
        record["observables/spatial_energy_density"].attrs.update(units="m^-3/eV", logical_axis_order="E,z")
    contents = source.read_bytes()
    commit["artifacts"][0].update(bytes=len(contents), sha256=hashlib.sha256(contents).hexdigest())
    view = materialize(generation, commit, "a" * 64, tmp_path / "views")
    figure = next(item for item in view["figures"] if item["id"] == "energy-density")
    assert figure["native_shape"] == [1921, 3]
    assert figure["display_aspect_ratio"] == "3:2"
    svg = (tmp_path / "views" / figure["path"]).read_text()
    assert 'viewBox="0 0 840 560"' in svg and "1921 × 3" in svg
    assert "-0.1473913140194004" not in svg
    assert source.read_bytes() == contents


def test_line_figure_legends_wrap_long_labels_without_dropping_their_units():
    import xml.etree.ElementTree as ET
    from qcl_negf_results.render import _svg

    label = "Занятые состояния, продольный импульс k=0.01239 nm⁻¹"
    svg = _svg("Спектры", "E, eV", "trace (−iG˂), eV⁻¹", [(label, [0., 1.], [1., 2.]),
        (label + " (другая точка)", [0., 1.], [2., 3.])], "not_evaluated")
    tree = ET.fromstring(svg)
    assert tree.attrib["viewBox"].split()[2] == "840"
    assert 'font-size="22"' in svg and 'font-size="20"' in svg
    ns = {"s": "http://www.w3.org/2000/svg"}
    legend = [item for item in tree.findall(".//s:text", ns) if item.findall("s:tspan", ns)]
    assert len(legend) == 2
    for item, expected in zip(legend, (label, label + " (другая точка)"), strict=True):
        assert " ".join(part.text for part in item.findall("s:tspan", ns)) == expected
        assert len(item.findall("s:tspan", ns)) >= 1


@pytest.mark.parametrize("monitor", [False, True])
def test_convergence_references_preserve_raw_values_and_invalid_gaps(monitor):
    from xml.etree import ElementTree as ET
    from qcl_negf_results.render import _curve_limits, _svg

    values = np.array([.1, 0., 1e-5, -.1, 1e-12, np.nan, 1e-6])
    original = values.copy()
    curves = [("r_K", np.arange(len(values)), values)]
    svg = _svg("Convergence", "Iteration", "Residual", curves, "unconverged", logarithmic=True, monitor=monitor)
    root = ET.fromstring(svg)
    ns = {"s": "http://www.w3.org/2000/svg"}
    references = root.findall(".//s:g[@class='convergence-reference']", ns)
    assert {float(row.attrib["data-reference-level"]) for row in references} == {1e-4, 1e-6, 1e-8}
    assert all(row.find("s:path", ns).attrib.get("stroke-dasharray") for row in references)
    assert "критерии принятия заданы отдельно" in svg
    assert _curve_limits(curves, True)[2] < -12
    assert _curve_limits([("r_K", [1, 2], [.1, .01])], True)[2] < -8
    assert not root.findall(".//s:polyline", ns), "invalid and nonpositive values must disconnect each displayed sample"
    np.testing.assert_equal(values, original)
    linear = _svg("Current", "Iteration", "A/m²", curves, "unconverged")
    assert 'class="convergence-reference"' not in linear


def test_renderer_rejects_cached_view_from_another_contract_set(tmp_path):
    generation, commit, digest = generation_fixture(tmp_path / "point")
    destination = tmp_path / "views"
    view = materialize(generation, commit, digest, destination)
    cache = destination / Path(view["figures"][0]["path"]).parent / "index.json"
    stale = json.loads(cache.read_text())
    stale["contract_set"] = "qcl-negf.results.0.10.5"
    atomic_json(cache, stale)
    with pytest.raises(ValueError, match="contract set"):
        materialize(generation, commit, digest, destination)
