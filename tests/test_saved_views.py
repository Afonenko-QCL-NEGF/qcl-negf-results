"""Literal saved-view oracles; no solver or state/full owner is needed."""
import hashlib
import json

import h5py
import numpy as np
import pytest

from qcl_negf_contracts.messages import ContractError
from qcl_negf_results.commits import artifact_row, json_bytes
from qcl_negf_results.saved_views import (AnalysisReader, SavedViewLimits, brief_result,
                                        describe_saved_views, read_saved_view)

IDENTITY = {"point_id": "p0", "execution_id": "ex", "attempt": 2, "plan_fingerprint": "a" * 64}
SELECTION = {"run_uuid": "run", "execution_id": "ex", "attempt": 2, "calcjob_uuid": "job"}


def metadata(rows):
    return {"plan": {"schema": "qcl-negf-scientific-plan-v2", "root_definition_id": "study",
        "root_kind": "study", "name": "Study", "fingerprint": "a"*64, "computation_count": 3,
        "inclusions": [], "executions": [{"id": "ex", "definition_id": "study", "variant_id": "base"}],
        "points": [{"id": f"p{i}", "execution_id": "ex", "temperature_K": 70.,
                    "voltage_per_period_V": i*.01, "branch": "b", "order": i} for i in range(3)]},
        "series": {"schema": "qcl-negf-series-result-v3", "contract_set": "qcl-negf.results.v1",
                   "plan_fingerprint": "a"*64, "points": rows},
        "source": {"kind": "aiida_retrieved", "locator_id": "registered", "calcjob_uuid": "job"}}


def row(i, current=True, **extra):
    return {"id": f"p{i}", "execution_id": "ex", "attempt": 2, "status": "completed",
            "quality": "strict", "converged": True,
            "observables": {**({"current_density_A_per_m2": i+1} if current else {}),
                "scientific_assessment": {"scientific_accepted": True, "stationary_candidate_accepted": True}}, **extra}


@pytest.mark.parametrize("middle", [None, row(1, False), row(1, observables={"current_density_A_per_m2": None}), row(1, status="failed")])
def test_brief_preserves_missing_middle_and_never_connects_endpoints(middle, monkeypatch):
    monkeypatch.setattr(h5py, "File", lambda *a, **k: pytest.fail("brief opened HDF5"))
    rows = [row(2), row(0)] + ([] if middle is None else [middle])
    brief = brief_result(metadata(rows), selection=SELECTION, maximum_bytes=1024*1024)
    assert [p["point_id"] for p in brief["points"]] == ["p0", "p1", "p2"]
    assert [p["coordinates"]["voltage_per_period_V"] for p in brief["points"]] == [0., .01, .02]
    assert [p["point_id"] for p in brief["points"] if p["line_connectable"]] == ["p0", "p2"]
    assert brief["current_segments"] == [["p0"], ["p2"]]
    assert brief["points"][1]["missing"] == (["planned_row_absent"] if middle is None else
        ["current_unavailable"] if middle.get("observables", {}).get("current_density_A_per_m2") is None else [])


def test_brief_preserves_retained_and_borrowed_attempts_in_frozen_order(monkeypatch):
    monkeypatch.setattr(h5py, "File", lambda *a, **k: pytest.fail("brief opened HDF5"))
    accepted = {"scientific_accepted": True, "stationary_candidate_accepted": True}
    paused_assessment = {"scientific_accepted": False, "reason": "saved paused assessment"}
    retained = {"id": "p0", "execution_id": "ex", "attempt": 1, "status": "completed",
                "observables": {"current_density_A_per_m2": 7., "scientific_assessment": accepted},
                "data": {"source_commit_sha256": "1" * 64}}
    current = {"id": "p1", "execution_id": "ex", "attempt": 2, "status": "completed",
               "observables": {"current_density_A_per_m2": 8., "scientific_assessment": accepted},
               "data": {"source_commit_sha256": "2" * 64}}
    paused = {"id": "p2", "execution_id": "ex", "attempt": 2, "status": "paused",
              "observables": {"current_density_A_per_m2": 9., "scientific_assessment": paused_assessment},
              "data": {"checkpoint_source_attempt": 1, "source_commit_sha256": "3" * 64,
                       "pause_reason": "resource_pressure", "resume_kind": "checkpoint",
                       "recovery_origin": "last_committed_before_resource_pause"}}
    result = brief_result(metadata([paused, current, retained]), selection=SELECTION)
    assert result["selection"] == SELECTION
    assert [point["point_id"] for point in result["points"]] == ["p0", "p1", "p2"]
    assert [point["source_attempt"] for point in result["points"]] == [1, 2, 1]
    assert [point["source_commit_sha256"] for point in result["points"]] == ["1" * 64, "2" * 64, "3" * 64]
    assert [point["recorded_statuses"]["scientific_assessment"] for point in result["points"]] == [accepted, accepted, paused_assessment]
    assert result["current_segments"] == [["p0", "p1"]]
    gap = brief_result(metadata([paused, retained]), selection=SELECTION)
    assert [point["point_id"] for point in gap["points"]] == ["p0", "p1", "p2"]
    assert gap["points"][1]["missing"] == ["planned_row_absent"]
    assert gap["points"][1]["source_attempt"] is None
    assert gap["current_segments"] == [["p0"]]


@pytest.mark.parametrize("attempt", [3, 0, -1, True, False, 1.0, "1", None])
def test_brief_rejects_recorded_invalid_or_future_row_attempt(attempt):
    with pytest.raises(ContractError) as error:
        brief_result(metadata([row(0, attempt=attempt)]), selection=SELECTION)
    assert error.value.code == "identity_mismatch"


@pytest.mark.parametrize("row_attempt,source_attempt", [(2, 3), (1, 2), (2, 0), (2, -1),
                                                       (2, True), (2, False), (2, 1.0), (2, "1"), (2, None)])
def test_brief_rejects_checkpoint_source_outside_real_row_bounds(row_attempt, source_attempt):
    saved = row(0, attempt=row_attempt, status="paused", data={"checkpoint_source_attempt": source_attempt})
    with pytest.raises(ContractError) as error:
        brief_result(metadata([saved]), selection=SELECTION)
    assert error.value.code == "identity_mismatch"


def test_brief_legacy_absent_attempt_stays_unknown():
    legacy = {"id": "p0", "execution_id": "ex", "status": "completed",
              "observables": {"current_density_A_per_m2": 7.}}
    result = brief_result(metadata([legacy]), selection=SELECTION)
    assert result["points"][0]["source_attempt"] is None
    assert result["points"][0]["recorded_statuses"]["scientific_accepted"] is None
    assert result["current_segments"] == []


def fixture(root, *, denominator=True):
    root.mkdir()
    path = root / "analysis.h5"
    def attrs(obj, **values):
        for key, value in values.items():
            obj.attrs[key] = np.bytes_(value) if isinstance(value, str) else value
    with h5py.File(path, "w") as h:
        m = h.create_group("metadata")
        attrs(m, schema="qcl-negf-physics-analysis-v4", schema_version="4.0",
              artifact_role="physics.analysis", contract_set="qcl-negf.results.v1",
              current_convention="positive electron flow toward +z", energy_reference="saved reference")
        m["identity_json"] = np.bytes_(json.dumps(IDENTITY))
        axes = {"z": ("axes/z_nm", [0., 2., 7.], "nm"),
                "E": ("axes/energy_eV", [-.5, .25], "eV"),
                "k": ("axes/k_per_nm", [0., .3], "nm^-1"),
                "state": ("axes/state_index", [0, 1, 2], "1")}
        for name, (native, values, units) in axes.items():
            h[native] = values; attrs(h[native], units=units, logical_axis_order=name)
        def data(native, values, names, units):
            h[native] = np.asarray(values, dtype="f8")
            attrs(h[native], units=units, logical_axis_order=",".join(names),
                  axis_coordinate_paths_json=json.dumps([axes["state" if n == "state_prime" else n][0] for n in names]))
        data("observables/spatial_energy_density", [[-11., 12., np.nan], [21., np.inf, -np.inf]], ["E", "z"], "m^-3/eV")
        data("observables/density_per_m3", [0., -2., 8.], ["z"], "m^-3")
        data("observables/spectral", [[101., 102.], [201., 202.]], ["E", "k"], "eV^-1")
        data("basis/effective_wavefunctions/real", [[11., 12., 13.], [21., 22., 23.], [31., 32., 33.]], ["z", "state"], "nm^-1/2")
        data("basis/effective_wavefunctions/imag", [[-11., -12., -13.], [-21., -22., -23.], [-31., -32., -33.]], ["z", "state"], "nm^-1/2")
        data("basis/effective_density_matrix_per_m2/real", [[2., 11., 12.], [21., 4., 23.], [31., 32., 9.]], ["state", "state_prime"], "m^-2")
        attrs(h["basis"], effective_gauge="saved gauge; degenerate states not matched")
        h["axes/spatial_weights"] = [.1, .2, .7]
        attrs(h["axes/spatial_weights"], units="dimensionless", integration_measure="dx", physical_measure_multiplier=1e-9, physical_measure_units="m")
        if denominator:
            h["inputs/N_dop_2D_per_m2"] = 10.
            attrs(h["inputs/N_dop_2D_per_m2"], units="m^-2")
            h["inputs/f_ion"] = .5
            attrs(h["inputs/f_ion"], units="1")
    owner = artifact_row(path, relative="analysis.h5", role="physics.analysis", media_type="application/x-hdf5", schema="qcl-negf-physics-analysis-v4")
    commit = {"schema": "qcl-negf.artifact-commit.v2", "contract_set": "qcl-negf.results.v1", "generation": 1, "identity": IDENTITY,
              "quality": "unconverged", "terminal_status": "failed", "scientific_accepted": False, "artifacts": [owner]}
    commit_path = root / "commit.json"; commit_path.write_bytes(json_bytes(commit))
    return commit_path


def test_manifest_never_reads_numeric_arrays_or_full_owner(tmp_path, monkeypatch):
    commit = fixture(tmp_path / "record")
    original = h5py.Dataset.__getitem__
    def guarded(dataset, key):
        if dataset.dtype.kind in "biufc":
            pytest.fail("manifest read numeric payload")
        return original(dataset, key)
    monkeypatch.setattr(h5py.Dataset, "__getitem__", guarded)
    manifest = describe_saved_views(commit, expected_identity=IDENTITY)
    assert manifest["integrity"] == {"metadata_checked": True, "owner_sha_verified": False, "native_validation": "not_performed"}
    heat = next(v for v in manifest["views"] if v["view_id"] == "spatial_energy_density")
    assert heat["shape"] == [2, 3] and heat["axes"] == ["E", "z"]
    assert manifest["files"][0]["role"] == "physics.analysis"


def test_native_cells_axes_invalid_reasons_and_explicit_components(tmp_path):
    commit = fixture(tmp_path / "record")
    view = read_saved_view(commit, "spatial_energy_density", [{"start": 0, "stop": 2, "step": 1}, {"start": 0, "stop": 3, "step": 2}], expected_identity=IDENTITY)
    assert view["values"] == [[-11., None], [21., None]]
    assert view["invalid_mask"] == [[None, "nan"], [None, "negative_infinity"]]
    assert view["axes"][1]["values"] == [0., 7.]
    assert view["axes"][1]["weights"] == [.1, .7]
    assert view["integrity"]["owner_sha_verified"] is True
    assert read_saved_view(commit, "spectral_trace", [{"start": 0, "stop": 2, "step": 1}, 1], expected_identity=IDENTITY)["values"] == [102., 202.]
    assert read_saved_view(commit, "effective_wavefunctions_imag", [1, 2], expected_identity=IDENTITY)["values"] == -23.


def test_population_target_doping_not_selected_subset_sum(tmp_path):
    commit = fixture(tmp_path / "with")
    data = read_saved_view(commit, "effective_populations", [{"start": 0, "stop": 2, "step": 1}], expected_identity=IDENTITY)
    assert data["values"] == [2., 4.]
    assert data["fractions"]["values"] == [.4, .8]
    assert data["fractions"]["denominator"]["value"] == 5.
    missing = read_saved_view(fixture(tmp_path / "without", denominator=False), "effective_populations", [0], expected_identity=IDENTITY)
    assert missing["fractions"]["values"] is None
    assert missing["fractions"]["reason"] == "saved_target_doping_unavailable"


def test_corrupt_sha_identity_and_budgets_refuse_before_payload(tmp_path, monkeypatch):
    commit = fixture(tmp_path / "record")
    with pytest.raises(ContractError):
        describe_saved_views(commit, expected_identity={**IDENTITY, "attempt": 3})
    with pytest.raises(ContractError):
        read_saved_view(commit, "density", [slice(None)], expected_identity=IDENTITY, limits=SavedViewLimits(maximum_selected_bytes=1))
    path = commit.parent / "analysis.h5"
    declaration = json.loads(commit.read_bytes())
    declaration["artifacts"][0]["sha256"] = "0"*64  # same-size healthy HDF5, wrong committed digest
    commit.write_bytes(json_bytes(declaration))
    with pytest.raises(ContractError):
        read_saved_view(commit, "density", [0], expected_identity=IDENTITY)
    monkeypatch.setattr(h5py, "File", lambda *a, **k: pytest.fail("over-cap owner opened"))
    with pytest.raises(ContractError):
        describe_saved_views(commit, expected_identity=IDENTITY, limits=SavedViewLimits(maximum_owner_bytes=1))


def test_selection_and_total_allocations_are_checked_before_numeric_reads(tmp_path, monkeypatch):
    commit = fixture(tmp_path / "record")
    original = h5py.Dataset.__getitem__
    def no_numeric(dataset, key):
        if dataset.dtype.kind in "biufc":
            pytest.fail("refused selection read numeric array")
        return original(dataset, key)
    monkeypatch.setattr(h5py.Dataset, "__getitem__", no_numeric)
    for selection, limits in (([slice(None)], SavedViewLimits(maximum_selected_bytes=24)),
                              ([{"start": 0, "stop": 4, "step": 1}], SavedViewLimits()),
                              ([{"start": 0, "stop": 3, "step": 0}], SavedViewLimits())):
        with pytest.raises(ContractError):
            read_saved_view(commit, "density", selection, expected_identity=IDENTITY, limits=limits)
    with pytest.raises(ContractError):
        read_saved_view(commit, "../../state_dimensionless/GR", [0], expected_identity=IDENTITY)


def test_missing_units_and_gauge_are_explicit_capabilities(tmp_path):
    commit = fixture(tmp_path / "record")
    path = commit.parent / "analysis.h5"
    with h5py.File(path, "r+") as h:
        del h["observables/density_per_m3"].attrs["units"]
        del h["basis"].attrs["effective_gauge"]
    declaration = json.loads(commit.read_bytes())
    declaration["artifacts"] = [artifact_row(path, relative="analysis.h5", role="physics.analysis", media_type="application/x-hdf5", schema="qcl-negf-physics-analysis-v4")]
    commit.write_bytes(json_bytes(declaration))
    manifest = describe_saved_views(commit, expected_identity=IDENTITY)
    rows = {row["view_id"]: row for row in manifest["views"]}
    assert rows["density"]["availability"] == "unavailable"
    assert rows["effective_wavefunctions_real"]["availability"] == "unavailable"
    assert "units" in rows["density"]["reason"]
    assert "gauge" in rows["effective_wavefunctions_real"]["reason"]


def test_reader_closes_on_interrupted_view_and_reports_total_sha_delivery(tmp_path):
    commit = fixture(tmp_path / "record")
    with pytest.raises(RuntimeError):
        with AnalysisReader(commit, expected_identity=IDENTITY) as reader:
            stream = reader._stream
            raise RuntimeError("caller cancelled")
    assert stream.closed
    assert reader._handle is None
    with AnalysisReader(commit, expected_identity=IDENTITY) as reader:
        result = reader.read("density", [0])
        counters = result["io_counters"]
        assert counters["sha_bytes"] == reader.owner.size
        assert counters["bytes_read_total"] >= reader.owner.size
        assert counters["logical_selected_bytes"] == 24  # value + coordinate + weight
        assert result["values"] == 0.
        assert result["invalid_mask"] is None
        assert result["axes"][0]["values"] == [0.]


def test_native_nested_assessment_positive_unknown_and_false_are_projected_verbatim(monkeypatch):
    monkeypatch.setattr(h5py, "File", lambda *a, **k: pytest.fail("brief opened HDF5"))
    certificate = {"scientific_accepted": True, "stationary_candidate_accepted": True,
                   "final_fixed_point_audit": {"status": "recorded_sentinel"}}
    native = {"id": "p1", "execution_id": "ex", "attempt": 2, "status": "completed", "quality": "strict",
              "converged": True, "observables": {"current_density_A_per_m2": 2., "scientific_assessment": certificate}}
    brief = brief_result(metadata([row(0), native, row(2)]), selection=SELECTION)
    assert brief["current_segments"] == [["p0", "p1", "p2"]]
    recorded = brief["points"][1]["recorded_statuses"]
    assert recorded["scientific_accepted"] is True
    assert recorded["stationary_assessment"] == certificate
    assert recorded["scientific_assessment"] == certificate
    assert recorded["assessment_source_path"] == "observables.scientific_assessment"
    for assessment, expected in ((None, None), ({"scientific_accepted": False}, False)):
        middle = {**native, "observables": {"current_density_A_per_m2": 2.}}
        if assessment is not None:
            middle["observables"]["scientific_assessment"] = assessment
        # An imagined top-level flag cannot promote missing native assessment.
        middle["scientific_accepted"] = True
        result = brief_result(metadata([row(0), middle, row(2)]), selection=SELECTION)
        assert result["current_segments"] == [["p0"], ["p2"]]
        assert result["points"][1]["recorded_statuses"]["scientific_accepted"] is expected


def test_attribute_datatype_and_scalar_fence_precedes_high_level_read(tmp_path, monkeypatch):
    with h5py.File(tmp_path / "attributes.h5", "w") as h:
        numeric = np.empty((), dtype=object)
        numeric[()] = np.array([1, 2], dtype="i8")
        h.attrs.create("vlen_numeric", numeric, dtype=h5py.vlen_dtype(np.dtype("i8")))
        compound = np.empty((), dtype=np.dtype([("payload", h5py.vlen_dtype(np.dtype("i8")))]))
        compound[()] = (np.array([3, 4], dtype="i8"),)
        h.attrs.create("compound_vlen", compound, dtype=compound.dtype)
        h.attrs["nonscalar"] = np.array([1., 2.])
        h.attrs.create("healthy_utf8", "ν".encode(), dtype=h5py.string_dtype("utf-8", length=16))
        h.attrs["healthy_float"] = np.float64(1.25)
        manager = type(h.attrs)
        original = manager.__getitem__
        forbidden = {"vlen_numeric", "compound_vlen", "nonscalar"}
        def fenced(attributes, name):
            if name in forbidden:
                pytest.fail("forbidden attribute reached high-level payload read")
            return original(attributes, name)
        monkeypatch.setattr(manager, "__getitem__", fenced)
        reader = object.__new__(AnalysisReader)
        reader.limits = SavedViewLimits()
        reader._metadata_bytes = 0
        for name in forbidden:
            with pytest.raises(ContractError) as failure:
                reader._attr(h, name)
            assert failure.value.code == "unsupported_contract"
        assert reader._metadata_bytes == 0
        assert reader._attr(h, "healthy_utf8") == "ν"
        assert reader._attr(h, "healthy_float") == 1.25


def native_identity_fixture(root, extra):
    commit = fixture(root)
    path = root / "analysis.h5"
    with h5py.File(path, "r+") as h:
        del h["metadata/identity_json"]
        h["metadata/identity_json"] = np.bytes_(json.dumps({**IDENTITY, **extra}))
    declaration = json.loads(commit.read_bytes())
    declaration["artifacts"] = [artifact_row(path, relative="analysis.h5", role="physics.analysis", media_type="application/x-hdf5", schema="qcl-negf-physics-analysis-v4")]
    commit.write_bytes(json_bytes(declaration))
    return commit


@pytest.mark.parametrize("recorded,expected", [({"run_uuid": "A"}, {"run_uuid": "B"}),
    ({"calcjob_uuid": "job-A"}, {"calcjob_uuid": "job-B"}), ({"run_uuid": 1}, {"run_uuid": True})])
def test_native_only_foreign_identity_cannot_be_relabelled(tmp_path, recorded, expected):
    commit = native_identity_fixture(tmp_path / "record", recorded)
    with pytest.raises(ContractError) as failure:
        describe_saved_views(commit, expected_identity={**IDENTITY, **expected})
    assert failure.value.code == "identity_mismatch"
    # Matching recorded values remain readable even when the commit is legacy.
    describe_saved_views(commit, expected_identity={**IDENTITY, **recorded})


def test_genuinely_absent_legacy_extra_identity_remains_unknown(tmp_path):
    commit = fixture(tmp_path / "record")
    expected = {**IDENTITY, "run_uuid": "provider-run", "calcjob_uuid": "provider-job"}
    view = read_saved_view(commit, "density", [0], expected_identity=expected)
    assert view["selection"]["run_uuid"] == "provider-run"
    assert view["selection"]["calcjob_uuid"] == "provider-job"
    assert view["identity_binding"]["native_recorded_fields"] == sorted(IDENTITY)
    assert view["identity_binding"]["provider_only_fields"] == ["calcjob_uuid", "run_uuid"]


def _with_json_dependency(commit):
    from qcl_negf_results.model import canonical_bytes
    dependency = commit.parent / "model.json"
    configuration = {"fixture_only": True}
    dependency.write_bytes(json_bytes({"schema": "qcl-negf-resolved-configuration-v3",
        "contract_set": "qcl-negf.results.v1", "configuration": configuration,
        "hash_encoding": "qcl-negf-canonical-bytes-v1",
        "configuration_hash": hashlib.sha256(canonical_bytes(configuration)).hexdigest()}))
    declaration = json.loads(commit.read_bytes())
    declaration["artifacts"][0]["dependencies"] = ["model.json"]
    declaration["artifacts"].append(artifact_row(dependency, relative="model.json",
        role="model", media_type="application/json", schema="qcl-negf-resolved-configuration-v3"))
    commit.write_bytes(json_bytes(declaration))
    return dependency


@pytest.mark.parametrize("target", ["commit", "owner", "dependency"])
def test_final_symlink_swap_after_ancestry_check_is_rejected(tmp_path, monkeypatch, target):
    import os
    import qcl_negf_results.saved_views as saved
    commit = fixture(tmp_path / "record")
    dependency = _with_json_dependency(commit)
    selected = {"commit": commit, "owner": commit.parent / "analysis.h5",
                "dependency": dependency}[target]
    original_open = os.open
    swapped = []
    def swap_then_open(path, flags, *args, **kwargs):
        if os.fspath(path) == str(selected) and not swapped:
            held = selected.with_name(selected.name + ".held")
            selected.rename(held)
            selected.symlink_to(held.name)
            swapped.append(True)
        return original_open(path, flags, *args, **kwargs)
    monkeypatch.setattr(saved.os, "open", swap_then_open)
    with pytest.raises(ContractError) as failure:
        describe_saved_views(commit, expected_identity=IDENTITY)
    assert failure.value.code == "corrupt_result"
    assert swapped == [True]  # Rejection must reach the final-component open.


@pytest.mark.parametrize("kind", ["fifo", "socket", "directory"])
def test_regular_open_refuses_nonregular_without_blocking(tmp_path, monkeypatch, kind):
    import os
    import socket
    import qcl_negf_results.saved_views as saved
    path = tmp_path / "special"
    original_open = os.open
    descriptors = []
    def capture(*args, **kwargs):
        descriptor = original_open(*args, **kwargs)
        descriptors.append(descriptor)
        return descriptor
    monkeypatch.setattr(saved.os, "open", capture)
    sock = None
    try:
        if kind == "fifo": os.mkfifo(path)
        elif kind == "socket":
            sock = socket.socket(socket.AF_UNIX)
            sock.bind(str(path))
        else: path.mkdir()
        with pytest.raises(ContractError) as failure:
            saved._open_regular(path)
        assert failure.value.code == "corrupt_result"
    finally:
        if sock is not None: sock.close()
    for descriptor in descriptors:
        with pytest.raises(OSError): os.fstat(descriptor)


def test_opened_inode_size_check_closes_fd_and_never_stats_path(tmp_path, monkeypatch):
    import os
    from pathlib import Path
    import qcl_negf_results.saved_views as saved
    path = tmp_path / "owner"
    path.write_bytes(b"four")
    original_open = os.open
    descriptors = []
    def capture(*args, **kwargs):
        descriptor = original_open(*args, **kwargs)
        descriptors.append(descriptor)
        return descriptor
    monkeypatch.setattr(saved.os, "open", capture)
    monkeypatch.setattr(Path, "stat", lambda *a, **k: pytest.fail("size checked by pathname"))
    with pytest.raises(ContractError) as failure:
        saved._open_regular(path, expected_size=3)
    assert failure.value.code == "corrupt_result"
    assert len(descriptors) == 1
    with pytest.raises(OSError): os.fstat(descriptors[0])


@pytest.mark.parametrize("stage", ["fstat", "fdopen"])
def test_interruption_before_stream_transfer_closes_descriptor(tmp_path, monkeypatch, stage):
    import os
    import qcl_negf_results.saved_views as saved
    path = tmp_path / "small"
    path.write_bytes(b"tiny")
    original_open, original_fstat = os.open, os.fstat
    descriptors = []
    def capture(*args, **kwargs):
        descriptor = original_open(*args, **kwargs)
        descriptors.append(descriptor)
        return descriptor
    def interrupted(*args, **kwargs): raise KeyboardInterrupt("literal interrupt")
    monkeypatch.setattr(saved.os, "open", capture)
    monkeypatch.setattr(saved.os, stage, interrupted)
    with pytest.raises(KeyboardInterrupt, match="literal interrupt"):
        saved._open_regular(path)
    assert len(descriptors) == 1
    with pytest.raises(OSError): original_fstat(descriptors[0])


def test_missing_required_nofollow_primitive_has_no_follow_fallback(tmp_path, monkeypatch):
    import qcl_negf_results.saved_views as saved
    monkeypatch.delattr(saved.os, "O_NOFOLLOW")
    monkeypatch.setattr(saved.os, "open", lambda *a, **k: pytest.fail("unsupported open attempted"))
    with pytest.raises(ContractError) as failure:
        saved._open_regular(tmp_path / "absent")
    assert failure.value.code == "unsupported_contract"


@pytest.mark.parametrize("failure_type", [ContractError, KeyboardInterrupt])
def test_constructor_metadata_failure_closes_owned_inode(tmp_path, monkeypatch, failure_type):
    import os
    import qcl_negf_results.saved_views as saved
    commit = fixture(tmp_path / "record")
    original_open = saved._open_regular
    owned = []
    def capture(path, **kwargs):
        raw = original_open(path, **kwargs)
        if str(path).endswith("analysis.h5"): owned.append((raw, raw.fileno()))
        return raw
    def broken_metadata(self):
        if failure_type is ContractError: raise ContractError("literal metadata error", "corrupt_result")
        raise KeyboardInterrupt("literal metadata interruption")
    monkeypatch.setattr(saved, "_open_regular", capture)
    monkeypatch.setattr(AnalysisReader, "_storage", broken_metadata)
    with pytest.raises(failure_type):
        AnalysisReader(commit, expected_identity=IDENTITY)
    assert len(owned) == 1 and owned[0][0].closed
    with pytest.raises(OSError): os.fstat(owned[0][1])


@pytest.mark.parametrize("failure_type", [RuntimeError, KeyboardInterrupt])
def test_hdf_close_failure_still_closes_raw_and_repeated_close_is_safe(tmp_path, failure_type):
    import os
    commit = fixture(tmp_path / "record")
    reader = AnalysisReader(commit, expected_identity=IDENTITY)
    handle, stream = reader._handle, reader._stream
    descriptor = stream._raw.fileno()
    class InterruptedClose:
        def close(self):
            handle.close()
            raise failure_type("literal close failure")
    reader._handle = InterruptedClose()
    with pytest.raises(failure_type, match="literal close failure"):
        reader.close()
    assert stream.closed
    with pytest.raises(OSError): os.fstat(descriptor)
    reader.close()
    assert reader._handle is reader._stream is None


def test_owner_hash_and_hdf_share_open_inode_after_path_replacement(tmp_path, monkeypatch):
    import os
    from pathlib import Path
    import qcl_negf_results.saved_views as saved
    commit = fixture(tmp_path / "original")
    foreign_commit = fixture(tmp_path / "foreign")
    foreign = foreign_commit.parent / "analysis.h5"
    with h5py.File(foreign, "r+") as handle:
        handle["observables/density_per_m3"][0] = 99.
    owner = commit.parent / "analysis.h5"
    original_hdf, original_verify = h5py.File, saved._verify_stream
    streams, inodes, verified = [], [], []
    original_path_open = Path.open
    def no_owner_reopen(path, *args, **kwargs):
        if path == owner: pytest.fail("owner pathname reopened")
        return original_path_open(path, *args, **kwargs)
    def capture_hdf(stream, *args, **kwargs):
        assert isinstance(stream, saved._BudgetStream)
        streams.append(stream)
        info = os.fstat(stream._raw.fileno())
        inodes.append((info.st_dev, info.st_ino))
        handle = original_hdf(stream, *args, **kwargs)
        os.replace(foreign, owner)  # Different healthy owner at the old pathname.
        monkeypatch.setattr(Path, "open", no_owner_reopen)
        return handle
    def capture_verify(stream, size, digest):
        assert stream is streams[0]
        info = os.fstat(stream._raw.fileno())
        assert (info.st_dev, info.st_ino) == inodes[0]
        assert (owner.stat().st_dev, owner.stat().st_ino) != inodes[0]
        verified.append(True)
        return original_verify(stream, size, digest)
    monkeypatch.setattr(saved.h5py, "File", capture_hdf)
    monkeypatch.setattr(saved, "_verify_stream", capture_verify)
    result = read_saved_view(commit, "density", [0], expected_identity=IDENTITY)
    assert result["values"] == 0.  # Literal original, not replacement sentinel 99.
    assert result["integrity"]["owner_sha_verified"] is True
    assert verified == [True] and len(streams) == 1 and streams[0].closed


@pytest.mark.parametrize("stage", ["hdf_constructor", "owner_hash"])
def test_hdf_or_hash_error_closes_owner_descriptor(tmp_path, monkeypatch, stage):
    import os
    import qcl_negf_results.saved_views as saved
    commit = fixture(tmp_path / "record")
    original_open = saved._open_regular
    owned = []
    def capture(path, **kwargs):
        raw = original_open(path, **kwargs)
        if str(path).endswith("analysis.h5"): owned.append((raw, raw.fileno()))
        return raw
    native_error = ValueError("literal native read failure")
    def broken(*args, **kwargs): raise native_error
    monkeypatch.setattr(saved, "_open_regular", capture)
    if stage == "hdf_constructor":
        monkeypatch.setattr(saved.h5py, "File", broken)
        with pytest.raises(ContractError, match="saved native metadata cannot be read") as failure:
            describe_saved_views(commit, expected_identity=IDENTITY)
    else:
        monkeypatch.setattr(saved, "_verify_stream", broken)
        with pytest.raises(ContractError, match="saved native selection cannot be read") as failure:
            read_saved_view(commit, "density", [0], expected_identity=IDENTITY)
    assert failure.value.code == "corrupt_result"
    assert failure.value.__cause__ is native_error
    assert len(owned) == 1 and owned[0][0].closed
    with pytest.raises(OSError): os.fstat(owned[0][1])


def test_raw_descriptor_closes_even_if_budget_stream_close_is_interrupted(tmp_path):
    import os
    commit = fixture(tmp_path / "record")
    reader = AnalysisReader(commit, expected_identity=IDENTITY)
    stream = reader._stream
    raw = stream._raw
    descriptor = raw.fileno()
    original_close = stream.close
    def interrupted(): raise KeyboardInterrupt("literal stream close interruption")
    stream.close = interrupted
    with pytest.raises(KeyboardInterrupt, match="literal stream close interruption"):
        reader.close()
    assert raw.closed
    with pytest.raises(OSError): os.fstat(descriptor)
    reader.close()
    # Restore the injected method before destruction of the detached stream.
    stream.close = original_close
    stream.close()


def test_regular_open_uses_readonly_nofollow_cloexec_nonblock(tmp_path, monkeypatch):
    import os
    import qcl_negf_results.saved_views as saved
    path = tmp_path / "ordinary"
    path.write_bytes(b"literal")
    original_open = os.open
    flags_seen = []
    def capture(path, flags):
        flags_seen.append(flags)
        return original_open(path, flags)
    monkeypatch.setattr(saved.os, "open", capture)
    with saved._open_regular(path, expected_size=7) as raw:
        descriptor = raw.fileno()
        assert raw.read() == b"literal"
        assert os.get_inheritable(descriptor) is False
    assert len(flags_seen) == 1
    assert flags_seen[0] & os.O_ACCMODE == os.O_RDONLY
    for flag in (os.O_NOFOLLOW, os.O_CLOEXEC, os.O_NONBLOCK):
        assert flags_seen[0] & flag == flag
    with pytest.raises(OSError): os.fstat(descriptor)
