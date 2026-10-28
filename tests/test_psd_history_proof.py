from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from qcl_negf_contracts.messages import ContractError
from qcl_negf_results.history import verify_cumulative_history
from qcl_negf_results.witnesses import PACKED_SCHEMA
from test_export import cumulative_fixture


def _selected(parent, count):
    table = parent.create_group("selected_blocks")
    table.attrs.update(schema=PACKED_SCHEMA, index_origin=0)
    sequences = list(range(1, count + 1, 2))
    table["record_sequence"] = np.asarray(sequences, dtype=np.int64)
    text_type = h5py.string_dtype("utf-8")
    table.create_dataset("record_attributes_json", data=[json.dumps({"sequence": n,
        "matrix_kind": "unoccupied", "E0_eV": 0.01}) for n in sequences], dtype=text_type)
    table.create_dataset("group_metadata_json", data=[json.dumps({
        "matrices_dimensionless": {}, "matrices_dimensionless/Gp": {}, "critical_eigenvector": {}})], dtype=text_type)
    table["record_group_metadata_index"] = np.zeros(len(sequences), dtype=np.int64)
    descriptors = [{"path": name, "shape": shape, "attributes": {"units": "1"}}
                   for name, shape in [("matrices_dimensionless/Gp/real", [3, 3]),
                       ("matrices_dimensionless/Gp/imag", [3, 3]),
                       ("critical_eigenvector/real", [3]), ("critical_eigenvector/imag", [3])]]
    table.create_dataset("dataset_metadata_json", data=[json.dumps(d) for d in descriptors], dtype=text_type)
    records, indices, offsets, values = [], [], [0], []
    special = np.array([0x7FF8000000001234, 0x8000000000000000, 0], dtype=np.uint64).view(np.float64)
    for index in range(len(sequences)):
        for descriptor, payload in enumerate((np.eye(3) * -2.34e-12, np.zeros((3, 3)), special, np.zeros(3))):
            records.append(index)
            indices.append(descriptor)
            values.extend(payload.reshape(-1))
            offsets.append(len(values))
    table["payload_record_index"] = np.asarray(records, dtype=np.int64)
    table["payload_metadata_index"] = np.asarray(indices, dtype=np.int64)
    table["payload_offsets"] = np.asarray(offsets, dtype=np.int64)
    table["payload_values"] = np.asarray(values, dtype=np.float64)


def _psd_histories(tmp_path: Path) -> tuple[Path, Path, dict]:
    generations = cumulative_fixture(tmp_path / "run")
    for generation in generations:
        with h5py.File(generation / "history.h5", "a") as handle:
            count = len(handle["scba/sequence"])
            group = handle["psd_history"]
            group["available"][:] = 1
            group["matrix_kind"][:] = ["unoccupied"] * count
            values = -np.arange(count, dtype=np.float64)
            values[1] = np.nan
            group["minimum_eigenvalue"] = values
            group["minimum_eigenvalue"].attrs.update(units="1", logical_shape=str(count))
            del group["selected_blocks"]
            _selected(group, count)
    identity = json.loads((generations[-1] / "commit.json").read_bytes())["identity"]
    return generations[0] / "history.h5", generations[-1] / "history.h5", identity


def test_current_native5_history_preserves_payload_bits_markers_and_original_scalar(tmp_path: Path) -> None:
    from copy import deepcopy
    from qcl_negf_results.state import StateReader
    from qcl_negf_results.diagnostic_views import DiagnosticReader, HistoryViewRequest
    from result_v2_fixture import source,record,metadata,reference,integer,ieee,symbol,IDENTITY
    matrix=np.eye(3,dtype='<c16')*-2.34e-12
    vector=np.zeros(3,dtype='<c16')
    vector.view('<u8')[:]=[0x7ff8000000001234,0,0x8000000000000000,0,0,0]
    with h5py.File(tmp_path/'witness.h5','w') as f:
        metadata(f,representation='array_payload')
        f['arrays/matrix']=matrix;f['arrays/eigenvector']=vector
    owner=reference(tmp_path,'witness.h5','qcl-negf-scientific-history-v5','application/x-hdf5')
    def array(name,shape):
        return dict(kind='array',eltype='ComplexF64',shape=[str(n) for n in shape],order='F',
                    dataset='/arrays/'+name,file=deepcopy(owner))
    row=record()
    witness=dict(matrix_kind=symbol('unoccupied'),energy_index=integer(1),momentum_index=integer(1),
        **{name:ieee() for name in ('minimum_eigenvalue','block_norm','backward_error','absolute_defect',
                                  'relative_defect','ratio','hermiticity_defect')},
        matrices=dict(kind='dict',entries=[dict(key='Gp',value=array('matrix',(3,3)))]),
        eigenvector=array('eigenvector',(3,)))
    marker={name:ieee() for name in ('raw_hole_charge','represented_capacity','occupied_fraction_a',
        'empty_fraction_c','relative_correction_Gn','relative_correction_Gp','equilibrium_mu_eV','fdt_raw',
        'fdt_normalized','equilibrium_abs_current_A_m2','delta_energy_eV','sampled_gamma_over_dE_q10',
        'sampled_gamma_over_dE_q50','sampled_gamma_over_dE_q90','sampled_spectral_weight_underresolved',
        'relative_mode_weight_floor','lo_shift_over_dE','field_shift_over_dE','lo_boundary_occupied_fraction',
        'lo_boundary_spectral_fraction','field_boundary_occupied_fraction','field_boundary_spectral_fraction',
        'seed_mu_eV','seed_number_ratio','fdt_raw_seed_mu','fdt_normalized_seed_mu')}
    marker.update({name:integer(1) for name in ('measured_iteration','linewidth_sampled_blocks','marker_cadence','linewidth_max_blocks')})
    marker.update({name:symbol('not_measured') for name in ('equilibrium_status','linewidth_status','linewidth_sampling_method','fresh_map_status')})
    marker.update(equilibrium_applicable=False,
        channels=dict(kind='array',eltype='SCBAChannelMarker',shape=['0'],order='F',values=[]),
        collisions=dict(kind='array',eltype='SCBACollisionMarker',shape=['0'],order='F',values=[]))
    marker['fdt_raw']=ieee('7ff8000000001234');marker['seed_number_ratio']=ieee('8000000000000000')
    row['core']['fields']['row']['fields'].update(witness=dict(type='SCBAPhysicsWitness',fields=witness),
        physical_markers=dict(type='SCBAPhysicalMarkers',fields=marker))
    update=deepcopy(row);update['stream_ordinal']='2';update['core']['fields']['sequence']=integer(9)
    update['core']['fields']['action']=symbol('witness_update')
    update['core']['fields']['row']['fields']['witness']['fields']['ratio']=ieee('4008000000000000')
    update['target']=dict(scope='same_leaf',source_identity=IDENTITY,source_epoch='epoch',append_ordinal='1',
        leaf=None,row_offset='0',anchor_state=None,anchor_history=None)
    ref=source(tmp_path,[row,update],array_ref=owner)
    with StateReader(tmp_path,state_ref=ref,expected_identity=IDENTITY) as reader:
        reader.import_history();batch=DiagnosticReader(reader).read(HistoryViewRequest('scba',('J',)))
        assert batch.records==(row,update) and batch.witness_updates==(update,)
        assert len(batch.rows)==1 and batch.rows[0]['cells']['J']['tagged']==row['core']['fields']['row']['fields']['J']
        stored=batch.records[0]['core']['fields']['row']['fields']
        assert stored['physical_markers']['fields']==marker
        payload=stored['witness']['fields']
        assert reader.read_array(payload['matrices']['entries'][0]['value'],(slice(None),)*2).values.tobytes()==matrix.tobytes()
        assert reader.read_array(payload['eigenvector'],(slice(None),)).values.tobytes()==vector.tobytes()


@pytest.mark.parametrize("change", ["missing_group", "scalar", "kind", "matrix", "units", "unrecognized",
    "offset", "shape", "sequence", "nan_payload", "signed_zero", "marker", "missing_markers"])
def test_current_evidence_cannot_be_lost_through_history_coalescing(tmp_path: Path, change: str) -> None:
    # Retired v4 consolidation remains rejected; the positive native5 evidence
    # oracle above and canonical target/cursor tamper tests carry current proofs.
    old, new, identity = _psd_histories(tmp_path)
    with h5py.File(old if change == "unrecognized" else new, "a") as handle:
        group = handle["psd_history"]
        packed = group["selected_blocks"]
        if change == "missing_group":
            del handle["psd_history"]
        elif change == "scalar":
            group["minimum_eigenvalue"][0] = 0.0
        elif change == "kind":
            group["matrix_kind"][0] = "occupied"
        elif change == "matrix":
            packed["payload_values"][0] = 0.0
        elif change == "units":
            attrs = json.loads(packed["record_attributes_json"][0])
            attrs["E0_eV"] = 1
            packed["record_attributes_json"][0] = json.dumps(attrs)
        elif change == "unrecognized":
            handle.create_group("future_scientific_evidence")
        elif change == "offset":
            packed["payload_offsets"][-1] += 1
        elif change == "shape":
            descriptor = json.loads(packed["dataset_metadata_json"][0])
            descriptor["shape"] = [999]
            packed["dataset_metadata_json"][0] = json.dumps(descriptor)
        elif change == "sequence":
            packed["record_sequence"][0] = 99
        elif change == "nan_payload":
            packed["payload_values"][18] = np.nan
        elif change == "signed_zero":
            packed["payload_values"][19] = 0.0
        elif change == "marker":
            handle["physical_markers/fdt_raw"][0] = 1.0
        elif change == "missing_markers":
            del handle["physical_markers"]
    with pytest.raises(ContractError, match="lossless"):
        verify_cumulative_history(old, new, identity)


@pytest.mark.parametrize("legacy", ["schema", "schema_version", "selected_layout", "missing_markers"])
def test_previous_native_format_is_explicitly_rejected(tmp_path: Path, legacy: str) -> None:
    old, new, identity = _psd_histories(tmp_path)
    with h5py.File(old, "a") as handle:
        if legacy == "schema":
            handle["metadata"].attrs["schema"] = "qcl-negf-scientific-history-v2"
        elif legacy == "schema_version":
            handle["metadata"].attrs["schema_version"] = "2.0"
        elif legacy == "selected_layout":
            del handle["psd_history/selected_blocks"]
            handle["psd_history"].create_group("selected_blocks").create_group("sequence-1")
        else:
            del handle["physical_markers"]
    with pytest.raises(ContractError, match="lossless"):
        verify_cumulative_history(old, new, identity)
