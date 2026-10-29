"""Bind native5 original context to its exact canonical owner and request.

The preserved test IDs now use fresh native5 constructor bytes. No legacy
container, solver getter or old scientific archive supplies an expected value.
"""
from copy import deepcopy
import json
import math
import struct
import tarfile

import h5py
import pytest

from qcl_negf_contracts.messages import ContractError
from qcl_negf_results.export import export_snapshot
from qcl_negf_results.provenance import verify_native_provenance
from qcl_negf_results.state import StateReader
from consumer_context_fixture import fixture, completed_fixture, change, fixed_json, PLAN
from result_v2_fixture import put, reference, ieee, source


def _tag(number):
    return ieee(struct.pack('>d', number).hex())


def _read_export(root, selected, observations, output, *, profile='science'):
    with StateReader(root, state_ref=selected, expected_identity=selected['identity']) as reader:
        receipt = export_snapshot(reader, output, observations_ref=observations,
            profile=profile, byte_budget=32*1024*1024, reserve_bytes=0)
    with tarfile.open(output / receipt['archive'], 'r:xz') as archive:
        return json.load(archive.extractfile('manifest.json'))


def _rebind_full(root, selected, observations, old_full, datasets=None):
    """Refresh literal envelopes after a manufactured Core owner mutation."""
    def bind(value, old, new):
        if isinstance(value, dict):
            if all(value.get(k) == v for k,v in old.items()): value.update(deepcopy(new))
            for child in value.values(): bind(child,old,new)
        elif isinstance(value,list):
            for child in value: bind(child,old,new)
    manifest=json.loads((root/selected['manifest']['path']).read_bytes())
    old_cp=deepcopy(manifest['checkpoint'])
    full=reference(root,'full.h5',old_full['schema'],'application/x-hdf5')
    cp=json.loads((root/old_cp['path']).read_bytes()); bind(cp,old_full,full)
    def relocate(node):
        if isinstance(node,dict):
            if node.get('kind')=='array' and 'file' in node and datasets:
                node['dataset']=datasets.get((node['dataset'],tuple(node['shape']),node['eltype']),node['dataset'])
            for child in node.values(): relocate(child)
        elif isinstance(node,list):
            for child in node: relocate(child)
    relocate(cp)
    cpref=put(root,old_cp['path'],cp,old_cp['schema'])
    old_d=next({k:r[k] for k in ('path','bytes','sha256','media_type','schema')} for r in manifest['artifacts'] if r['path']=='analysis.h5')
    with h5py.File(root/'analysis.h5','r+') as handle:
        d=json.loads(bytes(handle['final_diagnostics/declaration_json'][()]));bind(d,old_cp,cpref)
        del handle['final_diagnostics/declaration_json'];fixed_json(handle,'final_diagnostics/declaration_json',d)
    diagnostic=reference(root,'analysis.h5',old_d['schema'],'application/x-hdf5')
    for old,new in ((old_full,full),(old_cp,cpref),(old_d,diagnostic)): bind(manifest,old,new)
    selected['manifest']=put(root,selected['manifest']['path'],manifest,manifest['schema'])
    def updated(v):
        for old,new in ((old_full,full),(old_cp,cpref),(old_d,diagnostic)):bind(v,old,new)
        relocate(v)
    change(root,selected,observations,updated)


def _fixture(root, **kwargs):
    """Use actual native5 /arrays/ paths, beyond B0's declaration-only fixture."""
    selected,observations,value,checkpoint,values=fixture(root,**kwargs)
    datasets={}; declarations={}
    def walk(node):
        if isinstance(node,dict):
            if node.get('kind')=='array' and 'file' in node:
                key=(node['dataset'],tuple(node['shape']),node['eltype'])
                datasets.setdefault(key,'/arrays/a'+str(len(datasets)));declarations[key]=node
            for child in node.values():walk(child)
        elif isinstance(node,list):
            for child in node:walk(child)
    walk(checkpoint)
    with h5py.File(root/'full.h5','r+') as handle:
        handle.require_group('arrays')
        for key,new in datasets.items():
            old,shape,_=key
            handle.create_dataset(new,shape=tuple(map(int,shape)),dtype=handle[old].dtype)
    _rebind_full(root,selected,observations,checkpoint['payloads'][0],datasets)
    with h5py.File(root/observations['path'],'r') as handle:
        value=json.loads(bytes(handle['final_observations/declaration_json'][()]))
    checkpoint=json.loads((root/value['checkpoint']['path']).read_bytes())
    return selected,observations,value,checkpoint,values


def _completed_fixture(root):
    from unittest.mock import patch
    with patch('consumer_context_fixture.fixture',_fixture):
        return completed_fixture(root)


@pytest.mark.parametrize("file,name", [("analysis.h5", "identity_json"),
                                       ("recovery.h5", "point_identity_json")])
@pytest.mark.parametrize("field,value", [("point_id", "point-2"), ("execution_id", "other"),
    ("attempt", 2), ("plan_fingerprint", "c" * 64)])
def test_embedded_native_identity_cannot_be_relabelled_by_a_valid_commit(tmp_path, file, name, field, value):
    root,output=tmp_path/'run',tmp_path/'exports'
    selected,observations,_,cp,_=_fixture(root)
    embedded=dict(selected['identity']);embedded['plan_sha256' if field=='plan_fingerprint' else field]=str(value)
    target=observations['path'] if file=='analysis.h5' else 'full.h5'
    old_full=deepcopy(cp['payloads'][0])
    with h5py.File(root/target,'r+') as handle:
        del handle['metadata/identity_json'];fixed_json(handle,'metadata/identity_json',embedded)
    if target=='full.h5': _rebind_full(root,selected,observations,old_full)
    else:
        from consumer_context_fixture import refresh
        refresh(root,selected,observations)
    with pytest.raises(ContractError,match='identity'):
        _read_export(root,selected,observations,output,profile='full-state')
    assert not list(output.glob('*.tar.xz')) and not list(output.glob('*.json'))


def test_borrowed_checkpoint_is_checked_against_its_original_commit_attempt(tmp_path):
    root,output=tmp_path/'run',tmp_path/'exports'
    selected,observations,_,_,_,parent=_completed_fixture(root)
    original=(root/parent['manifest']['path']).read_bytes()
    manifest=_read_export(root,selected,observations,output,profile='full-state')
    proof=manifest['records'][0]['native_provenance']
    assert proof['source']['identity']['attempt']=='2'
    assert proof['restoration']['parent_state']==parent and parent['identity']['attempt']=='1'
    assert proof['original_evidence']['fields']['original_status']['value']=='invalid'
    assert proof['current_evidence']['status']=='validation_failed'
    assert (root/parent['manifest']['path']).read_bytes()==original
    assert parent['manifest'] in [v['reference'] for v in manifest['source_locator']]


def test_absent_native_identity_and_coordinates_are_explicitly_not_available(tmp_path):
    selected=source(tmp_path)
    with StateReader(tmp_path,state_ref=selected,expected_identity=selected['identity']) as reader:
        proof=verify_native_provenance(reader)
        assert proof['identity']['status']=='matched'
        assert proof['identity']['scope']=='verified_selected_manifest'
        assert proof['native_verified_references']==[]
        assert proof['coordinates']==dict(status='not_available',reason='explicit_final_observations_ref_required')
        assert proof['transport_provenance']['status']=='not_recorded'


@pytest.mark.parametrize("temperature", [70.0, 71.0])
def test_explicit_native_kelvin_value_matches_the_frozen_plan(tmp_path, temperature):
    root=tmp_path/'run'
    selected,observations,_,_,_=_fixture(root,plan_bytes=PLAN.replace(b'200.0',b'70.0'))
    def update(v):
        v['request']['temperature_K']=_tag(70.)
        v['context']['inputs']['T_L_K']=_tag(temperature)
        v['request']['coordinate_checks']['temperature'].update(requested=_tag(70.),actual=_tag(temperature))
    change(root,selected,observations,update)
    if temperature!=70.:
        with pytest.raises(ContractError,match='coordinate'):
            _read_export(root,selected,observations,tmp_path/'export')
    else:
        result=_read_export(root,selected,observations,tmp_path/'export')
        check=result['records'][0]['native_provenance']['coordinates']['temperature_K']
        assert check['status']=='verified' and check['requested']==check['actual']==_tag(70.)


@pytest.mark.parametrize("file,path", [("analysis.h5", "metadata/voltage_per_period_V"),
                                       ("recovery.h5", "inputs/V_period_V")])
@pytest.mark.parametrize("roundtrip", [False, True])
def test_reconstructed_voltage_is_exactly_matched_or_explicitly_not_verified(tmp_path, file, path, roundtrip):
    expected=.056;observed=math.nextafter(expected,math.inf) if roundtrip else expected
    root=tmp_path/'run'
    selected,observations,_,_,_=_fixture(root,plan_bytes=PLAN.replace(b'0.05',b'0.056'))
    def update(v):
        v['request']['voltage_per_period_V']=_tag(expected)
        v['context']['inputs']['V_period_V']=_tag(observed)
        v['request']['coordinate_checks']['voltage'].update(requested=_tag(expected),actual=_tag(observed),
            status='not_verified' if roundtrip else 'verified',reason='field_period_roundtrip' if roundtrip else None)
    change(root,selected,observations,update)
    # Both preserved cases now use the explicit O1 coordinate attestation; full
    # owner bytes are additionally qualified for the former recovery-owner case.
    result=_read_export(root,selected,observations,tmp_path/'export',profile='full-state' if file=='recovery.h5' else 'science')
    check=result['records'][0]['native_provenance']['coordinates']['voltage_per_period_V']
    assert check['status']==('not_verified' if roundtrip else 'verified')
    assert check['actual']==_tag(observed) and check['requested']==_tag(expected)
    assert check['source']=='final_observations.request.coordinate_checks.voltage'
    assert check['reason']==('field_period_roundtrip' if roundtrip else None)


@pytest.mark.parametrize("source_attempt", [1, 3])
def test_cumulative_history_keeps_earlier_attempts_and_rejects_future_ones(tmp_path, source_attempt):
    root=tmp_path/'run'
    selected,observations,_,_,_,parent=_completed_fixture(root)
    if source_attempt==3:
        manifest=json.loads((root/selected['manifest']['path']).read_bytes())
        manifest['lineage']['parent_state']['identity']['attempt']='3'
        selected['manifest']=put(root,selected['manifest']['path'],manifest,manifest['schema'])
        with pytest.raises(ContractError,match='attempt'):
            _read_export(root,selected,observations,tmp_path/'export')
    else:
        result=_read_export(root,selected,observations,tmp_path/'export')
        assert parent['manifest'] in [v['reference'] for v in result['source_locator']]
        proof=result['records'][0]['native_provenance']
        assert proof['restoration']['history']==json.loads((root/parent['manifest']['path']).read_bytes())['history']
        assert proof['restoration']['parent_state']['identity']['attempt']=='1'
        assert proof['source']['identity']['attempt']=='2'

def test_explicit_native5_provenance_uses_verified_original_request_and_nullable_status(tmp_path):
    from consumer_context_fixture import fixture
    from qcl_negf_results.state import StateReader
    from qcl_negf_results.provenance import verify_native_provenance
    selected, observations, value, _, _ = fixture(tmp_path)
    with StateReader(tmp_path, state_ref=selected, expected_identity=selected['identity']) as reader:
        plain = verify_native_provenance(reader)
        assert plain['coordinates'] == dict(status='not_available',reason='explicit_final_observations_ref_required')
        result = verify_native_provenance(reader, observations_ref=observations)
        assert result['source'] == selected
        assert result['coordinates']['temperature_K'] == dict(value['request']['coordinate_checks']['temperature'],
            source='final_observations.request.coordinate_checks.temperature')
        assert result['coordinates']['voltage_per_period_V']['actual'] == value['context']['inputs']['V_period_V']
        assert result['current_evidence']['scientific_accepted'] is None
        assert result['original_evidence']['fields']['original_status']['value'] == 'invalid'
        assert result['transport_provenance']['status'] == 'not_recorded'


def test_native5_provenance_has_no_path_or_identity_alias_fallback(tmp_path):
    with pytest.raises(TypeError,match='StateReader'):
        verify_native_provenance(tmp_path/'legacy.h5')
    with pytest.raises(TypeError):
        verify_native_provenance(tmp_path/'legacy.h5',{'point_id':'alias'},None)


@pytest.mark.parametrize('mutate', ['plan', 'model', 'parent'])
def test_native5_provenance_rejects_original_input_and_parent_tamper(tmp_path, mutate):
    selected,observations,_,_,_,parent=_completed_fixture(tmp_path)
    target=tmp_path/('plan.yaml' if mutate=='plan' else 'model.json' if mutate=='model' else parent['manifest']['path'])
    payload=bytearray(target.read_bytes());payload[-1]^=1;target.write_bytes(payload)
    with StateReader(tmp_path,state_ref=selected,expected_identity=selected['identity']) as reader:
        with pytest.raises(ContractError):verify_native_provenance(reader,observations_ref=observations)
