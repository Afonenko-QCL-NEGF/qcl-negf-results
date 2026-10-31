"""Native5 same-open integrity and actual bounded array reads; no scientific solve."""
import copy
from dataclasses import replace
import os

import h5py
import numpy as np
import pytest
from qcl_negf_contracts.messages import ContractError
from qcl_negf_results.state import StateReader, ResultReadLimits, verify_recovery_bundle
from result_v2_fixture import IDENTITY, source, record, array_source


def selected_array(reader):
    reader.import_history()
    return reader.history_batch().records[0]['core']['fields']['row']['fields']['density']


def test_logical_hyperslab_reads_only_selected_values_and_remains_readonly(tmp_path, monkeypatch):
    ref = array_source(tmp_path, 100_000, value=1.25)
    with StateReader(tmp_path, state_ref=ref, expected_identity=IDENTITY) as reader:
        declaration = selected_array(reader)
        original, selections = h5py.Dataset.__getitem__, []
        def bounded(dataset, key, **kw):
            if dataset.name.startswith('/arrays/'):
                selections.append(key)
                assert key == (slice(5, 7, 1),)
            return original(dataset, key, **kw)
        monkeypatch.setattr(h5py.Dataset, '__getitem__', bounded)
        block = reader.read_array(declaration, (slice(5, 7),), maximum_bytes=16)
        assert block.values.tolist() == [1.25, 1.25]
        assert not block.values.flags.writeable and block.values.nbytes == 16
        assert len(selections) == 1
        assert block.io_counters['logical_selected_bytes'] == 16
        description = reader.describe_array(declaration)
        assert description.logical_shape == description.stored_shape == (100_000,)


def test_array_allocation_budget_fails_before_value_read(tmp_path, monkeypatch):
    ref = array_source(tmp_path, 50)
    with StateReader(tmp_path, state_ref=ref, expected_identity=IDENTITY) as reader:
        declaration = selected_array(reader)
        original = h5py.Dataset.__getitem__
        def forbidden(dataset, key, **kw):
            assert not dataset.name.startswith('/arrays/'), 'oversized tensor selection read'
            return original(dataset, key, **kw)
        monkeypatch.setattr(h5py.Dataset, '__getitem__', forbidden)
        with pytest.raises(ContractError) as error:
            reader.read_array(declaration, (slice(None),), maximum_bytes=8)
        assert error.value.code == 'budget_exceeded'


def test_native_chunk_decode_buffer_respects_limit_before_numerical_read(tmp_path, monkeypatch):
    ref = array_source(tmp_path, 512)
    limits = ResultReadLimits(maximum_array_bytes=16)
    with StateReader(tmp_path, state_ref=ref, expected_identity=IDENTITY, limits=limits) as reader:
        declaration = selected_array(reader)
        assert reader.describe_array(declaration).logical_shape == (512,)
        original = h5py.Dataset.__getitem__
        def bounded(dataset,key,**kw):
            assert not dataset.name.startswith('/arrays/'), 'oversized HDF chunk decoded'
            return original(dataset,key,**kw)
        monkeypatch.setattr(h5py.Dataset,'__getitem__',bounded)
        with pytest.raises(ContractError,match='chunk') as error:
            reader.read_array(declaration,(slice(0,1),),maximum_bytes=8)
        assert error.value.code == 'budget_exceeded'


def test_record_coordinate_chunk_budget_checked_before_payload_access(tmp_path, monkeypatch):
    import json
    from result_v2_fixture import reference, put
    ref = source(tmp_path,[record()])
    with h5py.File(tmp_path / 'records.h5','r+') as handle:
        values = handle['records/offsets'][:]
        del handle['records/offsets']
        handle.create_dataset('records/offsets',data=values,chunks=(1024,),maxshape=(None,))
    page=json.loads((tmp_path / 'catalog.json').read_text())
    page['artifacts'][-1].update(reference(tmp_path,'records.h5',page['artifacts'][-1]['schema'],'application/x-hdf5'))
    manifest=json.loads((tmp_path / 'manifest.json').read_text())
    manifest['history']['head']=put(tmp_path,'catalog.json',page,page['schema'])
    ref['manifest']=put(tmp_path,'manifest.json',manifest,manifest['schema'])
    original=h5py.Dataset.__getitem__
    def no_coordinates(dataset,key,**kw):
        assert dataset.name!='/records/offsets','oversized record chunk decoded'
        return original(dataset,key,**kw)
    monkeypatch.setattr(h5py.Dataset,'__getitem__',no_coordinates)
    with StateReader(tmp_path,state_ref=ref,expected_identity=IDENTITY,
                     limits=ResultReadLimits(maximum_record_bytes=4096)) as reader:
        reader.import_history()
        with pytest.raises(ContractError,match='chunk'):
            reader.history_batch()


def test_integrity_file_size_is_an_irreducible_physical_read_obligation(tmp_path):
    ref = array_source(tmp_path, 8192)
    with StateReader(tmp_path, state_ref=ref, expected_identity=IDENTITY) as reader:
        declaration = selected_array(reader)
        reader.limits = replace(reader.limits, maximum_io_bytes=128)
        with pytest.raises(ContractError) as error:
            reader.read_array(declaration, (slice(0, 1),))
        assert error.value.code == 'budget_exceeded'


@pytest.mark.parametrize('field,value', [('attempt','2'),('branch_id','other'),('model_sha256','c'*64),('plan_sha256','d'*64),('attempt',True)])
def test_full_expected_identity_is_required_even_when_manifest_bytes_valid(tmp_path, field, value):
    ref = source(tmp_path)
    identity = copy.deepcopy(IDENTITY)
    identity[field] = value
    with pytest.raises(ContractError):
        StateReader(tmp_path, state_ref=ref, expected_identity=identity)


def test_cached_open_file_detects_path_replacement(tmp_path):
    ref = array_source(tmp_path, 64)
    with StateReader(tmp_path, state_ref=ref, expected_identity=IDENTITY) as reader:
        declaration = selected_array(reader)
        reader.read_array(declaration, (slice(0, 1),))
        replacement = tmp_path / 'replacement.h5'
        replacement.write_bytes((tmp_path / 'arrays.h5').read_bytes())
        os.replace(replacement, tmp_path / 'arrays.h5')
        with pytest.raises(ContractError) as error:
            reader.read_array(declaration, (slice(0, 1),))
        assert error.value.code == 'source_changed'


def test_same_size_tamper_with_stale_timestamp_fails_new_hash_boundary(tmp_path):
    ref = array_source(tmp_path, 64)
    before = (tmp_path / 'arrays.h5').stat()
    with h5py.File(tmp_path / 'arrays.h5', 'r+') as f:
        f['arrays/a00000001'][0] = 123.0
    os.utime(tmp_path / 'arrays.h5', ns=(before.st_atime_ns, before.st_mtime_ns))
    assert (tmp_path / 'arrays.h5').stat().st_size == before.st_size
    with StateReader(tmp_path, state_ref=ref, expected_identity=IDENTITY) as reader:
        declaration = selected_array(reader)
        with pytest.raises(ContractError, match='checksum'):
            reader.read_array(declaration, (slice(0, 1),))


def test_same_open_source_mutation_cannot_hide_by_restoring_mtime(tmp_path):
    ref = array_source(tmp_path, 64)
    path = tmp_path / 'arrays.h5'
    with StateReader(tmp_path, state_ref=ref, expected_identity=IDENTITY) as reader:
        declaration = selected_array(reader)
        reader.read_array(declaration, (slice(0, 1),))
        before = path.stat()
        with path.open('r+b') as output:
            last = output.seek(-1, 2)
            byte = output.read(1)
            output.seek(last)
            output.write(bytes([byte[0] ^ 1]))
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        with pytest.raises(ContractError) as error:
            reader.read_array(declaration, (slice(0, 1),))
        assert error.value.code == 'source_changed'


@pytest.mark.parametrize('where', ['root', 'manifest'])
def test_symbolic_links_never_authorize_an_external_source(tmp_path, where):
    ref = source(tmp_path / 'owned')
    if where == 'root':
        (tmp_path / 'link').symlink_to(tmp_path / 'owned', target_is_directory=True)
        root = tmp_path / 'link'
    else:
        raw = (tmp_path / 'owned/manifest.json').read_bytes()
        (tmp_path / 'external.json').write_bytes(raw)
        (tmp_path / 'owned/manifest.json').unlink()
        (tmp_path / 'owned/manifest.json').symlink_to(tmp_path / 'external.json')
        root = tmp_path / 'owned'
    with pytest.raises(ContractError):
        StateReader(root, state_ref=ref, expected_identity=IDENTITY)


@pytest.mark.parametrize('selection,exception', [((), ValueError), ((True,),ValueError), ((-1,),IndexError), ((slice(None,None,0),),ValueError)])
def test_array_requires_valid_explicit_logical_axis_selection(tmp_path, selection, exception):
    ref = array_source(tmp_path, 3)
    with StateReader(tmp_path, state_ref=ref, expected_identity=IDENTITY) as reader:
        declaration = selected_array(reader)
        with pytest.raises(exception):
            reader.read_array(declaration, selection)


def test_retired_recovery_directory_api_is_not_a_format_fallback(tmp_path):
    with pytest.raises(ContractError) as error:
        verify_recovery_bundle(tmp_path)
    assert error.value.code == 'unsupported_format'


def _profile_context(root):
    """Explicit synthetic original input refs for the public whole-call reader."""
    import json
    from consumer_context_fixture import fixture
    selected, _, _, _, _ = fixture(root, plan_kind='scientific_plan_v2')
    manifest = json.loads((root / selected['manifest']['path']).read_bytes())
    refs = {item['role']: {key: item[key] for key in
            ('path', 'bytes', 'sha256', 'schema', 'media_type')}
            for item in manifest['artifacts'] if item['role'] in ('plan', 'model')}
    return dict(artifact_root=str(root), state_ref=selected,
                expected_identity=selected['identity'], plan_ref=refs['plan'],
                model_ref=refs['model'], diagnostics_ref=None, observations_ref=None,
                limits=ResultReadLimits(), maximum_context_bytes=128*1024*1024), manifest


def test_selected_contracts_profile_reads_original_context_with_full_shared_cache(tmp_path):
    """A qualified schema update must admit the real owner call, cold and warm."""
    from qcl_negf_contracts import results as contracts
    from qcl_negf_results.final_observations import describe_native_metadata_context
    args, manifest = _profile_context(tmp_path)
    # Public Results/telemetry callsite domain, independently enumerated here.
    kinds = ('checkpoint-declaration', 'core-state', 'final-diagnostics',
             'final-observations', 'nested-sweep-plan', 'producer-phase-segment',
             'producer-phase-source', 'result-common', 'result-cursor',
             'result-manifest', 'result-pointer', 'scientific-catalog',
             'scientific-plan', 'scientific-record', 'telemetry-envelope')
    for kind in kinds:
        contracts._validator(kind + '.schema.json')
    before = contracts._validator.cache_info()
    first = describe_native_metadata_context(**args)
    second = describe_native_metadata_context(**args)
    assert first[0] == second[0] == manifest
    assert first[2].plan_raw == second[2].plan_raw == (tmp_path / 'plan.json').read_bytes()
    assert first[2].model_raw == second[2].model_raw == (tmp_path / 'model.json').read_bytes()
    assert first[2].source == second[2].source == args['state_ref']
    assert first[3:] == second[3:] == (None, None)
    assert before.currsize == contracts._validator.cache_info().currsize == 15
    assert first[2].io_counters['metadata_high_water_bytes'] <= 128*1024*1024


@pytest.mark.parametrize('resource', ['model-help.schema.json', 'telemetry-envelope.schema.json'])
def test_changed_cached_resource_rejects_before_factory_or_artifact_access(tmp_path, monkeypatch, resource):
    """Qualifying new help content must not trust mutable bundled cache content."""
    import sys
    from qcl_negf_contracts import results as contracts, schema_validator
    from qcl_negf_results.final_observations import describe_native_metadata_context
    args, _ = _profile_context(tmp_path)
    validator = contracts._validator('result-manifest.schema.json')
    document = validator._registry['https://qcl-negf.invalid/schema/' + resource].contents
    original = copy.deepcopy(document)
    if resource == 'model-help.schema.json':
        document['$defs']['FragmentInput']['properties']['reference_value_origin'] = {
            '$ref': '#/$defs/ReferenceValueOrigin'}  # Superseded non-null branch.
    else:
        document['title'] = 'unqualified telemetry resource'
    def forbidden_root(reader):
        pytest.fail('artifact root accessed before profile rejection')
    def observe(frame, event, arg):
        if event == 'call' and frame.f_code is schema_validator.__code__:
            pytest.fail('schema factory called before profile rejection')
    monkeypatch.setattr(StateReader, '_root_identity', forbidden_root)
    previous = sys.getprofile()
    try:
        sys.setprofile(observe)
        with pytest.raises(ContractError, match='validation profile') as caught:
            describe_native_metadata_context(**args)
        assert caught.value.code == 'incompatible_contract'
    finally:
        sys.setprofile(previous)
        document.clear()
        document.update(original)


@pytest.mark.parametrize('relative', ['model_help.py', 'schemas/model-help.schema.json'])
def test_changed_contracts_package_source_rejects_before_factory_or_artifact_access(tmp_path, relative):
    """A disk-only source change must reject even with pristine loaded validators."""
    import json
    from pathlib import Path
    import shutil
    import subprocess
    import sys
    import qcl_negf_contracts
    import qcl_negf_results
    args, _ = _profile_context(tmp_path / 'artifacts')
    package = tmp_path / 'source' / 'qcl_negf_contracts'
    shutil.copytree(Path(qcl_negf_contracts.__file__).parent, package,
                    ignore=shutil.ignore_patterns('__pycache__'))
    path = package / relative
    raw = path.read_bytes()
    before, after = (b'Public', b'public') if path.suffix == '.py' else (b'Model help', b'model help')
    changed = raw.replace(before, after, 1)
    assert changed != raw and len(changed) == len(raw)
    path.write_bytes(changed)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(package.parent),
        str(Path(qcl_negf_results.__file__).parent.parent))))
    script = '''
import json,sys
from qcl_negf_contracts import schema_validator
from qcl_negf_contracts.messages import ContractError
from qcl_negf_results.state import StateReader, ResultReadLimits
from qcl_negf_results.final_observations import describe_native_metadata_context
def root(reader):
    raise AssertionError('artifact access before profile rejection')
def observe(frame,event,arg):
    if event=='call' and frame.f_code is schema_validator.__code__:
        raise AssertionError('factory called before profile rejection')
StateReader._root_identity=root
sys.setprofile(observe)
try:
    args=json.loads(sys.argv[1])
    args['limits']=ResultReadLimits(**args['limits'])
    describe_native_metadata_context(**args)
except ContractError as error:
    assert error.code=='incompatible_contract' and 'validation profile' in str(error)
    print('REJECTED_BEFORE_EFFECT')
else:
    raise AssertionError('changed package admitted')
finally:
    sys.setprofile(None)
'''
    result = subprocess.run([sys.executable, '-c', script, json.dumps(args, default=lambda value: value.__dict__)], env=env,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == 'REJECTED_BEFORE_EFFECT'
