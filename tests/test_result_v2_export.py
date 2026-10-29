import pytest

from qcl_negf_contracts.messages import ContractError
from qcl_negf_results.state import StateReader
from qcl_negf_results import live_views
from result_v2_fixture import IDENTITY, record, source, producer_root, producer_state


def test_explicit_scalar_reduction_preserves_negative_current_and_units(tmp_path):
    ref = source(tmp_path, [record(1, 7), record(2, 99, 'c010000000000000')])
    assert hasattr(live_views, 'ScalarReduction'), 'missing native5 explicit streaming derive'
    with StateReader(tmp_path, state_ref=ref, expected_identity=IDENTITY) as reader:
        reader.import_history()
        request = live_views.ScalarReduction('scba', 'J', '1', '2')
        result = live_views.derive_history(reader, request, maximum_io_bytes=1024*1024)
        assert result['minimum']['value']['bits'] == 'c010000000000000'
        assert result['maximum']['value']['bits'] == '4008000000000000'
        assert result['units'] == 'A/m^2'
        assert result['finite_count'] == 2 and result['observations'] == 2


@pytest.mark.parametrize('stage,field', [('scba','not_a_field'), ('outer','density'), ('scba','ν')])
def test_scalar_reduction_rejects_unknown_or_nonscalar_layout_field(stage, field):
    with pytest.raises(ValueError):
        live_views.ScalarReduction(stage,field,'1','2')


def test_full_state_export_roundtrip_and_science_omissions(producer_root, tmp_path):
    from qcl_negf_results.export import preview, export_snapshot
    from qcl_negf_results.archive import receive
    ref = producer_state(producer_root, 'exact', recovery=True)
    with StateReader(producer_root, state_ref=ref, expected_identity=ref['identity']) as reader:
        science = preview(reader, profile='science')
        assert not science['capabilities']['exact_restore']['available']
        assert science['omitted_by_policy']
        receipt = export_snapshot(reader, tmp_path / 'exports', profile='full-state',
                                  byte_budget=32*1024*1024, reserve_bytes=0)
    restored = receive([tmp_path / 'exports' / receipt['archive']], tmp_path / 'restored', receipt=receipt)
    assert restored['source_state'] == ref
    with StateReader(restored['source_root'], state_ref=ref, expected_identity=ref['identity']) as reader:
        assert len(reader.checkpoint()['state']['fields']) == 17


def test_native5_domain_views_report_capability_without_opening_old_layout(tmp_path):
    from qcl_negf_results.saved_views import AnalysisReader
    from qcl_negf_results.diagnostic_views import DiagnosticReader
    from qcl_negf_results.equilibrium import diagnose_stored_equilibrium, EquilibriumReadLimits
    from qcl_negf_results.state import StateReader
    from consumer_context_fixture import semantic_fixture
    root=tmp_path/'source'
    selected,artifact,_,_,_=semantic_fixture(root)
    with StateReader(root,state_ref=selected,expected_identity=selected['identity']) as reader:
        with AnalysisReader(reader,artifact_ref=artifact) as analysis:
            assert analysis.describe()['source_state']==selected
            assert analysis.read('density',[slice(None)])['values']==[0.,-2.,8.]
        assert reader._handles and not reader._closed
    with pytest.raises(TypeError):
        AnalysisReader(tmp_path/'not-a-native4-layout',expected_identity=IDENTITY)
    with pytest.raises(TypeError,match='StateReader'):
        AnalysisReader(tmp_path/'not-a-native4-layout',artifact_ref=artifact)
    equilibrium = lambda owner, **kw: diagnose_stored_equilibrium(owner, observations_ref={}, limits=EquilibriumReadLimits(maximum_io_bytes=1024*1024), **kw)
    with pytest.raises(TypeError, match='StateReader'):
        equilibrium(tmp_path / 'not-a-native4-layout')
    with StateReader(root, state_ref=selected, expected_identity=selected['identity']) as reader:
        with pytest.raises(ContractError):
            equilibrium(reader)
    with pytest.raises(TypeError,match='StateReader'):
        DiagnosticReader(tmp_path / 'not-a-native4-layout')


def test_cli_requires_explicit_identity_and_state_ref(tmp_path, monkeypatch, capsys):
    import json
    from qcl_negf_results.cli import main
    from result_v2_fixture import encoded
    ref = source(tmp_path / 'source', [record()])
    (tmp_path / 'identity.json').write_bytes(encoded(IDENTITY))
    (tmp_path / 'state.json').write_bytes(encoded(ref))
    monkeypatch.setattr('sys.argv', ['qcl-negf-results', 'preview', str(tmp_path / 'source'),
        '--identity', str(tmp_path / 'identity.json'), '--state-ref', str(tmp_path / 'state.json'),
        '--profile', 'science'])
    main()
    result = json.loads(capsys.readouterr().out)
    assert result['source'] == ref
    assert result['scientific_accepted'] is None


def test_failed_directory_sync_rolls_back_publication_pair(tmp_path, monkeypatch):
    from qcl_negf_results import export
    ref = source(tmp_path / 'source', [record()])
    def fail(path):
        raise OSError('manufactured directory sync failure')
    monkeypatch.setattr(export, 'fsync_directory', fail)
    with StateReader(tmp_path / 'source', state_ref=ref, expected_identity=IDENTITY) as reader:
        with pytest.raises(OSError, match='sync failure'):
            export.export_snapshot(reader, tmp_path / 'exports', profile='science', byte_budget=4*1024*1024, reserve_bytes=0)
    assert list((tmp_path / 'exports').glob('*.tar.xz')) == []
    assert list((tmp_path / 'exports').glob('*.json')) == []


def test_export_selected_ancestry_and_capture_share_one_source_allowance(producer_root, tmp_path, monkeypatch):
    from qcl_negf_results import state
    from qcl_negf_results.export import export_snapshot
    ref = producer_state(producer_root, 'grandchild')
    with StateReader(producer_root, state_ref=ref, expected_identity=ref['identity']) as reader:
        account, deliveries = state._IOBudget.account, []
        def counted(budget, count):
            deliveries.append(count)
            account(budget, count)
        monkeypatch.setattr(state._IOBudget, 'account', counted)
        export_snapshot(reader, tmp_path / 'export', profile='science',
            byte_budget=16*1024*1024, reserve_bytes=0)
        assert reader.io_counters['bytes_read_total'] == sum(deliveries)


def test_receive_verification_preserves_declared_selected_capabilities_without_restore(tmp_path):
    from qcl_negf_results.archive import receive
    from result_v2_fixture import export_source
    ref = source(tmp_path / 'source', [record()])
    receipt = export_source(tmp_path / 'source',ref,tmp_path / 'export')
    result = receive([tmp_path / 'export' / receipt['archive']],receipt=receipt)
    assert 'raw_scientific_history' in result['capabilities']
    assert result['source_root'] is None and result['source_state'] is None


def test_full_state_receive_rejects_extra_unrelated_original_path(producer_root, tmp_path):
    import json
    from qcl_negf_results.archive import receive
    from result_v2_fixture import export_source, encoded
    from test_single_archive import rewrite_archive
    ref = producer_state(producer_root,'exact',recovery=True)
    receipt = export_source(producer_root,ref,tmp_path / 'export',profile='full-state')
    changed = tmp_path / 'extra.tar.xz'
    def extra(payloads):
        value = json.loads(payloads['manifest.json'])
        row = json.loads(json.dumps(value['source_locator'][0]))
        row['reference']['path'] = 'unrelated/orphan.json'
        value['source_locator'].append(row)
        payloads['manifest.json'] = encoded(value)
    rewrite_archive(tmp_path / 'export' / receipt['archive'],changed,extra)
    with pytest.raises(ContractError,match='unrelated|inventory'):
        receive([changed],tmp_path / 'restored')
    assert not (tmp_path / 'restored').exists()


def test_science_includes_declared_full_owner_only_when_selected_leaf_depends_on_it(producer_root, tmp_path):
    import json
    from result_v2_fixture import put, export_source, copy_constructor_closure
    from test_export import unpack
    ref=producer_state(producer_root,'exact',recovery=True)
    root=copy_constructor_closure(producer_root,tmp_path / 'source',ref)
    manifest=json.loads((root / ref['manifest']['path']).read_text())
    optional=next(row for row in manifest['artifacts'] if row['role']=='recovery'
        and row['media_type']=='application/x-hdf5')
    leaf=next(row for row in manifest['artifacts'] if row['role']=='science.history')
    # A fresh policy constructor augments the canonical dependency graph using
    # actual native5 bytes; it does not claim P04 emitted this additional edge.
    for depends in (False,True):
        leaf['dependencies']=[optional['path']] if depends else []
        ref['manifest']=put(root,('dependent' if depends else 'independent')+'.json',manifest,manifest['schema'])
        receipt=export_source(root,ref,tmp_path / ('with' if depends else 'without'))
        selected,_=unpack(receipt,tmp_path / ('with' if depends else 'without'))
        included={row['source_path'] for row in selected['records'][0]['included']}
        omitted={row['source_path'] for row in selected['omitted_by_policy']}
        assert (optional['path'] in included) is depends
        assert (optional['path'] in omitted) is (not depends)
        assert not selected['canonical_source_complete']


def test_explicit_analysis_science_closure_and_provenance_without_history(tmp_path):
    import json
    import tarfile
    from consumer_context_fixture import fixture
    from qcl_negf_results.export import preview, export_snapshot
    from qcl_negf_results.archive import receive
    root = tmp_path / 'source'
    selected, observations, value, _, _ = fixture(root)
    with StateReader(root, state_ref=selected, expected_identity=selected['identity']) as reader:
        with pytest.raises(ContractError, match='history|analysis'):
            preview(reader, profile='science')
        info = preview(reader, profile='science', observations_ref=observations,
                       diagnostics_ref=value['diagnostics'])
        assert info['integrity_scope'] == 'selected_analysis_metadata_verified'
        assert info['capabilities']['stored_observables']['available']
        assert info['capabilities']['stored_final_diagnostics']['available']
        assert not info['canonical_source_complete'] and info['scientific_accepted'] is None
        receipt = export_snapshot(reader, tmp_path / 'export', profile='science',
            observations_ref=observations, diagnostics_ref=value['diagnostics'],
            byte_budget=16*1024*1024, reserve_bytes=0)
    with tarfile.open(tmp_path / 'export' / receipt['archive'], 'r:xz') as archive:
        manifest = json.load(archive.extractfile('manifest.json'))
        actual = {row['reference']['path']: row['reference'] for row in manifest['source_locator']}
        assert set(actual) == {'selected.json', 'checkpoint.json', 'full.h5', 'analysis.h5',
                               'observations.h5', 'plan.yaml', 'model.json'}
        assert info['bytes'] == sum(int(row['bytes']) for row in actual.values())
        for reference in manifest['source_locator']:
            assert archive.extractfile(reference['object']).read() == (root / reference['reference']['path']).read_bytes()
        assert not manifest['complete'] and not manifest['canonical_source_complete']
        assert manifest['records'][0]['native_provenance']['coordinates']['temperature_K']['status'] == 'verified'
        assert manifest['coverage']['stored_observables'] == 'selected_metadata_verified'
    result = receive([tmp_path / 'export' / receipt['archive']], receipt=receipt)
    assert result['source_root'] is None


def test_explicit_diagnostics_selects_its_closure_and_omits_unselected_observations(tmp_path):
    from consumer_context_fixture import fixture
    from qcl_negf_results.export import preview
    selected, _, value, _, _ = fixture(tmp_path)
    with StateReader(tmp_path, state_ref=selected, expected_identity=selected['identity']) as reader:
        info = preview(reader, profile='science', diagnostics_ref=value['diagnostics'])
        assert info['files'] == 4
        assert {v['source_path'] for v in info['omitted_by_policy']} == {'observations.h5', 'plan.yaml', 'model.json'}
        assert info['capabilities']['stored_final_diagnostics']['available']
        assert not info['capabilities']['stored_observables']['available']


@pytest.mark.parametrize('selector,ref_kind', [('observations_ref','diagnostics'),
    ('diagnostics_ref','observations'), ('observations_ref','tampered'), ('diagnostics_ref','alias')])
def test_explicit_analysis_rejects_wrong_representation_or_nonexact_ref_before_publication(tmp_path, selector, ref_kind):
    from consumer_context_fixture import fixture
    from qcl_negf_results.export import export_snapshot
    selected, observations, value, _, _ = fixture(tmp_path / 'source')
    reference = dict(value['diagnostics'] if ref_kind == 'diagnostics' else observations)
    if ref_kind == 'tampered': reference['sha256'] = 'f'*64
    if ref_kind == 'alias': reference['extra'] = 'not exact'
    with StateReader(tmp_path / 'source', state_ref=selected, expected_identity=selected['identity']) as reader:
        with pytest.raises(ContractError):
            export_snapshot(reader, tmp_path / 'export', profile='science',
                byte_budget=16*1024*1024, reserve_bytes=0, **{selector:reference})
    assert not list((tmp_path / 'export').glob('*.tar.xz'))
    assert not list((tmp_path / 'export').glob('*.json'))


def test_both_analysis_selectors_must_select_the_exact_observations_diagnostic(tmp_path):
    from consumer_context_fixture import fixture
    from result_v2_fixture import put, reference
    import json
    import shutil
    from qcl_negf_results.export import preview
    selected, observations, value, _, _ = fixture(tmp_path)
    shutil.copyfile(tmp_path / 'analysis.h5', tmp_path / 'other-diagnostics.h5')
    other = reference(tmp_path, 'other-diagnostics.h5', value['diagnostics']['schema'], 'application/x-hdf5')
    manifest = json.loads((tmp_path / selected['manifest']['path']).read_bytes())
    manifest['artifacts'].append(dict(other,role='physics.analysis',profile='science',dependencies=['checkpoint.json']))
    selected['manifest'] = put(tmp_path, 'selected.json', manifest, manifest['schema'])
    with StateReader(tmp_path, state_ref=selected, expected_identity=selected['identity']) as reader:
        with pytest.raises(ContractError, match='diagnostic.*differ|diagnostic.*match'):
            preview(reader, profile='science', observations_ref=observations, diagnostics_ref=other)


def test_analysis_export_admits_actual_closure_before_copy_and_rejects_source_change(tmp_path, monkeypatch):
    from consumer_context_fixture import fixture
    from qcl_negf_results import export
    selected, observations, _, _, _ = fixture(tmp_path / 'source')
    with StateReader(tmp_path / 'source', state_ref=selected, expected_identity=selected['identity']) as reader:
        with monkeypatch.context() as no_copy:
            no_copy.setattr(export, 'copy_closed', lambda *a, **kw: pytest.fail('copy before closure admission'))
            with pytest.raises(ContractError, match='budget|exceed'):
                export.export_snapshot(reader, tmp_path / 'small', profile='science', observations_ref=observations,
                    byte_budget=1, reserve_bytes=0)
        def change(event):
            if event['phase'] == 'capturing':
                target = tmp_path / 'source' / observations['path']
                data = bytearray(target.read_bytes()); data[-1] ^= 1; target.write_bytes(data)
        with pytest.raises(ContractError):
            export.export_snapshot(reader, tmp_path / 'changed', profile='science', observations_ref=observations,
                byte_budget=16*1024*1024, reserve_bytes=0, progress=change)
    assert not list((tmp_path / 'changed').glob('*.tar.xz'))
    assert not list((tmp_path / 'changed').glob('*.json'))


def test_full_state_inventory_does_not_qualify_analysis_and_selection_changes_snapshot_identity(tmp_path):
    import json
    import tarfile
    from test_native_provenance import _fixture
    from qcl_negf_results.export import preview, export_snapshot
    selected, observations, _, _, _ = _fixture(tmp_path / 'source')
    with StateReader(tmp_path / 'source', state_ref=selected, expected_identity=selected['identity']) as reader:
        plain = preview(reader, profile='full-state')
        assert plain['integrity_scope'] == 'declared_metadata_availability'
        assert not plain['capabilities']['stored_final_diagnostics']['available']
        kwargs = dict(profile='full-state', byte_budget=32*1024*1024, reserve_bytes=0)
        raw = export_snapshot(reader, tmp_path / 'raw', **kwargs)
        chosen = export_snapshot(reader, tmp_path / 'chosen', observations_ref=observations, **kwargs)
        assert raw['snapshot_identity'] != chosen['snapshot_identity']
        with tarfile.open(tmp_path / 'raw' / raw['archive'], 'r:xz') as archive:
            manifest = json.load(archive.extractfile('manifest.json'))
            assert manifest['canonical_source_complete']
            assert manifest['coverage']['stored_final_diagnostics'] == 'explicit_final_diagnostics_ref_required'
            assert 'selected_final_diagnostics_metadata' not in manifest['capabilities']
        assert not reader.source.capabilities['stored_final_diagnostics']['available']


def test_selected_analysis_metadata_does_not_read_scientific_arrays(tmp_path, monkeypatch):
    import h5py
    from consumer_context_fixture import fixture
    from qcl_negf_results.export import preview
    from qcl_negf_results.provenance import verify_native_provenance
    selected, observations, value, _, _ = fixture(tmp_path)
    original=h5py.Dataset.__getitem__
    def metadata_only(dataset, key):
        assert '/arrays/' not in dataset.name, 'metadata projection read a scientific array'
        return original(dataset,key)
    with StateReader(tmp_path,state_ref=selected,expected_identity=selected['identity']) as reader:
        monkeypatch.setattr(h5py.Dataset,'__getitem__',metadata_only)
        info=preview(reader,profile='science',observations_ref=observations,diagnostics_ref=value['diagnostics'])
        proof=verify_native_provenance(reader,observations_ref=observations)
        assert info['scientific_accepted'] is None and proof['current_evidence']['scientific_accepted'] is None


@pytest.mark.parametrize('field,value', [('path','other.h5'),('bytes','12'),('sha256','0'*64),
    ('media_type','text/plain'),('schema','qcl-negf-physics-v5')])
def test_selected_analysis_complete_fileref_required(tmp_path, field, value):
    from consumer_context_fixture import fixture
    from qcl_negf_results.export import preview
    selected, observations, _, _, _=fixture(tmp_path)
    observations[field]=value
    with StateReader(tmp_path,state_ref=selected,expected_identity=selected['identity']) as reader:
        with pytest.raises(ContractError):preview(reader,profile='science',observations_ref=observations)


@pytest.mark.parametrize('cancel', [False, True])
def test_analysis_export_uses_one_owner_allowance_and_restores_context_on_cancellation(tmp_path, monkeypatch, cancel):
    from test_native_provenance import _completed_fixture
    from qcl_negf_results import state
    from qcl_negf_results.export import export_snapshot
    selected,observations,_,_,_,_= _completed_fixture(tmp_path/'source')
    with StateReader(tmp_path/'source',state_ref=selected,expected_identity=selected['identity']) as reader:
        account=state._IOBudget.account;deliveries=[];budgets=set()
        def counted(budget,count):
            budgets.add(id(budget));deliveries.append(count);account(budget,count)
        monkeypatch.setattr(state._IOBudget,'account',counted)
        def progress(event):
            if cancel and event['phase']=='capturing':raise KeyboardInterrupt('literal cancellation')
        def operation():return export_snapshot(reader,tmp_path/'export',profile='science',observations_ref=observations,
            byte_budget=32*1024*1024,reserve_bytes=0,progress=progress)
        if cancel:
            with pytest.raises(KeyboardInterrupt,match='literal cancellation'):operation()
            assert not list((tmp_path/'export').glob('*.tar.xz')) and not list((tmp_path/'export').glob('*.json'))
            assert not list((tmp_path/'export').glob('.export-pin-*'))
        else:operation()
        assert len(budgets)==1 and reader.io_counters['bytes_read_total']==sum(deliveries)
        assert reader._shared_budget is None and not reader._closed


def test_analysis_export_operation_allowance_includes_capture_after_metadata(tmp_path):
    from consumer_context_fixture import fixture
    from qcl_negf_results.state import ResultReadLimits
    from qcl_negf_results.export import export_snapshot
    selected,observations,_,_,_=fixture(tmp_path/'source')
    kwargs=dict(profile='science',observations_ref=observations,byte_budget=32*1024*1024,reserve_bytes=0)
    with StateReader(tmp_path/'source',state_ref=selected,expected_identity=selected['identity']) as reader:
        export_snapshot(reader,tmp_path/'ok',**kwargs)
        required=reader.io_counters['bytes_read_total']
    with StateReader(tmp_path/'source',state_ref=selected,expected_identity=selected['identity'],
                     limits=ResultReadLimits(maximum_io_bytes=required-1)) as reader:
        with pytest.raises(ContractError,match='budget|allowance|limit'):
            export_snapshot(reader,tmp_path/'short',**kwargs)
        assert reader.io_counters['bytes_read_total']<=required-1
    assert not list((tmp_path/'short').glob('*.json')) and not list((tmp_path/'short').glob('*.tar.xz'))


def test_selected_completed_analysis_preserves_parent_proof_without_optional_parent_payload(tmp_path):
    import json
    import tarfile
    from test_native_provenance import _completed_fixture
    from qcl_negf_results.export import export_snapshot
    selected,observations,_,_,_,parent=_completed_fixture(tmp_path/'source')
    with StateReader(tmp_path/'source',state_ref=selected,expected_identity=selected['identity']) as reader:
        receipt=export_snapshot(reader,tmp_path/'export',profile='science',observations_ref=observations,
            byte_budget=32*1024*1024,reserve_bytes=0)
    with tarfile.open(tmp_path/'export'/receipt['archive'],'r:xz') as archive:
        manifest=json.load(archive.extractfile('manifest.json'))
    included={row['reference']['path'] for row in manifest['source_locator']}
    omitted={row['source_path'] for row in manifest['omitted_by_policy']}
    assert parent['manifest']['path'] in included and 'checkpoint.json' in included
    assert 'child-full.h5' in included and 'full.h5' in omitted and 'full.h5' not in included
    assert 'child-observations.h5' in included and 'observations.h5' in omitted
    assert manifest['records'][0]['native_provenance']['restoration']['parent_state']==parent


def test_full_source_verification_reuses_selected_owner(tmp_path, monkeypatch):
    from test_native_provenance import _fixture
    from qcl_negf_results.export import export_snapshot
    selected,observations,_,_,_=_fixture(tmp_path/'source')
    with StateReader(tmp_path/'source',state_ref=selected,expected_identity=selected['identity']) as reader:
        def no_reopen(*args, **kwargs):pytest.fail('untracked duplicate selected StateReader')
        monkeypatch.setattr(StateReader,'_initialize',no_reopen)
        export_snapshot(reader,tmp_path/'export',profile='full-state',observations_ref=observations,
            byte_budget=32*1024*1024,reserve_bytes=0)


@pytest.mark.parametrize('fault', ['same_size','replacement','missing'])
def test_source_change_after_final_copy_is_refused_before_publication(tmp_path, fault):
    from qcl_negf_results.export import export_snapshot
    selected=source(tmp_path/'source',[record()]);target=tmp_path/'source'/selected['manifest']['path']
    copies=[]
    def progress(event):
        if event['phase']=='capturing':
            copies.append(event['source_path'])
            if len(copies)==3:
                if fault=='missing':target.unlink()
                elif fault=='replacement':
                    substitute=target.with_name('replacement.json');substitute.write_bytes(target.read_bytes());substitute.replace(target)
                else:
                    raw=bytearray(target.read_bytes());raw[-1]^=1;target.write_bytes(raw)
    with StateReader(tmp_path/'source',state_ref=selected,expected_identity=selected['identity']) as reader:
        with pytest.raises(ContractError) as error:
            export_snapshot(reader,tmp_path/'export',profile='science',byte_budget=4*1024*1024,reserve_bytes=0,progress=progress)
        assert error.value.code=='source_changed'
    assert len(copies)==3
    assert not list((tmp_path/'export').glob('*.tar.xz')) and not list((tmp_path/'export').glob('*.json'))
    assert not list((tmp_path/'export').glob('.export-pin-*'))


@pytest.mark.parametrize('selection', ['full_default','science_diagnostics','science_observations'])
def test_export_bytes_and_provenance_do_not_depend_on_unselected_reader_cache(tmp_path, selection):
    import json
    import tarfile
    from test_native_provenance import _fixture
    from qcl_negf_results.final_observations import FinalObservationsReader
    from qcl_negf_results.export import export_snapshot
    selected,observations,value,_,_=_fixture(tmp_path/'source');receipts=[]
    selectors=({'diagnostics_ref':value['diagnostics']} if selection=='science_diagnostics' else
               {'observations_ref':observations} if selection=='science_observations' else {})
    for warm in (False,True):
        with StateReader(tmp_path/'source',state_ref=selected,expected_identity=selected['identity']) as reader:
            if warm:
                with FinalObservationsReader(reader,artifact_ref=observations) as view:view.describe()
            output=tmp_path/str(warm)
            receipt=export_snapshot(reader,output,profile='full-state' if selection=='full_default' else 'science',
                byte_budget=32*1024*1024,reserve_bytes=0,**selectors)
            with tarfile.open(output/receipt['archive'],'r:xz') as archive:
                manifest=json.load(archive.extractfile('manifest.json'))
            proof=manifest['records'][0]['native_provenance']
            expected=sorted([row['reference'] for row in manifest['source_locator']
                if row['reference']['media_type']=='application/x-hdf5'],key=lambda row:row['path'])
            assert proof['native_verified_references']==expected
            assert proof['native_verified_references_scope']=='captured_native_bytes_and_headers'
            if selection=='science_diagnostics':
                assert observations not in proof['native_verified_references']
                assert proof['coordinates']['reason']=='explicit_final_observations_ref_required'
            elif selection=='science_observations':
                assert proof['request']==value['request'] and proof['current_evidence']['scientific_accepted'] is None
            else:
                assert manifest['coverage']['stored_final_diagnostics']=='explicit_final_diagnostics_ref_required'
            receipts.append(receipt)
    assert receipts[0]['snapshot_identity']==receipts[1]['snapshot_identity']
    assert receipts[0]['sha256']==receipts[1]['sha256']
    assert (tmp_path/'False'/receipts[0]['archive']).read_bytes()==(tmp_path/'True'/receipts[1]['archive']).read_bytes()


@pytest.mark.parametrize('selector', ['observations_ref','diagnostics_ref'])
def test_wrong_exact_ref_is_not_satisfied_by_warm_analysis_cache(tmp_path, selector):
    from consumer_context_fixture import fixture
    from qcl_negf_results.final_observations import FinalObservationsReader
    from qcl_negf_results.export import export_snapshot
    selected,observations,value,_,_=fixture(tmp_path/'source')
    with StateReader(tmp_path/'source',state_ref=selected,expected_identity=selected['identity']) as reader:
        with FinalObservationsReader(reader,artifact_ref=observations) as view:view.describe()
        wrong=dict(observations if selector=='observations_ref' else value['diagnostics']);wrong['sha256']='0'*64
        with pytest.raises(ContractError):
            export_snapshot(reader,tmp_path/'export',profile='science',byte_budget=32*1024*1024,reserve_bytes=0,**{selector:wrong})
    assert not list((tmp_path/'export').glob('*.tar.xz')) and not list((tmp_path/'export').glob('*.json'))
