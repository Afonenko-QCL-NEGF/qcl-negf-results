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
    equilibrium = lambda path, **kw: diagnose_stored_equilibrium(path, limits=EquilibriumReadLimits(maximum_io_bytes=1024*1024), **kw)
    for create in (AnalysisReader, DiagnosticReader, equilibrium):
        with pytest.raises(ContractError) as error:
            create(tmp_path / 'not-a-native4-layout', expected_identity=IDENTITY)
        assert error.value.code == 'capability_unavailable'


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
