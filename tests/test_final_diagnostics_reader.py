"""Independent tiny literal native5 observations, not a scientific pass oracle."""
from copy import deepcopy
import importlib
import importlib.util
import json
import math

import h5py
import numpy as np
import pytest

from qcl_negf_contracts.messages import ContractError
from qcl_negf_contracts.results import validate_final_diagnostics
from qcl_negf_results.state import StateReader, ResultReadLimits
from final_diagnostics_fixture import fixture, wire_values, refresh, change_declaration, fixed_json, SCALARS


def api():
    assert importlib.util.find_spec('qcl_negf_results.final_diagnostics') is not None, 'stored final diagnostics reader missing'
    return importlib.import_module('qcl_negf_results.final_diagnostics')


def test_literal_full_fields_are_bound_detached_and_not_promoted(tmp_path,monkeypatch):
    ref,artifact,value,checkpoint=fixture(tmp_path)
    original=h5py.Dataset.__getitem__
    def no_numeric(dataset,key):
        assert '/arrays/' not in dataset.name, 'describe read numeric payload'
        return original(dataset,key)
    with StateReader(tmp_path,state_ref=ref,expected_identity=ref['identity']) as reader:
        monkeypatch.setattr(h5py.Dataset,'__getitem__',no_numeric)
        view=api().FinalDiagnosticsReader(reader,artifact_ref=artifact)
        description=view.describe()
        assert description.declaration==value
        assert description.artifact==artifact and description.checkpoint==value['checkpoint']
        assert description.source==ref
        assert description.declaration['solution_metrics']['summary_metrics'][0]['value']['bits']=='7ff8000000000001'
        assert reader.source.status['process']['state']=='failed'
        assert not reader.source.status['physics']['state']=='passed'
        assert {item['path'] for item in reader.source.integrity['selected_payloads_verified']}=={'analysis.h5'}
        description.declaration['dimensions']['energy']='999'
        assert view.describe().declaration==value
        view.close(); view.close()
        assert reader.source.state==ref


def test_four_literal_arrays_logical_axes_no_transpose_and_bounded_selection(tmp_path):
    ref,artifact,_,_=fixture(tmp_path)
    with StateReader(tmp_path,state_ref=ref,expected_identity=ref['identity']) as reader:
        with api().FinalDiagnosticsReader(reader,artifact_ref=artifact) as view:
            for name,expected in wire_values().items():
                description=view.describe_integral(name)
                assert description.logical_shape==description.stored_shape==expected.shape
                block=view.read_integral(name,tuple(slice(None) for _ in expected.shape))
                assert block.values.tobytes()==expected.tobytes()
                assert not block.values.flags.writeable
            name='spectral_integral'
            assert view.read_integral(name,(1,0,1)).values==11+12j
            assert view.read_integral(name,(slice(0,2,2),slice(None),slice(1,2))).values.tolist()==[[[3-4j],[7-8j]]]
            assert view.read_integral(name,(slice(0,0),slice(None),slice(None))).values.size==0
            with pytest.raises(ContractError,match='buffer|budget|limit'):
                view.read_integral(name,(slice(None),)*3,maximum_bytes=16)


@pytest.mark.parametrize('bits',['0000000000000000','8000000000000000','7ff0000000000000',
                               'fff0000000000000','7ff8000000001234','7ff0000000000001'])
def test_all_ieee_bits_raw_scalar_and_dataset_survive(tmp_path,bits):
    arrays=wire_values(); arrays['spectral_trace_cumulative'].view('<u8')[0,0]=int(bits,16)
    ref,artifact,_,_=fixture(tmp_path,values=arrays)
    def changed(value):
        w=value['positivity']['data']['witnesses'][0]
        w.update(status='error',matrix_kind='none',energy_index={'kind':'integer','type':'Int64','decimal':'0'},
                 momentum_index={'kind':'integer','type':'Int64','decimal':'0'})
        for name in SCALARS: w[name]['bits']=bits
    change_declaration(tmp_path,ref,artifact,changed)
    with StateReader(tmp_path,state_ref=ref,expected_identity=ref['identity']) as reader:
        view=api().FinalDiagnosticsReader(reader,artifact_ref=artifact)
        assert all(view.describe().declaration['positivity']['data']['witnesses'][0][name]['bits']==bits for name in SCALARS)
        block=view.read_integral('spectral_trace_cumulative',(slice(0,1),slice(0,1)))
        assert int(block.values.view('<u8')[0,0])==int(bits,16)


@pytest.mark.parametrize('status',['error','not_measured'])
def test_unavailable_scan_is_exact_reason_without_fake_arrays(tmp_path,status):
    ref,artifact,value,_=fixture(tmp_path,status=status)
    with StateReader(tmp_path,state_ref=ref,expected_identity=ref['identity']) as reader:
        view=api().FinalDiagnosticsReader(reader,artifact_ref=artifact)
        assert view.describe().declaration==value
        with pytest.raises(ContractError,match='literal missing scan'):
            view.read_integral('spectral_integral',(slice(None),)*3)


@pytest.mark.parametrize('field,value',[('sha256','f'*64),('bytes','123'),('schema','qcl-negf-physics-v5')])
def test_selected_analysis_exact_ref_not_only_path(tmp_path,field,value):
    ref,artifact,_,_=fixture(tmp_path); artifact[field]=value
    with StateReader(tmp_path,state_ref=ref,expected_identity=ref['identity']) as reader:
        with pytest.raises(ContractError): api().FinalDiagnosticsReader(reader,artifact_ref=artifact).describe()


@pytest.mark.parametrize('field',['identity','invocation','state_id','checkpoint','dimensions','counts','units','axes','order'])
def test_declaration_context_and_shape_counts_cannot_cross_stitch(tmp_path,field):
    ref,artifact,_,_=fixture(tmp_path)
    def changed(v):
        if field=='identity': v['identity']['attempt']='foreign'
        elif field=='invocation': v['invocation']['source_epoch']='foreign'
        elif field=='state_id': v['state_id']='foreign'
        elif field=='checkpoint': v['checkpoint']['sha256']='f'*64
        elif field=='dimensions': v['dimensions']['basis']='3'
        elif field=='counts': v['positivity']['data']['integrals']['total_blocks_per_kind']['decimal']='7'
        else:
            d=v['positivity']['data']['integrals']['spectral_integral']
            d[field]={'units':'eV','axes':['a','k','b'],'order':'C'}[field]
    change_declaration(tmp_path,ref,artifact,changed)
    with StateReader(tmp_path,state_ref=ref,expected_identity=ref['identity']) as reader:
        with pytest.raises(ContractError): api().FinalDiagnosticsReader(reader,artifact_ref=artifact).describe()


@pytest.mark.parametrize('kind',['vlen','ascii','vector','null','compound','invalid_utf8','duplicate','bare_nan'])
def test_fixed_utf8_scalar_fence_before_high_level_value_read(tmp_path,monkeypatch,kind):
    ref,artifact,value,_=fixture(tmp_path)
    with h5py.File(tmp_path/'analysis.h5','r+') as f:
        del f['final_diagnostics/declaration_json']
        name='final_diagnostics/declaration_json'
        if kind=='vlen': f.create_dataset(name,data=json.dumps(value),dtype=h5py.string_dtype())
        elif kind=='ascii': fixed_json(f,name,value,cset='ascii')
        elif kind=='vector': f.create_dataset(name,data=[b'{}'])
        elif kind=='null': f.create_dataset(name,dtype='S4')
        elif kind=='compound': f.create_dataset(name,shape=(),dtype=[('x',h5py.string_dtype())])
        else: fixed_json(f,name,{'invalid_utf8':b'\xff','duplicate':b'{"x":1,"x":2}','bare_nan':b'{"x":NaN}'}[kind])
    refresh(tmp_path,ref,artifact)
    original=h5py.Dataset.__getitem__
    def guarded(dataset,key):
        if kind in ('vlen','ascii','vector','null','compound'):
            assert dataset.name!='/final_diagnostics/declaration_json', 'unsafe scalar read'
        return original(dataset,key)
    monkeypatch.setattr(h5py.Dataset,'__getitem__',guarded)
    with StateReader(tmp_path,state_ref=ref,expected_identity=ref['identity']) as reader:
        with pytest.raises(ContractError): api().FinalDiagnosticsReader(reader,artifact_ref=artifact).describe()


@pytest.mark.parametrize('kind',['shape','endian','float32','complex_order','custom_float','filter','soft','external','raw','virtual','scaleoffset'])
def test_analysis_array_storage_is_exact_separate_from_core_guard(tmp_path,kind):
    ref,artifact,_,_=fixture(tmp_path)
    with h5py.File(tmp_path/'analysis.h5','r+') as f:
        name='/final_diagnostics/arrays/spectral_integral_eigenvalues'
        del f[name]
        if kind=='shape': f[name]=np.zeros((2,3),dtype='<f8')
        elif kind=='endian': f[name]=np.zeros((2,2),dtype='>f8')
        elif kind=='float32': f[name]=np.zeros((2,2),dtype='<f4')
        elif kind=='complex_order':
            f[name]=np.zeros((2,2),dtype=[('i','<f4'),('r','<f4')])
        elif kind in ('custom_float','filter'):
            dtype=h5py.h5t.IEEE_F64LE.copy()
            plist=h5py.h5p.create(h5py.h5p.DATASET_CREATE)
            if kind=='custom_float': dtype.set_ebias(1022)
            else:
                plist.set_chunk((2,2));plist.set_filter(32015,h5py.h5z.FLAG_OPTIONAL)
            dataset=h5py.h5d.create(f.id,name.encode(),dtype,h5py.h5s.create_simple((2,2)),dcpl=plist)
            dataset.close();plist.close();dtype.close()
        elif kind=='soft': f[name]=h5py.SoftLink('/final_diagnostics/arrays/spectral_trace_cumulative')
        elif kind=='external': f[name]=h5py.ExternalLink('foreign.h5','/x')
        elif kind=='raw': f.create_dataset(name,shape=(2,2),dtype='<f8',external=[('foreign.raw',0,32)])
        elif kind=='virtual': f.create_virtual_dataset(name,h5py.VirtualLayout(shape=(2,2),dtype='<f8'))
        else: f.create_dataset(name,data=np.zeros((2,2)),scaleoffset=1)
    refresh(tmp_path,ref,artifact)
    with StateReader(tmp_path,state_ref=ref,expected_identity=ref['identity']) as reader:
        with pytest.raises(ContractError): api().FinalDiagnosticsReader(reader,artifact_ref=artifact).describe()


def test_small_compressed_chunk_rejected_before_payload_allocation(tmp_path,monkeypatch):
    ref,artifact,_,_=fixture(tmp_path,chunks=True,compression='gzip')
    with StateReader(tmp_path,state_ref=ref,expected_identity=ref['identity'],
                     limits=ResultReadLimits(maximum_array_bytes=32)) as reader:
        view=api().FinalDiagnosticsReader(reader,artifact_ref=artifact)
        view.describe()
        original=h5py.Dataset.__getitem__
        def no_array(dataset,key):
            assert '/arrays/' not in dataset.name, 'chunk payload read before budget guard'
            return original(dataset,key)
        monkeypatch.setattr(h5py.Dataset,'__getitem__',no_array)
        with pytest.raises(ContractError,match='chunk'):
            view.read_integral('spectral_integral',(0,0,0),maximum_bytes=16)


def test_operation_shared_budget_hash_and_source_replacement(tmp_path,monkeypatch):
    ref,artifact,_,_=fixture(tmp_path)
    from qcl_negf_results import state
    with StateReader(tmp_path,state_ref=ref,expected_identity=ref['identity']) as reader:
        deliveries=[]; original=state._IOBudget.account
        def accounted(budget,n): deliveries.append(n); original(budget,n)
        monkeypatch.setattr(state._IOBudget,'account',accounted)
        view=api().FinalDiagnosticsReader(reader,artifact_ref=artifact)
        d=view.describe()
        assert d.io_counters['bytes_read_total']==sum(deliveries)
        assert d.io_counters['sha_bytes']>=int(artifact['bytes'])
        (tmp_path/'analysis.h5').write_bytes((tmp_path/'analysis.h5').read_bytes())
        with pytest.raises(ContractError,match='changed'):
            view.read_integral('spectral_integral',(0,0,0))


def test_independent_tiny_integral_quadrature_values_are_read_not_recomputed(tmp_path):
    # A0=[[2,i],[-i,2]], A1=[[4,0],[0,-2]], A2=0, wE/(2π)=(1/4,1/2,1/8).
    # S=[[2.5,.25i],[-.25i,-.5]], λ=1±sqrt(37)/4, cumulative trace=(1,2,2).
    arrays=wire_values(); arrays['spectral_integral'][:]=np.array([[2.5,.25j],[-.25j,-.5]])
    arrays['spectral_integral_eigenvalues'][:]=[1-math.sqrt(37)/4,1+math.sqrt(37)/4]
    arrays['spectral_trace_cumulative'][:]=np.array([[1,1],[2,2],[2,2]])
    ref,artifact,_,_=fixture(tmp_path,values=arrays)
    with StateReader(tmp_path,state_ref=ref,expected_identity=ref['identity']) as reader:
        view=api().FinalDiagnosticsReader(reader,artifact_ref=artifact)
        for name in ('spectral_integral','spectral_integral_eigenvalues','spectral_trace_cumulative'):
            block=view.read_integral(name,tuple(slice(None) for _ in arrays[name].shape))
            assert np.array_equal(block.values,arrays[name])


def test_header_fixed_widths_share_aggregate_metadata_allowance(tmp_path,monkeypatch):
    from qcl_negf_results.native import _native_header
    fixture(tmp_path)
    with h5py.File(tmp_path/'analysis.h5','r+') as f:
        group=f['metadata']
        for name in tuple(group.attrs):
            value=group.attrs[name];del group.attrs[name]
            group.attrs.create(name,value,dtype='S512')
        observed=[];original=h5py.AttributeManager.__getitem__
        def counted(attrs,key): observed.append(key);return original(attrs,key)
        monkeypatch.setattr(h5py.AttributeManager,'__getitem__',counted)
        with pytest.raises(ContractError,match='aggregate|budget|limit'):
            _native_header(group,2048)
        assert len(observed)<=4


@pytest.mark.parametrize('reader_kind',['final','core'])
@pytest.mark.parametrize('change',['rewrite','replace','unlink'])
@pytest.mark.parametrize('moment',['before','after'])
def test_selected_owner_stays_bound_until_block_publication(tmp_path,monkeypatch,reader_kind,change,moment):
    from result_v2_fixture import array_source
    if reader_kind=='final':
        ref,artifact,_,_=fixture(tmp_path)
        path=tmp_path/'analysis.h5'
        target='/final_diagnostics/arrays/spectral_integral'
    else:
        ref=array_source(tmp_path,value=2.5)
        path=tmp_path/'arrays.h5'
        target='/arrays/a00000001'
    with StateReader(tmp_path,state_ref=ref,expected_identity=ref['identity']) as reader:
        if reader_kind=='final':
            view=api().FinalDiagnosticsReader(reader,artifact_ref=artifact)
            view.describe()
            read=lambda: view.read_integral('spectral_integral',(0,0,0))
        else:
            reader.import_history()
            declaration=reader.history_batch().records[0]['core']['fields']['row']['fields']['density']
            reader.describe_array(declaration)
            read=lambda: reader.read_array(declaration,(0,))
        original=h5py.Dataset.__getitem__
        changed=[]
        def mutate(dataset):
            if change=='rewrite':
                offset=(dataset.id.get_offset() if dataset.chunks is None else
                        dataset.id.get_chunk_info_by_coord((0,)).byte_offset)
                with path.open('r+b') as raw:
                    raw.seek(offset)
                    raw.write(np.asarray([777-321j if reader_kind=='final' else 777],dtype=dataset.dtype).tobytes())
                    raw.flush()
            elif change=='replace':
                replacement=path.with_suffix('.replacement')
                replacement.write_bytes(path.read_bytes())
                replacement.replace(path)
            else:
                path.unlink()
            changed.append(True)
        def selection(dataset,key):
            selected=dataset.name==target and not changed
            if selected and moment=='before': mutate(dataset)
            result=original(dataset,key)
            if selected and moment=='after': mutate(dataset)
            return result
        monkeypatch.setattr(h5py.Dataset,'__getitem__',selection)
        with pytest.raises(ContractError) as error: read()
        assert error.value.code=='source_changed'
        assert changed and reader.io_counters['logical_selected_bytes']==0


def test_healthy_warm_selection_has_no_owner_hash_rescan_and_closes_resources(tmp_path,monkeypatch):
    from qcl_negf_results import state
    ref,artifact,_,_=fixture(tmp_path)
    with StateReader(tmp_path,state_ref=ref,expected_identity=ref['identity']) as reader:
        view=api().FinalDiagnosticsReader(reader,artifact_ref=artifact)
        view.describe()
        owners=list(reader._handles.values())
        def no_rescan(*a,**k): pytest.fail('warm selected read rescanned full owner hash')
        monkeypatch.setattr(state,'_verify_stream',no_rescan)
        for _ in range(2):
            block=view.read_integral('spectral_integral',(0,0,0))
            assert block.values==1+2j and not block.values.flags.writeable
            assert block.io_counters['sha_bytes']==0
            assert block.io_counters['logical_selected_bytes']==16
        assert all(item['handle'].id.valid and not item['raw'].closed for item in owners)
    assert all(not item['handle'].id.valid and item['raw'].closed for item in owners)


@pytest.mark.parametrize('error_type',[RuntimeError,KeyboardInterrupt,SystemExit,__import__('asyncio').CancelledError])
@pytest.mark.parametrize('cleanup_error',[False,True])
def test_temporary_owner_cleanup_preserves_primary_error(tmp_path,monkeypatch,error_type,cleanup_error):
    from qcl_negf_results import state, native
    ref,artifact,_,_=fixture(tmp_path)
    stacks=[]
    original=state.ExitStack
    class TrackedStack(original):
        def __init__(self):
            super().__init__();self.resources=[];stacks.append(self)
        def enter_context(self,manager):
            resource=super().enter_context(manager)
            self.resources.append(resource)
            return resource
        def close(self):
            super().close()
            if cleanup_error: raise OSError('secondary cleanup error')
    monkeypatch.setattr(state,'ExitStack',TrackedStack)
    primary=error_type('primary validation interruption')
    def interrupted(*a,**k): raise primary
    monkeypatch.setattr(native,'validate_native_handle',interrupted)
    try:
        with StateReader(tmp_path,state_ref=ref,expected_identity=ref['identity']) as reader:
            with pytest.raises(error_type) as error:
                api().FinalDiagnosticsReader(reader,artifact_ref=artifact).describe()
            assert error.value is primary
            assert len(stacks)==1 and not reader._handles
            assert not stacks[0]._exit_callbacks
            raw,handle=stacks[0].resources
            assert raw.closed and not handle.id.valid
    finally:
        for stack in stacks: original.close(stack)
