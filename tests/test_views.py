from __future__ import annotations
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from qcl_negf_results.presentation import atomic_json
from qcl_negf_results.render import materialize, materialize_series





from contextlib import contextmanager
from qcl_negf_results.saved_views import AnalysisReader, SavedViewLimits
from qcl_negf_results.state import StateReader, ResultReadLimits
from qcl_negf_contracts.messages import ContractError
from consumer_context_fixture import render_fixture,history_batch,change

SELECTIONS={'potential_total':[slice(None)],'density':[slice(None)],
    'effective_wavefunctions_real':[slice(None),slice(None)],'spectral_trace':[slice(None),slice(None)],
    'spatial_energy_density':[slice(None),slice(None)]}


@contextmanager
def native_render(root,*,ne=2,selected_bytes=1024*1024):
    selected,artifact,value,checkpoint,values=render_fixture(root,ne=ne)
    with StateReader(root,state_ref=selected,expected_identity=selected['identity'],limits=ResultReadLimits(maximum_array_bytes=selected_bytes)) as owner:
        with AnalysisReader(owner,artifact_ref=artifact,limits=SavedViewLimits(maximum_selected_bytes=selected_bytes)) as analysis:
            yield analysis,selected,artifact,checkpoint


def test_native_fields_materialize_without_json_arrays_and_preserve_scientific_evidence(tmp_path):
    source=tmp_path/'point'
    with native_render(source) as (analysis,selected,artifact,checkpoint):
        source_before=(source/artifact['path']).read_bytes()
        scba=history_batch(selected,sequence=(1,2,3,4,5),residual=(1e-2,0.,1e-4,float('nan'),1e-6))
        outer=history_batch(selected,sequence=(6,7),outer=(1,2),residual=(1e-2,1e-3),stage='outer',checkpoint=checkpoint)
        view=materialize(analysis,tmp_path/'views',selections=SELECTIONS,history=(scba,outer))
        assert view['scientific_accepted'] is None and view['presentation_ready'] is True
        import struct
        assert struct.unpack('>d',bytes.fromhex(view['validation']['summary_metrics'][0]['value']['bits']))[0]==.43
        assert len(view['positivity']['data']['witnesses'])==6
        assert all(struct.unpack('>d',bytes.fromhex(v['minimum_eigenvalue']['bits']))[0]==-.001 for v in view['positivity']['data']['witnesses'])
        assert {figure['id'] for figure in view['figures']} >= {'potentials','density','wavefunctions','spectra','energy-density','convergence-scba','convergence-outer'}
        for figure in view['figures']:
            assert figure['source_state']==selected
            assert (tmp_path/'views'/figure['path']).is_file()
            assert 'values' not in figure and 'x' not in figure
        convergence=next(f for f in view['figures'] if f['id']=='convergence-scba')
        svg=(tmp_path/'views'/convergence['path']).read_text()
        assert 'log' in svg and 'разрывы/неположительные значения: 2' in svg
        assert 'nan' not in svg.lower()
        image=next(f for f in view['figures'] if f['id']=='energy-density')
        assert image['native_shape']==[2,3] and image['color_range']==[-2,2]
        assert image['media_type']=='image/svg+xml'
        svg=(tmp_path/'views'/image['path']).read_text()
        assert 'viewBox="0 0 840 560"' in svg and 'z, nm' in svg and 'E, eV' in svg
        assert 'линейная шкала' in svg and 'm⁻³' in svg and 'eV^-1' in svg
        assert 'href="data:image/png;base64,' in svg
        assert (tmp_path/'views'/image['raw_path']).read_bytes().startswith(b'\x89PNG')
        assert (source/artifact['path']).read_bytes()==source_before
        assert len(json.dumps(view).encode())<100_000



def test_immutable_renderer_cache_and_corrupt_hdf5_rejected(tmp_path,monkeypatch):
    source=tmp_path/'point'
    with native_render(source) as (analysis,selected,artifact,_):
        view=materialize(analysis,tmp_path/'views',selections=SELECTIONS)
        monkeypatch.setattr(h5py,'File',lambda *a,**k:pytest.fail('cache hit reopened numerical HDF5'))
        assert materialize(analysis,tmp_path/'views',selections=SELECTIONS)==view
        (source/artifact['path']).write_bytes(b'truncated')
        with pytest.raises(ContractError,match='changed'):
            materialize(analysis,tmp_path/'fresh-cache',selections=SELECTIONS)



def test_pointer_hash_and_path_escape_are_rejected(tmp_path):
    from copy import deepcopy
    root=tmp_path/'source';selected,artifact,_,_,_=render_fixture(root)
    corrupt=deepcopy(selected);corrupt['manifest']['sha256']='0'*64
    with pytest.raises(ContractError,match='checksum|hash|differ'):
        StateReader(root,state_ref=corrupt,expected_identity=selected['identity'])
    corrupt=deepcopy(selected);corrupt['manifest']['path']='../../escape.json'
    with pytest.raises(ContractError):StateReader(root,state_ref=corrupt,expected_identity=selected['identity'])


def test_series_graph_has_explicit_unaccepted_marker_not_interpolated_line(tmp_path):
    rows=[{"id":f"p{i}","execution_id":"ex","scientific_accepted":i!=1,"status":"completed","quality":"strict","converged":True,"coordinates":{"temperature_K":70,"voltage_per_period_V":i*.01,"branch":"forward","order":i},"observables":{"current_density_A_per_m2":i+1}} for i in range(3)]
    figures=materialize_series(rows,tmp_path)
    svg=(tmp_path/figures[0]["path"]).read_text()
    assert "непринята" in svg and svg.count("<circle") == 3
    assert "<polyline" not in svg  # neither isolated accepted endpoint disappears or crosses the failure
    assert rows[1]["observables"]["current_density_A_per_m2"]==2


def test_undeclared_axis_order_is_rejected_without_transpose_guess(tmp_path):
    root=tmp_path/'point';selected,artifact,_,_,_=render_fixture(root)
    change(root,selected,artifact,lambda v:v['views']['spatial_energy_density_per_eV_m3'].update(axes=['z','E']))
    with StateReader(root,state_ref=selected,expected_identity=selected['identity']) as owner:
        with pytest.raises(ContractError):materialize(AnalysisReader(owner,artifact_ref=artifact),tmp_path/'views',selections=SELECTIONS)



def test_empty_outer_history_does_not_hide_committed_scba_physics(tmp_path):
    with native_render(tmp_path/'point') as (analysis,selected,_,_):
        batch=history_batch(selected)
        result=materialize(analysis,tmp_path/'views',selections=SELECTIONS,history=(batch,))
        assert result['presentation_ready'] and any(f['id']=='convergence-scba' for f in result['figures'])
        assert not any(f['id']=='convergence-outer' for f in result['figures'])





@pytest.mark.parametrize("damage", ["hash", "identity", "sequence", "path"])
def test_live_history_rejects_corrupt_or_foreign_segments(tmp_path,damage):
    from copy import deepcopy
    root=tmp_path/'source';selected,artifact,_,_,_=render_fixture(root);batch=history_batch(selected)
    identity=deepcopy(selected['identity'])
    if damage=='identity':identity['point_id']='other'
    elif damage=='path':selected['manifest']['path']='../../foreign.json'
    elif damage=='hash':(root/artifact['path']).write_bytes(b'corrupt')
    else:
        # An explicitly passed batch still cannot duplicate native append order.
        batch.rows[1]['sequence']['decimal']='1';batch.records[1]['core']['fields']['sequence']['decimal']='1'
    with pytest.raises((ContractError,ValueError)):
        with StateReader(root,state_ref=selected,expected_identity=identity) as owner:
            materialize(AnalysisReader(owner,artifact_ref=artifact),tmp_path/'views',selections=SELECTIONS,history=(batch,))



def test_live_log_graph_does_not_bridge_nonfinite_or_nonpositive_history(tmp_path):
    with native_render(tmp_path/'point') as (analysis,selected,_,_):
        batch=history_batch(selected,residual=(.1,float('nan'),.001))
        result=materialize(analysis,tmp_path/'views',selections=SELECTIONS,history=(batch,))
        figure=next(f for f in result['figures'] if f['id']=='convergence-scba')
        svg=(tmp_path/'views'/figure['path']).read_text()
        assert '<polyline' not in svg and svg.count('<circle')==2
        assert 'nan' not in svg.lower()



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


def test_monitor_uses_native_latest_outer_cycle_and_full_history_keeps_boundaries(tmp_path,monkeypatch):
    import qcl_negf_results.render as render
    captured=[];original=render._svg
    def capture(*args,**kwargs):captured.append((args,kwargs));return original(*args,**kwargs)
    monkeypatch.setattr(render,'_svg',capture)
    with native_render(tmp_path/'point') as (analysis,selected,artifact,_):
        batch=history_batch(selected,sequence=(100,101,102,103,104),iteration=(1,2,3,1,2),outer=(1,1,1,2,2),residual=(.1,.001,.0001,.2,.02))
        before=(tmp_path/'point'/artifact['path']).read_bytes()
        value=materialize(analysis,tmp_path/'views',selections=SELECTIONS,history=(batch,))
        figure=next(v for v in value['figures'] if v['id']=='convergence-scba')
        assert figure['history_rows']==5 and figure['history_cycle_count']==2
        assert figure['monitor_context']==dict(phase='scba',attempt='1',outer_iteration=2,first_iteration=1,last_iteration=2,first_sequence=103,last_sequence=104,rows=2)
        monitor_args,monitor_kwargs=next(v for v in captured if v[1].get('monitor') and v[1].get('logarithmic'))
        assert monitor_args[1]=='Итерация SCBA, ν' and monitor_args[3]==[('r_K',[1.,2.],[.2,.02])]
        full_args,full_kwargs=next(v for v in captured if not v[1].get('monitor') and v[1].get('logarithmic'))
        xx,yy=full_args[3][0][1:]
        assert set(x for x in xx if np.isfinite(x))=={100,101,102,103,104}
        assert any(not np.isfinite(x) for x in xx)
        assert full_kwargs['boundaries']==[(103.,'μ=2 · a=1')]
        assert (tmp_path/'point'/artifact['path']).read_bytes()==before



def test_resume_keeps_native_indices_and_excludes_previous_attempt_from_monitor(tmp_path,monkeypatch):
    import qcl_negf_results.render as render
    captured=[];original=render._svg
    def capture(*args,**kwargs):
        if kwargs.get('monitor') and kwargs.get('logarithmic'):captured.extend(args[3])
        return original(*args,**kwargs)
    monkeypatch.setattr(render,'_svg',capture)
    with native_render(tmp_path/'point') as (analysis,selected,_,_):
        batch=history_batch(selected,sequence=(31,32,33,34),iteration=(1,2,3,4),outer=(7,7,7,7),attempts=('2','2','10','10'),residual=(.1,.01,.009,.008))
        value=materialize(analysis,tmp_path/'views',selections=SELECTIONS,history=(batch,))
        context=next(v for v in value['figures'] if v['id']=='convergence-scba')['monitor_context']
        assert context['attempt']=='10' and context['outer_iteration']==7
        assert context['first_iteration']==3 and context['last_iteration']==4
        assert captured==[('r_K',[3.,4.],[.009,.008])]



def test_cycle_spanning_closed_segments_does_not_reset_at_file_boundary(tmp_path,monkeypatch):
    from dataclasses import replace
    with native_render(tmp_path/'point') as (analysis,selected,_,_):
        batch=history_batch(selected,sequence=(1,2,3,4,5,6),iteration=(1,2,1,2,3,4),outer=(1,1,2,2,2,2),residual=(.1,.01,.2,.1,float('nan'),.01))
        first=replace(batch,records=batch.records[:3],rows=batch.rows[:3],at_cutoff=False)
        second=replace(batch,records=batch.records[3:],rows=batch.rows[3:])
        value=materialize(analysis,tmp_path/'views',selections=SELECTIONS,history=(first,second))
        figure=next(v for v in value['figures'] if v['id']=='convergence-scba')
        assert figure['history_cycle_count']==2 and figure['monitor_context']['rows']==4
        assert figure['monitor_context']['first_iteration']==1 and figure['monitor_context']['last_iteration']==4
        svg=(tmp_path/'views'/figure['monitor_path']).read_text()
        assert svg.count('<polyline')==1 and svg.count('<circle')==1



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
    with native_render(tmp_path/'point',ne=1921,selected_bytes=2*1024*1024) as (analysis,_,artifact,_):
        source=tmp_path/'point'/artifact['path'];contents=source.read_bytes()
        view=materialize(analysis,tmp_path/'views',selections={'spatial_energy_density':[slice(None),slice(None)]})
        figure=next(v for v in view['figures'] if v['id']=='energy-density')
        assert figure['native_shape']==[1921,3] and figure['display_aspect_ratio']=='3:2'
        svg=(tmp_path/'views'/figure['path']).read_text()
        assert 'viewBox="0 0 840 560"' in svg and '1921 × 3' in svg
        assert '-0.1473913140194004' not in svg
        assert source.read_bytes()==contents



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
    with native_render(tmp_path/'point') as (analysis,_,_,_):
        destination=tmp_path/'views';view=materialize(analysis,destination,selections=SELECTIONS)
        cache=destination/Path(view['figures'][0]['path']).parent/'index.json'
        stale=json.loads(cache.read_text());stale['contract_set']='qcl-negf.results.0.10.5';atomic_json(cache,stale)
        with pytest.raises(ValueError,match='contract'):materialize(analysis,destination,selections=SELECTIONS)



def test_native5_raster_preflight_is_arithmetic_and_nonuniform_does_not_force_screen_minimum():
    from qcl_negf_results.render import _raster_forecast, _heatmap_svg
    native, peak = _raster_forecast((4,3),(4,3),retained_bytes=1024)
    assert native > 8*12 and peak >= native+1024
    assert _raster_forecast((10**30,10**30),(2,2))[0] > 10**60
    image, raw, annotation = _heatmap_svg(np.arange(12.).reshape(4,3),[0.,1.,9.],[-.15,-.14,.05,.2],'eV^-1 m^-3','invalid')
    assert np.prod(annotation['display_shape']) <= 12
    assert annotation['native_shape'] == [4,3]
    assert 'image/png' in image and raw.startswith(b'\x89PNG')


def test_native5_materialize_explicit_selection_borrowed_cache_and_source_binding(tmp_path,monkeypatch):
    from consumer_context_fixture import semantic_fixture
    from qcl_negf_results.saved_views import AnalysisReader
    from qcl_negf_results.state import StateReader
    selected,artifact,_,_,_=semantic_fixture(tmp_path/'source')
    with StateReader(tmp_path/'source',state_ref=selected,expected_identity=selected['identity']) as owner:
        with AnalysisReader(owner,artifact_ref=artifact) as analysis:
            selections={'density':[slice(None)],'spatial_energy_density':[slice(None),slice(None)]}
            view=materialize(analysis,tmp_path/'views',selections=selections)
            assert view['source_state']==selected and view['scientific_accepted'] is None
            assert {f['id'] for f in view['figures']}=={'density','energy-density'}
            image=next(f for f in view['figures'] if f['id']=='energy-density')
            assert image['native_shape']==[2,3] and image['color_range']==[-21.,21.]
            monkeypatch.setattr(h5py,'File',lambda *a,**k:pytest.fail('warm native cache reopened HDF'))
            assert materialize(analysis,tmp_path/'views',selections=selections)==view
            with (tmp_path/'source'/artifact['path']).open('r+b') as stream:
                stream.seek(-1,2);cell=stream.read(1);stream.seek(-1,2);stream.write(bytes([cell[0]^1]))
            with pytest.raises(Exception,match='changed|owner'):
                materialize(analysis,tmp_path/'views',selections=selections)


def test_native5_acceptance_missing_and_null_never_default_true():
    from qcl_negf_results.presentation import accepted_result
    record=dict(status='completed',quality='strict',converged=True)
    assert not accepted_result(record)
    assert not accepted_result(dict(record,scientific_accepted=None))
    assert not accepted_result(dict(record,scientific_accepted=False))
    assert accepted_result(dict(record,scientific_accepted=True))


def test_native5_series_missing_middle_is_a_gap_not_a_deleted_point(tmp_path):
    rows=[dict(id=str(i),execution_id='literal',status='completed',quality='strict',converged=True,coordinates=dict(temperature_K=200.,branch='f',order=i,voltage_per_period_V=.1*i),
        observables=dict(current_density_A_per_m2=None if i==1 else float(i+1)),scientific_accepted=True) for i in range(3)]
    result=materialize_series(rows,tmp_path)
    svg=(tmp_path/result[0]['path']).read_text()
    assert '<polyline' not in svg
    assert svg.count('<circle')==2


def test_native5_renderer_history_current_cycle_native_indices_and_explicit_boundaries(tmp_path,monkeypatch):
    from consumer_context_fixture import semantic_fixture,history_batch
    from qcl_negf_results.saved_views import AnalysisReader
    from qcl_negf_results.state import StateReader
    import qcl_negf_results.render as render
    selected,artifact,_,_,_=semantic_fixture(tmp_path/'source')
    batch=history_batch(selected,sequence=(100,101,102,103,104),iteration=(1,2,3,1,2),outer=(1,1,1,2,2),residual=(.1,.001,.0001,.2,.02))
    captured=[];original=render._svg
    def capture(*args,**kwargs):captured.append((args,kwargs));return original(*args,**kwargs)
    monkeypatch.setattr(render,'_svg',capture)
    with StateReader(tmp_path/'source',state_ref=selected,expected_identity=selected['identity']) as owner:
        view=materialize(AnalysisReader(owner,artifact_ref=artifact),tmp_path/'views',selections={'density':[slice(None)]},history=(batch,))
    figure=next(v for v in view['figures'] if v['id']=='convergence-scba')
    assert figure['history_rows']==5 and figure['history_cycle_count']==2
    assert figure['monitor_context']==dict(phase='scba',attempt='1',outer_iteration=2,first_iteration=1,last_iteration=2,first_sequence=103,last_sequence=104,rows=2)
    assert next(v[0][3] for v in captured if v[1].get('monitor'))==[('r_K',[1.,2.],[.2,.02])]
    args,kwargs=next(v for v in captured if v[1].get('logarithmic') and not v[1].get('monitor'))
    assert set(x for x in args[3][0][1] if np.isfinite(x))=={100,101,102,103,104}
    assert any(not np.isfinite(x) for x in args[3][0][1])
    assert kwargs['boundaries']==[(103.,'μ=2 · a=1')]


@pytest.mark.parametrize('middle',[dict(status='failed'),dict(scientific_accepted=None),dict(current=float('nan')),dict(current=float('inf'))])
def test_native5_series_failed_unknown_and_nonfinite_break_only_their_segment(tmp_path,middle):
    rows=[dict(id=str(i),execution_id='literal',status='completed',quality='strict',converged=True,
        scientific_accepted=True,coordinates=dict(temperature_K=200.,branch='f',order=i,voltage_per_period_V=.1*i),
        observables=dict(current_density_A_per_m2=float(i+1))) for i in range(3)]
    rows[1].update({k:v for k,v in middle.items() if k!='current'})
    if 'current' in middle:rows[1]['observables']['current_density_A_per_m2']=middle['current']
    result=materialize_series(rows,tmp_path)
    svg=(tmp_path/result[0]['path']).read_text()
    assert '<polyline' not in svg
    assert svg.count('<circle')== (3 if middle.get('status')=='failed' or 'scientific_accepted' in middle else 2)


@pytest.mark.parametrize('shape',[(2,8192),(8192,2),(3,5)])
def test_native5_rowwise_png_exact_literal_colors_and_extrema_for_wide_deep_invalid(shape):
    from qcl_negf_results.render import _png
    import zlib,struct
    array=np.full(shape,np.nan)
    if shape!=(3,5):array[0,:7 if shape[1]>=7 else 2]=np.array([-1.,-.5,-0.,0.,.5,1.,np.inf])[:min(7,shape[1])]
    raw,annotation=_png(array)
    width,height=struct.unpack('>II',raw[16:24]);assert (height,width)==shape
    compressed=b'';position=8
    while position<len(raw):
        size=struct.unpack('>I',raw[position:position+4])[0];kind=raw[position+4:position+8]
        if kind==b'IDAT':compressed+=raw[position+8:position+8+size]
        position+=12+size
    scan=zlib.decompress(compressed)
    def pixel(row,column):
        start=(height-1-row)*(1+3*width)+1+3*column
        return tuple(scan[start:start+3])
    assert pixel(height-1,width-1)==(155,155,155)
    if shape==(3,5):assert annotation['native_range'] is None and annotation['missing_count']==15
    elif width>=7:
        assert [pixel(0,i) for i in range(7)]==[(0,0,255),(127,127,255),(255,255,255),(255,255,255),(255,127,127),(255,0,0),(155,155,155)]
        assert annotation['color_range']==[-1.,1.]
    else:assert pixel(0,0)==(0,0,255) and pixel(0,1)==(127,127,255)


def test_native5_raster_guard_refuses_huge_policy_metadata_before_numeric_payload(tmp_path,monkeypatch):
    from qcl_negf_results.saved_views import SavedViewLimits
    with native_render(tmp_path/'source') as (analysis,_,_,_):
        # Tight policy makes native-alone impossible even for this literal tiny
        # source. Large dimension arithmetic is separately allocation-free.
        analysis.limits=SavedViewLimits(maximum_selected_bytes=1,maximum_cells=1)
        original=h5py.Dataset.__getitem__
        def no_numeric(dataset,key):
            assert '/arrays/' not in dataset.name
            return original(dataset,key)
        monkeypatch.setattr(h5py.Dataset,'__getitem__',no_numeric)
        with pytest.raises(ContractError,match='native raster'):
            materialize(analysis,tmp_path/'views',selections={'spatial_energy_density':[slice(None),slice(None)]})


def test_native5_explicit_history_current_is_signed_linear_with_recorded_units(tmp_path,monkeypatch):
    import qcl_negf_results.render as render
    from result_v2_fixture import ieee
    import struct
    captured=[];original=render._svg
    def capture(*args,**kwargs):captured.append((args,kwargs));return original(*args,**kwargs)
    monkeypatch.setattr(render,'_svg',capture)
    with native_render(tmp_path/'source') as (analysis,selected,_,_):
        batch=history_batch(selected)
        for record,row,current in zip(batch.records,batch.rows,(-2.,0.,3.),strict=True):
            tag=ieee(struct.pack('>d',current).hex());record['core']['fields']['row']['fields']['J']=tag
            row['cells']['J']=dict(tagged=tag,value=current,invalid_reason=None,units='A/m^2',missing_reason=None)
        result=materialize(analysis,tmp_path/'views',selections={'density':[slice(None)]},history=(batch,))
        figure=next(v for v in result['figures'] if v['id']=='history-J-scba')
        assert figure['units']['y']=='A/m^2'
        args,kwargs=next(v for v in captured if v[0][0]=='J: scba')
        assert not kwargs.get('logarithmic',False)
        assert args[3]==[('J',[1.,2.,3.],[-2.,0.,3.])]


def test_native5_cache_publication_rechecks_owner_after_cached_asset_read(tmp_path,monkeypatch):
    with native_render(tmp_path/'source') as (analysis,_,artifact,_):
        destination=tmp_path/'views';selections={'density':[slice(None)]}
        view=materialize(analysis,destination,selections=selections)
        asset=destination/view['figures'][0]['path'];changed=[]
        def read_then_tamper():
            if not changed:
                with (tmp_path/'source'/artifact['path']).open('r+b') as raw:
                    raw.seek(-1,2);cell=raw.read(1);raw.seek(-1,2);raw.write(bytes([cell[0]^1]))
                changed.append(True)
        _track_cache_reads(monkeypatch,asset,lambda:None,after_read=read_then_tamper)
        with pytest.raises(ContractError) as failure:materialize(analysis,destination,selections=selections)
        assert failure.value.code=='source_changed' and changed==[True]


@pytest.mark.parametrize('field',['acceptance','quality','identity','empty_figures','source','units','native_range','selection'])
def test_b1_f1_warm_cache_metadata_has_no_authority_over_qualified_render(tmp_path,monkeypatch,field):
    with native_render(tmp_path/'source') as (analysis,_,_,_):
        destination=tmp_path/'views';selections={'density':[slice(None)]}
        expected=materialize(analysis,destination,selections=selections)
        index=destination/expected['cache_key']/'index.json'
        bad=json.loads(json.dumps(expected))
        if field=='acceptance':bad['scientific_accepted']=True
        elif field=='quality':bad['quality']='strict'
        elif field=='identity':bad['identity']['run_id']='foreign'
        elif field=='empty_figures':bad['figures']=[]
        elif field=='source':bad['figures'][0]['source_state']['identity']['run_id']='foreign'
        elif field=='units':bad['figures'][0]['units']['y']='A/m^2'
        elif field=='native_range':bad['figures'][0]['native_range']=[123,456]
        else:bad['figures'][0]['raw_selections'][0]['selection']=[dict(start=1,stop=2,step=1)]
        assert bad['source_binding']==expected['source_binding']
        index.write_text(json.dumps(bad))
        original=h5py.Dataset.__getitem__
        def no_numeric(dataset,key):
            assert '/arrays/' not in dataset.name,'corrupt warm cache must reject without numeric reconstruction'
            return original(dataset,key)
        monkeypatch.setattr(h5py.Dataset,'__getitem__',no_numeric)
        with pytest.raises(ContractError,match='cache|metadata'):
            materialize(analysis,destination,selections=selections)


def test_b1_f1_public_return_mutation_cannot_rewrite_memo_and_new_facade_is_warm(tmp_path,monkeypatch):
    with native_render(tmp_path/'source') as (analysis,_,artifact,_):
        destination=tmp_path/'views';selections={'density':[slice(None)]}
        public=materialize(analysis,destination,selections=selections)
        expected=json.loads(json.dumps(public))
        public['scientific_accepted']=True;public['figures'].clear()
        public['source_binding']['identity']['run_id']='foreign'
        original=h5py.Dataset.__getitem__
        def no_numeric(dataset,key):
            assert '/arrays/' not in dataset.name,'same-owner new facade must keep a healthy warm hit'
            return original(dataset,key)
        monkeypatch.setattr(h5py.Dataset,'__getitem__',no_numeric)
        with AnalysisReader(analysis.reader,artifact_ref=artifact) as next_facade:
            assert materialize(next_facade,destination,selections=selections)==expected


@pytest.mark.parametrize('field',['units','missing_reason'])
def test_b1_f1_history_projection_cannot_contradict_raw_pairs(tmp_path,field):
    from result_v2_fixture import ieee
    import struct
    with native_render(tmp_path/'source') as (analysis,selected,_,_):
        batch=history_batch(selected)
        for record,row,current in zip(batch.records,batch.rows,(-2.,0.,3.),strict=True):
            tag=ieee(struct.pack('>d',current).hex());record['core']['fields']['row']['fields']['J']=tag
            row['cells']['J']=dict(tagged=tag,value=current,invalid_reason=None,units='A/m^2',missing_reason=None)
        for row in batch.rows:row['cells']['J'][field]='mA/m^2' if field=='units' else 'invented missing'
        with pytest.raises(ContractError,match='units|missing|projection'):
            materialize(analysis,tmp_path/'views',selections={'density':[slice(None)]},history=(batch,))


def test_b1_f1_healthy_history_cache_is_warm_without_numeric_or_full_hash_reads(tmp_path,monkeypatch):
    with native_render(tmp_path/'source') as (analysis,selected,_,_):
        batch=history_batch(selected)
        destination=tmp_path/'views';selections={'density':[slice(None)]}
        expected=materialize(analysis,destination,selections=selections,history=(batch,))
        original=h5py.Dataset.__getitem__
        def no_numeric(dataset,key):
            assert '/arrays/' not in dataset.name
            return original(dataset,key)
        monkeypatch.setattr(h5py.Dataset,'__getitem__',no_numeric)
        actual=materialize(analysis,destination,selections=selections,history=(batch,))
        assert actual==json.loads(json.dumps(expected))
        assert analysis.reader.io_counters['sha_bytes']==0


def test_b1_f1_new_owner_recreates_unqualified_disk_metadata(tmp_path,monkeypatch):
    source=tmp_path/'source';destination=tmp_path/'views';selections={'density':[slice(None)]}
    with native_render(source) as (analysis,selected,artifact,_):
        expected=materialize(analysis,destination,selections=selections)
    index=destination/expected['cache_key']/'index.json'
    bad=json.loads(json.dumps(expected));bad['scientific_accepted']=True;bad['figures']=[]
    index.write_text(json.dumps(bad))
    reads=[];original=h5py.Dataset.__getitem__
    def numeric(dataset,key):
        if '/arrays/' in dataset.name:reads.append(dataset.name)
        return original(dataset,key)
    monkeypatch.setattr(h5py.Dataset,'__getitem__',numeric)
    with StateReader(source,state_ref=selected,expected_identity=selected['identity']) as owner:
        with AnalysisReader(owner,artifact_ref=artifact) as next_view:
            assert materialize(next_view,destination,selections=selections)==expected
    assert reads and all(name.rsplit('/',1)[1] in {'density_per_m3','z_nm','wX'} for name in reads)
    assert json.loads(index.read_text())['scientific_accepted'] is None


def test_b1_f1_one_entry_replacement_and_warm_workspace_admission(tmp_path,monkeypatch):
    from dataclasses import replace
    with native_render(tmp_path/'source') as (analysis,_,_,_):
        destination=tmp_path/'views'
        first=materialize(analysis,destination,selections={'density':[slice(None)]})
        old=analysis.reader._materialized_point_memo
        second=materialize(analysis,destination,selections={'density':[slice(0,2)]})
        memo=analysis.reader._materialized_point_memo
        assert type(memo) is tuple and type(memo[1]) is bytes and memo[0]==second['cache_key']!=first['cache_key']
        assert memo is not old and old not in tuple(analysis.reader.__dict__.values())
        # The private allowance uses the tighter actual owner selected-byte cap.
        # Same facade limits/key: the retained memo plus decode containers must
        # refuse before any selected payload, then admit after restoring space.
        analysis.reader.limits=replace(analysis.reader.limits,maximum_array_bytes=4096)
        original=h5py.Dataset.__getitem__
        def no_numeric(dataset,key):
            assert '/arrays/' not in dataset.name
            return original(dataset,key)
        monkeypatch.setattr(h5py.Dataset,'__getitem__',no_numeric)
        with pytest.raises(ContractError,match='memo') as error:
            materialize(analysis,destination,selections={'density':[slice(0,2)]})
        assert error.value.code=='budget_exceeded' and analysis.reader._materialized_point_memo is memo
        analysis.reader.limits=replace(analysis.reader.limits,maximum_array_bytes=131072)
        assert materialize(analysis,destination,selections={'density':[slice(0,2)]})==second


def test_b1_f1_matching_raw_missing_and_varying_units_remain_signed_gaps(tmp_path,monkeypatch):
    import math
    import struct
    from importlib.resources import files
    import qcl_negf_results.render as render
    from result_v2_fixture import ieee,symbol
    nodes=json.loads(files('qcl_negf_contracts').joinpath('core_layout.json').read_text())['graph']['nodes']
    pair=next(name for name,node in nodes.items() if node.get('fields')==[
        {'name':'first','type':'Symbol'},{'name':'second','type':'Symbol'}])
    captured=[];original=render._svg
    def capture(*args,**kwargs):captured.append((args,kwargs));return original(*args,**kwargs)
    monkeypatch.setattr(render,'_svg',capture)
    with native_render(tmp_path/'source') as (analysis,selected,_,_):
        batch=history_batch(selected,sequence=(1,2,3,4),residual=(.1,.01,.001,.0001))
        for i,(record,row,value) in enumerate(zip(batch.records,batch.rows,(-2.,-0.,3.,-4.),strict=True)):
            tag=ieee(struct.pack('>d',value).hex());core=record['core']['fields'];core['row']['fields']['J']=tag
            unit='A/m^2' if i<3 else 'mA/m^2'
            core['units'][0]['fields']['second']=unit
            missing='no_previous_iteration' if i==1 else None
            if missing:core['missing_reasons']=[dict(type=pair,fields=dict(first=symbol('J'),second=symbol(missing)))]
            row['cells']['J']=dict(tagged=tag,value=value,invalid_reason=None,units=unit,missing_reason=missing)
        result=materialize(analysis,tmp_path/'views',selections={'density':[slice(None)]},history=(batch,))
        figure=next(v for v in result['figures'] if v['id']=='history-J-scba')
        assert figure['units']['y']=='stored units vary'
        args,kwargs=next(v for v in captured if v[0][0]=='J: scba')
        assert not kwargs.get('logarithmic',False)
        _,xs,ys=args[3][0]
        assert xs[:3]==[1.,2.,3.] and math.isnan(xs[3]) and xs[4]==4.
        assert ys[0]==-2. and math.isnan(ys[1]) and ys[2]==3. and math.isnan(ys[3]) and ys[4]==-4.
        assert batch.rows[1]['cells']['J']['tagged']['bits']=='8000000000000000'


def _track_cache_reads(monkeypatch,target,before_read,*,after_read=lambda:None):
    """Observe real FileIO/Path.open reads at the shared Python I/O boundary."""
    import io
    import os
    calls=[];streams=[];opened=io.open
    class Tracked:
        def __init__(self,stream):self.stream=stream
        def __getattr__(self,name):return getattr(self.stream,name)
        def __enter__(self):return self
        def __exit__(self,*args):return self.stream.__exit__(*args)
        def read(self,size=-1):
            before_read()
            value=self.stream.read(size);calls.append((size,len(value)))
            after_read()
            return value
    def tracked(file,*args,**kwargs):
        stream=opened(file,*args,**kwargs)
        path=Path(os.readlink('/proc/self/fd/'+str(file))) if isinstance(file,int) else Path(file)
        if path==target:
            streams.append(stream)
            return Tracked(stream)
        return stream
    monkeypatch.setattr(io,'open',tracked)
    return calls,streams


def _cache_role(analysis,selected,destination,role):
    analysis.limits=SavedViewLimits(maximum_selected_bytes=262144,maximum_cells=16)
    selections={'spatial_energy_density':[slice(None),slice(None)]} if role=='png' else {'density':[slice(None)]}
    history=(history_batch(selected,sequence=(1,),residual=(.1,)),) if role=='monitor' else ()
    expected=materialize(analysis,destination,selections=selections,history=history)
    if role=='index':target=destination/expected['cache_key']/'index.json'
    else:
        key='raw_path' if role=='png' else 'monitor_path' if role=='monitor' else 'path'
        target=destination/next(f[key] for f in expected['figures'] if key in f)
    return selections,history,expected,target


@pytest.mark.parametrize('role',['svg','png','monitor'])
def test_b1_f2_actual_asset_growth_is_bounded_before_allocation(tmp_path,monkeypatch,role):
    import os
    with native_render(tmp_path/'source') as (analysis,selected,_,_):
        destination=tmp_path/'views'
        selections,history,expected,target=_cache_role(analysis,selected,destination,role)
        assert materialize(analysis,destination,selections=selections,history=history)==json.loads(json.dumps(expected))
        admitted=target.stat().st_size;workspace=16*262144+8*16;changed=[]
        def grow():
            if changed:return
            # Real growth after the read descriptor has been opened/qualified;
            # sparse extension avoids an adversarial fixture allocation.
            with target.open('ab',buffering=0) as stream:os.ftruncate(stream.fileno(),workspace+1024)
            changed.append(True)
        calls,streams=_track_cache_reads(monkeypatch,target,grow)
        with pytest.raises(ContractError):materialize(analysis,destination,selections=selections,history=history)
        assert changed==[True] and calls
        assert all(0<size<=65536 and actual<=size for size,actual in calls)
        assert sum(actual for _,actual in calls)<=admitted+1
        assert all(stream.closed for stream in streams)


@pytest.mark.parametrize('role',['svg','png','monitor','index'])
def test_b1_f2_same_size_path_replacement_during_actual_read_is_rejected(tmp_path,monkeypatch,role):
    with native_render(tmp_path/'source') as (analysis,selected,_,_):
        destination=tmp_path/'views'
        selections,history,_,target=_cache_role(analysis,selected,destination,role)
        replacement=target.with_suffix(target.suffix+'.replacement');replacement.write_bytes(target.read_bytes());changed=[]
        def replace():
            if not changed:replacement.replace(target);changed.append(True)
        calls,streams=_track_cache_reads(monkeypatch,target,replace)
        with pytest.raises(ContractError):materialize(analysis,destination,selections=selections,history=history)
        assert changed==[True] and calls and all(stream.closed for stream in streams)


@pytest.mark.parametrize('kind',['fifo','directory'])
def test_b1_f2_warm_index_nonregular_refuses_before_numeric_read(tmp_path,monkeypatch,kind):
    import os
    with native_render(tmp_path/'source') as (analysis,selected,_,_):
        destination=tmp_path/'views'
        selections,history,_,target=_cache_role(analysis,selected,destination,'index')
        target.unlink()
        if kind=='fifo':os.mkfifo(target)
        else:target.mkdir()
        original=h5py.Dataset.__getitem__
        def no_numeric(dataset,key):
            assert '/arrays/' not in dataset.name,'nonregular warm index cannot trigger a cold scientific read'
            return original(dataset,key)
        monkeypatch.setattr(h5py.Dataset,'__getitem__',no_numeric)
        with pytest.raises(ContractError):materialize(analysis,destination,selections=selections,history=history)


@pytest.mark.parametrize('role',['svg','index'])
def test_b1_f2_symlink_swap_at_actual_open_refuses_without_following(tmp_path,monkeypatch,role):
    import os
    with native_render(tmp_path/'source') as (analysis,selected,_,_):
        selections,history,_,target=_cache_role(analysis,selected,tmp_path/'views',role)
        other=target.with_suffix('.other');other.write_bytes(target.read_bytes())
        real_open=os.open;changed=[]
        def swap(path,*args,**kwargs):
            if Path(path)==target and not changed:
                target.unlink();target.symlink_to(other.name);changed.append(True)
            return real_open(path,*args,**kwargs)
        monkeypatch.setattr(os,'open',swap)
        with pytest.raises(ContractError,match='without following links'):
            materialize(analysis,tmp_path/'views',selections=selections,history=history)
        assert changed==[True]


@pytest.mark.parametrize('role',['svg','index'])
@pytest.mark.parametrize('exception',[RuntimeError,KeyboardInterrupt])
def test_b1_f2_actual_read_failure_closes_descriptor_and_preserves_primary(tmp_path,monkeypatch,role,exception):
    with native_render(tmp_path/'source') as (analysis,selected,_,_):
        selections,history,_,target=_cache_role(analysis,selected,tmp_path/'views',role)
        def fail():raise exception('primary read cancellation')
        calls,streams=_track_cache_reads(monkeypatch,target,fail)
        with pytest.raises(exception,match='primary read cancellation'):
            materialize(analysis,tmp_path/'views',selections=selections,history=history)
        assert calls==[] and streams and all(stream.closed for stream in streams)


def test_b1_f2_opened_oversize_asset_refuses_before_actual_read(tmp_path,monkeypatch):
    import os
    with native_render(tmp_path/'source') as (analysis,selected,_,_):
        selections,history,_,target=_cache_role(analysis,selected,tmp_path/'views','svg')
        os.truncate(target,16*262144+8*16+1024)
        def forbidden():pytest.fail('opened size admission must precede every asset read')
        calls,streams=_track_cache_reads(monkeypatch,target,forbidden)
        with pytest.raises(ContractError) as failure:
            materialize(analysis,tmp_path/'views',selections=selections,history=history)
        assert failure.value.code=='budget_exceeded' and calls==[]
        assert streams and all(v.closed for v in streams)


def test_b1_f2_healthy_multichunk_hash_is_bounded_without_source_payload_or_sha_rescan(tmp_path,monkeypatch):
    with native_render(tmp_path/'source',ne=2049,selected_bytes=2*1024*1024) as (analysis,_,_,_):
        selections={'spectral_trace':[slice(None),slice(None)]};destination=tmp_path/'views'
        expected=materialize(analysis,destination,selections=selections)
        target=destination/expected['figures'][0]['path'];size=target.stat().st_size
        assert size>65536
        calls,streams=_track_cache_reads(monkeypatch,target,lambda:None)
        original=h5py.Dataset.__getitem__
        def no_numeric(dataset,key):
            assert '/arrays/' not in dataset.name
            return original(dataset,key)
        monkeypatch.setattr(h5py.Dataset,'__getitem__',no_numeric)
        assert materialize(analysis,destination,selections=selections)==expected
        assert len(calls)>2 and all(0<n<=65536 and count<=n for n,count in calls)
        assert sum(count for _,count in calls)==size
        assert analysis.reader.io_counters['sha_bytes']==0 and all(v.closed for v in streams)
