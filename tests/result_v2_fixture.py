"""Small manufactured transport sources, never a solver or a scientific oracle."""
import copy
import hashlib
import json
import os
import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest

LAYOUT = "94561e874a62ad0749be8a13645ef0142994e074b33d271cbd57807f2cf63600"
IDENTITY = dict(run_id="run", execution_id="exec", point_id="point", attempt="1",
                branch_id="branch", model_sha256="a" * 64, plan_sha256="b" * 64)


def encoded(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                      separators=(",", ":")).encode()


def reference(root, path, schema, media="application/json"):
    raw = (root / path).read_bytes()
    return dict(path=path, bytes=str(len(raw)), sha256=hashlib.sha256(raw).hexdigest(),
                schema=schema, media_type=media)


def put(root, path, value, schema):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(encoded(value))
    return reference(root, path, schema)


def integer(value):
    return dict(kind="integer", type="Int64", decimal=str(value))


def ieee(bits="0000000000000000"):
    return dict(kind="ieee", type="f64", bits=bits)


def symbol(value):
    return dict(kind="symbol", value=value)


def record(ordinal=1, sequence=7, current="4008000000000000"):
    row = {name: ieee() for name in ("r_D", "r_A", "r_K", "r_Σ", "r_λ", "λ",
           "r_PSD", "r_caus", "r_roundoff", "r_Jchange", "r_population", "J",
           "raw_charge", "target_charge", "normalized_charge", "lambda_change")}
    row.update({"ν": integer(ordinal), "J": ieee(current), "witness": None,
                "physical_markers": None})
    key = dict(domain_revision=integer(0), discretization_id="grid", basis_id="basis",
               stage=symbol("scba"), outer_iteration=integer(1), inner_iteration=integer(ordinal))
    fields = dict(key, action=symbol("append"), sequence=integer(sequence), converged=False,
                  status=symbol("running"), quality=symbol("unresolved"),
                  strict_streak=integer(0), diagnostic_streak=integer(0),
                  units=[dict(type="Type_458abebe7784aa53", fields=dict(first=symbol("J"), second="A/m^2"))],
                  missing_reasons=[], row=dict(type="SCBAIteration", fields=row))
    return dict(schema="qcl-negf.scientific-record.v1", contract_set="qcl-negf.results.v2",
                identity=copy.deepcopy(IDENTITY), invocation=dict(source_epoch="epoch"),
                row_key=key, stream_ordinal=str(ordinal), target=None,
                core=dict(type="ScientificHistorySnapshot", fields=fields))


def metadata(handle, *, representation="record_leaf", identity=None):
    group = handle.create_group("metadata")
    group.attrs.update({k: np.bytes_(v) for k,v in dict(schema="qcl-negf-scientific-history-v5", schema_version="5.0",
        contract_set="qcl-negf.results.v2", artifact_role="science.history",
        representation=representation, core_layout_sha256=LAYOUT).items()})
    group.create_dataset("identity_json", data=np.bytes_(encoded(identity or IDENTITY)))
    group.create_dataset("invocation_json", data=np.bytes_('{"source_epoch":"epoch"}'))


def source(root, records=(), *, array_ref=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    pin = dict(head=None, entries="0", records="0", scba_rows="0", outer_rows="0",
               witness_updates="0", last_record="0")
    if records:
        path = root / "records.h5"
        pieces = [encoded(row) for row in records]
        offsets = np.cumsum([0] + [len(row) for row in pieces], dtype=np.uint64)
        with h5py.File(path, "w") as f:
            metadata(f)
            f["records/offsets"] = offsets
            f["records/json_utf8"] = np.frombuffer(b"".join(pieces), dtype=np.uint8)
            f["records/stream_ordinal"] = np.array([int(row["stream_ordinal"]) for row in records], dtype="<u8")
        leaf = reference(root, path.name, "qcl-negf-scientific-history-v5", "application/x-hdf5")
        counts = dict(scba_rows=0, outer_rows=0, witness_updates=0)
        for row in records:
            key = ('witness_updates' if row['core']['fields']['action']['value'] == 'witness_update'
                   else 'scba_rows' if row['core']['fields']['stage']['value'] == 'scba' else 'outer_rows')
            counts[key] += 1
        totals = dict(records=str(len(records)), last_record=str(len(records)), **{k:str(v) for k,v in counts.items()})
        artifacts = []
        if array_ref is not None:
            artifacts.append(dict(array_ref, role='science.history', profile='science', dependencies=[]))
        artifacts.append(dict(leaf, role='science.history', profile='science',
                              dependencies=[] if array_ref is None else [array_ref['path']]))
        page = dict(schema="qcl-negf.scientific-catalog.v1", contract_set="qcl-negf.results.v2",
                    identity=IDENTITY, invocation=dict(source_epoch="epoch"), generation="1",
                    previous=None, artifacts=artifacts,
                    entries=str(len(artifacts)), totals=totals, range=dict(first_record="1", last_record=str(len(records))))
        ref = put(root, "catalog.json", page, page["schema"])
        pin = dict(head=ref, entries=str(len(artifacts)), **totals)
    missing = dict(state="not_measured", reason="manufactured", evidence=None)
    manifest = dict(schema="qcl-negf.result-manifest.v1", contract_set="qcl-negf.results.v2",
        identity=IDENTITY, invocation=dict(source_epoch="epoch"), lineage=dict(parent_state=None),
        provenance=dict(format="qcl-negf.git-graph.v1", revision="a" * 40, components=[dict(
            path="components/QCLNEGF.jl", revision="b" * 40, url="https://example.test/core")]),
        numerical_binding=None, state_id="state-1", state_sequence="1", generation="1", closed=False,
        status=dict(process=dict(state="running", exit_code=None, reason=None),
            convergence=missing, physics=missing, discretization=missing, experimental=missing),
        artifacts=[], checkpoint=None, history=pin)
    ref = put(root, "manifest.json", manifest, manifest["schema"])
    return dict(identity=copy.deepcopy(IDENTITY), state_id="state-1", state_sequence="1", manifest=ref)


@pytest.fixture
def producer_root():
    value = os.environ.get("QCL_RESULT_V2_FIXTURE_ROOT")
    if not value:
        pytest.skip("explicit constructor-only producer fixture root not supplied")
    return Path(value)


def producer_state(root, directory, *, recovery=False):
    pointer = json.loads((root / directory / ("recovery/current.json" if recovery else "current.json")).read_text())
    manifest = json.loads((root / pointer["manifest"]["path"]).read_text())
    return {key: copy.deepcopy(manifest[key]) for key in ("identity", "state_id", "state_sequence")} | {"manifest": pointer["manifest"]}


def append(root, prior, row):
    manifest = json.loads((root / prior['manifest']['path']).read_text())
    n = int(manifest['generation']) + 1
    raw = encoded(row)
    name = f'records-{n}.h5'
    with h5py.File(root / name, 'w') as f:
        metadata(f)
        f['records/offsets'] = np.array([0, len(raw)], dtype='<u8')
        f['records/json_utf8'] = np.frombuffer(raw, dtype='u1')
        f['records/stream_ordinal'] = np.array([int(row['stream_ordinal'])], dtype='<u8')
    leaf = reference(root, name, 'qcl-negf-scientific-history-v5', 'application/x-hdf5')
    totals = {key: manifest['history'][key] for key in ('records','scba_rows','outer_rows','witness_updates','last_record')}
    for key in ('records','scba_rows','last_record'):
        totals[key] = str(int(totals[key])+1)
    page = dict(schema='qcl-negf.scientific-catalog.v1', contract_set='qcl-negf.results.v2',
        identity=IDENTITY, invocation=dict(source_epoch='epoch'), generation=str(n), previous=manifest['history'],
        artifacts=[dict(leaf, role='science.history', profile='science', dependencies=[])],
        entries=str(int(manifest['history']['entries'])+1), totals=totals,
        range=dict(first_record=row['stream_ordinal'], last_record=row['stream_ordinal']))
    page_ref = put(root, f'catalog-{n}.json', page, page['schema'])
    manifest.update(generation=str(n), state_sequence=str(n), state_id=f'state-{n}',
                    history=dict(head=page_ref, entries=page['entries'], **totals))
    manifest_ref = put(root, f'manifest-{n}.json', manifest, manifest['schema'])
    return {key: manifest[key] for key in ('identity','state_id','state_sequence')} | dict(manifest=manifest_ref)


def pointer(root, state_ref):
    manifest = json.loads((root / state_ref['manifest']['path']).read_text())
    path = 'executions/exec/point/attempt-1/current.json'
    put(root, path, dict(schema='qcl-negf.result-pointer.v1', contract_set='qcl-negf.results.v2',
        identity=IDENTITY, generation=manifest['generation'], manifest=state_ref['manifest']),
        'qcl-negf.result-pointer.v1')


def array_source(root, count=32, *, value=0.0):
    """Bounded constructor writes; density shape is manufactured, not physics."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with h5py.File(root / 'arrays.h5', 'w') as f:
        metadata(f, representation='array_payload')
        dataset = f.create_dataset('arrays/a00000001', shape=(count,), dtype='<f8', chunks=(min(max(count,1),8192),))
        for first in range(0, count, 8192):
            dataset[first:first+8192] = np.full(min(count-first,8192), value, dtype='<f8')
    owner = reference(root, 'arrays.h5', 'qcl-negf-scientific-history-v5', 'application/x-hdf5')
    array = dict(kind='array', eltype='Float64', order='F', shape=[str(count)], dataset='/arrays/a00000001', file=owner)
    row = record()
    row['row_key']['stage'] = symbol('outer')
    row['row_key']['inner_iteration'] = None
    core = row['core']['fields']
    core['stage'], core['inner_iteration'] = symbol('outer'), None
    values = {name:ieee() for name in ('r_P','r_U','r_n','r_neutral','r_J','r_Jchange','r_population','ζ','J')}
    values.update({'μ':integer(1), 'populations':array, 'density':copy.deepcopy(array)})
    core['row'] = dict(type='OuterIteration', fields=values)
    return source(root, [row], array_ref=owner)


def copy_constructor_closure(root, destination, state):
    """Copy only explicit refs of a small positive manufactured producer source."""
    destination.mkdir()
    queue, seen, total = [state['manifest']], {}, 0
    def refs(value):
        if isinstance(value, dict):
            if set(('path','bytes','sha256','media_type','schema')) <= value.keys():
                yield {key:value[key] for key in ('path','bytes','sha256','media_type','schema')}
            for item in value.values():
                yield from refs(item)
        elif isinstance(value, list):
            for item in value:
                yield from refs(item)
    while queue:
        reference = queue.pop()
        path = reference['path']
        assert not Path(path).is_absolute() and all(part not in ('','.','..') for part in path.split('/'))
        if path in seen:
            assert seen[path] == reference
            continue
        seen[path] = reference
        total += int(reference['bytes'])
        assert total <= 16*1024*1024, 'only tiny constructor fixtures may use this helper'
        source_path, target = root / path, destination / path
        target.parent.mkdir(parents=True, exist_ok=True)
        with source_path.open('rb') as original, target.open('xb') as output:
            shutil.copyfileobj(original, output, length=65536)
        assert target.stat().st_size == int(reference['bytes'])
        assert hashlib.sha256(target.read_bytes()).hexdigest() == reference['sha256']
        if reference['media_type'] == 'application/json':
            queue.extend(refs(json.loads(target.read_text())))
    return destination


def export_source(root, state, destination, **kwargs):
    """Tests still select the complete explicit StateRef and canonical identity."""
    from qcl_negf_results.state import StateReader
    from qcl_negf_results.export import export_snapshot
    with StateReader(root, state_ref=state, expected_identity=state['identity']) as reader:
        return export_snapshot(reader, destination, profile=kwargs.pop('profile', 'science'),
            byte_budget=kwargs.pop('byte_budget', 16*1024*1024),
            reserve_bytes=kwargs.pop('reserve_bytes', 0), **kwargs)
