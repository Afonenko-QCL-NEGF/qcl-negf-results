"""Stored final observations from one verified native5 source. No derive or solve."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from qcl_negf_contracts.messages import ContractError
from qcl_negf_contracts.results import validate_final_diagnostics
from .state import (StateReader, Document, ArrayDescription, ArrayBlock, _ref, _metadata_weight,
                    _context_copy, _context_contract, _context_before, _context_hold,
                    _context_workspace, _context_weight, _context_arena)
from .native import final_diagnostics_metadata, validate_final_diagnostics_array, _FINAL_ARRAYS


@dataclass(frozen=True)
class FinalDiagnosticsDescription:
    source: Document
    invocation: Document
    artifact: Document
    checkpoint: Document
    declaration: Document
    integrity: Document
    io_counters: Mapping[str, int | None]


class FinalDiagnosticsReader:
    """Borrow a selected StateReader; it owns all handles and physical budgets."""
    def __init__(self, reader: StateReader, *, artifact_ref: Document):
        if not isinstance(reader,StateReader):
            raise TypeError('explicit selected StateReader required')
        if not isinstance(artifact_ref,Mapping) or set(artifact_ref) != {'path','bytes','sha256','media_type','schema'}:
            raise ContractError('exact analysis FileRef required', 'corrupt_result')
        self._reader, self._reference = reader, _context_copy(dict(artifact_ref))
        self._source, self._closed = _context_copy(reader.source.state), False

    def __enter__(self):
        return self

    def __exit__(self,*args):
        self.close()

    def close(self):
        self._closed = True

    def _load(self):
        live=[]
        with _context_hold(live,self._source,self._reference):
            if self._closed:
                raise ValueError('final diagnostics reader is closed')
            reader = self._reader
            if reader.source.state != self._source:
                raise ContractError('selected final diagnostics source changed', 'source_changed')
            owner = next((row for row in reader._manifest['artifacts'] if row['path']==self._reference['path']),None)
            if (owner is None or _ref(owner)!=self._reference or owner['role']!='physics.analysis'
                    or owner['schema']!='qcl-negf-physics-analysis-v5' or owner['profile']!='science'):
                raise ContractError('selected exact analysis FileRef absent or differs', 'corrupt_result')
            checkpoint = reader._load_checkpoint()
            handle = reader._open_hdf(owner,identity=reader._manifest['identity'],
                invocation=reader._manifest['invocation'],representation='final_diagnostics')
            native = final_diagnostics_metadata(handle,identity=reader._manifest['identity'],
                invocation=reader._manifest['invocation'],maximum_bytes=reader.limits.maximum_metadata_bytes)
            value = native['declaration']
            live.append(native)
            if _metadata_weight((reader._manifest,checkpoint,value)) > reader.limits.maximum_metadata_bytes:
                raise ContractError('aggregate retained analysis context exceeds metadata limit','budget_exceeded')
            _context_contract('validate_final_diagnostics',value,checkpoint=checkpoint,manifest=reader._manifest,artifact_ref=self._reference)
            value = validate_final_diagnostics(value,checkpoint=checkpoint,
                manifest=reader._manifest,artifact_ref=self._reference)
            live[0]=value;del native
            data = value['positivity']['data']
            if data is not None:
                for name in _FINAL_ARRAYS:
                    validate_final_diagnostics_array(handle,data['integrals'][name])
            return value, handle

    def describe(self) -> FinalDiagnosticsDescription:
        with self._reader._consumer_operation():
            value, _ = self._load()
            _context_before('diagnostics description',extra=8192,roots=(value,),copies=(self._source,value['invocation'],self._reference,value['checkpoint']))
            with _context_hold(value), _context_workspace('diagnostics copy arguments',
                    8192+sum(_context_weight(v) for v in (self._source,value['invocation'],self._reference,value['checkpoint'])) if _context_arena.get() is not None else 0):
                return FinalDiagnosticsDescription(_context_copy(self._source),_context_copy(value['invocation']),
                    _context_copy(self._reference),_context_copy(value['checkpoint']),value,
                    dict(manifest_verified=True,analysis_file_verified=True,checkpoint_declaration_verified=True,
                        native_metadata_verified=True,selected_array_metadata_verified=value['positivity']['data'] is not None,
                        full_core_payloads_scope='not_assessed_by_this_operation',scope='selected_bytes'),
                    self._reader.io_counters)

    def _array(self,name):
        value, handle = self._load()
        if name not in _FINAL_ARRAYS:
            raise ValueError('unknown final diagnostics integral')
        if value['positivity']['data'] is None:
            raise ContractError(value['positivity']['reason'],'capability_unavailable')
        descriptor = value['positivity']['data']['integrals'][name]
        return descriptor, validate_final_diagnostics_array(handle,descriptor)

    def describe_integral(self,name: str) -> ArrayDescription:
        with self._reader._consumer_operation():
            descriptor, dataset = self._array(name)
            return ArrayDescription(_context_copy(descriptor),dataset.shape,dataset.shape,
                descriptor['eltype'],dataset.dtype.itemsize,'F',_context_copy(self._source))

    def read_integral(self,name: str,selection: tuple[int | slice,...],*,maximum_bytes: int=1024*1024) -> ArrayBlock:
        with self._reader._consumer_operation():
            descriptor, dataset = self._array(name)
            return self._reader._read_selected_array(descriptor,dataset,selection,maximum_bytes=maximum_bytes)


def describe_final_diagnostics(reader: StateReader, *, artifact_ref: Document) -> FinalDiagnosticsDescription:
    with FinalDiagnosticsReader(reader,artifact_ref=artifact_ref) as view:
        return view.describe()
