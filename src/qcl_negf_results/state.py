"""One result-v2 source, canonical cursors and bounded, verified native5 reads.

Metadata import and selected byte verification are separate. Nothing here runs a
solver or converts process completion into scientific acceptance.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, fields, is_dataclass
import copy
import hashlib
import io
import math
import os
import sys
from pathlib import Path
from threading import Lock
from typing import Any, Mapping

import h5py
import numpy as np
from qcl_negf_contracts import schema_validator
from qcl_negf_contracts.artifacts import relative_path
from qcl_negf_contracts.messages import ContractError
from qcl_negf_contracts.results import (validate_manifest, validate_pointer, validate_catalog_page,
    validate_checkpoint_declaration, validate_scientific_record, validate_cursor)
from .commits import decode_json, json_bytes

Document = Mapping[str, Any]
_REF_KEYS = ('path', 'bytes', 'sha256', 'media_type', 'schema')
_TOTALS = ('records', 'scba_rows', 'outer_rows', 'witness_updates', 'last_record')


def _fail(message, code='corrupt_result'):
    raise ContractError(message, code)


def _ref(value):
    return {key: value[key] for key in _REF_KEYS}


def _schema_check(validator, value):
    from jsonschema import ValidationError
    try:
        if _context_arena.get() is not None:
            _context_before('schema validation',extra=_context_weight(value)+4*_context_json_size(value)+4096,roots=(value,))
        validator.validate(value)
    except ValidationError as error:
        raise ContractError('invalid result declaration: ' + error.message, 'corrupt_result') from error


def _arrays(value):
    if isinstance(value, dict):
        if value.get('kind') == 'array' and 'file' in value:
            yield value
        for item in value.values():
            yield from _arrays(item)
    elif isinstance(value, list):
        for item in value:
            yield from _arrays(item)


def _metadata_weight(value):
    """Conservative retained Python-object estimate; not a process RSS limit."""
    pending, seen, total = [value], set(), 0
    while pending:
        item = pending.pop()
        if id(item) in seen:
            continue
        seen.add(id(item))
        total += sys.getsizeof(item)
        if isinstance(item, dict):
            pending.extend(item.keys())
            pending.extend(item.values())
        elif isinstance(item, (tuple, list)):
            pending.extend(item)
    return total



_context_arena = ContextVar('qcl_results_metadata_context_arena', default=None)
_CONTEXT_MAXIMUM = 128 * 1024 * 1024
_CONTEXT_VALIDATION = 112 * 1024 * 1024
_CONTEXT_PROFILE_IO = 6 * 1024 * 1024
_context_call_lock = Lock()
_CONTEXT_HEADER = 65536


_VALIDATION_PACKAGES = {
    'qcl_negf_contracts': (48,
    885024,
    '06177da7369959b136666bf827e2a16120c2bc78cfe06bcbab80479c0a02d702'),
    'jsonschema': (38,
    463467,
    '96a04a34eea376cc3afe479fd5115e5648679d15a22028bacfec62eaf8f08766'),
    'referencing': (13,
    109224,
    '693bb9caed2d635557d0ebce81ffc15ac2e5041371e68a18880139d22ed257e3'),
    'rpds': (2,
    971387,
    '8e16fe10328aac190e2984b00cad00c13ad42c7c902a77d9eed39540cd667928'),
    'attr': (13,
    191766,
    '17b91d3582e2b54702c3a96e07218e7543585ee7c01890198ef004d635dd8314'),
    'attrs': (6,
    1557,
    '7222d2a29028426c7bbcda934d1dfa8a80d51506cca2de28071477a38dea65d9'),
    'jsonschema_specifications': (24,
    39986,
    'd72dda1198d96400d7b31b68a69ff4f2c999a54a768ee44cd5ebdbb486521516')}
_VALIDATION_SCHEMAS = {
    'http://json-schema.org/draft-03/schema': 'f0fd1d2c48f2b39dccd425bdca913be1de4bf3ecddf292478ee8f26175271140',
    'http://json-schema.org/draft-04/schema': 'c8aa3d8de08d4e2048ed2d5a223c31f5dcf4dcbc2adcea628e0a522f0e1ba44a',
    'http://json-schema.org/draft-06/schema': '07d8be64c0c0d2ad7fd68509ea28ba3e5243703655ff2e5a1970cfc6abc090ae',
    'http://json-schema.org/draft-07/schema': '1ac84c2f322d91e3781863e6421917fb5c33ca33761f7bb7a4445dd22293ce01',
    'https://json-schema.org/draft/2019-09/meta/applicator': 'b7b34b2551fbbf139502ef1a5ed134b50214d6d15647e8498d8f33857fc7f8d8',
    'https://json-schema.org/draft/2019-09/meta/content': '4dbfe51279467891d13371cf8d2fe5a9c375c5779f8c6852df8fca95157efb2e',
    'https://json-schema.org/draft/2019-09/meta/core': '02cc9b0090db51b5b7ed2ba3b6c10cc8eb2b563a3f1c09aaa21cbc65a52494ad',
    'https://json-schema.org/draft/2019-09/meta/format': 'e6c874b36a3a0522ae8a059e139ea9232f4375f7835d4788322862723e4ebc63',
    'https://json-schema.org/draft/2019-09/meta/meta-data': '8c685e5129ef0e49f6af2881c26419b3cab9f2326dbba6e66971f23d642b7964',
    'https://json-schema.org/draft/2019-09/meta/validation': 'e4c01a8d9ca3ac0379ace5af307f3ed911df6257b7a98f9058ef89ffb3437582',
    'https://json-schema.org/draft/2019-09/schema': '2b629425923759fc850153f5629d42947365cd4f29775fe9590e1e220b5eef90',
    'https://json-schema.org/draft/2020-12/meta/applicator': '46a5141432ac51fff467e6c2230c19cba0b07d6a492c0fb062a1a852c370eed7',
    'https://json-schema.org/draft/2020-12/meta/content': 'adcc808be6d10c12a2fcdf3123ecc4b56b18df58fc86001f2740ddd10fdeedfd',
    'https://json-schema.org/draft/2020-12/meta/core': 'fc882f9fe8343beacb4b47b4a65441a9bfe2a15f279c66ceb6ddd65da7586edd',
    'https://json-schema.org/draft/2020-12/meta/format-annotation': 'f35165bc538974536af0c491819172f68e32e304e677742e67fe12f008731e60',
    'https://json-schema.org/draft/2020-12/meta/format-assertion': 'dd93ef25f948d92a237f15019808465bbd3fe58e2ed45fe9548afc2319eb1a45',
    'https://json-schema.org/draft/2020-12/meta/meta-data': '5b3b96d59a4a2f16fc72a0dcaf758aea40aae866a61ac9d0d71dd9d6a8603115',
    'https://json-schema.org/draft/2020-12/meta/unevaluated': '61f01418e98cdbb0ad0fbfbd593780bdadb7fb745a7adff646d26c103bc6d166',
    'https://json-schema.org/draft/2020-12/meta/validation': '34fba27426a2fe175b459326a4013434775f06dd33d1209cab9028decae1e240',
    'https://json-schema.org/draft/2020-12/schema': 'c1cacf82bea665da4dbaf58eb341b7af39420d4c9248a012aba3436ef1f802bb',
    'https://qcl-negf.invalid/schema/artifact-manifest.schema.json': '14f230fad5419dff856114ee839b35a9be3a9738b382d7fb88706a7971fc707b',
    'https://qcl-negf.invalid/schema/checkpoint-declaration.schema.json': '532c4a457eb63dd6aeb0089a3dc5c080307160ce383f55f05ce92759311b8abc',
    'https://qcl-negf.invalid/schema/core-state.schema.json': '56967f7039dac2692dfbcf35203d8412058577c30a0c1e4294fccbab2d957eb0',
    'https://qcl-negf.invalid/schema/export-receipt.schema.json': 'a4f84dda93cbd05e70f34ad74f002857fb8f7713d7cf73c2ac4effed60a3e087',
    'https://qcl-negf.invalid/schema/final-diagnostics.schema.json': 'de147df663138580b7af1a75f9b689abfe9f527056dd65f727f166a55f3f082f',
    'https://qcl-negf.invalid/schema/final-fixed-point-audit.schema.json': 'b4020be18f7cf2aa0d9436e0f4705867442a41e354a76c4fd526be0513ec5f00',
    'https://qcl-negf.invalid/schema/final-observations.schema.json': 'ee7dce5448500b1e3d4591875383ea76d1207013da7b4dd62c21062e5303fb03',
    'https://qcl-negf.invalid/schema/model-help.schema.json': 'abf98022602de243e0d7d5bb42ea3e8a80b7a12b4fddf9cc3ebdd3fc8592e8ac',
    'https://qcl-negf.invalid/schema/nested-sweep-plan.schema.json': '1fdfd5da1d9c3e2b1c99943ed40111046e12a065ac427a1d03ed7bac268c432b',
    'https://qcl-negf.invalid/schema/output-policy.schema.json': 'b5bf6d025994d351f3b99f2d036dc9a24438880bb0fc3ddab20b0d618aeee110',
    'https://qcl-negf.invalid/schema/outward-result.schema.json': '2bc4faef56a2d1812880cb4e58ad0d389e6bc6c3aa05fa0348a27246658c080c',
    'https://qcl-negf.invalid/schema/physics-snapshot-metadata.schema.json': '782dbb8ddac0e5c7372dc29392e9d023a6bb6357b63a3412173742632054995b',
    'https://qcl-negf.invalid/schema/physics-snapshot.schema.json': '9ccd7264032c0ae96b0933307e3367ee2663394050bacdb13b63e445f6327012',
    'https://qcl-negf.invalid/schema/point-result.schema.json': '1bfdd7da366e89b16060bd0ba99a77eff1f0fd0baa304b90fec975b596f2d4c5',
    'https://qcl-negf.invalid/schema/point-status.schema.json': 'f40939a5aa1f2cde4278e09bba28a8ebd115d34ecd1af27a03f3acb3be8a052a',
    'https://qcl-negf.invalid/schema/positivity-diagnostics.schema.json': '0974668de7e169d81c906e2515a3b1fc698e3dbb9678c281880b649675919e39',
    'https://qcl-negf.invalid/schema/producer-phase-segment.schema.json': '507d21392c272a38e016e5e2b4575a6c036735d74663e97acf63c6914e072313',
    'https://qcl-negf.invalid/schema/producer-phase-source.schema.json': '41edc85ef5808720164654ce1f454c8d958b9b8a69e516c948440d61e9e2e2fe',
    'https://qcl-negf.invalid/schema/progress.schema.json': '27f4511bf30dfb944af13ac7ec7aba4103bd88da4fbd689baae86aa6813a83d4',
    'https://qcl-negf.invalid/schema/resolved-configuration.schema.json': '93e1965c64ce501b649bd98b64a54e26982e5a6522072924e245113ef2edb5fb',
    'https://qcl-negf.invalid/schema/resource-estimate.schema.json': '33e1b4ae88d235a418330317ebdc754cceac43de1d3b37851decb44a1a465491',
    'https://qcl-negf.invalid/schema/result-common.schema.json': 'ff58b3aae1ee0d8ed7b8dbbc09bc50f8cf74ee6c2c5f9c3eebf318ab417f4eb0',
    'https://qcl-negf.invalid/schema/result-cursor.schema.json': '399899f09779ff97899e9c0fc2b2d487f7fb3e348cd8c1f813ca0631233a7db0',
    'https://qcl-negf.invalid/schema/result-manifest.schema.json': '12e46127a24fa58292f5f666d3f8b66dece25979ee9870e317dc775e974b47e6',
    'https://qcl-negf.invalid/schema/result-pointer.schema.json': '8e4ef83eda56bb16afb2f577eff5eeedcb0087ae0a53bd5b9aea2571b46247cf',
    'https://qcl-negf.invalid/schema/run-result.schema.json': '87e374ba80e31ddb7c7cd8af9004b97cb3fe3a584f2cdf5e06f99d2aec6ac560',
    'https://qcl-negf.invalid/schema/run.schema.json': 'ae5a87ae66a4b8adfc9fa5d082999d7f884b17388053ec70f4a4157122982a92',
    'https://qcl-negf.invalid/schema/runtime-event.schema.json': 'ad12e403a157ff42494b66b7a6314908e7b48738d58ea63d7fce42105faaf9b3',
    'https://qcl-negf.invalid/schema/scientific-catalog.schema.json': 'f9b0d6dba1e6d565afc342cff74da785b527b78f581b18945c1e8e6e3ee56125',
    'https://qcl-negf.invalid/schema/scientific-definition.schema.json': 'b12a8b2273d6c4630fcccb850ee2bf57b3585297fef0159447b4a57d0824e72b',
    'https://qcl-negf.invalid/schema/scientific-plan.schema.json': '1fa24cbd3a0a37da1e5fdc20de3e05307fd9777d8156819f28850212ab1bbf5a',
    'https://qcl-negf.invalid/schema/scientific-record.schema.json': 'bf6e02c8cd0d46f4221d5bd540a31adc5ea3e45f841011b5642d2ce7e669a62b',
    'https://qcl-negf.invalid/schema/scientific-series-result.schema.json': '644b068e5712efb85681f0865d4ce4a92c842cee0fb9194993a090335ca341cf',
    'https://qcl-negf.invalid/schema/scientific-worker-result.schema.json': 'eee0df0691a1caa1e310c0ceec8aab9033774dfe7c2a093e0d03fa5a0b60243a',
    'https://qcl-negf.invalid/schema/telemetry-envelope.schema.json': '803f24eb5eeb2c07a69b30ad2474215a860e764ca165cf89791e57a660fab850',
    'https://qcl-negf.invalid/schema/validation-diagnostics.schema.json': '35ecb1710da9f36cd510fc147c4974c92aef56a5b897ad1fac0cc31573268aaa'}
_VALIDATION_LAYOUT = '6515d7d00f248a16ff4d439ee66a2c757b5507c4c6cd29b0b6c4cdb481afe5ad'
_VALIDATION_ROW_FIELDS = ('domain_revision', 'discretization_id', 'basis_id', 'stage', 'outer_iteration', 'inner_iteration')
# Shared Contracts.results cache: fixed results + telemetry public callsites.
# All 15 cached bundles plus 3 direct bundles fit the qualified 112 MiB W reserve.
_VALIDATION_KINDS = ('checkpoint-declaration.schema.json',
    'core-state.schema.json',
    'final-diagnostics.schema.json',
    'final-observations.schema.json',
    'nested-sweep-plan.schema.json',
    'producer-phase-segment.schema.json',
    'producer-phase-source.schema.json',
    'result-common.schema.json',
    'result-cursor.schema.json',
    'result-manifest.schema.json',
    'result-pointer.schema.json',
    'scientific-catalog.schema.json',
    'scientific-plan.schema.json',
    'scientific-record.schema.json',
    'telemetry-envelope.schema.json')


_VALIDATION_LIBRARY_CODES = {'keywords': {'$dynamicRef': '7795e960ce3eef69f742a40e5d450a60726e306c29eeba0b42eec2a49a043a3d',
              '$ref': '178d49316272bceb54e01a6ca22aecfc1a31d0285d70bf70e32387037da9cebc',
              'additionalProperties': '56799d21076c67f5dcb61af5660bf2eb7fd77635ea12acbfbae81fc897f15df3',
              'allOf': '5c6eae8b526615b93207d1d0206bd8d95573237d3353c59f6e5c928f0b73a254',
              'anyOf': 'c5750f3f5da1cae509ce5597f6ab06b8f4f2d4a08d82e645538b65ccdf000a12',
              'const': '0338c7dd820dd1202ff83010a4733f5b557879bffd2a92301a92a9b0312ec972',
              'contains': '45bf300a4b587e39efa54283b9379ed8da8d6db00f11c53edf8fb65a148b8f6e',
              'dependentRequired': '3205af1fcd5c23bbc6e6f2174778723359ad1935c98210d2a8a701b3cfed64f5',
              'dependentSchemas': 'd875ea9b1b8ba9f105afc6c059d6873de66468ba1e77990c1f72cc0850067cd5',
              'enum': 'ac9ba205336f62762cdb224fbb41540c4818b22944cd37d30fe7aa98482f0fe4',
              'exclusiveMaximum': '52fa0a0faf21b8bc801fe24e81e957628773c70d7cdc4f2adba14e4779a38bd1',
              'exclusiveMinimum': '28c2ba2a3e9274f6677f61b3fd4e67502c0543c9817eaed40ece60659b1fdc1e',
              'format': '2ffdf4c6f1a01432a02c246ba6cc6eb2ba2664e2e6a0b686570ee9e251baf546',
              'if': '32cfba966fcdb25e451b966243992f6b4622e2e2dee39f756b225c6c3c9e5b49',
              'items': '731398b66a3815e5b899511cdcc29789d91602b2ed949a44b4a6aa10227e0cc2',
              'maxItems': 'd94f6cdd0ff6d187ec2932f6648ea834a0f57936713c6f7fbe8ab64222984fc6',
              'maxLength': 'a645a2da522e9d96bceef7223e9ba22906f98f22303b2c9aac990d6183c769d2',
              'maxProperties': '3e0e11d1a82f10dc2215f6cf7b121046f1f734a03ca87da7d567e5574d60a12c',
              'maximum': '6ed245f522c6104eb541274489594a24ed583774263f9573d625617ae262d241',
              'minItems': '9f3eb3a5f6e2758c6b54f0da80454f26e7a83642daa2bdd6da94a4e9772952e7',
              'minLength': 'd9407d2486af25c102664a76fc99f169047d7e98250cd7f38fd1d016866c4034',
              'minProperties': '99f6f622bc36cdda3eb7993c3221dd7582fef184f16933ed69675769ce723547',
              'minimum': '63096701547f94402322973701e16201ea7daead732647e4bdbcbe4856a26e2a',
              'multipleOf': '134f1d4f06635409136beb6ed639eca4a2a19feed2486285704aabdc47ae59b5',
              'not': 'b25117ad9aab1e33d08b4cb32c5438837aae1774bcaad7ff673d0946948af0a4',
              'oneOf': '39f2d22952509711259f50aa4b58574b616a94fe93c21a0a83719e23b46854f9',
              'pattern': '2c96d077837f8300c922b565376c7e4572f92dfffa11f8aa3c9f723baf336ad2',
              'patternProperties': '68b9632e0993a374deea17ccd88ca706784f43b76d01de93792a95b7a1344c16',
              'prefixItems': 'a692869be6ab8c86e8f7d50aa8c091bbc2ff400ff7f3492ab75efca61833b82e',
              'properties': '422b8fee150959324cc516b9cb187d215b9b98093d33cb9b233cf48d3bbbf764',
              'propertyNames': '0cac1db3eed7a14a7fe3bf8333e3231daf082954afd400f2d8f2bdea0190748e',
              'required': 'ab4e05fe5cf3301bfa7ecb0a50ec7169ef770d50addd8b9d8eb94086efef2bfb',
              'type': '35595f19594a97acea18fc1d2460f12c2c3188d20c96ed93bd91bae4e073b057',
              'unevaluatedItems': '6c0b6313df5905721ae8e76c88c5a86fa3da6d7c5f889ae95aae4bcad38a0398',
              'unevaluatedProperties': '1e5043201eb9de41e6170133a44c1c610e77f952253786ff9c5e15c9469ac644',
              'uniqueItems': '5f6fa939889501276e528151dbd7e8d8a595425b5b4e6d5addfe1d5470e9b350'},
 'methods': {'Draft.ID_OF': '1c0875f974c14f362017e938a9d5d9ebcfcd2137f45621dc8f961452dbe9b9d0',
             'Draft.__attrs_post_init__': 'a895d887a2c13a1259f051103756a4a2d2c689f12341262c1e212f2ca24b3e6e',
             'Draft.__init__': '081365938dc329e8cf906e38179f2944904a554fb10e79c0e3c77186ad2a209c',
             'Draft.descend': 'a8e1942f11960e2e2084327962d8c6f47bae8e796ba739e35ececf92dd24ff22',
             'Draft.evolve': '84b199d42cd24ff2f6763ec207273ab1282ac297b5929d5fe9eda558db330570',
             'Draft.iter_errors': '93f38bed592727573719194d2d16447e46fb802aae29e817b0b56874d5b7ad75',
             'Draft.validate': 'eb90ca04adf3e852b0103c930469815892d4842c801974482f0fd56c07f4d821',
             'Registry.crawl': '7a5c16c06f9679ff7134619450b631ba38ad8965d9df2c966039a606a085ddbb',
             'Registry.with_resources': 'd138267fc02dcdc7baa39e38282ee39bc3ba5353af10381e4ab1fae39116a297',
             'Resolver._evolve': 'c0d8e743377e801338e8132bb3213da29b4705072715138a88ed97dfe673d2c7',
             'Resolver.lookup': 'ba0fa8c99da03d9d27ab62b7be902e6df65a463f1ff4a6f958e7181d36edf4b4',
             'Resource.from_contents': '8999f405d40f54a2c7d9369dfb6c0bda55c87e91e5437b8d4bcb5bc62495784f',
             'retrieve': '8f88b24d3f153448e93b1a79e208b62293d2af346b9a75411d9fcba2a9227e88'},
 'type_checkers': {'array': 'b5369251118c95bfdb6dab28a0d1e267eb791a29af0d41814b6a1ad2a76ff0a1',
                   'boolean': '3595ebb702256f85ee4bdc61b9f7b24580b545cadb900268228cd69d2074b9d4',
                   'integer': 'df999c0e79f7b274d52fd039130810b1421551e30d09b61cba28ff502c923554',
                   'null': '80a83c38c11ab69fe958f76f6cd816e10dd7d16c72eae0644dc46ee2a4011a0d',
                   'number': 'a53fc732350de60f889d867a291c0ec880012d936629033f0bcd4df47198ca9e',
                   'object': '656991e97bb91e3f3a43aebe83aa4d95c8a9f2f67f7aadb91678a6cd0397dbc0',
                   'string': '8ca1d787fba3367550e611cfe3c68e99176f433f1e1bd12cdb4b10f9fb1da635'}}


def _validation_profile(arena, previous=None):
    """Qualify one pinned offline implementation and its actual mutable caches.

    CPython's known unbounded lru-cache layout is inspected, never cleared or
    prewarmed. Unknown source/layout/content fails before the owner factory.
    All inspection/encoding/hash workspace is prepaid in validation_reservation.
    """
    import functools
    import gc
    import importlib
    import importlib.util
    import json
    import marshal
    import operator
    import types

    def require(condition):
        if not condition:_fail('unknown or changed validation profile', 'incompatible_contract')

    def read(stream,count):
        if count>arena.profile_io_remaining:
            _fail('validation profile source I/O allowance exhausted','budget_exceeded')
        arena.profile_io_remaining-=count  # Request prepaid even on EOF/failure.
        raw=stream.read(count);arena.profile_bytes_read+=len(raw)
        return raw

    @contextmanager
    def opened(path):
        stream=path.open('rb',buffering=0);primary=None
        try:yield stream
        except BaseException as error:primary=error;raise
        finally:
            try:stream.close()
            except BaseException as error:
                if primary is None:raise
                primary.add_note('validation profile cleanup also raised '+type(error).__name__)

    require(sys.implementation.name=='cpython' and sys.version_info[:3]==(3,14,7))
    modules={}
    for name,(count,size,expected) in _VALIDATION_PACKAGES.items():
        profiles=((count,size,expected),)
        if name=='rpds':
            profiles+=((2,981691,'8f378f9a10adbb21d170f679fc52b0c9d36766a1540f297a03ac18d1060f041a'),)
            size=max(profile[1] for profile in profiles)
        spec=importlib.util.find_spec(name)
        require(spec is not None and spec.origin is not None)
        directory=Path(spec.origin).resolve().parent
        records=[];delivered=0
        for path in directory.rglob('*'):
            if '__pycache__' in path.parts or not path.is_file():continue
            if name!='jsonschema_specifications' and path.suffix not in ('.py','.json','.so'):continue
            require(not path.is_symlink() and len(records)<count)
            digest=hashlib.sha256()
            length=path.stat().st_size
            require(length<=size-delivered)
            with opened(path) as stream:
                remaining=length
                while remaining:
                    requested=min(65536,remaining);chunk=read(stream,requested)
                    require(len(chunk)==requested)
                    remaining-=len(chunk);delivered+=len(chunk);digest.update(chunk)
                require(not read(stream,1))  # Growth/replace cannot extend the admitted read.
            records.append((str(path.relative_to(directory)),digest.hexdigest()))
        records.sort()
        actual=hashlib.sha256(json.dumps(records,separators=(',',':')).encode()).hexdigest()
        require((len(records),delivered,actual) in profiles)
        modules[name]=importlib.import_module(name)
        require(type(modules[name]) is types.ModuleType)

    contracts=modules['qcl_negf_contracts']
    from qcl_negf_contracts import results
    from jsonschema import Draft202012Validator
    from jsonschema._types import TypeChecker
    from referencing import Registry,Resource
    from referencing._core import Resolver,_fail_to_retrieve
    from jsonschema_specifications import REGISTRY
    require(schema_validator is contracts.schema_validator is results.schema_validator)
    def code_digest(function):
        if type(function) is types.MethodType:function=function.__func__
        require(type(function) is types.FunctionType)
        def normalized(code):
            return code.replace(co_filename='',co_consts=tuple(normalized(v) if type(v) is types.CodeType else v for v in code.co_consts))
        return hashlib.sha256(marshal.dumps(normalized(function.__code__))).hexdigest()
    # These are the factory's immediate resource/JSON bindings, not a sandbox
    # for arbitrary mutations of trusted CPython/stdlib/import machinery. Code
    # fingerprints prevent replacing both aliases with the same foreign target.
    import importlib.resources as resources
    import importlib.resources._common as resource_common
    require(type(resources) is types.ModuleType and type(resource_common) is types.ModuleType
            and type(json) is types.ModuleType and json is sys.modules['json'])
    for module in (contracts,results):
        require(type(module.__package__) is str and module.__package__=='qcl_negf_contracts' and module.files is resources.files
                and module.json is json)
    resource_files=resources.files
    require(resource_files is resource_common.files
            and type(resource_files) is types.FunctionType
            and resource_files.__globals__ is resource_common.__dict__
            and code_digest(resource_files)=='ff64631fdb0a542e5f3ccb64be3557cb90f049d567124e81112cff361b219fe5')
    closure=resource_files.__closure__
    require(type(closure) is tuple and len(closure)==2)
    target,sentinel=(cell.cell_contents for cell in closure)
    require(type(target) is types.FunctionType and target is resource_files.__wrapped__
            and target.__globals__ is resource_common.__dict__ and target.__closure__ is None
            and type(target.__defaults__) is tuple and len(target.__defaults__)==1
            and target.__defaults__[0] is None and target.__kwdefaults__ is None
            and code_digest(target)=='40ff2d5a513b2c72d793682d363f834fabb196b2800313a542609b428673b3a2'
            and type(sentinel) is object and type(resource_files.__defaults__) is tuple
            and len(resource_files.__defaults__)==2
            and all(value is sentinel for value in resource_files.__defaults__)
            and resource_files.__kwdefaults__ is None)
    loads=json.loads
    require(type(loads) is types.FunctionType and loads.__globals__ is json.__dict__
            and loads.__closure__ is None and loads.__defaults__ is None
            and type(loads.__kwdefaults__) is dict and len(loads.__kwdefaults__)==6
            and all(type(key) is str and key in ('cls','object_hook','parse_float','parse_int',
                'parse_constant','object_pairs_hook') and value is None
                for key,value in loads.__kwdefaults__.items())
            and code_digest(loads)=='7a9fc010b4ec723309efe8667317339405121e56a39bef309690889abfd9bdc2')
    methods={**{'Draft.'+name:getattr(Draft202012Validator,name) for name in
        ('__init__','__attrs_post_init__','evolve','descend','iter_errors','validate')},
        **{'Registry.'+name:getattr(Registry,name) for name in ('with_resources','crawl')},
        **{'Resolver.'+name:getattr(Resolver,name) for name in ('lookup','_evolve')},
        'Resource.from_contents':Resource.from_contents,'retrieve':_fail_to_retrieve,
        'Draft.ID_OF':Draft202012Validator.ID_OF}
    require({k:code_digest(v) for k,v in methods.items()}==_VALIDATION_LIBRARY_CODES['methods'])
    require(type(Draft202012Validator.VALIDATORS) is dict and type(Draft202012Validator.TYPE_CHECKER) is TypeChecker)
    require(type(Draft202012Validator.TYPE_CHECKER._type_checkers) is modules['rpds'].HashTrieMap)
    require({k:code_digest(v) for k,v in Draft202012Validator.VALIDATORS.items()}==_VALIDATION_LIBRARY_CODES['keywords'])
    require({k:code_digest(v) for k,v in Draft202012Validator.TYPE_CHECKER._type_checkers.items()}==_VALIDATION_LIBRARY_CODES['type_checkers'])
    require(type(Draft202012Validator._APPLICABLE_VALIDATORS) is operator.methodcaller
            and repr(Draft202012Validator._APPLICABLE_VALIDATORS)=="operator.methodcaller('items')")
    # Compare loaded function code with the already fingerprinted source; file
    # identity alone does not qualify monkeypatched/stale loaded callables.
    for module,names in ((contracts,('schema_names','load_schema','schema_validator')),
                         (results,('_validator','_layout','_row_fields'))):
        length,expected=(1953,'32cbb629679afd8a752f8de69982889b0b24dfe1e5692c127ac8ccdb351c03be') if module is contracts else (
            51736,'d80c9985cb59a2e1d514695288c63e18b71fe3131294478806c57cac4c9f705f')
        with opened(Path(module.__file__)) as stream:
            raw=read(stream,length)
            require(len(raw)==length and not read(stream,1))
        require(hashlib.sha256(raw).hexdigest()==expected)
        code=compile(raw,module.__file__,'exec',dont_inherit=True)
        expected_codes={v.co_name:v for v in code.co_consts if type(v) is types.CodeType}
        for name in names:
            function=getattr(module,name)
            if module is results:
                require(type(function) is functools._lru_cache_wrapper)
                # __wrapped__ is writable metadata; the C wrapper calls the
                # distinct user_function referent. Qualify that actual target.
                referents=gc.get_referents(function)
                require(len(referents)==7)
                targets=[value for value in referents if type(value) is types.FunctionType]
                require(len(targets)==1 and targets[0] is function.__wrapped__)
                function=targets[0]
            require(type(function) is types.FunctionType and function.__globals__ is module.__dict__
                    and function.__code__==expected_codes[name] and function.__closure__ is None
                    and function.__defaults__ is None and function.__kwdefaults__ is None)

    def canonical(value):
        # Fixed bundled document bounds, not a general object-graph framework.
        # No user mapping/encoder callback can run while qualifying a cache.
        remaining=2*1024*1024
        active=set()
        def visit(item,depth=0):
            nonlocal remaining
            require(depth<=32)
            kind=type(item)
            require(item is None or kind in (bool,int,float,str,dict,list))
            remaining-=128+6*len(item) if kind is str else 128
            require(remaining>=0)
            if kind in (dict,list):
                require(id(item) not in active);active.add(id(item))
                for key,child in (item.items() if kind is dict else enumerate(item)):
                    if kind is dict:require(type(key) is str);visit(key,depth+1)
                    visit(child,depth+1)
                active.remove(id(item))
        visit(value)
        return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,
            separators=(',',':'),allow_nan=False).encode()).hexdigest()

    def cache(function,maximum):
        require(type(function) is functools._lru_cache_wrapper)
        require(set(function.__dict__)=={'cache_parameters','__module__','__name__','__qualname__',
            '__doc__','__annotate__','__type_params__','__wrapped__'})
        info=function.cache_info()
        require(info.maxsize is None and info.currsize<=maximum)
        referents=gc.get_referents(function)
        require(len(referents)==7)
        dictionaries=[v for v in referents if type(v) is dict and v is not function.__dict__]
        require(len(dictionaries)==1 and len(dictionaries[0])==info.currsize)
        return dictionaries[0],info

    require(type(REGISTRY) is Registry)
    standard={}
    for uri,resource in REGISTRY.items():
        require(type(uri) is str and type(resource) is Resource)
        standard[uri]=resource.contents
    require(len(standard)==20 and len(REGISTRY._anchors)==9)
    for uri,document in standard.items():require(canonical(document)==_VALIDATION_SCHEMAS.get(uri))
    validators,info=cache(results._validator,len(_VALIDATION_KINDS))
    identities={}
    base='https://qcl-negf.invalid/schema/'
    for name,validator in validators.items():
        require(type(name) is str and name in _VALIDATION_KINDS and type(validator) is Draft202012Validator)
        registry=validator._registry;resolver=validator._resolver
        require(type(registry) is Registry and type(resolver) is Resolver and type(resolver._registry) is Registry)
        require(len(registry)==36 and len(registry._anchors)==0 and len(registry._uncrawled)==36)
        require(registry._retrieve is REGISTRY._retrieve and resolver._registry._retrieve is REGISTRY._retrieve)
        require(len(resolver._registry)==56 and resolver._registry._anchors==REGISTRY._anchors)
        require(validator.schema is registry[base+name].contents)
        require(validator._ref_resolver is None and validator.format_checker is None)
        expected=[(Draft202012Validator.VALIDATORS[k],k,v) for k,v in validator.schema.items()
                  if k in Draft202012Validator.VALIDATORS]
        require(type(validator._validators) is list and len(validator._validators)==len(expected))
        for actual,wanted in zip(validator._validators,expected):
            require(type(actual) is tuple and len(actual)==3 and actual[0] is wanted[0]
                    and actual[1]==wanted[1] and actual[2] is wanted[2])
        for uri,resource in registry.items():
            require(type(resource) is Resource and uri.startswith(base) and canonical(resource.contents)==_VALIDATION_SCHEMAS.get(uri))
            require(resource._specification is Resource.from_contents(resource.contents)._specification)
            require(resolver._registry[uri].contents is resource.contents)
        for uri,document in standard.items():require(resolver._registry[uri].contents is document)
        identities[name]=id(validator)
    layout,_=cache(results._layout,1)
    require(not layout or (set(layout)=={()} and canonical(layout[()])==_VALIDATION_LAYOUT))
    row_fields,_=cache(results._row_fields,1)
    require(not row_fields or (set(row_fields)=={()} and type(row_fields[()]) is tuple
                              and row_fields[()]==_VALIDATION_ROW_FIELDS))
    if previous is not None:
        old_info,old_identities=previous
        require(info.hits>=old_info.hits and info.misses>=old_info.misses
                and all(identities.get(k)==v for k,v in old_identities.items()))
    return info,identities


def _context_validation_profile(arena, previous=None):
    arena.before('validation profile inspection')
    try:return _validation_profile(arena,previous)
    except ContractError:raise
    except (KeyError,AttributeError,TypeError,ValueError,OSError,RuntimeError,RecursionError) as error:
        raise ContractError('unknown or changed validation profile','incompatible_contract') from error


@contextmanager
def _context_call():
    if _context_arena.get() is not None or not _context_call_lock.acquire(blocking=False):
        _fail('metadata context reentry or concurrent operation','incompatible_contract')
    try:yield
    finally:_context_call_lock.release()


def _context_weight(value, *, seen=None, active=None, depth=0):
    """Explicit metadata units, not interpreter/HDF allocator or process RSS.

    A forest counts an aliased container once; distinct copies count separately.
    Scalar/string occurrences retain the conservative public W accounting.
    """
    if depth > 64: _fail('metadata context depth exceeds allowance', 'budget_exceeded')
    seen = set() if seen is None else seen
    active = set() if active is None else active
    kind = type(value)
    if value is None or kind in (bool, int, float): return 64
    if kind in (str, bytes): return 128 + 6 * len(value)
    if kind in (list, tuple, dict) or isinstance(value, OrderedDict) or is_dataclass(value):
        identity = id(value)
        if identity in active: _fail('metadata context cycle', 'budget_exceeded')
        if identity in seen: return 0
        seen.add(identity); active.add(identity)
        try:
            values = ((f.name, getattr(value, f.name)) for f in fields(value)) if is_dataclass(value) else (
                value.items() if isinstance(value, dict) else enumerate(value))
            return 128 + sum(_context_weight(k, seen=seen, active=active, depth=depth+1) +
                             _context_weight(v, seen=seen, active=active, depth=depth+1) for k,v in values)
        finally: active.remove(identity)
    if kind is set:
        return 128 + sum(_context_weight(v, seen=seen, active=active, depth=depth+1) for v in value)
    _fail('foreign object in metadata context', 'invalid_input')


def _context_json_size(value, depth=0):
    """UTF8 JSON length without allocating an encoded tree/string."""
    if depth > 64: _fail('metadata JSON depth', 'budget_exceeded')
    kind=type(value)
    if value is None:return 4
    if kind is bool:return 4 if value else 5
    if kind in (int,float):return len(json_bytes(value))
    if kind is str:
        total=2
        for c in value:
            n=ord(c)
            if 0xd800<=n<=0xdfff:_fail('invalid metadata UTF8')
            total+=2 if c in '\\"\b\f\n\r\t' else 6 if n<32 else 1 if n<128 else 2 if n<2048 else 3 if n<65536 else 4
        return total
    if kind is dict:
        return 2+max(0,len(value)-1)+sum(_context_json_size(k,depth+1)+1+_context_json_size(v,depth+1) for k,v in value.items())
    if kind in (list,tuple):return 2+max(0,len(value)-1)+sum(_context_json_size(v,depth+1) for v in value)
    _fail('foreign JSON metadata', 'invalid_input')


def _context_raw_tree_bound(raw):
    """No JSON materialization: bound scalar/container/string occurrences."""
    total=0;stack=[];inside=False;escaped=False;string_size=0;token=False
    def begin():
        nonlocal total
        if stack and stack[-1][0]==91 and stack[-1][1]:
            total+=64;stack[-1][1]=False
    for byte in raw:
        if inside:
            if byte==34 and not escaped:
                total+=128+6*string_size;inside=False;string_size=0
            else:
                string_size+=1
                if escaped:escaped=False
                elif byte==92:escaped=True
            continue
        if byte in (32,9,10,13,44,58,93,125):
            token=False
            if byte==44 and stack and stack[-1][0]==91:stack[-1][1]=True
            if byte in (93,125) and stack:stack.pop()
            continue
        if byte==34:
            begin();inside=True;token=False
        elif byte in (91,123):
            begin();total+=128;stack.append([byte,True]);token=False
            if len(stack)>64:_fail('metadata JSON depth', 'budget_exceeded')
        elif not token:
            begin();total+=64;token=True
    # Malformed strings still cannot allocate beyond their raw character bound.
    if inside:total+=128+6*string_size
    return total


class _MetadataArena:
    """One prepaid owner call: resident metadata plus next allocation scratch.

    Temporary capacity can be reused within this call. No caller credit is
    returned, and a failed call cannot be resumed with this arena.
    """
    def __init__(self, maximum):
        if type(maximum) is not int or not 0<maximum<=_CONTEXT_MAXIMUM:
            raise ValueError('maximum_context_bytes must be an integer in 1..134217728')
        self.validation_reservation=_CONTEXT_VALIDATION;self.profile_bytes_read=0;self.profile_io_remaining=_CONTEXT_PROFILE_IO
        self.maximum=maximum;self.high_water=0;self.trace=[];self.readers=[];self.held=[];self.scratch=0;self.failure=None
        self.before('arena header')

    def roots(self):
        for reader in self.readers:
            for name in ('_state_ref','_manifest','_pages','_index','_known_states','_catalog_by_hash',
                    '_published_paths','_catalog_by_leaf','_page_proofs','_parent_anchor_cache',
                    '_declarations','_verified_refs','_metadata_bindings','_checkpoint'):
                value=getattr(reader,name,None)
                if value is not None:yield value
            for key,item in getattr(reader,'_handles',{}).items():
                yield key;yield item['stamp']
            budget=getattr(reader,'_budget',None)
            guard=getattr(budget,'_source_guard',None)
            if guard is not None:yield guard
        yield from self.held

    def before(self, operation, *, extra=0, roots=(), copies=()):
        seen=set();active=set()
        resident=sum(_context_weight(v,seen=seen,active=active) for v in self.roots())
        resident+=sum(_context_weight(v,seen=seen,active=active) for v in roots)
        caches=sum(reader.limits.maximum_array_bytes*len(getattr(reader,'_handles',{}))
                   for reader in self.readers if hasattr(reader,'limits'))
        added=self.validation_reservation+self.scratch+caches+extra+sum(_context_weight(v) for v in copies)
        # Trace/header/traversal bookkeeping is also a finite part of admission.
        traversal=128*(len(seen)+65)
        total=_CONTEXT_HEADER+2048*(len(self.trace)+1)+traversal+resident+added
        if total>self.maximum:
            self.failure=dict(operation=operation,resident=resident,scratch=self.scratch,hdf_cache_allowance=caches,
                validation_reservation=self.validation_reservation,requested=added,traversal=traversal,total=total,maximum=self.maximum,completed_preflights=len(self.trace))
            _fail('metadata context arena exceeded before '+operation, 'budget_exceeded')
        self.high_water=max(self.high_water,total)
        self.trace.append((operation,resident,added,total))


@contextmanager
def _context_hold(*values):
    arena=_context_arena.get()
    if arena is None:yield;return
    start=len(arena.held);arena.held.extend(values)
    try:yield
    finally:del arena.held[start:]


def _context_before(operation, *, extra=0, roots=(), copies=()):
    arena=_context_arena.get()
    if arena is not None:arena.before(operation,extra=extra,roots=roots,copies=copies)


@contextmanager
def _context_workspace(operation, maximum):
    """Keep admitted foreign parser workspace live across nested owner hooks."""
    arena=_context_arena.get()
    if arena is None:yield;return
    arena.before(operation,extra=maximum)
    arena.scratch+=maximum
    try:yield
    finally:arena.scratch-=maximum


def _context_copy(value):
    _context_before('deepcopy',roots=(value,),copies=(value,))
    return copy.deepcopy(value)


def _context_decoding(raw, source):
    if _context_arena.get() is None:return
    tree=_context_raw_tree_bound(raw)
    ordinary=512*(1+sum(raw.count(c) for c in (b'{',b'}',b'[',b']',b',',b':')))+4*len(raw)
    # commits.decode_json materializes json.loads + validate_json container copy.
    # Pair-hook tuples/string decoding are covered before the protected call.
    scratch=max(ordinary,3*tree+4*len(raw)+4096)
    _context_before('JSON '+source,extra=scratch,roots=(raw,))


def _context_contract(name, value, **contexts):
    if _context_arena.get() is None:return
    w=_context_weight;size=_context_json_size
    def checked(v):return 2*w(v)+16*size(v)+4096
    def manifest(m):return checked(m)+w(m)
    def checkpoint(c,m):return max(checked(c),w(c)+manifest(m))+4096
    def diagnostic(d,c,m,r):
        return max(checked(d),w(d)+manifest(m),w(d)+w(m)+checkpoint(c,m),
            w(d)+w(m)+w(c)+checked(r))+4096
    extra=checked(value)
    if name=='validate_manifest':extra=manifest(value)
    elif name=='validate_checkpoint_declaration':extra=checkpoint(value,contexts.get('manifest',{}))
    elif name=='validate_final_diagnostics':
        extra=diagnostic(value,contexts['checkpoint'],contexts['manifest'],contexts['artifact_ref'])
    elif name=='validate_final_observations':
        c=contexts['checkpoint'];m=contexts['manifest'];d=contexts['diagnostics']
        retained=w(value)+w(m)+w(c)
        extra=max(checked(value),w(value)+manifest(m),w(value)+w(m)+checkpoint(c,m),
            retained+diagnostic(d,c,m,value['diagnostics']),
            retained+w(d)+checked(contexts['plan'])+checked(contexts['model']))+4096
        for key in ('parent_manifest','parent_checkpoint'):
            if contexts.get(key) is not None:extra+=checked(contexts[key])
    elif name=='validate_catalog_page':
        if contexts.get('previous_page') is not None:extra=max(extra,w(value)+checked(contexts['previous_page'])+checked(contexts.get('previous_ref',{})))
    elif name=='validate_scientific_record':
        if contexts.get('parent_checkpoint') is not None:extra=max(extra,w(value)+checkpoint(contexts['parent_checkpoint'],{}))+checked(contexts.get('parent_state',{}))+checked(contexts.get('parent_manifest_ref',{}))
    elif name=='validate_model':extra=4*w(value)+16*size(value)+4096
    _context_before('Contracts '+name,extra=extra,roots=(value,contexts))


def _context_encoded(value):
    if _context_arena.get() is not None:
        _context_before('JSON encoding',extra=16*_context_json_size(value),roots=(value,))
    return json_bytes(value)


class _IOBudget:
    """Physical stream delivery, including integrity and native validation.

    This counts delivered bytes, not disk traffic or logical dataset bytes.
    Checking an EOF-clipped request before the callback prevents overshoot.
    """
    def __init__(self, maximum: int | None):
        if maximum is not None and (type(maximum) is not int or maximum < 1):
            raise ValueError("maximum_io_bytes must be a positive integer")
        self.maximum = maximum
        self.phase = "metadata"
        self.bytes_read_total = 0
        self.sha_bytes = 0
        self.hdf_bytes = 0
        self.metadata_bytes = 0
        self.logical_selected_bytes = 0
        self._source_guard = None

    def require(self, count: int) -> None:
        if self.maximum is not None and count > self.maximum - self.bytes_read_total:
            raise ContractError("total physical stream I/O budget exceeded", "budget_exceeded")

    def account(self, count: int) -> None:
        self.bytes_read_total += count
        if self.phase == "sha":
            self.sha_bytes += count
        elif self.phase == "hdf":
            self.hdf_bytes += count
        else:
            self.metadata_bytes += count

    def counters(self) -> dict[str, int | None]:
        return {name: getattr(self, name) for name in (
            "bytes_read_total", "sha_bytes", "hdf_bytes", "metadata_bytes", "logical_selected_bytes")} | {
                "maximum_io_bytes": self.maximum}


class _BudgetStream(io.RawIOBase):
    """A single read-only file object used by SHA and the HDF5 fileobj driver."""
    def __init__(self, raw: Any, budget: _IOBudget):
        super().__init__()
        self._raw, self._budget = raw, budget

    @property
    def name(self) -> str:
        return self._raw.name

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._raw.tell()

    def seek(self, offset: int, whence: int = 0) -> int:
        return self._raw.seek(offset, whence)

    def effective_request(self, size: int) -> int:
        position = self._raw.tell()
        end = self._raw.seek(0, 2)
        self._raw.seek(position)
        remaining = max(0, end - position)
        return remaining if size < 0 else min(size, remaining)

    def read(self, size: int = -1) -> bytes:
        self._budget.require(self.effective_request(size))
        data = self._raw.read(size)
        self._budget.account(len(data))
        return data

    def readinto(self, target: Any) -> int:
        self._budget.require(self.effective_request(len(target)))
        count = self._raw.readinto(target)
        self._budget.account(count)
        return count

    def close(self) -> None:
        if not self.closed:
            self._raw.close()
        super().close()


def _verify_stream(stream: Any, size: int, digest: str) -> None:
    stream.seek(0, 2)
    if stream.tell() != size:
        _fail("state payload byte length differs from its commit")
    stream.seek(0)
    _context_before('SHA streaming buffer',extra=262144+4096)
    if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
        _fail("state payload checksum differs from its commit")
    stream.seek(0)


def _same_json_value(left: Any, right: Any) -> bool:
    """Compare decoded JSON values without Python's bool/int/float coercion."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _same_json_value(left[key], right[key]) for key in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _same_json_value(a, b) for a, b in zip(left, right))
    return left == right


@dataclass(frozen=True)
class ResultReadLimits:
    maximum_metadata_bytes: int = 16 * 1024 * 1024
    maximum_io_bytes: int = 256 * 1024 * 1024
    maximum_catalog_pages: int = 4096
    maximum_catalog_entries: int = 65536
    maximum_catalog_bytes: int = 64 * 1024 * 1024
    maximum_open_files: int = 8
    maximum_record_bytes: int = 1024 * 1024
    maximum_batch_bytes: int = 4 * 1024 * 1024
    maximum_batch_records: int = 256
    maximum_hdf_objects: int = 65536
    maximum_hdf_depth: int = 64
    maximum_array_bytes: int = 1024 * 1024

    def __post_init__(self):
        if any(type(getattr(self, item.name)) is not int or getattr(self, item.name) < 1
               for item in fields(self)):
            raise ValueError('read limits require positive integers')


@dataclass(frozen=True)
class ResultDescription:
    state: Document
    invocation: Document
    history: Document
    lineage: Document
    status: Document
    closed: bool
    capabilities: Document
    integrity: Document
    io_counters: Mapping[str, int | None]


@dataclass(frozen=True)
class HistoryBatch:
    source: Document
    records: tuple[Document, ...]
    cursor: Document
    at_cutoff: bool
    io_counters: Mapping[str, int | None]


@dataclass(frozen=True)
class ArrayDescription:
    declaration: Document
    logical_shape: tuple[int, ...]
    stored_shape: tuple[int, ...]
    eltype: str
    element_bytes: int
    order: str
    source: Document


@dataclass(frozen=True)
class ArrayBlock:
    values: Any
    selection: tuple[int | slice, ...]
    declaration: Document
    source: Document
    io_counters: Mapping[str, int | None]


def _internal_storage(handle, *, maximum_objects=65536, maximum_depth=64):
    """Bounded ownership walk, without reading dataset values."""
    seen, pending = set(), [(handle, 0)]
    while pending:
        group, depth = pending.pop()
        address = h5py.h5o.get_info(group.id).addr
        if address in seen:
            continue
        if depth > maximum_depth or len(seen) >= maximum_objects:
            _fail('native object inventory exceeds read limit', 'budget_exceeded')
        seen.add(address)
        if len(group) > maximum_objects - len(seen):
            _fail('native group inventory exceeds read limit', 'budget_exceeded')
        for name in group:
            if not isinstance(group.get(name, getlink=True), h5py.HardLink):
                _fail('native source rejects soft/external links')
            child = group[name]
            if isinstance(child, h5py.Group):
                pending.append((child, depth + 1))
            elif isinstance(child, h5py.Dataset):
                if child.external or child.is_virtual:
                    _fail('native source rejects external/virtual storage')
                address = h5py.h5o.get_info(child.id).addr
                seen.add(address)
                if len(seen) > maximum_objects:
                    _fail('native object inventory exceeds read limit', 'budget_exceeded')


class StateReader:
    """One explicitly selected immutable result source; no legacy format branch.

    Each operation has a physical stream budget. Open handles reuse SHA proof;
    cache eviction/reopen is a new verification boundary. Returned JSON is copied.
    """
    def __init__(self, artifact_root: str | Path, *, state_ref: Document,
                 expected_identity: Document, limits: ResultReadLimits | None = None):
        self._initialize(artifact_root, state_ref=state_ref, expected_identity=expected_identity, limits=limits)

    def _initialize(self, artifact_root, *, state_ref, expected_identity, limits=None,
                    verified_manifest=None, budget=None, lineage_depth=0, metadata_bindings=None,
                    metadata_owner=None):
        arena=_context_arena.get()
        if arena is not None:
            arena.readers.append(self)
            _context_before('reader initialization',roots=(state_ref,expected_identity))
        self.limits = limits or ResultReadLimits()
        if not isinstance(self.limits, ResultReadLimits):
            raise TypeError('limits must be ResultReadLimits')
        self.root = Path(os.path.abspath(artifact_root))
        self._root_stamp = self._root_identity()
        self._closed = False
        self._handles = OrderedDict()
        self._pages, self._index, self._known_states = [], {}, {}
        self._catalog_by_hash, self._published_paths, self._catalog_by_leaf = {}, set(), {}
        self._page_proofs = {}
        self._parent_anchor_cache = None
        self._catalog_ready, self._catalog_size = False, 0
        self._index_weight = self._known_weight = self._pending_state_weight = 0
        self._checkpoint = None
        self._declarations, self._verified_refs = {}, {}
        self._pointer_path, self._pointer_raw = None, None
        self._shared_budget = budget
        # FD stamps accompany already decoded metadata; they never supply bytes,
        # resolve a FileRef or substitute for its original hash validation.
        self._metadata_bindings = (metadata_bindings if metadata_bindings is not None
            else dict(entries={}, weight=0))
        if metadata_bindings is None:
            self._metadata_bindings['weight'] = _metadata_weight(self._metadata_bindings)
        self._metadata_context_owner = metadata_owner
        self._lineage_depth = lineage_depth
        if lineage_depth > self.limits.maximum_hdf_depth:
            _fail('lineage proof depth exceeds metadata limit', 'budget_exceeded')
        self._operation()
        self._state_ref = _context_copy(dict(state_ref))
        _schema_check(schema_validator('result-common.schema.json').evolve(schema={
            '$ref': '#/$defs/StateRef', '$defs': schema_validator('result-common.schema.json').schema['$defs']
        }), self._state_ref)
        if not _same_json_value(self._state_ref['identity'], dict(expected_identity)):
            _fail('complete selected identity differs from StateRef', 'identity_mismatch')
        original=verified_manifest if verified_manifest is not None else self._json(self._state_ref['manifest'])
        _context_contract('validate_manifest',original)
        self._manifest = validate_manifest(original)
        del original
        for key in ('identity', 'state_id', 'state_sequence'):
            if self._manifest[key] != self._state_ref[key]:
                _fail('manifest differs from selected StateRef')
        self._register_manifest()
        self._remember_state()

    def _root_identity(self):
        from ._capture_io import regular
        # Opening a known child is unnecessary; nofollow traversal is performed
        # by regular for every actual file. Reject symlink root itself as well.
        current = self.root
        for part in (current, *current.parents):
            if part.is_symlink():
                _fail('artifact root traverses a symbolic link')
        stat = self.root.stat()
        return stat.st_dev, stat.st_ino

    def _operation(self):
        if getattr(self, '_closed', False):
            raise ValueError('state reader is closed')
        if hasattr(self, '_root_stamp') and self._root_identity() != self._root_stamp:
            _fail('artifact root replaced', 'source_changed')
        self._budget = getattr(self, '_shared_budget', None) or _IOBudget(self.limits.maximum_io_bytes)
        for item in getattr(self, '_handles', {}).values():
            item['stream']._budget = self._budget

    @contextmanager
    def _consumer_operation(self, maximum_io_bytes=None, *, maximum_owner_bytes=None,
                            verify_source_stability=False, maximum_stability_bytes=None):
        """One allowance through nested checkpoint/catalog/ancestor operations."""
        self._operation()
        budget, prior = self._budget, self._shared_budget
        prior_owner = getattr(self, '_consumer_owner_limit', None)
        prior_guard = budget._source_guard
        maximum = budget.maximum
        try:
            self._consumer_owner_limit = (maximum_owner_bytes if prior_owner is None else prior_owner
                if maximum_owner_bytes is None else min(prior_owner,maximum_owner_bytes))
            if maximum_io_bytes is not None:
                if type(maximum_io_bytes) is not int or maximum_io_bytes < 1:
                    raise ValueError('positive consumer I/O allowance required')
                bound = budget.bytes_read_total + maximum_io_bytes
                budget.maximum = bound if maximum is None else min(maximum, bound)
            self._shared_budget = budget
            if verify_source_stability and prior_guard is None:
                cap = self.limits.maximum_metadata_bytes
                working = cap
                if maximum_stability_bytes is not None:
                    if type(maximum_stability_bytes) is not int or maximum_stability_bytes < 1:
                        raise ValueError('positive source stability allowance required')
                    cap = min(cap, maximum_stability_bytes)
                    working = maximum_stability_bytes
                header = dict(entries={}, weight=0, maximum=cap, working=working,
                              source=self._state_ref, root=self._root_stamp)
                weight = _metadata_weight(header)
                if weight > cap or weight + self._metadata_bindings['weight'] > working:
                    _fail('source binding header exceeds allowance', 'budget_exceeded')
                budget._source_guard = dict(header, weight=weight, source=_context_copy(self._state_ref))
                del header  # No duplicate temporary header remains across yield.
                for path, opened in self._metadata_bindings['entries'].items():
                    self._note_source_binding(path, opened)
            yield budget
            if verify_source_stability and prior_guard is None:
                self._check_source_bindings()
        finally:
            self._shared_budget = prior
            self._consumer_owner_limit = prior_owner
            budget.maximum = maximum
            budget._source_guard = prior_guard

    def _note_source_binding(self, path, opened, *, metadata=False):
        """Retain an actual authenticated FD stamp, never a later path snapshot."""
        def charge(book, maximum):
            old = book['entries'].get(path)
            if old is not None:
                if old != opened:
                    _fail('consumed immutable source binding changed', 'source_changed')
                return 0
            # Deep row weight plus 128 B covers the dict slot/growth allowance.
            # Paths/stamp tuples may be shared; counting them again is conservative.
            addition = _metadata_weight((path, opened)) + 128
            if (len(book['entries']) >= self.limits.maximum_catalog_entries
                    or book['weight'] + addition > maximum):
                _fail('source binding metadata exceeds allowance', 'budget_exceeded')
            return addition
        jobs = []
        metadata_charge = 0
        if metadata:
            if not hasattr(self, '_metadata_bindings'):
                self._metadata_bindings = dict(entries={}, weight=0)
                self._metadata_bindings['weight'] = _metadata_weight(self._metadata_bindings)
            context_owner = getattr(self, '_metadata_context_owner', None) or self
            names = ('_index_weight', '_known_weight', '_pending_state_weight', '_catalog_size')
            retained = max(sum(getattr(owner, name, 0) for name in names)
                           for owner in (self, context_owner))
            maximum = min(self.limits.maximum_catalog_bytes, context_owner.limits.maximum_catalog_bytes)
            metadata_charge = charge(self._metadata_bindings, maximum-retained)
            jobs.append((self._metadata_bindings, metadata_charge))
        guard = self._budget._source_guard
        if guard is not None:
            guard_charge = charge(guard, guard['maximum'])
            if (guard['weight'] + guard_charge + self._metadata_bindings['weight']
                    + metadata_charge > guard['working']):
                _fail('source binding workspace exceeds allowance', 'budget_exceeded')
            jobs.append((guard, guard_charge))
        _context_before('source binding entries',extra=512*len(jobs),roots=(path,opened))
        for book, addition in jobs:
            if addition:
                book['entries'][path] = opened
                book['weight'] += addition

    def _source_stability_bytes(self, *, seal=False):
        guard = self._budget._source_guard
        if guard is None:
            return 0
        if seal:
            # All matrix owners have already been described. No unforecast new
            # trace allocation may appear while numerical buffers are retained.
            guard['maximum'] = guard['weight']
        return guard['weight'] + self._metadata_bindings['weight']

    def _check_source_bindings(self):
        from .commits import safe_path
        guard = self._budget._source_guard
        if guard is None:
            return
        if self._state_ref != guard['source']:
            _fail('selected operation source changed', 'source_changed')
        try:
            if self._root_identity() != guard['root']:
                _fail('artifact root replaced', 'source_changed')
            for path, opened in guard['entries'].items():
                current = safe_path(self.root, path).stat(follow_symlinks=False)
                actual = (current.st_dev, current.st_ino, current.st_size,
                          current.st_mtime_ns, current.st_ctime_ns)
                if actual != opened:
                    _fail('consumed source binding changed', 'source_changed')
            for key, item in self._handles.items():
                if key[0] in guard['entries']:
                    self._check_hdf_owner(key, item)
        except (OSError, ValueError, ContractError) as error:
            raise ContractError('consumed source binding changed', 'source_changed') from error

    @classmethod
    def from_pointer(cls, artifact_root, *, expected_identity, pointer, limits=None):
        if pointer not in ('live', 'recovery'):
            raise ValueError('pointer must be live or recovery')
        selected = dict(expected_identity)
        path = ('executions/' + selected['execution_id'] + '/' + selected['point_id'] +
                '/attempt-' + selected['attempt'] + ('/current.json' if pointer == 'live'
                else '/recovery/current.json'))
        # Validate selection/path through canonical pure schema before file access.
        common = schema_validator('result-common.schema.json')
        _schema_check(common.evolve(schema={'$ref': '#/$defs/Identity', '$defs': common.schema['$defs']}), selected)
        from ._capture_io import beneath, small_json
        cap = limits or ResultReadLimits()
        budget = _IOBudget(cap.maximum_io_bytes)
        value, raw = small_json(beneath(Path(artifact_root), path), cap.maximum_metadata_bytes, budget=budget)
        value = validate_pointer(value)
        if value['identity'] != selected:
            _fail('pointer differs from selected identity', 'identity_mismatch')
        # Capture the hashed manifest once to derive its StateRef; the private
        # initializer consumes that exact verified document and physical budget.
        temporary = cls.__new__(cls)
        temporary.root, temporary.limits, temporary._budget = Path(os.path.abspath(artifact_root)), cap, budget
        manifest = validate_manifest(temporary._json(value['manifest']))
        ref = {key: manifest[key] for key in ('identity', 'state_id', 'state_sequence')}
        ref['manifest'] = value['manifest']
        result = cls.__new__(cls)
        result._initialize(artifact_root, state_ref=ref, expected_identity=selected, limits=cap,
                           verified_manifest=manifest, budget=budget,
                           metadata_bindings=temporary._metadata_bindings)
        result._shared_budget = None
        if manifest['generation'] != value['generation']:
            result.close()
            _fail('pointer generation differs from manifest')
        result._pointer_path, result._pointer_raw = path, raw
        return result

    def __enter__(self):
        return self

    def __exit__(self, *args):
        try:
            self.close()
        except BaseException as cleanup_error:
            if len(args) < 2 or args[1] is None:
                raise
            args[1].add_note('native owner cleanup also raised ' + type(cleanup_error).__name__)

    def close(self):
        # Mark closed and detach first: cancellation in one stack must not leak
        # another verified owner or make a second close repeat its callbacks.
        owners = tuple(self._handles.values())
        self._handles.clear()
        self._closed = True
        first = None
        for value in owners:
            try:
                value['stack'].close()
            except BaseException as error:
                if first is None:
                    first = error
                else:
                    first.add_note('another native owner cleanup raised ' + type(error).__name__)
        if first is not None:
            raise first

    def _json(self, reference):
        from ._capture_io import beneath, regular, stamp
        if int(reference['bytes']) > self.limits.maximum_metadata_bytes:
            _fail('metadata exceeds read limit', 'budget_exceeded')
        with regular(beneath(self.root, reference['path']), size=int(reference['bytes'])) as raw:
            before = stamp(raw, include_change=True)
            stream = _BudgetStream(raw, self._budget)
            phase, self._budget.phase = self._budget.phase, 'metadata'
            try:
                _context_before('original JSON read',extra=128+6*int(reference['bytes']))
                payload = stream.read(int(reference['bytes']))
            finally:
                self._budget.phase = phase
            if stamp(raw, include_change=True) != before:
                _fail('metadata changed during authenticated read', 'source_changed')
            if hashlib.sha256(payload).hexdigest() != reference['sha256']:
                _fail('metadata differs from immutable reference')
            self._note_source_binding(reference['path'], before, metadata=True)
        _context_decoding(payload,reference['path'])
        return decode_json(payload, reference['path'])

    def _register(self, reference, *, table=None):
        _context_before('reference inventory entry',extra=1024,roots=(reference,))
        target = self._index if table is None else table
        relative_path(reference['path'])
        simple = _ref(reference)
        old = target.get(simple['path'])
        if old is not None and _ref(old) != simple:
            _fail('immutable path rebound to different bytes')
        value = _context_copy(dict(reference))
        if target is self._index:
            changed_weight = _metadata_weight(value) - (_metadata_weight(old) if old is not None else 0)
            if (self._index_weight + changed_weight + self._known_weight + self._pending_state_weight
                    + self._catalog_size + self._metadata_bindings['weight'] > self.limits.maximum_catalog_bytes):
                _fail('retained metadata context exceeds limit', 'budget_exceeded')
            self._index_weight += changed_weight
        target[simple['path']] = value
        if len(target) > self.limits.maximum_catalog_entries:
            _fail('imported reference inventory exceeds limit', 'budget_exceeded')

    def _register_manifest(self):
        self._register(self._state_ref['manifest'])
        for row in self._manifest['artifacts']:
            self._register(row)

    def _remember_state(self):
        digest = self._state_ref['manifest']['sha256']
        if digest not in self._known_states:
            _context_before('known source entry',extra=1024)
            weight = _metadata_weight(self._state_ref) + _metadata_weight(self._manifest)
            if self._known_weight + weight + self._index_weight + self._catalog_size + self._metadata_bindings['weight'] > self.limits.maximum_catalog_bytes:
                _fail('retained metadata context exceeds limit', 'budget_exceeded')
            self._known_states[digest] = (self._state_ref, self._manifest)
            self._known_weight += weight
        self._pending_state_weight = 0

    @property
    def io_counters(self):
        return self._budget.counters()

    @property
    def source(self):
        ready = self._catalog_ready
        def cap(value, reason):
            return dict(available=bool(value), reason=None if value else reason)
        checkpoint = self._checkpoint is not None
        history = int(self._manifest['history']['records']) > 0
        _context_before('source description copies',extra=8192,copies=(self._state_ref,self._manifest['invocation'],self._manifest['history'],self._manifest['lineage'],self._manifest['status'],list(self._verified_refs.values())))
        capabilities = dict(history_read=cap(ready, 'catalog_import_required'),
            array_slice=cap(bool(self._declarations), 'verified_array_declaration_required'),
            exact_checkpoint=cap(checkpoint, 'checkpoint_verification_required' if self._manifest['checkpoint'] else 'selected_state_has_no_checkpoint'),
            exact_restore=cap(False, 'core_owned'), saved_views=cap(False, 'explicit_final_observations_ref_required'),
            stored_observables=cap(False, 'explicit_final_observations_ref_required'),
            stored_final_diagnostics=cap(False, 'explicit_final_diagnostics_ref_required'),
            streaming_export=cap(history or self._manifest['checkpoint'], 'no_selected_scientific_data'),
            science_export=cap(history, 'history_or_explicit_analysis_ref_required'),
            full_state_export=cap(self._manifest['checkpoint'] is not None, 'selected_state_has_no_checkpoint'),
            scalar_derive=cap(ready, 'catalog_import_required'))
        with _context_hold(capabilities), _context_workspace('source copy arguments',
                8192+sum(_context_weight(v) for v in (self._state_ref,self._manifest['invocation'],self._manifest['history'],self._manifest['lineage'],self._manifest['status'],list(self._verified_refs.values()))) if _context_arena.get() is not None else 0):
            return ResultDescription(_context_copy(self._state_ref), _context_copy(self._manifest['invocation']),
                _context_copy(self._manifest['history']), _context_copy(self._manifest['lineage']),
                _context_copy(self._manifest['status']), self._manifest['closed'], capabilities,
                dict(manifest_verified=True, catalog_verified=ready,
                     selected_payloads_verified=_context_copy(list(self._verified_refs.values())), scope='selected_bytes'),
                self.io_counters)

    @staticmethod
    def _pin(page, reference):
        return dict(head=_ref(reference), entries=page['entries'], **page['totals'])

    def import_history(self):
        self._operation()
        self._import_history()
        return self.source

    def _import_history(self):
        live=[]
        with _context_hold(live):
            pin = self._manifest['history']
            known = self._catalog_by_hash
            ref, pending, seen = pin['head'], [], set()
            live.extend((pending,seen))
            size, delta, added_paths, delta_weights, index_change = self._catalog_size, {}, set(), {}, 0
            live.extend((delta,added_paths,delta_weights))
            def register(reference):
                nonlocal index_change
                old = delta.get(reference['path'], self._index.get(reference['path']))
                if old is not None and _ref(old) != _ref(reference):
                    _fail('immutable path rebound to different bytes')
                self._register(reference, table=delta)
                existing = self._index.get(reference['path'])
                changed = _metadata_weight(delta[reference['path']]) - (_metadata_weight(existing) if existing is not None else 0)
                index_change += changed - delta_weights.get(reference['path'], 0)
                delta_weights[reference['path']] = changed
                if reference['path'] not in self._index:
                    added_paths.add(reference['path'])
                if len(self._index) + len(added_paths) > self.limits.maximum_catalog_entries:
                    _fail('imported reference inventory exceeds limit', 'budget_exceeded')
                if (size + self._index_weight + index_change + self._known_weight
                        + self._pending_state_weight + self._metadata_bindings['weight'] > self.limits.maximum_catalog_bytes):
                    _fail('retained catalog/context metadata exceeds limit', 'budget_exceeded')
            anchors = self._parent_anchors()
            for reference in anchors.values():
                register(dict(reference, role='science.history', dependencies=[]))
            while ref is not None and ref['sha256'] not in known:
                if ref['path'] in seen or len(pending) + len(self._pages) >= self.limits.maximum_catalog_pages:
                    _fail('catalog cycle or page limit', 'budget_exceeded')
                seen.add(ref['path'])
                page = self._json(ref)
                size += _metadata_weight(page) + _metadata_weight(ref)
                if size + self._index_weight + self._known_weight + self._pending_state_weight + self._metadata_bindings['weight'] > self.limits.maximum_catalog_bytes:
                    _fail('decoded catalog inventory exceeds limit', 'budget_exceeded')
                pending.append((_context_copy(ref), page))
                # Syntax is checked before following an untrusted predecessor.
                _schema_check(schema_validator('scientific-catalog.schema.json'), page)
                ref = page['previous']['head'] if page['previous'] else None
            if self._pages and pending and ref != self._pages[-1][0]:
                _fail('new history does not extend imported head', 'source_changed')
            prior = known.get(ref['sha256']) if ref else None
            previous_pin = self._pin(prior[1], prior[0]) if prior else None
            prior_page = prior[1] if prior else None
            prior_ref = prior[0] if prior else None
            additions, new_paths_published = [], set()
            live.extend((additions,new_paths_published))
            for reference, raw_page in reversed(pending):
                _context_contract('validate_catalog_page',raw_page,previous=previous_pin,previous_page=prior_page,previous_ref=prior_ref)
                page = validate_catalog_page(raw_page, previous=previous_pin,
                    previous_page=prior_page, previous_ref=prior_ref)
                live.append(page)
                if page['identity'] != self._manifest['identity'] or page['invocation'] != self._manifest['invocation']:
                    _fail('catalog identity or epoch differs from manifest')
                rows = page['artifacts']
                if len(rows) not in (1, 2):
                    _fail('native5 batch requires optional array owner then one record leaf')
                new_paths = {row['path'] for row in rows}
                if new_paths & self._published_paths or new_paths & new_paths_published:
                    _fail('catalog republishes a nonadjacent immutable path')
                if new_paths & anchors.keys():
                    _fail('catalog republishes an ancestor append dependency')
                if len(rows) == 2 and (rows[0]['dependencies'] or rows[0]['path'] not in rows[-1]['dependencies']):
                    _fail('ordered native5 array dependency differs')
                for row in rows:
                    register(row)
                for row in rows:
                    if any(path == row['path'] or not (path in self._published_paths
                            or path in new_paths_published or path in new_paths or path in anchors)
                            for path in row['dependencies']):
                        _fail('catalog dependency absent, future or self-referential')
                    if row is rows[0] and len(rows) == 2 and row['dependencies']:
                        _fail('array owner cannot depend on another leaf')
                register(reference)
                new_paths_published.update(new_paths)
                additions.append((reference, page))
                prior_ref, prior_page, previous_pin = reference, page, self._pin(page, reference)
            actual_pin = previous_pin if pending else (self._pin(*reversed(self._pages[-1])) if self._pages else None)
            if pin['head'] is None:
                if self._pages:
                    _fail('history disappeared', 'source_changed')
            elif actual_pin != pin:
                _fail('catalog head counters differ from manifest pin')
            first_new = len(self._pages)
            _context_before('catalog aggregate indexes',extra=4096+1024*len(additions),roots=(additions,delta,new_paths_published))
            self._pages.extend(additions)
            for offset, (reference, page) in enumerate(additions, first_new):
                leaf = page['artifacts'][-1]
                self._catalog_by_leaf[(leaf['path'], leaf['sha256'])] = (offset, reference, page)
            self._index.update(delta)
            self._index_weight += index_change
            self._catalog_by_hash.update({reference['sha256']: (reference, page) for reference, page in additions})
            self._published_paths.update(new_paths_published)
            self._catalog_size, self._catalog_ready = size, True

    def _check_hdf_owner(self, key, item):
        """Check the actual retained FD and its immutable root-relative binding."""
        from ._capture_io import stamp
        from .commits import safe_path
        try:
            if self._root_identity() != self._root_stamp:
                _fail('artifact root replaced', 'source_changed')
            current = safe_path(self.root, key[0]).stat(follow_symlinks=False)
            value = (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns, current.st_ctime_ns)
            opened = stamp(item['raw'], include_change=True)
        except (OSError, ValueError, ContractError) as error:
            raise ContractError('trusted payload binding changed', 'source_changed') from error
        if value != item['stamp'] or opened != item['stamp']:
            _fail('trusted payload source changed', 'source_changed')

    def _open_hdf(self, reference, *, identity=None, invocation=None, representation=None):
        from ._capture_io import regular, stamp
        from .native import validate_native_handle
        owner_limit = getattr(self, '_consumer_owner_limit', None)
        if owner_limit is not None and int(reference['bytes']) > owner_limit:
            _fail('selected native owner exceeds consumer byte limit', 'budget_exceeded')
        key = (reference['path'], reference['sha256'])
        from .commits import safe_path
        path = safe_path(self.root, reference['path'])
        if key in self._handles:
            item = self._handles[key]
            self._check_hdf_owner(key, item)
            self._handles.move_to_end(key)
        else:
            stack = ExitStack()
            try:
                raw = stack.enter_context(regular(path, size=int(reference['bytes'])))
                before = stamp(raw, include_change=True)
                stream = _BudgetStream(raw, self._budget)
                self._budget.phase = 'sha'
                _verify_stream(stream, int(reference['bytes']), reference['sha256'])
                self._budget.phase = 'hdf'
                _context_before('HDF metadata handles/cache',extra=self.limits.maximum_array_bytes+128*self.limits.maximum_hdf_objects)
                handle = stack.enter_context(h5py.File(stream, 'r', rdcc_nbytes=self.limits.maximum_array_bytes))
                _internal_storage(handle, maximum_objects=self.limits.maximum_hdf_objects,
                                  maximum_depth=self.limits.maximum_hdf_depth)
                validate_native_handle(handle, reference.get('role'), reference['schema'],
                                       maximum_metadata_bytes=self.limits.maximum_metadata_bytes)
                if stamp(raw, include_change=True) != before:
                    _fail('payload changed while being verified')
                item = dict(stack=stack, raw=raw, stream=stream, handle=handle, stamp=before)
                self._check_hdf_owner(key, item)
                self._handles[key] = item
                self._verified_refs[reference['path']] = _ref(reference)
                while len(self._handles) > self.limits.maximum_open_files:
                    _, old = self._handles.popitem(last=False)
                    old['stack'].close()
                    self._verified_refs = {path: value for path, value in self._verified_refs.items()
                        if any(key[0] == path for key in self._handles)}
            except BaseException as error:
                self._handles.pop(key, None)
                self._verified_refs.pop(reference['path'], None)
                try:
                    stack.close()
                except BaseException as cleanup_error:
                    error.add_note('temporary native owner cleanup also raised ' + type(cleanup_error).__name__)
                raise
        self._budget.phase = 'hdf'
        handle = item['handle']
        if owner_limit is not None:
            _internal_storage(handle, maximum_objects=self.limits.maximum_hdf_objects,
                              maximum_depth=self.limits.maximum_hdf_depth)
        from .native import native_metadata
        native_metadata(handle, identity=identity, invocation=invocation,
                        representation=representation, maximum_bytes=self.limits.maximum_metadata_bytes)
        self._note_source_binding(reference['path'], item['stamp'])
        return handle

    def _record(self, leaf, offset, *, identity, invocation, page=None):
        handle = self._open_hdf(leaf, identity=identity, invocation=invocation, representation='record_leaf')
        n = self._record_count(handle)
        if type(offset) is not int or not 0 <= offset < n:
            _fail('record offset outside physical record leaf')
        start, stop = (int(x) for x in handle['records/offsets'][offset:offset+2])
        if stop - start > self.limits.maximum_record_bytes:
            _fail('record JSON exceeds read limit', 'budget_exceeded')
        _context_before('record UTF8 read',extra=128+8*(stop-start))
        payload = handle['records/json_utf8'][start:stop].tobytes()
        _context_decoding(payload,'scientific record')
        record = decode_json(payload, 'scientific record')
        ordinal = int(handle['records/stream_ordinal'][offset])
        if record.get('identity') != identity or record.get('invocation') != invocation or record.get('stream_ordinal') != str(ordinal):
            _fail('record identity/epoch/ordinal differs from native coordinates')
        if page is not None and not int(page['range']['first_record']) <= ordinal <= int(page['range']['last_record']):
            _fail('record ordinal outside containing page')
        item = next(item for item in self._handles.values() if item['handle'] is handle)
        self._note_source_binding(leaf['path'], item['stamp'], metadata=True)
        return record

    def _record_count(self, handle):
        from .native import record_table
        for item in self._handles.values():
            if item['handle'] is handle:
                # A warm coordinate proof authenticates content, not admission
                # under a later consumer's tighter decoded-chunk allowance.
                for name in ('offsets','json_utf8','stream_ordinal'):
                    dataset = handle['records/'+name]
                    if dataset.chunks and math.prod(dataset.chunks)*dataset.dtype.itemsize > self.limits.maximum_record_bytes:
                        _fail('native record chunk decode buffer exceeds limit', 'budget_exceeded')
                if 'record_count' not in item:
                    item['record_count'] = record_table(handle,
                        maximum_block_bytes=self.limits.maximum_record_bytes)
                return item['record_count']
        _fail('record handle is no longer trusted', 'source_changed')

    def _validate_record(self, record, leaf, offset, *, manifest=None):
        source = manifest or self._manifest
        target = record.get('target')
        kwargs = {}
        parent_reader = None
        if target is not None and target.get('scope') == 'parent_checkpoint':
            parent_ref = source['lineage']['parent_state']
            if parent_ref is None:
                _fail('parent update has no actual lineage parent')
            parent_reader = self._parent_reader(parent_ref)
            kwargs = dict(parent_checkpoint=parent_reader._load_checkpoint(),
                parent_state=parent_ref, parent_manifest_ref=parent_ref['manifest'])
        try:
            _context_contract('validate_scientific_record',record,**kwargs)
            checked = validate_scientific_record(record, **kwargs)
            core = checked['core']['fields']
            stage, row = core['stage']['value'], core['row']
            if stage not in ('scba', 'outer') or row['type'] != ('SCBAIteration' if stage == 'scba' else 'OuterIteration'):
                _fail('scientific stage and typed row differ')
            counter = row['fields']['ν' if stage == 'scba' else 'μ']
            expected = core['inner_iteration'] if stage == 'scba' else core['outer_iteration']
            if (counter != expected or stage == 'outer' and core['inner_iteration'] is not None
                    or int(core['outer_iteration']['decimal']) < 1
                    or stage == 'scba' and int(expected['decimal']) < 1
                    or int(core['sequence']['decimal']) < 1
                    or int(core['domain_revision']['decimal']) < 0):
                _fail('scientific row counters/domain/sequence differ from snapshot')
            owner = self._index.get(leaf['path'])
            dependencies = set(owner.get('dependencies', ())) if owner is not None else set()
            for declaration in _arrays(checked):
                reference = declaration['file']
                if (reference['path'] not in dependencies or reference['path'] not in self._index
                        or _ref(self._index[reference['path']]) != reference):
                    _fail('record array owner absent from verified leaf dependency closure')
            if target is not None:
                target_leaf = _ref(leaf) if target['scope'] == 'same_leaf' else target['leaf']
                target_offset = int(target['row_offset'])
                if target['scope'] == 'same_leaf' and target_offset >= offset:
                    _fail('same-leaf update does not target an earlier physical append')
                if target['scope'] != 'parent_checkpoint':
                    target_owner = self._index.get(target_leaf['path'])
                    target_context = self._catalog_by_leaf.get((target_leaf['path'], target_leaf['sha256']))
                    if (target_owner is None or target_context is None
                            or _ref(target_owner) != target_leaf
                            or _ref(target_context[2]['artifacts'][-1]) != target_leaf):
                        _fail('update target FileRef differs from frozen catalog record leaf')
                    if target['scope'] == 'sealed_leaf':
                        current_context = self._catalog_by_leaf.get((leaf['path'], leaf['sha256']))
                        if (current_context is None or target_context[0] >= current_context[0]
                                or target_leaf['path'] not in current_context[2]['artifacts'][-1]['dependencies']):
                            _fail('sealed update target is not an earlier declared catalog dependency')
                    target_leaf = _ref(target_owner)
                original = self._record(target_leaf, target_offset, identity=target['source_identity'],
                    invocation=dict(source_epoch=target['source_epoch']))
                _context_contract('validate_scientific_record',original)
                validate_scientific_record(original)
                if (original['target'] is not None or original['core']['fields']['action']['value'] != 'append'
                        or original['row_key'] != checked['row_key']
                        or original['stream_ordinal'] != target['append_ordinal']):
                    _fail('update target differs from original append action/key/ordinal')
            return checked
        finally:
            if parent_reader is not None:
                parent_reader.close()

    def _cursor(self, leaf=None, offset=0, ordinal=0):
        return dict(schema='qcl-negf.result-cursor.v1', contract_set='qcl-negf.results.v2',
            identity=_context_copy(self._manifest['identity']), invocation=_context_copy(self._manifest['invocation']),
            manifest=_context_copy(self._state_ref['manifest']), history=_context_copy(self._manifest['history']),
            leaf=None if leaf is None else _ref(leaf), row_offset=str(offset), last_record=str(ordinal))

    def _cursor_page(self, cursor):
        if cursor['leaf'] is None:
            return None
        context = self._catalog_by_leaf.get((cursor['leaf']['path'], cursor['leaf']['sha256']))
        if context is not None:
            _, reference, page = context
            if _ref(page['artifacts'][-1]) == cursor['leaf']:
                return dict(reference=reference, document=page)
        _fail('cursor record leaf is not reachable from pinned source')

    def _check_cursor(self, cursor):
        _schema_check(schema_validator('result-cursor.schema.json'), cursor)
        context = self._cursor_page(cursor)
        validate_cursor(cursor, manifest=self._manifest, manifest_ref=self._state_ref['manifest'], catalog_page=context)
        if cursor['leaf'] is not None:
            row = self._record(cursor['leaf'], int(cursor['row_offset']),
                identity=self._manifest['identity'], invocation=self._manifest['invocation'], page=context['document'])
            self._validate_record(row, cursor['leaf'], int(cursor['row_offset']))
            if row['stream_ordinal'] != cursor['last_record']:
                _fail('cursor physical position differs from scientific ordinal')
            return row
        return None

    def history_batch(self, cursor=None, *, maximum_records=256):
        self._operation()
        if not self._catalog_ready:
            _fail('explicit catalog import required', 'capability_unavailable')
        if type(maximum_records) is not int or not 1 <= maximum_records <= self.limits.maximum_batch_records:
            raise ValueError('maximum_records exceeds configured batch limit')
        current = self._cursor() if cursor is None else _context_copy(dict(cursor))
        previous_row = self._check_cursor(current)
        previous_sequence = int(previous_row['core']['fields']['sequence']['decimal']) if previous_row else 0
        output, consumed, last = [], 0, int(current['last_record'])
        first_page = (self._catalog_by_leaf[(current['leaf']['path'], current['leaf']['sha256'])][0]
                      if current['leaf'] is not None else 0)
        for page_index in range(first_page, len(self._pages)):
            _, page = self._pages[page_index]
            if int(page['range']['last_record']) <= last:
                continue
            leaf = page['artifacts'][-1]
            start = int(current['row_offset']) + 1 if current['leaf'] == _ref(leaf) else 0
            handle = self._open_hdf(leaf, identity=self._manifest['identity'],
                invocation=self._manifest['invocation'], representation='record_leaf')
            count = self._record_count(handle)
            expected = int(page['range']['last_record']) - int(page['range']['first_record']) + 1
            if count != expected:
                _fail('physical record count differs from catalog range')
            for offset in range(start, count):
                raw = self._record(leaf, offset, identity=self._manifest['identity'],
                    invocation=self._manifest['invocation'], page=page)
                row = self._validate_record(raw, leaf, offset)
                if int(row['stream_ordinal']) != last + 1:
                    _fail('scientific record ordinal gap or replay')
                sequence = int(row['core']['fields']['sequence']['decimal'])
                if sequence <= previous_sequence:
                    _fail('Core delivery sequence replay or regression')
                previous_sequence = sequence
                proof = self._page_proofs.setdefault(leaf['sha256'],
                    dict(next_offset=0, scba_rows=0, outer_rows=0, witness_updates=0))
                if proof.get('invalid'):
                    _fail('actual scientific record categories differ from catalog totals')
                if offset == proof['next_offset']:
                    category = ('witness_updates' if row['core']['fields']['action']['value'] == 'witness_update'
                        else 'scba_rows' if row['core']['fields']['stage']['value'] == 'scba' else 'outer_rows')
                    proof[category] += 1
                    proof['next_offset'] += 1
                    if proof['next_offset'] == count:
                        for category in ('scba_rows','outer_rows','witness_updates'):
                            before = int(page['previous'][category]) if page['previous'] else 0
                            if proof[category] != int(page['totals'][category]) - before:
                                proof['invalid'] = True
                                _fail('actual scientific record categories differ from catalog totals')
                consumed += len(json_bytes(row))
                if consumed > self.limits.maximum_batch_bytes:
                    _fail('batch JSON exceeds byte limit', 'budget_exceeded')
                output.append(_context_copy(row))
                last += 1
                current = self._cursor(leaf, offset, last)
                if len(output) == maximum_records:
                    break
            if len(output) == maximum_records:
                break
        declarations = {_context_encoded(item): _context_copy(item) for row in output for item in _arrays(row)}
        if self._checkpoint is not None:
            declarations.update({_context_encoded(item): _context_copy(item) for item in _arrays(self._checkpoint)})
        self._declarations = declarations
        return HistoryBatch(_context_copy(self._state_ref), tuple(output), current,
            last == int(self._manifest['history']['last_record']), self.io_counters)

    def checkpoint(self):
        self._operation()
        return _context_copy(self._load_checkpoint())

    def _load_checkpoint(self):
        live=[]
        with _context_hold(live):
            if self._checkpoint is not None:
                return self._checkpoint
            reference = self._manifest['checkpoint']
            if reference is None:
                _fail('selected state has no checkpoint', 'capability_unavailable')
            raw_checkpoint=self._json(reference)
            _context_contract('validate_checkpoint_declaration',raw_checkpoint,manifest=self._manifest)
            checkpoint = validate_checkpoint_declaration(raw_checkpoint, manifest=self._manifest)
            live.append(checkpoint);del raw_checkpoint
            self._import_history()
            parent_reader, parent_checkpoint = None, None
            parent_ref = self._manifest['lineage']['parent_state']
            if parent_ref is not None:
                with self._parent_reader(parent_ref) as parent_reader:
                    parent_checkpoint = parent_reader._load_checkpoint()
            for ref in checkpoint['payloads']:
                if ref['path'] not in self._index or _ref(self._index[ref['path']]) != ref:
                    _fail('checkpoint payload differs from manifest dependencies')
            for target in checkpoint['witness_targets']:
                locator = target['locator']
                local = (locator['source_identity'] == self._manifest['identity']
                         and locator['source_epoch'] == self._manifest['invocation']['source_epoch'])
                if local:
                    if not any(_ref(page['artifacts'][-1]) == locator['leaf'] for _, page in self._pages):
                        _fail('checkpoint target is not in frozen scientific catalog')
                elif parent_checkpoint is None or target not in parent_checkpoint['witness_targets']:
                    _fail('ancestor checkpoint target is not retained by actual immediate parent')
                row = self._record(locator['leaf'], int(locator['row_offset']), identity=locator['source_identity'],
                    invocation=dict(source_epoch=locator['source_epoch']))
                _context_contract('validate_scientific_record',row)
                validate_scientific_record(row)
                if (row['target'] is not None or row['row_key'] != target['row_key']
                        or row['stream_ordinal'] != locator['append_ordinal']):
                    _fail('checkpoint witness locator differs from original append')
            self._checkpoint = checkpoint
            if _context_arena.get() is not None:
                allowance=sum(_context_weight(item)+16*_context_json_size(item)+256 for item in _arrays(checkpoint))
                _context_before('checkpoint declaration copies',extra=allowance)
            self._declarations.update({_context_encoded(item): _context_copy(item) for item in _arrays(checkpoint)})
            return checkpoint

    def _parent_reader(self, reference):
        parent = StateReader.__new__(StateReader)
        parent._initialize(self.root, state_ref=reference, expected_identity=reference['identity'],
            limits=self.limits, budget=self._budget, lineage_depth=self._lineage_depth + 1,
            metadata_bindings=self._metadata_bindings,
            metadata_owner=self._metadata_context_owner or self)
        parent._consumer_owner_limit = getattr(self, '_consumer_owner_limit', None)
        return parent

    def _parent_anchors(self):
        """Authenticated parent metadata only; no HDF or prefix payload scan."""
        live=[]
        with _context_hold(live):
            if self._parent_anchor_cache is not None:
                return self._parent_anchor_cache
            reference = self._manifest['lineage']['parent_state']
            anchors = {}
            live.append(anchors)
            if reference is not None:
                with self._parent_reader(reference) as parent:
                    checkpoint_ref = parent._manifest['checkpoint']
                    if checkpoint_ref is None:
                        _fail('lineage parent has no exact checkpoint declaration')
                    original=parent._json(checkpoint_ref)
                    _context_contract('validate_checkpoint_declaration',original,manifest=parent._manifest)
                    checkpoint = validate_checkpoint_declaration(original, manifest=parent._manifest)
                    live.append(checkpoint);del original
                    for target in checkpoint['witness_targets']:
                        leaf = target['locator']['leaf']
                        prior = anchors.get(leaf['path'])
                        if prior is not None and prior != leaf:
                            _fail('parent anchors rebound an immutable path')
                        anchors[leaf['path']] = _context_copy(leaf)
            self._parent_anchor_cache = anchors
            return anchors

    def _array(self, declaration):
        key = json_bytes(declaration)
        if key not in self._declarations:
            _fail('array descriptor not from retained verified state/records', 'source_changed')
        declaration = self._declarations[key]
        reference = declaration['file']
        if reference['path'] not in self._index or _ref(self._index[reference['path']]) != reference:
            _fail('array owner not in verified source dependencies')
        handle = self._open_hdf(self._index[reference['path']], identity=self._manifest['identity'],
            invocation=self._manifest['invocation'], representation='array_payload')
        from .native import validate_native_array
        dataset = validate_native_array(handle, declaration)
        return declaration, dataset

    def describe_array(self, declaration):
        self._operation()
        wire, dataset = self._array(declaration)
        return ArrayDescription(_context_copy(wire), tuple(int(v) for v in wire['shape']),
            tuple(dataset.shape), wire['eltype'], dataset.dtype.itemsize, 'F', _context_copy(self._state_ref))

    def read_array(self, declaration, selection, *, maximum_bytes=1024*1024):
        self._operation()
        wire, dataset = self._array(declaration)
        return self._read_selected_array(wire, dataset, selection, maximum_bytes=maximum_bytes)

    def _read_selected_array(self, wire, dataset, selection, *, maximum_bytes):
        """Shared bounded selection; caller authenticates the descriptor owner."""
        if type(maximum_bytes) is not int or not 0 < maximum_bytes <= self.limits.maximum_array_bytes:
            raise ValueError('maximum_bytes exceeds configured array read limit')
        if not isinstance(selection, tuple) or len(selection) != dataset.ndim:
            raise ValueError('one explicit selector required per logical axis')
        normalized, counts = [], []
        for item, size in zip(selection, dataset.shape):
            if type(item) is int:
                if not 0 <= item < size:
                    raise IndexError('array index outside logical shape')
                normalized.append(item)
            elif isinstance(item, slice):
                if item.step is not None and item.step <= 0:
                    raise ValueError('array selection requires positive step')
                a, b, step = item.indices(size)
                normalized.append(slice(a, b, step))
                counts.append(len(range(a, b, step)))
            else:
                raise ValueError('array selectors must be integers or slices')
        required = math.prod(counts) * dataset.dtype.itemsize
        if required > maximum_bytes:
            _fail('selected array buffer exceeds limit', 'budget_exceeded')
        if dataset.chunks and math.prod(dataset.chunks)*dataset.dtype.itemsize > self.limits.maximum_array_bytes:
            _fail('native array chunk decode buffer exceeds limit', 'budget_exceeded')
        # Local analysis descriptors have no FileRef: bind to the real opened
        # HDF owner, shared with Core arrays, rather than infer a filename.
        owner = next(((key, item) for key, item in self._handles.items()
                      if item['handle'].id == dataset.file.id), None)
        if owner is None:
            _fail('selected array has no retained verified owner', 'source_changed')
        self._check_hdf_owner(*owner)
        values = np.asarray(dataset[tuple(normalized)])
        values.flags.writeable = False
        self._check_hdf_owner(*owner)
        self._budget.logical_selected_bytes += values.nbytes
        return ArrayBlock(values, tuple(normalized), _context_copy(wire), _context_copy(self._state_ref), self.io_counters)

    def poll(self):
        self._operation()
        if self._pointer_path is None:
            return self.source
        from ._capture_io import beneath, small_json
        value, raw = small_json(beneath(self.root, self._pointer_path), self.limits.maximum_metadata_bytes, budget=self._budget)
        if raw == self._pointer_raw:
            return self.source
        pointer = validate_pointer(value)
        if pointer['identity'] != self._manifest['identity']:
            _fail('selected pointer source replaced', 'source_changed')
        candidate = validate_manifest(self._json(pointer['manifest']))
        if (candidate['identity'] != self._manifest['identity'] or candidate['invocation'] != self._manifest['invocation']
                or candidate['lineage'] != self._manifest['lineage']
                or candidate['provenance'] != self._manifest['provenance']
                or int(candidate['generation']) <= int(self._manifest['generation'])
                or int(candidate['state_sequence']) <= int(self._manifest['state_sequence'])
                or candidate['generation'] != pointer['generation']):
            _fail('pointer source/regression differs from current invocation', 'source_changed')
        old_manifest, old_ref = self._manifest, self._state_ref
        changed_paths = {pointer['manifest']['path'], *(row['path'] for row in candidate['artifacts'])}
        original_entries = {path: self._index.get(path) for path in changed_paths}
        original_weight = self._index_weight
        self._manifest = candidate
        self._state_ref = {key: candidate[key] for key in ('identity', 'state_id', 'state_sequence')}
        self._state_ref['manifest'] = pointer['manifest']
        self._pending_state_weight = _metadata_weight(self._state_ref) + _metadata_weight(candidate)
        try:
            self._register_manifest()
            if self._catalog_ready:
                self._import_history()
        except Exception:
            self._manifest, self._state_ref = old_manifest, old_ref
            self._index_weight, self._pending_state_weight = original_weight, 0
            for path, value in original_entries.items():
                if value is None:
                    self._index.pop(path, None)
                else:
                    self._index[path] = value
            raise
        self._checkpoint, self._declarations = None, {}
        self._pointer_raw = raw
        self._remember_state()
        return self.source

    def rebase_cursor(self, cursor):
        self._operation()
        _schema_check(schema_validator('result-cursor.schema.json'), cursor)
        digest = cursor['manifest']['sha256']
        if digest not in self._known_states:
            _fail('cursor source not observed in this invocation', 'source_changed')
        ref, manifest = self._known_states[digest]
        if cursor['identity'] != self._manifest['identity'] or cursor['invocation'] != self._manifest['invocation']:
            _fail('cursor source replaced', 'source_changed')
        validate_cursor(cursor, manifest=manifest, manifest_ref=ref['manifest'], catalog_page=self._cursor_page(cursor))
        rebased = _context_copy(dict(cursor))
        rebased['manifest'], rebased['history'] = _context_copy(self._state_ref['manifest']), _context_copy(self._manifest['history'])
        self._check_cursor(rebased)
        return rebased


def verify_recovery_bundle(*args, **kwargs):
    """Retired v1 directory API; use explicit StateRef and checkpoint()."""
    _fail('legacy recovery directory unsupported; select result-v2 StateRef', 'unsupported_format')
