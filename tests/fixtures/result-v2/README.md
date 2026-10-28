# Native5 constructor fixtures and independent reader oracles

The portable tests use `tests/result_v2_fixture.py` to manufacture small canonical result-v2
manifests/catalogs and native5 files. They never construct a solver or represent a scientific run.
Shape, tagged scalar/IEEE values, identity, array ownership, cursor position, negative cases and
budgets are checked independently in `test_result_v2_reader.py`, `test_result_v2_cursor.py` and
`test_result_v2_export.py`.

For the separate actual Julia producer packet, set `QCL_RESULT_V2_FIXTURE_ROOT` explicitly. The P06
acceptance packet uses P04 F2 constructor output, Runner source `72df0c` published by merge
`b1c1af789685b24af8f35810ad76e32f0a0aae06`, Contracts `5167810d0a1ce436dae1b7f101de681463e432a6`,
and Core layout SHA256 `94561e874a62ad0749be8a13645ef0142994e074b33d271cbd57807f2cf63600`. Its
separately retained `P04-F2-fixtures.json` has SHA256
`0d1d3bda18819ab05ac62c3c66a4e35c9241fae4b4d8acb744536d41ea95cf54`. The external 7,701,135-byte,
356-entry fixture packet intentionally contains negative files and orphans. Do not recursively
import it as a healthy source or replace it with historical scientific arrays. Binary fixtures stay
outside Git.

The producer and Python reader do not share an encoder. Python literal checks cover the complete
SCBA scalar row, all 16 snapshot fields, units/missing reasons, outer scalars, both full
populations/density vectors, exact-state schema and selected complex witness matrices/eigenvectors.
The independent codec oracle checks signed zero, infinities, distinct NaN payloads, nonsquare
complex arrays, rank-four axis positions, empty axes, Bool enum and Int64 extremes. H5py shape
already equals logical shape; no Python transpose is allowed. Unsupported ranks are negative tests
against the published 129-node graph.

`copy_constructor_closure` follows only explicit authenticated FileRefs and has a 16 MiB test-only
admission cap. It supports fresh thin history-only child cases and missing/tamper/identity negatives
without globbing old payloads. Tests of parent/grandparent updates bind the actual immediate parent
and retained original append; iteration counters are not source identity.

Full-state export/receive is exercised against actual producer graph and the same public reader.
Science subset tests keep optional full owners omitted and preserve original canonical JSON as
provenance without claiming restoration. Storage tests separately exercise a 450 MB manufactured
native array and an incompressible transport above 200 MB using bounded buffers. These constructor
checks establish L-level transport/reader behavior, not numerical restart, scientific convergence,
physical validation, installed composition or service readiness. Native5 independent
analysis/saved/diagnostic/equilibrium views remain a mandatory owning followup before research
release acceptance.
