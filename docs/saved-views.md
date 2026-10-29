# Native5 saved views: staged capability

The result-v2 reader exposes complete typed records and bounded array slices through `StateReader`;
see the [public reader examples](../README.md). `AnalysisReader`, `describe_saved_views` and
`read_saved_view` currently return `capability_unavailable` while their native5 semantic adapter is
still being migrated. The explicit raw `FinalObservationsReader` already reads the published
`qcl-negf.final-observations.v1` metadata and 26 stored local arrays from one StateReader. They do
not interpret native4 paths inside a native5 payload or start an implicit derive.

This restriction is a staged migration. Required saved observables, axes, units, weights,
source-state binding and independent analysis evidence must be restored by the owning Runner/Results
research-view followup before final installed research acceptance. A raw tensor declaration does not
supply missing scientific axis semantics automatically. Existing mathematical definitions remain
requirements; unsupported capability is not scientific failure, successful analysis or permanent
permission to remove the view.

`StateReader.source.capabilities` and separate status evidence make this limitation visible to
Portal providers. A missing required selected reference is an integrity error; an unavailable
optional representation has an explicit capability reason.

## Raw observations and original input proof

Import `FinalObservationsReader` or `describe_final_observations` from
`qcl_negf_results.final_observations`. Both require the already selected `StateReader` and explicit
`artifact_ref` (all five FileRef fields). This raw reader returns stored observables and context; it
does not reconstruct a private Core problem, diagonalize a Hamiltonian, derive density or certify
science. Original terminal evidence, current revalidation and nullable scientific acceptance remain
separate.

The closed local arrays have ranks one/two, exact LE Float64/ComplexF64/Int64 or unsigned Boolean
enum, logical axes and coordinate/weight references from the owning schema. The native guard
requires fixed UTF8 scalar JSON metadata, internal storage and the producer's contiguous or
shuffle/deflate profile. Selected decoded chunk and returned-buffer bounds apply even to scalar or
empty selections; these limits do not claim a bound on total process RSS or native-library scratch.

Plan/model files are opened through the same reader allowance and checked on their actual nofollow
FD and path before publication. Model canonical identity is independent of the file SHA. Nested YAML
is admitted by bounded events before object construction: no aliases, anchors, merge keys, duplicate
keys, tag directives, arbitrary constructors or nonfinite values. Portable standard tags and scalar
resolution follow exact YAML.jl0.4.17 primary source rather than PyYAML's default implicit resolver;
21 literal portable trees/type distinctions/Float64 bits separately match actual
Julia1.13/YAML0.4.17 output. Full original-plan interoperability remains a separate gate. Binary
integer spellings are rejected consistently with the owning YAML constructor; they are never
silently reinterpreted. JSON plans use the strict duplicate-free, finite decoder and exact
scientific-plan/request joins.

For completed restore, the immediate `lineage.parent_state`, parent checkpoint, original evidence
and frozen history pin are verified under the shared allowance. The reader does not search another
attempt or a newer checkpoint. A valid raw read is not exact-restore numerical equivalence,
independent equilibrium acceptance or a complete scientific archive export.
