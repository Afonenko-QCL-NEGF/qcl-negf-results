# Native5 saved views: staged capability

The result-v2 reader exposes complete typed records and bounded array slices through `StateReader`;
see the [public reader examples](../README.md). `AnalysisReader`, `describe_saved_views` and
`read_saved_view` currently return `capability_unavailable` because a verified independent native5
analysis representation has not yet been published. They do not interpret native4 paths inside a
native5 payload or start an implicit derive.

This restriction is a staged migration. Required saved observables, axes, units, weights,
source-state binding and independent analysis evidence must be restored by the owning Runner/Results
research-view followup before final installed research acceptance. A raw tensor declaration does not
supply missing scientific axis semantics automatically. Existing mathematical definitions remain
requirements; unsupported capability is not scientific failure, successful analysis or permanent
permission to remove the view.

`StateReader.source.capabilities` and separate status evidence make this limitation visible to
Portal providers. A missing required selected reference is an integrity error; an unavailable
optional representation has an explicit capability reason.
