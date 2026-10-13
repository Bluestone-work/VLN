# Native single-action full-return audit

HEAD: `2963a0f33d757250c709d401ef0f7309ada71b84`

The frozen source is `research/results/semantic_node_memory_multiview_001`. The
fail-closed audit is `research/results/native_single_action_full_return_44_002/exact_audit_003/`.
It checks every disabled native trajectory against the source trace and evaluator,
then checks each enabled branch's exact event key, GraphMap digest, policy RNG,
pre-pose, pre-metrics, native prefix, action construction, and one-time
alternative intervention. A failed check excludes the state; no approximate join
is performed.

## Causal dataset

- source episodes/routes: 44
- source scenes: 11
- source decision states: 373
- states with admissible alternatives: 369
- alternative actions: 3,878
- schedules: 215; runs: 430
- BENEFIT: 430
- HARM: 3,447
- UNCERTAIN: 1

Audit status: **PASS**. All 3,878 states are valid. Disabled traces are byte
identical to the 373-row source trace; all disabled evaluator dictionaries match
the source evaluator. The current checkpoint hash is recorded as provenance; the
historical manifests contain the same checkpoint path and checkpoint identity.

## Fixed label

`success > failure`, then higher nDTW, higher SDTW, shorter path, fewer
primitives, and fewer collision events. Deadbands are fixed at `1e-6` for nDTW/
SDTW and `1e-5` for path cost. Labels are not tuned from probe results.

## Provenance

The pair table stores final metrics only under outcome fields. Online features
come from the current node's native graph record and the source capture's
pre-action 360° observation. Alternative endpoint observations, goal distance,
reference route and final outcome are excluded from features.
