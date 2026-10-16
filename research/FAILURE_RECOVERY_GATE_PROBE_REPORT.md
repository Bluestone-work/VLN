# Failure recovery gate probe

HEAD: `2dbfe7f`.

This is a development-only probe on the existing outcome-enriched critical
failure cohort. It is not an unbiased benchmark: native failures and critical
states were selected before this analysis. No policy, controller, Graph
Transformer or STOP rule was changed, and no online intervention was run.

## Data

The source is `critical_causal_unseen/full_001/cases/`:

- 71 critical states from 22 native-failure episodes and 10 scenes;
- 985 exact full-return branches;
- after removing 71 branches tied with native, 914 labeled branches;
- 275 BENEFIT and 639 HARM branches;
- 50 of 64 evaluable states contain at least one BENEFIT branch.

Labels are fixed full-return comparisons. Features are pre-action only:
native candidate logit and margin, STOP/proposal flags, graph size,
backtracking length, candidate graph type and geometry, high-level step,
pre-action primitive count and pre-action collision count. Endpoint
observations, goal distance, reference path and final outcome are excluded
from features.

## Scene-disjoint probe

| Model | AUROC | AUPRC | Within-state pairwise |
| --- | ---: | ---: | ---: |
| Native margin / STOP fields | 0.5304 | 0.4032 | 0.6497 |
| Native + graph / geometry / pre-action history | 0.4941 | 0.3294 | 0.5444 |

The first block contains only the native candidate score, selected score,
margin, STOP and proposal flags. The second adds graph topology, candidate
geometry, backtracking length and pre-action execution counters. Both use
leave-one-scene-out fitting with inverse-state and class balancing.

The native-only block ranks a BENEFIT action above a HARM action in 64.97% of
within-state pairs. Adding the extra fields reduces this to 54.44%. This does
not support a safe late gate based on native uncertainty, graph geometry or
recent execution counters.

## Decision

**NO-GO for this simple recovery gate.** The failure cohort demonstrates real
causal rescue opportunities, but the tested pre-action fields do not reliably
identify them. The high pooled opportunity rate must not be interpreted as a
deployable policy signal because this cohort was selected after failure.

The next single high-value experiment is a fresh scene-disjoint failure cohort
with a frozen representation of ordered instruction progress plus a short
action-history sequence. The representation should be tested offline first,
using the same exact full-return labels. It should include explicit
abstention, and it must pass a native-success retention gate before any shadow
or intervention run. If that richer state still has near-chance candidate
ranking, stop the waypoint-recovery direction rather than changing VLMs or
tuning thresholds.

Artifacts:

- `research/tools/probe_failure_recovery_gate.py`
- `research/results/failure_recovery_gate_probe_002/summary.json`
- `research/results/failure_recovery_gate_probe_002/rows.jsonl`
