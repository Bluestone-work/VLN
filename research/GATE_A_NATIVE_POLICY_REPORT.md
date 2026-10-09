# Gate A: Dense Candidates With Unchanged Native ETPNav Policy

## Question

Can proposal coverage from a denser candidate set be converted into closed-loop navigation improvement when the released ETPNav graph encoder, SAP head, argmax policy, controller, STOP logic, checkpoint, seed, sensors, and 15-decision budget remain unchanged?

## Frozen protocol

The experiment uses the frozen Gate-A population: 31 unresolved routes in 18 scenes plus 16 metadata-only native-success controls in 16 scenes. The checkpoint is `data/logs/checkpoints/release_r2r/ckpt.iter12000.pth`, seed is `100`, and both arms use the same evaluator and controller settings.

Arm A0 uses the released waypoint NMS (`max_predictions=5`, `sigma=(7,5)`). Arm A1 uses the same heatmap with `A0 union dense-NMS` (`max_predictions=12`, `sigma=(4,3)`). Every A1 candidate is scored by the original graph encoder and SAP head; the selected action remains native argmax. No oracle return, route, goal, or external selector is used.

The graph implementation maintains a provenance-separated native ghost subgraph in A1. Native proposals are merged only with native ghosts and dense proposals only with dense ghosts. This is an implementation guard for the required nested action-set comparison: across every matched prefix state, the target geometry of every admissible A0 graph action is present in A1. It does not alter A0 behavior, the encoder, the head, or action selection.

## Closed-loop results

| Cohort | Routes | Scenes | A0 SR | A1 SR | Rescued | Destroyed | Mean SPL delta | Mean nDTW delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Gate-A unresolved | 31 | 18 | 0/31 (0.000) | 9/31 (0.290) | 9 | 0 | +0.2245 | +0.0762 |
| Native-success controls | 16 | 16 | 16/16 (1.000) | 13/16 (0.813) | 0 | 3 | -0.2130 | -0.0954 |
| All paired routes | 47 | 34 | 16/47 (0.340) | 22/47 (0.468) | 9 | 3 | +0.0756 | +0.0178 |

On the unresolved population, A1 directly selected a new candidate at the first target divergence on all 31 routes. The candidate-set nesting audit found zero violations over all matched prefix states. The controls show a material regression risk: 3 of 16 native successes were destroyed, and average SPL/nDTW decreased on controls. Therefore this is evidence for an online proposal-coverage opportunity, not a generally safe improvement.

Candidate provenance averaged 4.45 (train) / 4.53 (unseen) candidates per A0 decision and 12.73 / 12.65 per A1 decision. The graph action set is larger because historical ghosts are retained; those counts are reported separately in `analysis_002/per_route.csv` and should not be confused with current heatmap proposal counts.

## Required routes

| Episode | Cohort | A0 | A1 | Interpretation |
|---|---|---:|---:|---|
| `8343` | train | fail | fail | A1 did not realize the earlier oracle opportunity |
| `1052` | val_unseen | fail | success | online native policy realized a dense-candidate rescue |
| `1584` | val_unseen | fail | fail | dense proposal coverage alone was insufficient |

The complete paired route table is in [`per_route.csv`](/home/wj/VLN-CE/ETPNav/research/results/gate_a_native_policy/analysis_002/per_route.csv). Machine-readable aggregate results are in [`policy_summary.json`](/home/wj/VLN-CE/ETPNav/research/results/gate_a_native_policy/analysis_002/policy_summary.json), with scene aggregation in [`per_scene.csv`](/home/wj/VLN-CE/ETPNav/research/results/gate_a_native_policy/analysis_002/per_scene.csv).

## Interpretation and gate

**GO for a proposal-coverage mechanism as a diagnostic opportunity, with a safety caveat.** The unchanged native policy can exploit newly available candidates: 9/31 frozen unresolved routes are rescued online, and the gain is present in both train and held-out source cohorts. This is stronger than the privileged Gate-A ceiling alone because the result comes from complete reset-to-termination rollouts.

The result does not justify claiming a benchmark improvement or immediately changing the waypoint predictor. The control harm (3/16 destroyed successes), longer paths (+3.53 mean primitive/path-length units over all routes), and uneven realization of oracle opportunities mean the next experiment must be a larger outcome-blind native-success control replication and a provenance-level analysis of why native argmax sometimes chooses harmful dense candidates. No classifier, RL, VLM, or navigator redesign is warranted by this gate alone.

## Reproducibility

The four final rollouts are stored locally under `research/results/gate_a_native_policy/runs/{train_a0_v2,train_a1_v6,unseen_a0_v2,unseen_a1_v6}`. The committed reproducibility subset is `final_manifests/` plus `evaluator/`; the full local runs retain stdout/stderr, graph-option JSONL, traces, and checkpoint/config provenance.
