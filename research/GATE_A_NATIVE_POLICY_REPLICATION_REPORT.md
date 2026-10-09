# Gate A Native-Policy Control Replication

## Research question

Does the dense candidate proposal gain remain safe on routes where the unchanged native ETPNav policy already succeeds?

## Frozen cohort and protocol

The cohort was frozen from two previously frozen outcome-blind samplers: 64 train routes across 16 scenes and 66 val-unseen routes across 11 scenes. The 31 Gate-A routes were removed by episode ID before any result from this replication was read. All remaining 99 routes were retained: 55 train routes and 44 val-unseen routes across 27 scenes. Selection used only source membership, episode ID, and scene metadata.

A0 used the released waypoint NMS (`max_predictions=5`, `sigma=(7,5)`). A1 used the same heatmap with the frozen `A0 union dense-NMS` configuration (`max_predictions=12`, `sigma=(4,3)`). Both arms used the same released checkpoint, seed 100, sensors, controller, STOP logic, 15-decision budget, and evaluator. A1 passed every candidate through the native graph encoder and SAP head and selected native argmax. No oracle return, reference route, goal, classifier, reranker, or learned selector was used.

The native/dense ghost merge pools were provenance-separated in A1 so that the final graph action set remained nested. The matched-prefix audit found zero A0-subset violations across all 99 routes.

## Results

| Cohort | Routes | Scenes | A0 SR | A1 SR | Rescued | Destroyed | SPL delta | nDTW delta | Path delta | Primitive delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Train controls | 55 | 16 | 55/55 (1.000) | 52/55 (0.945) | 0 | 3 | -0.1072 | -0.0489 | +1.34 | +9.95 |
| Val-unseen controls | 44 | 11 | 44/44 (1.000) | 29/44 (0.659) | 0 | 15 | -0.2562 | -0.0722 | +2.40 | +19.09 |
| All controls | 99 | 27 | 99/99 (1.000) | 81/99 (0.818) | 0 | 18 | -0.1734 | -0.0593 | +1.81 | +14.01 |

The harm is therefore not confined to the earlier 16-route control sample. It is stronger on the unseen scenes: 15/44 native successes are destroyed. The all-control evaluator aggregates are:

| Arm | SR | SPL | nDTW | SDTW | Path length | Primitive steps | Collisions | High-level steps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A0 | 1.000 | 0.874 | 0.825 | 0.825 | 9.82 | 62.84 | 0.0648 | 7.61 |
| A1 | 0.818 | 0.701 | 0.766 | 0.689 | 11.63 | 76.85 | 0.0973 | 7.53 |

## Provenance and cost

Across captured decisions, A0 exposed a mean of 4.68 heatmap proposals and A1 exposed 12.95, including 8.27 newly added proposals. A1 directly selected a new target at the first matched divergence on 96/99 routes. The rate was similar for retained routes (79/81) and harmed routes (17/18), so simple candidate count or the mere presence of a new candidate does not explain the harm.

| Group | Routes | Mean A1 candidates | Mean new candidates | Mean SPL delta | Mean nDTW delta | Mean primitive delta | Mean collision delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| A1 retained success | 81 | 12.88 | 8.22 | -0.0388 | -0.0210 | +11.10 | +0.0196 |
| A1 destroyed success | 18 | 13.28 | 8.49 | -0.7790 | -0.2317 | +27.11 | +0.0908 |

This is a policy-safety failure: exposing more actions to an unchanged argmax policy causes it to choose harmful alternatives on native-success routes. The dense set is not merely unused; it changes decisions and downstream graph execution.

## Decision

**NO-GO for unconditional proposal expansion with the unchanged native policy.**

The earlier 31-route unresolved experiment established a limited online rescue opportunity. This 99-route outcome-blind replication establishes the necessary counter-result: dense candidates destroy 18/99 native successes, with larger harm on held-out scenes. The combined evidence does not support deploying A1 directly or claiming a general proposal-coverage improvement.

Do not train a waypoint predictor, PPO/RL policy, VLM, or generic reranker from this result. The proposal-expansion branch should remain closed unless a separately justified safety mechanism can abstain from harmful new candidates and is validated on an independently frozen cohort. The previous Gate-B/C failures mean that mechanism is not currently available.

## Artifacts

- [`GATE_A_NATIVE_POLICY_REPLICATION_REPORT.md`](/home/wj/VLN-CE/ETPNav/research/GATE_A_NATIVE_POLICY_REPLICATION_REPORT.md)
- [`summary.json`](/home/wj/VLN-CE/ETPNav/research/results/gate_a_native_policy_replication_001/analysis_001/summary.json)
- [`per_route.csv`](/home/wj/VLN-CE/ETPNav/research/results/gate_a_native_policy_replication_001/analysis_001/per_route.csv)
- [`per_scene.csv`](/home/wj/VLN-CE/ETPNav/research/results/gate_a_native_policy_replication_001/analysis_001/per_scene.csv)
- [`manifest.json`](/home/wj/VLN-CE/ETPNav/research/results/gate_a_native_policy_replication_001/manifest.json)

Full local runs remain under `research/results/gate_a_native_policy_replication_001/runs/`; evaluator JSON is written under `data/logs/eval_results/gate_a_native_policy_replication_*`.
