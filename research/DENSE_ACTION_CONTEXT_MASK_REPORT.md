# A2-MASK Dense Context Only Report

## Experimental question

How much of dense-action harm remains when dense ghosts participate in graph encoding and SAP scoring, but dense actions are masked from final execution?

The frozen cohort is the 44-route, 11-scene val-unseen native-success control replication. Checkpoint, seed, route file, controller, STOP mechanism, sliding, evaluator, success threshold, and decision budget are unchanged.

## Four arms

| Arm | Graph context | Executable set |
|---|---|---|
| A0 Native | native | native |
| A1 Dense Full | native + dense | native + dense |
| A2-ISO Native Isolation | native + dense | native + dense; native uses isolated score |
| A2-MASK Dense Context Only | native + dense | native only after full SAP scoring |

## Navigation results

| Arm | SR | SPL | nDTW | SDTW | Path | Primitive | Collision |
|---|---:|---:|---:|---:|---:|---:|---:|
| A0 | 44/44 | 0.8182 | 0.7451 | 0.7451 | 11.074 | 75.273 | 0.107 |
| A1 | 29/44 | 0.5620 | 0.6728 | 0.5430 | 13.475 | 94.364 | 0.135 |
| A2-ISO | 33/44 | 0.5862 | 0.6482 | 0.5714 | 12.711 | 96.523 | 0.210 |
| A2-MASK | 34/44 | 0.6452 | 0.6864 | 0.5819 | 12.089 | 82.227 | 0.090 |

Destroyed native-success routes:

- A1: 15 routes: `14, 166, 382, 392, 413, 730, 776, 931, 1026, 1093, 1149, 1552, 1608, 1708, 1778`
- A2-ISO: 11 routes: `166, 392, 730, 776, 931, 1026, 1093, 1149, 1552, 1608, 1708`
- A2-MASK: 10 routes: `14, 370, 381, 382, 640, 833, 931, 1093, 1149, 1608`

A2-MASK blocks the direct dense-action path completely, yet still loses 10/44 native successes. This is direct closed-loop evidence that representation contamination and native ranking drift alone can cause substantial harm. It also improves over A1 in this cohort, but not enough to approach A0.

## Mask fidelity

- 357/357 captured decision rows contain full-context logits and native masks.
- Dense selected actions: `0`.
- Rows violating the native execution mask: `0`.
- Dense options remain present in the graph and have full-context scores in the trace; they are only marked non-admissible after scoring.

## Scene-cluster bootstrap

10,000 resamples over 11 scenes were used. 95% intervals for metric deltas are:

| Comparison | SR | SPL | nDTW | Primitive | Collision |
|---|---|---|---|---|---|
| A2-MASK - A0 | [-0.348, -0.082] | [-0.284, -0.068] | [-0.140, -0.020] | [-4.93, 32.24] | [-0.044, 0.006] |
| A2-MASK - A1 | [-0.033, 0.292] | [-0.050, 0.239] | [-0.081, 0.117] | [-60.59, 5.24] | [-0.066, -0.018] |
| A2-ISO - A1 | [0.000, 0.164] | [-0.060, 0.081] | [-0.077, 0.022] | [-58.07, 25.48] | [0.001, 0.140] |

The primary A2-MASK-A0 SR and nDTW intervals exclude zero. The A2-MASK-A1 improvement is uncertain on this 11-scene cohort, while collision reduction relative to A1 is more stable.

## Mechanism conclusion

This is **Case C/H1+H2 coupled** in the protocol terminology:

- H1 is independently harmful: A2-MASK cannot execute dense actions but still destroys 10 native successes.
- H2 remains harmful: A1 destroys 15, and A2-ISO only partially recovers to 11; dense scores are still not calibrated for the expanded action distribution.
- A2-MASK does not support a shared-graph safe override architecture. Dense candidates need a shadow representation and a separate safety decision before they can affect execution.

**NO-GO** for unconditional dense expansion, ranker training, g3D-LF, or RL. The next task is the pre-registered single-override causal dataset.

## Reproduction

- Protocol: [DENSE_ACTION_CONTEXT_MASK_PROTOCOL.md](/home/wj/VLN-CE/ETPNav/research/DENSE_ACTION_CONTEXT_MASK_PROTOCOL.md)
- Config: [DENSE_A2_MASK_CONTROL_VAL_UNSEEN_001.json](/home/wj/VLN-CE/ETPNav/research/configs/DENSE_A2_MASK_CONTROL_VAL_UNSEEN_001.json)
- Analyzer: [analyze_dense_context_mask.py](/home/wj/VLN-CE/ETPNav/research/tools/analyze_dense_context_mask.py)
- Results: `research/results/dense_a2_mask_control_val_unseen_001/`
