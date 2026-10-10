# Dense Action OOD and Calibration Audit

This is a read-only analysis of the frozen val-unseen A0/A1/A2 control traces. It does not fit a model, choose a threshold, or use continuation outcomes as features.

## Alignment

Native actions are defined by exact action signatures present in the A0 admissible set at the same prefix. A1 actions absent from that set are classified as dense. Prefixes are truncated at the first A0/A1 selected-action divergence; later rows are excluded because they describe different states.

The audit contains 72 common-prefix decision states across 43 first-divergence states. This is smaller than the raw 301-row A1 trace because post-divergence states are intentionally excluded.

## Main result

At the first A0/A1 divergence:

- 42/43 states (`97.7%`) had the highest dense score above the highest native score;
- mean dense-minus-native top-score gap was `1.461`;
- median gap was `1.427`;
- 42/43 first divergences selected a dense action;
- mean native absolute full-vs-isolated logit shift on the aligned states was `1.943`.

This is direct evidence for a ranking/calibration failure on the expanded action distribution after accounting for the separate representation contamination effect. Dense actions are not merely being selected because native actions changed: at the decision where A1 first departs from A0, the dense score distribution almost always outranks the native distribution.

## Outcome-stratified evidence

| aligned group | states | dense top > native top | mean top gap | mean native shift |
|---|---:|---:|---:|---:|
| all common prefixes | 72 | 58.3% | 0.534 | 1.943 |
| first A0/A1 divergence | 43 | 97.7% | 1.461 | 1.943 |
| prefixes of A1-destroyed routes | 21 | 66.7% | 0.912 | 2.017 |
| prefixes of A1-retained routes | 51 | 54.9% | 0.379 | 1.913 |
| prefixes of A2-destroyed routes | 15 | 66.7% | 1.171 | 1.897 |

The destroyed/retained split is descriptive and route-level outcomes are used only after feature extraction. It is not a trained classifier result. The small number of aligned destroyed prefixes limits precision, but the first-divergence result is strong and mechanism-specific.

## Interpretation

The two mechanisms are both present:

1. **Representation contamination (H1):** full-graph insertion changes native embeddings/logits. On aligned states, the mean absolute native logit shift is `1.943`.
2. **Dense-action OOD/calibration failure (H2):** at the first policy divergence, dense actions almost universally occupy the top score position, with a positive mean score gap of `1.461`.

A2 addresses only the first mechanism for native actions and reduced native-success destruction from 15/44 to 11/44 in the navigation control. It does not calibrate dense scores, and it remains unsafe by the navigation criteria. The current evidence therefore does not justify training a selector or adding g3D-LF as a feature source.

## Decision

**NO-GO for unconditional dense action expansion and for immediate ranker/RL training.** The next valid research step is a pre-registered representation/calibration intervention that constrains dense scores using a frozen calibration rule, evaluated on a new scene-disjoint cohort. It must be tested as a safety experiment with native-success preservation as the primary gate, not tuned on this cohort.

## Reproduction

Command:

```bash
python research/tools/analyze_dense_action_calibration.py
```

Outputs:

- `research/results/dense_action_calibration_001/summary.json`
- `research/results/dense_action_calibration_001/decision_records.csv`

The analysis script is [analyze_dense_action_calibration.py](/home/wj/VLN-CE/ETPNav/research/tools/analyze_dense_action_calibration.py).
