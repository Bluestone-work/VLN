# Visual waypoint value probe

HEAD: `fac2b20aaf4cb809121bdd89e08fb9396cf30a14`

This is an offline probe only. The released ETPNav checkpoint, Graph
Transformer, controller and STOP behavior were not changed.

## Dataset optimization

Source: `research/results/native_single_action_full_return_44_002/directional_pairs_002/pairs.jsonl`

- 3,877 labeled alternative actions;
- 369 independent decision states;
- 11 scenes;
- 430 BENEFIT and 3,447 HARM rows;
- 182 states contain both BENEFIT and HARM alternatives;
- one UNCERTAIN row is excluded;
- 329 STOP alternatives have no directional sector and use an explicit visual
  abstention;
- 3,548 non-STOP rows have an exact current-node RGB-D sector;
- no endpoint image, goal distance, reference route or final outcome is used as
  a feature.

The prepared dataset is
`research/results/visual_waypoint_value_dataset_002/`. Each training fold
weights candidates inversely by the number of candidates in their state, then
balances BENEFIT and HARM within the training fold. Test folds are held out by
scene, so rows from one state never cross the split.

The first visual representation is deterministic RGB spatial pooling,
color histograms and depth statistics. The second representation is a frozen
ImageNet ResNet18 sector embedding, with no fine-tuning.

## Scene-disjoint OOF results

| arm | AUROC | AUPRC | within-state pairwise | rescue | destroyed native successes |
|---|---:|---:|---:|---:|---:|
| F0 native | 0.722 | 0.213 | 0.779 | 71/182 | 124/187 |
| F0 + geometry | 0.727 | 0.215 | 0.783 | 71/182 | 127/187 |
| F0 + pooled RGB-D | 0.723 | 0.215 | 0.774 | 73/182 | 127/187 |
| F0 + geometry + pooled RGB-D | 0.728 | 0.217 | 0.777 | 71/182 | 126/187 |
| pooled RGB-D only | 0.608 | 0.148 | 0.558 | 56/182 | 132/187 |
| F0 + frozen ResNet18 sector | 0.700 | 0.209 | 0.764 | 67/182 | 127/187 |
| F0 + geometry + ResNet18 sector | 0.704 | 0.211 | 0.764 | 67/182 | 126/187 |
| frozen ResNet18 sector only | 0.569 | 0.134 | 0.545 | 42/182 | 96/187 |

The paired scene-bootstrap 95% interval for pooled RGB-D over F0 is
`[-0.0098, 0.0120]` AUROC and `[-0.0194, 0.0105]` pairwise accuracy. For
ResNet18 over F0 it is `[-0.0736, 0.0298]` AUROC and `[-0.0572, 0.0142]`
pairwise accuracy. Neither interval establishes a positive visual increment.

## Interpretation

The state-balanced dataset fixes the previous row-frequency problem. It does
not reveal a stable local visual value signal. Pooled RGB-D produces only two
additional rescue states while increasing destructive selections. A generic
frozen ImageNet feature is worse than the native branch in this probe, so the
result is not evidence that a stronger generic image encoder will solve the
problem.

This is still not a proof that all visual short-horizon prediction is
impossible. The current experiment uses one-step current-sector features and
does not model latent transitions, multi-step history, or instruction-grounded
future affordance. It establishes that the current local sector representation
does not provide a safe, scene-disjoint residual signal over native ETPNav.

## Decision

**NO-GO for immediate visual residual fusion.** Do not alter the native policy
based on these probes. The next defensible experiment is a small
action-conditioned latent transition/value model using one to three native
continuation steps, evaluated with the same exact full-return labels and
scene-held-out protocol. If that model also fails to add conditional signal in
native-wrong states, stop pursuing local visual waypoint value and focus on
history/transition modeling or a documented negative result.

Artifacts:

- `research/tools/prepare_visual_value_dataset.py`
- `research/tools/probe_visual_waypoint_value.py`
- `research/tools/extract_resnet_sector_features.py`
- `research/results/visual_waypoint_value_dataset_002/dataset_manifest.json`
- `research/results/visual_waypoint_value_probe_001/summary.json`
- `research/results/visual_waypoint_value_probe_003/summary.json`
