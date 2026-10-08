# Gate B: Native-Relative Intervention and Abstention Feasibility

## Dataset

Samples compare one alternative native graph action with the native action at
the same captured state. Labels use full-return outcomes only after the
pre-action feature table was frozen:

- `INTERVENE`: alternative strictly dominates native on SR, SPL, nDTW, goal
  error, path length, and primitive count, with one strict improvement;
- `KEEP`: native strictly dominates the alternative;
- `AMBIGUOUS`: mixed or incomparable outcomes, retained without forcing a
  binary label.

The corrected dataset has 1,377 samples, 117 states, 31 routes, and 18 scenes:
`INTERVENE=76`, `KEEP=533`, `AMBIGUOUS=768`. INTERVENE occurs in 48 states,
18 routes, and 13 scenes, so it is not concentrated in one scene. The train
and val_unseen scene sets are disjoint (8 versus 10 scenes). All audited routes
are native failures; native-success harm therefore cannot be estimated from
this dataset and is never reported as zero.

Labels compare the alternative continuation directly with the native
continuation from the same captured state. The episode-level control is used
only for stratification. AMBIGUOUS rows are excluded from binary fitting and
remain in held-out evaluation, so their rate is visible rather than silently
treated as KEEP or INTERVENE.

Features are pre-action only: graph logits and differences, normalized ranks,
candidate geometry/path estimates, candidate count, current/history proposal
flag, policy entropy and top-1/top-2 margin, STOP flags, and high-level step.
Goal distance, reference route, future collision/success, SPL/nDTW, final path,
and primitive outcomes are excluded.

## Models and held-out result

Only logistic regression, ridge linear classifier, and a 16-unit one-hidden-layer
MLP were fitted on the train cohort. Held-out evaluation is by unseen scene
cohort, never random rows. At the predefined thresholds 0.50/0.70/0.80/0.90/0.95,
the conservative high-threshold region does not provide the required signal:

| Model | Threshold | Coverage | Intervention precision | Harmful KEEP rate | Ambiguous intervention rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| logistic | 0.95 | 0.625 | 0.350 | 0.000 | 0.650 |
| ridge | 0.70 | 0.750 | 0.396 | 0.000 | 0.604 |
| small MLP | 0.95 | 0.547 | 0.314 | 0.000 | 0.686 |

Ridge at thresholds 0.80/0.90/0.95 abstains completely (zero coverage). The
apparent zero KEEP harm is not a safety result: the held-out cohort has no
native-success routes, and most selected interventions are AMBIGUOUS rather
than beneficial.

## Decision

**NO-GO for graph-value selective intervention.** Frozen pre-action features
do not identify a high-precision, nonzero-coverage intervention region on
held-out scenes. Do not proceed to single-intervention online tests, PPO/RL,
VLM/LLM, or more feature/threshold search from this dataset. The research
branch should stop here unless a separately justified dataset includes native
success routes and a new pre-registered signal is proposed.

Artifacts (the `004` directory is the corrected, reproducible run):

- `research/results/gate_b_native_relative_004/summary.json`
- `research/results/gate_b_native_relative_004/samples.csv`
- `research/results/gate_b_native_relative_004/per_route.csv`
- `research/results/gate_b_native_relative_004/models_002/summary.json`
- `research/results/gate_b_native_relative_004/models_002/heldout_predictions.csv`
- `research/results/gate_b_native_relative_004/figures_002/gate_b_precision_coverage.png`

`gate_b_native_relative_001` is retained as the original run. Its counts and
metrics are identical, but the corrected builder is the authoritative source.
