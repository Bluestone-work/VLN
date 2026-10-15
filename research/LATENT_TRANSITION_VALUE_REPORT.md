# Action-conditioned latent transition/value probe

HEAD before this report: `9162520`

This experiment tests whether a predicted short-term future visual state adds
waypoint value beyond native ETPNav. It does not modify the policy, graph
transformer, controller or STOP behavior.

## Exact transition data

The full endpoint replay reused the frozen 44-route multi-view capture and the
exact native controller. It enumerated 4,251 admissible branches across 373
decision states, 44 episodes and 11 scenes. All 373 native selected-action
sentinels passed endpoint, primitive, collision, RNG and terminal checks.

The endpoint RGB-D is used only as a training target for a frozen ImageNet
ResNet18 latent dynamics model. It is never provided as an online feature to
the value head.

The causal join contains 3,877 alternatives:

- BENEFIT: 430;
- HARM: 3,447;
- UNCERTAIN: 1 excluded.

## Model

For each candidate action, the transition model receives only:

```text
current pre-action RGB sector ResNet feature
action geometry
native logit summary
```

It predicts a 16-dimensional PCA latent of the post-action endpoint RGB
observation. A separate value probe then receives the predicted latent and
native F0 features. Scene-held-out folds are used throughout.

`F0 + oracle endpoint` is an analysis-only upper-bound diagnostic. It uses the
actual endpoint visual latent and is not deployable.

## Full 44-route results

| arm | AUROC | AUPRC | within-state pairwise |
|---|---:|---:|---:|
| F0 native | 0.722 | 0.213 | 0.779 |
| F0 + geometry | 0.727 | 0.215 | 0.783 |
| F0 + predicted transition | 0.719 | 0.218 | 0.774 |
| F0 + oracle endpoint diagnostic | 0.723 | 0.218 | 0.781 |

Predicted transition minus F0 scene-bootstrap 95% intervals:

```text
AUROC:    [-0.0158,  0.0074]
AUPRC:    [-0.0040,  0.0117]
Pairwise: [-0.0199,  0.0065]
```

The oracle endpoint diagnostic also has intervals containing zero. Therefore
the result is not explained only by an imperfect transition predictor: actual
post-action endpoint appearance itself does not show a stable incremental
waypoint-value signal in this probe.

## Pilot scope check

An outcome-blind 11-episode pilot was run first:

- 958 branches;
- 84 states;
- 7 scenes;
- 32 BENEFIT and 842 HARM alternatives;
- all 84 native sentinels passed.

The pilot's predicted-transition AUROC was 0.604 versus F0 0.588, but its
scene-bootstrap AUROC difference was `[-0.0240, 0.0271]`. The full cohort does
not reproduce a positive effect.

## Decision

**NO-GO for this one-step visual latent transition/value method.** A predicted
short-term visual state does not add stable scene-disjoint signal beyond native
ETP features, and no intervention is permitted.

This does not prove that all world-model approaches are impossible. It rules out
the current one-step endpoint-appearance target and value head. The remaining
plausible direction would need to model instruction-conditioned traversability
or graph/history transition state, rather than endpoint RGB appearance. If that
direction is pursued, it should be a separately registered experiment with a
new causal target and no online endpoint input.

Artifacts:

- `research/tools/probe_latent_transition_value.py`
- `research/results/native_transition_endpoint_full_001/summary.json`
- `research/results/latent_transition_value_full_001/summary.json`
- `research/results/latent_transition_value_pilot_003/summary.json`
