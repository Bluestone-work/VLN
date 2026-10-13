# Native Ghost Endpoint Semantic Report

This experiment enumerated all admissible native graph actions for two frozen
val-unseen episodes. It produced 80 isolated controller branches over 10 decision
states. The selected native action was replayed as a sentinel in every state.

```text
sentinel pass: 10 / 10
action branches: 80
states: 10
```

For each branch, post-controller endpoint RGB-D was saved. A local CLIP ViT-B/32
probe then ranked endpoint images by similarity to instruction landmark phrases.
The selected native action was the top endpoint-semantic action in only 1/10
states, with mean selected rank 6.2 among the admissible actions.

This is useful evidence that endpoint semantics are action-dependent and differ
from the current-panorama probe. It is not yet evidence that the CLIP ranking is
correct: the smoke has no counterfactual full-return labels in this report, only
controller replay fidelity. The result therefore supports continuing with a
causal BENEFIT/HARM join, while forbidding deployment or model training from this
probe alone.

The next experiment is to join these endpoint records with existing single
override/full-return outcomes for the same state and action identity, then test
whether endpoint semantic evidence separates beneficial and harmful native
alternatives. Any join lacking exact state/action identity must be discarded.
