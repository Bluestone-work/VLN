# Gate C Instruction-Progress Representation Protocol

## Frozen question

Does adding deployable, frozen instruction-progress and candidate-semantic
compatibility information make native-relative long-horizon intervention labels
separable on held-out scenes?

## Frozen data and labels

Use `research/results/gate_b_native_relative_004/samples.csv` unchanged:
1,377 native-relative samples, 117 states, 31 routes, 18 scenes,
INTERVENE=76, KEEP=533, AMBIGUOUS=768. Use the existing train versus
val_unseen scene-disjoint split. Labels are strict full-return dominance;
H=1/H=2 progress is never used as supervision.

## Nested feature levels

- **F0:** the exact Gate B geometry/logit/policy feature schema, unchanged.
- **F1:** F0 plus statistics of existing frozen 768-D native and alternative
  `gmap_embeds` (norms, difference norm, cosine, coordinate-wise mean/std and
  absolute-difference summaries).
- **F2:** F1 plus frozen instruction-progress alignment from `txt_embeds` and
  the current graph state: expected/peak token index, entropy, prefix/suffix
  mass, and candidate/state-to-token alignment statistics.
- **F3:** F2 plus candidate compatibility with current and suffix token spans,
  including alternative-minus-native deltas.

No held-out result may alter this hierarchy. High-dimensional embeddings use
fixed summary statistics; no transformer, VLM, LLM, RL, PPO, waypoint
predictor, adaptive density, or collision trigger is introduced.

## Fitting and evaluation

Fit logistic regression, ridge linear classification, and a 16-unit MLP on
INTERVENE/KEEP rows from training scenes only. Standardization is fit on the
training cohort. Evaluate every row, including AMBIGUOUS, on held-out scenes.
For each model use only thresholds 0.50, 0.70, 0.80, 0.90, and 0.95.

Report intervention precision/recall/coverage, KEEP harm rate, AMBIGUOUS
intervention rate, risk-coverage and precision-coverage curves, plus route and
scene support. Low confidence means ABSTAIN/KEEP NATIVE.

## Pre-registered gate

Conditional GO requires an F2/F3 held-out precision improvement over F0, a
fixed high threshold with nonzero coverage and precision at least 0.70,
reduced AMBIGUOUS intervention rate, and positive support across multiple
routes and at least three held-out scenes. Precision at least 0.80 with
nonzero coverage is a strong signal. Otherwise the decision is NO-GO for
instruction-progress selective intervention and no online intervention or
larger model follows.

All files are written to a new immutable result directory and never overwrite
Gate B artifacts.
