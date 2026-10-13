# Human semantic oracle feasibility report

HEAD: `2498831861b782be482b6ffdd4590da4023f64b5`

This report is intentionally pending human annotation. The frozen, outcome
blind annotation artifact is
`research/annotations/directional_semantic_oracle_v1/`.

Scene split:

- development scene: `data/scene_datasets/mp3d/QUCTc6BB5sX/QUCTc6BB5sX.glb`
- test scenes: `X7HyMhZNoso`, `oLBMNvg9in8`, `EU6Fwq7SyZv`
- rule: SHA-256 lexicographic ordering of scene IDs, first four only; selected
  before annotation and without reading causal labels

Sampling:

- states: 48 (12 per selected scene)
- unique sectors: 241
- action-bearing sectors only; one nearest 30° sector per state
- development sectors: 48
- test sectors: 193
- source: 44-route / 11-scene pre-action multi-view capture

Annotation:

- status: **PENDING HUMAN ANNOTATION**
- preannotations: frozen Qwen hypotheses only; not ground truth
- final labels: `annotations.jsonl` is not present yet
- landmark distribution: pending
- instruction-support yes/no/uncertain: pending
- confidence distribution: pending
- missing / invalid: pending

Causal test:

F0: pending human labels

Human-only: pending human labels

F0 + Human: pending human labels

Complementarity: pending human labels

Rescue opportunity: pending human labels

Scene-level results: pending human labels

Consistency across scenes: pending human labels

Conclusion:

No oracle feasibility conclusion is made before the frozen test sectors receive
manual labels. The Qwen preannotation and current extractor result cannot be
used as a substitute for human ground truth.

GO / NO-GO: **PENDING HUMAN ANNOTATION**

Next single highest-value experiment: annotate the 193 test sectors using the
frozen protocol, validate `annotations.jsonl`, then run the scene-held-out
oracle causal evaluator. Do not expose outcomes to annotators.
