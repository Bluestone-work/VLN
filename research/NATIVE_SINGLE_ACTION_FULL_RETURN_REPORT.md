# Native single action full return

HEAD: `32deb0b4772edf7f3f992f55a624ee700714d2a5` (frozen native source; the
analysis-only runner is in the working tree and is committed with this report)

The exact pair artifact is `research/results/native_single_action_full_return_003/causal_pairs/pairs.jsonl`.
It contains one native/alternative row for every enumerated non-native action.

## Frozen population

- source capture: `semantic_memory_alignment_smoke_001`
- split: val-unseen, seed 100, release `ckpt.iter12000.pth`
- episodes: 2 (`1624`, `1778`)
- scenes: 1 (`pLe4wQe7qrG`)
- decision states: 10
- admissible branches: 80 (10 native sentinels + 70 alternatives)
- schedules: 37, each with one event per episode

## Exactness audit

All 37 disabled/native runs reproduced the 34-row source trajectory exactly,
including action, pose and evaluator metrics. Every alternative row passed the
exact event key, graph/action identity, pre-pose and pre-RNG checks. The audit is
`research/results/native_single_action_full_return_003/causal_pairs/exact_audit.json`.

The native selected branch is therefore a valid sentinel. Enabled runs execute
the selected alternative exactly once and then use the unchanged native policy
and controller until termination. Alternative endpoint traces are retained with
their immediate pose, distance and primitive actions; final episode metrics are
joined from the same evaluator protocol.

## Fixed outcome labels

The preregistered lexicographic rule is success, then nDTW, then SDTW, then
shorter path, fewer primitives and fewer collision events. Deadbands are fixed
at `1e-6` for nDTW/SDTW and `1e-5` for path cost. No threshold was tuned from
the outcomes.

```text
states: 10
actions: 70 alternatives (+ 10 native sentinels)
routes/episodes: 2
scenes: 1
BENEFIT: 2
HARM: 68
UNCERTAIN: 0
```

This is a causal full-return dataset, but it is a smoke cohort from one scene.
It is sufficient to unblock exact outcome joining and to expose the next
analysis gate; it is not sufficient for a scene-disjoint claim.

## Safety and limitations

No semantic feature uses an alternative endpoint observation, goal distance,
reference route or final outcome. Those values occur only in the outcome table.
No policy, PPO/GRPO objective, waypoint predictor or Graph Transformer was
changed. The 44-episode multi-view capture remains the larger validated source,
but exact full-return pairs have only been collected for the two-episode endpoint
enumeration so far.

Conclusion: **exact causal pair construction PASS for this frozen smoke cohort;
scene-disjoint semantic efficacy remains untested.**

Next gate: collect the same exact pairs over the frozen 44-episode/11-scene
cohort before fitting or interpreting a semantic probe.
