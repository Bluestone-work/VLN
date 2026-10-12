# Offline Semantic Episodic Memory Protocol

The first phase is analysis-only. It reads frozen native A0 traces and graph-option records, creates one record per visited decision node and admissible native ghost, and never feeds the record into ETPNav. Raw instruction, position, action identity, logits, existing embedding provenance, and all missing-data reasons are retained.

The semantic adapter may use a validated frozen vision-language encoder only when a rendered RGB panorama and a stable viewpoint key are available. The current trace cohort does not satisfy that join: it has observation hashes but no pixels/viewpoint mapping. The adapter therefore emits `visual_feature_status=unavailable` and does not invent captions, object labels, place categories, relations, or confidence values. Instruction phrases are retained as hypotheses only, never as observed visual evidence.

Dead-end rules are frozen: no outcome, reference-path geometry, future goal distance, final success, or GT shortest path may be an input feature. They may only be used as analysis labels after memory extraction. STOP feasibility uses actual final outcome only for retrospective ranking evaluation. Progress feasibility checks phrase evidence order only when evidence exists. Native ghost feasibility compares semantic evidence with native logits only on the same admissible native action set.

The three experiments report valid sample count, missing-alignment count, per-scene coverage, and negative evidence. A semantic result is not considered positive unless the visual/instruction score exists independently of native logits and has a pre-registered scene-level confidence interval. No policy training, graph mutation, dense waypoint, PPO, GRPO, or threshold search is allowed.
