# Semantic Episodic Memory Feasibility Report

`HEAD: 8a11e50eac5370ec285b2132c6ac92d8041b5d18` was audited without changing any existing dense-action, A2, or single-override implementation.

## Current ETP node stores

Real nodes store position, panorama visual embedding, visit step, graph edges, weighted shortest paths, and a policy-generated stop score. Ghosts store proposal/mean positions, running visual embedding sum and count, front-node identities, and optional real positions. Runtime graph inputs add relative bearing/elevation/distance, graph distance, graph step, visited mask, and native provenance. Instruction context is recomputed by the graph/language transformer; it is not persisted as a structured node memory.

## Missing explicit information

There is no object or landmark identity, place category, instruction progress, completed sub-instruction, or transition relation. These may be implicit in latent visual/contextual embeddings, but the implementation has no field that can be inspected or audited as semantic evidence. Node orientation is also not persisted as a node field.

## Offline extraction

The frozen native A0 captures were processed by `research/tools/extract_semantic_memory.py`.

| Cohort | Decision states | Records | Instruction phrase hypotheses | Valid visual semantic records |
|---|---:|---:|---:|---:|
| train A0 | 380 | 4,099 | 3,979 | 0 |
| val-unseen A0 | 373 | 4,251 | 3,649 | 0 |

The output is under `research/semantic_memory/`. Every record retains raw node/action metadata, instruction text, candidate hypotheses, native logit/position where available, and an explicit `unavailable_no_panorama_join` status. The CLIP view feature HDF5 exists in the repository, but the frozen traces expose observation hashes and runtime numeric graph ids rather than a stable simulator viewpoint-to-HDF5 key. Joining it would require a new validated panorama capture/replay step.

## STOP

Semantic STOP feasibility is not estimable from this frozen extraction. There are 0 valid instruction-grounded visual evidence records, so node ranking, top-1 goal-node recall, MRR, and scene bootstrap CIs are undefined. The existing `node_stop_score` is a policy output and is not silently reused as semantic evidence. No representative semantic STOP failure can be claimed.

## Instruction progress

The phrase list is a lexical hypothesis from the instruction only; it contains no observation-grounded match or ordering evidence. Therefore strongest-evidence node, phrase-order recovery, progress confusion, and “passed landmark then failed” analyses all have zero valid samples. Treating instruction text alone as progress would be leakage and was rejected.

## Native ghost selection

The native action set and logits are available, but semantic evidence for any native ghost is absent. The number of actions with a semantic-vs-logit disagreement is zero by construction, and no claim about a better native action is valid. This also means the experiment cannot distinguish semantic usefulness from existing instruction-conditioned graph embeddings.

## Negative evidence and failure analysis

The main negative result is infrastructure-level: object/place/relation recognition cannot be recovered from these artifacts. A generic caption would be unsupported. The current cohort cannot test hallucination, relation accuracy, complementarity, or whether latent ETP features already encode the same signal. The CLIP feature store is not enough without a deterministic viewpoint join. This is a data/interface failure, not evidence that semantic memory has no value.

## Decision

Semantic memory is **PARTIAL / UNTESTED**, not demonstrated complementary. Current status is **NO-GO for model changes**. Do not add semantic nodes/tokens to the Graph Transformer, and do not start PPO/GRPO or dense-action experiments.

The single highest-value next experiment is a small, frozen, scene-balanced panorama replay capture that records `(episode, high_level_step, real node/ghost id, RGB panorama pixels or CLIP key, instruction, graph state)` with deterministic observation hashes. Once this alignment gate passes, run the same three offline tests before considering a read-only late/residual fusion branch. The preferred future architecture remains a shadow semantic side memory with confidence-gated late fusion, because adding graph nodes risks the already demonstrated representation contamination.
