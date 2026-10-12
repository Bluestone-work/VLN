# Semantic Episodic Memory Feasibility Report

## Scope

This experiment audited the released ETPNav checkpoint without changing policy,
GraphMap, waypoint prediction, controller, STOP mechanism, evaluator, or action
set. The alignment cohort is the frozen 44-episode val-unseen native-success
control set (`seed=100`, `ckpt.iter12000.pth`). It produced 373 decision states
across 11 scenes. RGB-D observations were captured before each high-level action
and were written independently from the existing post-action trace.

## Current ETP node memory

Visited and ghost nodes retain position/geometry, visual feature tensors, graph
connectivity and graph distance, step/visit state, stop-related scores, and the
front-node/ghost aggregation relationships used by GraphMap. Language-conditioned
information exists inside the policy's contextual latent representations and
logits, but the persistent node schema has no explicit object or landmark
identity, place category, instruction-progress index, or transition relation such
as “pass the sofa” or “turn at the painting”. These omissions are documented in
`SEMANTIC_MEMORY_AUDIT.md`.

## Capture fidelity

All 373 graph rows match a unique trace key. All 373 RGB and 373 depth files are
present and their 746 stored SHA-256 values match. No row advertises privileged
labels in features. The post-action trace hashes intentionally do not match the
pre-action semantic hashes (0/746 matches), which confirms the two timestamps are
distinct rather than indicating file corruption. The machine-readable audit is
`research/results/semantic_memory_alignment_val_unseen_001/alignment_audit.json`.

The frozen policy reproduction remained 44/44 successful, with SPL 0.818203 and
nDTW 0.745077. This is an analysis-only capture, not a new policy result.

## Offline semantic probe

The local `openai_clip` ViT-B/32 checkpoint was used only as an image/text probe
over the saved pre-action RGB frames. It did not generate labels, alter policy,
or consume outcome/reference-path information. Every decision node received a
probe record, but the probe is not a validated object detector or relation model.

STOP feasibility is currently weak. The Spearman correlation between native STOP
logit and the instruction-phrase CLIP score is 0.0126 over 314 valid node pairs;
the 10,000 scene bootstrap CI is [-0.0869, 0.1216]. This does not show an
independent goal-node signal. A privileged goal-node semantic label was not
invented for this audit.

Instruction-progress feasibility is also weak. The strongest prompted phrase
sequence was monotone in 21/44 episodes (47.7%), with 164 lexical-order
inversions overall. This is a descriptive consistency check, not a claim that
the phrases were actually visible or completed.

Native ghost selection was not validly testable from this capture. One current
panorama is shared by all native ghosts; there is no action-conditioned corridor,
endpoint view, or per-ghost semantic observation. Reusing the current image score
for every ghost would be leakage of an unobserved relation and would not answer
the selection question.

The broad CLIP label agreed with one instruction landmark phrase in only 26.1%
of 314 pairs. This low agreement, together with the near-zero STOP correlation,
is negative evidence against treating generic image/text similarity as semantic
memory.

## Failure analysis

The smoke and full cohort show three limitations. First, generic CLIP prompts can
rank scene categories while missing the instruction's relation and transition.
Second, a landmark-like score does not establish that the landmark was reached,
passed, or should trigger STOP. Third, current capture has no per-action future
view, so it cannot distinguish a useful native ghost from a visually plausible
but wrong branch. The MP3D scenes also have no semantic annotations, preventing a
direct object-level ground-truth audit.

## Decision

**PARTIAL / NO-GO for changing the model.** The capture infrastructure is valid
and reproducible, and explicit memory remains structurally absent from GraphMap.
However, this first offline probe does not demonstrate complementary semantic
signal for STOP, instruction progress, or native ghost selection. Do not add
semantic nodes to the Graph Transformer, train a fusion head, or start RL.

The single highest-value next experiment is a small, preregistered
action-conditioned semantic capture: render or replay each native ghost's
controller corridor/endpoint observation from the same state, then test whether
instruction-grounded evidence separates the native action selected by ETPNav from
another admissible native ghost. It must preserve the current graph and remain a
read-only side-memory analysis. If that experiment still fails to add signal,
stop this semantic-memory direction.
