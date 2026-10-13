# Visited semantic node memory validation protocol

Source HEAD: `0abb78b4e368ad3807081d3f52bd8680ffb334f2`.
This protocol freezes extraction/scoring before evaluating the new semantic outputs.
Earlier CLIP results and cohort native-success selection are already known.

Only the frozen 44 native-success val-unseen episodes (11 scenes, 373 decisions)
are included. The old capture has one RGB-D view, despite its `panoramas` folder
name. Reuse `run_option_capture.py --trace --semantic-capture` with the existing
configuration and current multiview capture implementation to fill missing views.
Require original and new action dictionaries, primitive traces, graph IDs,
visited masks, logits, options/embeddings, pose, and base image hashes to agree.
Audit all 12 RGB + 12 depth files per decision; sensor suffixes denote local yaw,
positive left, per simulator configuration (base RGB is 0 degrees).

A frozen Qwen2-VL-2B-Instruct model, revision
`895c3a49bc3fa70a340399125c650a463535e71c`, extracts JSON from a labeled 12-view
contact sheet. The model is downloaded into a separate cache and dependencies
into a separate environment; the legacy ETPNav environment is unchanged.
Instruction phrases are generated in a separate text-only call, preserving
instruction order and a final goal phrase. No goal coordinates, reference paths,
distances, actions, logits, scene labels, or outcomes enter extraction.
Unknown/occluded objects, bearings, transitions and completion must abstain.
Model confidence is uncalibrated, never a calibrated probability. A single
panorama cannot establish `passed`, `entered`, or completed turns; temporal
completion remains unknown unless supported by past observed evidence.

Persist structured place, objects/bearings, spatial relations and per-phrase
visual evidence before a deterministic hashed structural embedding. Ghosts are
excluded. Keep raw model outputs, parse failures, source hashes and prompt.
Use final-phrase evidence as semantic STOP score. Native node STOP score is
softmax(logits)[0], with latest visit overwriting prior scores, matching GraphMap.
At every prefix compare only available visited nodes; never use future memory.
Fusion uses fixed 0.5 native/semantic percentile ranks, without tuning on labels.
Unknown semantic scores abstain. Uncalibrated confidence gates at fixed
0, .25, .5, .75, .9 are descriptive risk-coverage points, not selected policies.

Evaluation is a separate program reading frozen feature files plus privileged
trace/dataset labels. Goal-node recall uses minimum observed geodesic goal
distance (ties within 1e-5); also report success-radius (<=3 m) retrieval accuracy.
Repeated observations are collapsed by node with latest available record.
Report episode-final STOP and all-prefix history retrieval separately, with
prefixes >=2 nodes additionally to remove trivial recall. Scene-cluster bootstrap
resamples 11 scenes with replacement (2,000 draws, seed 100), preserving all
within-scene rows. Report Spearman/Kendall and paired improvement/harm intervals.

Progress measures are descriptive consistency/coverage, not ground-truth
completion accuracy. No phrase visibility/completion annotations exist. Report
that missing denominator, do not treat a forced monotonic accumulator as accuracy.
All 44 episodes succeed, so failed-episode rescue and failure-generalization are
unidentifiable. Historical node distance improvements are retrieval opportunities,
not executed return outcomes. nDTW of truncated observed paths is an explicitly
optimistic retrospective diagnostic, not the effect of an executable return.

Complementarity compares semantic score against STOP logit and native node STOP
probability. A scene-held-out ridge probe predicts semantic score from captured
native STOP contextual embeddings (not mislabeled as visited raw visual features).
Conditional tables use both exact nearest-node and success-radius labels.
No thresholds, weights or encoder parameters are tuned using val-unseen outcomes.

GO requires positive lower scene CI for net beneficial-minus-harmful retrieval
changes and retained native successes, plus valid directional/relational evidence.
Unvalidated evidence, missing completion labels or no stable conditional gain
cannot justify deployment. NO-GO is scoped to this extractor/experiment, not
proof that all possible semantic memory is useless. Only an offline GO permits
read-only shadow rollout. Only shadow trajectory identity permits residual fusion.
No dense/A2/RL_TOPO changes, ghost semantics, RL, or backbone training.
