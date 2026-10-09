# Long-horizon Graph Value Data Pool Audit

Date: 2026-10-09

The only locally available alternative training pool is RxR VLN-CE. It does
not solve the independence problem. Against the frozen 59-scene exclusion set,
RxR `train_guide` contains 60,300 records across 59 scenes, with only one
scene outside the exclusion set: `mp3d/YmJkqBEsHnH/YmJkqBEsHnH.glb`.
`train_follower` contains 58,752 records across the same 59 scenes and the same
single remaining scene. The GT variants contain no records in this checkout.

The R2R pool has only two remaining scenes and four routes, while the RxR pool
adds no independent multi-scene coverage. Mixing R2R and RxR would also change
the instruction/task distribution, and would still not create six independent
scenes. No route sample or continuation outcome was generated from RxR.

Decision: **NO-GO for an expanded local data-pool confirmation.** The next
experiment requires an explicitly obtained external scene pool or a declared
non-independent cross-scene study with a different estimand. Until then,
model fitting and intervention remain paused.

Source hashes and counts are recorded in
`research/results/long_horizon_graph_value_data_pool_audit_003/summary.json`.
