# Long-horizon Graph Value Feasibility

Date: 2026-10-09

No new B0/B1/B2 fit was run after the pilot. The pilot source is a historical
training capture and cannot satisfy the frozen scene-disjoint validation gate.
This report records the current evidence and the resulting stop decision.

The strongest existing offline result is the corrected fresh critical census:
5,668 strict full-return dominance pairs, with a frozen linear scorer at 0.9386
pair accuracy versus 0.8520 for native graph logit in scene-grouped in-cohort
folds. The independent 16-state/8-scene confirmation reported 0.8947 versus
0.8421 on 19 strict pairs. These are feasibility signals only: strict dominance
discards mixed outcomes, the cohorts were failure-critical, and the corrected
fresh-to-confirmation transfer was 0.8421 for both learned and native ranking.

The later route-disjoint execution study changed 27/96 routes: SR 92.7083 ->
95.8333, SPL 86.9455 -> 89.2480, nDTW 83.7453 -> 84.7011 and primitives
58.125 -> 56.448. It rescued three and lost zero native successes, but scene
bootstrap intervals were [−0.603,+5.621] SPL, [−1.198,+3.368] nDTW and
[−6.365,+2.146] primitives. The prospective replication later failed the frozen
cost gate and showed scene-level intervals crossing zero. No online policy claim
is made from either cohort.

Decision: **NO-GO for scaling the current graph-value model and for g3D-LF/RL.**
The required next experiment is a newly frozen, outcome-blind, scene-disjoint
full-return dataset with the protocol in `LONG_HORIZON_GRAPH_VALUE_PROTOCOL.md`,
followed by B0/B1/B2 evaluation from raw outcomes. Do not tune on the historical
validation or intervention results.
