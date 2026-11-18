# TETP multi-query stability and native-drift audit

Date: 2026-11-19
Checkpoint: `data/logs/checkpoints/release_r2r/ckpt.iter12000.pth`
Split: `val_unseen`, R2R, seed 100

## Implementation

The TETP query now runs three real navigation encoder passes. The first pass is
the deterministic action pass. Two additional passes temporarily enable only
Dropout modules under `vln_bert.global_encoder` and `global_sap_head`; language
and panorama features, GraphMap inputs, and candidate identities remain fixed.
Each candidate record stores all score passes, mean, standard deviation, and
independent pass/fail flags for native memory, local support, query stability,
and score envelope.

`TETP_NATIVE_DRIFT_MODE=hard` remains the default and preserves fail-closed
execution. `audit` is an explicit counterfactual arm: native drift is recorded
as advisory so the other conditions can be tested in an actual rollout. It is
not the production safety setting.

## Results

| arm | episodes | SR | SPL | nDTW | candidate decisions | candidate admissions |
|---|---:|---:|---:|---:|---:|---:|
| hard, 3 query passes | 44 | 68.18% | 63.33% | 78.47% | 334 | 0 |
| audit, 3 query passes | 4 | 25.00% | 25.00% | 64.24% | 37 | 4 |

For the 44-route hard arm, every decision used exactly three passes. There were
1,695 candidate records; score standard deviation was nonzero for all records
(median 0.3024, 95th percentile 0.7386, maximum 2.1678). Native drift had
median 1.2038 and never passed the fixed `1e-4` certificate. The independent
conditions passed for local support on 1,687/1,695 candidates, stability on
675/1,695, and score envelope on 1,634/1,695. The 30/30 native-success routes
were retained, matching the previous dual-GraphMap control exactly on all 44
episode metrics.

The 4-route audit arm admitted four candidates after ignoring native drift;
its SR fell to 25%. This demonstrates that the other conditions are now
experimentally exercised, but also confirms that native drift cannot be
silently relaxed. The audit arm is evidence for keeping the default hard gate,
not a performance claim.

Raw traces:

- `data/logs/tetp_mc_hard44.jsonl`
- `data/logs/tetp_mc_audit4.jsonl`
- `data/logs/tetp_mc_smoke4.jsonl`
- `data/logs/tetp_mc_a2_smoke4.jsonl` (negative isolation ablation)

Episode metrics for the hard arm are in
`data/logs/eval_results/tetp_mc_hard44/`.
