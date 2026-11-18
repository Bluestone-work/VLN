# TETP dual-GraphMap evaluation

Configuration: release R2R checkpoint `ckpt.iter12000.pth`, `val_unseen`, 44
episodes, seed 100, `ACTION_ABSTRACTION.LEVEL=dense_native_union`,
`ACTION_ABSTRACTION.TETP_ENABLED=true`, native drift threshold `1e-4`.

| arm | SR | SPL | nDTW | baseline-success routes retained |
|---|---:|---:|---:|---:|
| native baseline | 68.18% | 63.58% | 77.23% | 30/30 |
| TETP dual graph | 68.18% | 63.33% | 78.47% | 30/30 |

The TETP trace contains 334 decisions. All candidates abstained because the
native drift certificate failed; no outcome or goal-distance label was used.
The median native drift was 1.20. TETP produced zero rescue routes in this
control cohort and zero native-success regressions.

Raw artifacts:

- `data/logs/tetp_dual_eval44.jsonl`
- `data/logs/eval_results/tetp_dual_eval44/stats_ckpt_59_val_unseen.json`
- `data/logs/eval_results/tetp_dual_eval44/stats_ep_ckpt_59_val_unseen_r0_w1.json`
