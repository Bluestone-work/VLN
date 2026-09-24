# ETPNav Baseline Report

Compatibility reproduction. No algorithmic modifications were made. Compatibility changes: official Habitat-Sim 0.1.7 headless EGL build, `EGL_PLATFORM=surfaceless`, legacy dependency/API versions, MP3D symlink reuse, and explicit `MODEL.pretrained_path` overrides because the official eval branches omitted them.

## R2R-CE

```text
split: val_unseen
episodes: 1839
checkpoint: data/logs/checkpoints/release_r2r/ckpt.iter12000.pth
pretrained: data/pretrained/ETP/mlm.sap_r2r/ckpts/model_step_82500.pt
SR: 0.570962 | SPL: 0.491014 | OSR: 0.643284 | NE: 4.734436
path_length: 11.569253 | steps_taken: 80.274063 | collisions: 0.127652
nDTW: 0.623051 | sDTW: 0.467859 | ghost_cnt: 23.849920
```

Log: `logs/r2r_full_baseline.log`  Result: `data/logs/eval_results/release_r2r/stats_ckpt_59_val_unseen.json`

## RxR-CE

```text
split: val_unseen
episodes: 11006
checkpoint: data/logs/checkpoints/release_rxr/ckpt.iter19600.pth
pretrained: data/pretrained/ETP/mlm.sap_rxr/ckpts/model_step_90000.pt
SR: 0.554334 | SPL: 0.451361 | OSR: 0.642195 | NE: 5.728692
path_length: 18.065586 | steps_taken: 166.566330 | collisions: 0.363841
nDTW: 0.623533 | sDTW: 0.456984 | ghost_cnt: 33.551247
```

Log: `logs/rxr_full_baseline.log`  Result: `data/logs/eval_results/release_rxr/stats_ckpt_97_val_unseen.json`

Per-rank results: `data/logs/eval_results/release_rxr/stats_ep_ckpt_97_val_unseen_r0_w2.json` and `stats_ep_ckpt_97_val_unseen_r1_w2.json`.

## Environment

```text
environment: etpnav_legacy
python: 3.6.13
torch: 1.9.1+cu111
torchvision: 0.10.1+cu111
habitat-lab: 0.1.7
habitat-sim: 0.1.7 headless EGL
GPU: 2 x NVIDIA GeForce RTX 4090
git commit: 160e65fe5d23ad83d024d57d710e47be6892fc68
```

Smoke tests are runtime validation only and excluded from benchmark metrics. Hashes are in `results/r2r_baseline_sha256.txt` and `results/rxr_baseline_sha256.txt`; structured results are in `results/r2r_baseline_result.json` and `results/rxr_baseline_result.json`.
