oringin:
orm.LayerNorm.bias', 'bert.global_encoder.encoder.x_layers.1.lang_self_att.self.query.weight', 'bert.global_encoder.encoder.x_layers.0.lang_self_att.self.query.bias', 'bert.global_encoder.encoder.x_layers.0.lang_output.dense04380373107888, 'success': 0.5897714907508161, 'oracle_success': 0.6583242655059848, 'path_length': 11.64230551579572, 'collisions': 0.10146474407042089, 'spl': 0.504799631522164, 'ndtw': 0.6341327001813917, 'sdtw': 0.4808760215520864, 'ghost_cnt': 24.1240478781284}
2026-09-23 22:59:43,232 rank 0's 920-episode results: {'steps_taken': 79.83260869565217, 'distance_to_goal': 4.964242558039563, 'success': 0.5521739130434783, 'oracle_success': 0.6282608695652174, 'path_length': 11.496279129321161, 'collisions': 0.15381000809056514, 'spl': 0.47724394672176323, 'ndtw': 0.6119808226033393, 'sdtw': 0.45485683622074446, 'ghost_cnt': 23.57608695652174}
2026-09-23 22:59:43,244 Episodes evaluated: 1839
2026-09-23 22:59:43,244 Average episode steps_taken: 80.274063
2026-09-23 22:59:43,244 Average episode distance_to_goal: 4.734436
2026-09-23 22:59:43,244 Average episode success: 0.570962
2026-09-23 22:59:43,245 Average episode oracle_success: 0.643284
2026-09-23 22:59:43,245 Average episode path_length: 11.569253
2026-09-23 22:59:43,245 Average episode collisions: 0.127652
2026-09-23 22:59:43,245 Average episode spl: 0.491014
2026-09-23 22:59:43,245 Average episode ndtw: 0.623051
2026-09-23 22:59:43,245 Average episode sdtw: 0.467859
2026-09-23 22:59:43,246 Average episode ghost_cnt: 23.849920

cd /home/wj/VLN-CE/ETPNav

CUDA_VISIBLE_DEVICES=0,1 \
/home/wj/miniconda3/envs/etpnav_legacy/bin/python \
-m torch.distributed.launch \
--nproc_per_node=2 \
--master_port 2345 \
run.py \
--exp_name stage3_long_r2r \
--run-type train \
--exp-config run_r2r/iter_train.yaml \
RL_TOPO.ENABLED True \
RL_TOPO.DEBUG True \
RL_TOPO.DEBUG_TRANSITIONS False \
IL.iters 200 \
IL.log_every 20 \
NUM_ENVIRONMENTS 8 \
GPU_NUMBERS 2 \
SIMULATOR_GPU_IDS '[0,1]' \
TORCH_GPU_IDS '[0,1]' \
IL.load_from_ckpt False \
IL.is_requeue False

data/logs/checkpoints/stage3_long_r2r/ckpt.iter200.pth

2026-09-24 11:15:49,407 Average episode steps_taken: 45.000000
2026-09-24 11:15:49,407 Average episode distance_to_goal: 8.147552
2026-09-24 11:15:49,407 Average episode success: 0.154432
2026-09-24 11:15:49,407 Average episode oracle_success: 0.178902
2026-09-24 11:15:49,407 Average episode path_length: 6.446091
2026-09-24 11:15:49,408 Average episode collisions: 0.119974
2026-09-24 11:15:49,408 Average episode spl: 0.145143
2026-09-24 11:15:49,408 Average episode ndtw: 0.460951
2026-09-24 11:15:49,408 Average episode sdtw: 0.130389
2026-09-24 11:15:49,409 Average episode ghost_cnt: 16.305057