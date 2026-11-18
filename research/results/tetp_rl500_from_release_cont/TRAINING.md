# TETP/RL training continuation

The valid continuation started from `data/logs/checkpoints/release_r2r/ckpt.iter12000.pth`
and ran 500 PPO steps to iteration 12500. The target `IL.iters` was therefore
12500; setting it to 500 would execute zero steps after loading iteration 12000.

The final checkpoint is intentionally not committed because it is about 950 MB.
The run log is retained at:

`data/logs/running_log/tetp_rl500_from_release_cont_train.log`

On the same 44-route control evaluation, the continuation kept the native
policy metrics unchanged (SR 68.18%, SPL 63.58%, nDTW 77.23%); PPO
`argmax_action_change_rate` remained zero.
