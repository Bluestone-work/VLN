"""CBR-GRPO entry point.

The official R1 trainer remains the baseline implementation.  This registry
entry adds a safe gate and reusable branch probe; the optimization hook stays
disabled until simulator restoration passes the probe on the target dataset.
"""

from habitat_baselines.common.baseline_registry import baseline_registry

from vlnce_baselines.GRPO_trainer_ETP_R1 import RLTrainer as GRPOR1Trainer
from vlnce_baselines.common.counterfactual_rollout import (
    run_counterfactual_branches,
)


@baseline_registry.register_trainer(name="CBR-GRPO-R1")
class CBRGRPOR1Trainer(GRPOR1Trainer):
    """R1 GRPO trainer with an explicit, fail-closed CBR switch.

    ``cf_enabled=False`` is the default and follows the exact official R1
    training path.  Enabling CBR currently requires callers to invoke
    ``run_cbr_probe`` first; this prevents claiming a counterfactual update
    when an environment cannot restore its state exactly.
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.cf_enabled = bool(getattr(config.GRPO, "cf_enabled", False))
        self.cf_probe_passed = False

    def run_cbr_probe(self, env, gmap, prev_vp, actions, rollout_fn):
        results = run_counterfactual_branches(
            env, gmap, prev_vp, actions,
            rollout_fn,
            max_branches=int(getattr(self.config.GRPO, "cf_topk", 3)),
        )
        self.cf_probe_passed = True
        return results

    def train(self):
        if self.cf_enabled and not self.cf_probe_passed:
            raise RuntimeError(
                "GRPO.cf_enabled=True but no verified CBR state-restore probe "
                "has passed; run run_cbr_probe() before enabling CBR training."
            )
        return super().train()

