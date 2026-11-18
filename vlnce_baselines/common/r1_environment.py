"""ETP-R1 environment registration layered on the existing VLN-CE worker.

The simulator/controller implementation remains the repository's tested
``VLNCEDaggerEnv``.  R1 gets a separate registry name and only the small
worker APIs required by the official GRPO sampler.
"""
from habitat_baselines.common.baseline_registry import baseline_registry

from vlnce_baselines.common.environments import VLNCEDaggerEnv


@baseline_registry.register_env(name="R1Env")
class R1Env(VLNCEDaggerEnv):
    def current_dist_to_goal(self, is_train=False):
        return super().current_dist_to_goal()

    def point_dist_to_goal(self, pos, is_train=False):
        return super().point_dist_to_goal(pos)

    def reset_current_episode(self):
        """Reset the worker simulator to the same episode start state."""
        self._env._reset_stats()
        self.prev_episode_id = "something different"
        self._env.reconfigure(self._env._config)
        observations = self._env.task.reset(episode=self._env.current_episode)
        self._env.task.measurements.reset_measures(
            episode=self._env.current_episode, task=self._env.task
        )
        return observations

    def capture_agent_state(self):
        state = self._env.sim.get_agent_state()
        return {
            "position": state.position.copy(),
            "rotation": state.rotation,
            "episode_id": str(self._env.current_episode.episode_id),
        }

    def restore_agent_state(self, state):
        if str(self._env.current_episode.episode_id) != str(state["episode_id"]):
            raise ValueError("cannot restore a state from another episode")
        self._env.sim.set_agent_state(state["position"], state["rotation"])
