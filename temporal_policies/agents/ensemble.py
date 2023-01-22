from typing import Optional

from temporal_policies.agents import base, wrapper
from temporal_policies import envs, networks


class EnsembleAgent(wrapper.WrapperAgent):
    """Agent wrapper that predicts a lower-confidence bound on the Q-value
    using the empirical variance of a bootstrap ensemble."""

    def __init__(
        self,
        policy: base.Agent,
        env: Optional[envs.Env] = None,
        scale: float = 1.0,
        clip: bool = True, 
        device: str = "auto",
    ):
        """Constructs the ensemble agent.

        Args:
            policy: Main agent with an ensemble of Q-functions.
            env: Policy env (unused, but included for API consistency).
            scale: Lower-confidence bound scale.
            device: Torch device.
        """
        super().__init__(
            state_space=policy.state_space,
            action_space=policy.action_space,
            observation_space=policy.observation_space,
            actor=policy.actor,
            critic=networks.critics.ContinuousEnsembleCritic(policy.critic, scale, clip),
            encoder=policy.encoder,
            device=device,
        )
