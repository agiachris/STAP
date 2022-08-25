from typing import Optional

from temporal_policies.agents import base, wrapper
from temporal_policies import envs, networks


class GaussianAgent(wrapper.WrapperAgent):
    """Agent wrapper that samples from a Gaussian distribution centered around
    another policy."""

    def __init__(
        self,
        policy: base.Agent,
        env: Optional[envs.Env] = None,
        std: float = 0.5,
        device: str = "auto",
    ):
        """Constructs the random agent.

        Args:
            policy: Main policy whose predictions are used as the mean.
            env: Policy env (unused, but included for API consistency).
            std: Standard deviation.
            device: Torch device.
        """
        super().__init__(
            state_space=policy.state_space,
            action_space=policy.action_space,
            observation_space=policy.observation_space,
            actor=networks.actors.GaussianActor(policy.actor, std, policy.action_space),
            critic=policy.critic,
            encoder=policy.encoder,
            device=device,
        )
