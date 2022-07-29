from typing import Optional, Type

from temporal_policies.agents import base, wrapper
from temporal_policies import envs
from temporal_policies import scod


class SCODCriticAgent(wrapper.WrapperAgent):
    """Agent wrapper that returns an uncertainty-transformed metric over the critic."""

    def __init__(
        self,
        policy: base.Agent,
        scod_wrapper: Type[scod.WrapperSCOD],
        env: Optional[envs.Env] = None,
        device: str = "auto",
    ):
        """Constructs the random agent.

        Args:
            policy: Main policy whose predictions are used as the mean.
            scod_wrapper: SCOD wrapper around the critic.
            env: Policy env (unused, but included for API consistency).
            std: Standard deviation.
            device: Torch device.
        """
        super().__init__(
            state_space=policy.state_space,
            action_space=policy.action_space,
            observation_space=policy.observation_space,
            actor=policy.actor,
            critic=scod_wrapper,
            encoder=policy.encoder,
            device=device,
        )
