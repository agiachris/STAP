from typing import Optional, Sequence, Union

import numpy as np  # type: ignore
import torch  # type: ignore

from temporal_policies.agents import base as agents
from temporal_policies import envs, networks
from temporal_policies.utils import spaces


class ConstantAgent(agents.Agent):
    """An agent that outputs a single action."""

    def __init__(
        self,
        env: envs.Env,
        action: Optional[Union[torch.Tensor, np.ndarray, Sequence[float]]] = None,
        device: str = "auto",
    ):
        """Constructs the constant agent.

        Args:
            env: Policy env.
            action: Constant action.
            device: Torch device.
        """
        dim_states = len(env.state_space.shape)
        if action is None:
            action = spaces.null_tensor(env.action_space)

        super().__init__(
            state_space=env.observation_space,
            action_space=env.action_space,
            observation_space=env.observation_space,
            actor=networks.Constant(action, input_dim=dim_states),
            critic=networks.Constant(0.0, input_dim=dim_states),
            encoder=torch.nn.Identity(),
            device=device,
        )
