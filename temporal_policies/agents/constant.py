from typing import Optional, Sequence, Union

import gym  # type: ignore
import numpy as np  # type: ignore
import torch  # type: ignore

from temporal_policies.agents import base as agents
from temporal_policies import envs, networks
from temporal_policies.utils import spaces, tensors


class ConstantAgent(agents.Agent):
    """An agent that outputs a single action."""

    def __init__(
        self,
        env: Optional[envs.Env] = None,
        action: Optional[Union[torch.Tensor, np.ndarray, Sequence[float]]] = None,
        action_space: Optional[gym.spaces.Space] = None,
        observation_space: Optional[gym.spaces.Space] = None,
        device: str = "auto",
    ):
        """Constructs the constant agent.

        Args:
            env: Optional policy env. If env is not available, action_space, and
                observation_space must be provided.
            action: Constant action.
            action_space: Action space if env is not given.
            observaton_space: Observation space if env is not given.
            device: Torch device.
        """
        if env is not None:
            action_space = env.action_space if action_space is None else action_space
            observation_space = (
                env.observation_space
                if observation_space is None
                else observation_space
            )

        assert observation_space is not None
        assert action_space is not None

        if action is None:
            action = spaces.null_tensor(action_space)

        dim_states = len(observation_space.shape)
        dim_batch = tensors.dim(action) - len(action_space.shape)

        super().__init__(
            state_space=observation_space,
            action_space=action_space,
            observation_space=observation_space,
            actor=networks.Constant(action, input_dim=dim_states + dim_batch),
            critic=networks.Constant(0.0, input_dim=dim_states),
            encoder=torch.nn.Identity(),
            device=device,
        )
