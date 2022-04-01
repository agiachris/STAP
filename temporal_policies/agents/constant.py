from typing import Optional

import gym  # type: ignore
import torch  # type: ignore

from temporal_policies.agents import base as agents
from temporal_policies import networks


class ConstantAgent(agents.Agent):
    """An agent that outputs a single action."""

    def __init__(
        self,
        action: torch.Tensor,
        state_space: gym.spaces.Space,
        action_space: Optional[gym.spaces.Space] = None,
        observation_space: Optional[gym.spaces.Space] = None,
    ):
        """Constructs the constant agent.

        Args:
            action: Constant action.
            state_space: State space.
            action_space: Optional action space. Default inferred from action.
            observation_space: Optional observation space. Default equal to state space.
        """
        if action_space is None:
            action_space = gym.spaces.Box(
                low=-float("inf"), high=float("inf"), shape=action.shape, dtype=action.dtype
            )
        if observation_space is None:
            observation_space = state_space

        dim_states = len(state_space.shape)

        super().__init__(
            state_space=state_space,
            action_space=action_space,
            observation_space=observation_space,
            actor=networks.Constant(action, dim=dim_states),
            critic=networks.Constant(torch.tensor(0.0), dim=dim_states),
            encoder=torch.nn.Identity,
        )
