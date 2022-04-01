from typing import Optional

import gym  # type: ignore
import torch  # type: ignore

from temporal_policies.agents import base as agents
from temporal_policies import networks


class RandomAgent(agents.Agent):
    """Agent that outputs random actions."""

    def __init__(
        self,
        state_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        observation_space=Optional[gym.spaces.Space],
    ):
        """Constructs the random agent.

        Args:
            state_space: State space.
            action_space: Action space.
            observation_space: Optional observation space. Default equal to state space.
        """
        if not isinstance(action_space, gym.spaces.Box):
            raise NotImplementedError

        if observation_space is None:
            observation_space = state_space

        dim_states = len(state_space.shape)

        super().__init__(
            state_space=state_space,
            action_space=action_space,
            observation_space=observation_space,
            actor=networks.Random(
                action_space.low, action_space.high, input_dim=dim_states
            ),
            critic=networks.Constant(torch.tensor(0.0), input_dim=dim_states),
            encoder=torch.nn.Identity,
        )
