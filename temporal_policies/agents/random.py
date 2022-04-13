from typing import Optional

import gym  # type: ignore
import torch  # type: ignore

from temporal_policies.agents import base as agents
from temporal_policies import envs, networks


class RandomAgent(agents.Agent):
    """Agent that outputs random actions."""

    def __init__(
        self,
        env: Optional[envs.Env] = None,
        action_space: Optional[gym.spaces.Space] = None,
        observation_space: Optional[gym.spaces.Space] = None,
        device: str = "auto",
    ):
        """Constructs the random agent.

        Args:
            env: Optional policy env. If env is not available, action_space, and
                observation_space must be provided.
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

        dim_states = len(observation_space.shape)

        super().__init__(
            state_space=observation_space,
            action_space=action_space,
            observation_space=observation_space,
            actor=networks.Random(
                action_space.low, action_space.high, input_dim=dim_states
            ),
            critic=networks.Constant(0.0, input_dim=dim_states),
            encoder=torch.nn.Identity(),
            device=device,
        )
