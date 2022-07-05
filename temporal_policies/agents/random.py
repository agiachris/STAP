from typing import Optional

import gym
import torch

from temporal_policies.agents import base, wrapper
from temporal_policies import envs, networks


class RandomAgent(wrapper.WrapperAgent):
    """Agent that outputs random actions."""

    def __init__(
        self,
        env: Optional[envs.Env] = None,
        policy: Optional[base.Agent] = None,
        action_space: Optional[gym.spaces.Space] = None,
        observation_space: Optional[gym.spaces.Space] = None,
        device: str = "auto",
    ):
        """Constructs the random agent.

        The random agent's critic will use the given policy's critic, or output
        0 if a policy is not provided.

        Args:
            env: Optional policy env. If env is not available, action_space, and
                observation_space must be provided.
            policy: Policy whose critic will be used.
            action_space: Action space if env or policy is not given.
            observaton_space: Observation space if env or policy is not given.
            device: Torch device.
        """
        if policy is not None:
            if action_space is None:
                action_space = policy.action_space
            if observation_space is None:
                observation_space = policy.observation_space

            state_space = policy.state_space
            dim_states = len(state_space.shape)
            critic = policy.critic
            encoder = policy.encoder
        else:
            if env is not None:
                if action_space is None:
                    action_space = env.action_space
                if observation_space is None:
                    observation_space = env.observation_space

            assert action_space is not None
            assert observation_space is not None

            state_space = observation_space
            dim_states = len(state_space.shape)
            critic = networks.Constant(0.0, input_dim=dim_states)
            encoder = torch.nn.Identity()

        super().__init__(
            state_space=state_space,
            action_space=action_space,
            observation_space=observation_space,
            actor=networks.Random(
                action_space.low, action_space.high, input_dim=dim_states
            ),
            critic=critic,
            encoder=encoder,
            device=device,
        )
