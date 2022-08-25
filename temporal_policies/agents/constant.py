from typing import Optional, Sequence, Union

import gym
import numpy as np
import torch

from temporal_policies.agents import base as agents
from temporal_policies import encoders, envs, networks
from temporal_policies.utils import spaces, tensors


class ConstantAgent(agents.Agent):
    """An agent that outputs a single action."""

    def __init__(
        self,
        env: Optional[envs.Env] = None,
        action: Optional[Union[torch.Tensor, np.ndarray, Sequence[float]]] = None,
        policy: Optional[agents.Agent] = None,
        action_space: Optional[gym.spaces.Box] = None,
        observation_space: Optional[gym.spaces.Box] = None,
        device: str = "auto",
    ):
        """Constructs the constant agent.

        Args:
            env: Optional policy env. If env is not available, agent or
                action_space/observation_space must be provided.
            action: Constant action.
            policy: Optional policy. If policy is not available, env or
                action_space/observation_space must be provided.
            action_space: Action space if env and policy are not given.
            observaton_space: Observation space if env and policy are not given.
            device: Torch device.
        """
        if policy is not None:
            if action_space is None:
                action_space = policy.action_space
            if observation_space is None:
                observation_space = policy.state_space
        if env is not None:
            if action_space is None:
                action_space = env.action_space
            if observation_space is None:
                observation_space = env.observation_space

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
            actor=networks.actors.ConstantActor(action, dim_states, dim_batch),
            critic=networks.critics.ConstantCritic(0.0, dim_states),
            encoder=encoders.IdentityEncoder(env, action_space, observation_space),
            device=device,
        )
