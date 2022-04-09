import torch  # type: ignore

from temporal_policies.agents import base as agents
from temporal_policies import envs, networks


class RandomAgent(agents.Agent):
    """Agent that outputs random actions."""

    def __init__(
        self,
        env: envs.Env,
        device: str = "auto",
    ):
        """Constructs the random agent.

        Args:
            env: Policy env.
            device: Torch device.
        """
        dim_states = len(env.observation_space.shape)

        super().__init__(
            state_space=env.observation_space,
            action_space=env.action_space,
            observation_space=env.observation_space,
            actor=networks.Random(
                env.action_space.low, env.action_space.high, input_dim=dim_states
            ),
            critic=networks.Constant(0.0, input_dim=dim_states),
            encoder=torch.nn.Identity(),
            device=device,
        )
