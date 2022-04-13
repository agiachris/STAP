from typing import Optional

from temporal_policies.agents import base, random, wrapper
from temporal_policies import envs, networks


class OracleAgent(wrapper.WrapperAgent):
    """Dummy agent that features an oracle critic."""

    def __init__(
        self,
        env: envs.Env,
        policy: Optional[base.Agent] = None,
        # action: Optional[Union[torch.Tensor, np.ndarray, Sequence[float]]] = None,
        device: str = "auto",
    ):
        """Constructs the oracle agent.

        If action is specified, the agent's actor will always output it.
        Otherwise, the actor will output random actions.

        Args:
            env: Policy env.
            action: Optional constant action.
            device: Torch device.
        """
        if policy is None:
            policy = random.RandomAgent(env)

        super().__init__(
            state_space=env.state_space,
            action_space=env.action_space,
            observation_space=env.observation_space,
            actor=networks.actors.OracleActor(env, policy),
            critic=networks.critics.OracleCritic(env),
            encoder=networks.encoders.OracleEncoder(env),
            device=device,
        )

        self._env = env

    @property
    def env(self) -> envs.Env:
        """Last generated env."""
        return self._env

    @env.setter
    def env(self, env: envs.Env) -> None:
        """Sets the last generated env."""
        self._env = env
        self.actor.env = env
        self.critic.env = env
        self.encoder.env = env
