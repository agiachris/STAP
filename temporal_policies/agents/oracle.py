from typing import Optional

from temporal_policies.agents import base, random, wrapper
from temporal_policies.encoders import base as encoders
from temporal_policies import envs, networks


class OracleAgent(wrapper.WrapperAgent):
    """Dummy agent that features an oracle critic."""

    def __init__(
        self,
        env: envs.Env,
        policy: Optional[base.Agent] = None,
        device: str = "auto",
    ):
        """Constructs the oracle agent.

        The oracle agent's actor will use the given policy's actor.

        Args:
            env: Policy env.
            policy: Policy whose actor will be used.
            device: Torch device.
        """
        if policy is None:
            policy = random.RandomAgent(env)

        super().__init__(
            state_space=policy.state_space,
            action_space=policy.action_space,
            observation_space=policy.observation_space,
            actor=networks.actors.OracleActor(env, policy),
            critic=networks.critics.OracleCritic(env),
            encoder=encoders.Encoder(env, networks.encoders.OracleEncoder),
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
        self.actor.env = env  # type: ignore
        self.critic.env = env  # type: ignore
        self.encoder.network.env = env  # type: ignore

    def reset_cache(self) -> None:
        assert isinstance(self.actor, networks.actors.OracleActor)
        self.actor.reset_cache()
        assert isinstance(self.critic, networks.critics.OracleCritic)
        self.critic.reset_cache()
