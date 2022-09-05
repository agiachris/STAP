from typing import Dict, Tuple

import numpy as np

from temporal_policies import envs
from temporal_policies.networks.critics.base import Critic
from temporal_policies.utils import tensors


class OracleCritic(Critic):
    """Dummy critic that returns ground truth rewards from simulation."""

    def __init__(self, env: envs.Env):
        """Constructs the oracle critic.

        Args:
            env: Env for simulation.
        """
        super().__init__()
        self.env = env

        if not hasattr(self.env, "_oracle_sim_result"):
            self.env._oracle_sim_result: Dict[Tuple[str, Tuple, Tuple], float] = {}  # type: ignore

    def reset_cache(self) -> None:
        self.env._oracle_sim_result.clear()  # type: ignore

    @tensors.torch_wrap
    @tensors.vmap(dims=1)
    def forward(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Outputs the reward from the given state and action.

        Args:
            state: Environment state.
            action: Action.
        """
        # primitive = self.env.get_primitive()
        # if not hasattr(self, "_action"):
        #     self._action = type(primitive).__name__
        # else:
        #     assert type(primitive).__name__ == self._action

        try:
            result = self.env._oracle_sim_result[  # type: ignore
                str(self.env.get_primitive()), tuple(state), tuple(action)
            ]
        except KeyError:
            raise ValueError
            prev_state = self.env.get_state()
            self.env.set_state(state)

            result = self.env.step(action)

            self.env.set_state(prev_state)

            # Cache results for oracle critic.
            self.env._oracle_sim_result[  # type: ignore
                str(self.env.get_primitive()), tuple(state), tuple(action)
            ] = result

        _, reward, _, _, _ = result

        return np.array(reward, dtype=np.float32)
