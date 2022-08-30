from typing import Tuple, Union

import torch
import numpy as np

from temporal_policies.utils import utils
from .task_manager import TaskManager


def _has_required(req, **kwargs):
    assert all(x in kwargs for x in req)


def _strip_kwargs(req, opt, **kwargs):
    req_kwargs = {k: v for k, v in kwargs.items() if k in req}
    opt_kwargs = {k: v for k, v in kwargs.items() if k in opt}
    return req_kwargs, opt_kwargs


def _parse_value_kwargs(**kwargs):
    req = ["states"]
    opt = []
    _has_required(req, **kwargs)
    return _strip_kwargs(req, opt, **kwargs)


def _parse_q_value_kwargs(**kwargs):
    req = ["states", "actions"]
    opt = []
    _has_required(req, **kwargs)
    return _strip_kwargs(req, opt, **kwargs)


def _parse_policy_kwargs(**kwargs):
    req = ["states"]
    opt = ["variance", "samples", "bounds", "sample"]
    _has_required(req, **kwargs)
    return _strip_kwargs(req, opt, **kwargs)


def _parse_random_policy_kwargs(**kwargs):
    req = ["envs"]
    opt = ["samples"]
    _has_required(req, **kwargs)
    return _strip_kwargs(req, opt, **kwargs)


def _parse_simulate_kwargs(**kwargs):
    req = ["envs", "actions"]
    opt = []
    _has_required(req, **kwargs)
    return _strip_kwargs(req, opt, **kwargs)


def _parse_model_kwargs(**kwargs):
    req = ["states", "actions"]
    opt = []
    _has_required(req, **kwargs)
    return _strip_kwargs(req, opt, **kwargs)


class IOManager(TaskManager):
    def __init__(
        self,
        default_actor="random",
        default_critic="q_fn",
        default_dynamics="ground_truth",
        **kwargs
    ):
        """Interfaces input representations from the planner (e.g., states, environments, actions)
        and produces the respective outputs (e.g., actions, simulated environments).

        args:
            default_actor: default actor type to evaluate for valid actions
            default_critic: default critic type to evaluate for unspecified primitive returns
            default_dynamics: default model type to evaluate for unspecified primitive dynamics
        """
        super().__init__(**kwargs)
        assert default_actor in ["policy", "random"]
        assert default_critic in ["q_fn", "v_fn"]
        assert default_dynamics in ["ground_truth", "learned"]
        self._default_actor = default_actor
        self._default_critic = default_critic
        self._default_dynamics = default_dynamics

    def _use_q_function(self, idx):
        critic = self._task[idx].get("critic", None)
        use_q = critic == "q_fn" or (critic is None and self._default_critic == "q_fn")
        return use_q

    def _critic_interface(self, idx, **kwargs):
        """Generic critic interface: query V(s) or Q(s, a) for geometric feasibility score."""
        returns = None
        if self._use_q_function(idx):
            # Query Q(s, a)
            req_kwargs, opt_kwargs = _parse_q_value_kwargs(**kwargs)
            returns = self._q_value_fn(idx, **req_kwargs, **opt_kwargs)
        else:
            # Query V(s)
            req_kwargs, opt_kwargs = _parse_value_kwargs(**kwargs)
            returns = self._value_fn(idx, **req_kwargs, **opt_kwargs)
        return returns

    def _use_policy_actor(self, idx):
        actor = self._task[idx].get("actor", None)
        use_policy = actor == "policy" or (
            actor is None and self._default_actor == "policy"
        )
        return use_policy

    def _actor_interface(self, idx, **kwargs):
        """Generic actor interface: query policy or random actor for actions."""
        actions = None
        if self._use_policy_actor(idx):
            # Query policy
            req_kwargs, opt_kwargs = _parse_policy_kwargs(**kwargs)
            if "variance" in opt_kwargs:
                actions = self._sample_policy(idx, **req_kwargs, **opt_kwargs)
            else:
                actions = self._policy(idx, **req_kwargs, **opt_kwargs)
        else:
            # Query random action
            req_kwargs, opt_kwargs = _parse_random_policy_kwargs(**kwargs)
            actions = self._random_policy(**req_kwargs, **opt_kwargs)

        use_learned_dynamics = self._use_learned_dynamics(idx)
        if not use_learned_dynamics and utils.contains_tensors(actions):
            actions = utils.to_np(actions)
        elif use_learned_dynamics and isinstance(actions, np.ndarray):
            actions = utils.to_tensor(actions).to(self._device)
        return actions

    def _use_learned_dynamics(self, idx_action: int) -> bool:
        task_dynamics = self._task[idx_action].get("dynamics", self._default_dynamics)
        return task_dynamics == "learned"

    def _simulate_interface(
        self, idx_action: int, **simulation_kwargs
    ) -> Tuple[np.ndarray, Union[bool, np.ndarray]]:
        """Generic simulation interface: query learned or ground truth dynamics for next state."""
        output_states, output_success = None, None
        if self._use_learned_dynamics(idx_action):
            # Simulate learned dynamics model
            req_kwargs, opt_kwargs = _parse_model_kwargs(**simulation_kwargs)
            states, actions, is_np, is_batched = self._pre_process_states_actions(
                **req_kwargs, **opt_kwargs
            )
            output_success = np.ones(states.size(0), dtype=bool) if is_batched else True
            output_states = self._simulate_model(idx_action, states, actions)
            output_states = self._post_process_outputs(output_states, is_np, is_batched)
        else:
            # Simulated environment dynamics
            req_kwargs, opt_kwargs = _parse_simulate_kwargs(**simulation_kwargs)
            req_kwargs["actions"] = utils.to_np(req_kwargs["actions"])
            assert isinstance(req_kwargs["actions"], np.ndarray)
            output_states, output_success = self._simulate_env(
                idx_action, **req_kwargs, **opt_kwargs
            )
        return output_states, output_success

    def _pre_process_inputs(
        self, inputs: Union[torch.Tensor, np.ndarray]
    ) -> Tuple[torch.Tensor, bool, bool]:
        """Batch and tensorize inputs."""
        is_np = not utils.contains_tensors(inputs)
        is_batched = inputs.ndim > 1
        if is_np:
            inputs = utils.to_tensor(inputs).to(self._device)
        if not is_batched:
            inputs = utils.unsqueeze(inputs, 0)
        return inputs, is_np, is_batched

    @staticmethod
    def _post_process_outputs(
        outputs: torch.Tensor, is_np: bool, is_batched: bool
    ) -> Union[torch.Tensor, np.ndarray]:
        """Post-process outputs based on their input configuration."""
        if not is_batched:
            outputs = utils.squeeze(outputs, 0)
        if is_np:
            outputs = utils.to_np(outputs)
        return outputs

    def _pre_process_states_actions(
        self,
        states: Union[torch.Tensor, np.ndarray],
        actions: Union[torch.Tensor, np.ndarray],
    ) -> Tuple[torch.Tensor, torch.Tensor, bool, bool]:
        """Batch and tensorize states and actions."""
        states, is_np_s, is_batched_s = self._pre_process_inputs(states)
        actions, is_np_a, is_batched_a = self._pre_process_inputs(actions)
        is_np = is_np_s or is_np_a
        is_batched = is_batched_s or is_batched_a

        num_s, num_a = states.size(0), actions.size(0)
        if num_a == num_s:
            return states, actions, is_np, is_batched
        if num_s < num_a and num_s == 1:
            states = states.tile(num_a, 1)
        elif num_s > num_a and num_a == 1:
            actions = actions.tile(num_s, 1)
        else:
            raise ValueError("Cannot match uneven number of states to actions")
        return states, actions, is_np, is_batched

    def _value_fn(self, idx, states):
        """Compute V(s).
        Note: subject to modification for various value function implementations.
        """
        states, is_np, is_batched = self._pre_process_inputs(states)
        returns = self._get_model(idx).network.critic(states)
        return self._post_process_outputs(returns, is_np, is_batched)

    def _q_value_fn(self, idx_action: int, states, actions):
        """Compute Q(s, a)."""
        states, actions, is_np, is_batched = self._pre_process_states_actions(
            states, actions
        )
        if self._use_learned_dynamics(idx_action):
            # TODO: Find way to map repeated action indices to unique policy indices.
            idx_policy = torch.full(
                (states.shape[0],), idx_action, dtype=torch.int64
            ).to(self._device)
            states = self.dynamics_model.decode(states, idx_policy)

        q1, q2 = self._get_model(idx_action).network.critic(states, actions)
        returns = torch.min(q1, q2).cpu().detach()
        return self._post_process_outputs(returns, is_np, is_batched)

    def _random_policy(self, envs, samples=1):
        """Query environment action spaces for a number of samples."""
        if isinstance(envs, list):
            return np.array([self._random_policy(env, samples) for env in envs])
        if samples == 1:
            return envs.action_space.sample()
        return np.array([envs.action_space.sample() for _ in range(samples)])

    def _policy(self, idx_action: int, states, **kwargs):
        """Query the policy for actions given states."""
        states, is_np, is_batched = self._pre_process_inputs(states)
        if self._use_learned_dynamics(idx_action):
            # TODO: Find way to map repeated action indices to unique policy indices.
            idx_policy = torch.full(
                (states.shape[0],), idx_action, dtype=torch.int64
            ).to(self._device)
            states = self.dynamics_model.decode(states, idx_policy)

        actions = self._get_model(idx_action).predict(states, is_batched=True, **kwargs)
        return self._post_process_outputs(actions, is_np, is_batched)

    def _sample_policy(
        self, idx, states, variance, samples=1, bounds=[-1, 1], **kwargs
    ):
        """Sample from Multivariate Gaussian distribution centered around the policy
        predicted mean with the specified variance.
        """
        actions = self._policy(idx, states, **kwargs)
        loc, is_np, is_batched = self._pre_process_inputs(actions)

        # Batch mean and covariance tensors
        cov = torch.eye(loc.size(1)).to(self._device)
        if isinstance(variance, float):
            cov *= variance
        elif isinstance(variance, list):
            cov = cov @ torch.tensor(variance).to(self._device)
        elif isinstance(variance, np.ndarray):
            cov = cov @ torch.from_numpy(variance).to(self._device)
        else:
            raise TypeError("Variance must be float or list type")

        # Sample from Multivariate Gaussian
        cov = cov.tile(loc.size(0), 1, 1)
        dist = torch.distributions.MultivariateNormal(loc=loc, covariance_matrix=cov)
        actions = torch.clamp(dist.sample((samples,)), *bounds)
        if samples == 1:
            actions = actions.squeeze(0)
        else:
            actions = actions.transpose(1, 0)
        return self._post_process_outputs(actions, is_np, is_batched)

    def _simulate_env(self, idx, envs, actions):
        """Simulate forward environments (in-place) given first actions."""
        if isinstance(envs, list):
            assert len(envs) == len(actions)
            # print(actions.shape)
            outputs = [
                self._simulate_env(idx, env, action)
                for env, action in zip(envs, actions)
            ]
            next_states, success = list(zip(*outputs))
            return (
                np.array(next_states).squeeze(),
                np.array(success, dtype=bool).squeeze(),
            )

        for _ in range(envs._max_episode_steps):
            # print(actions)
            state, _, terminated, truncated, info = envs.step(actions)
            if terminated or truncated:
                break
            actions = self._actor_interface(idx, envs=envs, states=state)
        return envs._get_observation(), info["success"]

    def _simulate_model(
        self,
        idx_action: int,
        states: Union[torch.Tensor, np.ndarray],
        actions: Union[torch.Tensor, np.ndarray],
    ) -> Union[torch.Tensor, np.ndarray]:
        """Simulate forward dynamics with learned model."""
        states, actions, is_np, is_batched = self._pre_process_states_actions(
            states, actions
        )

        # TODO: Find way to map repeated action indices to unique policy indices.
        idx_policy = torch.full((actions.shape[0],), idx_action, dtype=torch.int32)

        next_states = self.dynamics_model.forward(states, idx_policy, actions)
        return self._post_process_outputs(next_states, is_np, is_batched)

    def _encode_state(
        self, idx_action: int, states: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        """Encode into latent state."""
        states, is_np, is_batched = self._pre_process_inputs(states)

        # TODO: Find way to map repeated action indices to unique policy indices.
        idx_policy = torch.full((states.shape[0],), idx_action, dtype=torch.int32)

        latents = self.dynamics_model.encode(states, idx_policy)
        return self._post_process_outputs(latents, is_np, is_batched)
