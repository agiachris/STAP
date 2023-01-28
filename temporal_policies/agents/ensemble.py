from typing import Optional, Union, Type

from temporal_policies.agents import base, wrapper
from temporal_policies import envs
from temporal_policies.utils import configs
from temporal_policies.networks import critics


class EnsembleAgent(wrapper.WrapperAgent):
    """Agent wrapper that uses a bootstrap ensemble of Q-functions to produce
    estimates of expected rewards that factor-in epistemic uncertainty."""

    def __init__(
        self,
        policy: base.Agent,
        env: Optional[envs.Env] = None,
        critic_class: Union[str, Type[critics.ContinuousEnsembleCritic]] = critics.EnsembleLCBCritic,
        pessimistic: bool = True,
        clip: bool = True,
        lcb_scale: Optional[float] = None,
        ood_threshold: Optional[float] = None,
        ood_value: Optional[float] = None,
        device: str = "auto",
    ):
        """Constructs the EnsembleAgent.

        Args:
            policy: Main agent with an ensemble of Q-functions.
            env: Policy env (unused, but included for API consistency).
            pessimistic: Estimated rewards from min(Qi) instead of mean(Qi).
            clip: Clip Q-values between [0, 1].
            lcb_scale (critics.EnsembleLCBCritic): Lower confidence bound (LCB) scale factor.
            ood_threshold (critics.EnsembleThresholdCritic): Out-of-distribution threshold on std(Qi).
            ood_value (critics.EnsembleThresholdCritic): Value assignment to out-of-distribution detected sample.
            device: Torch device.
        """
        critic_class = configs.get_class(critic_class, critics)
        if not issubclass(critic_class, critics.ContinuousEnsembleCritic):
            raise ValueError("Must supply valid subclass of ContinuousEnsembleCritic.")

        if issubclass(critic_class, critics.EnsembleLCBCritic):
            critic = critic_class(
                scale=lcb_scale,
                critic=policy.critic,
                pessimistic=pessimistic,
                clip=clip
            )
        elif issubclass(critic_class, critics.EnsembleThresholdCritic):
            critic = critic_class(
                threshold=ood_threshold,
                value=ood_value,
                critic=policy.critic,
                pessimistic=pessimistic,
                clip=clip
            )
        elif issubclass(critic_class, critics.EnsembleOODCritic):
            critic = critic_class(
                threshold=ood_threshold,
                critic=policy.critic,
                pessimistic=pessimistic,
                clip=clip,
            )
        else:
            raise ValueError(f"{critic_class} not supported by EnsembleAgent.")
            
        super().__init__(
            state_space=policy.state_space,
            action_space=policy.action_space,
            observation_space=policy.observation_space,
            actor=policy.actor,
            critic=critic,
            encoder=policy.encoder,
            device=device,
        )
