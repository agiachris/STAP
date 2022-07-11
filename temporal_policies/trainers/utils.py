import pathlib
from typing import Any, Dict, Optional, Sequence, Union

from temporal_policies import agents, dynamics, encoders, trainers
from temporal_policies.dynamics import load as load_dynamics
from temporal_policies.dynamics import LatentDynamics, load_policy_checkpoints
from temporal_policies.utils import configs


class TrainerFactory(configs.Factory):
    """Trainer factory."""

    def __init__(
        self,
        path: Optional[Union[str, pathlib.Path]] = None,
        config: Optional[Union[str, pathlib.Path, Dict[str, Any]]] = None,
        agent: Optional[agents.RLAgent] = None,
        dynamics: Optional[dynamics.LatentDynamics] = None,
        encoder: Optional[encoders.Encoder] = None,
        checkpoint: Optional[Union[str, pathlib.Path]] = None,
        policy_checkpoints: Optional[Sequence[Union[str, pathlib.Path]]] = None,
        agent_trainers: Optional[Sequence["trainers.AgentTrainer"]] = None,
        device: str = "auto",
    ):
        """Creates the trainer factory from a config or checkpoint.

        Args:
            path: Training output path.
            config: Optional dynamics config path or dict. Must be provided if
                checkpoint is None.
            agent: Agent to be trained.
            dynamics: Dynamics model to be trained.
            checkpoint: Optional trainer checkpoint. Must be provided if config
                is None.
            policy_checkpoints: Optional list of policy checkpoints for dynamics.
            agent_trainers: Optional list of agent trainers for dynamics.
            device: Torch device.
        """
        if checkpoint is not None:
            ckpt_config = load_config(checkpoint)
            if config is None:
                config = ckpt_config

        if config is None:
            raise ValueError("Either config or checkpoint must be specified")

        super().__init__(config, "trainer", trainers)

        if issubclass(self.cls, trainers.AgentTrainer):
            if agent is None:
                if checkpoint is None:
                    raise ValueError("Either agent or checkpoint must be specified")
                ckpt_agent = agents.load(checkpoint=checkpoint, device=device)
                if not isinstance(ckpt_agent, agents.RLAgent):
                    raise ValueError("Checkpoint agent must be an RLAgent instance")
                agent = ckpt_agent

            self.kwargs["agent"] = agent
        elif issubclass(self.cls, (trainers.DynamicsTrainer, trainers.UnifiedTrainer)):
            if dynamics is None:
                if checkpoint is None:
                    raise ValueError("Either dynamics or checkpoint must be specified")
                ckpt_dynamics = load_dynamics(checkpoint=checkpoint, device=device)
                if not isinstance(ckpt_dynamics, LatentDynamics):
                    raise ValueError("Checkpoint dynamics must be a LatentDynamics instance")
                dynamics = ckpt_dynamics

            self.kwargs["dynamics"] = dynamics

            if issubclass(self.cls, trainers.DynamicsTrainer):
                if agent_trainers is not None:
                    self.kwargs["agent_trainers"] = agent_trainers
                elif policy_checkpoints is None:
                    if checkpoint is None:
                        raise ValueError(
                            "One of agent_trainers, policy_checkpoints, or "
                            "checkpoint must be specified"
                        )
                    policy_checkpoints = load_policy_checkpoints(checkpoint)

                self.kwargs["policy_checkpoints"] = policy_checkpoints
        elif issubclass(self.cls, trainers.AutoencoderTrainer):
            if encoder is None:
                if checkpoint is None:
                    raise ValueError("Either encoder or checkpoint must be specified")
                ckpt_encoder = encoders.load(checkpoint=checkpoint, device=device)
                if not isinstance(ckpt_encoder, encoders.Autoencoder):
                    raise ValueError("Checkpoint encoder must be an Autoencoder instance")
                encoder = ckpt_encoder

            self.kwargs["encoder"] = encoder

            if agent_trainers is not None:
                self.kwargs["agent_trainers"] = agent_trainers
            elif policy_checkpoints is None:
                if checkpoint is None:
                    raise ValueError(
                        "One of agent_trainers, policy_checkpoints, or "
                        "checkpoint must be specified"
                    )
                policy_checkpoints = load_policy_checkpoints(checkpoint)

            self.kwargs["policy_checkpoints"] = policy_checkpoints
        else:
            raise NotImplementedError

        if checkpoint is not None:
            if self.config["trainer"] != ckpt_config["trainer"]:
                raise ValueError(
                    f"Config trainer [{self.config['trainer']}] and checkpoint "
                    f"trainer [{ckpt_config['trainer']}] must be the same"
                )
            self.kwargs["checkpoint"] = checkpoint
            self.kwargs["path"] = pathlib.Path(checkpoint).parent.parent
        else:
            self.kwargs["path"] = path

        self.kwargs["device"] = device


def load(
    path: Optional[Union[str, pathlib.Path]] = None,
    config: Optional[Union[str, pathlib.Path, Dict[str, Any]]] = None,
    agent: Optional[agents.RLAgent] = None,
    dynamics: Optional[dynamics.LatentDynamics] = None,
    encoder: Optional[encoders.Encoder] = None,
    checkpoint: Optional[Union[str, pathlib.Path]] = None,
    policy_checkpoints: Optional[Sequence[Union[str, pathlib.Path]]] = None,
    agent_trainers: Optional[Sequence["trainers.AgentTrainer"]] = None,
    device: str = "auto",
    **kwargs,
) -> trainers.Trainer:
    """Loads the trainer factory from a config or checkpoint.

    Args:
        path: Training output path.
        config: Optional dynamics config path or dict. Must be provided if
            checkpoint is None.
        agent: Agent to be trained.
        dynamics: Dynamics model to be trained.
        checkpoint: Optional trainer checkpoint. Must be provided if config is
            None.
        policy_checkpoints: Optional list of policy checkpoints for dynamics.
        agent_trainers: Optional list of agent trainers for dynamics.
        device: Torch device.
        **kwargs: Optional trainer constructor kwargs.

    Returns:
        Trainer instance.
    """
    trainer_factory = TrainerFactory(
        path=path,
        config=config,
        agent=agent,
        dynamics=dynamics,
        encoder=encoder,
        checkpoint=checkpoint,
        policy_checkpoints=policy_checkpoints,
        agent_trainers=agent_trainers,
        device=device,
    )
    return trainer_factory(**kwargs)


def load_config(path: Union[str, pathlib.Path]) -> Dict[str, Any]:
    """Loads a trainer config from path.

    Args:
        path: Path to the config, config directory, or checkpoint.

    Returns:
        Trainer config dict.
    """
    return configs.load_config(path, "trainer")
