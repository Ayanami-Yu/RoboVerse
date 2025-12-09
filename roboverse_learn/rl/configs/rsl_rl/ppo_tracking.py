from typing import Literal, Optional
from metasim.utils import configclass
from roboverse_learn.rl.configs.rsl_rl.ppo import RslRlPPOConfig
from roboverse_learn.rl.configs.rsl_rl.algorithm import RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


SimBackend = Literal[
    "isaacgym",
    "isaacsim",
    "isaaclab",
    "mujoco",
    "genesis",
    "mjx",
]


@configclass
class RslRlPPOTrackingConfig(RslRlPPOConfig):
    """RSL-RL PPO configs for motion tracking task."""
    # Experiment / runner settings
    max_iterations = 30000
    save_interval = 500
    empirical_normalization = True  # deprecated
    wandb_project: str = "rsl_rl_ppo_tracking"

    # Environment / device
    task = "beyondmimic.tracking.isaaclab"
    robot = "g1_tracking"  # unused
    sim: SimBackend = "isaacsim"

    # Logging
    use_wandb: bool = True

    # WandB registry for loading motions
    registry_name: Optional[str] = None

    # Policy configuration
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    # Algorithm configuration
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.05,  # TODO the only difference?
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

    def __post_init__(self) -> None:
        super().__post_init__()

        if ":" not in self.registry_name:  # Check if the registry name includes alias, if not, append ":latest"
            self.registry_name += ":latest"

        import pathlib
        import wandb

        api = wandb.Api()
        artifact = api.artifact(self.registry_name)
        self.motion_file = str(pathlib.Path(artifact.download()) / "motion.npz")
