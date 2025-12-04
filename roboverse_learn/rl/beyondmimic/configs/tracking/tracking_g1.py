from metasim.utils import configclass
from typing import Callable
from dataclasses import MISSING

from roboverse_learn.rl.beyondmimic.configs.cfg_base import BaseEnvCfg
from roboverse_learn.rl.beyondmimic.configs.algorithm import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)
from roboverse_learn.rl.beyondmimic.configs.cfg_queries import ContactForces
from roboverse_learn.rl.beyondmimic.configs.cfg_randomizers import (
    MaterialRandomizer,
    MassRandomizer,
)
from roboverse_learn.rl.beyondmimic.mdp.commands import MotionCommandCfg
from roboverse_learn.rl.beyondmimic.mdp import (
    events,
    terminations,
    rewards,
    observations,
    events,
)


VELOCITY_RANGE = {
    # linear velocity
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    # angular velocity
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
}


# TODO should be moved to a separate file or the same location as config superclass
@configclass
class CfgTerm:
    func: Callable = MISSING
    params: dict[str, any] | None = None


@configclass
class ObsTerm(CfgTerm):
    noise_range: tuple[float, float] | None = None


@configclass
class RewTerm(CfgTerm):
    weight: float = 1.0


@configclass
class ObservationsCfg:

    @configclass
    class PolicyCfg:
        command = ObsTerm(func=observations.generated_commands)
        motion_anchor_pos_b = ObsTerm(func=observations.motion_anchor_pos_b, noise_range=(-0.25, 0.25))
        motion_anchor_ori_b = ObsTerm(func=observations.motion_anchor_ori_b, noise_range=(-0.05, 0.05))
        base_lin_vel = ObsTerm(func=observations.base_lin_vel, noise_range=(-0.5, 0.5))
        base_ang_vel = ObsTerm(func=observations.base_ang_vel, noise_range=(-0.2, 0.2))
        joint_pos = ObsTerm(func=observations.joint_pos_rel, noise_range=(-0.01, 0.01))
        joint_vel = ObsTerm(func=observations.joint_vel_rel, noise_range=(-0.5, 0.5))
        actions = ObsTerm(func=observations.last_action)

    @configclass
    class PrivilegedCfg:
        command = ObsTerm(func=observations.generated_commands)
        motion_anchor_pos_b = ObsTerm(func=observations.motion_anchor_pos_b)
        motion_anchor_ori_b = ObsTerm(func=observations.motion_anchor_ori_b)
        body_pos = ObsTerm(func=observations.robot_body_pos_b)
        body_ori = ObsTerm(func=observations.robot_body_ori_b)
        base_lin_vel = ObsTerm(func=observations.base_lin_vel)
        base_ang_vel = ObsTerm(func=observations.base_ang_vel)
        joint_pos = ObsTerm(func=observations.joint_pos_rel)
        joint_vel = ObsTerm(func=observations.joint_vel_rel)
        actions = ObsTerm(func=observations.last_action)

    # observation groups
    policy = PolicyCfg()
    critic = PrivilegedCfg()


@configclass
class RewardsCfg:
    motion_global_anchor_pos = RewTerm(func=rewards.motion_global_anchor_position_error_exp, weight=0.5, params={"std": 0.3})
    motion_global_anchor_ori = RewTerm(func=rewards.motion_global_anchor_orientation_error_exp, weight=0.5, params={"std": 0.4})
    motion_body_pos = RewTerm(func=rewards.motion_relative_body_position_error_exp, weight=1.0, params={"std": 0.3})
    motion_body_ori = RewTerm(func=rewards.motion_relative_body_orientation_error_exp, weight=1.0, params={"std": 0.4})
    motion_body_lin_vel = RewTerm(func=rewards.motion_global_body_linear_velocity_error_exp, weight=1.0, params={"std": 1.0})
    motion_body_ang_vel = RewTerm(func=rewards.motion_global_body_angular_velocity_error_exp, weight=1.0, params={"std": 3.14})
    action_rate_l2 = RewTerm(func=rewards.action_rate_l2, weight=-1e-1)
    joint_limit = RewTerm(func=rewards.joint_pos_limits, weight=-10.0)
    undesired_contacts = RewTerm(func=rewards.undesired_contacts, weight=-0.1, params={"threshold": 1.0, "body_names": r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$).+$"})  # TODO check if `body_names` is correctly parsed


@configclass
class TrackingG1EnvCfg(BaseEnvCfg):
    """Environment configuration for humanoid motion tracking task."""

    control = BaseEnvCfg.Control(action_scale=0.25, soft_joint_pos_limit_factor=0.9)

    commands = MotionCommandCfg(
        anchor_body_name="torso_link",
        body_names=[  # for indexing motion body links
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ],
        resampling_time_range=(1.0e9, 1.0e9),
        pose_range={
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.01, 0.01),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-0.2, 0.2),
        },
        velocity_range=VELOCITY_RANGE,
        joint_position_range=(-0.1, 0.1),
    )
    observations = ObservationsCfg()  # TODO compute obs
    rewards = RewardsCfg()

    # NOTE extra obs will be included in `env_states.extras["contact_forces"]`
    callbacks_query = {"contact_forces": ContactForces(history_length=3)}
    # TODO fully align domain randomization with BeyondMimic
    callbacks_setup = {
        "material_randomizer": MaterialRandomizer(
            obj_name="g1_tracking",
            static_friction_range=(0.3, 1.6),
            dynamic_friction_range=(0.3, 1.2),
            restitution_range=(0.0, 0.5),
            num_buckets=64,
        ),
        # TODO change `MassRandomizer` to `randomize_rigid_body_com()` from BeyondMimic
        "mass_randomizer": MassRandomizer(
            obj_name="g1_tracking",
            body_names="torso_link",
            mass_distribution_params=(-1.0, 3.0),  # TODO change this
            operation="add",
        ),
        # NOTE `env` will be passed to the functions inside `AgentTask._bind_callbacks()`
        "add_joint_default_pos": (
            events.randomize_joint_default_pos,
            {
                "pos_distribution_params": (-0.01, 0.01),
                "operation": "add",
            }
        ),
    }
    callbacks_post_step = {
        # TODO slightly different from BeyondMimic – check if this works
        "push_robot": (
            events.push_by_setting_velocity,
            {
                "interval_range_s": (1.0, 3.0),
                "velocity_range": VELOCITY_RANGE,
            },
        )
    }
    callbacks_terminate = {
        "time_out": terminations.time_out,
        "anchor_pos": (terminations.bad_anchor_pos_z_only, {"threshold": 0.25}),
        "anchor_ori": (terminations.bad_anchor_ori, {"threshold": 0.8}),
        "ee_body_pos": (terminations.bad_motion_body_pos_z_only, {"threshold": 0.25, "body_names": ["left_ankle_roll_link", "right_ankle_roll_link", "left_wrist_yaw_link", "right_wrist_yaw_link"]}),
    }


@configclass
class TrackingG1RslRlTrainCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 500
    experiment_name = ""  # same as task name
    empirical_normalization = True
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
