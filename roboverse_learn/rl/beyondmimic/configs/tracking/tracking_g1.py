import math
from metasim.utils import configclass
from metasim.scenario.scenario import ScenarioCfg
from typing import Callable

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
from roboverse_learn.rl.beyondmimic.helper.motion_utils import MotionCommand, MotionCommandCfg  # TODO
from roboverse_learn.rl.beyondmimic.configs.callback_funcs import (
    termination_funcs,
    reset_funcs,
    step_funcs,
    reward_funcs,
    observation_funcs as obs_funcs,
    event_funcs,
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
    func: Callable
    params: dict[str, any] | None = None


@configclass
class ObsTerm(CfgTerm):
    noise_range: tuple[float, float] | None = None


@configclass
class RewTerm(CfgTerm):
    weight: float


# TODO add `cmd` as a default param to the functions
@configclass
class ObservationsCfg:
    # TODO add `mdp.generated_commands` from BeyondMimic?
    @configclass
    class PolicyCfg:
        command = ObsTerm(func=obs_funcs.generated_commands)  # TODO
        motion_anchor_pos_b = ObsTerm(func=obs_funcs.motion_anchor_pos_b, noise_range=(-0.25, 0.25))
        motion_anchor_ori_b = ObsTerm(func=obs_funcs.motion_anchor_ori_b, noise_range=(-0.05, 0.05))
        base_lin_vel = ObsTerm(func=obs_funcs.base_lin_vel, noise_range=(-0.5, 0.5))
        base_ang_vel = ObsTerm(func=obs_funcs.base_ang_vel, noise_range=(-0.2, 0.2))
        joint_pos = ObsTerm(func=obs_funcs.joint_pos_rel, noise_range=(-0.01, 0.01))
        joint_vel = ObsTerm(func=obs_funcs.joint_vel_rel, noise_range=(-0.5, 0.5))
        actions = ObsTerm(func=obs_funcs.last_action)  # TODO

    @configclass
    class PrivilegedCfg:
        command = ObsTerm(func=obs_funcs.generated_commands)  # TODO
        motion_anchor_pos_b = ObsTerm(func=obs_funcs.motion_anchor_pos_b)
        motion_anchor_ori_b = ObsTerm(func=obs_funcs.motion_anchor_ori_b)
        body_pos = ObsTerm(func=obs_funcs.robot_body_pos_b)
        body_ori = ObsTerm(func=obs_funcs.robot_body_ori_b)
        base_lin_vel = ObsTerm(func=obs_funcs.base_lin_vel)
        base_ang_vel = ObsTerm(func=obs_funcs.base_ang_vel)
        joint_pos = ObsTerm(func=obs_funcs.joint_pos_rel)
        joint_vel = ObsTerm(func=obs_funcs.joint_vel_rel)
        actions = ObsTerm(func=obs_funcs.last_action)  # TODO

    policy = PolicyCfg()
    critic = PrivilegedCfg()


@configclass
class RewardsCfg:  # TODO check how reward weight is used in unitree_rl (weight * func_return_value * sim_dt?)
    motion_global_anchor_pos = RewTerm(func=reward_funcs.motion_global_anchor_position_error_exp, weight=0.5, params={"std": 0.3})
    motion_global_anchor_ori = RewTerm(func=reward_funcs.motion_global_anchor_orientation_error_exp, weight=0.5, params={"std": 0.4})
    motion_body_pos = RewTerm(func=reward_funcs.motion_relative_body_position_error_exp, weight=1.0, params={"std": 0.3})
    motion_body_ori = RewTerm(func=reward_funcs.motion_relative_body_orientation_error_exp, weight=1.0, params={"std": 0.4})
    motion_body_lin_vel = RewTerm(func=reward_funcs.motion_global_body_linear_velocity_error_exp, weight=1.0, params={"std": 1.0})
    motion_body_ang_vel = RewTerm(func=reward_funcs.motion_global_body_angular_velocity_error_exp, weight=1.0, params={"std": 3.14})
    action_rate_l2 = RewTerm(func=reward_funcs.action_rate_l2, weight=0.05, params={"std": 3.14})
    joint_limit = RewTerm(func=reward_funcs.joint_pos_limits, weight=-10.0)
    undesired_contacts = RewTerm(func=reward_funcs.undesired_contacts, weight=-0.1, params={"threshold": 1, "body_names": [r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$).+$"]})  # TODO check if `body_names` is correctly parsed


@configclass
class CommandsCfg:
    motion = MotionCommandCfg(
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


@configclass
class TrackingG1EnvCfg(BaseEnvCfg):
    """Environment configuration for humanoid motion tracking task."""

    episode_length_s = 10.0
    # TODO check these two values in BeyondMimic
    obs_len_history = 5
    priv_obs_len_history = 5

    control = BaseEnvCfg.Control(action_scale=0.25, soft_joint_pos_limit_factor=0.9)

    observations = ObservationsCfg()  # TODO
    rewards = RewardsCfg()

    # NOTE extra obs will be included in `env_states.extras["contact_forces"]`
    callbacks_query = {"contact_forces": ContactForces(history_length=3)}
    callbacks_setup = {  # TODO check if these physics params have the same effect in both Isaac Lab and Metasim
        "material_randomizer": MaterialRandomizer(
            obj_name="g1_dof29",
            static_friction_range=(0.3, 1.6),
            dynamic_friction_range=(0.3, 1.2),
            restitution_range=(0.0, 0.0),
            num_buckets=64,
        ),
        # TODO change `MassRandomizer` to `randomize_rigid_body_com()` from BeyondMimic
        "mass_randomizer": MassRandomizer(
            obj_name="g1_dof29",
            body_names="torso_link",
            mass_distribution_params=(-1.0, 3.0),
            operation="add",
        ),
        # NOTE `env` will be passed to the functions inside `AgentTask._bind_callbacks()`
        "add_joint_default_pos": (
            event_funcs.randomize_joint_default_pos,
            {
                "pos_distribution_params": (-0.01, 0.01),
                "operation": "add",
            }
        ),
    }
    callbacks_reset = {
        "random_root_state": (
            reset_funcs.random_root_state,
            {
                "pose_range": [
                    [-0.5, -0.5, 0.0, 0, 0, -3.14],  # x,y,z roll,pitch,yaw
                    [0.5, 0.5, 0.0, 0, 0, 3.14],
                ],
                "velocity_range": [[0] * 6, [0] * 6],
            },
        ),
        "reset_joints_by_scale": (
            reset_funcs.reset_joints_by_scale,
            {"position_range": (1.0, 1.0), "velocity_range": (-1.0, 1.0)},
        ),
    }
    callbacks_post_step = {
        # TODO slightly different from BeyondMimic – check if this works
        "push_robot": (
            step_funcs.push_by_setting_velocity,
            {
                "interval_range_s": (1.0, 3.0),
                "velocity_range": VELOCITY_RANGE,
            },
        )
    }
    callbacks_terminate = {
        "time_out": termination_funcs.time_out,
        "base_height": (
            termination_funcs.root_height_below_minimum,
            {"minimum_height": 0.2},
        ),
        "bad_orientation": (termination_funcs.bad_orientation, {"limit_angle": 0.8}),
    }

    def __post_init__(self, motion_file: str, scenario: ScenarioCfg, resample: Callable, device: str):
        self.commands = MotionCommand(motion_file, scenario, resample, device)  # TODO


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
