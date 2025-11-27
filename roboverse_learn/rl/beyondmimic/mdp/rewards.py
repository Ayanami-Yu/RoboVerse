from __future__ import annotations

import torch

from metasim.types import TensorState
from metasim.utils.math import quat_error_magnitude

from roboverse_learn.rl.beyondmimic.mdp.commands import MotionCommand
from roboverse_pack.tasks.beyondmimic.base.types import EnvTypes
from roboverse_learn.rl.beyondmimic.helper.utils import get_indexes
from roboverse_learn.rl.beyondmimic.configs.cfg_queries import ContactForces


# adapted from BeyondMimic rewards.py

def motion_global_anchor_position_error_exp(env: EnvTypes, env_states: TensorState, std: float) -> torch.Tensor:
    robot_state = env_states.robots[env.name]
    error = torch.sum(torch.square(env.commands.anchor_pos_w - robot_state.root_state[:, :3]), dim=-1)
    return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(env: EnvTypes, env_states: TensorState, std: float) -> torch.Tensor:
    robot_state = env_states.robots[env.name]
    error = quat_error_magnitude(env.commands.anchor_quat_w, robot_state.root_state[:, 3:7]) ** 2
    return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
    env: EnvTypes, env_states: TensorState, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    robot_state = env_states.robots[env.name]
    body_indexes = get_indexes(env, body_names, env_states.robots[env.name].body_names)
    error = torch.sum(
        torch.square(env.commands.body_pos_relative_w[:, body_indexes] - robot_state.body_state[:, body_indexes, :3]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
    env: EnvTypes, env_states: TensorState, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    robot_state = env_states.robots[env.name]
    body_indexes = get_indexes(env, body_names, env_states.robots[env.name].body_names)
    error = (
        quat_error_magnitude(env.commands.body_quat_relative_w[:, body_indexes], robot_state.body_state[:, body_indexes, 3:7])
        ** 2
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
    env: EnvTypes, env_states: TensorState, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Linear velocity tracking error."""
    robot_state = env_states.robots[env.name]
    body_indexes = get_indexes(env, body_names, env_states.robots[env.name].body_names)
    error = torch.sum(
        torch.square(env.commands.body_lin_vel_w[:, body_indexes] - robot_state.body_state[:, body_indexes, 7:10]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
    env: EnvTypes, env_states: TensorState, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    robot_state = env_states.robots[env.name]
    body_indexes = get_indexes(env, body_names, env_states.robots[env.name].body_names)
    error = torch.sum(
        torch.square(env.commands.body_ang_vel_w[:, body_indexes] - robot_state.body_state[:, body_indexes, 10:13]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


# adapted from `isaaclab.envs.mdp.rewards.py`

def action_rate_l2(env: EnvTypes, env_states: TensorState) -> torch.Tensor:  # TODO check where action is stored
    """Penalize the rate of change of the actions using L2 squared kernel."""
    return torch.sum(
        torch.square(env.actions - env.history_buffer["actions"][-1]), dim=1
    )  # [n_envs, n_dims]


def joint_pos_limits(env: EnvTypes, env_states: TensorState) -> torch.Tensor:
    """Penalize joint positions if they cross the soft limits.

    This is computed as a sum of the absolute value of the difference between the joint position and the soft limits.
    """
    robot_state = env_states.robots[env.name]
    out_of_limits = -(robot_state.joint_pos - env.soft_dof_pos_limits[:, 0]).clip(
        max=0.0
    )
    out_of_limits += (robot_state.joint_pos - env.soft_dof_pos_limits[:, 1]).clip(
        min=0.0
    )
    return torch.sum(out_of_limits, dim=1)


def undesired_contacts(
    env: EnvTypes,
    env_states: TensorState,
    threshold: float,
    body_names: str | tuple[str] = "(?!.*ankle.*).*",
) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""
    indexes = get_indexes(env, body_names, env_states.robots[env.name].body_names)
    contact_forces: ContactForces = env_states.extras["contact_forces"][env.name]
    is_contact = (
        # TODO check correspondence with `contact_sensor.data.net_forces_w_history`
        contact_forces.contact_forces_history[:, :, indexes, :]  # [n_envs, history_length, n_bodies, 3] -> [n_envs, 3, 30, 3]
        .norm(dim=-1)
        .max(dim=1)[0]
        > threshold
    )

    # sum over contacts for each environment
    return torch.sum(is_contact, dim=1)  # [n_envs]
