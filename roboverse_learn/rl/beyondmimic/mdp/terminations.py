from __future__ import annotations

import torch

from metasim.types import TensorState
from metasim.utils.math import quat_rotate_inverse

from roboverse_pack.tasks.beyondmimic.base.types import EnvTypes
from roboverse_learn.rl.beyondmimic.mdp.commands import MotionCommand
from roboverse_learn.rl.beyondmimic.helper.utils import get_body_indexes


# FIXME `env_states` receives params from `LeggedRobotTask._terminated()` but not used
# TODO check if this is needed; if so, add a `cmd` param
def time_out(env: EnvTypes, env_states: TensorState) -> torch.Tensor:
    """Terminate the episode when the episode length exceeds the maximum episode length."""
    return env._episode_steps >= env.max_episode_steps


# adapted from BeyondMimic terminations.py  # TODO

def bad_anchor_pos_z_only(env: EnvTypes, env_states: TensorState, threshold: float) -> torch.Tensor:
    robot_state = env_states.robots[env.name]
    return torch.abs(env.commands.anchor_pos_w[:, 2] - robot_state.root_state[:, 2]) > threshold


def bad_anchor_ori(env: EnvTypes, env_states: TensorState, threshold: float) -> torch.Tensor:
    robot_state = env_states.robots[env.name]
    motion_projected_gravity_b = quat_rotate_inverse(env.commands.anchor_quat_w, env.gravity_vec)  # [n_envs, 3]
    robot_projected_gravity_b = quat_rotate_inverse(robot_state.root_state[:, 3:7], env.gravity_vec)

    # check whether the robot's tilt magnitude deviates too much (how relatively "upright"), and ignores which way it leans
    return (motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]).abs() > threshold


def bad_motion_body_pos_z_only(env: EnvTypes, env_states: TensorState, threshold: float, body_names: list[str]) -> torch.Tensor:
    robot_state = env_states.robots[env.name]
    body_indexes = get_body_indexes(env.commands, body_names)
    error = torch.abs(env.commands.body_pos_relative_w[:, body_indexes, 2] - robot_state.body_state[:, body_indexes, 2])
    return torch.any(error > threshold, dim=-1)
