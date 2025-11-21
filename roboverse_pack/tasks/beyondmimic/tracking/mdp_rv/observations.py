"""RoboVerse-compatible observation computation for motion tracking."""

from __future__ import annotations

import torch

from metasim.types import TensorState
from metasim.utils.math import matrix_from_quat, subtract_frame_transforms

from .commands import MotionCommandRV


def compute_observations(
    env_states: TensorState,
    motion_command: MotionCommandRV,
    robot_name: str,
    last_actions: torch.Tensor | None,
    enable_corruption: bool = True,
    noise_params: dict[str, tuple[float, float]] | None = None,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Compute policy observations from environment state.

    Args:
        env_states: Current environment states
        motion_command: Motion command instance
        robot_name: Name of the robot
        last_actions: Last actions taken
        enable_corruption: Whether to add noise
        noise_params: Noise parameters for each observation component
        device: Device to use

    Returns:
        Observation tensor
    """
    robot_state = env_states.robots[robot_name]
    num_envs = robot_state.root_state.shape[0]

    obs_components = []

    # Command (joint pos + vel)
    command = motion_command.command
    obs_components.append(command)

    # Motion anchor position in robot anchor body frame
    anchor_pos_b = motion_anchor_pos_b(env_states, motion_command, robot_name)
    if enable_corruption and noise_params and "motion_anchor_pos_b" in noise_params:
        noise_min, noise_max = noise_params["motion_anchor_pos_b"]
        noise = torch.rand_like(anchor_pos_b) * (noise_max - noise_min) + noise_min
        anchor_pos_b = anchor_pos_b + noise
    obs_components.append(anchor_pos_b)

    # Motion anchor orientation in robot anchor body frame
    anchor_ori_b = motion_anchor_ori_b(env_states, motion_command, robot_name)
    if enable_corruption and noise_params and "motion_anchor_ori_b" in noise_params:
        noise_min, noise_max = noise_params["motion_anchor_ori_b"]
        noise = torch.rand_like(anchor_ori_b) * (noise_max - noise_min) + noise_min
        anchor_ori_b = anchor_ori_b + noise
    obs_components.append(anchor_ori_b)

    # Base linear velocity
    base_lin_vel = robot_state.root_state[:, 7:10]
    if enable_corruption and noise_params and "base_lin_vel" in noise_params:
        noise_min, noise_max = noise_params["base_lin_vel"]
        noise = torch.rand_like(base_lin_vel) * (noise_max - noise_min) + noise_min
        base_lin_vel = base_lin_vel + noise
    obs_components.append(base_lin_vel)

    # Base angular velocity
    base_ang_vel = robot_state.root_state[:, 10:13]
    if enable_corruption and noise_params and "base_ang_vel" in noise_params:
        noise_min, noise_max = noise_params["base_ang_vel"]
        noise = torch.rand_like(base_ang_vel) * (noise_max - noise_min) + noise_min
        base_ang_vel = base_ang_vel + noise
    obs_components.append(base_ang_vel)

    # Joint positions (relative to default)
    joint_pos = robot_state.joint_pos
    if enable_corruption and noise_params and "joint_pos" in noise_params:
        noise_min, noise_max = noise_params["joint_pos"]
        noise = torch.rand_like(joint_pos) * (noise_max - noise_min) + noise_min
        joint_pos = joint_pos + noise
    obs_components.append(joint_pos)

    # Joint velocities
    joint_vel = robot_state.joint_vel
    if enable_corruption and noise_params and "joint_vel" in noise_params:
        noise_min, noise_max = noise_params["joint_vel"]
        noise = torch.rand_like(joint_vel) * (noise_max - noise_min) + noise_min
        joint_vel = joint_vel + noise
    obs_components.append(joint_vel)

    # Last actions
    if last_actions is not None:
        obs_components.append(last_actions)
    else:
        obs_components.append(torch.zeros(num_envs, motion_command.num_actions, device=device))

    return torch.cat(obs_components, dim=1)


def compute_privileged_observations(
    env_states: TensorState,
    motion_command: MotionCommandRV,
    robot_name: str,
    last_actions: torch.Tensor | None,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Compute privileged observations from environment state (no noise).

    Args:
        env_states: Current environment states
        motion_command: Motion command instance
        robot_name: Name of the robot
        last_actions: Last actions taken
        device: Device to use

    Returns:
        Privileged observation tensor
    """
    robot_state = env_states.robots[robot_name]
    num_envs = robot_state.root_state.shape[0]

    obs_components = []

    # Command
    command = motion_command.command
    obs_components.append(command)

    # Motion anchor position in robot anchor body frame (no noise)
    anchor_pos_b = motion_anchor_pos_b(env_states, motion_command, robot_name)
    obs_components.append(anchor_pos_b)

    # Motion anchor orientation in robot anchor body frame (no noise)
    anchor_ori_b = motion_anchor_ori_b(env_states, motion_command, robot_name)
    obs_components.append(anchor_ori_b)

    # Robot body positions in anchor body frame
    body_pos_b = robot_body_pos_b(env_states, motion_command, robot_name)
    obs_components.append(body_pos_b)

    # Robot body orientations in anchor body frame
    body_ori_b = robot_body_ori_b(env_states, motion_command, robot_name)
    obs_components.append(body_ori_b)

    # Base linear velocity (no noise)
    base_lin_vel = robot_state.root_state[:, 7:10]
    obs_components.append(base_lin_vel)

    # Base angular velocity (no noise)
    base_ang_vel = robot_state.root_state[:, 10:13]
    obs_components.append(base_ang_vel)

    # Joint positions (no noise)
    joint_pos = robot_state.joint_pos
    obs_components.append(joint_pos)

    # Joint velocities (no noise)
    joint_vel = robot_state.joint_vel
    obs_components.append(joint_vel)

    # Last actions
    if last_actions is not None:
        obs_components.append(last_actions)
    else:
        obs_components.append(torch.zeros(num_envs, motion_command.num_actions, device=device))

    return torch.cat(obs_components, dim=1)


def motion_anchor_pos_b(env_states: TensorState, motion_command: MotionCommandRV, robot_name: str) -> torch.Tensor:
    """Get motion command anchor position in robot anchor body frame."""
    robot_state = env_states.robots[robot_name]
    robot_anchor_body_index = motion_command.robot_anchor_body_index

    robot_anchor_pos_w = robot_state.body_state[:, robot_anchor_body_index, :3]
    robot_anchor_quat_w = robot_state.body_state[:, robot_anchor_body_index, 3:7]

    pos, _ = subtract_frame_transforms(
        robot_anchor_pos_w,
        robot_anchor_quat_w,
        motion_command.anchor_pos_w,
        motion_command.anchor_quat_w,
    )

    return pos


def motion_anchor_ori_b(env_states: TensorState, motion_command: MotionCommandRV, robot_name: str) -> torch.Tensor:
    """Get motion command anchor orientation in robot anchor body frame as rotation matrix (first 2 columns)."""
    robot_state = env_states.robots[robot_name]
    robot_anchor_body_index = motion_command.robot_anchor_body_index

    robot_anchor_pos_w = robot_state.body_state[:, robot_anchor_body_index, :3]
    robot_anchor_quat_w = robot_state.body_state[:, robot_anchor_body_index, 3:7]

    _, ori = subtract_frame_transforms(
        robot_anchor_pos_w,
        robot_anchor_quat_w,
        motion_command.anchor_pos_w,
        motion_command.anchor_quat_w,
    )
    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(mat.shape[0], -1)


def robot_body_pos_b(env_states: TensorState, motion_command: MotionCommandRV, robot_name: str) -> torch.Tensor:
    """Get robot body positions in anchor body frame."""
    robot_state = env_states.robots[robot_name]
    robot_anchor_body_index = motion_command.robot_anchor_body_index
    body_indexes = motion_command.body_indexes

    robot_anchor_pos_w = robot_state.body_state[:, robot_anchor_body_index, :3]
    robot_anchor_quat_w = robot_state.body_state[:, robot_anchor_body_index, 3:7]
    robot_body_pos_w = robot_state.body_state[:, body_indexes, :3]

    num_bodies = len(motion_command.body_names)
    pos_b, _ = subtract_frame_transforms(
        robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        robot_body_pos_w,
        robot_state.body_state[:, body_indexes, 3:7],
    )

    return pos_b.reshape(robot_state.root_state.shape[0], -1)


def robot_body_ori_b(env_states: TensorState, motion_command: MotionCommandRV, robot_name: str) -> torch.Tensor:
    """Get robot body orientations in anchor body frame as rotation matrices (first 2 columns)."""
    robot_state = env_states.robots[robot_name]
    robot_anchor_body_index = motion_command.robot_anchor_body_index
    body_indexes = motion_command.body_indexes

    robot_anchor_pos_w = robot_state.body_state[:, robot_anchor_body_index, :3]
    robot_anchor_quat_w = robot_state.body_state[:, robot_anchor_body_index, 3:7]
    robot_body_quat_w = robot_state.body_state[:, body_indexes, 3:7]

    num_bodies = len(motion_command.body_names)
    _, ori_b = subtract_frame_transforms(
        robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        robot_state.body_state[:, body_indexes, :3],
        robot_body_quat_w,
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)
