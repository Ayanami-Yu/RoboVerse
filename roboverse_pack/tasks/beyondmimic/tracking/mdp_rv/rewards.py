"""RoboVerse-compatible reward computation for motion tracking."""

from __future__ import annotations

import torch

from metasim.types import TensorState
from metasim.utils.math import quat_error_magnitude

from .commands import MotionCommandRV


def compute_rewards(
    env_states: TensorState,
    motion_command: MotionCommandRV,
    robot_name: str,
    last_actions: torch.Tensor | None,
    reward_weights: dict[str, float] | None = None,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Compute rewards from environment state.

    Args:
        env_states: Current environment states
        motion_command: Motion command instance
        robot_name: Name of the robot
        last_actions: Last actions taken
        reward_weights: Weights for each reward term
        device: Device to use

    Returns:
        Reward tensor
    """
    if reward_weights is None:
        reward_weights = {}

    robot_state = env_states.robots[robot_name]
    num_envs = robot_state.root_state.shape[0]
    total_reward = torch.zeros(num_envs, dtype=torch.float32, device=device)

    # Motion global anchor position error (exponential)
    if "motion_global_anchor_pos" in reward_weights:
        weight = reward_weights["motion_global_anchor_pos"]
        std = 0.3
        robot_anchor_body_index = motion_command.robot_anchor_body_index
        robot_anchor_pos_w = robot_state.body_state[:, robot_anchor_body_index, :3]
        error = torch.sum(torch.square(motion_command.anchor_pos_w - robot_anchor_pos_w), dim=-1)
        reward = torch.exp(-error / std**2)
        total_reward += weight * reward

    # Motion global anchor orientation error (exponential)
    if "motion_global_anchor_ori" in reward_weights:
        weight = reward_weights["motion_global_anchor_ori"]
        std = 0.4
        robot_anchor_body_index = motion_command.robot_anchor_body_index
        robot_anchor_quat_w = robot_state.body_state[:, robot_anchor_body_index, 3:7]
        error = quat_error_magnitude(motion_command.anchor_quat_w, robot_anchor_quat_w) ** 2
        reward = torch.exp(-error / std**2)
        total_reward += weight * reward

    # Motion relative body position error (exponential)
    if "motion_body_pos" in reward_weights:
        weight = reward_weights["motion_body_pos"]
        std = 0.3
        body_indexes = motion_command.body_indexes
        robot_body_pos_w = robot_state.body_state[:, body_indexes, :3]
        error = torch.sum(torch.square(motion_command.body_pos_relative_w - robot_body_pos_w), dim=-1)
        reward = torch.exp(-error.mean(-1) / std**2)
        total_reward += weight * reward

    # Motion relative body orientation error (exponential)
    if "motion_body_ori" in reward_weights:
        weight = reward_weights["motion_body_ori"]
        std = 0.4
        body_indexes = motion_command.body_indexes
        robot_body_quat_w = robot_state.body_state[:, body_indexes, 3:7]
        error = quat_error_magnitude(motion_command.body_quat_relative_w, robot_body_quat_w) ** 2
        reward = torch.exp(-error.mean(-1) / std**2)
        total_reward += weight * reward

    # Motion global body linear velocity error (exponential)
    if "motion_body_lin_vel" in reward_weights:
        weight = reward_weights["motion_body_lin_vel"]
        std = 1.0
        body_indexes = motion_command.body_indexes
        robot_body_lin_vel_w = robot_state.body_state[:, body_indexes, 7:10]
        error = torch.sum(torch.square(motion_command.body_lin_vel_w - robot_body_lin_vel_w), dim=-1)
        reward = torch.exp(-error.mean(-1) / std**2)
        total_reward += weight * reward

    # Motion global body angular velocity error (exponential)
    if "motion_body_ang_vel" in reward_weights:
        weight = reward_weights["motion_body_ang_vel"]
        std = 3.14
        body_indexes = motion_command.body_indexes
        robot_body_ang_vel_w = robot_state.body_state[:, body_indexes, 10:13]
        error = torch.sum(torch.square(motion_command.body_ang_vel_w - robot_body_ang_vel_w), dim=-1)
        reward = torch.exp(-error.mean(-1) / std**2)
        total_reward += weight * reward

    # Action rate L2 penalty
    if "action_rate_l2" in reward_weights and last_actions is not None:
        weight = reward_weights["action_rate_l2"]
        # Compute action rate (difference from previous)
        # For simplicity, use L2 norm of actions
        action_rate = torch.norm(last_actions, dim=-1)
        total_reward += weight * action_rate

    # Joint limit penalty
    if "joint_limit" in reward_weights:
        weight = reward_weights["joint_limit"]
        # Get joint limits from handler
        # This is a simplified version - actual implementation would check against limits
        joint_pos = robot_state.joint_pos
        # Penalty for being near limits (simplified)
        joint_limit_penalty = torch.zeros(num_envs, device=device)
        total_reward += weight * joint_limit_penalty

    # Undesired contacts penalty
    if "undesired_contacts" in reward_weights:
        weight = reward_weights["undesired_contacts"]
        # Contact information would need to be queried from handler
        # For now, set to zero
        contact_penalty = torch.zeros(num_envs, device=device)
        total_reward += weight * contact_penalty

    return total_reward
