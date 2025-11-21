"""RoboVerse-compatible termination computation for motion tracking."""

from __future__ import annotations

import torch

from metasim.types import TensorState
from metasim.utils.math import quat_rotate_inverse

# Import quat_rotate_inverse from isaaclab if not available in metasim
try:
    from isaaclab.utils.math import quat_rotate_inverse
except ImportError:
    pass

from .commands import MotionCommandRV


def compute_terminations(
    env_states: TensorState,
    motion_command: MotionCommandRV,
    robot_name: str,
    thresholds: dict[str, float] | None = None,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Compute termination flags from environment state.

    Args:
        env_states: Current environment states
        motion_command: Motion command instance
        robot_name: Name of the robot
        thresholds: Termination thresholds
        device: Device to use

    Returns:
        Termination tensor
    """
    if thresholds is None:
        thresholds = {}

    robot_state = env_states.robots[robot_name]
    num_envs = robot_state.root_state.shape[0]
    terminated = torch.zeros(num_envs, dtype=torch.bool, device=device)

    robot_anchor_body_index = motion_command.robot_anchor_body_index

    # Anchor position z-only termination
    if "anchor_pos_z" in thresholds:
        threshold = thresholds["anchor_pos_z"]
        robot_anchor_pos_w = robot_state.body_state[:, robot_anchor_body_index, :3]
        error_z = torch.abs(motion_command.anchor_pos_w[:, 2] - robot_anchor_pos_w[:, 2])
        terminated = terminated | (error_z > threshold)

    # Anchor orientation termination
    if "anchor_ori" in thresholds:
        threshold = thresholds["anchor_ori"]
        robot_anchor_quat_w = robot_state.body_state[:, robot_anchor_body_index, 3:7]
        # Compute projected gravity difference
        gravity_vec = torch.tensor([0.0, 0.0, -9.81], device=device).unsqueeze(0).repeat(num_envs, 1)
        motion_projected_gravity_b = quat_rotate_inverse(motion_command.anchor_quat_w, gravity_vec)
        robot_projected_gravity_b = quat_rotate_inverse(robot_anchor_quat_w, gravity_vec)
        error = torch.abs(motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2])
        terminated = terminated | (error > threshold)

    # End-effector body position z-only termination
    if "ee_body_pos_z" in thresholds:
        threshold = thresholds["ee_body_pos_z"]
        body_indexes = motion_command.body_indexes
        body_names = motion_command.body_names
        # Check specific end-effector bodies
        ee_body_names = [
            "left_ankle_roll_link",
            "right_ankle_roll_link",
            "left_wrist_yaw_link",
            "right_wrist_yaw_link",
        ]
        ee_indices = [i for i, name in enumerate(body_names) if name in ee_body_names]
        if len(ee_indices) > 0:
            ee_body_indexes = body_indexes[ee_indices]
            robot_ee_pos_w = robot_state.body_state[:, ee_body_indexes, 2]  # z positions
            motion_ee_pos_w = motion_command.body_pos_relative_w[:, ee_indices, 2]
            error_z = torch.abs(motion_ee_pos_w - robot_ee_pos_w)
            terminated = terminated | torch.any(error_z > threshold, dim=-1)

    return terminated
