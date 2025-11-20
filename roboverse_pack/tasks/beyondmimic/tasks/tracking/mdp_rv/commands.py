"""RoboVerse-compatible motion command for tracking."""

from __future__ import annotations

import math
import os
from collections.abc import Sequence

import numpy as np
import torch

from metasim.sim.base import BaseSimHandler
from metasim.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    yaw_quat,
)

# Import sample_uniform from isaaclab or implement it
try:
    from isaaclab.utils.math import sample_uniform
except ImportError:
    import torch

    def sample_uniform(lower, upper, shape, device="cpu"):
        """Sample uniform random values."""
        if isinstance(lower, (int, float)):
            lower = torch.tensor(lower, device=device)
        if isinstance(upper, (int, float)):
            upper = torch.tensor(upper, device=device)
        return torch.rand(shape, device=device) * (upper - lower) + lower


class MotionLoader:
    """Loads and provides access to motion capture data for robot tracking."""

    def __init__(self, motion_file: str, body_indexes: Sequence[int], device: str = "cpu"):
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        data = np.load(motion_file)
        self.fps = data["fps"]
        self.joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
        self.joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
        self._body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
        self._body_quat_w = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)
        self._body_lin_vel_w = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
        self._body_ang_vel_w = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)
        self._body_indexes = body_indexes
        self.time_step_total = self.joint_pos.shape[0]

    @property
    def body_pos_w(self) -> torch.Tensor:
        """Body positions in world frame for tracked bodies."""
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        """Body orientation quaternions in world frame for tracked bodies."""
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        """Body linear velocities in world frame for tracked bodies."""
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        """Body angular velocities in world frame for tracked bodies."""
        return self._body_ang_vel_w[:, self._body_indexes]


class MotionCommandRV:
    """RoboVerse-compatible command for motion tracking."""

    def __init__(
        self,
        motion_file: str,
        robot_name: str,
        handler: BaseSimHandler,
        num_envs: int,
        anchor_body_name: str,
        body_names: list[str],
        pose_range: dict[str, tuple[float, float]],
        velocity_range: dict[str, tuple[float, float]],
        joint_position_range: tuple[float, float],
        resampling_time_range: tuple[float, float],
        adaptive_kernel_size: int = 1,
        adaptive_lambda: float = 0.8,
        adaptive_uniform_ratio: float = 0.1,
        adaptive_alpha: float = 0.001,
        device: str | torch.device = "cpu",
    ):
        """Initialize motion command.

        Args:
            motion_file: Path to motion file (.npz)
            robot_name: Name of the robot in the handler
            handler: RoboVerse simulation handler
            num_envs: Number of environments
            anchor_body_name: Name of anchor body
            body_names: List of body names to track
            pose_range: Pose randomization ranges
            velocity_range: Velocity randomization ranges
            joint_position_range: Joint position randomization range
            resampling_time_range: Time range for resampling
            adaptive_kernel_size: Size of adaptive sampling kernel
            adaptive_lambda: Lambda for adaptive sampling
            adaptive_uniform_ratio: Uniform ratio for adaptive sampling
            adaptive_alpha: Alpha for adaptive sampling
            device: Device to use
        """
        self.handler = handler
        self.robot_name = robot_name
        self.num_envs = num_envs
        self.device = device
        self.anchor_body_name = anchor_body_name
        self.body_names = body_names

        # Get robot body and joint information
        robot = handler.robots[0]
        self.robot_body_names = robot.body_names
        self.robot_joint_names = handler.get_joint_names(robot_name)

        # Find body indices
        self.robot_anchor_body_index = self.robot_body_names.index(anchor_body_name)
        self.motion_anchor_body_index = body_names.index(anchor_body_name)
        self.body_indexes = torch.tensor(
            [self.robot_body_names.index(name) for name in body_names], dtype=torch.long, device=device
        )

        # Load motion data
        self.motion = MotionLoader(motion_file, self.body_indexes.cpu().tolist(), device=device)

        # Initialize time steps and relative poses
        self.time_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.body_pos_relative_w = torch.zeros(num_envs, len(body_names), 3, device=device)
        self.body_quat_relative_w = torch.zeros(num_envs, len(body_names), 4, device=device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        # Adaptive sampling setup
        # Estimate bin count based on episode length and simulation timestep
        dt = handler.sim_params.dt if hasattr(handler, "sim_params") else 0.005
        decimation = handler.scenario.decimation if hasattr(handler, "scenario") else 25
        episode_length_steps = int(10.0 / (dt * decimation))  # Assume 10s episode
        self.bin_count = int(self.motion.time_step_total // episode_length_steps) + 1
        self.bin_failed_count = torch.zeros(self.bin_count, dtype=torch.float, device=device)
        self._current_bin_failed = torch.zeros(self.bin_count, dtype=torch.float, device=device)
        self.kernel = torch.tensor([adaptive_lambda**i for i in range(adaptive_kernel_size)], device=device)
        self.kernel = self.kernel / self.kernel.sum()

        # Configuration
        self.pose_range = pose_range
        self.velocity_range = velocity_range
        self.joint_position_range = joint_position_range
        self.resampling_time_range = resampling_time_range
        self.adaptive_kernel_size = adaptive_kernel_size
        self.adaptive_lambda = adaptive_lambda
        self.adaptive_uniform_ratio = adaptive_uniform_ratio
        self.adaptive_alpha = adaptive_alpha

        # Metrics
        self.metrics = {
            "error_anchor_pos": torch.zeros(num_envs, device=device),
            "error_anchor_rot": torch.zeros(num_envs, device=device),
            "error_anchor_lin_vel": torch.zeros(num_envs, device=device),
            "error_anchor_ang_vel": torch.zeros(num_envs, device=device),
            "error_body_pos": torch.zeros(num_envs, device=device),
            "error_body_rot": torch.zeros(num_envs, device=device),
            "error_joint_pos": torch.zeros(num_envs, device=device),
            "error_joint_vel": torch.zeros(num_envs, device=device),
            "sampling_entropy": torch.zeros(num_envs, device=device),
            "sampling_top1_prob": torch.zeros(num_envs, device=device),
            "sampling_top1_bin": torch.zeros(num_envs, device=device),
        }

        # Initialize with first frame
        self._resample_command(list(range(num_envs)))

    def update(self):
        """Update motion command (called each step)."""
        self.time_steps += 1
        env_ids = torch.where(self.time_steps >= self.motion.time_step_total)[0]
        if len(env_ids) > 0:
            self._resample_command(env_ids.tolist())

        # Update relative poses
        self._update_relative_poses()

        # Update metrics
        self._update_metrics()

        # Update adaptive sampling
        self.bin_failed_count = (
            self.adaptive_alpha * self._current_bin_failed + (1 - self.adaptive_alpha) * self.bin_failed_count
        )
        self._current_bin_failed.zero_()

    def reset(self, env_ids: list[int]):
        """Reset motion command for specified environments."""
        if len(env_ids) == 0:
            return
        self._resample_command(env_ids)

    def _update_relative_poses(self):
        """Update relative body poses based on anchor."""
        states = self.handler.get_states()
        robot_state = states.robots[self.robot_name]

        # Get robot anchor pose
        robot_anchor_pos_w = robot_state.body_state[:, self.robot_anchor_body_index, :3]
        robot_anchor_quat_w = robot_state.body_state[:, self.robot_anchor_body_index, 3:7]

        # Get motion anchor pose
        anchor_pos_w = self.anchor_pos_w
        anchor_quat_w = self.anchor_quat_w

        # Compute relative transformation
        anchor_pos_w_repeat = anchor_pos_w[:, None, :].repeat(1, len(self.body_names), 1)
        anchor_quat_w_repeat = anchor_quat_w[:, None, :].repeat(1, len(self.body_names), 1)
        robot_anchor_pos_w_repeat = robot_anchor_pos_w[:, None, :].repeat(1, len(self.body_names), 1)
        robot_anchor_quat_w_repeat = robot_anchor_quat_w[:, None, :].repeat(1, len(self.body_names), 1)

        delta_pos_w = robot_anchor_pos_w_repeat.clone()
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)

    def _update_metrics(self):
        """Update tracking metrics."""
        states = self.handler.get_states()
        robot_state = states.robots[self.robot_name]

        # Anchor metrics
        robot_anchor_pos_w = robot_state.body_state[:, self.robot_anchor_body_index, :3]
        robot_anchor_quat_w = robot_state.body_state[:, self.robot_anchor_body_index, 3:7]
        robot_anchor_lin_vel_w = robot_state.body_state[:, self.robot_anchor_body_index, 7:10]
        robot_anchor_ang_vel_w = robot_state.body_state[:, self.robot_anchor_body_index, 10:13]

        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, robot_anchor_quat_w)
        self.metrics["error_anchor_lin_vel"] = torch.norm(self.anchor_lin_vel_w - robot_anchor_lin_vel_w, dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(self.anchor_ang_vel_w - robot_anchor_ang_vel_w, dim=-1)

        # Body metrics
        robot_body_pos_w = robot_state.body_state[:, self.body_indexes, :3]
        robot_body_quat_w = robot_state.body_state[:, self.body_indexes, 3:7]

        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - robot_body_pos_w, dim=-1).mean(dim=-1)
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, robot_body_quat_w).mean(dim=-1)

        # Joint metrics
        robot_joint_pos = robot_state.joint_pos
        robot_joint_vel = robot_state.joint_vel

        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - robot_joint_vel, dim=-1)

    def _adaptive_sampling(self, env_ids: list[int]):
        """Perform adaptive sampling for motion resampling."""
        # For now, use uniform sampling (adaptive sampling requires termination info)
        # This can be enhanced later if needed
        sampled_bins = torch.randint(0, self.bin_count, (len(env_ids),), device=self.device)
        self.time_steps[env_ids] = (
            (sampled_bins.float() + torch.rand(len(env_ids), device=self.device))
            / self.bin_count
            * (self.motion.time_step_total - 1)
        ).long()

        # Metrics
        H = math.log(self.bin_count)
        self.metrics["sampling_entropy"][env_ids] = 1.0  # Normalized entropy
        self.metrics["sampling_top1_prob"][env_ids] = 1.0 / self.bin_count
        self.metrics["sampling_top1_bin"][env_ids] = sampled_bins.float() / self.bin_count

    def _resample_command(self, env_ids: list[int]):
        """Resample motion command for specified environments."""
        if len(env_ids) == 0:
            return

        self._adaptive_sampling(env_ids)

        # Get root pose from motion data
        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()

        # Apply pose randomization
        range_list = [self.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos[env_ids] += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])

        # Apply velocity randomization
        range_list = [self.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel[env_ids] += rand_samples[:, :3]
        root_ang_vel[env_ids] += rand_samples[:, 3:]

        # Apply joint position randomization
        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()
        joint_pos += sample_uniform(*self.joint_position_range, joint_pos.shape, joint_pos.device)

        # Set robot state
        states = self.handler.get_states()
        robot_state = states.robots[self.robot_name]

        # Set joint positions and velocities
        robot_state.joint_pos[env_ids] = joint_pos[env_ids]
        robot_state.joint_vel[env_ids] = joint_vel[env_ids]

        # Set root state
        root_state = torch.cat(
            [root_pos[env_ids], root_ori[env_ids], root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1
        )
        robot_state.root_state[env_ids, :13] = root_state

        # Update handler states
        self.handler.set_states(states=states, env_ids=env_ids)

    # Properties for accessing motion data
    @property
    def joint_pos(self) -> torch.Tensor:
        """Target joint positions from motion data at current time step."""
        return self.motion.joint_pos[self.time_steps]

    @property
    def joint_vel(self) -> torch.Tensor:
        """Target joint velocities from motion data at current time step."""
        return self.motion.joint_vel[self.time_steps]

    @property
    def body_pos_w(self) -> torch.Tensor:
        """Target body positions in world frame from motion data."""
        return self.motion.body_pos_w[self.time_steps]

    @property
    def body_quat_w(self) -> torch.Tensor:
        """Target body orientations in world frame from motion data."""
        return self.motion.body_quat_w[self.time_steps]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        """Target body linear velocities in world frame from motion data."""
        return self.motion.body_lin_vel_w[self.time_steps]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        """Target body angular velocities in world frame from motion data."""
        return self.motion.body_ang_vel_w[self.time_steps]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        """Target anchor body position in world frame from motion data."""
        return self.motion.body_pos_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        """Target anchor body orientation in world frame from motion data."""
        return self.motion.body_quat_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        """Target anchor body linear velocity in world frame from motion data."""
        return self.motion.body_lin_vel_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        """Target anchor body angular velocity in world frame from motion data."""
        return self.motion.body_ang_vel_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def command(self) -> torch.Tensor:
        """Command observation containing joint positions and velocities."""
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def num_actions(self) -> int:
        """Number of actions (joints)."""
        return len(self.robot_joint_names)
