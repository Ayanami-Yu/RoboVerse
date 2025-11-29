import os
import math
from typing import Sequence, TYPE_CHECKING
import numpy as np
import torch
from dataclasses import MISSING
from metasim.types import TensorState
from metasim.utils import configclass
from metasim.utils.math import (
    quat_apply,
    sample_uniform,
    quat_mul,
    yaw_quat,
    quat_from_euler_xyz,
)
from roboverse_learn.rl.beyondmimic.helper.math import quat_inv, quat_error_magnitude
from roboverse_learn.rl.beyondmimic.helper.string_utils import find_bodies
if TYPE_CHECKING:
    from roboverse_pack.tasks.beyondmimic.tracking.tracking_g1 import TrackingG1Task


# adapted from BeyondMimic commands.py

class MotionLoader:
    """Load motion data from a file and provide access to target joint states and body positions and orientations in world frame."""
    def __init__(self, motion_file: str, body_indexes: Sequence[int], device: str = "cpu"):
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        data = np.load(motion_file)
        self.fps = data["fps"]

        # target joint states from motion file
        self.joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)  # [n_timesteps, n_dofs] -> [6574, 29]
        self.joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)  # [n_timesteps, n_dofs]
        # target body positions and orientations in world frame
        self._body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)  # [n_timesteps, n_bodies, 3] -> [6574, 30, 3]
        self._body_quat_w = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)  # [n_timesteps, n_bodies, 4]

        # target body velocities
        self._body_lin_vel_w = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)  # [n_timesteps, n_bodies, 3]
        self._body_ang_vel_w = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)  # [n_timesteps, n_bodies, 3]
        self._body_indexes = body_indexes  # [n_indexes] -> [14]
        self.time_step_total = self.joint_pos.shape[0]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]


@configclass
class MotionCommandCfg:
    motion_file: str = MISSING
    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING  # a subset of body link names to track

    resampling_time_range: tuple[float, float] = MISSING
    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}
    joint_position_range: tuple[float, float] = (-0.52, 0.52)

    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001

    # TODO add `debug_vis` from BeyondMimic?


class MotionCommand:
    def __init__(self, env: "TrackingG1Task", cfg: MotionCommandCfg):
        self.env = env
        self.cfg = cfg
        self.device = env.device

        # time left before resampling
        self.time_left = torch.zeros(env.num_envs, device=env.device)

        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        self.body_indexes = torch.tensor(find_bodies(self.cfg.body_names, env.sorted_body_names, preserve_order=True)[0])

        self.motion = MotionLoader(cfg.motion_file, self.body_indexes, env.device)
        self.time_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.body_pos_relative_w = torch.zeros(env.num_envs, len(self.cfg.body_names), 3, device=env.device)
        self.body_quat_relative_w = torch.zeros(env.num_envs, len(self.cfg.body_names), 4, device=env.device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        self.bin_count = int(self.motion.time_step_total // (1 / env.step_dt)) + 1
        self.bin_failed_count = torch.zeros(self.bin_count, dtype=torch.float, device=env.device)
        self._current_bin_failed = torch.zeros(self.bin_count, dtype=torch.float, device=env.device)
        self.kernel = torch.tensor(
            [self.cfg.adaptive_lambda ** i for i in range(self.cfg.adaptive_kernel_size)], device=env.device
        )
        self.kernel = self.kernel / self.kernel.sum()

        # metrics used for logging
        metrics = ["error_anchor_pos", "error_anchor_rot", "error_anchor_lin_vel", "error_anchor_ang_vel", "error_body_pos", "error_body_rot", "error_joint_pos", "error_joint_vel", "sampling_entropy", "sampling_top1_prob", "sampling_top1_bin"]
        self.metrics = {k: torch.zeros(env.num_envs, device=env.device) for k in metrics}

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        episode_failed = self.env.terminated_buf[env_ids]
        if torch.any(episode_failed):
            current_bin_index = torch.clamp(
                (self.time_steps * self.bin_count) // max(self.motion.time_step_total, 1), 0, self.bin_count - 1
            )
            fail_bins = current_bin_index[env_ids][episode_failed]
            self._current_bin_failed[:] = torch.bincount(fail_bins, minlength=self.bin_count)

        # sample
        sampling_probabilities = self.bin_failed_count + self.cfg.adaptive_uniform_ratio / float(self.bin_count)
        sampling_probabilities = torch.nn.functional.pad(
            sampling_probabilities.unsqueeze(0).unsqueeze(0),
            (0, self.cfg.adaptive_kernel_size - 1),  # Non-causal kernel
            mode="replicate",
        )
        sampling_probabilities = torch.nn.functional.conv1d(sampling_probabilities, self.kernel.view(1, 1, -1)).view(-1)

        sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()

        sampled_bins = torch.multinomial(sampling_probabilities, len(env_ids), replacement=True)

        self.time_steps[env_ids] = (
            (sampled_bins + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device))
            / self.bin_count
            * (self.motion.time_step_total - 1)
        ).long()

        # metrics
        H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
        H_norm = H / math.log(self.bin_count)
        pmax, imax = sampling_probabilities.max(dim=0)
        self.metrics["sampling_entropy"][:] = H_norm
        self.metrics["sampling_top1_prob"][:] = pmax
        self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count

    def _resample_command(self, env_ids: Sequence[int], env_states: TensorState):
        if len(env_ids) == 0:
            return
        self._adaptive_sampling(env_ids)

        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()

        # perturb root position and orientation
        range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos[env_ids] += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])

        # perturb root velocities
        range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel[env_ids] += rand_samples[:, :3]
        root_ang_vel[env_ids] += rand_samples[:, 3:]

        # perturb joint positions and velocities then clamp to limits
        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()
        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)
        soft_joint_pos_limits = self.env.soft_dof_pos_limits[env_ids]
        joint_pos[env_ids] = torch.clip(
            joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
        )
        env_states.robots[self.env.name].joint_pos[env_ids] = joint_pos[env_ids]
        env_states.robots[self.env.name].joint_vel[env_ids] = joint_vel[env_ids]
        env_states.robots[self.env.name].root_state[env_ids, :] = torch.cat([root_pos[env_ids], root_ori[env_ids], root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1)
        self.env.handler.set_states(env_states, env_ids)  # TODO test if this is correct

    def _update_command(self, env_states: TensorState):  # TODO change corresponding entries
        # pick new time steps using adaptive sampling for the envs that have reached the end of the motion
        self.time_steps += 1
        env_ids = torch.where(self.time_steps >= self.motion.time_step_total)[0]
        self._resample_command(env_ids, env_states)  # resample when motion ends

        # TODO clear metric values?

        robot_state = env_states.robots[self.env.name]

        # put reference motion at the robot's XY (rotate it so its heading matches the robot's heading) while keeping the motion's height (Z)
        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)  # [n_envs, n_bodies, 3], n_bodies = 14
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)  # [n_envs, n_bodies, 4]
        robot_anchor_pos_w_repeat = robot_state.root_state[:, 0:3][:, None, :].repeat(1, len(self.cfg.body_names), 1)  # [n_envs, n_bodies, 3]
        robot_anchor_quat_w_repeat = robot_state.root_state[:, 3:7][:, None, :].repeat(1, len(self.cfg.body_names), 1)  # [n_envs, n_bodies, 4]

        # let XY come from the robot anchor, and Z comes from the motion anchor
        # avoid penalizing global XY drift while preserving vertical posture from the motion
        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]  # overwrite Z

        # first compute relative rotation between robot and motion anchors, then keep only the yaw component
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))
        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)

        self.bin_failed_count = (
            self.cfg.adaptive_alpha * self._current_bin_failed + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
        )
        self._current_bin_failed.zero_()

    def _update_metrics(self, env_states: TensorState):
        """Update metrics for logging."""
        robot_state = env_states.robots[self.env.name]
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - robot_state.root_state[:, :3], dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, robot_state.root_state[:, 3:7])
        self.metrics["error_anchor_lin_vel"] = torch.norm(self.anchor_lin_vel_w - robot_state.root_state[:, 7:10], dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(self.anchor_ang_vel_w - robot_state.root_state[:, 10:13], dim=-1)

        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - robot_state.body_state[:, self.body_indexes, :3], dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, robot_state.body_state[:, self.body_indexes, 3:7]).mean(
            dim=-1
        )

        self.metrics["error_body_lin_vel"] = torch.norm(self.body_lin_vel_w - robot_state.body_state[:, self.body_indexes, 7:10], dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_ang_vel"] = torch.norm(self.body_ang_vel_w - robot_state.body_state[:, self.body_indexes, 10:13], dim=-1).mean(
            dim=-1
        )

        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - robot_state.joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - robot_state.joint_vel, dim=-1)

    def reset(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = torch.arange(self.env.num_envs, dtype=torch.int64, device=self.device)
        if len(env_ids) != 0:
            # resample the time left before resampling (will be used by `compute()`)
            self.time_left[env_ids] = self.time_left[env_ids].uniform_(*self.cfg.resampling_time_range)
            self._resample_command(env_ids, self.env.handler.get_states())  # TODO is using `get_states()` here correct?

    def compute(self, env_states: TensorState):
        """Compute the command."""
        # update the metrics based on current state
        self._update_metrics(env_states)
        # reduce the time left before resampling by the timestep passed since the last call
        self.time_left -= self.env.step_dt
        # resample the command if necessary
        resample_env_ids = (self.time_left <= 0.0).nonzero().flatten()
        if len(resample_env_ids) > 0:
            self._resample(resample_env_ids)
        # update the command
        self._update_command(env_states)

    @property
    def command(self) -> torch.Tensor:
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def joint_pos(self) -> torch.Tensor:
        return self.motion.joint_pos[self.time_steps]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.motion.joint_vel[self.time_steps]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.time_steps] + self.env.handler.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.time_steps, self.motion_anchor_body_index] + self.env.handler.scene.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps, self.motion_anchor_body_index]
