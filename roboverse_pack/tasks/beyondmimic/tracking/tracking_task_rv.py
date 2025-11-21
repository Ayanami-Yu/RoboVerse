"""RoboVerse-compatible task implementation for BeyondMimic motion tracking."""

from __future__ import annotations

import os
from collections import deque
from copy import deepcopy
from dataclasses import InitVar, dataclass

import torch
from gymnasium import spaces

from metasim.scenario.scenario import ScenarioCfg
from metasim.sim.base import BaseSimHandler
from metasim.task.rl_task import RLTaskEnv
from metasim.types import Action, Reward, TensorState
from metasim.utils.hf_util import check_and_download_single
from metasim.utils.state import list_state_to_tensor
from roboverse_learn.rl.beyondmimic.helper.utils import pattern_match
from roboverse_pack.tasks.beyondmimic.tracking.mdp_rv.commands import MotionCommandRV
from roboverse_pack.tasks.beyondmimic.tracking.mdp_rv.observations import (
    compute_observations,
    compute_privileged_observations,
)
from roboverse_pack.tasks.beyondmimic.tracking.mdp_rv.rewards import compute_rewards
from roboverse_pack.tasks.beyondmimic.tracking.mdp_rv.terminations import compute_terminations

# Import sample_uniform
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


@dataclass
class TrackingTaskCfg:
    """Configuration for the tracking task."""

    # Motion command settings
    motion_file: str = ""
    anchor_body_name: str = "torso_link"
    body_names: InitVar[list[str] | None] = None
    pose_range: dict[str, tuple[float, float]] | None = None
    velocity_range: dict[str, tuple[float, float]] | None = None
    joint_position_range: tuple[float, float] = (-0.1, 0.1)
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)

    # Adaptive sampling settings
    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001

    # Episode settings
    episode_length_s: float = 10.0
    max_episode_steps: int = 2000

    # Observation settings
    enable_corruption: bool = True
    observation_noise: dict[str, tuple[float, float]] | None = None

    # Reward weights
    reward_weights: dict[str, float] | None = None

    # Termination thresholds
    termination_thresholds: dict[str, float] | None = None

    class InitialStates:
        """Initial states using BeyondMimic's configs."""

        objects = {}
        robots = {
            "g1_dof29": {
                "pos": [0.0, 0.0, 0.76],
                "rot": [1.0, 0.0, 0.0, 0.0],
                "joint_pos": {
                    ".*_hip_pitch_joint": -0.312,
                    ".*_knee_joint": 0.669,
                    ".*_ankle_pitch_joint": -0.363,
                    ".*_elbow_joint": 0.6,
                    "left_shoulder_roll_joint": 0.2,
                    "left_shoulder_pitch_joint": 0.2,
                    "right_shoulder_roll_joint": -0.2,
                    "right_shoulder_pitch_joint": 0.2,
                    "left_wrist_roll_joint": 0.15,
                    "right_wrist_roll_joint": -0.15,
                },
                "joint_vel": {".*": 0.0},
            },
        }

    initial_states = InitialStates()

    def __post_init__(self, body_names: list[str] | None):
        """Set default values."""
        if body_names:
            self.body_names = body_names
        if self.pose_range is None:
            self.pose_range = {
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (-0.01, 0.01),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-0.2, 0.2),
            }
        if self.velocity_range is None:
            self.velocity_range = {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.2, 0.2),
                "roll": (-0.52, 0.52),
                "pitch": (-0.52, 0.52),
                "yaw": (-0.78, 0.78),
            }
        if self.observation_noise is None:
            self.observation_noise = {
                "motion_anchor_pos_b": (-0.25, 0.25),
                "motion_anchor_ori_b": (-0.05, 0.05),
                "base_lin_vel": (-0.5, 0.5),
                "base_ang_vel": (-0.2, 0.2),
                "joint_pos": (-0.01, 0.01),
                "joint_vel": (-0.5, 0.5),
            }
        if self.reward_weights is None:
            self.reward_weights = {
                "motion_global_anchor_pos": 0.5,
                "motion_global_anchor_ori": 0.5,
                "motion_body_pos": 1.0,
                "motion_body_ori": 1.0,
                "motion_body_lin_vel": 1.0,
                "motion_body_ang_vel": 1.0,
                "action_rate_l2": -1e-1,
                "joint_limit": -10.0,
                "undesired_contacts": -0.1,
            }
        if self.termination_thresholds is None:
            self.termination_thresholds = {
                "anchor_pos_z": 0.25,
                "anchor_ori": 0.8,
                "ee_body_pos_z": 0.25,
            }


class TrackingTaskRV(RLTaskEnv):
    """RoboVerse-compatible task for motion tracking."""

    max_episode_steps = 2000

    def __init__(
        self,
        scenario: ScenarioCfg,
        task_cfg: TrackingTaskCfg | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        """Initialize the tracking task.

        Args:
            scenario: RoboVerse scenario configuration
            task_cfg: Task-specific configuration
            device: Device to run on
        """
        assert len(scenario.robots) > 0, "At least one robot must be specified"
        self.robot = scenario.robots[0]  # TODO handle multiple robots
        self.robot_name = scenario.robots[0].name

        self.task_cfg = task_cfg
        self.max_episode_steps = task_cfg.max_episode_steps

        if device is None:
            device = "cpu" if scenario.simulator == "mujoco" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)

        self.scenario = scenario
        self.num_envs = self.scenario.num_envs

        # Initialize simulation handler
        if isinstance(self.scenario, BaseSimHandler):
            self.handler = self.scenario
        else:
            self._instantiate_env(self.scenario)
        if self.traj_filepath is not None:
            check_and_download_single(self.traj_filepath)

        self._prepare_callbacks()
        self._episode_steps = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        # Get initial states
        self._initial_states = list_state_to_tensor(self.handler, self._get_initial_states(), self.device)

        self._observation_space: spaces.Space | None = None
        self._action_space: spaces.Space | None = None

        self.asymmetric_obs = False

        # Initialize motion command after handler is available
        self._initialize_motion_command()

        # Observation and action buffers
        self.obs_buf_queue: deque[torch.Tensor] = deque(maxlen=1)
        self.priv_obs_buf_queue: deque[torch.Tensor] = deque(maxlen=1)
        self.last_actions: torch.Tensor | None = None

        # TODO no need to call `self._bind_callbacks()` here?

        self.reset(env_ids=list(range(self.num_envs)))

        # Compute observation dimensions
        states = self.handler.get_states()
        first_obs = self._observation(states)
        self.num_obs = first_obs.shape[-1]
        self.num_priv_obs = self._privileged_observation(states).shape[-1]

        # Action bounds from joint limits (ordered by joint_names)
        limits = self.robot.joint_limits
        self.joint_names = self.handler.get_joint_names(self.robot.name)
        self._action_low = torch.tensor(
            [limits[j][0] for j in self.joint_names], dtype=torch.float32, device=self.device
        )
        self._action_high = torch.tensor(
            [limits[j][1] for j in self.joint_names], dtype=torch.float32, device=self.device
        )
        self.num_actions = self._action_low.shape[0]  # TODO check equivalence with `len(self.robot.actuators)`

    def _initialize_motion_command(self):
        """Initialize the motion command system."""
        if not os.path.isfile(self.task_cfg.motion_file):
            raise FileNotFoundError(f"Motion file not found: {self.task_cfg.motion_file}")

        self.motion_command = MotionCommandRV(
            motion_file=self.task_cfg.motion_file,
            robot_name=self.robot_name,
            handler=self.handler,
            num_envs=self.num_envs,
            anchor_body_name=self.task_cfg.anchor_body_name,
            body_names=self.task_cfg.body_names,
            pose_range=self.task_cfg.pose_range,
            velocity_range=self.task_cfg.velocity_range,
            joint_position_range=self.task_cfg.joint_position_range,
            resampling_time_range=self.task_cfg.resampling_time_range,
            adaptive_kernel_size=self.task_cfg.adaptive_kernel_size,
            adaptive_lambda=self.task_cfg.adaptive_lambda,
            adaptive_uniform_ratio=self.task_cfg.adaptive_uniform_ratio,
            adaptive_alpha=self.task_cfg.adaptive_alpha,
            device=self.device,
        )

    def _get_initial_states(self) -> list[dict]:
        """Return per-env initial states."""
        robot_state = self.task_cfg.initial_states.robots[self.robot.name]

        sorted_joint_names = self.handler.get_joint_names(self.robot.name, sort=True)
        joint_pos = pattern_match(robot_state["joint_pos"], sorted_joint_names)
        joint_vel = pattern_match(robot_state["joint_vel"], sorted_joint_names)

        template = {
            "objects": {},
            "robots": {
                self.robot.name: {
                    "pos": torch.tensor(robot_state["pos"], dtype=torch.float32),
                    "rot": torch.tensor(robot_state["rot"], dtype=torch.float32),
                    "dof_pos": {name: joint_pos[name] for name in joint_pos},
                    "dof_vel": {name: joint_vel[name] for name in joint_vel},
                }
            },
        }
        return [deepcopy(template) for _ in range(self.scenario.num_envs)]

    def _observation(self, env_states: TensorState) -> torch.Tensor:
        """Compute observations from environment state."""
        obs = compute_observations(
            env_states=env_states,
            motion_command=self.motion_command,
            robot_name=self.robot_name,
            last_actions=self.last_actions,
            enable_corruption=self.task_cfg.enable_corruption,
            noise_params=self.task_cfg.observation_noise,
            device=self.device,
        )
        return obs

    def _privileged_observation(self, env_states: TensorState) -> torch.Tensor:
        """Compute privileged observations from environment state."""
        priv_obs = compute_privileged_observations(
            env_states=env_states,
            motion_command=self.motion_command,
            robot_name=self.robot_name,
            last_actions=self.last_actions,
            device=self.device,
        )
        return priv_obs

    def _reward(self, env_states: TensorState) -> Reward:
        """Compute rewards from environment state."""
        reward = compute_rewards(
            env_states=env_states,
            motion_command=self.motion_command,
            robot_name=self.robot_name,
            last_actions=self.last_actions,
            reward_weights=self.task_cfg.reward_weights,
            device=self.device,
        )
        return reward

    def _terminated(self, env_states: TensorState) -> torch.Tensor:
        """Compute termination flags from environment state."""
        terminated = compute_terminations(
            env_states=env_states,
            motion_command=self.motion_command,
            robot_name=self.robot_name,
            thresholds=self.task_cfg.termination_thresholds,
            device=self.device,
        )
        return terminated

    def _time_out(self, env_states: TensorState | None) -> torch.Tensor:
        """Compute timeout flags."""
        return self._episode_steps >= self.max_episode_steps

    def step(self, actions: Action) -> tuple[torch.Tensor, Reward, torch.Tensor, torch.Tensor, dict]:
        """Step the environment."""
        # Store actions for observation computation
        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        if actions.ndim == 1:
            actions = actions.unsqueeze(0)

        # Normalize actions from [-1, 1] to joint limits
        real_actions = self.unnormalise_action(actions)
        self.last_actions = real_actions

        # Update motion command
        self.motion_command.update()

        # Step physics
        self.handler.set_dof_targets(real_actions)
        self.handler.simulate()

        # Get new state
        states = self.handler.get_states()

        # Compute observations, rewards, terminations
        obs = self._observation(states).to(self.device)
        priv_obs = self._privileged_observation(states).to(self.device)
        reward = self._reward(states).to(self.device)
        terminated = self._terminated(states).bool().to(self.device)
        time_out = self._time_out(states).bool().to(self.device)

        # Update observation buffers
        self.obs_buf_queue.append(obs)
        self.priv_obs_buf_queue.append(priv_obs)

        # Handle resets
        episode_done = terminated | time_out
        done_indices = episode_done.nonzero(as_tuple=False).squeeze(-1)
        if done_indices.numel():
            self.reset(env_ids=done_indices.tolist())
            states_after = self.handler.get_states()
            obs_after = self._observation(states_after).to(self.device)
            obs[done_indices] = obs_after[done_indices]

        info = {
            "privileged_observation": priv_obs,
            "episode_steps": self._episode_steps.clone(),
        }

        return obs, reward, terminated, time_out, info

    def reset(self, states=None, env_ids=None) -> tuple[torch.Tensor, dict]:
        """Reset selected environments."""
        if env_ids is None:
            env_ids = list(range(self.num_envs))

        self._episode_steps[env_ids] = 0

        # Reset motion command for selected envs
        self.motion_command.reset(env_ids)

        # Reset environment states
        raw_states = self._initial_states if states is None else states
        states_to_set = self._prepare_states(raw_states, env_ids)
        self.handler.set_states(states=states_to_set, env_ids=env_ids)

        # Get initial observations
        states = self.handler.get_states()
        first_obs = self._observation(states).to(self.device)
        first_priv_obs = self._privileged_observation(states).to(self.device)

        # Initialize observation buffers
        if len(self.obs_buf_queue) == 0:
            self.obs_buf_queue.append(first_obs)
            self.priv_obs_buf_queue.append(first_priv_obs)
        else:
            self.obs_buf_queue[0] = first_obs
            self.priv_obs_buf_queue[0] = first_priv_obs

        info = {"privileged_observation": first_priv_obs}
        return first_obs, info

    @property
    def obs_buf(self) -> torch.Tensor:
        """Get current observation buffer."""
        if len(self.obs_buf_queue) == 0:
            raise RuntimeError("Observation buffer not initialized.")
        return torch.cat(list(self.obs_buf_queue), dim=1)

    @property
    def priv_obs_buf(self) -> torch.Tensor:
        """Get current privileged observation buffer."""
        if len(self.priv_obs_buf_queue) == 0:
            raise RuntimeError("Privileged observation buffer not initialized.")
        return torch.cat(list(self.priv_obs_buf_queue), dim=1)

    @property
    def num_privileged_obs(self) -> int:
        """Get number of privileged observations."""
        return self.num_priv_obs
