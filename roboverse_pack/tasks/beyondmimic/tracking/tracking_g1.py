from __future__ import annotations

import copy

import torch

from metasim.scenario.lights import DomeLightCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.scenario.simulator_params import SimParamCfg
from metasim.task.registry import register_task
from metasim.types import TensorState
from metasim.utils.math import quat_rotate_inverse
from roboverse_learn.rl.beyondmimic.configs.tracking.tracking_g1 import (
    TrackingG1EnvCfg,
    TrackingG1RslRlTrainCfg,
)
from roboverse_pack.tasks.beyondmimic.base import LeggedRobotTask


@register_task("beyondmimic.tracking")
class TrackingG1Task(LeggedRobotTask):
    """Registered humanoid locomotion task."""

    env_cfg_cls = TrackingG1EnvCfg
    train_cfg_cls = TrackingG1RslRlTrainCfg
    task_name = "tracking_g1"

    scenario = ScenarioCfg(
        robots=["g1_dof29"],
        objects=[],
        cameras=[],
        num_envs=2,
        simulator="isaaclab",
        headless=True,
        env_spacing=2.5,
        decimation=4,
        sim_params=SimParamCfg(
            dt=0.005,
            substeps=1,
            num_threads=10,
            solver_type=1,
            num_position_iterations=4,
            num_velocity_iterations=0,
            contact_offset=0.01,
            rest_offset=0.0,
            bounce_threshold_velocity=0.5,
            max_depenetration_velocity=1.0,
            default_buffer_size_multiplier=5,
            replace_cylinder_with_capsule=True,
            friction_correlation_distance=0.025,
            friction_offset_threshold=0.04,
        ),
        lights=[
            DomeLightCfg(
                intensity=800.0,
                color=(0.85, 0.9, 1.0),
            )
        ],
    )

    def __init__(
        self,
        scenario: ScenarioCfg,
        device: str | torch.device,
        env_cfg: TrackingG1EnvCfg,
    ) -> None:
        scenario_copy = copy.deepcopy(scenario)
        scenario_copy.__post_init__()

        # record terminated envs for adapting sampling
        self.terminated_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.truncated_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        super().__init__(scenario=scenario_copy, config=env_cfg, device=device)

    def _init_buffers(self):  # TODO change command dim and obs dim
        # commands + base_ang_vel + projected_gravity + dof pos/vel/prev actions
        self.num_obs_single = 3 + 3 + 3 + self.num_actions * 3
        # commands + base_lin_vel + base_ang_vel + projected_gravity + dof pos/vel/prev actions
        self.num_priv_obs_single = 3 + 3 + 3 + 3 + self.num_actions * 3
        # Rewrite some hyperparameters
        # TODO consider encapsulating observation-related params into a config class
        self.obs_clip_limit = 100.0
        self.obs_scale = torch.ones(size=(self.num_obs_single,), dtype=torch.float, device=self.device)
        self.priv_obs_scale = torch.ones(size=(self.num_priv_obs_single,), dtype=torch.float, device=self.device)
        self.obs_noise = torch.zeros(size=(self.num_obs_single,), dtype=torch.float, device=self.device)

        # Observation scale
        self.obs_scale[3:6] = 0.2  # angular velocity
        self.obs_scale[9 + self.num_actions : 9 + 2 * self.num_actions] = 0.05  # joint velocity

        # Priviliged observation scale
        # TODO adopt observation scales of BeyondMimic
        self.priv_obs_scale[6:9] = 0.2  # angular velocity
        self.priv_obs_scale[12 + self.num_actions : 12 + 2 * self.num_actions] = 0.05  # joint velocity

        # Noise vector
        # [0:3] -> commands
        self.obs_noise[3:6] = 0.2  # [3:6] -> base_ang_vel
        self.obs_noise[6:9] = 0.05  # projected_gravity
        self.obs_noise[9 : 9 + self.num_actions] = 0.01
        self.obs_noise[9 + self.num_actions : 9 + 2 * self.num_actions] = 1.5  # joint velocities
        return super()._init_buffers()

    def _compute_task_observations(self, env_states: TensorState):
        robot_state = env_states.robots[self.robot.name]
        base_quat = robot_state.root_state[:, 3:7]
        base_lin_vel = quat_rotate_inverse(base_quat, robot_state.root_state[:, 7:10])
        base_ang_vel = quat_rotate_inverse(base_quat, robot_state.root_state[:, 10:13])
        projected_gravity = quat_rotate_inverse(base_quat, self.gravity_vec)

        q = env_states.robots[self.name].joint_pos - self.default_dof_pos
        dq = env_states.robots[self.name].joint_vel - self.default_dof_vel

        obs_buf = torch.cat(
            (
                self.commands_manager.value,  # 3
                base_ang_vel,  # 3
                projected_gravity,  # 3
                q,  # |A|
                dq,  # |A|
                self.actions,  # |A|
                # gait
            ),
            dim=-1,
        )
        priv_obs_buf = torch.cat(
            (
                self.commands_manager.value,  # 3
                base_lin_vel,  # 3
                base_ang_vel,  # 3
                projected_gravity,  # 3
                q,  # |A|
                dq,  # |A|
                self.actions,  # |A|
                # gait
            ),
            dim=-1,
        )

        # NOTE BeyondMimic: data += torch.rand_like(data) * (cfg.n_max - cfg.n_min) + cfg.n_min (data of shape [n_envs, n_dims])
        obs_buf += (2 * torch.rand_like(obs_buf) - 1) * self.obs_noise

        # clip observations -> scale observations
        obs_buf = obs_buf.clip(-self.obs_clip_limit, self.obs_clip_limit) * self.obs_scale
        priv_obs_buf = priv_obs_buf.clip(-self.obs_clip_limit, self.obs_clip_limit) * self.priv_obs_scale

        return obs_buf, priv_obs_buf  # TODO check if noise will be added later

    def _terminated(self, env_states: TensorState | None) -> torch.BoolTensor:
        """Override to record terminated (with time-out excluded) envs for adapting sampling."""
        self.terminated_buf[:] = False
        self.truncated_buf[:] = False
        for _key in self.terminate_callback.keys():
            _func, _params = self.terminate_callback[_key]
            _flag = (_func(self, env_states, **_params)).detach().clone().to(torch.bool)
            if _key == "time_out":
                self.truncated_buf = torch.logical_or(self.truncated_buf, _flag)
            else:
                self.terminated_buf = torch.logical_or(self.terminated_buf, _flag)
            self.episode_not_terminations[_key] += _flag.to(torch.float)
        return torch.logical_or(self.terminated_buf, self.truncated_buf)
