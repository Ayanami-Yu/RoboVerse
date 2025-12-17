from __future__ import annotations

import copy
from dataclasses import asdict

import torch

from metasim.scenario.lights import DomeLightCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.scenario.simulator_params import SimParamCfg
from metasim.task.registry import register_task
from metasim.types import TensorState
from roboverse_learn.rl.configs.rsl_rl.ppo_tracking import RslRlPPOTrackingConfig
from roboverse_pack.tasks.beyondmimic.metasim.configs.tracking_g1 import TrackingG1EnvCfg

from .base_legged_robot import LeggedRobotTask


@register_task("motion-tracking")
class TrackingG1Task(LeggedRobotTask):
    """Registered BeyondMimic motion tracking task."""

    # env_cfg_cls = TrackingG1EnvCfg
    # train_cfg_cls = TrackingG1RslRlTrainCfg
    # task_name = "tracking_g1"

    scenario = ScenarioCfg(
        robots=["g1_tracking"],
        objects=[],
        cameras=[],
        num_envs=2,
        simulator="isaacsim",
        headless=True,
        env_spacing=2.5,
        decimation=1,  # NOTE task-level decimation is defined by `self.cfg.decimation`
        sim_params=SimParamCfg(  # TODO why this is not even the default values of `PhysxCfg` in Isaac Lab?
            dt=0.005,
            substeps=1,
            num_threads=10,
            solver_type=1,
            # num_position_iterations=4,
            num_position_iterations=255,
            # num_velocity_iterations=0,
            num_velocity_iterations=255,
            # contact_offset=0.01,  # not used?
            # rest_offset=0.0,  # not used?
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
        args: RslRlPPOTrackingConfig,
        device: str | torch.device,
        # env_cfg: TrackingG1EnvCfg,
        reset_in_env_wrapper: bool = True,
    ) -> None:
        scenario_copy = copy.deepcopy(scenario)
        scenario_copy.__post_init__()

        cfg = TrackingG1EnvCfg()
        cfg.commands.motion_file = args.motion_file

        super().__init__(scenario=scenario_copy, config=cfg, device=device)
        if not reset_in_env_wrapper:
            self.reset()

    def _compute_observation_group(self, env_states: TensorState, group_name: str):
        """Compute all observations of a given group and concatenate them into a single tensor."""
        obs_terms = getattr(self.cfg.observations, group_name)
        group_obs = []
        for term_cfg in asdict(obs_terms).values():
            if term_cfg["params"]:
                obs: torch.Tensor = term_cfg["func"](self, env_states, **term_cfg["params"]).clone()
            else:
                obs: torch.Tensor = term_cfg["func"](self, env_states).clone()
            if term_cfg["noise_range"]:
                obs += (
                    torch.rand_like(obs) * (term_cfg["noise_range"][1] - term_cfg["noise_range"][0])
                    + term_cfg["noise_range"][0]
                )  # [n_envs, n_dims]
            group_obs.append(obs)
        return torch.cat(group_obs, dim=-1)

    def _observation(self, env_states: TensorState):
        obs_buf = dict()
        for group_name in ["policy", "critic"]:
            obs_buf[group_name] = self._compute_observation_group(env_states, group_name)

        # return obs_buf["policy"], obs_buf["critic"]
        return obs_buf

    """def _terminated(self, env_states: TensorState | None) -> torch.BoolTensor:
        # Override to record terminated (with time-out excluded) envs for adapting sampling
        self.terminated_buf[:] = False
        self.truncated_buf[:] = False
        for _key in self.terminate_callback.keys():
            _func, _params = self.terminate_callback[_key]
            _flag = (_func(self, env_states, **_params)).detach().clone().to(torch.bool)
            if _key == "time_out":
                self.truncated_buf = torch.logical_or(self.truncated_buf, _flag)
            else:
                self.terminated_buf = torch.logical_or(self.terminated_buf, _flag)
            self.episode_not_terminated[_key] += _flag.to(torch.float)
        return torch.logical_or(self.terminated_buf, self.truncated_buf)"""
