from __future__ import annotations

import math
from collections import deque
from copy import deepcopy
from dataclasses import asdict

import torch

from metasim.scenario.scenario import ScenarioCfg
from metasim.utils.state import TensorState
from roboverse_learn.rl.beyondmimic.configs.cfg_base import BaseEnvCfg
from roboverse_learn.rl.beyondmimic.helper.string_utils import find_bodies
from roboverse_learn.rl.beyondmimic.helper.utils import (
    get_axis_params,
    pattern_match,
)
from roboverse_learn.rl.beyondmimic.mdp.commands import MotionCommand
from roboverse_pack.robots import G1Dof12Cfg, Go2Cfg

from .base_agent import AgentTask


class LeggedRobotTask(AgentTask):
    """A base task env for legged robots."""

    def __init__(
        self,
        scenario: ScenarioCfg,
        config: BaseEnvCfg,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__(scenario=scenario, config=config, device=device)
        self.name = self.robot.name if hasattr(self, "robot") else getattr(self, "name", None)
        self.num_actions = len(self.robot.actuators)
        self.sim_dt = self.scenario.sim_params.dt
        self.sorted_body_names = self.handler.get_body_names(self.name, sort=True)
        self.sorted_joint_names = self.handler.get_joint_names(self.name, sort=True)

        self._instantiate_cfg(self.cfg)
        self._init_joint_cfg()
        self._init_buffers()
        self._setup()
        self.reset()

    def _setup(self):
        for _setup_fn, _params in self.setup_callback.values():
            _setup_fn(env=self, **_params)  # TODO check if this will break `MaterialRandomizer` and `MassRandomizer`

    def _observation(self, env_states: TensorState) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return (policy_obs, privileged_obs). Implemented by subclasses."""
        raise NotImplementedError

    def _instantiate_cfg(self, config: BaseEnvCfg | None):
        self.cfg = config  # FIXME reassignment?
        # value assignments from configs
        self.decimation = self.cfg.control.decimation
        self.step_dt = self.sim_dt * self.decimation
        self.action_clip = self.cfg.control.action_clip
        self.action_scale = self.cfg.control.action_scale

        self.common_step_counter = 0
        self.commands = MotionCommand(env=self, cfg=self.cfg.commands)

    def _init_joint_cfg(self):
        """Parse default joint positions and torque limits from cfg."""
        robot: G1Dof12Cfg | Go2Cfg = self.robot
        sorted_joint_names: list[str] = self.sorted_joint_names
        original_joint_names: list[str] = self.handler.get_joint_names(self.robot.name, sort=False)
        self.sorted_to_original_joint_indexes = torch.tensor(
            find_bodies(original_joint_names, sorted_joint_names, preserve_order=True)[0], device=self.device
        )
        self.original_to_sorted_joint_indexes = torch.tensor(
            find_bodies(sorted_joint_names, original_joint_names, preserve_order=True)[0], device=self.device
        )

        torque_limits = (  # TODO check its order
            robot.torque_limits
            if hasattr(robot, "torque_limits")  # False
            else {name: actuator_cfg.torque_limit for name, actuator_cfg in robot.actuators.items()}
        )

        sorted_limits = [torque_limits[name] for name in sorted_joint_names]
        self.torque_limits = (
            torch.tensor(sorted_limits, device=self.device) * self.cfg.control.torque_limits_factor
        )  # (n_dof,)

        p_gains = []
        d_gains = []
        for name in sorted_joint_names:
            actuator_cfg = robot.actuators[name]
            p_gains.append(actuator_cfg.stiffness if actuator_cfg.stiffness is not None else 0.0)
            d_gains.append(actuator_cfg.damping if actuator_cfg.damping is not None else 0.0)

        self.p_gains = torch.tensor(p_gains, device=self.device)
        self.d_gains = torch.tensor(d_gains, device=self.device)

        # check if manual PD control is needed (if any joints use effort control)
        control_types = robot.control_type
        self.manual_pd_on = any(mode == "effort" for mode in control_types.values()) if control_types else False

        dof_pos_limits = robot.joint_limits
        sorted_dof_pos_limits = [dof_pos_limits[joint] for joint in sorted_joint_names]
        self.dof_pos_limits = torch.tensor(sorted_dof_pos_limits, device=self.device)  # (n_dofs, 2)

        soft_limit_factor = getattr(
            self.cfg.control,
            "soft_joint_pos_limit_factor",
            getattr(self.robot, "soft_joint_pos_limit_factor", 1.0),
        )
        _mid = (self.dof_pos_limits[:, 0] + self.dof_pos_limits[:, 1]) / 2.0
        _diff = self.dof_pos_limits[:, 1] - self.dof_pos_limits[:, 0]

        # NOTE same as `ArticulationData.soft_joint_pos_limits` in Isaac Lab
        soft_dof_pos_limits = torch.zeros_like(self.dof_pos_limits, device=self.device)
        soft_dof_pos_limits[:, 0] = _mid - 0.5 * _diff * soft_limit_factor
        soft_dof_pos_limits[:, 1] = _mid + 0.5 * _diff * soft_limit_factor
        self.soft_dof_pos_limits_sorted = soft_dof_pos_limits.unsqueeze(0).repeat(
            self.num_envs, 1, 1
        )  # (n_envs, n_dofs, 2)
        self.soft_dof_pos_limits_original = self.soft_dof_pos_limits_sorted[:, self.sorted_to_original_joint_indexes, :]

        dof_vel_limits = getattr(
            robot,
            "joint_velocity_limits",
            [getattr(robot.actuators[name], "velocity_limit", torch.inf) for name in sorted_joint_names],
        )
        self.dof_vel_limits = torch.tensor(dof_vel_limits, device=self.device)  # (n_dofs, 2)
        self.soft_dof_vel_limits = self.dof_vel_limits * getattr(
            self.cfg.control,
            "soft_joint_vel_limit_factor",
            getattr(self.robot, "soft_joint_vel_limit_factor", 1.0),
        )

        default_joint_pos = self.cfg.initial_states.robots[robot.name].get(
            "default_joint_pos", robot.default_joint_positions
        )
        default_joint_pos = pattern_match(default_joint_pos, sorted_joint_names)
        sorted_joint_pos = [default_joint_pos[name] for name in sorted_joint_names]
        default_dof_pos = torch.tensor(sorted_joint_pos, device=self.device)  # (n_dofs,)
        self.default_dof_pos = default_dof_pos.unsqueeze(0).repeat(self.num_envs, 1)  # (n_envs, n_dofs)

        default_joint_vel = getattr(robot, "default_joint_velocities", 0)
        if isinstance(default_joint_vel, dict):
            default_joint_vel = pattern_match(default_joint_vel, sorted_joint_names)
        sorted_joint_vel = (
            [default_joint_vel[name] for name in sorted_joint_names]
            if isinstance(default_joint_vel, dict)
            else [default_joint_vel for _ in sorted_joint_names]
        )
        default_dof_vel = torch.tensor(sorted_joint_vel, device=self.device)  # (n_dofs,)
        self.default_dof_vel = default_dof_vel.unsqueeze(0).repeat(self.num_envs, 1)  # (n_envs, n_dofs)

    def _pre_physics_step(self, actions: torch.Tensor):
        """Apply pre-physics callbacks."""
        for pre_fn, _params in self.pre_physics_step_callback.values():
            pre_fn(self, **_params)
        # NOTE corresponds to action clipping in `RslRlVecEnvWrapper.step()` in Isaac Lab
        return torch.clip(actions, -self.action_clip, self.action_clip)

    def _init_buffers(self):
        self.actions = torch.zeros(
            size=(self.num_envs, self.num_actions),
            dtype=torch.float,
            device=self.device,
        )
        self.actions_offset = self.default_dof_pos.clone() if self.cfg.control.action_offset else 0.0

        self.rew_buf = torch.zeros(size=(self.num_envs,), dtype=torch.float, device=self.device)
        self.reset_buf = torch.zeros(
            size=(self.num_envs,), dtype=torch.bool, device=self.device
        )  # both time-out and reset

        # record terminated envs for adapting sampling
        self.terminated_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.truncated_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # set gravity vector
        self.up_axis_idx = 2
        self.gravity_vec = torch.tensor(
            get_axis_params(-1.0, self.up_axis_idx),
            dtype=torch.float,
            device=self.device,
        ).repeat((self.num_envs, 1))

        # reset commands
        self.commands.reset()  # TODO no need to record the command metrics here?

        # history buffer (action) for reward computation
        self.history_buffer = {}
        self.history_buffer["actions"] = deque([self.actions.clone() * 0.0], maxlen=2)
        # self.history_buffer["joint_vel"] = deque([self.actions.clone() * 0.0], maxlen=2)  # TODO remove this

        # for observation history
        env_states = self.handler.get_states()
        obs_single, priv_obs_single = self._observation(env_states)
        self.obs_buf_queue = deque(
            [deepcopy(obs_single * 0.0) for _ in range(self.cfg.obs_len_history)],
            maxlen=self.cfg.obs_len_history,
        )
        self.priv_obs_buf_queue = deque(
            [deepcopy(priv_obs_single * 0.0) for _ in range(self.cfg.priv_obs_len_history)],
            maxlen=self.cfg.priv_obs_len_history,
        )

        # for logging
        self.episode_not_terminated = {
            _key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for _key in self.terminate_callback.keys()
        }  # FIXME why is this float instead of bool?
        self.episode_rewards = {
            name: torch.zeros(
                self.num_envs,
                dtype=torch.float,
                device=self.device,
                requires_grad=False,
            )
            for name in asdict(self.cfg.rewards).keys()
        }

    def _compute_effort(self, actions: torch.Tensor, env_states: TensorState) -> torch.Tensor:
        """Compute effort from actions using PD control."""
        # get current joint positions and velocities
        sorted_dof_pos = env_states.robots[self.name].joint_pos
        sorted_dof_vel = env_states.robots[self.name].joint_vel

        # compute PD control effort
        effort = self.p_gains * (actions - sorted_dof_pos) - self.d_gains * sorted_dof_vel

        # apply torque limits
        effort = torch.clip(effort, -self.torque_limits, self.torque_limits)
        return effort.to(torch.float32)

    def reset(self, env_ids: torch.Tensor | list[int] | None = None):
        """Reset selected envs (defaults to all)."""
        if env_ids is None:
            env_ids = torch.tensor(list(range(self.num_envs)), device=self.device)

        self.extras["episode"] = {}

        # update the curriculum for environments that need a reset
        if self.cfg.curriculum.enabled:
            for _name, _func in self.cfg.curriculum.funcs.items():
                _return_val = _func(self, env_ids)
                self.extras["episode"]["Curriculum/" + _name] = _return_val

        # reset the internal buffers of the scene elements  # TODO is this needed?

        # apply events such as randomizations for environments that need a reset
        for _reset_fn, _params in self.reset_callback.values():
            _ = _reset_fn(self, env_ids, **_params)
        self.set_states(states=self.setup_initial_env_states, env_ids=env_ids)

        # reset observations
        for _obs in self.obs_buf_queue:
            _obs[env_ids] = 0.0
        for _priv_obs in self.priv_obs_buf_queue:
            _priv_obs[env_ids] = 0.0

        # reset actions and other history buffers
        self.actions[env_ids] = 0.0
        for history in self.history_buffer.values():
            for item in history:
                item[env_ids] = 0.0

        # reset rewards
        # self.rew_buf[env_ids] = 0.0  # FIXME why did this even exist?
        for key in self.episode_rewards.keys():
            self.extras["episode"]["Episode_Reward/" + key] = (
                torch.mean(self.episode_rewards[key][env_ids]) / self.cfg.max_episode_length_s
            )
            self.episode_rewards[key][env_ids] = 0.0

        # reset curriculum

        # reset commands
        metrics = self.commands.reset(env_ids=env_ids)  # TODO
        for metric_name, metric_value in metrics.items():
            self.extras["episode"][f"Metrics_Motion/{metric_name}"] = metric_value

        # reset events

        # reset terminations
        for key in self.episode_not_terminated.keys():
            self.extras["episode"]["Episode_Termination/" + key] = (
                torch.mean(self.episode_not_terminated[key][env_ids]) / self.max_episode_steps
            )
            self.episode_not_terminated[key][env_ids] = 0.0

        # reset the episode length buffer
        self._episode_steps[env_ids] = 0

    def step(
        self,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Apply actions, simulate for `decimation` steps, and compute RLTask-style outputs."""
        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions, device=self.device, dtype=torch.float32)
        if actions.ndim == 1:
            actions = actions.unsqueeze(0)

        # NOTE `self.actions` is in the original order since it's computed inside `OnPolicyRunner`
        actions = self._pre_physics_step(actions)
        self.actions[:] = actions  # original order
        processed_actions = (  # TODO modify action scale according to BeyondMimic
            (self.actions * self.action_scale + self.actions_offset)
            .clip(-self.action_clip, self.action_clip)
            .clone()[:, self.original_to_sorted_joint_indexes]
        )  # sorted order
        env_states = self.get_states()
        for _ in range(self.decimation):
            applied_action = (
                self._compute_effort(processed_actions, env_states) if self.manual_pd_on else processed_actions
            )
            env_states = self._physics_step(applied_action)

        self._post_physics_step(env_states)

        # NOTE for RSL-RL v2.3.0
        self.extras["observations"] = {"policy": self.obs_buf, "critic": self.priv_obs_buf}

        return (
            self.obs_buf,
            self.rew_buf,
            self.reset_buf,
            self.extras,
        )

    def _post_physics_step(self, env_states: TensorState):
        self._episode_steps += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)

        # check termination conditions and compute rewards
        self.reset_buf[:] = self._terminated(env_states)
        self.rew_buf[:] = self._reward(env_states)

        # reset envs and MDP
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset(env_ids=reset_env_ids)
            # TODO check the corresponding code to the commented lines
            # update articulation kinematics
            # self.scene.write_data_to_sim()
            # self.sim.forward()

        # update commands
        self.commands.compute(env_states)

        # step interval events
        for _step_fn, _params in self.post_physics_step_callback.values():
            _step_fn(self, env_states, **_params)

        # compute observations after resets
        obs_single, priv_single = self._observation(env_states)
        # append to observation history buffers
        self.obs_buf_queue.append(obs_single)
        if priv_single is not None and self.priv_obs_buf_queue.maxlen > 0:
            self.priv_obs_buf_queue.append(priv_single)

        # copy to the history buffer
        # TODO verify whether history length of 1 equals not using history
        for key, history in self.history_buffer.items():
            if hasattr(self, key):
                history.append(getattr(self, key).clone())
            elif hasattr(env_states.robots[self.name], key):
                history.append(getattr(env_states.robots[self.name], key).clone())
            else:
                raise ValueError(f"History buffer key {key} not found in task or robot states.")

    def _reward(self, env_states: TensorState):
        rew_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        for name, term_cfg in asdict(self.cfg.rewards).items():
            # TODO multiply `step_dt` or not
            if term_cfg["params"]:
                value = term_cfg["func"](self, env_states, **term_cfg["params"]) * term_cfg["weight"] * self.step_dt
                # value = term_cfg["func"](self, env_states, **term_cfg["params"]) * term_cfg["weight"]
            else:
                value = term_cfg["func"](self, env_states) * term_cfg["weight"] * self.step_dt
                # value = term_cfg["func"](self, env_states) * term_cfg["weight"]
            rew_buf += value  # update total reward
            self.episode_rewards[name] += value  # update episodic sum

        return rew_buf

    def _terminated(self, env_states: TensorState | None) -> torch.BoolTensor:
        reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for _key in self.terminate_callback.keys():
            _terminate_fn, _params = self.terminate_callback[_key]
            _terminate_flag = (_terminate_fn(self, env_states, **_params)).detach().clone().to(torch.bool)
            reset_buf = torch.logical_or(reset_buf, _terminate_flag)
            self.episode_not_terminated[_key] += _terminate_flag.to(torch.float)
        return reset_buf

    def _get_initial_states(self):
        """Return list of per-env initial states derived from config."""
        sorted_joint_names = self.handler.get_joint_names(self.robot.name, sort=True)

        robot_state = self.cfg.initial_states.robots[self.robot.name]
        pos = robot_state.get("pos", [0.0, 0.0, 0.5])  # TODO verify the default values
        rot = robot_state.get("rot", [1.0, 0.0, 0.0, 0.0])

        joint_pos = robot_state.get(
            "joint_pos",
            robot_state.get("default_joint_pos", self.robot.default_joint_positions),
        )
        joint_pos = pattern_match(joint_pos, sorted_joint_names)

        joint_vel = robot_state.get(
            "joint_vel",
            robot_state.get("default_joint_vel", getattr(self.robot, "default_joint_velocities", {})),
        )
        joint_vel = pattern_match(joint_vel, sorted_joint_names)

        template = {
            "objects": {},
            "robots": {
                self.robot.name: {
                    "pos": torch.tensor(pos, dtype=torch.float32),
                    "rot": torch.tensor(rot, dtype=torch.float32),
                    "dof_pos": {name: joint_pos[name] for name in joint_pos},
                    "dof_vel": {name: joint_vel[name] for name in joint_vel},
                }
            },
        }
        return [deepcopy(template) for _ in range(self.scenario.num_envs)]

    @property
    def max_episode_steps(self):
        """Maximum episode length in steps."""
        return math.ceil(self.cfg.max_episode_length_s / self.step_dt)
