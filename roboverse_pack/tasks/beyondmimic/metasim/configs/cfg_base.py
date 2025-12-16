from __future__ import annotations

from dataclasses import MISSING
from typing import Callable

from metasim.utils import configclass


@configclass
class CallbacksCfg:
    """Configuration for callbacks."""

    setup: dict = {}
    reset: dict = {}  # func_name: (func(env, env_ids,**kwargs), kwargs)
    pre_step: dict = {}  # func_name: (func(env, actions, **kwargs), kwargs)
    post_step: dict = {}  # func_name: (func(env, env_states, **kwargs), kwargs)
    terminate: dict = {}  # func_name: (func(env, env_states, **kwargs), kwargs)
    query: dict = {}


@configclass
class BaseEnvCfg:
    """The base class of environment configuration for legged robots."""

    max_episode_length_s = 10.0
    obs_len_history = 1  # number of past observations to include in the observation
    priv_obs_len_history = 1  # number of past privileged observations to include in the privileged observation

    @configclass
    class Control:
        """Configuration for control."""

        torque_limits_factor: float = 1.0  # scale torque limits from urdf  # TODO check this
        soft_joint_pos_limit_factor: float = 0.9
        action_clip: float | None = None  # TODO check this
        action_scale = 0.25
        action_offset = True  # offset actions by `default_dof_pos_original` specified in the task class
        decimation = 4  # task-level

    control = Control()

    @configclass
    class Curriculum:
        """Configuration for curriculum."""

        enabled = False
        funcs: dict[str, Callable] = MISSING

    curriculum = Curriculum()

    class InitialStates:
        """Configuration for initial states."""

        objects = {}
        robots = {
            "g1_tracking": {  # TODO check if this will override `IsaacsimHandler._add_robot()`
                "pos": [0.0, 0.0, 0.76],
                "default_joint_pos": {
                    ".*_hip_pitch_joint": -0.312,
                    ".*_knee_joint": 0.669,
                    ".*_ankle_pitch_joint": -0.363,
                    ".*_elbow_joint": 0.6,
                    "left_shoulder_roll_joint": 0.2,
                    "left_shoulder_pitch_joint": 0.2,
                    "right_shoulder_roll_joint": -0.2,
                    "right_shoulder_pitch_joint": 0.2,
                },
            }
        }

    initial_states = InitialStates()

    callbacks_setup: dict[str, tuple[Callable, dict] | Callable] = {}
    # func_name: (func(env, env_ids,**kwargs), kwargs)
    callbacks_reset: dict[str, tuple[Callable, dict] | Callable] = {}
    # func_name: (func(env, env_states, **kwargs), kwargs)
    callbacks_pre_step: dict[str, tuple[Callable, dict] | Callable] = {}
    # func_name: (func(env, actions, **kwargs), kwargs)
    callbacks_post_step: dict[str, tuple[Callable, dict] | Callable] = {}
    # func_name: (func(env, env_states, **kwargs), kwargs)
    callbacks_terminate: dict[str, tuple[Callable, dict] | Callable] = MISSING
    callbacks_query: dict[str, tuple[Callable, dict] | Callable] = MISSING

    def __post_init__(self):

        def _normalize(value) -> dict:
            return {} if value is MISSING else value

        self.callbacks = CallbacksCfg()
        self.callbacks.query = _normalize(self.callbacks_query)
        self.callbacks.terminate = _normalize(self.callbacks_terminate)
        self.callbacks.setup = _normalize(self.callbacks_setup)
        self.callbacks.reset = _normalize(self.callbacks_reset)
        self.callbacks.pre_step = _normalize(self.callbacks_pre_step)
        self.callbacks.post_step = _normalize(self.callbacks_post_step)

        # Type check for callbacks
        for cb_attr in [
            "setup",
            "reset",
            "pre_step",
            "post_step",
            "terminate",
            "query",
        ]:
            cb_dict = getattr(self.callbacks, cb_attr)
            for func_name, func_tuple in cb_dict.items():
                if not (
                    callable(func_tuple)
                    or (
                        isinstance(func_tuple, tuple)
                        and len(func_tuple) == 2
                        and (callable(func_tuple[0]) or isinstance(func_tuple[0], object))
                        and isinstance(func_tuple[1], dict)
                    )
                ):
                    raise ValueError(
                        f"Callback {func_name} in {cb_attr} must be a callable or a tuple of (callable, dict)."
                    )
