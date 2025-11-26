from __future__ import annotations

import torch
from typing import Sequence

from metasim.types import TensorState
from metasim.utils.math import (
    quat_apply,
    wrap_to_pi,
    sample_uniform,
    quat_mul,
    yaw_quat,
    quat_from_euler_xyz,
)

from roboverse_pack.tasks.beyondmimic.base.types import EnvTypes
from roboverse_learn.rl.beyondmimic.configs.cfg_base import BaseEnvCfg
from roboverse_learn.rl.beyondmimic.helper.math import quat_inv
from roboverse_learn.rl.beyondmimic.helper.motion_utils import MotionCommand


# adapted from `isaaclab.envs.mdp.events.py`

def push_by_setting_velocity(
    env: EnvTypes,
    env_states: TensorState,
    interval_range_s: tuple | int,
    velocity_range: dict[str, tuple[float, float]],
):
    """Randomly set robot's root velocity to simulate a push.

    The function takes a dictionary of velocity ranges for each axis and rotation. The keys of the dictionary are "x", "y", "z", "roll", "pitch", and "yaw". The values are tuples of the form (min, max).
    """
    if not hasattr(env, "push_interval"):
        env.push_interval = (
            sample_uniform(
                interval_range_s[0],
                interval_range_s[1],
                (env.num_envs, 1),
                device=env.device,
            ).flatten()
            / env.step_dt  # convert seconds to simulation steps
        ).to(torch.int)
    push_env_ids = (
        torch.logical_and(
            env._episode_steps % env.push_interval == 0, env._episode_steps > 0
        )
        .nonzero(as_tuple=False)
        .flatten()
    )
    if len(push_env_ids) == 0:
        return
    ranges = torch.tensor([velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]], device=env.device)
    env_states.robots[env.name].root_state[push_env_ids, 7:13] += sample_uniform(
        ranges[:, 0], ranges[:, 1], (len(push_env_ids), 3), device=env.device
    )  # add random velocity to root's linear and angular velocities

    env.handler.set_states(env_states, push_env_ids.tolist())
