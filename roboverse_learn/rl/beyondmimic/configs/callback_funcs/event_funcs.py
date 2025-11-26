import torch
from typing import Literal

from roboverse_learn.rl.beyondmimic.configs.cfg_randomizers import randomize_prop_by_op
from roboverse_pack.tasks.beyondmimic.base.types import EnvTypes


# adapted from BeyondMimic events.py

def randomize_joint_default_pos(  # startup
    env: EnvTypes,
    env_ids: torch.Tensor | None,
    joint_ids: torch.Tensor | None,
    pos_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize the joint default positions which may be different from URDF due to calibration errors."""
    # save nominal value for export
    # TODO check where `default_dof_pos_nominal` will be used
    env.default_dof_pos_nominal = torch.clone(env.default_dof_pos[0])

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scenario.num_envs, device=env.device)

    # resolve joint indices
    if joint_ids is None:
        joint_ids = torch.arange(len(env.sorted_joint_names), device=env.device)

    if pos_distribution_params is not None:
        pos = env.default_dof_pos.unsqueeze(0).repeat(env.scenario.num_envs, 1).to(env.device).clone()  # [n_envs, n_dofs]
        pos = randomize_prop_by_op(
            pos, pos_distribution_params, env_ids, joint_ids, operation=operation, distribution=distribution
        )[env_ids][:, joint_ids]

        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]
        env.default_dof_pos[env_ids, joint_ids] = pos
