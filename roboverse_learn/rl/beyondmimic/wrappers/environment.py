from __future__ import annotations

from typing import Union
import torch
from tensordict import TensorDict

from roboverse_pack.tasks.beyondmimic.base import EnvTypes


class RslRlVecEnvWrapper:
    def __init__(self, env: EnvTypes, train_cfg: dict | object = None):
        self.env = env
        self.train_cfg = train_cfg

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, Union[torch.Tensor, None], torch.Tensor, torch.Tensor, dict]:
        _ = self.env.step(actions)
        return self.obs_buf["policy"], self.env.rew_buf, self.env.reset_buf, self.env.extras

    def get_observations(self) -> TensorDict:
        # return self.obs_buf  # FIXME this is for RSL-RL 3.1.1
        return self.obs_buf["policy"], {"observations": {"policy": self.obs_buf["policy"], "critic": self.obs_buf["critic"]}}  # NOTE this aligns with RSL-RL 2.3.0 used by `RslRlVecEnvWrapper` in Isaac Lab v2.1.0

    @property
    def num_envs(self):
        return self.env.num_envs

    @property
    def num_actions(self):
        return self.env.num_actions

    @property
    def max_episode_length(self):
        return self.env.max_episode_steps

    @property
    def episode_length_buf(self):
        return self.env._episode_steps

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        self.env._episode_steps = value

    @property
    def device(self):
        return self.env.device

    @property
    def cfg(self):
        return self.train_cfg

    @property
    def obs_buf(self) -> TensorDict:
        # NOTE this returns both policy and privileged obs, but `AgentTask.obs_buf()` only returns policy obs (a 2D tensor)
        return TensorDict(policy=self.env.obs_buf, critic=self.env.priv_obs_buf)
