from __future__ import annotations

from typing import Union
import torch
from tensordict import TensorDict

from roboverse_pack.tasks.beyondmimic.base import EnvTypes


class RslRlEnvWrapper:
    def __init__(self, env: EnvTypes, train_cfg: dict | object = None):
        self.env = env
        self.train_cfg = train_cfg

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, Union[torch.Tensor, None], torch.Tensor, torch.Tensor, dict]:
        _ = self.env.step(actions)
        return self.obs_buf, self.env.rew_buf, self.env.reset_buf, self.env.extras

    def get_observations(self) -> TensorDict:
        return self.obs_buf

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
        return TensorDict(policy=self.env.obs_buf, critic=self.env.priv_obs_buf)
