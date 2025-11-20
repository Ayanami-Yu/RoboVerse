"""RSL-RL wrapper for RoboVerse environments."""

from __future__ import annotations

from typing import Union

import torch
from tensordict import TensorDict

from rsl_rl.env import VecEnv


class RslRlVecEnvWrapperRV(VecEnv):
    """RSL-RL wrapper for RoboVerse task environments."""

    def __init__(self, env, train_cfg: dict | object = None):
        """Initialize the wrapper.

        Args:
            env: RoboVerse task environment
            train_cfg: Training configuration
        """
        self.env = env
        self.train_cfg = train_cfg

        # Get observation and action dimensions
        self.num_obs = env.num_obs
        self.num_privileged_obs = getattr(env, "num_priv_obs", env.num_obs)
        self.num_actions = env.num_actions
        self.max_episode_length = env.max_episode_steps

    def get_observations(self) -> TensorDict:
        """Return the current observations.

        Returns:
            observations (TensorDict): Observations from the environment.
        """
        return TensorDict(
            policy=self.env.obs_buf,
            critic=self.env.priv_obs_buf,
        )

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, Union[torch.Tensor, None], torch.Tensor, torch.Tensor, dict]:
        """Step the environment.

        Args:
            actions: Actions to take

        Returns:
            obs, privileged_obs, rewards, dones, infos
        """
        obs, reward, terminated, time_out, info = self.env.step(actions)
        done = terminated | time_out

        # Get privileged observations from info
        priv_obs = info.get("privileged_observation", obs)

        return obs, priv_obs, reward, done, info

    def reset(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Reset the environment.

        Returns:
            obs, privileged_obs
        """
        obs, info = self.env.reset()
        priv_obs = info.get("privileged_observation", obs)
        return obs, priv_obs

    @property
    def num_envs(self):
        """Number of environments."""
        return self.env.num_envs

    @property
    def device(self):
        """Device."""
        return self.env.device

    @property
    def obs_buf(self) -> TensorDict:
        """Observation buffer."""
        return TensorDict(
            policy=self.env.obs_buf,
            critic=self.env.priv_obs_buf,
        )

    @property
    def rew_buf(self) -> torch.Tensor:
        """Reward buffer."""
        # RSL-RL expects rewards to be stored, but RoboVerse doesn't store them
        # Return zeros as placeholder
        return torch.zeros(self.num_envs, device=self.device)

    @property
    def reset_buf(self) -> torch.Tensor:
        """Reset buffer."""
        # RSL-RL expects reset flags to be stored
        # Return zeros as placeholder (actual resets handled in step)
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    @property
    def episode_length_buf(self) -> torch.Tensor:
        """Episode length buffer."""
        return self.env._episode_steps

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        """Set episode length buffer."""
        self.env._episode_steps = value

    @property
    def extras(self) -> dict:
        """Extra information."""
        return {}
