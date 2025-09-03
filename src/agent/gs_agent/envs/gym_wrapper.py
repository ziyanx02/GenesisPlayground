from typing import Any, Final

import gymnasium as gym
import numpy.typing as npt
import torch
from gymnasium.spaces import Box, Discrete

from gs_agent.bases.env_wrapper import BaseEnvWrapper

_DEFAULT_DEVICE: Final[torch.device] = torch.device("cpu")


class GymEnvWrapper(BaseEnvWrapper):
    """Wrapper for a single Gymnasium env to typed (batch) step components (pseudo-batch).

    """

    def __init__(
        self, env: gym.Env[Any, Any], device: torch.device = _DEFAULT_DEVICE
    ) -> None:
        super().__init__(env)
        self._curr_obs: torch.Tensor = torch.tensor(self.env.reset()[0])

    # ---------------------------
    # BatchEnvWrapper API (batch)
    # ---------------------------
    def reset(self) -> tuple[torch.Tensor, dict[str, Any]]:
        obs, info = self.env.reset()
        self._curr_obs = torch.tensor(obs)
        return self._curr_obs, info

    def step(
        self, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        gym_actions = action.clone().cpu().numpy().squeeze()
        obs, reward, terminated, truncated, info = self.env.step(gym_actions)

        done = terminated | truncated
        # If episode is done, reset the environment
        if done:
            obs, info = self.env.reset()

        self._curr_obs = torch.tensor(obs)

        reward_batch = torch.as_tensor([[float(reward)]])
        done_batch = torch.as_tensor([[1.0 if done else 0.0]])
        return self._curr_obs, reward_batch, done_batch, info

    def get_observation(self) -> torch.Tensor:
        return self._curr_obs

    @property
    def env_spec(self) -> gym.Env[Any, Any]:
        return self.env

    @property
    def num_envs(self) -> int:
        return 1

    def close(self) -> None:
        self.env.close()

    def render(self) -> None:
        self.env.render()