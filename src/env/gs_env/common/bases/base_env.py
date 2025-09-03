from abc import ABC, abstractmethod

import torch
from typing import Any


class BaseGymEnv(ABC):
    """
    Abstract base class for Gym-like environments using TensorDict.
    """

    # TODO
    _observation_space: Any
    _action_space: Any
    _info_space: Any
    _num_env: int

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @property
    def observation_space(self) -> Any:
        return self._observation_space

    @property
    def action_space(self) -> Any:
        return self._action_space

    @property
    def info_space(self) -> Any:
        return self._info_space

    @property
    def num_envs(self) -> int:
        return self._num_env

    @abstractmethod
    def reset(self, envs_idx: torch.IntTensor | None = None) -> None:
        """
        Reset the environment and return the initial observation as a TensorDict.
        """
        ...

    @abstractmethod
    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Take an action and return:
        - 'observation'
        - 'reward'
        - 'done'
        - 'info'
        """
        ...

    @property
    @abstractmethod
    def action_dim(self) -> int:
        """
        The number of actions available in the action space.
        """
        ...

    @property
    @abstractmethod
    def actor_obs_dim(self) -> int:
        """
        The dimension of the actor's observation space.
        """
        ...

    @property
    @abstractmethod
    def critic_obs_dim(self) -> int:
        """
        The dimension of the critic's observation space.
        """
        ...
