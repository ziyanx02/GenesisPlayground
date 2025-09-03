from abc import ABC, abstractmethod

import torch

import gs_env.common.bases.spaces as spaces


class BaseGymEnv(ABC):
    """
    Abstract base class for Gym-like environments using TensorDict.
    """

    _observation_space: spaces.Space
    _action_space: spaces.Space
    _info_space: spaces.Space
    _num_env: int

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @property
    def observation_space(self) -> spaces.Space:
        return self._observation_space

    @property
    def action_space(self) -> spaces.Space:
        return self._action_space

    @property
    def info_space(self) -> spaces.Space:
        return self._info_space

    @property
    def num_envs(self) -> int:
        return self._num_env

    @abstractmethod
    def reset(self, envs_idx: torch.IntTensor | None = None) -> None:
        """
        Reset the environment and return the initial observation as a TensorDict.
        """
        pass

    @abstractmethod
    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Take an action and return:
        - 'observation'
        - 'reward'
        - 'done'
        - 'info'
        """
        pass

    @property
    @abstractmethod
    def action_dim(self) -> int:
        """
        The number of actions available in the action space.
        """
        pass

    @property
    @abstractmethod
    def actor_obs_dim(self) -> int:
        """
        The dimension of the actor's observation space.
        """
        pass

    @property
    @abstractmethod
    def critic_obs_dim(self) -> int:
        """
        The dimension of the critic's observation space.
        """
        pass
