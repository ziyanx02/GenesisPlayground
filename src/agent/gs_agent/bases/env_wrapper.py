from abc import ABC, abstractmethod

from typing import Any, Final
import torch

class BaseEnvWrapper(ABC):
    """
    Base class for all environment wrappers.
    """

    def __init__(self, env: Any) -> None:
        self.env: Final[Any] = env
        
    @abstractmethod
    def get_observations(self) -> tuple[torch.Tensor, dict[str, Any]]:
        ...
    
    @abstractmethod
    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        ...
         
    @property
    @abstractmethod
    def action_dim(self) -> int:
        ...
        
    @property
    @abstractmethod
    def actor_obs_dim(self) -> int:
        ...
         
    @property
    @abstractmethod
    def critic_obs_dim(self) -> int:
        ...
        
    @property
    @abstractmethod
    def num_envs(self) -> int:
        ...