from abc import ABC, abstractmethod

from typing import Any
import torch



class BaseEnvWrapper(ABC):
    """
    Base class for all environment wrappers.
    """

    def __init__(self, env: Any) -> None:
        ...

        
    @abstractmethod
    def get_observations(self) -> tuple[torch.Tensor, dict]:
        ...
    
    @abstractmethod
    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        ...
        
    @abstractmethod
    def get_depth_image(self, normalize: bool = True) -> torch.Tensor:
        ...
        
    @abstractmethod
    def get_rgb_image(self, normalize: bool = True) -> torch.Tensor:
        ...
        
    @property
    @abstractmethod
    def action_dim(self) -> int:
        ...
        
    @property
    @abstractmethod
    def actor_obs_dim(self) -> int:
        ...
        
        
    @abstractmethod
    def critic_obs_dim(self) -> int:
        ...
        
    @property
    @abstractmethod
    def depth_shape(self) -> tuple[int, int] | None:
        ...

    @property
    @abstractmethod
    def rgb_shape(self) -> tuple[int, int] | None:
        ...
        
    @property
    @abstractmethod
    def img_resolution(self) -> tuple[int, int]:
        ...
        
    @property
    @abstractmethod
    def num_envs(self) -> int:
        ...