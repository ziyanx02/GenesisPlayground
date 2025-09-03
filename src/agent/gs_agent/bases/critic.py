
from abc import ABC, abstractmethod

from torch import nn
import torch



class BaseStateValueFunction(nn.Module, ABC):
    """State value function V(s)."""

    @abstractmethod
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: [B, O]  ->  returns V: [B, 1]."""
        ...


class BaseQValueFunction(nn.Module, ABC):
    """Q-value function Q(s, a)."""

    @abstractmethod
    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """obs: [B, O], act: [B, A]  ->  returns Q: [B, 1]."""
        ...