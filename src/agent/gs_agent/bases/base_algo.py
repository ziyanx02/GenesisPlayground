from abc import ABC, abstractmethod
from typing import Any
from typing import Final
from gs_agent.bases.base_env_wrapper import BaseEnvWrapper

import torch

class BaseAlgo(ABC):
    """
    Base class for all algorithms in the Genesis Playground system.
    
    This abstract base class defines the common interface that all algorithms
    must implement, ensuring consistency across different algorithm types.
    """
    
    def __init__(self, env: BaseEnvWrapper, cfg: Any, device: torch.device) -> None:
        """
        Initialize the base algorithm.
        
        Args:
            env: The environment to train the algorithm on.
            cfg: The configuration for the algorithm.
            device: The device to train the algorithm on.
        """
        self.env: Final[BaseEnvWrapper] = env
        self.cfg: Final[Any] = cfg
        self.device: Final[torch.device] = device
     
        
    @abstractmethod
    def train(self, num_iters: int, log_dir: str | None = None) -> None:
        """
        Train the algorithm for a given number of iterations.
        """
        ...
    