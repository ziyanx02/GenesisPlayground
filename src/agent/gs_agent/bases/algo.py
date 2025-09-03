from abc import ABC, abstractmethod
from typing import Any
from typing import Final
from gs_agent.bases.env_wrapper import BaseEnvWrapper
from gs_agent.bases.policy import Policy

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
    def train_one_episode(self) -> dict:
        """
        Train the algorithm for a given episode.

        Returns:
            A dictionary containing the training results.
        """
        ...
    
    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save the algorithm to a file.
        """
        ...
    
    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load the algorithm from a file.
        """
        ...
        
    @abstractmethod
    def train_mode(self) -> None:
        """
        Set the algorithm to train mode.
        """
        ...
    
    @abstractmethod
    def eval_mode(self) -> None:
        """
        Set the algorithm to eval mode.
        """
        ... 

    @abstractmethod
    def get_inference_policy(self) -> Policy:
        """
        Get the inference policy for evaluation.
        """
        ...