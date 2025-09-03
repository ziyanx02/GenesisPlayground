from abc import ABC, abstractmethod
from gs_agent.bases.env_wrapper import BaseEnvWrapper
from typing import Any
import torch

class BaseRunner(ABC):
    def __init__(self):
        ...

    @abstractmethod
    def train(self, metric_logger: Any) -> dict:
        """
        Train the algorithm for a given number of episodes.
        """
        ...
         
    @abstractmethod
    def load_checkpoint(self, path: str) -> None:
        """
        Load the checkpoint of the algorithm.
        """
        ...
        
    @abstractmethod
    def get_policy(self) -> Policy:
        """
        Get the policy for evaluation.
        """
        ...