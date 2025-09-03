from abc import ABC, abstractmethod

from typing import TypeVar, Generic

TTransition = TypeVar("TTransition")

class BaseBuffer(ABC, Generic[TTransition]):
    """
    Abstract base class for on-policy and imitation learning buffers.
    """

    def __init__(self) -> None:
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset the buffer state."""
        ...

    @abstractmethod
    def append(self, transition: TTransition) -> None:
        """Append a transition to the buffer."""
        ...

    @abstractmethod
    def is_full(self) -> bool:
        """Check if the buffer is full."""
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of data points stored."""
        ...
