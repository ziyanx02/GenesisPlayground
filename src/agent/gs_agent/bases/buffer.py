from abc import ABC, abstractmethod


class BaseBuffer(ABC):
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
    def is_full(self) -> bool:
        """Check if the buffer is full."""
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of data points stored."""
        ...
