from abc import ABC, abstractmethod


class BaseBuffer(ABC):
    """
    Abstract base class for on-policy and imitation learning buffers.
    """

    def __init__(
        self,
        num_envs: int,
        max_steps: int,
        actor_obs_size: int,
        critic_obs_size: int,
        action_size: int,
        rgb_shape: tuple[int, int, int] | None = None,
        depth_shape: tuple[int, int, int] | None = None,
        device: str = "cpu",
    ):
        self._num_envs = num_envs
        self._max_steps = max_steps
        self._actor_obs_dim = actor_obs_size
        self._critic_obs_dim = critic_obs_size
        self._action_dim = action_size
        self._rgb_shape = rgb_shape
        self._depth_shape = depth_shape
        self._device = device
        self._idx = 0

    @abstractmethod
    def reset(self):
        """Reset the buffer state."""
        pass

    @abstractmethod
    def is_full(self) -> bool:
        """Check if the buffer is full."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of data points stored."""
        pass
