import abc

import torch

import gs_env.common.bases.spaces as spaces


class BaseSensor(abc.ABC):
    """
    Base class for all sensors
    """

    _observation_space_dict: dict[str, spaces.Space]

    @abc.abstractmethod
    def get_observation(self, envs_idx: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Get the observation from the sensor.

        Args:
            envs_idx (torch.IntTensor): Indices of environments in the batch to get observation.
        """

    @abc.abstractmethod
    def initialize(self) -> None:
        """Initialize the sensor."""

    @property
    def observation_space_dict(self) -> dict[str, spaces.Space]:
        """
        Get the observation space dictionary.
        """
        return self._observation_space_dict


class BaseCamera(BaseSensor, abc.ABC):
    """
    Base class for all camera devices.
    """

    def __init__(self, silent: bool = True) -> None:
        """
        Initialize the camera device.
        :param silent: If True, suppress output.
        """
        super().__init__()

        self._silent = silent

    def initialize(self) -> None:
        pass

    @abc.abstractmethod
    def start(self) -> None:
        """
        Start the camera device.
        """
        pass

    @abc.abstractmethod
    def stop(self) -> None:
        """
        Stop the camera device.
        """
        pass

    @abc.abstractmethod
    def get_frame(self) -> dict:
        """
        Get a frame from the camera device.
        """
        pass

    @property
    @abc.abstractmethod
    def resolution(self) -> tuple[int, int]:
        """
        Get the resolution of the camera device.
        """
        pass

    @property
    @abc.abstractmethod
    def fps(self) -> int:
        """
        Get the frames per second of the camera device.
        """
        pass

    @property
    @abc.abstractmethod
    def intrinsics(self) -> torch.Tensor:
        """
        Get the camera intrinsics.
        """
        pass

    def get_observation(self, envs_idx: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.get_frame()
