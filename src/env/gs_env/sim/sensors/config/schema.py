from typing import TypeAlias

from gs_schemas.base_types import GenesisEnum, genesis_pydantic_config
from pydantic.dataclasses import dataclass

# ------------------------------------------------------------
# Camera
# ------------------------------------------------------------


@dataclass(config=genesis_pydantic_config(frozen=True))
class OakCameraArgs:
    # aligned with real
    silent: bool
    resolution: tuple[int, int]
    fps: int
    exposure: float | None
    white_balance: float | None

    # sim-specific
    name: str
    pos: tuple[float, float, float]
    lookat: tuple[float, float, float]
    fov: int  # TODO: cannot set hfov
    # TODO: no intrinsic (maybe focus_dist but not sure about the conversion)
    GUI: bool


@dataclass(config=genesis_pydantic_config(frozen=True))
class RealSenseCameraArgs: ...


@dataclass(config=genesis_pydantic_config(frozen=True))
class ZEDCameraArgs: ...


CameraArgs: TypeAlias = OakCameraArgs | RealSenseCameraArgs | ZEDCameraArgs


# ------------------------------------------------------------
# Proprioceptive
# ------------------------------------------------------------


class ProprioceptiveSensorType(GenesisEnum):
    EE_LINK_POS = "EE_LINK_POS"
    EE_LINK_QUAT = "EE_LINK_QUAT"
    JOINT_ANGLES = "JOINT_ANGLES"
    GRIPPER_WIDTH = "GRIPPER_WIDTH"


@dataclass(config=genesis_pydantic_config(frozen=True))
class ProprioceptiveSensorArgs:
    sensor_type: ProprioceptiveSensorType


SensorArgs: TypeAlias = CameraArgs | ProprioceptiveSensorArgs
