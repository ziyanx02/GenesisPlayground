import genesis as gs
from gs_schemas.base_types import genesis_pydantic_config
from pydantic.dataclasses import dataclass

from gs_env.sim.objects.config.schema import ObjectArgs
from gs_env.sim.robots.config.schema import RobotArgs
from gs_env.sim.scenes.config.schema import SceneArgs
from gs_env.sim.sensors.config.schema import SensorArgs


@dataclass(config=genesis_pydantic_config(frozen=True))
class GenesisInitArgs:
    seed: int
    precision: str
    logging_level: str
    backend: (
        gs.constants.backend | None
    )  # While we avoid using None, this is an exception where it finds a suitable backend automatically


@dataclass(config=genesis_pydantic_config(frozen=True))
class EnvArgs:
    gs_init_args: GenesisInitArgs
    scene_args: SceneArgs
    robot_args: RobotArgs
    objects_args: list[ObjectArgs]
    sensors_args: list[SensorArgs]
    reward_args: dict | None = None
    img_resolution: tuple[int, int] | None = None
