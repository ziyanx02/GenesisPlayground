import genesis as gs
import numpy as np
from gs_schemas.base_types import genesis_pydantic_config
from pydantic import BaseModel

from gs_env.sim.objects.config.schema import ObjectArgs
from gs_env.sim.robots.config.schema import (
    HumanoidRobotArgs,
    ManipulatorRobotArgs,
    QuadrupedRobotArgs,
)
from gs_env.sim.scenes.config.schema import SceneArgs
from gs_env.sim.sensors.config.schema import SensorArgs


class GenesisInitArgs(BaseModel):
    model_config = genesis_pydantic_config(frozen=True)
    seed: int
    precision: str
    logging_level: str
    backend: (
        gs.constants.backend | None
    )  # While we avoid using None, this is an exception where it finds a suitable backend automatically


class EnvArgs(BaseModel):
    model_config = genesis_pydantic_config(frozen=True, arbitrary_types_allowed=True)

    env_name: str
    gs_init_args: GenesisInitArgs
    scene_args: SceneArgs
    robot_args: ManipulatorRobotArgs | QuadrupedRobotArgs | HumanoidRobotArgs
    objects_args: list[ObjectArgs]
    sensors_args: list[SensorArgs]
    reward_term: str = "reward"
    reward_args: dict[str, float] | dict[str, dict[str, float | tuple[float, float, float]]]
    img_resolution: tuple[int, int]


class LeggedRobotEnvArgs(EnvArgs):
    action_latency: int = 1
    obs_history_len: int = 1
    obs_scales: dict[str, float]
    obs_noises: dict[str, float]
    actor_obs_terms: list[str]
    critic_obs_terms: list[str]
    reset_roll_range: tuple[float, float] = (-0.15, 0.15)
    reset_pitch_range: tuple[float, float] = (-0.15, 0.15)
    reset_yaw_range: tuple[float, float] = (-np.pi, np.pi)
    reset_dof_pos_range: tuple[float, float] = (-0.15, 0.15)
    terminate_after_collision_on: list[str]


class WalkingEnvArgs(LeggedRobotEnvArgs):
    command_resample_time: float = 10.0
    commands_range: tuple[tuple[float, float], ...]


class MotionEnvArgs(LeggedRobotEnvArgs):
    motion_file: str
    no_terminate_before_motion_time: float = 1.0
    terminate_after_base_pos_error: float = 0.5
    terminate_after_base_height_error: float = 0.15
    terminate_after_base_rot_error: float = 0.3
    terminate_after_dof_pos_error: float = 8.0


class ManipulationEnvArgs(EnvArgs):
    """Configuration for manipulation environments (e.g., in-hand rotation)."""
    action_latency: int = 1
    obs_history_len: int = 1
    obs_scales: dict[str, float]
    obs_noises: dict[str, float]
    actor_obs_terms: list[str]
    critic_obs_terms: list[str]
    cube_args: dict[str, float | tuple[float, float, float]]
