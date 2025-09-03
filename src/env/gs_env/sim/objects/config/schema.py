from typing import TypeAlias

import genesis as gs
from gs_schemas.base_types import genesis_pydantic_config
from pydantic.dataclasses import dataclass

from gs_env.common.utils.gs_utils import add_to_gs_method


@add_to_gs_method(gs.materials.Rigid)
@dataclass(config=genesis_pydantic_config(frozen=True))
class RigidMaterialArgs:
    rho: float
    friction: float | None
    needs_coup: bool
    coup_friction: float
    coup_softness: float
    coup_restitution: float
    sdf_cell_size: float
    sdf_min_res: int
    sdf_max_res: int
    gravity_compensation: float


@dataclass(config=genesis_pydantic_config(frozen=True))
class PrimitiveMorphArgs:
    pos: tuple[float, float, float]
    euler: tuple[int, int, int]
    quat: tuple | None
    visualization: bool
    collision: bool
    requires_jac_and_IK: bool
    is_free: bool

    fixed: bool


@add_to_gs_method(gs.morphs.Box)
@dataclass(config=genesis_pydantic_config(frozen=True))
class BoxMorphArgs(PrimitiveMorphArgs):
    lower: tuple | None
    upper: tuple | None
    size: tuple | None


@add_to_gs_method(gs.morphs.Cylinder)
@dataclass(config=genesis_pydantic_config(frozen=True))
class CylinderMorphArgs(PrimitiveMorphArgs):
    height: float
    radius: float


@add_to_gs_method(gs.morphs.Sphere)
@dataclass(config=genesis_pydantic_config(frozen=True))
class SphereMorphArgs(PrimitiveMorphArgs):
    radius: float


@dataclass(config=genesis_pydantic_config(frozen=True))
class PrimitiveObjectArgs:
    material_args: RigidMaterialArgs
    morph_args: PrimitiveMorphArgs
    visualize_contact: bool
    vis_mode: str


@dataclass(config=genesis_pydantic_config(frozen=True))
class MeshObjectArgs:
    file: str
    up: tuple[int, int, int]
    front: tuple[int, int, int]
    scale: float
    coacd_options: gs.options.misc.CoacdOptions


@dataclass(config=genesis_pydantic_config(frozen=True))
class PartNetMobilityObjectArgs(MeshObjectArgs): ...  # TODO


@dataclass(config=genesis_pydantic_config(frozen=True))
class ObjaverseObjectArgs: ...  # TODO


@dataclass(config=genesis_pydantic_config(frozen=True))
class BlenderkitObjectArgs: ...  # TODO


ObjectArgs: TypeAlias = (
    PrimitiveObjectArgs
    | MeshObjectArgs
    | PartNetMobilityObjectArgs
    | ObjaverseObjectArgs
    | BlenderkitObjectArgs
)
