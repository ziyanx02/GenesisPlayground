from __future__ import annotations

from typing import TypeAlias

import genesis as gs
import torch
from gs_schemas.base_types import GenesisEnum, genesis_pydantic_config
from pydantic import field_validator, BaseModel

from gs_env.common.utils.asset_utils import get_urdf_path


class RigidMaterialArgs(BaseModel):
    model_config = genesis_pydantic_config(frozen=True)
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


class URDFMorphArgs(BaseModel):
    model_config = genesis_pydantic_config(frozen=True)
    # Morph
    pos: tuple[float, float, float]
    euler: tuple[int, int, int]
    quat: tuple[float, float, float, float] | None
    visualization: bool
    collision: bool
    requires_jac_and_IK: bool
    is_free: bool

    # FileMorph
    file: str
    scale: float | tuple[float, float, float]
    convexify: bool
    recompute_inertia: bool

    # URDF
    fixed: bool
    prioritize_urdf_material: bool
    merge_fixed_links: bool
    links_to_keep: list[str]

    decimate: bool


class CtrlType(GenesisEnum):
    # Arm Control Types
    JOINT_POSITION = "JOINT_POSITION"
    EE_POSE_ABS = "EE_POSE_ABS"  # Absolute pose
    EE_POSE_REL = "EE_POSE_REL"  # Delta pose from current EE pose

class IKSolver(GenesisEnum):
    GS = "GS"  # Genesis Solver
    PIN = "PIN"  # Pinocchio Solver



class ManipulatorRobotArgs(BaseModel):
    model_config = genesis_pydantic_config(frozen=True)

    material_args: RigidMaterialArgs
    morph_args: URDFMorphArgs
    visualize_contact: bool
    vis_mode: str
    ctrl_type: CtrlType
    ik_solver: IKSolver
    ee_link_name: str
    show_target: bool
    gripper_link_names: list[str]
    default_arm_dof: dict[str, float]
    default_gripper_dof: dict[str, float] | None = None

    @field_validator("morph_args")
    def check_morph_args(cls, morph_args: URDFMorphArgs) -> URDFMorphArgs:
        expected_file = get_urdf_path("piper", "piper")
        assert (
            morph_args.file == expected_file
        ), f"File in morph args {morph_args.file} is not consistent with the expected {expected_file}"

        return morph_args





class JointPosAction(BaseModel):
    model_config = genesis_pydantic_config(frozen=True)

    gripper_width: float
    joint_pos: torch.Tensor  # (n_dof,)


class EEPoseAbsAction(BaseModel):
    model_config = genesis_pydantic_config(frozen=True)

    gripper_width: float
    ee_link_pos: torch.Tensor  # (3,)
    ee_link_quat: torch.Tensor  # (4,)


class EEPoseRelAction(BaseModel):
    model_config = genesis_pydantic_config(frozen=True)

    gripper_width: float
    ee_link_pos: torch.Tensor  # (3,)
    ee_link_quat: torch.Tensor  # (4,)


BaseAction: TypeAlias = (
    JointPosAction | EEPoseAbsAction | EEPoseRelAction
)
