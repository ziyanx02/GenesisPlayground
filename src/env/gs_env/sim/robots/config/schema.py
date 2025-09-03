from __future__ import annotations

from typing import TypeAlias

import genesis as gs
import torch
from gs_schemas.base_types import GenesisEnum, genesis_pydantic_config
from pydantic import field_validator
from pydantic.dataclasses import dataclass

from gs_env.common.utils.asset_utils import get_urdf_path


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
class URDFMorphArgs:
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
    JOINT_VELOCITY = "JOINT_VELOCITY"
    JOINT_FORCE = "JOINT_FORCE"
    #
    EE_POSE_ABS = "EE_POSE_ABS"  # Absolute pose
    EE_POSE_REL = "EE_POSE_REL"  # Delta pose from current EE pose
    EE_VELOCITY = "EE_VELOCITY"
    #
    IMPEDANCE = "IMPEDANCE"
    HYBRID = "HYBRID"


class IKSolver(GenesisEnum):
    GS = "GS"  # Genesis Solver
    PIN = "PIN"  # Pinocchio Solver


@dataclass(config=genesis_pydantic_config(frozen=True))
class ManipulatorRobotArgs:
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


@dataclass(config=genesis_pydantic_config(frozen=True))
class QuadrotorRobotArgs: ...


@dataclass(config=genesis_pydantic_config(frozen=True))
class QuadrupedRobotArgs: ...


@dataclass(config=genesis_pydantic_config(frozen=True))
class HumanoidRobotArgs: ...


RobotArgs: TypeAlias = (
    ManipulatorRobotArgs | QuadrotorRobotArgs | QuadrupedRobotArgs | HumanoidRobotArgs
)


@dataclass(config=genesis_pydantic_config(frozen=True))
class JointPosAction:
    """
    Joint position action.
    """

    gripper_width: float
    joint_pos: torch.Tensor  # (n_dof,)


@dataclass(config=genesis_pydantic_config(frozen=True))
class JointVelAction:
    """
    Joint velocity action.
    """

    gripper_width: float
    joint_vel: torch.Tensor


@dataclass(config=genesis_pydantic_config(frozen=True))
class JointTorqueAction:
    """
    Joint torque action.
    """

    gripper_width: float
    joint_force: torch.Tensor  # (n_dof,)


@dataclass(config=genesis_pydantic_config(frozen=True))
class EEPoseAbsAction:
    """
    End-effector pose absolute action.
    """

    gripper_width: float
    ee_link_pos: torch.Tensor  # (3,)
    ee_link_quat: torch.Tensor  # (4,)


@dataclass(config=genesis_pydantic_config(frozen=True))
class EEPoseRelAction:
    """
    End-effector pose relative action.
    """

    gripper_width: float
    ee_link_pos_delta: torch.Tensor  # (3,)
    ee_link_ang_delta: torch.Tensor  # (3,)  # rpy


BaseAction: TypeAlias = (
    JointPosAction | JointVelAction | JointTorqueAction | EEPoseAbsAction | EEPoseRelAction
)
