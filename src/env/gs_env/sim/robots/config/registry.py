from typing import TypeAlias

from gs_env.common.utils.asset_utils import get_urdf_path
from gs_env.sim.robots.config.schema import (
    CtrlType,
    IKSolver,
    ManipulatorRobotArgs,
    RigidMaterialArgs,
    URDFMorphArgs,
)

# ------------------------------------------------------------
# Material
# ------------------------------------------------------------

MaterialArgs: TypeAlias = RigidMaterialArgs


MaterialArgsRegistry: dict[str, MaterialArgs] = {}


MaterialArgsRegistry["default"] = RigidMaterialArgs(
    rho=200.0,
    friction=None,
    needs_coup=True,
    coup_friction=0.1,
    coup_softness=0.002,
    coup_restitution=0.0,
    sdf_cell_size=0.005,
    sdf_min_res=32,
    sdf_max_res=128,
    gravity_compensation=1,
)


# ------------------------------------------------------------
# Morph
# ------------------------------------------------------------

MorphArgs: TypeAlias = URDFMorphArgs


MorphArgsRegistry: dict[str, MorphArgs] = {}


MorphArgsRegistry["piper_default"] = URDFMorphArgs(
    pos=(0.0, 0.0, 0.0),
    euler=(0, 0, 0),
    quat=None,
    visualization=True,
    collision=True,
    requires_jac_and_IK=True,
    is_free=True,
    file=get_urdf_path("piper", "piper"),
    scale=1.0,
    convexify=False,
    recompute_inertia=False,
    fixed=True,
    prioritize_urdf_material=False,
    merge_fixed_links=False,
    links_to_keep=[],
    decimate=True,
)

MorphArgsRegistry["franka_default"] = URDFMorphArgs(
    pos=(0.0, 0.0, 0.0),
    euler=(0, 0, 0),
    quat=None,
    visualization=True,
    collision=True,
    requires_jac_and_IK=True,
    is_free=True,
    file=get_urdf_path("piper", "piper"),
    scale=1.0,
    convexify=False,
    recompute_inertia=False,
    fixed=True,
    prioritize_urdf_material=False,
    merge_fixed_links=False,
    links_to_keep=[],
    decimate=True,
)


# ------------------------------------------------------------
# Robot
# ------------------------------------------------------------


RobotArgsRegistry: dict[str, ManipulatorRobotArgs] = {}

RobotArgsRegistry["piper_default"] = ManipulatorRobotArgs(
    material_args=MaterialArgsRegistry["default"],
    morph_args=MorphArgsRegistry["piper_default"],
    visualize_contact=False,
    vis_mode="visual",
    ctrl_type=CtrlType.EE_POSE_REL,
    ik_solver=IKSolver.GS,  # TODO: need to be aligned with real
    ee_link_name="gripper_base",  # or "gripper_tip"
    show_target=True,
    gripper_link_names=[
        "link7",
        "link8",
    ],
    default_arm_dof={
        "joint1": 0.0,
        "joint2": 0.0,
        "joint3": 0.0,
        "joint4": -2.27,
        "joint5": 0.0,
        "joint6": 2.27,
    },
    default_gripper_dof={
        "joint7": 0.03,
        "joint8": -0.03,
    },  # open gripper
)
