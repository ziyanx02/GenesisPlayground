import numpy as np
import torch

import gs_env.common.bases.spaces as spaces

# Default Floating Point Type
NP_SCALAR = np.float32
TORCH_SCALAR = torch.float32
# Default integer type
NP_INT = np.int32
TORCH_INT = torch.int32


def GRIPPER_WIDTH_SPACE(
    low: float = 0.0, high: float = 0.08, name: str = "gripper_width"
) -> spaces.BoxWithNames:
    """
    Create a space for gripper width.
    Args:
        low (float): Lower bound for gripper width.
        high (float): Upper bound for gripper width.
        name (str): Name of the space.
    """
    return spaces.BoxWithNames(
        low=np.array([low], dtype=NP_SCALAR),
        high=np.array([high], dtype=NP_SCALAR),
        shape=(1,),
        dtype=NP_SCALAR,
        names=[name],
    )


def POSITION_SPACE(
    low: float | list[float] = -np.inf, high: float | list[float] = np.inf
) -> spaces.BoxWithNames:
    """
    Create a 3D position space.
    Args:
        low (float | list[float]): Lower bound for position coordinates.
        high (float | list[float]): Upper bound for position coordinates.
    """
    low_arr = (
        np.full(3, low, dtype=NP_SCALAR) if np.isscalar(low) else np.array(low, dtype=np.float32)
    )
    high_arr = (
        np.full(3, high, dtype=NP_SCALAR) if np.isscalar(high) else np.array(high, dtype=NP_SCALAR)
    )
    return spaces.BoxWithNames(
        low=low_arr,
        high=high_arr,
        shape=(3,),
        dtype=NP_SCALAR,
        names=["pos_x", "pos_y", "pos_z"],
    )


def LINEAR_VELOCITY_SPACE(
    low: float | list[float] = -np.inf, high: float | list[float] = np.inf
) -> spaces.BoxWithNames:
    """
    Create a 3D linear velocity space.
    Args:
        low (float | list[float]): Lower bound for linear velocity coordinates.
        high (float | list[float]): Upper bound for linear velocity coordinates.
    """
    low_arr = (
        np.full(3, low, dtype=NP_SCALAR) if np.isscalar(low) else np.array(low, dtype=np.float32)
    )
    high_arr = (
        np.full(3, high, dtype=NP_SCALAR) if np.isscalar(high) else np.array(high, dtype=NP_SCALAR)
    )
    return spaces.BoxWithNames(
        low=low_arr,
        high=high_arr,
        shape=(3,),
        dtype=NP_SCALAR,
        names=["lin_vel_x", "lin_vel_y", "lin_vel_z"],
    )


def QUATERNION_SPACE(
    low: float | list[float] = -1.0, high: float | list[float] = 1.0
) -> spaces.BoxWithNames:
    """
    Create a quaternion space for representing orientations.
    Args:
        low (float | list[float]): Lower bound for quaternion components.
        high (float | list[float]): Upper bound for quaternion components.
    """
    low_arr = (
        np.full(4, low, dtype=NP_SCALAR) if np.isscalar(low) else np.array(low, dtype=np.float32)
    )
    high_arr = (
        np.full(4, high, dtype=NP_SCALAR) if np.isscalar(high) else np.array(high, dtype=NP_SCALAR)
    )
    return spaces.BoxWithNames(
        low=low_arr,
        high=high_arr,
        shape=(4,),
        dtype=NP_SCALAR,
        names=["quat_w", "quat_x", "quat_y", "quat_z"],
    )


def RPY_SPACE(
    low: float | list[float] = -np.inf, high: float | list[float] = np.inf
) -> spaces.BoxWithNames:
    """
    Create a 3D roll-pitch-yaw (RPY) space.
    Args:
        low (float | list[float]): Lower bound for angular velocity coordinates.
        high (float | list[float]): Upper bound for angular velocity coordinates.
    """
    low_arr = (
        np.full(3, low, dtype=NP_SCALAR) if np.isscalar(low) else np.array(low, dtype=np.float32)
    )
    high_arr = (
        np.full(3, high, dtype=NP_SCALAR) if np.isscalar(high) else np.array(high, dtype=NP_SCALAR)
    )
    return spaces.BoxWithNames(
        low=low_arr,
        high=high_arr,
        shape=(3,),
        dtype=NP_SCALAR,
        names=["roll_x", "pitch_y", "yaw_z"],
    )


def ANGULAR_VELOCITY_SPACE(
    low: float | list[float] = -np.inf, high: float | list[float] = np.inf
) -> spaces.BoxWithNames:
    """
    Create a 3D angular velocity space.
    Args:
        low (float | list[float]): Lower bound for angular velocity coordinates.
        high (float | list[float]): Upper bound for angular velocity coordinates.
    """
    low_arr = (
        np.full(3, low, dtype=NP_SCALAR) if np.isscalar(low) else np.array(low, dtype=np.float32)
    )
    high_arr = (
        np.full(3, high, dtype=NP_SCALAR) if np.isscalar(high) else np.array(high, dtype=NP_SCALAR)
    )
    return spaces.BoxWithNames(
        low=low_arr,
        high=high_arr,
        shape=(3,),
        dtype=NP_SCALAR,
        names=["ang_vel_x", "ang_vel_y", "ang_vel_z"],
    )


def JOINT_SPACE(
    n_dof: int,
    low: np.ndarray | float = -np.inf,
    high: np.ndarray | float = np.inf,
    names: list[str] | None = None,
) -> spaces.BoxWithNames:
    """
    Create a joint position DOF space.
    Args:
        n_dof (int): Number of degrees of freedom.
        low (float | np.ndarray): Lower bound for joint positions.
        high (float | np.ndarray): Upper bound for joint positions.
        names (list[str] | None): Optional names for each DOF.
    """
    assert n_dof > 0, "DOF must be greater than 0"

    # Handle scalar or vector bounds
    low_array = (
        np.full(n_dof, low, dtype=NP_SCALAR) if np.isscalar(low) else np.array(low, dtype=NP_SCALAR)
    )
    high_array = (
        np.full(n_dof, high, dtype=NP_SCALAR)
        if np.isscalar(high)
        else np.array(high, dtype=NP_SCALAR)
    )

    assert low_array.shape == (n_dof,), f"low must be shape ({n_dof},), got {low_array.shape}"
    assert high_array.shape == (n_dof,), f"high must be shape ({n_dof},), got {high_array.shape}"

    # Default joint names
    if names is None:
        names = [f"joint_{i}" for i in range(n_dof)]

    return spaces.BoxWithNames(
        low=low_array,
        high=high_array,
        shape=(n_dof,),
        dtype=NP_SCALAR,
        names=names,
    )


def JOINT_POS_SPACE(
    n_dof: int,
    low: np.ndarray | float = -np.inf,
    high: np.ndarray | float = np.inf,
    names: list[str] | None = None,
) -> spaces.BoxWithNames:
    """
    Create a joint position DOF space.

    Args:
        n_dof (int): Number of degrees of freedom.
        low (float | np.ndarray): Lower bound for joint positions.
        high (float | np.ndarray): Upper bound for joint positions.
        names (list[str] | None): Optional names for each DOF.

    Returns:
        spaces.Box: A Box space representing the joint position DOF space.
    """
    if names is None:
        names = [f"joint_pos_{i}" for i in range(n_dof)]
    return JOINT_SPACE(n_dof, low, high, names)


def JOINT_VEL_SPACE(
    n_dof: int,
    low: np.ndarray | float = -np.inf,
    high: np.ndarray | float = np.inf,
    names: list[str] | None = None,
) -> spaces.BoxWithNames:
    """
    Create a joint velocity DOF space.

    Args:
        n_dof (int): Number of degrees of freedom.
        low (float | np.ndarray): Lower bound for joint velocities.
        high (float | np.ndarray): Upper bound for joint velocities.
        names (list[str] | None): Optional names for each DOF.

    Returns:
        spaces.Box: A Box space representing the joint velocity DOF space.
    """
    if names is None:
        names = [f"joint_vel_{i}" for i in range(n_dof)]
    return JOINT_SPACE(n_dof, low, high, names)


def JOINT_TORQUE_SPACE(
    n_dof: int,
    low: np.ndarray | float = -np.inf,
    high: np.ndarray | float = np.inf,
    names: list[str] | None = None,
) -> spaces.BoxWithNames:
    """
    Create a joint torque DOF space.

    Args:
        n_dof (int): Number of degrees of freedom.
        low (float | np.ndarray): Lower bound for joint torques.
        high (float | np.ndarray): Upper bound for joint torques.
        names (list[str] | None): Optional names for each DOF.

    Returns:
        spaces.Box: A Box space representing the joint torque DOF space.
    """
    if names is None:
        names = [f"joint_torque_{i}" for i in range(n_dof)]
    return JOINT_SPACE(n_dof, low, high, names)
