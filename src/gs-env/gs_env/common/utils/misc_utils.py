import numpy as np

#
from gs_env.common.bases import spaces
from gs_env.common.utils import gs_types
from gs_env.sim.robots.config.schema import CtrlType


def get_space_dim(space: spaces.Space) -> int:
    if isinstance(space, spaces.Box | spaces.BoxWithNames):
        return int(np.prod(space.shape))
    elif isinstance(space, spaces.Discrete):
        return 1
    elif isinstance(space, spaces.Dict):
        return sum(get_space_dim(subspace) for subspace in space.spaces.values())
    else:
        raise NotImplementedError(f"Unsupported space type: {type(space)}")


# TODO: Move this to somewhere more appropriate
def get_action_space(ctrl_type: CtrlType, n_dof: int) -> spaces.Dict:
    action_space_dict = {"gripper_width": gs_types.GRIPPER_WIDTH_SPACE(0.0, 0.08)}
    match ctrl_type:
        case CtrlType.JOINT_POSITION:
            action_space_dict.update({"joint_pos": gs_types.JOINT_POS_SPACE(n_dof, -1.0, 1.0)})
        case CtrlType.JOINT_VELOCITY:
            action_space_dict.update({"joint_vel": gs_types.JOINT_VEL_SPACE(n_dof, -1.0, 1.0)})
        case CtrlType.JOINT_FORCE:
            action_space_dict.update(
                {"joint_torque": gs_types.JOINT_TORQUE_SPACE(n_dof, -1.0, 1.0)}
            )
        case CtrlType.EE_POSE_ABS:
            action_space_dict.update(
                {
                    "ee_link_pos": gs_types.POSITION_SPACE(),
                    "ee_link_quat": gs_types.QUATERNION_SPACE(),
                }
            )
        case CtrlType.EE_POSE_REL:
            action_space_dict.update(
                {
                    "ee_link_pos_delta": gs_types.POSITION_SPACE(),
                    "ee_link_ang_delta": gs_types.RPY_SPACE(),  # roll, pitch, yaw
                }
            )
        case CtrlType.EE_VELOCITY:
            action_space_dict.update(
                {
                    "ee_link_vel": gs_types.LINEAR_VELOCITY_SPACE(-1.0, 1.0),
                    "ee_link_omega": gs_types.ANGULAR_VELOCITY_SPACE(-1.0, 1.0),
                }
            )
        case CtrlType.IMPEDANCE:
            raise NotImplementedError("Impedance control type is not implemented yet.")
        case CtrlType.HYBRID:
            raise NotImplementedError("Hybrid control type is not implemented yet.")
        case _:
            raise ValueError(f"Unknown control type: {ctrl_type}")
    return spaces.Dict(action_space_dict)
