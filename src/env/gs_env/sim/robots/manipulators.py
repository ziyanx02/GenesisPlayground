from typing import Any

import genesis as gs
import torch
import numpy as np
from genesis.engine.entities.rigid_entity import RigidEntity, RigidLink
from gymnasium import spaces

from gs_env.common.bases.base_robot import BaseGymRobot
from gs_env.sim.robots.config.schema import (
    CtrlType,
    EEPoseAbsAction,
    EEPoseRelAction,
    JointPosAction,
    ManipulatorRobotArgs,
)


class ManipulatorBase(BaseGymRobot):
    def __init__(
        self,
        num_envs: int,
        scene: gs.Scene,
        args: ManipulatorRobotArgs,
        device: torch.device,
    ) -> None:
        super().__init__()
        # == set members ==
        self._device = device
        self._scene = scene
        self._num_envs = num_envs
        self._args = args

        # == Genesis configurations ==
        material: gs.materials.Rigid = gs.materials.Rigid()
        morph: gs.morphs.URDF = gs.morphs.URDF(
            file=args.morph_args.file,
            pos=args.morph_args.pos,
            euler=args.morph_args.euler,
            fixed=True,
        )
        #
        robot_entity = scene.add_entity(
            material=material,
            morph=morph,
            visualize_contact=args.visualize_contact,
            vis_mode=args.vis_mode,
        )
        assert isinstance(robot_entity, RigidEntity), (
            "Robot entity must be an instance of gs.Entity"
        )
        self._robot_entity = robot_entity

        # == action space ==
        n_dof = len(args.default_arm_dof.keys()) + len(args.default_gripper_dof.keys())
        action_space_dict: dict[str, spaces.Space[Any]] = {}
        match args.ctrl_type:
            case CtrlType.JOINT_POSITION:
                action_space_dict.update(
                    {"joint_pos": spaces.Box(shape=(n_dof,), low=-np.inf, high=np.inf)}
                )
            case CtrlType.EE_POSE_ABS:
                action_space_dict.update(
                    {
                        "ee_link_pos": spaces.Box(shape=(3,), low=-np.inf, high=np.inf),
                        "ee_link_quat": spaces.Box(shape=(4,), low=-np.inf, high=np.inf),
                    }
                )
            case CtrlType.EE_POSE_REL:
                action_space_dict.update(
                    {
                        "ee_link_pos_delta": spaces.Box(shape=(3,), low=-np.inf, high=np.inf),
                        "ee_link_ang_delta": spaces.Box(
                            shape=(3,), low=-np.inf, high=np.inf
                        ),  # roll, pitch, yaw
                    }
                )
            case CtrlType.DR_JOINT_POSITION:
                raise RuntimeError("DR control not supported for manipulator.")

        self._action_space = spaces.Dict(action_space_dict)

        # == some buffer initialization ==
        self._init()

    def _init(self) -> None:
        self._arm_dof_dim = len(self._args.default_arm_dof.keys())  # total number of joints
        self._gripper_dof_dim = 0
        if self._args.default_gripper_dof is not None:
            self._gripper_dof_dim = len(
                self._args.default_gripper_dof.keys()
            )  # number of gripper joints
        self._dof_dim = self._arm_dof_dim + self._gripper_dof_dim

        all_dof_names = list(self._args.default_arm_dof.keys())
        if self._args.default_gripper_dof is not None:
            all_dof_names += list(self._args.default_gripper_dof.keys())
        self._all_dof_names = all_dof_names

        # About torque calculation and domain randomization
        dof_kp, dof_kd = [], []
        for dof_name in all_dof_names:
            dof_kp.append(self._args.dof_kp[dof_name])
            dof_kd.append(self._args.dof_kd[dof_name])
        self._dof_kp = torch.tensor(dof_kp, device=self._device)
        self._dof_kd = torch.tensor(dof_kd, device=self._device)
        self._batched_dof_kp = self._dof_kp[None, :].repeat(self._num_envs, 1)
        self._batched_dof_kd = self._dof_kd[None, :].repeat(self._num_envs, 1)
        self._motor_strength = torch.ones((self._num_envs, self._dof_dim), device=self._device)
        self._motor_offset = torch.zeros((self._num_envs, self._dof_dim), device=self._device)

        default_dof_pos = list(self._args.default_arm_dof.values())
        if self._args.default_gripper_dof is not None:
            default_dof_pos += list(self._args.default_gripper_dof.values())
        self._default_dof_pos = torch.tensor(default_dof_pos, dtype=torch.float32, device=self._device)

        # Buffers
        self._dof_pos = torch.zeros((self._num_envs, self._dof_dim), dtype=torch.float32, device=self._device)
        self._dof_vel = torch.zeros((self._num_envs, self._dof_dim), dtype=torch.float32, device=self._device)
        self._torque = torch.zeros((self._num_envs, self._dof_dim), device=self._device)

        # == set up control dispatch ==
        self._dispatch = {
            CtrlType.JOINT_POSITION.value: self._apply_joint_pos,
            CtrlType.EE_POSE_ABS.value: self._apply_ee_pose_abs,
            CtrlType.EE_POSE_REL.value: self._apply_ee_pose_rel,
        }

    def post_build_init(self, eval_mode: bool = False) -> None:
        """Initialize limits and constraints after scene is built."""
        # Get DOF indices for all joints
        all_dofs_idx_local = [
            self._robot_entity.get_joint(name).dofs_idx_local[0] for name in self._all_dof_names
        ]
        self._all_dof_idx_local = all_dofs_idx_local
        all_finger_links_idx_local = [
            self._robot_entity.get_link(name).idx_local for name in self._args.gripper_link_names
        ]
        self._all_finger_links_idx_local = all_finger_links_idx_local

        # Set up DOF position limits
        self._dof_pos_limits = torch.stack(
            self._robot_entity.get_dofs_limit(all_dofs_idx_local), dim=1
        )

        # Apply soft limits (reduce range to avoid hitting hard limits)
        soft_range = self._args.soft_dof_pos_range
        for i in range(self._dof_pos_limits.shape[0]):
            # Calculate midpoint and range
            m = (self._dof_pos_limits[i, 0] + self._dof_pos_limits[i, 1]) / 2
            r = self._dof_pos_limits[i, 1] - self._dof_pos_limits[i, 0]
            # Apply soft limits
            self._dof_pos_limits[i, 0] = m - 0.5 * r * soft_range
            self._dof_pos_limits[i, 1] = m + 0.5 * r * soft_range

        # Set up torque limits
        self._torque_limits = self._robot_entity.get_dofs_force_range(all_dofs_idx_local)[1]

    def reset(self, envs_idx: torch.IntTensor | None = None) -> None:
        if envs_idx is None or len(envs_idx) == 0:
            return
        self.go_home(envs_idx)

    def go_home(self, envs_idx: torch.IntTensor) -> None:
        self._robot_entity.set_dofs_position(
            self._default_dof_pos.repeat(len(envs_idx), 1),
            envs_idx=envs_idx,
            dofs_idx_local=self._all_dof_idx_local,
            zero_velocity=True,
        )
        self._dof_pos[envs_idx] = self._default_dof_pos[None, :].repeat(len(envs_idx), 1)
        self._dof_vel[envs_idx] = 0.0
        self._torque[envs_idx] = 0.0

    def reset_to_pose(self, joint_positions: torch.Tensor) -> None:
        """Reset robot to a specific joint configuration."""

        # Ensure the tensor has the right shape (batch_size, num_joints)
        if joint_positions.dim() == 1:
            joint_positions = joint_positions.unsqueeze(0)

        # If only arm joints are provided, pad with default gripper values
        if joint_positions.shape[1] < len(self._default_dof_pos):
            arm_joints = joint_positions
            # Get gripper joint values, handling potential duplicates
            gripper_values = (
                list(self._args.default_gripper_dof.values())
                if self._args.default_gripper_dof
                else []
            )
            # Remove duplicates while preserving order
            gripper_values = list(dict.fromkeys(gripper_values))
            gripper_joints = (
                torch.tensor(gripper_values, dtype=torch.float32, device=self._device)
                .unsqueeze(0)
                .repeat(joint_positions.shape[0], 1)
            )
            joint_positions = torch.cat([arm_joints, gripper_joints], dim=1)

        # Create envs_idx for all environments
        envs_idx = torch.arange(joint_positions.shape[0], device=self._device)
        self._robot_entity.set_qpos(joint_positions, envs_idx=envs_idx)

    def apply_action(
        self, action: JointPosAction | EEPoseAbsAction | EEPoseRelAction | torch.Tensor
    ) -> None:
        """
        Apply the action to the robot.
        """
        if isinstance(action, torch.Tensor):
            match self.ctrl_type:
                case CtrlType.JOINT_POSITION:
                    action = JointPosAction(joint_pos=action, gripper_width=0.0)
                case CtrlType.EE_POSE_ABS:
                    action = EEPoseAbsAction(
                        ee_link_pos=action[..., :3],
                        ee_link_quat=action[..., 3:7],
                        gripper_width=0.0,
                    )
                case CtrlType.EE_POSE_REL:
                    action = EEPoseRelAction(
                        ee_link_pos_delta=action[..., :3],
                        ee_link_ang_delta=action[..., 3:6],
                        gripper_width=0.0,
                    )
                case CtrlType.DR_JOINT_POSITION:
                    raise RuntimeError("DR control not supported for manipulator.")
        self._dispatch[self._args.ctrl_type](action)
        self._torque[:] = self._batched_dof_kp * (action.joint_pos - self._dof_pos + self._motor_offset) - self._batched_dof_kd * self._dof_vel
        self._dof_pos[:] = self._robot_entity.get_dofs_position(self._all_dof_idx_local)
        self._dof_vel[:] = self._robot_entity.get_dofs_velocity(self._all_dof_idx_local)

    def _apply_joint_pos(self, act: JointPosAction) -> None:
        """
        Apply joint position control to the robot.
        """
        assert act.joint_pos.shape == (
            self._num_envs,
            self._dof_dim,
        ), "Joint position action must match the number of joints."
        self._robot_entity.control_dofs_position(position=act.joint_pos)

    def _apply_ee_pose_abs(self, act: EEPoseAbsAction) -> None:
        """
        Apply end-effector pose control to the robot.
        """
        assert act.ee_link_pos.shape == (
            self._num_envs,
            3,
        ), "End-effector position must be a 3D vector."
        assert act.ee_link_quat.shape == (
            self._num_envs,
            4,
        ), "End-effector quaternion must be a 4D vector."

        target_pos = act.ee_link_pos.to(self._device)
        target_quat = act.ee_link_quat.to(self._device)

        q_pos = self._robot_entity.inverse_kinematics(
            link=self._ee_link,
            pos=target_pos,
            quat=target_quat,
            dofs_idx_local=self._arm_dof_idx,
            max_samples=10,  # number of IK samples
            max_solver_iters=20,  # maximum solver iterations
        )
        if isinstance(q_pos, torch.Tensor):
            q_pos[:, self._fingers_dof_idx] = torch.tensor(
                [act.gripper_width, act.gripper_width], device=self._device
            )
            self._robot_entity.control_dofs_position(position=q_pos)

    def _apply_ee_pose_rel(self, act: EEPoseRelAction) -> None:
        """
        Apply relative end-effector pose control to the robot.
        """
        assert act.ee_link_pos_delta.shape == (
            self._num_envs,
            3,
        ), "End-effector position delta must be a 3D vector."
        assert act.ee_link_ang_delta.shape == (
            self._num_envs,
            3,
        ), "End-effector angle delta must be a 3D vector."
        q_pos = self._dls_ik(act)
        # set gripper width
        q_pos[:, self._fingers_dof_idx] = torch.tensor(
            [act.gripper_width, -act.gripper_width], device=self._device
        )
        self._robot_entity.control_dofs_position(position=q_pos)

    def _dls_ik(self, action: EEPoseRelAction) -> torch.Tensor:
        """
        Damped least squares inverse kinematics
        """
        delta_pose = torch.cat([action.ee_link_pos_delta, action.ee_link_ang_delta], dim=-1)
        lambda_val = 0.01
        jacobian = self._robot_entity.get_jacobian(link=self._ee_link)
        jacobian_T = jacobian.transpose(1, 2)
        lambda_matrix = (lambda_val**2) * torch.eye(n=jacobian.shape[1], device=self._device)
        delta_joint_pos = (
            jacobian_T
            @ torch.inverse(jacobian @ jacobian_T + lambda_matrix)
            @ delta_pose.unsqueeze(-1)
        ).squeeze(-1)
        return self._robot_entity.get_qpos() + delta_joint_pos

    @property
    def base_pos(self) -> torch.Tensor:
        return self._robot_entity.get_pos()

    @property
    def torque(self) -> torch.Tensor:
        return self._torque

    @property
    def dof_pos(self) -> torch.Tensor:
        return self._dof_pos

    @property
    def dof_vel(self) -> torch.Tensor:
        return self._dof_vel
    
    @property
    def default_dof_pos(self) -> torch.Tensor:
        return self._default_dof_pos
    
    @property
    def dof_names(self) -> list[str]:
        return self._all_dof_names
    
    @property
    def fingertip_pos(self) -> torch.Tensor:
        return self._robot_entity.get_links_pos(links_idx_local=self._all_finger_links_idx_local)

    def __getattr__(self, item: str) -> Any:
        # Use object.__getattribute__ to avoid recursion with hasattr
        try:
            return object.__getattribute__(self._robot_entity, item)
        except AttributeError as err:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{item}'"
            ) from err

    @property
    def ctrl_type(self) -> CtrlType:
        return self._args.ctrl_type

    @property
    def dof_pos_limits(self) -> torch.Tensor:
        """Get DOF position limits (soft limits)."""
        return self._dof_pos_limits

    @property
    def torque_limits(self) -> torch.Tensor:
        """Get torque limits for all DOFs."""
        return self._torque_limits


class FrankaRobot(ManipulatorBase):
    def __init__(
        self,
        num_envs: int,
        scene: gs.Scene,
        args: ManipulatorRobotArgs,
        device: torch.device,
    ) -> None:
        super().__init__(num_envs=num_envs, scene=scene, args=args, device=device)


class WUJIHand(ManipulatorBase):
    """
    WUJI Hand - 5-finger dexterous hand with 20 DOF (4 joints per finger).
    The hand base is fixed in space, only the finger joints are actuated.
    """

    def __init__(
        self,
        num_envs: int,
        scene: gs.Scene,
        args: ManipulatorRobotArgs,
        device: torch.device,
    ) -> None:
        super().__init__(num_envs=num_envs, scene=scene, args=args, device=device)

        # Store fingertip links for contact detection and manipulation tasks
        self._fingertip_links = [
            self._robot_entity.get_link(name) for name in args.gripper_link_names
        ]
        self._fingertip_links_idx = [link.idx_local for link in self._fingertip_links]

    @property
    def fingertip_links_idx(self) -> list[int]:
        """Get the indices of fingertip links for contact detection."""
        return self._fingertip_links_idx
