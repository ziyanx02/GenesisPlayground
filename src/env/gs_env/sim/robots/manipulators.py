from collections.abc import Callable
from typing import Any

import genesis as gs
from gymnasium import spaces
import torch
from gs_env.common.bases.base_robot import BaseGymRobot
from gs_env.common.utils.gs_utils import to_gs_and_assert
from gs_env.common.utils.math_utils import quat_from_angle_axis, quat_mul, quat_mul
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
        material: gs.materials.Rigid = to_gs_and_assert(args.material_args, gs.materials.Rigid)
        morph: gs.morphs.URDF = to_gs_and_assert(args.morph_args, gs.morphs.URDF)
        
        #
        self._robot = scene.add_entity(
            material=material,
            morph=morph,
            visualize_contact=args.visualize_contact,
            vis_mode=args.vis_mode,
        )

        # == action space ==
        n_dof = len(args.default_arm_dof.keys())
        action_space_dict = {"gripper_width": spaces.Box(0.0, 0.08)}
        match args.ctrl_type:
            case CtrlType.JOINT_POSITION:
                action_space_dict.update({"joint_pos": spaces.Box(shape=(n_dof,), low=-1.0, high=1.0)})
            case CtrlType.EE_POSE_ABS:
                action_space_dict.update(
                    {
                        "ee_link_pos": spaces.Box(shape=(3,), low=-1.0, high=1.0),
                        "ee_link_quat": spaces.Box(shape=(4,), low=-1.0, high=1.0),
                    }
                )
            case CtrlType.EE_POSE_REL:
                action_space_dict.update(
                    {
                        "ee_link_pos_delta": spaces.Box(shape=(3,), low=-1.0, high=1.0),
                        "ee_link_ang_delta": spaces.Box(shape=(3,), low=-1.0, high=1.0),  # roll, pitch, yaw
                    }
                )
            case _:  # type: ignore
                raise ValueError(f"Unknown control type: {args.ctrl_type}")

        # == some buffer initialization ==
        self._init()

    def _init(self) -> None:
        self._arm_dof_dim = len(self._args.default_arm_dof.keys())  # total number of joints
        self._gripper_dim = 0
        if self._args.default_gripper_dof is not None:
            self._gripper_dim = len(
                self._args.default_gripper_dof.keys()
            )  # number of gripper joints

        #
        self._arm_dof_idx = torch.arange(self._arm_dof_dim, device=self._device)
        self._fingers_dof = torch.arange(
            self._arm_dof_dim,
            self._arm_dof_dim + self._gripper_dim,
            device=self._device,
        )

        #
        self._ee_link = self._robot.get_link(self._args.ee_link_name)
        self._left_finger_link = self._robot.get_link(self._args.gripper_link_names[0])
        self._right_finger_link = self._robot.get_link(self._args.gripper_link_names[1])
        #
        self._default_joint_angles = list(self._args.default_arm_dof.values())
        if self._args.default_gripper_dof is not None:
            self._default_joint_angles += list(self._args.default_gripper_dof.values())

        # == set up control dispatch ==
        self._dispatch = {
            CtrlType.JOINT_POSITION.value: self._apply_joint_pos,
            CtrlType.EE_POSE_ABS.value: self._apply_ee_pose_abs,
            CtrlType.EE_POSE_REL.value: self._apply_ee_pose_rel,
        }

    def reset(self, envs_idx: torch.IntTensor | None = None) -> None:
        if envs_idx is None or len(envs_idx) == 0:
            return
        self.go_home(envs_idx)

    def go_home(self, envs_idx: torch.IntTensor) -> None:
        default_joint_angles = torch.tensor(
            self._default_joint_angles, dtype=torch.float32, device=self._device
        ).repeat(len(envs_idx), 1)
        self._robot.set_qpos(default_joint_angles, envs_idx=envs_idx)

    def apply_action(self, action: JointPosAction | EEPoseAbsAction | EEPoseRelAction | torch.Tensor) -> None:
        """
        Apply the action to the robot.
        """
        if isinstance(action, torch.Tensor):
            match self.ctrl_type:
                case CtrlType.JOINT_POSITION:
                    action = JointPosAction(joint_pos=action, gripper_width=0.0)
                case CtrlType.EE_POSE_ABS:
                    action = EEPoseAbsAction(
                        ee_link_pos=action[:, :3],
                        ee_link_quat=action[:, 3:7],
                        gripper_width=0.0,
                    )
                case CtrlType.EE_POSE_REL:
                    action = EEPoseRelAction(
                        ee_link_pos=action[:, :3],
                        ee_link_quat=action[:, 3:7],
                        gripper_width=0.0,
                    )
                case _:
                    raise ValueError(f"Unsupported control type: {self.ctrl_type}")
        self._dispatch[self._args.ctrl_type](action)

    def _apply_joint_pos(self, act: JointPosAction) -> None:
        """
        Apply joint position control to the robot.
        """
        assert act.joint_pos.shape == (
            self._num_envs,
            self._arm_dof_dim,
        ), "Joint position action must match the number of joints."
        q_target = act.joint_pos.to(self._device)
        self._robot.control_dofs_position(position=q_target) 

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

        q_pos = self._robot.inverse_kinematics(
            link=self._ee_link,
            pos=target_pos,
            quat=target_quat,
            dofs_idx_local=self._arm_dof_idx,
            max_samples=10,  # number of IK samples
            max_solver_iters=20,  # maximum solver iterations
        )
        self._robot.control_dofs_position(position=q_pos)

    def _apply_ee_pose_rel(self, act: EEPoseRelAction) -> None:
        """
        Apply relative end-effector pose control to the robot.
        """
        assert act.ee_link_pos.shape == (
            self._num_envs,
            3,
        ), "End-effector position delta must be a 3D vector."
        assert act.ee_link_quat.shape == (
            self._num_envs,
            3,
        ), "End-effector angle delta must be a 3D vector."

        current_pos = self._ee_link.get_pos()
        current_quat = self._ee_link.get_quat()

        target_pos = current_pos + act.ee_link_pos.to(self._device)
        target_quat = quat_mul(
            quat_from_angle_axis(
                torch.linalg.vector_norm(act.ee_link_quat, dim=-1),
                act.ee_link_quat
                / (torch.linalg.vector_norm(act.ee_link_ang_delta, dim=-1).unsqueeze(-1) + 1e-6),
            ),
            current_quat,
        )

        q_pos = self._robot.inverse_kinematics(
            link=self._ee_link,
            pos=target_pos,
            quat=target_quat,
            dofs_idx_local=self._arm_dof_idx,
            max_samples=10,
            max_solver_iters=20,
        )
        q_pos[:, self._fingers_dof] = torch.tensor(
            [act.gripper_width, -act.gripper_width], device=self._device
        )
        self._robot.control_dofs_position(position=q_pos)

    @property
    def base_pos(self) -> torch.Tensor:
        return self._robot.get_pos()

    @property
    def ee_pose(self) -> torch.Tensor:
        pos, quat = self._ee_link.get_pos(), self._ee_link.get_quat()
        return torch.cat([pos, quat], dim=-1)

    @property
    def left_finger_pose(self) -> torch.Tensor:
        pos, quat = self._left_finger_link.get_pos(), self._left_finger_link.get_quat()
        return torch.cat([pos, quat], dim=-1)

    @property
    def right_finger_pose(self) -> torch.Tensor:
        pos, quat = (
            self._right_finger_link.get_pos(),
            self._right_finger_link.get_quat(),
        )
        return torch.cat([pos, quat], dim=-1)

    def __getattr__(self, item: str) -> Any:
        if hasattr(self._robot, item):
            return getattr(self._robot, item)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{item}'")

    @property
    def ctrl_type(self) -> CtrlType:
        return self._args.ctrl_type


class PiperRobot(ManipulatorBase):
    def __init__(
        self, num_envs: int, scene: gs.Scene, args: ManipulatorRobotArgs, device: torch.device
    ) -> None:
        super().__init__(num_envs=num_envs, scene=scene, args=args, device=device)
