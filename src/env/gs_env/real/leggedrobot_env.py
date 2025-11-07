from collections.abc import Callable

import numpy as np
import torch
from gymnasium import spaces
from transforms3d import quaternions

from gs_env.common.bases.base_robot import BaseGymRobot
from gs_env.real.unitree.utils.low_state_controller import LowStateCmdHandler
from gs_env.real.unitree.utils.low_state_handler import LowStateMsgHandler
from gs_env.sim.envs.config.schema import LeggedRobotEnvArgs
from gs_env.sim.robots.config.schema import (
    BaseAction,
    CtrlType,
    DRJointPosAction,
    JointPosAction,
)

_DEFAULT_DEVICE = torch.device("cpu")


class UnitreeLeggedEnv(BaseGymRobot):
    def __init__(
        self,
        args: LeggedRobotEnvArgs,
        action_scale: float = 0.0,
        interactive: bool = False,
        device: torch.device = _DEFAULT_DEVICE,
    ) -> None:
        super().__init__()
        self._args = args
        if interactive:
            self._robot = LowStateCmdHandler(args.robot_args)
            self._robot.init()
            self._robot.start()
        else:
            self._robot = LowStateMsgHandler(args.robot_args)
            self._robot.init()
        self._action_space = spaces.Box(shape=(self.action_dim,), low=-np.inf, high=np.inf)
        self._action_scale = args.robot_args.action_scale * action_scale
        self._device = device
        self.prev_target_pos = np.array(self.robot.default_dof_pos, dtype=np.float32)[None, :]
        self.dt = 1.0 / self._args.robot_args.ctrl_freq

        # == set up control dispatch ==
        self._dispatch: dict[CtrlType, Callable[[BaseAction], None]] = {  # type: ignore
            CtrlType.JOINT_POSITION.value: self._apply_joint_pos,
            CtrlType.DR_JOINT_POSITION.value: self._apply_dr_joint_pos,
        }

        # self._dispatch: dict[CtrlType, Callable[[BaseAction], None]] = {  # type: ignore
        #     CtrlType.JOINT_POSITION.value: self._apply_joint_pos,
        #     CtrlType.DR_JOINT_POSITION.value: self._apply_dr_joint_pos,
        # }

    def reset(self, envs_idx: torch.IntTensor | None = None) -> None:
        # TODO: implement reset to reset_pos
        pass

    def apply_action(self, action: torch.Tensor) -> None:
        if isinstance(action, torch.Tensor):
            match self._args.robot_args.ctrl_type:
                case CtrlType.DR_JOINT_POSITION:
                    action = DRJointPosAction(joint_pos=action)
                case CtrlType.JOINT_POSITION:
                    action = JointPosAction(joint_pos=action, gripper_width=0.0)
                case _:
                    raise ValueError(f"Unsupported control type: {self._args.robot_args.ctrl_type}")

        self._dispatch[self._args.robot_args.ctrl_type](action)

    def _apply_joint_pos(self, action: torch.Tensor) -> None:
        action_np = action.joint_pos[0].detach().cpu().numpy()
        target_pos = self.robot.default_dof_pos + action_np * self._action_scale
        self.robot.target_pos = target_pos
        self.robot.target_vel = np.zeros_like(target_pos)  # hold still in vel

    def _apply_dr_joint_pos(self, action: torch.Tensor) -> None:
        action_np = action.joint_pos[0].detach().cpu().numpy()
        target_pos = self.robot.default_dof_pos + action_np * self._action_scale
        target_vel = (target_pos - self.prev_target_pos) / float(self.dt)
        self.prev_target_pos = target_pos.copy()
        self.robot.target_pos = target_pos
        self.robot.target_vel = target_vel

    def emergency_stop(self) -> None:
        self.robot.emergency_stop()

    @property
    def is_emergency_stop(self) -> bool:
        return self.robot.is_emergency_stop

    @property
    def robot(self) -> LowStateCmdHandler:
        return self._robot  # type: ignore

    @property
    def action_dim(self) -> int:
        return self._robot.num_dof

    @property
    def action_space(self) -> spaces.Box:
        return self._action_space

    @property
    def action_scale(self) -> float:
        return self._action_scale

    @property
    def dof_names(self) -> list[str]:
        return self.robot.dof_names

    @property
    def default_dof_pos(self) -> torch.Tensor:
        return torch.tensor(self.robot.default_dof_pos, device=self._device, dtype=torch.float32)[
            None, :
        ]

    @property
    def dof_pos(self) -> torch.Tensor:
        return torch.tensor(self.robot.joint_pos, device=self._device, dtype=torch.float32)[None, :]

    @property
    def dof_vel(self) -> torch.Tensor:
        return torch.tensor(self.robot.joint_vel, device=self._device, dtype=torch.float32)[None, :]

    @property
    def quat(self) -> torch.Tensor:
        return torch.tensor(self.robot.quat, device=self._device, dtype=torch.float32)[None, :]

    @property
    def projected_gravity(self) -> torch.Tensor:
        projected_gravity = quaternions.rotate_vector(
            v=np.array([0, 0, -1]),
            q=quaternions.qinverse(self.robot.quat),
        )
        return torch.tensor(projected_gravity, device=self._device, dtype=torch.float32)[None, :]

    # @property
    # def base_ang_vel(self) -> torch.Tensor:
    #     return torch.tensor(self.robot.ang_vel, device=self._device, dtype=torch.float32)[None, :]
    # SHOULD BE GLOBAL ANGULAR VELOCITY, NOT IMPLEMENTED YET

    @property
    def base_ang_vel_local(self) -> torch.Tensor:
        return torch.tensor(self.robot.ang_vel, device=self._device, dtype=torch.float32)[None, :]

    @property
    def device(self) -> torch.device:
        return self._device
