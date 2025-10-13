from collections.abc import Callable
from typing import Any

import torch
import numpy as np
from gymnasium import spaces
from transforms3d import quaternions

from gs_env.common.bases.base_robot import BaseGymRobot
from gs_env.real.unitree.utils.low_state_handler import LowStateMsgHandler  #noqa
from gs_env.real.unitree.utils.low_state_controller import LowStateCmdHandler
from gs_env.sim.envs.config.schema import LeggedRobotEnvArgs

_DEFAULT_DEVICE = torch.device("cpu")


class UnitreeLeggedEnv(BaseGymRobot):
    def __init__(
        self, args: LeggedRobotEnvArgs,
        action_scale: float = 0.0,
        device: torch.device = _DEFAULT_DEVICE
    ):
        super().__init__()
        self._args = args
        self._low_state_controller = LowStateCmdHandler(args.robot_args)
        self.controller.init()
        self.controller.start()
        self._action_space = spaces.Box(shape=(self.num_dof,), low=-np.inf, high=np.inf)
        self._action_scale = args.robot_args.action_scale * action_scale
        self._device = device

    def reset(self, envs_idx: torch.IntTensor | None = None) -> None:
        # TODO: implement reset to reset_pos
        pass

    def apply_action(self, action: torch.Tensor) -> None:
        action_np = action.cpu().numpy()
        target_pos = self.controller.default_pos + action_np * self._action_scale
        self.controller.target_pos = target_pos

    @property
    def controller(self) -> LowStateCmdHandler:
        return self._low_state_controller

    @property
    def num_dof(self) -> int:
        return self._low_state_controller.num_dof

    @property
    def action_space(self) -> spaces.Box:
        return self._action_space

    @property
    def dof_pos(self) -> torch.Tensor:
        return torch.tensor(self.controller.joint_pos, device=self._device)

    @property
    def dof_vel(self) -> torch.Tensor:
        return torch.tensor(self.controller.joint_vel, device=self._device)

    @property
    def projected_gravity(self) -> torch.Tensor:
        projected_gravity = quaternions.rotate_vector(
                v=np.array([0, 0, -1]),
                q=quaternions.qinverse(self.controller.quat),
            )
        return torch.tensor(projected_gravity, device=self._device)

    @property
    def base_ang_vel(self) -> torch.Tensor:
        return torch.tensor(self.controller.ang_vel, device=self._device)
